# SPDX-License-Identifier: MIT
"""Build pinned DataEcon/SQLite source with MSVC for Windows x86-64.

The archive is supplied explicitly and verified before extraction. Output must
be a new directory. No upstream checkout or global compiler settings are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sysconfig
from contextlib import chdir
from importlib.metadata import version as distribution_version
from pathlib import Path
from zipfile import ZipFile

SOURCE_COMMIT = "1a108688a044380f808bebf64079e32dbb9cd1a4"
SOURCE_URL = f"https://codeload.github.com/bankofcanada/DataEcon/zip/{SOURCE_COMMIT}"
SOURCE_SHA256 = "d2b48df3a47d173c43354fe033438253bb46d878ae6c3824c6224fbd4e6a2d71"
HEADER_SHA256 = "b593ac15cf4d262705d2b19b1cc3cf7936322e8cb28079ed0fa07c2049a97c10"
EPOCH_BEFORE = b"static const uint32_t EPOCH_s = 82;"
EPOCH_AFTER = b"#define EPOCH_s UINT32_C(82)"


def file_hash(path: Path) -> str:
    """Return the SHA-256 of a build input or output."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_source(archive: Path, output: Path) -> Path:
    """Verify the pinned archive and extract into a fresh build directory."""
    if file_hash(archive) != SOURCE_SHA256:
        raise ValueError("DataEcon source archive SHA-256 does not match the pinned input.")
    if output.exists():
        raise FileExistsError(
            "Use a new output directory; existing build output is never replaced."
        )
    source_parent = output / "source"
    prefix = f"DataEcon-{SOURCE_COMMIT}/"
    with ZipFile(archive) as zipped:
        for member in zipped.infolist():
            destination = (source_parent / member.filename).resolve()
            if not member.filename.startswith(prefix) or not destination.is_relative_to(
                source_parent.resolve()
            ):
                raise ValueError("Unexpected archive member path.")
        output.mkdir(parents=True)
        zipped.extractall(source_parent)
    source = source_parent / f"DataEcon-{SOURCE_COMMIT}"
    if file_hash(source / "include/daec.h") != HEADER_SHA256:
        raise ValueError("Extracted DataEcon header does not match the pinned ABI.")
    return source


def apply_msvc_patch(source: Path) -> dict[str, str]:
    """Make one internal date constant valid in MSVC file-scope C initializers.

    MSVC rejects references to a static const variable in these initializers.
    UINT32_C preserves the same unsigned value as an integer constant expression.
    No public declaration or arithmetic expression is otherwise changed.
    """
    path = source / "src/libdaec/dates.c"
    original = path.read_bytes()
    if original.count(EPOCH_BEFORE) != 1:
        raise ValueError("Expected unmodified DataEcon EPOCH_s declaration exactly once.")
    patched = original.replace(EPOCH_BEFORE, EPOCH_AFTER)
    path.write_bytes(patched)
    return {
        "file": "src/libdaec/dates.c",
        "before": EPOCH_BEFORE.decode(),
        "after": EPOCH_AFTER.decode(),
        "original_sha256": hashlib.sha256(original).hexdigest(),
        "patched_sha256": hashlib.sha256(patched).hexdigest(),
    }


def public_exports(header: Path) -> list[str]:
    """Select exactly the public functions in the pinned C header."""
    exports = re.findall(r"^\s*(?:int|const char \*)\s*(de_\w+)\(", header.read_text(), re.M)
    if len(exports) != 41 or len(set(exports)) != 41:
        raise ValueError(
            "Unexpected public DataEcon export list; review the header before building."
        )
    return exports


def build(archive: Path, output: Path) -> None:
    """Compile the pinned source and produce DLL, import library, header and notices."""
    if sysconfig.get_platform() != "win-amd64":
        raise RuntimeError("This build requires Windows x86-64 Python and MSVC Build Tools.")
    archive, output = archive.resolve(), output.resolve()
    source = extract_source(archive, output)
    patch = apply_msvc_patch(source)
    exports = public_exports(source / "include/daec.h")
    for subdir in ("include", "lib", "bin", "licenses", "objects"):
        (output / subdir).mkdir()
    shutil.copyfile(source / "include/daec.h", output / "include/daec.h")
    # Copy content, not upstream file attributes; these are editable build outputs.
    shutil.copyfile(source / "LICENSE.md", output / "licenses/DataEcon.txt")
    shutil.copyfile(source / "src/sqlite3/README.md", output / "licenses/SQLite-source.md")
    (output / "licenses/SQLite.txt").write_text(
        "SQLite 3.50.2 is bundled in DataEcon 0.4.0. Its source disclaims copyright.\n"
        "See src/sqlite3/sqlite3.c in the extracted DataEcon source tree.\n",
        encoding="utf-8",
    )
    definition = output / "lib/daec.def"
    definition.write_text("LIBRARY libdaec.dll\nEXPORTS\n" + "\n".join(exports) + "\n")

    # Keep source/archive validation usable without build-only dependencies.
    from setuptools._distutils.ccompiler import new_compiler  # noqa: PLC0415

    compiler = new_compiler(compiler="msvc")
    compiler.initialize()
    banner = subprocess.run([compiler.cc], capture_output=True, text=True, check=False)
    compiler_version = (banner.stdout + banner.stderr).strip().splitlines()
    compile_flags = ["/std:c11"]
    macros = [("_CRT_SECURE_NO_WARNINGS", "1")]
    # Relative source names avoid duplicated absolute paths in distutils' object
    # paths on Windows and keep __FILE__ diagnostics independent of the user path.
    with chdir(source):
        sources = [*sorted(Path("src/libdaec").glob("*.c")), Path("src/sqlite3/sqlite3.c")]
        objects = compiler.compile(
            [str(path) for path in sources],
            output_dir=str(output / "objects"),
            include_dirs=["include", "src/libdaec", "src/sqlite3"],
            macros=macros,
            extra_postargs=compile_flags,
        )
        compiler.link_shared_object(
            objects,
            str(output / "bin/libdaec.dll"),
            extra_postargs=[f"/DEF:{definition}", f"/IMPLIB:{output / 'lib/daec.lib'}"],
        )
    metadata = {
        "source_commit": SOURCE_COMMIT,
        "source_url": SOURCE_URL,
        "source_archive_sha256": SOURCE_SHA256,
        "dataecon_version": "0.4.0",
        "sqlite_version": "3.50.2",
        "platform": sysconfig.get_platform(),
        "setuptools_version": distribution_version("setuptools"),
        "windows_sdk_versions": sorted(
            {
                match.group(1)
                # setuptools keeps system include paths on the compiler class;
                # the instance list contains only explicitly added paths.
                for path in [*compiler.include_dirs, *getattr(type(compiler), "include_dirs", [])]
                if (match := re.search(r"[/\\]include[/\\](10\.[0-9.]+)[/\\]", path))
            }
        ),
        "compiler": compiler_version,
        "compile_options": compiler.compile_options,
        "extra_compile_flags": compile_flags,
        "macros": macros,
        "patch": patch,
        "exports": exports,
        "outputs": {
            path: file_hash(output / path)
            for path in ("include/daec.h", "bin/libdaec.dll", "lib/daec.lib")
        },
    }
    (output / "build-info.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Built DataEcon 0.4.0 from pinned source. Set TSECON_DATAECON_ROOT to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive", type=Path, help="Pinned DataEcon source ZIP (see docs/dataecon.md)"
    )
    parser.add_argument("--output", required=True, type=Path, help="New native build directory")
    args = parser.parse_args()
    build(args.archive, args.output)
