# SPDX-License-Identifier: MIT
"""Build pinned DataEcon/SQLite as a private PIC archive on Linux/macOS."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from build_dataecon_windows import (
    SOURCE_COMMIT,
    SOURCE_SHA256,
    SOURCE_URL,
    apply_msvc_patch,
    extract_source,
    file_hash,
)


def target_flags(system: str, machine: str, deployment: str | None) -> list[str]:
    """Reject undeclared targets before creating build output."""
    flags = ["-std=gnu11", "-O2", "-fPIC", "-fvisibility=hidden", "-pthread"]
    if system == "linux" and machine == "x86_64":
        return flags
    if system == "darwin" and machine == "arm64" and deployment == "11.0":
        return [*flags, "-arch", "arm64", "-mmacosx-version-min=11.0"]
    raise RuntimeError(
        "Supported targets: Linux x86-64 or macOS arm64 with deployment target 11.0."
    )


def build(archive: Path, output: Path) -> None:
    flags = target_flags(sys.platform, platform.machine(), os.getenv("MACOSX_DEPLOYMENT_TARGET"))
    compiler = shlex.split(os.environ.get("CC", "cc"))
    archiver = shlex.split(os.environ.get("AR", "ar"))
    compiler_version = subprocess.check_output([*compiler, "--version"], text=True)
    output = output.resolve()
    source = extract_source(archive.resolve(), output)
    # The same integer constant-expression adjustment is portable C. Keep one
    # guarded patch implementation and record it for both toolchain families.
    patch = apply_msvc_patch(source)
    for name in ("include", "lib", "objects"):
        (output / name).mkdir()
    shutil.copyfile(source / "include/daec.h", output / "include/daec.h")
    sources = [*sorted(source.glob("src/libdaec/*.c")), source / "src/sqlite3/sqlite3.c"]
    if len(sources) != 14:
        raise ValueError("Unexpected pinned native source inventory.")
    objects = []
    for index, path in enumerate(sources):
        obj = output / "objects" / f"{index}_{path.stem}.o"
        subprocess.run(
            [
                *compiler,
                *flags,
                "-Iinclude",
                "-Isrc/libdaec",
                "-Isrc/sqlite3",
                "-c",
                str(path.relative_to(source)),
                "-o",
                str(obj),
            ],
            cwd=source,
            check=True,
        )
        objects.append(str(obj))
    library = output / "lib/libdaec.a"
    subprocess.run([*archiver, "rcs", str(library), *objects], check=True)
    manifest = {
        "source_commit": SOURCE_COMMIT,
        "source_url": SOURCE_URL,
        "source_archive_sha256": SOURCE_SHA256,
        "dataecon_version": "0.4.0",
        "sqlite_version": "3.50.2",
        "linkage": "static-hidden",
        "platform": sys.platform,
        "architecture": platform.machine(),
        "deployment_target": os.getenv("MACOSX_DEPLOYMENT_TARGET"),
        "compiler": compiler_version.splitlines(),
        "compile_flags": flags,
        "patch": patch,
        "outputs": {name: file_hash(output / name) for name in ("include/daec.h", "lib/libdaec.a")},
    }
    (output / "build-info.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Built private DataEcon archive: {library}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.archive, args.output)
