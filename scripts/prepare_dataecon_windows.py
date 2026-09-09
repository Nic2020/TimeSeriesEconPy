# SPDX-License-Identifier: MIT
"""Prepare the pinned Windows DataEcon artifact for a local MSVC extension build.

Pass an extracted DataEcon_jll 0.4.0+0 x86_64 Windows artifact as the argument.
This verifies the header/DLL, copies the license and creates an MSVC import
library. It does not download binaries or require Julia at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools._distutils.ccompiler import new_compiler

HEADER_SHA256 = "b593ac15cf4d262705d2b19b1cc3cf7936322e8cb28079ed0fa07c2049a97c10"
DLL_SHA256 = "1b11c18a26e5ab3e65fcccced1c0357c58717c59e8cb4a8cbc57ade3703e7004"
SYMBOLS = [
    "de_version",
    "de_open",
    "de_open_readonly",
    "de_close",
    "de_find_object",
    "de_axis_range",
    "de_store_tseries",
    "de_load_tseries",
    "de_get_attribute",
    "de_error",
    "de_clear_error",
    "de_pack_year_period_date",
    "de_unpack_year_period_date",
]


def prepare(artifact: Path, destination: Path) -> None:
    """Verify native inputs and prepare header, DLL, notice and import library."""
    if sys.platform != "win32":
        raise RuntimeError("This preparation script supports Windows x86-64 only.")
    for relative, expected in (("include/daec.h", HEADER_SHA256), ("bin/libdaec.dll", DLL_SHA256)):
        actual = hashlib.sha256((artifact / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Unexpected SHA-256 for {relative}: {actual}")
    compiler = new_compiler()
    compiler.initialize()
    for subdir in ("include", "lib", "bin", "licenses"):
        (destination / subdir).mkdir(parents=True, exist_ok=True)
    for source, target in (
        ("include/daec.h", "include/daec.h"),
        ("bin/libdaec.dll", "bin/libdaec.dll"),
        ("share/licenses/DataEcon/LICENSE.md", "licenses/DataEcon.txt"),
    ):
        output = destination / target
        if not output.exists() or output.read_bytes() != (artifact / source).read_bytes():
            shutil.copyfile(artifact / source, output)
    definition = destination / "lib/daec.def"
    definition.write_text("LIBRARY libdaec.dll\nEXPORTS\n" + "\n".join(SYMBOLS) + "\n")
    subprocess.run(
        [
            compiler.lib,
            "/nologo",
            "/machine:x64",
            f"/def:{definition}",
            f"/out:{destination / 'lib/daec.lib'}",
        ],
        check=True,
    )
    print(f"Set TSECON_DATAECON_ROOT to {destination.resolve()} before building.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path, default=Path("build/dataecon"))
    args = parser.parse_args()
    prepare(args.artifact, args.output.resolve())
