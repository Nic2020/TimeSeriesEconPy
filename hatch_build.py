# SPDX-License-Identifier: MIT
"""Hatchling custom build hook: compile Cython extensions in-place.

Discovers every ``.pyx`` file under ``src/tsecon/`` at wheel-build time,
runs :func:`Cython.Build.cythonize` to emit C sources, then drives
``setuptools.command.build_ext`` to compile them into platform-native
extension modules (``.so`` on Linux/macOS, ``.pyd`` on Windows). The
compiled artifacts land next to their ``.pyx`` siblings. The hook includes
the exact files returned by this interpreter's build, avoiding stale artifacts
from another Python ABI in the same checkout.

The hook is a no-op when the build target is not the wheel (e.g. the
sdist build, which ships ``.pyx`` sources and lets the consumer's wheel
build compile them).

The same entry point is callable from the command line for ad-hoc local
builds outside of ``hatchling``::

    uv run python hatch_build.py

This rebuilds in place, matching what ``pip install -e .`` would produce.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hatchling.builders.hooks.plugin.interface import BuildHookInterface as _Hook
else:
    try:
        from hatchling.builders.hooks.plugin.interface import (
            BuildHookInterface as _Hook,
        )
    except ImportError:  # pragma: no cover — only the standalone-script path takes this
        _Hook = object  # type: ignore[assignment,misc]

ROOT = Path(__file__).resolve().parent
SRC_PKG = ROOT / "src" / "tsecon"


def _discover_pyx_modules() -> list[tuple[str, Path]]:
    """Return ``(dotted_module_name, .pyx_path)`` pairs under ``src/tsecon``."""
    pairs: list[tuple[str, Path]] = []
    for pyx in SRC_PKG.rglob("*.pyx"):
        rel = pyx.relative_to(ROOT / "src")
        module = ".".join(rel.with_suffix("").parts)
        if module == "tsecon.dataecon._native" and not os.environ.get("TSECON_DATAECON_ROOT"):
            continue
        pairs.append((module, pyx))
    return pairs


def _configure_static_dataecon(extension: Any, native_root: Path) -> None:
    """Link a verified private archive and retain its input provenance."""
    archive = native_root / "lib/libdaec.a"
    for path in (
        native_root / "include/daec.h",
        archive,
        native_root / "build-info.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing DataEcon build input: {path}")
    manifest = json.loads((native_root / "build-info.json").read_text(encoding="utf-8"))
    if manifest.get("linkage") != "static-hidden" or manifest["platform"] != sys.platform:
        raise ValueError("DataEcon archive linkage/platform does not match this build.")
    for relative in ("include/daec.h", "lib/libdaec.a"):
        if (
            hashlib.sha256((native_root / relative).read_bytes()).hexdigest()
            != manifest["outputs"][relative]
        ):
            raise ValueError(f"DataEcon build input differs from manifest: {relative}")
    extension.include_dirs.append(str(native_root / "include"))
    extension.extra_objects.append(str(archive))
    extension.depends.extend([str(archive), str(native_root / "include/daec.h")])
    extension.extra_compile_args.append("-fvisibility=hidden")
    if sys.platform == "linux":
        extension.libraries.extend(["pthread", "dl", "m"])
        extension.extra_link_args.append("-Wl,--exclude-libs,ALL")
    binary_dir = SRC_PKG / "dataecon" / "_binary"
    binary_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(native_root / "build-info.json", binary_dir / "build-info.json")


def _configure_dataecon(extension: Any) -> None:
    """Attach the platform native input and its packaged provenance."""
    native_root = Path(os.environ["TSECON_DATAECON_ROOT"]).resolve()
    if sys.platform in ("linux", "darwin"):
        _configure_static_dataecon(extension, native_root)
        return
    if sys.platform != "win32":
        raise RuntimeError("Unsupported DataEcon build platform.")
    for relative in ("include/daec.h", "lib/daec.lib", "bin/libdaec.dll"):
        if not (native_root / relative).is_file():
            raise FileNotFoundError(f"Missing DataEcon build input: {native_root / relative}")
    extension.include_dirs.append(str(native_root / "include"))
    extension.library_dirs.append(str(native_root / "lib"))
    extension.libraries.append("daec")
    extension.depends.extend(
        [str(native_root / "include/daec.h"), str(native_root / "lib/daec.lib")]
    )
    binary_dir = SRC_PKG / "dataecon" / "_binary"
    binary_dir.mkdir(parents=True, exist_ok=True)
    native_dll = native_root / "bin/libdaec.dll"
    bundled_dll = binary_dir / "libdaec.dll"
    # Julia artifacts can be read-only. Avoid replacing an identical DLL
    # on rebuild, and do not propagate source file permissions.
    if not bundled_dll.exists() or bundled_dll.read_bytes() != native_dll.read_bytes():
        if bundled_dll.exists():
            bundled_dll.chmod(bundled_dll.stat().st_mode | stat.S_IWRITE)
        shutil.copyfile(native_dll, bundled_dll)
    build_info = native_root / "build-info.json"
    if build_info.is_file():
        shutil.copyfile(build_info, binary_dir / "build-info.json")


def build_extensions_inplace() -> list[Path]:
    """Compile selected ``.pyx`` files under ``src/tsecon`` into sibling extensions.

    Returns the list of produced extension paths (``.so`` / ``.pyd``).
    Lazy-imports Cython + NumPy + setuptools so this module can be loaded
    in environments that don't yet have the build deps installed.
    """
    pairs = _discover_pyx_modules()
    if not pairs:
        return []

    import numpy as np
    from Cython.Build import cythonize
    from setuptools import Distribution, Extension
    from setuptools.command.build_ext import build_ext

    extensions = []
    for module, pyx in pairs:
        extension = Extension(
            module,
            [str(pyx.relative_to(ROOT))],
            include_dirs=[np.get_include()],
            # Match Julia's --check-bounds=no default: any out-of-bounds in
            # our kernels is a kernel-author bug, never a runtime user
            # condition (callers validate inputs before invoking).
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
        if module == "tsecon.dataecon._native":
            _configure_dataecon(extension)
        extensions.append(extension)

    ext_modules = cythonize(
        extensions,
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        },
    )

    # package_dir tells setuptools that the `tsecon` package lives under
    # src/tsecon/; without it, `inplace=1` would try to write the compiled
    # extension to ./tsecon/_*.pyd (which doesn't exist in our src-layout).
    dist = Distribution(
        {
            "name": "TimeSeriesEconPy",
            "ext_modules": ext_modules,
            "package_dir": {"tsecon": "src/tsecon"},
            "packages": ["tsecon"],
        }
    )
    cmd = build_ext(dist)
    cmd.inplace = 1
    cmd.ensure_finalized()
    cmd.run()

    # Normalize Windows short-name temp paths used by builds from an sdist.
    return [Path(cmd.get_ext_fullpath(module)).resolve() for module, _ in pairs]


class CythonBuildHook(_Hook):  # type: ignore[misc,valid-type]
    """Hatchling build hook that compiles Cython extensions in-place."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Run before each build target — compile Cython kernels for wheels."""
        if self.target_name != "wheel":
            return
        extensions = build_extensions_inplace()
        # Include only this interpreter's build products. Broad globs pick up
        # stale .pyd/.so files when multiple Python versions share a checkout.
        build_data.setdefault("artifacts", []).extend(
            path.relative_to(ROOT).as_posix() for path in extensions
        )
        if os.environ.get("TSECON_DATAECON_ROOT"):
            if sys.platform == "win32":
                build_data.setdefault("artifacts", []).append(
                    "src/tsecon/dataecon/_binary/libdaec.dll"
                )
            if (Path(os.environ["TSECON_DATAECON_ROOT"]) / "build-info.json").is_file():
                build_data["artifacts"].append("src/tsecon/dataecon/_binary/build-info.json")
        # Force a platform-specific wheel tag (not py3-none-any) since the
        # wheel now contains compiled extensions.
        build_data["infer_tag"] = True
        build_data["pure_python"] = False


if __name__ == "__main__":
    produced = build_extensions_inplace()
    for path in produced:
        print(f"built: {path}")
    sys.exit(0)
