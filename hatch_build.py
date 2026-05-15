# SPDX-License-Identifier: MIT
"""Hatchling custom build hook: compile Cython extensions in-place.

Discovers every ``.pyx`` file under ``src/tsecon/`` at wheel-build time,
runs :func:`Cython.Build.cythonize` to emit C sources, then drives
``setuptools.command.build_ext`` to compile them into platform-native
extension modules (``.so`` on Linux/macOS, ``.pyd`` on Windows). The
compiled artifacts land next to their ``.pyx`` siblings, so the wheel
includes them via the ``[tool.hatch.build.targets.wheel].artifacts`` glob
in ``pyproject.toml``.

Background: see ``claude_files/decisions/17_cython_dispatch_strategy.md``.

The hook is a no-op when the build target is not the wheel (e.g. the
sdist build, which ships ``.pyx`` sources and lets the consumer's wheel
build compile them).

The same entry point is callable from the command line for ad-hoc local
builds outside of ``hatchling``::

    uv run python hatch_build.py

This rebuilds in place, matching what ``pip install -e .`` would produce.
"""

from __future__ import annotations

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
        pairs.append((module, pyx))
    return pairs


def build_extensions_inplace() -> list[Path]:
    """Compile every ``.pyx`` under ``src/tsecon`` into a sibling extension.

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

    extensions = [
        Extension(
            module,
            [str(pyx.relative_to(ROOT))],
            include_dirs=[np.get_include()],
            # Match Julia's --check-bounds=no default: any out-of-bounds in
            # our kernels is a kernel-author bug, never a runtime user
            # condition (callers validate inputs before invoking).
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
        for module, pyx in pairs
    ]

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

    return [
        Path(ROOT / "src" / (mod.replace(".", "/"))).with_suffix("").parent
        / f"{mod.rsplit('.', 1)[-1]}*"
        for mod, _ in pairs
    ]


class CythonBuildHook(_Hook):  # type: ignore[misc,valid-type]
    """Hatchling build hook that compiles Cython kernels in-place."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Run before each build target — compile Cython kernels for wheels."""
        if self.target_name != "wheel":
            return
        build_extensions_inplace()
        # Force a platform-specific wheel tag (not py3-none-any) since the
        # wheel now contains compiled extensions.
        build_data["infer_tag"] = True
        build_data["pure_python"] = False


if __name__ == "__main__":
    produced = build_extensions_inplace()
    for path in produced:
        print(f"built: {path}")
    sys.exit(0)
