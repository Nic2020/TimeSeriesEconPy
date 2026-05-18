# SPDX-License-Identifier: MIT
"""Lock the ``tsecon.x13`` package skeleton — module presence + empty surface.

M2.0 ships the package skeleton with empty ``__all__`` lists; this test
fails loudly if any of the five private siblings disappear, get renamed,
or grow an unintended public surface before the M2.1+ implementation
sessions wire them in. Once M2.1 lands, the ``__all__`` assertions here
shift to checking the planned public surface arrived in the planned
location (not in the wrong sibling).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

_SUBMODULES = ("_x13", "_consts", "_spec", "_write", "_result")


def test_subpackage_imports() -> None:
    """``import tsecon.x13`` succeeds without side effects."""
    mod = importlib.import_module("tsecon.x13")
    assert mod.__all__ == []


@pytest.mark.parametrize("name", _SUBMODULES)
def test_sibling_imports(name: str) -> None:
    """Each private sibling imports and exposes an empty ``__all__``."""
    mod = importlib.import_module(f"tsecon.x13.{name}")
    assert mod.__all__ == [], (
        f"tsecon.x13.{name} has unexpected public surface: {mod.__all__}. "
        "Lock the public surface in tsecon.x13.__init__ instead of the "
        "private sibling."
    )


def test_no_top_level_reexport() -> None:
    """``tsecon`` does not re-export ``x13`` until M2.1 lands a symbol.

    Decision 24 explicitly keeps ``tsecon.x13`` accessible via the
    ``import tsecon.x13`` path only. Top-level re-export waits for the
    first non-empty ``__all__`` in M2.1.
    """
    import tsecon  # noqa: PLC0415

    assert "x13" not in tsecon.__all__


def test_skeleton_file_layout() -> None:
    """Lock the on-disk file layout so M2.1+ commits don't drift the shape."""
    from tsecon import x13 as x13_pkg  # noqa: PLC0415

    pkg_root = Path(x13_pkg.__file__).parent
    for name in _SUBMODULES:
        assert (pkg_root / f"{name}.py").is_file(), (
            f"src/tsecon/x13/{name}.py is missing; the M2.0 skeleton locked this filename."
        )
