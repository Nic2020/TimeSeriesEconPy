# SPDX-License-Identifier: MIT
"""Lock the ``tsecon.x13`` package skeleton — module presence + public surface.

M2.0 shipped the package skeleton with empty ``__all__`` lists; M2.1
(session 49) populates ``_spec.py`` and ``_consts.py`` and wires the
top-level re-export. The tests here pin:

* the subpackage import path,
* the per-sibling ``__all__`` contract (``_spec`` populated; the other
  four private siblings still empty until their own milestone),
* the top-level ``tsecon.__all__`` re-export presence (flipped in M2.1),
* the on-disk file layout (locked at M2.0 so M2.x can't drift it).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# Siblings ordered by sub-milestone. ``_spec`` and ``_consts`` populated in
# M2.1; the remaining three stay empty until M2.4 / M2.5.
_EMPTY_SURFACE_SIBLINGS = ("_x13", "_write", "_result")
_POPULATED_SIBLINGS = ("_consts", "_spec")
_ALL_SIBLINGS = _EMPTY_SURFACE_SIBLINGS + _POPULATED_SIBLINGS


def test_subpackage_imports() -> None:
    """``import tsecon.x13`` succeeds and exposes the M2.1 + M2.2 surface."""
    mod = importlib.import_module("tsecon.x13")
    # M2.1 (31): 26 X13var leaves + 5 supporting types
    # (X13var, X13default, RegimeChange, ArimaSpec, ArimaModel).
    # M2.2 (+17): Span + 8 spec-container dataclasses (X13series, X13arima,
    # X13automdl, X13transform, X13regression, X13forecast, X13seats,
    # X13x11) + 8 builder functions (series, arima, automdl, transform,
    # regression, forecast, seats, x11).
    assert len(mod.__all__) == 48
    assert "ao" in mod.__all__
    assert "ArimaSpec" in mod.__all__
    assert "RegimeChange" in mod.__all__
    assert "Span" in mod.__all__
    assert "X13series" in mod.__all__
    assert "series" in mod.__all__
    assert "x11" in mod.__all__


@pytest.mark.parametrize("name", _EMPTY_SURFACE_SIBLINGS)
def test_empty_sibling_imports(name: str) -> None:
    """Each not-yet-populated private sibling imports with an empty ``__all__``."""
    mod = importlib.import_module(f"tsecon.x13.{name}")
    assert mod.__all__ == [], (
        f"tsecon.x13.{name} has unexpected public surface: {mod.__all__}. "
        "Lock the public surface in tsecon.x13.__init__ instead of the "
        "private sibling."
    )


@pytest.mark.parametrize("name", _POPULATED_SIBLINGS)
def test_populated_sibling_imports(name: str) -> None:
    """M2.1 populates ``_consts`` (private constants, ``__all__`` stays empty)
    and ``_spec`` (re-exported through ``tsecon.x13``)."""
    mod = importlib.import_module(f"tsecon.x13.{name}")
    if name == "_consts":
        # Constants are module-private; consumed by sibling modules, not users.
        assert mod.__all__ == []
    else:
        # _spec re-exports the X13var family (M2.1, 31) + the spec-container
        # dataclasses and builder functions (M2.2, +17) = 48.
        assert len(mod.__all__) == 48, (
            f"tsecon.x13.{name}.__all__ has {len(mod.__all__)} entries; "
            "expected 48 (M2.1's 31 + M2.2's 17: Span + 8 X13*** dataclasses "
            "+ 8 spec-builder functions)."
        )


def test_top_level_reexport() -> None:
    """``tsecon.__all__`` re-exports ``x13`` (flipped from M2.0's "not yet")."""
    import tsecon  # noqa: PLC0415

    assert "x13" in tsecon.__all__
    assert tsecon.x13 is importlib.import_module("tsecon.x13")


def test_skeleton_file_layout() -> None:
    """Lock the on-disk file layout so M2.2+ commits don't drift the shape."""
    from tsecon import x13 as x13_pkg  # noqa: PLC0415

    pkg_root = Path(x13_pkg.__file__).parent
    for name in _ALL_SIBLINGS:
        assert (pkg_root / f"{name}.py").is_file(), (
            f"src/tsecon/x13/{name}.py is missing; the M2.0 skeleton locked this filename."
        )
