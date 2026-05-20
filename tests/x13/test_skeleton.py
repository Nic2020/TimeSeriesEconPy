# SPDX-License-Identifier: MIT
"""Lock the ``tsecon.x13`` package skeleton — module presence + public surface.

M2.0 shipped the package skeleton with empty ``__all__`` lists; M2.1
(session 49) populates ``_spec.py`` and ``_consts.py`` and wires the
top-level re-export. M2.4 (session 52) populates ``_write.py`` and adds
the :class:`X13spec` aggregator + :func:`newspec` / :func:`validateX13spec`
surface. The tests here pin:

* the subpackage import path,
* the per-sibling ``__all__`` contract (``_spec`` + ``_write`` populated;
  ``_consts`` private; ``_x13`` / ``_result`` still empty until M2.5),
* the top-level ``tsecon.__all__`` re-export presence (flipped in M2.1),
* the on-disk file layout (locked at M2.0 so M2.x can't drift it).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

# Siblings ordered by sub-milestone. ``_spec`` + ``_consts`` populated in
# M2.1; ``_write`` populated in M2.4; ``_x13`` / ``_result`` stay empty
# until M2.5.
_EMPTY_SURFACE_SIBLINGS = ("_x13", "_result")
_POPULATED_SIBLINGS = ("_consts", "_spec", "_write")
_ALL_SIBLINGS = _EMPTY_SURFACE_SIBLINGS + _POPULATED_SIBLINGS


# M2.1 (31): 26 X13var leaves + 5 supporting types
# (X13var, X13default, RegimeChange, ArimaSpec, ArimaModel).
# M2.2 (+17): Span + 8 spec-container dataclasses + 8 builder functions.
# M2.3 (+22): 11 rare-spec-container dataclasses + 11 rare builder functions.
# M2.4 (+5): X13spec, newspec, validateX13spec, x13write, impose_line_length.
# _spec.py: M2.1 (31) + M2.2 (17) + M2.3 (22) + M2.4 (3: X13spec, newspec,
# validateX13spec). The writer surface (x13write, impose_line_length) lives
# in _write.py, not _spec.py.
_EXPECTED_X13_SURFACE_LEN = 75
_EXPECTED_SPEC_SURFACE_LEN = 73
_EXPECTED_WRITE_SURFACE_LEN = 2


def test_subpackage_imports() -> None:
    """``import tsecon.x13`` succeeds and exposes the M2.1+M2.2+M2.3+M2.4 surface."""
    mod = importlib.import_module("tsecon.x13")
    assert len(mod.__all__) == _EXPECTED_X13_SURFACE_LEN
    assert "ao" in mod.__all__
    assert "ArimaSpec" in mod.__all__
    assert "RegimeChange" in mod.__all__
    assert "Span" in mod.__all__
    assert "X13series" in mod.__all__
    assert "X13spec" in mod.__all__
    assert "X13x11regression" in mod.__all__
    assert "series" in mod.__all__
    assert "x11" in mod.__all__
    assert "outlier" in mod.__all__
    assert "x11regression" in mod.__all__
    # M2.4 entries
    assert "newspec" in mod.__all__
    assert "validateX13spec" in mod.__all__
    assert "x13write" in mod.__all__
    assert "impose_line_length" in mod.__all__


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
    """M2.1 populates ``_consts`` (private) + ``_spec``; M2.4 adds ``_write``."""
    mod = importlib.import_module(f"tsecon.x13.{name}")
    if name == "_consts":
        # Constants are module-private; consumed by sibling modules, not users.
        assert mod.__all__ == []
    elif name == "_spec":
        assert len(mod.__all__) == _EXPECTED_SPEC_SURFACE_LEN, (
            f"tsecon.x13._spec.__all__ has {len(mod.__all__)} entries; "
            f"expected {_EXPECTED_SPEC_SURFACE_LEN} (M2.1's 31 + M2.2's 17 + "
            "M2.3's 22 + M2.4's 3: X13spec, newspec, validateX13spec)."
        )
    elif name == "_write":
        assert mod.__all__ == ["impose_line_length", "x13write"], (
            f"tsecon.x13._write.__all__ has unexpected entries: {mod.__all__}."
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
