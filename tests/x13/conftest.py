# SPDX-License-Identifier: MIT
"""Shared fixtures for the ``tsecon.x13`` test suite.

Lands the path-resolution + load-on-demand helpers M2.1 onwards will use
for the per-builder ``.spc`` round-trip tests. Keeping the fixture
infrastructure here (rather than in each test file) avoids the
copy-paste drift the upstream Julia test suite carries.

Two helpers ship in this M2.0 stub:

* :func:`x13_fixture_dir` (pytest fixture) — absolute path to
  ``tests/x13/fixtures/``. M2.1 commits will populate it with
  ``<builder>.spc`` + ``<builder>.out`` reference files extracted from
  the Julia upstream's test suite.
* :func:`load_fixture` (pytest fixture) — returns a callable that reads
  a fixture file by short name (``load_fixture("series.spc")`` →
  ``str``). Mirrors the read-only fixture-loading pattern from
  ``tests/test_fconvert_calendar.py`` (date-of-year fixture lookups).

The fixtures directory does not yet exist; the loader raises
:class:`FileNotFoundError` with the canonical message if M2.1+ code
asks for a fixture before it's been added.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

_FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def x13_fixture_dir() -> Path:
    """Absolute path to ``tests/x13/fixtures/``.

    May not exist yet — the M2.0 skeleton ships no fixtures. M2.1+
    commits populate it. Tests that require fixtures should ``skip``
    if the directory is absent.
    """
    return _FIXTURE_DIR


@pytest.fixture(scope="session")
def load_fixture(x13_fixture_dir: Path) -> Callable[[str], str]:
    """Return a callable that reads a fixture file by short name."""

    def _load(name: str) -> str:
        path = x13_fixture_dir / name
        if not path.is_file():
            msg = (
                f"X13 fixture {name!r} not found at {path!s}. "
                "M2.1+ commits will populate tests/x13/fixtures/."
            )
            raise FileNotFoundError(msg)
        return path.read_text(encoding="utf-8")

    return _load
