# SPDX-License-Identifier: MIT
"""Smoke tests: package imports and metadata is well-formed."""

from __future__ import annotations

import re

import tsecon
from tsecon import MIRRORS_JULIA_SHA


def test_package_imports() -> None:
    """The top-level package can be imported."""
    assert tsecon.__version__


def test_mirror_sha_is_well_formed() -> None:
    """``MIRRORS_JULIA_SHA`` is a 40-character hex string."""
    assert re.fullmatch(r"[0-9a-f]{40}", MIRRORS_JULIA_SHA), (
        f"MIRRORS_JULIA_SHA={MIRRORS_JULIA_SHA!r} is not a 40-char hex SHA"
    )


def test_version_is_pep440() -> None:
    """``__version__`` is a valid PEP 440 version string."""
    pep440 = re.compile(
        r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*"
        r"((a|b|rc)(0|[1-9][0-9]*))?"
        r"(\.post(0|[1-9][0-9]*))?"
        r"(\.dev(0|[1-9][0-9]*))?$"
    )
    assert pep440.match(tsecon.__version__), (
        f"__version__={tsecon.__version__!r} is not valid PEP 440"
    )
