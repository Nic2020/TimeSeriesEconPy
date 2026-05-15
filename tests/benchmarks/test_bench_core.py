# SPDX-License-Identifier: MIT
"""In-Python regression benchmarks (counterpart to the comparison harness).

These mirror the eight scenarios in
``benchmarks/compare/scenarios.py``, but run under ``pytest-benchmark``
rather than ``timeit`` so they participate in pytest's normal test
collection and statistics. Skipped by default — run with::

    uv run pytest tests/benchmarks/ --benchmark-only

or to update a saved baseline::

    uv run pytest tests/benchmarks/ --benchmark-only --benchmark-save=NAME

The cross-language harness (``benchmarks/compare/run.py``) is the
canonical paper-grade comparator; this file's job is to flag regressions
between commits *inside* the Python implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

# Reuse the scenario registry from the comparison harness — keeping the
# two suites in lockstep avoids drift.
HARNESS_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "compare"
sys.path.insert(0, str(HARNESS_DIR))
from scenarios import RUN, SETUP  # noqa: E402

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

pytestmark = pytest.mark.benchmark


@pytest.mark.parametrize("name", sorted(SETUP.keys()))
def test_scenario_runs_under_pytest_benchmark(
    name: str,
    benchmark: BenchmarkFixture,
) -> None:
    """Run each comparison-harness scenario under pytest-benchmark.

    The assertion only checks the run returned (defeats DCE in the rare
    case where pytest-benchmark optimises through a no-op). Regression
    thresholds are set by ``--benchmark-compare`` / ``--benchmark-save``
    workflows rather than by hard-coded bounds — environmental jitter
    makes absolute bounds brittle.
    """
    state = SETUP[name]()
    run = RUN[name]
    result: Any = benchmark(run, state)
    assert result is not None or name == "indexing_mit_lookup_100"  # sum may be 0.0


def test_all_scenarios_have_descriptions() -> None:
    """The harness expects every scenario to have a description string."""
    from scenarios import DESCRIPTION  # noqa: PLC0415

    assert set(DESCRIPTION.keys()) == set(SETUP.keys()) == set(RUN.keys())
    for name, desc in DESCRIPTION.items():
        assert isinstance(desc, str), f"description for {name!r} must be a string"
        assert desc, f"description for {name!r} must be non-empty"
