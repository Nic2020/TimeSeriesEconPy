# SPDX-License-Identifier: MIT
"""Tests for :func:`tsecon.rec` — the Python port of Julia's ``@rec`` macro.

Mirrors the Fibonacci docstring example from
``TimeSeriesEcon.jl/src/recursive.jl`` and adds Python-specific cases
(frequency mismatch, multi-series read, auto-resize on out-of-range
assignment, empty range).
"""

from __future__ import annotations

import numpy as np
import pytest

from tsecon import (
    MIT,
    MITRange,
    TSeries,
    mitrange,
    qq,
    rec,
)
from tsecon.frequencies import Unit, Yearly


def _unit(i: int) -> MIT:
    return MIT(Unit(), i)


class TestRecFibonacci:
    """The Julia docstring example, ported."""

    def test_fibonacci_unit(self) -> None:
        s = TSeries(_unit(1), np.array([1.0, 1.0]))
        rec(MITRange(_unit(3), _unit(10)), s, lambda t: s[t - 1] + s[t - 2])
        expected = [1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0]
        assert s.firstdate == _unit(1)
        assert s.lastdate == _unit(10)
        np.testing.assert_array_equal(s.values, expected)

    def test_fibonacci_returns_none(self) -> None:
        s = TSeries(_unit(1), np.array([1.0, 1.0]))
        result = rec(MITRange(_unit(3), _unit(10)), s, lambda t: s[t - 1] + s[t - 2])
        assert result is None


class TestRecAR1:
    """An AR(1) recurrence over a quarterly range, reading from a second series."""

    def test_consumption_reads_income(self) -> None:
        rng = mitrange(qq(2020, 1), qq(2024, 4))
        consumption = TSeries(qq(2020, 1), np.zeros(len(rng)))
        income = TSeries(qq(2020, 1), np.arange(1.0, len(rng) + 1.0))
        beta = 0.6
        consumption[qq(2020, 1)] = income[qq(2020, 1)]  # seed
        rec(
            mitrange(qq(2020, 2), qq(2024, 4)),
            consumption,
            lambda t: beta * consumption[t - 1] + (1 - beta) * income[t],
        )
        # Compare against a hand-written for loop.
        expected = TSeries(qq(2020, 1), np.zeros(len(rng)))
        expected[qq(2020, 1)] = income[qq(2020, 1)]
        for t in mitrange(qq(2020, 2), qq(2024, 4)):
            expected[t] = beta * expected[t - 1] + (1 - beta) * income[t]
        np.testing.assert_allclose(consumption.values, expected.values)


class TestRecFrequencyMismatch:
    def test_range_yearly_target_quarterly_raises(self) -> None:
        target = TSeries(qq(2020, 1), np.zeros(8))
        rng_yearly = mitrange(MIT(Yearly(), 2020), MIT(Yearly(), 2023))
        with pytest.raises(TypeError, match="frequency"):
            rec(rng_yearly, target, lambda t: 0.0)

    def test_range_unit_target_quarterly_raises(self) -> None:
        target = TSeries(qq(2020, 1), np.zeros(4))
        with pytest.raises(TypeError, match="frequency"):
            rec(MITRange(_unit(1), _unit(4)), target, lambda t: 0.0)


class TestRecEmptyRange:
    def test_empty_range_is_noop(self) -> None:
        s = TSeries(_unit(1), np.array([1.0, 2.0, 3.0]))
        # An "empty" mitrange (stop < start) has length 0.
        rng = MITRange(_unit(5), _unit(4))
        assert len(rng) == 0
        rec(rng, s, lambda t: 999.0)
        # Nothing changed.
        np.testing.assert_array_equal(s.values, [1.0, 2.0, 3.0])


class TestRecAutoResize:
    """Mirrors Julia's "assignment past the end extends storage" behaviour."""

    def test_assignment_past_end_extends_target(self) -> None:
        # Quarterly series of length 2; rec extends through the 8th quarter.
        s = TSeries(qq(2020, 1), np.array([1.0, 1.0]))
        rec(
            mitrange(qq(2020, 3), qq(2021, 4)),
            s,
            lambda t: s[t - 1] + s[t - 2],
        )
        # Fibonacci across 8 quarters.
        expected = [1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0]
        assert s.firstdate == qq(2020, 1)
        assert s.lastdate == qq(2021, 4)
        np.testing.assert_array_equal(s.values, expected)


class TestRecQuarterlyArithmetic:
    """A simple quarterly recurrence to lock the MIT-arithmetic round-trip."""

    def test_double_previous(self) -> None:
        s = TSeries(qq(2020, 1), np.array([1.0]))
        rec(mitrange(qq(2020, 2), qq(2020, 4)), s, lambda t: 2.0 * s[t - 1])
        np.testing.assert_array_equal(s.values, [1.0, 2.0, 4.0, 8.0])
        assert s.firstdate == qq(2020, 1)
        assert s.lastdate == qq(2020, 4)
