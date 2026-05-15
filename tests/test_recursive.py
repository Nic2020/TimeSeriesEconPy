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
    rec_linear,
    rec_linear_is_cython,
)
from tsecon._rec_kernels import rec_linear_numpy
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


# ---------------------------------------------------------------------------
# rec_linear — Cython-backed specialisation for closed-form linear recurrences.
# Decision 17 / MASTER_PLAN § M1.5. The same tests cover both the Cython
# path (when the compiled extension is importable) and the pure-NumPy
# fallback — public `rec_linear` dispatches based on availability.
# ---------------------------------------------------------------------------


class TestRecLinearFibonacci:
    """Fibonacci via rec_linear — the canonical lag-(1, 2) recurrence."""

    def test_fibonacci_unit(self) -> None:
        s = TSeries(_unit(1), np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        rec_linear(s, [1.0, 1.0], [1, 2], MITRange(_unit(3), _unit(10)))
        expected = [1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0]
        np.testing.assert_array_equal(s.values, expected)

    def test_returns_none(self) -> None:
        s = TSeries(_unit(1), np.array([1.0, 1.0, 0.0, 0.0]))
        result = rec_linear(s, [1.0, 1.0], [1, 2], MITRange(_unit(3), _unit(4)))
        assert result is None


class TestRecLinearAgreesWithRec:
    """The whole point of the spike: identical output to the lambda form."""

    @pytest.mark.parametrize(
        ("coeffs", "lags"),
        [
            ([1.0, 1.0], [1, 2]),  # Fibonacci
            ([0.5, 0.3], [1, 2]),  # AR(2)
            ([0.7], [1]),  # AR(1)
            ([0.4, 0.3, 0.2], [1, 2, 3]),  # AR(3)
            ([1.0, -0.5], [1, 4]),  # gappy lag polynomial
        ],
    )
    def test_matches_lambda_rec(
        self,
        coeffs: list[float],
        lags: list[int],
    ) -> None:
        max_lag = max(lags)
        init = np.arange(1.0, max_lag + 1)
        rng = mitrange(qq(2020, 1) + max_lag, qq(2020, 1) + max_lag + 99)
        # rec via lambda
        ref = TSeries(qq(2020, 1), np.concatenate([init, np.zeros(100)]))
        coeffs_arr = np.asarray(coeffs)
        lags_arr = np.asarray(lags)
        rec(
            rng,
            ref,
            lambda t: float(
                sum(c * ref[t - int(k)] for c, k in zip(coeffs_arr, lags_arr, strict=True))
            ),
        )
        # rec_linear path
        got = TSeries(qq(2020, 1), np.concatenate([init, np.zeros(100)]))
        rec_linear(got, coeffs, lags, rng)
        np.testing.assert_allclose(got.values, ref.values, rtol=1e-12, atol=0.0)


class TestRecLinearKernelFallback:
    """Lock the NumPy reference kernel against a hand-computed answer.

    Exercises ``rec_linear_numpy`` directly so the fallback path is
    validated independent of the public dispatcher.
    """

    def test_numpy_kernel_fibonacci(self) -> None:
        values = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        rec_linear_numpy(values, 2, 4, np.array([1.0, 1.0]), np.array([1, 2], dtype=np.int64))
        np.testing.assert_array_equal(values, [1.0, 1.0, 2.0, 3.0, 5.0, 8.0])

    def test_is_cython_flag_is_boolean(self) -> None:
        flag = rec_linear_is_cython()
        assert isinstance(flag, bool)


class TestRecLinearValidation:
    def test_frequency_mismatch_raises(self) -> None:
        target = TSeries(qq(2020, 1), np.array([1.0, 1.0, 0.0, 0.0]))
        rng = MITRange(MIT(Yearly(), 2020), MIT(Yearly(), 2023))
        with pytest.raises(TypeError, match="frequency"):
            rec_linear(target, [1.0], [1], rng)

    def test_non_unit_step_raises(self) -> None:
        target = TSeries(qq(2020, 1), np.array([1.0, 1.0, 0.0, 0.0, 0.0]))
        rng = MITRange(qq(2020, 1), qq(2021, 4), step=2)
        with pytest.raises(ValueError, match="step=1"):
            rec_linear(target, [1.0], [1], rng)

    def test_mismatched_coeffs_lags_raises(self) -> None:
        target = TSeries(qq(2020, 1), np.array([1.0, 1.0, 0.0, 0.0]))
        with pytest.raises(ValueError, match="same length"):
            rec_linear(target, [1.0, 1.0], [1, 2, 3], mitrange(qq(2020, 3), qq(2020, 4)))

    def test_empty_coeffs_raises(self) -> None:
        target = TSeries(qq(2020, 1), np.array([1.0, 0.0]))
        with pytest.raises(ValueError, match="non-empty"):
            rec_linear(target, [], [], mitrange(qq(2020, 2), qq(2020, 2)))

    def test_zero_lag_raises(self) -> None:
        target = TSeries(qq(2020, 1), np.array([1.0, 0.0]))
        with pytest.raises(ValueError, match="lags must be >= 1"):
            rec_linear(target, [1.0], [0], mitrange(qq(2020, 2), qq(2020, 2)))

    def test_negative_lag_raises(self) -> None:
        target = TSeries(qq(2020, 1), np.array([1.0, 0.0]))
        with pytest.raises(ValueError, match="lags must be >= 1"):
            rec_linear(target, [1.0], [-1], mitrange(qq(2020, 2), qq(2020, 2)))

    def test_initial_conditions_missing_raises(self) -> None:
        target = TSeries(qq(2020, 2), np.array([1.0, 0.0]))
        with pytest.raises(ValueError, match="initial conditions"):
            rec_linear(target, [1.0, 1.0], [1, 2], mitrange(qq(2020, 3), qq(2020, 3)))

    def test_non_float64_dtype_raises(self) -> None:
        target = TSeries(qq(2020, 1), np.array([1, 1, 0, 0], dtype=np.int64))
        with pytest.raises(TypeError, match="float64"):
            rec_linear(target, [1.0], [1], mitrange(qq(2020, 2), qq(2020, 4)))


class TestRecLinearEmptyRange:
    def test_empty_range_is_noop(self) -> None:
        s = TSeries(_unit(1), np.array([1.0, 1.0, 99.0]))
        rng = MITRange(_unit(5), _unit(4))
        assert len(rng) == 0
        rec_linear(s, [1.0, 1.0], [1, 2], rng)
        np.testing.assert_array_equal(s.values, [1.0, 1.0, 99.0])


class TestRecLinearAutoResize:
    """rec_linear must auto-extend target the way the lambda-form rec does."""

    def test_extends_past_lastdate(self) -> None:
        s = TSeries(_unit(1), np.array([1.0, 1.0]))
        rec_linear(s, [1.0, 1.0], [1, 2], MITRange(_unit(3), _unit(8)))
        expected = [1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0]
        assert s.firstdate == _unit(1)
        assert s.lastdate == _unit(8)
        np.testing.assert_array_equal(s.values, expected)


class TestRecLinearQuarterly:
    """Locks the quarterly-MIT round-trip just as the rec tests do."""

    def test_ar2_quarterly(self) -> None:
        start = qq(2020, 1)
        s = TSeries(start, np.zeros(102))
        s[start] = 1.0
        s[start + 1] = 1.0
        rec_linear(s, [0.5, 0.3], [1, 2], MITRange(start + 2, start + 101))
        # Hand-checked first few entries (initial 1.0, 1.0).
        # s[3] = 0.5 * s[2] + 0.3 * s[1] = 0.5 + 0.3 = 0.8
        # s[4] = 0.5 * s[3] + 0.3 * s[2] = 0.4 + 0.3 = 0.7
        np.testing.assert_allclose(s.values[2], 0.8)
        np.testing.assert_allclose(s.values[3], 0.7)
        # The series should be bounded: |coeffs[0]| + |coeffs[1]| < 1 → decay.
        assert abs(s.values[-1]) < 1.0
