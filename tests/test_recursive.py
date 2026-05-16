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

# Cython kernel is optional — only present when the wheel was built with a C
# toolchain. Tests that exercise it skip when ``_CY`` is False; the import is
# guarded with try/except so the file is loadable on toolchain-less installs.
try:
    from tsecon._rec_kernels_cy import (  # type: ignore[import-not-found]
        rec_linear_cython,
    )

    _CY = True
except ImportError:
    _CY = False


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


# ---------------------------------------------------------------------------
# Kernel-direct equivalence: Cython ≡ NumPy reference
# ---------------------------------------------------------------------------


class TestRecLinearKernelsAgreeOnArrays:
    """Cython kernel matches the NumPy reference output to within FP tolerance.

    Retrofits the kernel-direct equivalence class that ``test_stats_kernels.py``
    and ``test_fconvert_kernels.py`` have but ``test_recursive.py`` (session 17,
    the first kernel port) was written without. Closes review file
    ``F13_rec_linear_kernel_direct_test_missing``: prior coverage compared the
    high-level ``rec_linear`` wrapper against the lambda-based ``rec``
    reference, but never the two kernels (``_numpy`` / ``_cython``) against
    each other on raw arrays. A regression where both diverge from each other
    by ``O(rtol)`` while both still rounded the same way against ``rec`` at
    the public-API level would not have been caught.
    """

    @pytest.mark.parametrize(
        ("coeffs", "lags"),
        [
            ([1.0, 1.0], [1, 2]),  # Fibonacci-shaped
            ([0.5, 0.3], [1, 2]),  # AR(2) decay
            ([0.7], [1]),  # AR(1)
            ([0.4, 0.3, 0.2], [1, 2, 3]),  # AR(3)
            ([1.0, -0.5], [1, 4]),  # gappy lag polynomial
            ([0.2, 0.2, 0.2, 0.2, 0.2], [1, 2, 3, 4, 5]),  # AR(5) average
            ([0.9, -0.4], [2, 5]),  # gappy at lag-2 + lag-5
            ([0.1, 0.2, 0.3, 0.4], [1, 3, 5, 7]),  # odd-lag polynomial
        ],
    )
    @pytest.mark.parametrize("length", [15, 50, 150])
    def test_kernels_match_on_arrays(
        self,
        length: int,
        coeffs: list[float],
        lags: list[int],
    ) -> None:
        if not _CY:
            pytest.skip("Cython rec_linear kernel not compiled")
        rng = np.random.default_rng(seed=20260516)
        max_lag = max(lags)
        if length <= max_lag:
            pytest.skip(f"length={length} must exceed max_lag={max_lag} to leave room for ≥1 step")
        # Build the init buffer once, then make two independent copies — the
        # kernels mutate in place, so a shared buffer would conflate runs.
        base = np.zeros(length, dtype=np.float64)
        base[:max_lag] = rng.standard_normal(max_lag)
        values_numpy = base.copy()
        values_cython = base.copy()
        coeffs_arr = np.asarray(coeffs, dtype=np.float64)
        lags_arr = np.asarray(lags, dtype=np.int64)
        offset = max_lag
        count = length - max_lag
        rec_linear_numpy(values_numpy, offset, count, coeffs_arr, lags_arr)
        rec_linear_cython(values_cython, offset, count, coeffs_arr, lags_arr)
        np.testing.assert_allclose(values_numpy, values_cython, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# Edge cases for the rec_linear kernel (NaN, single-element, non-contiguous)
# ---------------------------------------------------------------------------


class TestRecLinearKernelEdgeCases:
    """Empty / degenerate inputs and NaN propagation follow the contract.

    Closes review file ``F06_rec_indexing_edge_case_gaps`` — the edge-case
    template crystallised by ``test_stats_kernels.py`` (session 20) and
    ``test_fconvert_kernels.py`` (session 21) was not retrofitted onto
    ``test_recursive.py`` (session 17). NaN in initial conditions is the
    realistic failure mode: TSeries auto-resize fills NaN gaps, so a
    recurrence over a series with leading NaN should propagate.
    """

    def test_rec_linear_numpy_propagates_nan_in_init(self) -> None:
        values = np.array([np.nan, 1.0, 0.0, 0.0, 0.0])
        rec_linear_numpy(
            values, 2, 3, np.asarray([1.0, 1.0]), np.asarray([1, 2], dtype=np.int64)
        )
        assert np.isnan(values[2])
        assert np.isnan(values[3])
        assert np.isnan(values[4])

    def test_rec_linear_cython_propagates_nan_in_init(self) -> None:
        if not _CY:
            pytest.skip("Cython rec_linear kernel not compiled")
        values = np.array([np.nan, 1.0, 0.0, 0.0, 0.0])
        rec_linear_cython(
            values, 2, 3, np.asarray([1.0, 1.0]), np.asarray([1, 2], dtype=np.int64)
        )
        assert np.isnan(values[2])
        assert np.isnan(values[3])
        assert np.isnan(values[4])

    def test_rec_linear_numpy_single_element_range(self) -> None:
        # count == 1: kernel writes exactly one position.
        values = np.array([1.0, 2.0, 0.0])
        rec_linear_numpy(
            values, 2, 1, np.asarray([1.0, 1.0]), np.asarray([1, 2], dtype=np.int64)
        )
        np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])

    def test_rec_linear_cython_single_element_range(self) -> None:
        if not _CY:
            pytest.skip("Cython rec_linear kernel not compiled")
        values = np.array([1.0, 2.0, 0.0])
        rec_linear_cython(
            values, 2, 1, np.asarray([1.0, 1.0]), np.asarray([1, 2], dtype=np.int64)
        )
        np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])

    def test_rec_linear_numpy_handles_non_contiguous_values(self) -> None:
        """The NumPy reference uses indexed Python reads/writes, so any 1-D float64 stride works.

        The matching ``_is_kernel_eligible`` check in
        ``tsecon._kernel_dispatch`` routes non-contiguous buffers to this
        path; the Cython kernel can't accept a non-contiguous view (typed
        memoryview cast requires ``[::1]``). This test exercises the
        kernel directly on ``arr[::2]`` so the fallback path is locked.
        """
        # 10 float64 elements, stride=16 (non-contiguous view of length-20 source).
        storage = np.zeros(20, dtype=np.float64)
        view = storage[::2]
        assert not view.flags.c_contiguous
        view[0] = 1.0
        view[1] = 1.0
        rec_linear_numpy(
            view, 2, 8, np.asarray([1.0, 1.0]), np.asarray([1, 2], dtype=np.int64)
        )
        expected_fib = [1.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0]
        np.testing.assert_array_equal(view, expected_fib)

    def test_rec_linear_cython_rejects_non_contiguous_values(self) -> None:
        """Cython memoryview cast rejects non-C-contiguous arrays.

        Locks the asymmetric-contract observation that motivates the
        dispatcher's contiguity check in ``_kernel_dispatch``. If a
        future patch loosens the ``cdef double[::1]`` constraint to
        ``cdef double[:]``, this test will start failing and the
        dispatcher's branch will need a reconsider.
        """
        if not _CY:
            pytest.skip("Cython rec_linear kernel not compiled")
        storage = np.zeros(20, dtype=np.float64)
        view = storage[::2]
        view[0] = 1.0
        view[1] = 1.0
        with pytest.raises(ValueError, match="contiguous"):
            rec_linear_cython(
                view, 2, 8, np.asarray([1.0, 1.0]), np.asarray([1, 2], dtype=np.int64)
            )
