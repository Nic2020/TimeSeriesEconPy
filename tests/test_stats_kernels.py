# SPDX-License-Identifier: MIT
"""Tests for the M1.5 third Cython port — scalar-reduction stats kernels.

Covers the kernel pair (``_stats_kernels.{mean,var,std,cor}_numpy`` and the
matching ``_stats_kernels_cy`` Cython kernels when compiled), the public
``mean / var / std / cor`` dispatcher path through ``_stats.py``, and the
``stats_is_cython`` introspection helper. Behaviour-level tests for
``mean / std / var / cor`` already live in ``test_options_and_bdaily.py``
:class:`TestStatisticsBDaily`; this file validates the kernel-direct
contract and the equivalence between the Cython kernels (when present)
and their NumPy reference siblings.
"""

from __future__ import annotations

import math
import warnings
from fractions import Fraction

import numpy as np
import pytest

from tsecon import (
    MVTSeries,
    TSeries,
    cor,
    mean,
    qq,
    stats_is_cython,
    std,
    var,
)
from tsecon._stats_kernels import _scale_exponent, cor_numpy, mean_numpy, std_numpy, var_numpy

# Cython kernels are optional — they're only present when the wheel was built
# with a C toolchain. Tests that exercise them call ``stats_is_cython()`` and
# skip when False; the imports are guarded with try/except so the file is
# still loadable on toolchain-less installs.
try:
    from tsecon._stats_kernels_cy import (  # type: ignore[import-not-found]
        cor_cython,
        mean_cython,
        std_cython,
        var_cython,
    )

    _CY = True
except ImportError:
    _CY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gen_arrays() -> list[np.ndarray]:
    """Several length-N float64 inputs for parametric kernel tests."""
    rng = np.random.default_rng(seed=20260516)
    return [
        np.arange(100.0),
        np.arange(1.0, 101.0),
        rng.standard_normal(100),
        rng.standard_normal(50),
        rng.standard_normal(2),
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.zeros(10),
        np.full(10, 7.5),
    ]


# ---------------------------------------------------------------------------
# Kernel-direct equivalence: Cython ≡ NumPy reference
# ---------------------------------------------------------------------------


class TestStatsKernelsAgreeOnArrays:
    """Cython kernels match the NumPy reference output to within FP tolerance.

    The two-pass variance kernel uses naive left-to-right summation
    while NumPy uses pairwise summation; for length-100 well-conditioned
    inputs the agreement is well within ``rtol=1e-12``. Same precedent
    as the rec_linear kernel-equivalence tests
    (``TestRecLinearAgreesWithRec``).
    """

    @pytest.mark.parametrize("arr", _gen_arrays())
    def test_mean_kernel_matches_numpy(self, arr: np.ndarray) -> None:
        if not _CY:
            pytest.skip("Cython stats kernels not compiled")
        a = np.ascontiguousarray(arr, dtype=np.float64)
        np.testing.assert_allclose(
            mean_cython(a),
            mean_numpy(a),
            rtol=1e-12,
            atol=1e-15,
        )

    @pytest.mark.parametrize("arr", _gen_arrays())
    @pytest.mark.parametrize("ddof", [0, 1])
    def test_var_kernel_matches_numpy(self, arr: np.ndarray, ddof: int) -> None:
        if not _CY:
            pytest.skip("Cython stats kernels not compiled")
        if arr.shape[0] - ddof <= 0:
            pytest.skip("kernel contract requires len(values) > ddof")
        a = np.ascontiguousarray(arr, dtype=np.float64)
        np.testing.assert_allclose(
            var_cython(a, ddof),
            var_numpy(a, ddof),
            rtol=1e-12,
            atol=1e-15,
        )

    @pytest.mark.parametrize("arr", _gen_arrays())
    @pytest.mark.parametrize("ddof", [0, 1])
    def test_std_kernel_matches_numpy(self, arr: np.ndarray, ddof: int) -> None:
        if not _CY:
            pytest.skip("Cython stats kernels not compiled")
        if arr.shape[0] - ddof <= 0:
            pytest.skip("kernel contract requires len(values) > ddof")
        a = np.ascontiguousarray(arr, dtype=np.float64)
        np.testing.assert_allclose(
            std_cython(a, ddof),
            std_numpy(a, ddof),
            rtol=1e-12,
            atol=1e-15,
        )

    def test_cor_kernel_matches_numpy(self) -> None:
        if not _CY:
            pytest.skip("Cython stats kernels not compiled")
        rng = np.random.default_rng(seed=20260516)
        for _ in range(8):
            x = np.ascontiguousarray(rng.standard_normal(100), dtype=np.float64)
            y = np.ascontiguousarray(rng.standard_normal(100), dtype=np.float64)
            np.testing.assert_allclose(
                cor_cython(x, y),
                cor_numpy(x, y),
                rtol=1e-12,
                atol=1e-15,
            )

    # Constant-array semantics: both kernels return nan + RuntimeWarning when
    # at least one input has zero variance. The parametrisation covers
    # (i) a "normal-magnitude" constant where np.corrcoef would itself fire
    # the divide-by-zero RuntimeWarning, (ii) an FP-noisy-magnitude constant
    # at 1e-60 where np.corrcoef silently returns 1.0 (the M1.6.0 baseline
    # failure that motivated this lock), (iii) one constant + one variable
    # input (only one side degenerate). See tsecon._stats.cor docstring.
    @pytest.mark.parametrize(
        ("x_arr", "y_arr"),
        [
            (np.full(100, 7.5), np.full(100, 7.5)),
            (np.full(100, 1e-60), np.full(100, 1e-60)),
            (np.full(100, 7.5), np.arange(100.0)),
            (np.arange(100.0), np.full(100, 7.5)),
        ],
    )
    def test_cor_constant_array_returns_nan_both_kernels(
        self, x_arr: np.ndarray, y_arr: np.ndarray
    ) -> None:
        x = np.ascontiguousarray(x_arr, dtype=np.float64)
        y = np.ascontiguousarray(y_arr, dtype=np.float64)
        with pytest.warns(RuntimeWarning, match="constant input"):
            assert np.isnan(cor_numpy(x, y))
        if _CY:
            with pytest.warns(RuntimeWarning, match="constant input"):
                assert np.isnan(cor_cython(x, y))


# ---------------------------------------------------------------------------
# Correlation scaling — subnormal squares and overflow
# ---------------------------------------------------------------------------

# The Hypothesis-found subnormal-square example: centred values near 1e-158 square to
# subnormals, where the two kernels' summation orders disagreed by a
# relative 6.2e-8. Its exact Pearson correlation is -1/3 (``y`` is ``x``
# reversed, so the centred products are three ``-1/16`` terms and one
# ``9/16`` term against ``12/16`` of centred squares).
SUBNORMAL_SQUARE_INPUT = np.array([1.78536692e-158, 0.0, 0.0, 0.0])
TINY = np.nextafter(0.0, 1.0)  # the smallest positive subnormal, 5e-324
HUGE = np.finfo(np.float64).max  # 1.8e308


def _exact_squared_cor(x: np.ndarray, y: np.ndarray) -> tuple[Fraction, int]:
    """``(sxy**2 / (sxx * syy), sign(sxy))`` in exact rational arithmetic."""
    xs = [Fraction(v) for v in x.tolist()]
    ys = [Fraction(v) for v in y.tolist()]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [v - mx for v in xs]
    dy = [v - my for v in ys]
    sxx = sum(d * d for d in dx)
    syy = sum(d * d for d in dy)
    sxy = sum(a * b for a, b in zip(dx, dy, strict=True))
    return sxy * sxy / (sxx * syy), (sxy > 0) - (sxy < 0)


def _kernels() -> list:
    return [cor_numpy, cor_cython] if _CY else [cor_numpy]


def _assert_matches_exact(x: np.ndarray, y: np.ndarray, rtol: float = 1e-12) -> None:
    squared, sign = _exact_squared_cor(x, y)
    expected = sign * math.sqrt(float(squared))
    for kernel in _kernels():
        result = kernel(x, y)
        assert isinstance(result, float)
        np.testing.assert_allclose(result, expected, rtol=rtol, atol=0)


class TestCorrelationScaling:
    """Both kernels scale each input by a power of two before squaring."""

    def test_subnormal_square_example_matches_the_exact_correlation(self) -> None:
        x = np.ascontiguousarray(SUBNORMAL_SQUARE_INPUT)
        y = np.ascontiguousarray(x[::-1])
        _assert_matches_exact(x, y)
        for kernel in _kernels():
            np.testing.assert_allclose(kernel(x, y), -1 / 3, rtol=1e-12, atol=0)
        if _CY:
            np.testing.assert_allclose(cor_cython(x, y), cor_numpy(x, y), rtol=1e-10, atol=0)

    @pytest.mark.parametrize(
        "factors", [(1e-100,), (1.0,), (1e158,), (1e300,), (1e300, 1e10)], ids=str
    )
    def test_subnormal_square_example_is_scale_invariant(self, factors: tuple[float, ...]) -> None:
        # Every scaled copy, from 1e-258 up to 1e152, has the same
        # correlation, -1/3 (the last factor is applied in two steps because
        # 1e310 is not a float64).
        x = SUBNORMAL_SQUARE_INPUT
        for factor in factors:
            x = x * factor
        x = np.ascontiguousarray(x)
        assert np.all(np.isfinite(x))
        y = np.ascontiguousarray(x[::-1])
        for kernel in _kernels():
            np.testing.assert_allclose(kernel(x, y), -1 / 3, rtol=1e-12, atol=0)

    def test_large_magnitudes_no_longer_overflow(self) -> None:
        # Centred values near 1e155 square past float64's range; the raw
        # ``np.corrcoef`` gives nan with an overflow warning, the scaled
        # kernels give the exact correlation.
        x = np.array([1e155, -1e155, 3e154, 2.0])
        y = np.ascontiguousarray(x[::-1])
        with warnings.catch_warnings(record=True) as raw:
            warnings.simplefilter("always")
            assert np.isnan(np.corrcoef(x, y)[0, 1])
        # NumPy 2.x reports the overflow in ``dot`` and the invalid divide;
        # NumPy 1.26 only the divide. Either way the raw call warns and fails.
        assert {str(w.message) for w in raw} >= {"invalid value encountered in divide"}
        assert all(issubclass(w.category, RuntimeWarning) for w in raw)
        _assert_matches_exact(x, y)

    def test_tiny_variances_no_longer_underflow(self) -> None:
        # ``[1.1e-203, 0.0]`` has a variance that underflows to zero unscaled
        # (the case the property test used to skip); both kernels give -1
        # (to the ulp the ``sqrt(sxx) * sqrt(syy)`` denominator has always
        # cost on a two-point input, e.g. ``[1.0, 0.0]``).
        x = np.array([1.1e-203, 0.0])
        y = np.ascontiguousarray(x[::-1])
        for kernel in _kernels():
            np.testing.assert_allclose(kernel(x, y), -1.0, rtol=1e-15, atol=0)
            np.testing.assert_allclose(kernel(x * 1e203, y * 1e203), -1.0, rtol=1e-15, atol=0)

    def test_scale_exponent_brings_the_maximum_into_the_unit_binade_per_value(self) -> None:
        for values, expected in [
            (np.array([0.75, -0.5]), 0),
            (np.array([3.0, -1.0]), 2),
            (np.array([-1e-158, 0.0]), -524),
            (np.array([1e155, 2.0]), 515),
            (np.array([TINY, 0.0]), -1073),
            (np.array([HUGE, 0.0]), 1024),
            (np.array([0.0, 0.0]), 0),
            (np.array([np.nan, 1.0]), 0),
            (np.array([np.inf, 1.0]), 0),
            (np.array([1.0, -np.inf]), 0),
        ]:
            exponent = _scale_exponent(values)
            assert exponent == expected
            largest = float(np.max(np.abs(values)))
            if largest and math.isfinite(largest):
                # Applied per value with ldexp: the standalone factor 2**1073
                # would overflow, the scaled values never do.
                scaled = np.ldexp(values, -exponent)
                assert 0.5 <= float(np.max(np.abs(scaled))) < 1.0
                assert np.all(np.isfinite(scaled))

    # Finite-range boundaries: the smallest subnormal, the subnormal/normal
    # border, the largest float and same-sign or mixed-sign values whose
    # unscaled sums overflow. Each is checked against the exact rational
    # correlation of the float64 inputs; ``y`` is ``x`` reversed unless
    # given, so the expected values are simple (-1, -1/2, ...).
    @pytest.mark.parametrize(
        ("x", "y"),
        [
            (np.array([TINY, 0.0]), None),
            (np.array([TINY, 3 * TINY, 0.0, 2 * TINY]), None),
            (np.array([np.finfo(np.float64).tiny, 0.0, 1e-308]), None),
            (np.array([np.nextafter(np.finfo(np.float64).tiny, 0.0), 0.0, 1e-308]), None),
            (np.array([1e308, 1e308, 0.0]), None),
            (np.array([1e308, -1e308, 5e307, 0.0]), None),
            (np.array([HUGE, 0.0, -HUGE]), None),
            (np.array([HUGE, HUGE, 0.0, -HUGE]), None),
            (np.array([1e308, 1e308, 0.0]), np.array([TINY, 0.0, 2 * TINY])),
            (np.array([TINY, 0.0, 2 * TINY]), np.array([-1e308, 1e308, 1e308])),
        ],
        ids=[
            "smallest-subnormal",
            "subnormal-mix",
            "normal-border",
            "largest-subnormal",
            "same-sign-huge",
            "mixed-sign-huge",
            "max-float",
            "max-float-mixed",
            "independent-scales",
            "independent-scales-reversed",
        ],
    )
    def test_finite_range_boundaries_match_the_exact_correlation(
        self, x: np.ndarray, y: np.ndarray | None
    ) -> None:
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(x[::-1]) if y is None else np.ascontiguousarray(y)
        _assert_matches_exact(x, y)
        if _CY:
            np.testing.assert_allclose(cor_cython(x, y), cor_numpy(x, y), rtol=1e-10, atol=0)

    def test_smallest_subnormal_and_huge_sums_have_simple_expected_values(self) -> None:
        for kernel in _kernels():
            np.testing.assert_allclose(
                kernel(np.array([TINY, 0.0]), np.array([0.0, TINY])), -1.0, rtol=1e-15, atol=0
            )
            np.testing.assert_allclose(
                kernel(np.array([1e308, 1e308, 0.0]), np.array([0.0, 1e308, 1e308])),
                -0.5,
                rtol=1e-15,
                atol=0,
            )

    def test_ordinary_inputs_are_bit_identical_to_the_unscaled_arithmetic(self) -> None:
        # Where the unscaled arithmetic stays in the normal range the scaled
        # intermediates round the same way, so the NumPy kernel equals the raw
        # ``np.corrcoef`` scalar bit for bit and the Cython kernel equals a
        # transcription of its unscaled loop. Checked on 200 seeded inputs at
        # magnitudes 1e-100..1e100: evidence over that domain, not a proof.
        # The transcription rounds every operation separately, so the
        # Unix statistics builds use ``-ffp-contract=off`` (hatch_build.py);
        # Windows uses MSVC's precise mode. Without that Unix flag,
        # a compiler that fused ``s += a * b`` into one rounding (Apple clang
        # by default) moves 103 of these 200 results by one ulp.
        rng = np.random.default_rng(seed=20260915)
        for _ in range(200):
            n = int(rng.integers(2, 300))
            scale = 10.0 ** rng.uniform(-100, 100)
            x = np.ascontiguousarray(rng.standard_normal(n) * scale)
            y = np.ascontiguousarray(rng.standard_normal(n) * scale + 0.5 * x)
            assert cor_numpy(x, y) == float(np.corrcoef(x, y)[0, 1])
            if _CY:
                assert cor_cython(x, y) == _unscaled_cython_transcription(x, y)

    @pytest.mark.parametrize(
        "bad", [[np.nan, 1.0, 2.0], [np.inf, 1.0, 2.0], [1.0, np.inf, -np.inf], [-np.inf, 0.5, 1.0]]
    )
    def test_nonfinite_inputs_still_propagate_nan(self, bad: list[float]) -> None:
        x = np.array(bad)
        y = np.array([1.0, 2.0, 5.0])
        with warnings.catch_warnings():
            # ``np.corrcoef`` itself warns on an infinite centred value, as it
            # did before the scaling; the kernels add no warning of their own.
            warnings.simplefilter("ignore", RuntimeWarning)
            for kernel in _kernels():
                assert math.isnan(kernel(x, y))
                assert math.isnan(kernel(y, x))


def _unscaled_cython_transcription(x: np.ndarray, y: np.ndarray) -> float:
    """The pre-scaling ``cor_cython`` loop in Python floats, for bit comparison."""
    xs = x.tolist()
    ys = y.tolist()
    n = len(xs)
    sx = xs[0]
    sy = ys[0]
    for i in range(1, n):
        sx += xs[i]
        sy += ys[i]
    mx = sx / n
    my = sy / n
    sxx = syy = sxy = 0.0
    for i in range(n):
        dx = xs[i] - mx
        dy = ys[i] - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    return sxy / (math.sqrt(sxx) * math.sqrt(syy))


# ---------------------------------------------------------------------------
# Public API agreement — mean/var/std/cor still match np.* after dispatch
# ---------------------------------------------------------------------------


class TestPublicStatsAgreeWithNumpy:
    """``mean(t)`` / ``std(t)`` / ``var(t)`` / ``cor(x, y)`` agree with np.* .

    Belt-and-braces: the existing ``TestStatisticsBDaily`` covers the
    behaviour, but explicit equivalence tests against ``np.mean`` /
    ``np.std`` / ``np.var`` / ``np.corrcoef`` lock the Cython-dispatch
    branch's output shape and value to the NumPy reference, so a future
    regression in the dispatcher (wrong dtype check, ravel-with-copy,
    ddof off-by-one) shows up as a failing test rather than a silent
    drift.
    """

    @pytest.mark.parametrize("arr", _gen_arrays())
    def test_mean_tseries_matches_numpy(self, arr: np.ndarray) -> None:
        t = TSeries(qq(2020, 1), arr.astype(np.float64))
        np.testing.assert_allclose(float(mean(t)), float(np.mean(arr)), rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("arr", _gen_arrays())
    def test_var_tseries_matches_numpy_ddof1(self, arr: np.ndarray) -> None:
        if arr.shape[0] < 2:
            pytest.skip("ddof=1 needs at least 2 elements")
        t = TSeries(qq(2020, 1), arr.astype(np.float64))
        np.testing.assert_allclose(
            float(var(t)), float(np.var(arr, ddof=1)), rtol=1e-12, atol=1e-15
        )

    @pytest.mark.parametrize("arr", _gen_arrays())
    def test_std_tseries_matches_numpy_ddof1(self, arr: np.ndarray) -> None:
        if arr.shape[0] < 2:
            pytest.skip("ddof=1 needs at least 2 elements")
        t = TSeries(qq(2020, 1), arr.astype(np.float64))
        np.testing.assert_allclose(
            float(std(t)), float(np.std(arr, ddof=1)), rtol=1e-12, atol=1e-15
        )

    def test_cor_two_tseries_matches_numpy(self) -> None:
        rng = np.random.default_rng(seed=20260516)
        x = TSeries(qq(2020, 1), rng.standard_normal(100))
        y = TSeries(qq(2020, 1), rng.standard_normal(100))
        np.testing.assert_allclose(
            float(cor(x, y)),
            float(np.corrcoef(x.values, y.values)[0, 1]),
            rtol=1e-12,
            atol=1e-15,
        )

    def test_mean_mvts_matches_numpy(self) -> None:
        # Julia's mean(::MVTSeries) iterates the matrix flat. The
        # dispatcher ravels a contiguous 2-D array into a 1-D view (no
        # copy), so the kernel applies just as for TSeries.
        rng = np.random.default_rng(seed=20260516)
        values = rng.standard_normal((100, 5))
        m = MVTSeries(qq(2020, 1), ["a", "b", "c", "d", "e"], values)
        np.testing.assert_allclose(float(mean(m)), float(np.mean(values)), rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# Kernel-direct fallback (always-callable NumPy reference)
# ---------------------------------------------------------------------------


class TestStatsKernelFallback:
    """The NumPy kernels are always callable, regardless of compile state."""

    def test_mean_numpy_returns_python_float(self) -> None:
        result = mean_numpy(np.arange(100.0))
        assert isinstance(result, float)
        assert result == 49.5

    def test_var_numpy_returns_python_float(self) -> None:
        result = var_numpy(np.arange(100.0), 1)
        assert isinstance(result, float)
        # var of arange(100) with ddof=1 is 841.6666...
        np.testing.assert_allclose(result, np.var(np.arange(100.0), ddof=1), rtol=1e-12)

    def test_std_numpy_returns_python_float(self) -> None:
        result = std_numpy(np.arange(100.0), 1)
        assert isinstance(result, float)
        np.testing.assert_allclose(result, np.std(np.arange(100.0), ddof=1), rtol=1e-12)

    def test_cor_numpy_returns_python_float(self) -> None:
        rng = np.random.default_rng(seed=20260516)
        x = rng.standard_normal(100)
        y = rng.standard_normal(100)
        result = cor_numpy(x, y)
        assert isinstance(result, float)
        np.testing.assert_allclose(result, np.corrcoef(x, y)[0, 1], rtol=1e-12)

    def test_stats_is_cython_returns_bool(self) -> None:
        assert isinstance(stats_is_cython(), bool)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestStatsKernelEdgeCases:
    """Empty / degenerate inputs follow NumPy semantics."""

    def test_mean_length_1_returns_value(self) -> None:
        assert mean_numpy(np.array([42.0])) == 42.0

    def test_cor_perfect_positive_correlation(self) -> None:
        x = np.arange(1.0, 101.0)
        y = 2.0 * x + 1.0
        np.testing.assert_allclose(cor_numpy(x, y), 1.0, rtol=1e-12)

    def test_cor_perfect_negative_correlation(self) -> None:
        x = np.arange(1.0, 101.0)
        y = -3.0 * x + 7.0
        np.testing.assert_allclose(cor_numpy(x, y), -1.0, rtol=1e-12)

    def test_mean_propagates_nan(self) -> None:
        result = mean_numpy(np.array([1.0, 2.0, np.nan, 4.0]))
        assert np.isnan(result)

    def test_public_cor_constant_tseries_returns_nan(self) -> None:
        # Public-surface lock for the constant-input convention: tsecon.cor
        # on two constant TSeries emits a RuntimeWarning and returns nan,
        # regardless of which kernel is dispatched. Locks the docstring's
        # "Notes" claim against a future regression where one kernel diverges.
        t = TSeries(qq(2020, 1), np.full(10, 7.5))
        with pytest.warns(RuntimeWarning, match="constant input"):
            result = cor(t, t)
        assert np.isnan(result)
