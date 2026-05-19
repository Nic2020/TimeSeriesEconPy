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
from tsecon._stats_kernels import cor_numpy, mean_numpy, std_numpy, var_numpy

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
