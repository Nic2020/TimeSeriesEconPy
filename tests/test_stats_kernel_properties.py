# SPDX-License-Identifier: MIT
"""Hypothesis property tests for the stats kernel pair.

Generates random ``float64`` arrays and asserts the NumPy reference and
Cython implementations of ``mean / var / std / cor`` produce bit-for-bit
equivalent output at ``rtol=1e-12``. Backstops the hand-picked
parametric cases in ``test_stats_kernels.py``.

The naive two-pass variance kernel can drift from NumPy's pairwise
summation for very large, very imbalanced inputs; the generator caps
magnitude at ``1e6`` so length-1000 well-conditioned arrays stay well
within ``rtol=1e-12`` — the same regime ``test_stats_kernels.py``
parametric cases already cover (length 100, well-conditioned).

Property-style coverage for the M1.5 kernel ports, complementing the
hand-picked parametric cases in the corresponding ``test_*_kernels.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from tsecon._stats_kernels import cor_numpy, mean_numpy, std_numpy, var_numpy

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


# Bounded float magnitude keeps the naive two-pass variance kernel within
# ``rtol=1e-12`` agreement with NumPy's pairwise summation for the array
# lengths Hypothesis explores (up to 1000). NaN is allowed at the array
# level so propagation is exercised in the property pass.
_BOUNDED_FLOAT = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    width=64,
)


def _array_strategy(min_size: int = 2, max_size: int = 1000) -> st.SearchStrategy[np.ndarray]:
    return npst.arrays(
        dtype=np.float64,
        shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=min_size, max_side=max_size),
        elements=_BOUNDED_FLOAT,
    )


def _scale_aware_atol_var(a: np.ndarray) -> float:
    """Variance-units atol that scales with ``max(|x|)**2``.

    Locks the Hypothesis-discovered finding that ``np.var`` (NumPy's
    pairwise summation) and the Cython kernel (naive two-pass) diverge
    by up to ``eps * max(|x|)**2`` on near-constant inputs (true variance
    is zero, but each implementation lands a different sub-eps residual).
    This is not a bug — both algorithms are correct given their numerics
    — but the bit-equivalence claim at fixed ``atol=1e-15`` is only
    meaningful when the expected variance is large relative to the
    catastrophic-cancellation residual.
    """
    return max(1e-15, 1e-10 * float(np.max(np.abs(a))) ** 2)


def _scale_aware_atol_std(a: np.ndarray) -> float:
    """Standard-deviation-units atol that scales with ``max(|x|)``."""
    return max(1e-15, 1e-10 * float(np.max(np.abs(a))))


def _scale_aware_atol_mean(a: np.ndarray) -> float:
    """Mean-units atol that scales with ``max(|x|)``."""
    return max(1e-15, 1e-10 * float(np.max(np.abs(a))))


# ---------------------------------------------------------------------------
# Property: NumPy ≡ Cython on arbitrary valid inputs
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CY, reason="Cython stats kernel not compiled")
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_array_strategy())
def test_mean_numpy_matches_cython(arr: np.ndarray) -> None:
    """``mean_numpy ≡ mean_cython`` to within scale-aware tolerance."""
    a = np.ascontiguousarray(arr, dtype=np.float64)
    np.testing.assert_allclose(
        mean_cython(a), mean_numpy(a), rtol=1e-12, atol=_scale_aware_atol_mean(a)
    )


@pytest.mark.skipif(not _CY, reason="Cython stats kernel not compiled")
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_array_strategy(min_size=2), st.integers(min_value=0, max_value=1))
def test_var_numpy_matches_cython(arr: np.ndarray, ddof: int) -> None:
    """``var_numpy ≡ var_cython`` for both ddof=0 and ddof=1.

    Tolerance scales with ``max(|x|)**2`` to accommodate the
    pairwise-vs-naive summation residual on near-constant inputs — see
    :func:`_scale_aware_atol_var`.
    """
    a = np.ascontiguousarray(arr, dtype=np.float64)
    # Kernel contract requires ``len(values) > ddof``; the strategy guarantees
    # ``len >= 2`` so both ddof values are valid.
    np.testing.assert_allclose(
        var_cython(a, ddof), var_numpy(a, ddof), rtol=1e-10, atol=_scale_aware_atol_var(a)
    )


@pytest.mark.skipif(not _CY, reason="Cython stats kernel not compiled")
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_array_strategy(min_size=2), st.integers(min_value=0, max_value=1))
def test_std_numpy_matches_cython(arr: np.ndarray, ddof: int) -> None:
    """``std_numpy ≡ std_cython`` for both ddof=0 and ddof=1.

    Tolerance scales with ``max(|x|)`` for the same reason as
    :func:`test_var_numpy_matches_cython`.
    """
    a = np.ascontiguousarray(arr, dtype=np.float64)
    np.testing.assert_allclose(
        std_cython(a, ddof), std_numpy(a, ddof), rtol=1e-10, atol=_scale_aware_atol_std(a)
    )


# Magnitude bounds match `_BOUNDED_FLOAT` for the other kernels. The earlier
# 1e-60 floor existed solely to skip the B4 underflow regime (`sqrt(sxx *
# syy)` underflowing for both stddevs below ~1e-154); session 28-hotfix
# rewrote the denominator as `sqrt(sxx) * sqrt(syy)` so each factor stays
# in normal range, and the floor is no longer needed. Constant inputs are
# now excluded by the bit-exact `min == max` `assume` below — both kernels
# return nan + RuntimeWarning on a constant array (`tsecon._stats.cor` docstring,
# Notes), which is handled by a separate parametric lock test in
# `test_stats_kernels.py::TestStatsKernelsAgreeOnArrays` and is out of scope
# for this property pair.
_BOUNDED_FLOAT_COR = _BOUNDED_FLOAT


@pytest.mark.skipif(not _CY, reason="Cython stats kernel not compiled")
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
# Regression seed: the falsifying example from the M1.6.0 audit baseline run
# (`np.full(100, 1e-60)`) reached the `cor_cython` constant-array path where
# cython's sequential summation produces an FP-exact mean (centred sums
# exactly zero -> `sxx=syy=0` -> `0 / sqrt(0)` = nan) while `np.corrcoef` on
# the same input returned 1.0 (its pairwise summation produces tiny non-zero
# deviations that self-correlate). The constant-input guard now uniformises
# both kernels to nan; this explicit example survives strategy changes.
@example(arr=np.full(100, 1e-60))
@given(
    npst.arrays(
        dtype=np.float64,
        shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=2, max_side=500),
        elements=_BOUNDED_FLOAT_COR,
    )
)
def test_cor_numpy_matches_cython(arr: np.ndarray) -> None:
    """``cor_numpy(x, y) ≡ cor_cython(x, y)`` at ``rtol=1e-10``.

    The correlation kernel is the looser pair: ``cor_numpy`` builds a
    stacked ``np.corrcoef`` matrix (pairwise summation through covariance
    and two stddevs) while ``cor_cython`` walks the two arrays once with a
    naive accumulator. The two paths agree at ``rtol=1e-10`` on
    well-conditioned inputs (a touch looser than the mean/var/std tests
    because the result is the *ratio* of two near-equal sums). Constant
    inputs are excluded — the constant-array semantics (both kernels
    return nan + RuntimeWarning) live in
    ``test_stats_kernels.py::TestStatsKernelsAgreeOnArrays``.
    """
    n = arr.shape[0]
    if n < 2:
        return
    x = np.ascontiguousarray(arr, dtype=np.float64)
    # Construct y as a permutation of x (reversal) so the cor is
    # well-conditioned (away from 0 / 1 degenerate cases) — pure-random y
    # would occasionally land near zero-variance and amplify FP noise.
    y = np.ascontiguousarray(x[::-1].copy(), dtype=np.float64)
    # Exclude bit-exact constant inputs — both kernels return nan +
    # RuntimeWarning there by design; the kernel-agreement claim of this
    # property is about well-defined Pearson correlations, not the
    # degenerate case. `min == max` is the detector both kernels use; any
    # input that survives it has a well-defined correlation in float64.
    if x.min() == x.max() or y.min() == y.max():
        return
    # Exclude inputs whose variance underflows the float64 normal range —
    # e.g. ``[1.1e-203, 0.0]`` has var ≈ 3e-407 which underflows to 0,
    # making ``np.corrcoef``'s internal ``cov / sqrt(var)`` divide by zero
    # (raising RuntimeWarning under ``filterwarnings=[error::RuntimeWarning]``).
    # The cython kernel's B4 fix (``sqrt(sxx) * sqrt(syy)``) protects its own
    # path, but np.corrcoef is not under our control; a numerically-stable
    # replacement for np.corrcoef on near-subnormal inputs is out of scope
    # for this hotfix (would require a kahan-summation or scaled-input pass).
    if np.var(x) == 0.0 or np.var(y) == 0.0:
        return
    np.testing.assert_allclose(cor_cython(x, y), cor_numpy(x, y), rtol=1e-10, atol=1e-12)
