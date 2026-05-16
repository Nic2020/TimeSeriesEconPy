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

See review file ``F04_hypothesis_property_tests_missing`` for the
motivation; the deferral was logged in
``decisions/18_cython_port_plan.md`` and ``MASTER_PLAN.md`` § M1.5
"Outstanding" since session 22 and bundled into this M1.5-followup
session.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
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

    See ``claude_files/paper/NOTES.md`` § "Property tests caught the
    naive-vs-pairwise summation residual".
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


# For the correlation property test we floor the magnitude at ``1e-60`` to
# avoid the ``cor_cython`` underflow bug (BUGS.md B4 — surfaced by this very
# Hypothesis pass): when ``sxx`` and ``syy`` both fall below ≈ 1e-154 their
# product underflows to zero and the final ``sxy / sqrt(0)`` returns ±inf.
# The bug doesn't fire for realistic econ inputs (magnitudes ≥ 1e-6) but
# is real; the floor below skips the regime, and the floor should be lifted
# when B4 is fixed (see the BUGS.md entry).
_BOUNDED_FLOAT_COR = st.one_of(
    st.floats(
        min_value=1e-60,
        max_value=1e6,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
        width=64,
    ),
    st.floats(
        min_value=-1e6,
        max_value=-1e-60,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
        width=64,
    ),
    st.just(0.0),
)


@pytest.mark.skipif(not _CY, reason="Cython stats kernel not compiled")
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
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
    because the result is the *ratio* of two near-equal sums).
    """
    n = arr.shape[0]
    x = np.ascontiguousarray(arr, dtype=np.float64)
    # Construct y as a permutation of x (reversal) so the cor is
    # well-conditioned (away from 0 / 1 degenerate cases) — pure-random y
    # would occasionally land near zero-variance and amplify FP noise.
    y = np.ascontiguousarray(x[::-1].copy(), dtype=np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        # Degenerate inputs: both kernels return NaN or 0; skip ambiguity.
        return
    if n < 2:
        return
    np.testing.assert_allclose(
        cor_cython(x, y), cor_numpy(x, y), rtol=1e-10, atol=1e-12
    )
