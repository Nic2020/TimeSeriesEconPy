# SPDX-License-Identifier: MIT
"""Hypothesis property tests for the cumsum-anchored kernel pair.

Generates random ``(values, offset, count, anchor_value, anchor_relative_idx)``
tuples and asserts the NumPy reference and Cython implementations of
:func:`tsecon._math_kernels.cumsum_anchored_numpy` produce equivalent
output at ``rtol=1e-12``. Backstops the hand-picked parametric cases in
``test_math.py::TestCumsumKernelsAgreeOnArrays`` — both kernels
implement the same anchored-cumsum semantics (cumsum over the slice +
constant-shift correction), so the agreement is structural rather than
coincidental, and any divergence is a real bug in one of the two
kernels.

Mirrors the four existing per-port property test files
(``test_rec_linear_properties.py``, ``test_indexing_properties.py``,
``test_stats_kernel_properties.py``, ``test_fconvert_kernel_properties.py``)
established by the M1.5 ports and extended here for the M1.6.2 fifth
port.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from tsecon._math_kernels import cumsum_anchored_numpy

try:
    from tsecon._math_kernels_cy import (  # type: ignore[import-not-found]
        cumsum_anchored_cython,
    )

    _CY = True
except ImportError:
    _CY = False


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


# Bounded float magnitude keeps the cumsum from overflowing across long
# chunks — both kernels are bit-equivalent regardless of magnitude, but
# ``assert_allclose`` at ``rtol=1e-12`` is only meaningful for finite,
# well-conditioned outputs. The bound ``|x| < 1e6`` lets length-200
# chunks sum to ~2e8 without floating-point issues.
_FINITE_FLOAT = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    width=64,
)
_FINITE_OR_NAN = st.one_of(_FINITE_FLOAT, st.just(float("nan")))
_FINITE_ANCHOR = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    width=64,
)


@st.composite
def _cumsum_inputs(  # type: ignore[no-untyped-def]
    draw,
) -> tuple[np.ndarray, int, int, float, int]:
    """Generate ``(values, offset, count, anchor_value, anchor_relative_idx)`` per the contract.

    Constraints satisfied:
      * ``values.dtype == float64`` and ``values`` is C-contiguous.
      * ``count >= 1`` (count==0 is locked separately in the parametric
        test; the random case carries actual work).
      * ``0 <= offset`` and ``offset + count <= len(values)``.
      * Either ``anchor_relative_idx == -1`` (the ``undiff_inplace``
        regime where reference cumsum is 0) or ``0 <= anchor_relative_idx
        < count`` (anchor inside the chunk).
      * ``anchor_value`` is finite, bounded so the shifted output stays
        in normal float range.

    The strategy mixes NaN into the value buffer with low probability so
    that the kernels' NaN-propagation agreement is exercised without
    overwhelming the well-conditioned cases.
    """
    count = draw(st.integers(min_value=1, max_value=200))
    # Always-present prefix and suffix to verify the kernel respects
    # offset / count boundaries (positions outside the chunk are bit-exact
    # untouched on exit).
    prefix_len = draw(st.integers(min_value=0, max_value=5))
    suffix_len = draw(st.integers(min_value=0, max_value=5))
    total = prefix_len + count + suffix_len
    include_nan = draw(st.booleans())
    elements = _FINITE_OR_NAN if include_nan else _FINITE_FLOAT
    values = np.asarray(draw(npst.arrays(dtype=np.float64, shape=total, elements=elements)))
    anchor_value = draw(_FINITE_ANCHOR)
    # Anchor regime: -1 (before chunk) or in [0, count).
    anchor_relative_idx = draw(
        st.one_of(st.just(-1), st.integers(min_value=0, max_value=count - 1))
    )
    return values, prefix_len, count, anchor_value, anchor_relative_idx


# ---------------------------------------------------------------------------
# Property: NumPy ≡ Cython on arbitrary valid inputs
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CY, reason="Cython cumsum-anchored kernel not compiled")
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_cumsum_inputs())
def test_cumsum_anchored_numpy_matches_cython(
    inputs: tuple[np.ndarray, int, int, float, int],
) -> None:
    """NumPy and Cython kernels agree at ``rtol=1e-12`` on Hypothesis-generated inputs."""
    values, offset, count, anchor_value, anchor_relative_idx = inputs
    v_numpy = values.copy()
    v_cython = values.copy()
    cumsum_anchored_numpy(v_numpy, offset, count, anchor_value, anchor_relative_idx)
    cumsum_anchored_cython(v_cython, offset, count, anchor_value, anchor_relative_idx)
    # ``equal_nan=True`` because the input strategy occasionally injects NaN;
    # both kernels propagate it identically.
    np.testing.assert_allclose(v_numpy, v_cython, rtol=1e-12, atol=1e-15, equal_nan=True)


# ---------------------------------------------------------------------------
# Property: positions outside [offset, offset+count) are bit-exact untouched
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_cumsum_inputs())
def test_cumsum_anchored_numpy_leaves_outside_positions_untouched(
    inputs: tuple[np.ndarray, int, int, float, int],
) -> None:
    """The NumPy reference must not touch positions outside the integrated chunk."""
    values, offset, count, anchor_value, anchor_relative_idx = inputs
    before = values.copy()
    cumsum_anchored_numpy(values, offset, count, anchor_value, anchor_relative_idx)
    np.testing.assert_array_equal(values[:offset], before[:offset])
    np.testing.assert_array_equal(values[offset + count :], before[offset + count :])


# ---------------------------------------------------------------------------
# Property: anchor invariant — the integrated value at the anchor position
# must equal anchor_value (modulo FP rounding) when the anchor is inside
# the chunk and the input has no NaNs.
# ---------------------------------------------------------------------------


@st.composite
def _cumsum_inputs_finite_anchor_inside(  # type: ignore[no-untyped-def]
    draw,
) -> tuple[np.ndarray, int, int, float, int]:
    """``_cumsum_inputs`` restricted to: anchor inside chunk + no NaN values."""
    count = draw(st.integers(min_value=1, max_value=200))
    prefix_len = draw(st.integers(min_value=0, max_value=5))
    suffix_len = draw(st.integers(min_value=0, max_value=5))
    total = prefix_len + count + suffix_len
    values = np.asarray(draw(npst.arrays(dtype=np.float64, shape=total, elements=_FINITE_FLOAT)))
    anchor_value = draw(_FINITE_ANCHOR)
    anchor_relative_idx = draw(st.integers(min_value=0, max_value=count - 1))
    return values, prefix_len, count, anchor_value, anchor_relative_idx


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_cumsum_inputs_finite_anchor_inside())
def test_cumsum_anchored_numpy_satisfies_anchor_invariant(
    inputs: tuple[np.ndarray, int, int, float, int],
) -> None:
    """``result[offset + anchor_relative_idx] == anchor_value`` after the kernel runs.

    The anchor is mathematically exact, but FP rounding in the cumsum +
    subtract-and-add-back roundtrip leaves a residual on the order of
    ``ulp(cumsum_magnitude)``. We allow up to ``8 * count * eps`` of the
    chunk's max magnitude, which is well inside numpy's pairwise-sum
    accuracy bound for length-N sums.
    """
    values, offset, count, anchor_value, anchor_relative_idx = inputs
    chunk_before = values[offset : offset + count].copy()
    cumsum_anchored_numpy(values, offset, count, anchor_value, anchor_relative_idx)
    chunk_max = float(np.max(np.abs(chunk_before))) if count > 0 else 0.0
    cumsum_scale = chunk_max * count
    atol = 8 * np.finfo(np.float64).eps * max(cumsum_scale, abs(anchor_value), 1.0)
    np.testing.assert_allclose(
        values[offset + anchor_relative_idx], anchor_value, rtol=0.0, atol=atol
    )
