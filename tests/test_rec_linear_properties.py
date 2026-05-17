# SPDX-License-Identifier: MIT
"""Hypothesis property tests for the rec_linear kernel pair.

Generates random ``(values, offset, count, coeffs, lags)`` tuples and
asserts the NumPy reference and Cython implementations of
:func:`tsecon._rec_kernels.rec_linear_numpy` produce bit-for-bit
equivalent output at ``rtol=1e-12``. Backstops the hand-picked
parametric cases in ``test_recursive.py`` — both kernels run the same
inner-loop arithmetic (innermost accumulation over ``k`` in lockstep),
so the agreement is structural rather than coincidental, and any
divergence is a real bug in one of the two kernels.

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

from tsecon._rec_kernels import rec_linear_numpy

try:
    from tsecon._rec_kernels_cy import (  # type: ignore[import-not-found]
        rec_linear_cython,
    )

    _CY = True
except ImportError:
    _CY = False


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


# Bounded coefficient magnitude keeps the recurrence from overflowing to inf
# across ``count`` iterations — the kernels are bit-identical regardless of
# magnitude, but ``assert_allclose`` at ``rtol=1e-12`` is only meaningful for
# finite, well-conditioned outputs. The bound ``|coeff| < 0.9 / n_terms``
# gives |Σ c_k| < 0.9 < 1, so the recurrence decays.
_FINITE_FLOAT = st.floats(
    min_value=-1.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    width=64,
)


@st.composite
def _rec_linear_inputs(  # type: ignore[no-untyped-def]
    draw,
) -> tuple[np.ndarray, int, int, int, np.ndarray, np.ndarray]:
    """Generate ``(values, offset, count, step, coeffs, lags)`` satisfying the kernel contract.

    Constraints satisfied (forward case, ``step == +1``):
      * ``coeffs.shape == lags.shape``
      * ``min(lags) >= 1``
      * ``offset >= max(lags)`` (every read is in-range)
      * ``offset + count <= len(values)`` (every write is in-range)

    Constraints satisfied (backward case, ``step == -1``, M1.6.1):
      * ``coeffs.shape == lags.shape``
      * ``max(lags) <= -1`` (lags mirrored relative to forward direction)
      * ``offset + count <= len(values)`` writes go backward from offset
        so the actual write extent is ``[offset - (count - 1), offset]``;
        we require ``offset - (count - 1) >= 0``.
      * ``offset - min(lags) < len(values)`` (the furthest read forward
        of offset is in-range; lags are negative so this is ``offset + |min|``).

    Initial conditions in the non-written positions are bounded; coefficients
    are bounded so the recurrence cannot overflow.
    """
    direction = draw(st.sampled_from([1, -1]))
    n_terms = draw(st.integers(min_value=1, max_value=4))
    # Unique magnitudes in 1..8; sign matches direction so the recurrence
    # always reads an already-written or initial-condition position.
    mags = sorted(draw(st.lists(st.integers(1, 8), min_size=n_terms, max_size=n_terms, unique=True)))
    max_mag = mags[-1]
    count = draw(st.integers(min_value=1, max_value=40))
    total_length = max_mag + count
    coeff_scale = 0.9 / n_terms
    coeffs = np.asarray(
        draw(
            st.lists(
                st.floats(
                    min_value=-coeff_scale,
                    max_value=coeff_scale,
                    allow_nan=False,
                    allow_infinity=False,
                    allow_subnormal=False,
                    width=64,
                ),
                min_size=n_terms,
                max_size=n_terms,
            )
        ),
        dtype=np.float64,
    )
    lags = np.asarray([direction * m for m in mags], dtype=np.int64)
    values = np.zeros(total_length, dtype=np.float64)
    if direction == 1:
        # Forward: offset = max_mag (so reads land in [0, max_mag-1]); init the
        # non-written prefix with bounded floats.
        offset = max_mag
        init = np.asarray(
            draw(
                npst.arrays(
                    dtype=np.float64,
                    shape=offset,
                    elements=_FINITE_FLOAT,
                )
            ),
        )
        values[:offset] = init
    else:
        # Backward: offset = count - 1 (so writes land in [0, count-1]); init
        # the trailing positions [count, total_length-1] = max_mag positions
        # with bounded floats. The kernel reads values[offset - lag] = values[
        # (count-1) + |lag|] which lies in that init buffer.
        offset = count - 1
        init = np.asarray(
            draw(
                npst.arrays(
                    dtype=np.float64,
                    shape=max_mag,
                    elements=_FINITE_FLOAT,
                )
            ),
        )
        values[count:] = init
    return values, offset, count, direction, coeffs, lags


# ---------------------------------------------------------------------------
# Property: NumPy ≡ Cython on arbitrary valid inputs (forward + backward)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CY, reason="Cython rec_linear kernel not compiled")
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_rec_linear_inputs())
def test_rec_linear_numpy_matches_cython(
    inputs: tuple[np.ndarray, int, int, int, np.ndarray, np.ndarray],
) -> None:
    """NumPy and Cython kernels agree at ``rtol=1e-12`` on Hypothesis-generated inputs."""
    values, offset, count, step, coeffs, lags = inputs
    v_numpy = values.copy()
    v_cython = values.copy()
    rec_linear_numpy(v_numpy, offset, count, step, coeffs, lags)
    rec_linear_cython(v_cython, offset, count, step, coeffs, lags)
    # ``equal_nan=True`` because NaN may appear in the init buffer (kernels
    # propagate it identically); inf-comparison left default since the
    # generator bounds prevent overflow.
    np.testing.assert_allclose(v_numpy, v_cython, rtol=1e-12, atol=1e-15, equal_nan=True)


# ---------------------------------------------------------------------------
# Property: the kernel is deterministic on a fixed input (idempotence under
# re-run from the same init buffer).
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_rec_linear_inputs())
def test_rec_linear_numpy_is_deterministic(
    inputs: tuple[np.ndarray, int, int, int, np.ndarray, np.ndarray],
) -> None:
    """Two runs of the NumPy kernel from the same init produce identical output."""
    values, offset, count, step, coeffs, lags = inputs
    a = values.copy()
    b = values.copy()
    rec_linear_numpy(a, offset, count, step, coeffs, lags)
    rec_linear_numpy(b, offset, count, step, coeffs, lags)
    np.testing.assert_array_equal(a, b)
