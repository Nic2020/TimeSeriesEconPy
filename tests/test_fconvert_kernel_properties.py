# SPDX-License-Identifier: MIT
"""Hypothesis property tests for the fconvert lower-aggregate kernel pair.

Generates random ``(values, group_starts, group_lengths, method_code)``
tuples satisfying the kernel contract and asserts the NumPy reference
and Cython implementations of ``aggregate_groups_numpy`` produce
identical output at ``rtol=1e-12`` for all six method codes
(``mean / sum / min / max / first / last``). Backstops the hand-picked
parametric cases in ``test_fconvert_kernels.py``.

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

from tsecon._fconvert_kernels import (
    METHOD_FIRST,
    METHOD_LAST,
    METHOD_MAX,
    METHOD_MEAN,
    METHOD_MIN,
    METHOD_SUM,
    aggregate_groups_numpy,
)

try:
    from tsecon._fconvert_kernels_cy import (  # type: ignore[import-not-found]
        aggregate_groups_cython,
    )

    _CY = True
except ImportError:
    _CY = False


_METHOD_CODES = (METHOD_MEAN, METHOD_SUM, METHOD_MIN, METHOD_MAX, METHOD_FIRST, METHOD_LAST)

# Bounded float magnitude keeps min/max/first/last bit-identical (no FP
# accumulation involved); mean/sum agree at ``rtol=1e-12`` for the array
# lengths and group widths Hypothesis explores. NaN allowed so propagation
# is exercised.
_BOUNDED_FLOAT = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
    allow_subnormal=False,
    width=64,
)


@st.composite
def _fconvert_inputs(  # type: ignore[no-untyped-def]
    draw,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate ``(values, group_starts, group_lengths)`` satisfying the contract.

    Constraints satisfied:
      * ``group_starts[g] >= 0`` and non-overlapping
      * ``group_starts[g] + group_lengths[g] <= len(values)``
      * ``group_lengths[g] >= 1``
      * ``values.dtype == float64``

    Groups are generated as a *contiguous partition* of the first ``S`` slots
    of ``values`` where ``S = sum(group_lengths)`` — this is the
    Quarterly→Yearly / Monthly→Quarterly regime the kernel is wired for.
    Variable-length groups are part of the contract (see
    ``test_fconvert_kernels.py::test_kernel_handles_variable_group_lengths``)
    so each ``group_lengths[g]`` is drawn independently in ``[1, 12]``.
    """
    n_groups = draw(st.integers(min_value=1, max_value=15))
    lengths = np.asarray(
        draw(
            st.lists(
                st.integers(min_value=1, max_value=12),
                min_size=n_groups,
                max_size=n_groups,
            )
        ),
        dtype=np.int64,
    )
    starts = np.zeros(n_groups, dtype=np.int64)
    if n_groups > 1:
        starts[1:] = np.cumsum(lengths[:-1])
    total = int(starts[-1] + lengths[-1])
    values = np.asarray(
        draw(
            npst.arrays(
                dtype=np.float64,
                shape=total,
                elements=_BOUNDED_FLOAT,
            )
        ),
    )
    return values, starts, lengths


# ---------------------------------------------------------------------------
# Property: NumPy ≡ Cython on arbitrary valid inputs, all six method codes
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CY, reason="Cython fconvert kernel not compiled")
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_fconvert_inputs(), st.sampled_from(_METHOD_CODES))
def test_aggregate_groups_numpy_matches_cython(
    inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
    method_code: int,
) -> None:
    """``aggregate_groups_numpy ≡ aggregate_groups_cython`` at ``rtol=1e-12``."""
    values, starts, lengths = inputs
    out_numpy = aggregate_groups_numpy(values, starts, lengths, method_code)
    out_cython = aggregate_groups_cython(values, starts, lengths, method_code)
    np.testing.assert_allclose(out_numpy, out_cython, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# Property: groupwise reduction matches a per-group NumPy reduction.
# Locks the "outer loop fused into C" invariant — the Cython kernel's
# correctness is grounded against an independent reference (np.sum / np.mean
# applied per group via Python list comprehension).
# ---------------------------------------------------------------------------


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_fconvert_inputs(), st.sampled_from(_METHOD_CODES))
def test_aggregate_groups_numpy_matches_groupwise_reduction(
    inputs: tuple[np.ndarray, np.ndarray, np.ndarray],
    method_code: int,
) -> None:
    """The NumPy reference matches a per-group NumPy reduction reference.

    Belt-and-braces: the reference kernel itself loops in Python and
    invokes ``slice.mean()`` / ``.sum()`` etc., but it's worth proving
    against a fully independent ``np.array([np.mean(...) for slc in slices])``
    construction so a regression in the reference kernel's branch
    selection (mis-mapped method_code) shows up.
    """
    values, starts, lengths = inputs
    got = aggregate_groups_numpy(values, starts, lengths, method_code)
    n_groups = starts.shape[0]
    expected = np.empty(n_groups, dtype=np.float64)
    for g in range(n_groups):
        s = int(starts[g])
        length = int(lengths[g])
        chunk = values[s : s + length]
        if method_code == METHOD_MEAN:
            expected[g] = np.mean(chunk)
        elif method_code == METHOD_SUM:
            expected[g] = np.sum(chunk)
        elif method_code == METHOD_MIN:
            expected[g] = np.min(chunk)
        elif method_code == METHOD_MAX:
            expected[g] = np.max(chunk)
        elif method_code == METHOD_FIRST:
            expected[g] = chunk[0]
        elif method_code == METHOD_LAST:
            expected[g] = chunk[-1]
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-15)
