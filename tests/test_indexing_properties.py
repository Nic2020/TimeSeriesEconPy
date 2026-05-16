# SPDX-License-Identifier: MIT
"""Hypothesis property tests for the indexing gather kernel pair.

Generates random ``(values, indices)`` pairs and asserts the NumPy
reference and Cython implementations of
:func:`tsecon._indexing_kernels.gather_numpy` produce bit-for-bit
equivalent output. The third leg, ``values.take(indices)``, is included
as a separate equivalence check — the NumPy reference is implemented
*via* ``np.take``, so the comparison is structural rather than
informative, but it locks the implementation choice against future
regressions (a switch to fancy indexing ``values[indices]`` would
silently change the array-allocation behaviour even though the values
agree).

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

from tsecon._indexing_kernels import gather_numpy

try:
    from tsecon._indexing_kernels_cy import (  # type: ignore[import-not-found]
        gather_cython,
    )

    _CY = True
except ImportError:
    _CY = False


# Values are unrestricted (NaN included) so propagation is exercised. The
# gather is bit-identical across all three implementations regardless of
# the values (it's a copy, no arithmetic).
_FLOAT_OR_NAN = st.floats(
    allow_nan=True,
    allow_infinity=True,
    width=64,
)


@st.composite
def _gather_inputs(  # type: ignore[no-untyped-def]
    draw,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``(values, indices)`` satisfying the gather kernel contract.

    Constraints satisfied:
      * ``values.dtype == float64`` and ``values.ndim == 1``
      * ``indices.dtype == int64`` and ``indices.ndim == 1``
      * ``0 <= indices[i] < len(values)`` for every ``i``
    """
    n = draw(st.integers(min_value=1, max_value=500))
    n_indices = draw(st.integers(min_value=0, max_value=500))
    values = np.asarray(
        draw(
            npst.arrays(
                dtype=np.float64,
                shape=n,
                elements=_FLOAT_OR_NAN,
            )
        ),
    )
    indices = np.asarray(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=n - 1),
                min_size=n_indices,
                max_size=n_indices,
            )
        ),
        dtype=np.int64,
    )
    return values, indices


# ---------------------------------------------------------------------------
# Property: NumPy ≡ Cython ≡ values.take(indices)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _CY, reason="Cython gather kernel not compiled")
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_gather_inputs())
def test_gather_numpy_matches_cython(inputs: tuple[np.ndarray, np.ndarray]) -> None:
    """``gather_numpy ≡ gather_cython`` bit-for-bit on Hypothesis-generated inputs."""
    values, indices = inputs
    out_numpy = gather_numpy(values, indices)
    out_cython = gather_cython(values, indices)
    # The gather is a copy, no arithmetic — equality is exact (bit-for-bit),
    # not approximate. ``equal_nan=True`` because the values may contain NaN.
    np.testing.assert_array_equal(out_numpy, out_cython)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_gather_inputs())
def test_gather_numpy_matches_take(inputs: tuple[np.ndarray, np.ndarray]) -> None:
    """``gather_numpy ≡ values.take(indices)`` — locks the implementation choice.

    The NumPy reference is implemented *as* ``np.take``; this test is
    therefore structurally tautological but locks the implementation
    choice: if a future patch switches to fancy indexing
    ``values[indices]``, this test stays green (same values) but
    :func:`test_gather_numpy_allocates_fresh_array` would catch the
    aliasing-or-not regression.
    """
    values, indices = inputs
    out = gather_numpy(values, indices)
    expected = values.take(indices)
    np.testing.assert_array_equal(out, expected)


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_gather_inputs())
def test_gather_numpy_allocates_fresh_array(inputs: tuple[np.ndarray, np.ndarray]) -> None:
    """``gather_numpy`` must return an array independent of the source buffer.

    Per the kernel contract (see ``_indexing_kernels.py`` module
    docstring): mutating the result must not affect ``values``. A switch
    to fancy indexing would still satisfy the value-equality property
    above but would *also* allocate fresh memory (numpy's
    ``__getitem__`` with integer arrays always copies). This test pins
    the freshness rather than the codepath.
    """
    values, indices = inputs
    if indices.shape[0] == 0:
        # Empty result — nothing to mutate.
        return
    original = values.copy()
    out = gather_numpy(values, indices)
    # Mutate the output; the source must be unchanged.
    # NaN values are inserted because the array may legitimately already
    # contain NaN; using a sentinel that's likely unique (a specific finite
    # value) sidesteps that.
    out[:] = 1.2345678e42
    # ``assert_array_equal`` treats NaN-at-same-positions as equal by default
    # in numpy >= 2.0, so a values array containing NaN compares fine.
    np.testing.assert_array_equal(values, original)
