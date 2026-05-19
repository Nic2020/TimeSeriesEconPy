# SPDX-License-Identifier: MIT
"""Tests for :func:`tsecon.lookup` — vectorised time-series gather.

Covers the public wrapper's validation contract, scalar/array shape
handling, MIT/int key dispatch, and bit-for-bit equivalence against the
per-element ``__getitem__`` reference (which is what users would write
in the absence of :func:`lookup`). The kernel pair
(``_indexing_kernels.gather_numpy`` and
``_indexing_kernels_cy.gather_cython`` when compiled) is also exercised
directly so the NumPy fallback path stays validated regardless of
local toolchain availability.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsecon import MIT, TSeries, lookup, lookup_is_cython, qq, yy
from tsecon._indexing_kernels import gather_numpy
from tsecon.frequencies import Unit, Yearly

# Cython kernel is optional — only present when the wheel was built with a C
# toolchain. Tests that exercise it skip when ``_CY`` is False; the import is
# guarded with try/except so the file is loadable on toolchain-less installs.
try:
    from tsecon._indexing_kernels_cy import (  # type: ignore[import-not-found]
        gather_cython,
    )

    _CY = True
except ImportError:
    _CY = False


def _unit(i: int) -> MIT:
    return MIT(Unit(), i)


# ---------------------------------------------------------------------------
# Kernel equivalence
# ---------------------------------------------------------------------------


class TestLookupAgreesWithGetitem:
    """The whole point: identical output to a per-element ``t[k]`` loop."""

    @pytest.mark.parametrize(
        ("start_factory", "indices_to_pick"),
        [
            (lambda: qq(2020, 1), [0, 5, 10, 50, 99]),
            (lambda: qq(2020, 1), list(range(100))),  # all positions
            (lambda: qq(2020, 1), [99, 0, 50, 25, 75]),  # non-monotonic
            (lambda: qq(2020, 1), [42]),  # length-1
            (lambda: yy(2020), [0, 1, 2, 3]),  # yearly frequency
        ],
    )
    def test_mit_keys_match_getitem(
        self,
        start_factory: object,
        indices_to_pick: list[int],
    ) -> None:
        start = start_factory()  # type: ignore[operator]
        values = np.arange(100.0)
        t = TSeries(start, values)
        keys = [start + i for i in indices_to_pick]
        got = lookup(t, keys)
        expected = np.array([t[k] for k in keys])
        np.testing.assert_array_equal(got, expected)

    @pytest.mark.parametrize(
        "indices",
        [
            [0, 5, 10, 50, 99],
            list(range(100)),
            [99, 0, 50, 25, 75],
            [42],
        ],
    )
    def test_int_keys_match_getitem(self, indices: list[int]) -> None:
        t = TSeries(qq(2020, 1), np.arange(100.0))
        got = lookup(t, indices)
        expected = np.array([t[i] for i in indices])
        np.testing.assert_array_equal(got, expected)

    def test_repeated_keys_supported(self) -> None:
        """Gather may include duplicates — ``np.take`` allows this and so do we."""
        t = TSeries(qq(2020, 1), np.arange(100.0))
        got = lookup(t, [0, 0, 0, 99, 99])
        np.testing.assert_array_equal(got, [0.0, 0.0, 0.0, 99.0, 99.0])


# ---------------------------------------------------------------------------
# Return-type / shape contract
# ---------------------------------------------------------------------------


class TestLookupReturnShape:
    def test_returns_ndarray(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(100.0))
        result = lookup(t, [qq(2020, 1), qq(2020, 2)])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_dtype_matches_source(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(100.0))
        result = lookup(t, [0, 1, 2])
        assert result.dtype == np.float64

    def test_empty_keys_returns_empty(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(100.0))
        result = lookup(t, [])
        assert result.shape == (0,)
        assert result.dtype == np.float64

    def test_result_is_independent_of_source(self) -> None:
        """Per the contract: mutating the result must not alias ``t``."""
        t = TSeries(qq(2020, 1), np.arange(100.0).copy())
        result = lookup(t, [0, 1, 2])
        result[0] = 999.0
        assert t[0] == 0.0  # unchanged

    def test_accepts_ndarray_keys(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(100.0))
        keys = np.array([0, 5, 10, 50, 99], dtype=np.int64)
        got = lookup(t, keys)
        np.testing.assert_array_equal(got, [0.0, 5.0, 10.0, 50.0, 99.0])

    def test_int_dtype_preserved(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(100, dtype=np.int64))
        result = lookup(t, [0, 5, 99])
        assert result.dtype == np.int64
        np.testing.assert_array_equal(result, [0, 5, 99])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestLookupValidation:
    def test_mit_frequency_mismatch_first_raises(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(100.0))
        bad_key = MIT(Yearly(), 2020)
        with pytest.raises(TypeError, match="frequency"):
            lookup(t, [bad_key])

    def test_mit_frequency_mismatch_later_raises(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(100.0))
        keys = [qq(2020, 1), qq(2020, 2), MIT(Yearly(), 2020)]
        with pytest.raises(TypeError, match="frequency"):
            lookup(t, keys)

    def test_mixed_mit_int_raises(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(100.0))
        keys = [qq(2020, 1), 5]  # MIT then int
        with pytest.raises(TypeError, match="homogeneous"):
            lookup(t, keys)

    def test_int_out_of_range_low_raises(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(100.0))
        with pytest.raises(IndexError, match="out of range"):
            lookup(t, [-1, 0, 1])

    def test_int_out_of_range_high_raises(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(100.0))
        with pytest.raises(IndexError, match="out of range"):
            lookup(t, [0, 99, 100])

    def test_mit_out_of_range_raises(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(100.0))
        with pytest.raises(IndexError, match="out of range"):
            lookup(t, [qq(2020, 1), qq(2100, 1)])

    def test_unsupported_key_type_raises(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(100.0))
        with pytest.raises(TypeError, match="must be MIT or int"):
            lookup(t, ["q1", "q2"])  # type: ignore[list-item]

    def test_2d_array_keys_raises(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(100.0))
        with pytest.raises(ValueError, match="1-D"):
            lookup(t, np.array([[0, 1], [2, 3]]))


# ---------------------------------------------------------------------------
# Direct kernel / introspection
# ---------------------------------------------------------------------------


class TestGatherKernelFallback:
    """Lock the NumPy reference kernel against a hand-checked answer.

    Exercises ``gather_numpy`` directly so the fallback path is
    validated independent of the public dispatcher.
    """

    def test_numpy_kernel_basic_gather(self) -> None:
        values = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        indices = np.array([0, 2, 4, 1, 3], dtype=np.int64)
        got = gather_numpy(values, indices)
        np.testing.assert_array_equal(got, [0.0, 20.0, 40.0, 10.0, 30.0])

    def test_numpy_kernel_empty_indices(self) -> None:
        values = np.array([0.0, 1.0, 2.0])
        got = gather_numpy(values, np.array([], dtype=np.int64))
        assert got.shape == (0,)
        assert got.dtype == np.float64

    def test_is_cython_flag_is_boolean(self) -> None:
        flag = lookup_is_cython()
        assert isinstance(flag, bool)


# ---------------------------------------------------------------------------
# Mirrors the benchmark scenario shape (regression catch)
# ---------------------------------------------------------------------------


class TestLookupBenchmarkShape:
    """A 100-element MIT / int lookup matches the benchmark scenarios."""

    def test_mit_lookup_100_matches_loop_sum(self) -> None:
        start = qq(2020, 1)
        t = TSeries(start, np.arange(100.0))
        keys = [start + i for i in range(100)]
        loop_sum = sum(float(t[k]) for k in keys)
        vec_sum = float(lookup(t, keys).sum())
        assert loop_sum == vec_sum == sum(range(100))

    def test_int_lookup_100_matches_loop_sum(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(100.0))
        keys = list(range(100))
        loop_sum = sum(float(t[k]) for k in keys)
        vec_sum = float(lookup(t, keys).sum())
        assert loop_sum == vec_sum == sum(range(100))


class TestLookupUnit:
    """Unit-frequency lookup with hand-checked positions."""

    def test_unit_gather(self) -> None:
        t = TSeries(_unit(1), np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
        got = lookup(t, [_unit(1), _unit(3), _unit(5)])
        np.testing.assert_array_equal(got, [10.0, 30.0, 50.0])


# ---------------------------------------------------------------------------
# Edge cases for the gather kernel (NaN passthrough, non-contiguous source)
# ---------------------------------------------------------------------------


class TestGatherKernelEdgeCases:
    """Degenerate inputs follow the documented contract.

    Closes review file ``F06_rec_indexing_edge_case_gaps`` — the edge-case
    template crystallised by ``test_stats_kernels.py`` (session 20) and
    ``test_fconvert_kernels.py`` (session 21) was not retrofitted onto
    ``test_indexing.py`` (session 19). NaN-in-source is realistic
    (resize-on-assign creates NaN gaps in TSeries storage); a gather
    must pass through NaN unchanged, not raise or silently substitute.
    """

    def test_gather_numpy_passes_through_nan(self) -> None:
        values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        indices = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        got = gather_numpy(values, indices)
        # First / third / fifth positions are regular; second and fourth must
        # remain NaN. Use isnan because NaN != NaN.
        assert got[0] == 1.0
        assert got[2] == 3.0
        assert got[4] == 5.0
        assert np.isnan(got[1])
        assert np.isnan(got[3])

    def test_gather_cython_passes_through_nan(self) -> None:
        if not _CY:
            pytest.skip("Cython gather kernel not compiled")
        values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        indices = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        got = gather_cython(values, indices)
        assert got[0] == 1.0
        assert got[2] == 3.0
        assert got[4] == 5.0
        assert np.isnan(got[1])
        assert np.isnan(got[3])

    def test_gather_numpy_handles_non_contiguous_source(self) -> None:
        """The NumPy reference uses ``np.take``, which handles any stride pattern.

        The matching ``_is_kernel_eligible`` check in
        ``tsecon._kernel_dispatch`` routes non-contiguous sources to this
        path (see ``indexing.py`` final ``return gather_numpy`` branch);
        the Cython kernel rejects non-contiguous views at the typed
        memoryview cast.
        """
        # 10 float64 elements, stride=16 (non-contiguous view of length-20 source).
        storage = np.arange(20.0)
        view = storage[::2]
        assert not view.flags.c_contiguous
        indices = np.array([0, 5, 9, 0, 2], dtype=np.int64)
        got = gather_numpy(view, indices)
        np.testing.assert_array_equal(got, [0.0, 10.0, 18.0, 0.0, 4.0])

    def test_gather_cython_rejects_non_contiguous_source(self) -> None:
        """Cython memoryview cast rejects non-C-contiguous source arrays.

        Locks the asymmetric-contract observation that motivates the
        ``_is_kernel_eligible`` contiguity check in
        ``tsecon._kernel_dispatch``. If a future patch loosens the
        ``cdef double[::1]`` constraint to ``cdef double[:]``, this test
        will start failing and the dispatcher's branch will need a
        reconsider.
        """
        if not _CY:
            pytest.skip("Cython gather kernel not compiled")
        storage = np.arange(20.0)
        view = storage[::2]
        indices = np.array([0, 2, 4], dtype=np.int64)
        with pytest.raises(ValueError, match="contiguous"):
            gather_cython(view, indices)
