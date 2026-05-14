# SPDX-License-Identifier: MIT
"""Tests for MVTSeries, ported from ``TimeSeriesEcon.jl/test/test_mvtseries.jl``.

Scope (session 9, 2026-05-14): construction, single- and two-argument
indexing, dot access, copy / similar / equals, integer / boolean indexing,
math helper overloads (``shift`` / ``lag`` / ``lead`` / ``diff`` / ``moving``
/ ``undiff``), and JSON round-trip. The Julia ``@testset "MV bcast"`` and
``@testset "MVTSeries show"`` blocks are deferred to the follow-up
broadcasting session.
"""

from __future__ import annotations

import copy as _copy

import numpy as np
import pytest

from tsecon import (
    MIT,
    MVTSeries,
    TSeries,
    diff,
    lag,
    lead,
    mitrange,
    mm,
    moving,
    moving_average,
    moving_sum,
    qq,
    shift,
    shift_inplace,
    undiff,
    yy,
)
from tsecon.frequencies import Quarterly, Unit

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruct:
    def test_empty_from_mit(self) -> None:
        m = MVTSeries(qq(2020, 1))
        assert m.shape == (0, 0)
        assert m.frequency == Quarterly()
        assert m.firstdate == qq(2020, 1)

    def test_empty_with_one_name(self) -> None:
        m = MVTSeries(qq(2020, 1), "a")
        assert m.shape == (0, 1)
        assert m.column_names == ("a",)

    def test_empty_with_tuple_names(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"))
        assert m.shape == (0, 2)
        assert m.column_names == ("a", "b")

    def test_empty_with_list_names(self) -> None:
        m = MVTSeries(qq(2020, 1), ["a", "b"])
        assert m.shape == (0, 2)

    def test_range_default_initializer(self) -> None:
        m = MVTSeries(mitrange(qq(2020, 1), qq(2020, 4)), ("a",))
        assert m.shape == (4, 1)
        assert np.isnan(m.values).all()

    def test_range_with_scalar(self) -> None:
        m = MVTSeries(mitrange(qq(2020, 1), qq(2020, 4)), ("a", "b"), 5)
        assert m.shape == (4, 2)
        assert (m.values == 5).all()

    def test_range_with_init_function(self) -> None:
        m = MVTSeries(mitrange(qq(2020, 1), qq(2020, 4)), ("a", "b"), np.zeros)
        assert m.shape == (4, 2)
        assert (m.values == 0).all()

    def test_range_with_ndarray(self) -> None:
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        m = MVTSeries(mitrange(qq(2020, 1), qq(2020, 2)), ("a", "b"), data)
        assert m.shape == (2, 2)
        assert (m.values == data).all()

    def test_range_with_size_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="do not match"):
            MVTSeries(mitrange(qq(2020, 1), qq(2020, 4)), ("a", "b"), np.zeros((3, 2)))

    def test_range_with_column_count_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="do not match"):
            MVTSeries(mitrange(qq(2020, 1), qq(2020, 4)), ("a", "b"), np.zeros((4, 3)))

    def test_unit_range_empty(self) -> None:
        m = MVTSeries(mitrange(MIT(Unit(), 1), MIT(Unit(), 5)))
        assert m.shape == (5, 0)

    def test_kwargs_only_from_tseries(self) -> None:
        a = TSeries(mitrange(MIT(Unit(), 1), MIT(Unit(), 5)), np.arange(1.0, 6.0))
        b = TSeries(mitrange(MIT(Unit(), 3), MIT(Unit(), 8)), np.arange(10.0, 16.0))
        m = MVTSeries(a=a, b=b)
        assert m.shape == (8, 2)
        assert m.column_names == ("a", "b")

    def test_range_kwargs_form(self) -> None:
        # MVTSeries(2020Q1:2021Q1; hex=TSeries(2019Q1, 1..20), why=zeros(5), zed=3)
        rng = mitrange(qq(2020, 1), qq(2021, 1))
        hex_series = TSeries(qq(2019, 1), np.arange(1.0, 21.0))
        m = MVTSeries(rng, hex=hex_series, why=np.zeros(5), zed=3)
        assert isinstance(m["hex"], TSeries)
        assert m.range == rng
        # Provided TSeries is truncated to MVTSeries range; firstdate(2019Q1)
        # so values at 2020Q1..2021Q1 are positions 5..9 → values 5.0..9.0.
        assert (m.hex.values == np.arange(5.0, 10.0)).all()

    def test_construction_from_init_tseries(self) -> None:
        # MVTSeries(2021Q1:2022Q4, (:a, :b, :c), init)
        init = TSeries(qq(2020, 1), np.arange(1.0, 17.0))
        rng = mitrange(qq(2021, 1), qq(2022, 4))
        m = MVTSeries(rng, ("a", "b", "c"), init)
        # All three columns initialized from the overlap of init range and m range.
        for name in m.column_names:
            assert np.allclose(m[name].values, init[rng].values)

    def test_classmethod_empty(self) -> None:
        rng = mitrange(yy(2020), yy(2022))
        m = MVTSeries.empty(rng, ("a", "b"), dtype=np.int64)
        assert m.dtype == np.int64
        assert m.shape == (3, 2)

    def test_classmethod_zeros(self) -> None:
        m = MVTSeries.zeros(mitrange(qq(2020, 1), qq(2020, 4)), ("a", "b"))
        assert (m.values == 0).all()

    def test_classmethod_ones(self) -> None:
        m = MVTSeries.ones(mitrange(qq(2020, 1), qq(2020, 4)), ("a", "b"))
        assert (m.values == 1).all()

    def test_classmethod_fill(self) -> None:
        m = MVTSeries.fill(mitrange(qq(2020, 1), qq(2020, 4)), ("a", "b"), 7.5)
        assert (m.values == 7.5).all()

    def test_first_arg_required(self) -> None:
        with pytest.raises(TypeError, match="firstdate"):
            MVTSeries()

    def test_wrap_by_default_aliases_buffer(self) -> None:
        # Decision 16: a compatible 2-D ndarray is wrapped, not copied.
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        m = MVTSeries(qq(2020, 1), ("a", "b"), data)
        assert m.values is data

    def test_copy_kwarg_forces_allocation(self) -> None:
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        m = MVTSeries(qq(2020, 1), ("a", "b"), data, copy=True)
        assert m.values is not data
        assert (m.values == data).all()

    def test_column_view_writes_through_to_matrix(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.zeros((4, 2)))
        m.a[qq(2020, 1)] = 99
        assert m.values[0, 0] == 99


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


class TestAccessors:
    def test_firstdate_lastdate_range(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.zeros((4, 2)))
        assert m.firstdate == qq(2020, 1)
        assert m.lastdate == qq(2020, 4)
        assert m.range == mitrange(qq(2020, 1), qq(2020, 4))

    def test_frequency(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a",), np.zeros((4, 1)))
        assert m.frequency == Quarterly()

    def test_dtype(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a",), np.zeros((4, 1), dtype=np.int32))
        assert m.dtype == np.int32

    def test_shape_ndim(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.zeros((4, 2)))
        assert m.shape == (4, 2)
        assert m.ndim == 2

    def test_columns_dict_keys_preserves_order(self) -> None:
        m = MVTSeries(qq(2020, 1), ("c", "a", "b"), np.zeros((2, 3)))
        assert tuple(m.columns.keys()) == ("c", "a", "b")
        assert m.column_names == ("c", "a", "b")

    def test_len(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.zeros((4, 2)))
        assert len(m) == 4

    def test_contains(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.zeros((4, 2)))
        assert "a" in m
        assert "c" not in m

    def test_is_empty(self) -> None:
        assert MVTSeries(qq(2020, 1)).is_empty()
        assert MVTSeries(qq(2020, 1), ("a",)).is_empty()
        assert not MVTSeries(qq(2020, 1), ("a",), np.zeros((3, 1))).is_empty()


# ---------------------------------------------------------------------------
# Indexing: single argument
# ---------------------------------------------------------------------------


class TestSingleIndex:
    @pytest.fixture
    def m(self) -> MVTSeries:
        return MVTSeries(qq(2020, 1), ("a", "b"), np.arange(1.0, 9.0).reshape(4, 2))

    def test_mit_returns_row_vector(self, m: MVTSeries) -> None:
        row = m[qq(2020, 1)]
        assert isinstance(row, np.ndarray)
        assert row.shape == (2,)
        assert (row == np.array([1.0, 2.0])).all()

    def test_mit_out_of_range_raises(self, m: MVTSeries) -> None:
        with pytest.raises(IndexError):
            _ = m[qq(2019, 1)]

    def test_mit_mixed_freq_raises(self, m: MVTSeries) -> None:
        with pytest.raises(TypeError, match="frequencies"):
            _ = m[yy(2020)]

    def test_mitrange_returns_mvtseries(self, m: MVTSeries) -> None:
        sub = m[mitrange(qq(2020, 1), qq(2020, 2))]
        assert isinstance(sub, MVTSeries)
        assert sub.shape == (2, 2)
        assert sub.column_names == ("a", "b")
        assert (sub.values == np.array([[1.0, 2.0], [3.0, 4.0]])).all()

    def test_string_returns_column_tseries(self, m: MVTSeries) -> None:
        col = m["a"]
        assert isinstance(col, TSeries)
        assert (col.values == np.array([1.0, 3.0, 5.0, 7.0])).all()
        # Identity preserved: the same TSeries is cached.
        assert m["a"] is m["a"]

    def test_tuple_of_names_returns_subset(self, m: MVTSeries) -> None:
        sub = m[("a", "b")]
        assert isinstance(sub, MVTSeries)
        assert sub.column_names == ("a", "b")

    def test_list_of_names_returns_subset(self, m: MVTSeries) -> None:
        sub = m[["a", "b"]]
        assert isinstance(sub, MVTSeries)
        assert sub.column_names == ("a", "b")

    def test_unknown_column_raises(self, m: MVTSeries) -> None:
        with pytest.raises(KeyError):
            _ = m["c"]

    def test_integer_falls_through(self, m: MVTSeries) -> None:
        # Integer pass-through to ndarray: m[0] is the first row.
        assert (m[0] == np.array([1.0, 2.0])).all()

    def test_mit_slice(self, m: MVTSeries) -> None:
        sub = m[qq(2020, 1) : qq(2020, 2)]
        assert isinstance(sub, MVTSeries)
        assert sub.shape == (2, 2)


# ---------------------------------------------------------------------------
# Indexing: two arguments
# ---------------------------------------------------------------------------


class TestTwoIndex:
    @pytest.fixture
    def m(self) -> MVTSeries:
        return MVTSeries(qq(2020, 1), ("a", "b"), np.arange(1.0, 9.0).reshape(4, 2))

    def test_mit_str_returns_scalar(self, m: MVTSeries) -> None:
        assert m[qq(2020, 1), "a"] == 1.0
        assert m[qq(2020, 2), "b"] == 4.0

    def test_mit_list_returns_row_subset(self, m: MVTSeries) -> None:
        sub = m[qq(2020, 1), ["a", "b"]]
        assert isinstance(sub, np.ndarray)
        assert (sub == np.array([1.0, 2.0])).all()

    def test_mitrange_str_returns_tseries(self, m: MVTSeries) -> None:
        col = m[mitrange(qq(2020, 1), qq(2020, 3)), "a"]
        assert isinstance(col, TSeries)
        assert col.firstdate == qq(2020, 1)
        assert (col.values == np.array([1.0, 3.0, 5.0])).all()

    def test_mitrange_list_returns_mvtseries(self, m: MVTSeries) -> None:
        sub = m[mitrange(qq(2020, 1), qq(2020, 2)), ["a", "b"]]
        assert isinstance(sub, MVTSeries)
        assert sub.shape == (2, 2)
        assert sub.column_names == ("a", "b")

    def test_colon_first(self, m: MVTSeries) -> None:
        # mvts[:, "a"] → column
        col = m[:, "a"]
        assert isinstance(col, TSeries)
        assert col is m["a"]

    def test_colon_second(self, m: MVTSeries) -> None:
        # mvts[mit, :] → row vector
        row = m[qq(2020, 1), :]
        assert (row == np.array([1.0, 2.0])).all()

    def test_double_colon_returns_self(self, m: MVTSeries) -> None:
        assert m[:, :] is m

    def test_unknown_column_in_two_arg_raises(self, m: MVTSeries) -> None:
        with pytest.raises(KeyError):
            _ = m[qq(2020, 1), "c"]


# ---------------------------------------------------------------------------
# Assignment
# ---------------------------------------------------------------------------


class TestSetItem:
    def test_set_via_string(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.zeros((4, 2)))
        m["a"] = np.array([10.0, 20.0, 30.0, 40.0])
        assert (m.values[:, 0] == np.array([10.0, 20.0, 30.0, 40.0])).all()

    def test_set_via_attribute(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.zeros((4, 2)))
        m.a = 5
        assert (m.values[:, 0] == 5).all()

    def test_set_attribute_unknown_column_raises(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.zeros((4, 2)))
        with pytest.raises(AttributeError, match="new column"):
            m.c = 5

    def test_set_via_mit(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.zeros((4, 2)))
        m[qq(2020, 1)] = np.array([1.0, 2.0])
        assert (m.values[0, :] == np.array([1.0, 2.0])).all()

    def test_set_via_mit_str(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.zeros((4, 2)))
        m[qq(2020, 1), "b"] = 99
        assert m.values[0, 1] == 99

    def test_set_via_mitrange_str(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.zeros((4, 2)))
        m[mitrange(qq(2020, 1), qq(2020, 3)), "a"] = np.array([1.0, 2.0, 3.0])
        assert (m.values[:3, 0] == np.array([1.0, 2.0, 3.0])).all()

    def test_set_column_from_tseries(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.zeros((4, 2)))
        src = TSeries(qq(2020, 1), np.array([1.0, 2.0, 3.0, 4.0]))
        m.a = src
        assert (m.values[:, 0] == np.array([1.0, 2.0, 3.0, 4.0])).all()

    def test_set_column_from_partial_tseries_aligns_by_mit(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.zeros((4, 2)))
        src = TSeries(qq(2020, 2), np.array([20.0, 30.0]))  # only middle two periods
        m.a = src
        # Untouched periods (2020Q1, 2020Q4) stay at 0.
        assert (m.values[:, 0] == np.array([0.0, 20.0, 30.0, 0.0])).all()

    def test_set_subset_by_name_collection(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.zeros((4, 2)))
        m[("a", "b")] = np.ones((4, 2))
        assert (m.values == 1).all()

    def test_assign_mvts_overlap(self) -> None:
        # https://github.com/bankofcanada/TimeSeriesEcon.jl/pull/49
        # `A[2021M1:2021M12] = B` writes overlap, leaves rest alone.
        # 36 monthly rows starting at 2020M1 ⇒ range 2020M1..2022M12.
        a_data = np.arange(108.0).reshape(36, 3)
        a = MVTSeries(mm(2020, 1), ("a", "b", "c"), a_data.copy())
        b = MVTSeries(mm(2021, 1), ("a", "c"), np.ones((12, 2)))
        a[mitrange(mm(2021, 1), mm(2021, 12))] = b
        # The non-overlapping rows are unchanged.
        assert np.allclose(a[mitrange(mm(2020, 1), mm(2020, 12)), :].values, a_data[:12, :])
        assert np.allclose(a[mitrange(mm(2022, 1), mm(2022, 12)), :].values, a_data[24:, :])
        # The :b column is untouched everywhere.
        assert np.allclose(a["b"].values, a_data[:, 1])
        # The :a and :c columns are 1 over the 12 months of 2021.
        assert np.allclose(a[mitrange(mm(2021, 1), mm(2021, 12)), "a"].values, np.ones(12))
        assert np.allclose(a[mitrange(mm(2021, 1), mm(2021, 12)), "c"].values, np.ones(12))


# ---------------------------------------------------------------------------
# Integer indexing pass-through
# ---------------------------------------------------------------------------


class TestIntegerIndexing:
    @pytest.fixture
    def m(self) -> MVTSeries:
        # Use Quarterly so we exercise non-Unit frequency, but stick to 4 rows.
        data = np.array([[1.0, 6.0], [2.0, 7.0], [3.0, 8.0], [4.0, 9.0]])
        return MVTSeries(qq(2020, 1), ("a", "b"), data.copy())

    def test_int_row(self, m: MVTSeries) -> None:
        assert (m[0] == np.array([1.0, 6.0])).all()

    def test_int_pair(self, m: MVTSeries) -> None:
        assert m[0, 0] == 1.0
        assert m[2, 1] == 8.0

    def test_int_slice(self, m: MVTSeries) -> None:
        assert (m[1:3] == m.values[1:3]).all()

    def test_set_int_pair(self, m: MVTSeries) -> None:
        m[0, 0] = 99
        assert m.values[0, 0] == 99


# ---------------------------------------------------------------------------
# Boolean indexing
# ---------------------------------------------------------------------------


class TestBoolIndexing:
    @pytest.fixture
    def m(self) -> MVTSeries:
        return MVTSeries(yy(2020), tuple("abcd"), np.arange(20.0).reshape(5, 4))

    def test_bool_row_mask(self, m: MVTSeries) -> None:
        mask = np.array([True, False, True, False, True])
        sub = m[mask]
        assert sub.shape == (3, 4)
        assert (sub == np.array([[0, 1, 2, 3], [8, 9, 10, 11], [16, 17, 18, 19]])).all()

    def test_bool_row_mask_with_str_col(self, m: MVTSeries) -> None:
        mask = np.array([True, False, True, False, True])
        assert (m[mask, "a"] == np.array([0.0, 8.0, 16.0])).all()

    def test_bool_row_mask_with_list_cols(self, m: MVTSeries) -> None:
        mask = np.array([True, False, True, False, True])
        sub = m[mask, ["a", "c"]]
        assert sub.shape == (3, 2)

    def test_bool_row_setitem(self, m: MVTSeries) -> None:
        mask = np.array([True, False, True, False, True])
        m[mask] = np.zeros((3, 4))
        assert (m.values[mask, :] == 0).all()


# ---------------------------------------------------------------------------
# Copy / similar / deepcopy
# ---------------------------------------------------------------------------


class TestCopy:
    def test_copy_independent(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.ones((4, 2)))
        c = m.copy()
        c.values[0, 0] = 99
        assert m.values[0, 0] == 1

    def test_copy_deep_kwarg_is_independent(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.ones((4, 2)))
        c = m.copy(deep=True)
        c.values[0, 0] = 99
        assert m.values[0, 0] == 1

    def test_deepcopy_honors_memo(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.ones((4, 2)))
        memo: dict[int, object] = {}
        new = _copy.deepcopy(m, memo)
        assert id(m) in memo
        assert memo[id(m)] is new

    def test_similar_matches_shape(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.ones((4, 2)))
        s = m.similar()
        assert s.shape == m.shape
        assert s.dtype == m.dtype

    def test_similar_with_dtype_override(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.ones((4, 2)))
        s = m.similar(dtype=np.int64)
        assert s.dtype == np.int64

    def test_copy_columns_view_back_into_new_buffer(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.ones((4, 2)))
        c = m.copy()
        c.a[qq(2020, 1)] = 99
        # The copy's column writes through to the copy's matrix, not m's.
        assert c.values[0, 0] == 99
        assert m.values[0, 0] == 1


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------


class TestEquality:
    def test_equals_round_trip(self) -> None:
        a = MVTSeries(qq(2020, 1), ("a", "b"), np.ones((4, 2)))
        b = MVTSeries(qq(2020, 1), ("a", "b"), np.ones((4, 2)))
        assert a.equals(b)

    def test_equals_differs_on_firstdate(self) -> None:
        a = MVTSeries(qq(2020, 1), ("a", "b"), np.ones((4, 2)))
        b = MVTSeries(qq(2020, 2), ("a", "b"), np.ones((4, 2)))
        assert not a.equals(b)

    def test_equals_differs_on_column_names(self) -> None:
        a = MVTSeries(qq(2020, 1), ("a", "b"), np.ones((4, 2)))
        b = MVTSeries(qq(2020, 1), ("a", "c"), np.ones((4, 2)))
        assert not a.equals(b)

    def test_eq_elementwise(self) -> None:
        a = MVTSeries(qq(2020, 1), ("a", "b"), np.arange(8.0).reshape(4, 2))
        b = MVTSeries(qq(2020, 1), ("a", "b"), np.arange(8.0).reshape(4, 2))
        assert (a == b).all()

    def test_eq_scalar(self) -> None:
        a = MVTSeries(qq(2020, 1), ("a", "b"), np.ones((4, 2)))
        assert (a == 1).all()

    def test_not_hashable(self) -> None:
        a = MVTSeries(qq(2020, 1), ("a",), np.ones((4, 1)))
        with pytest.raises(TypeError):
            hash(a)

    def test_allclose(self) -> None:
        a = MVTSeries(qq(2020, 1), ("a", "b"), np.ones((4, 2)))
        b = MVTSeries(qq(2020, 1), ("a", "b"), np.full((4, 2), 1.0 + 1e-12))
        assert a.allclose(b)


# ---------------------------------------------------------------------------
# Math overloads (shift / lag / lead / diff)
# ---------------------------------------------------------------------------


class TestShiftLagLead:
    @pytest.fixture
    def m(self) -> MVTSeries:
        return MVTSeries(qq(2020, 1), ("a", "b"), np.arange(1.0, 9.0).reshape(4, 2))

    def test_shift_positive(self, m: MVTSeries) -> None:
        s = shift(m, 1)
        assert s.firstdate == qq(2019, 4)
        assert s.shape == m.shape
        # Values are unchanged (only the firstdate moves).
        assert (s.values == m.values).all()

    def test_shift_negative(self, m: MVTSeries) -> None:
        s = shift(m, -1)
        assert s.firstdate == qq(2020, 2)

    def test_shift_does_not_alias(self, m: MVTSeries) -> None:
        s = shift(m, 1)
        s.values[0, 0] = 99
        assert m.values[0, 0] != 99

    def test_shift_inplace_mutates_columns(self, m: MVTSeries) -> None:
        s = shift_inplace(m, 1)
        assert s is m
        assert m.firstdate == qq(2019, 4)
        # The cached column anchors also reflect the new firstdate.
        assert m["a"].firstdate == qq(2019, 4)

    def test_lag(self, m: MVTSeries) -> None:
        assert lag(m, 1).firstdate == qq(2020, 2)

    def test_lead(self, m: MVTSeries) -> None:
        assert lead(m, 1).firstdate == qq(2019, 4)

    def test_lag_equals_shift_minus_k(self, m: MVTSeries) -> None:
        assert lag(m, 2).firstdate == shift(m, -2).firstdate


class TestDiff:
    def test_diff_default(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.arange(1.0, 9.0).reshape(4, 2))
        d = diff(m)
        assert isinstance(d, MVTSeries)
        assert d.shape == (3, 2)
        assert d.range == mitrange(qq(2020, 2), qq(2020, 4))
        # Each row is a constant difference of 2 (1→3, 3→5, 5→7 in col a; same for b).
        assert (d.values == 2).all()

    def test_diff_k_negative_2(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a",), np.arange(1.0, 6.0).reshape(5, 1))
        d = diff(m, -2)
        assert d.shape == (3, 1)
        # 5 quarters starting at 2020Q1 → 2020Q1..2021Q1; diff(k=-2) drops 2.
        assert d.range == mitrange(qq(2020, 3), qq(2021, 1))

    def test_diff_window_too_large_raises(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a",), np.arange(1.0, 4.0).reshape(3, 1))
        with pytest.raises(ValueError, match="window"):
            _ = diff(m, -3)


# ---------------------------------------------------------------------------
# Math overloads (moving / moving_sum / moving_average)
# ---------------------------------------------------------------------------


class TestMovingMVTS:
    def test_moving_window_2_positive(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.arange(1.0, 9.0).reshape(4, 2))
        mv = moving(m, 2)
        assert mv.shape == (3, 2)
        # The first row is the mean of rows 0 and 1: ([1+3, 2+4])/2 = [2, 3].
        assert (mv.values[0] == np.array([2.0, 3.0])).all()
        # firstdate is shifted forward by n-1 = 1.
        assert mv.firstdate == qq(2020, 2)

    def test_moving_window_negative(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.arange(1.0, 9.0).reshape(4, 2))
        mv = moving(m, -2)
        # Forward-looking — firstdate doesn't shift.
        assert mv.firstdate == qq(2020, 1)

    def test_moving_sum_is_moving_times_n(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.arange(1.0, 9.0).reshape(4, 2))
        avg = moving_average(m, 3)
        sm = moving_sum(m, 3)
        assert np.allclose(sm.values, 3 * avg.values)

    def test_moving_zero_window_raises(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.zeros((4, 2)))
        with pytest.raises(ValueError, match="nonzero"):
            _ = moving(m, 0)

    def test_moving_window_exceeds_length_raises(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.zeros((4, 2)))
        with pytest.raises(ValueError, match="exceeds"):
            _ = moving(m, 5)


# ---------------------------------------------------------------------------
# Math overloads (undiff)
# ---------------------------------------------------------------------------


class TestUndiffMVTS:
    def test_undiff_scalar_default(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.arange(1.0, 9.0).reshape(4, 2))
        d = diff(m)
        # Anchor at firstdate(d)-1 = qq(2020,1) with scalar 0 by default.
        # The MVTSeries scalar broadcasts to all columns.
        # We give the explicit anchor below — here use the default-zero scalar form.
        u = undiff(d)
        # u should be cumsum(d) plus a constant per column so that u[firstdate-1] = 0;
        # the anchor period is *outside* d.range, so d is extended with a zero row.
        assert isinstance(u, MVTSeries)
        assert u.shape == (4, 2)

    def test_undiff_round_trip_with_anchor(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.arange(1.0, 9.0).reshape(4, 2))
        d = diff(m)
        u = undiff(d, anchor=(m.firstdate, m[m.firstdate]))
        assert np.allclose(u.values, m.values)

    def test_undiff_vector_anchor(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.arange(1.0, 9.0).reshape(4, 2))
        d = diff(m)
        u = undiff(d, anchor=np.array([1.0, 2.0]))
        # Anchor falls outside d.range, so d is zero-extended; the anchor row
        # should equal [1.0, 2.0] = m[firstdate].
        assert np.allclose(u[qq(2020, 1)], np.array([1.0, 2.0]))

    def test_undiff_anchor_mvts(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.arange(1.0, 9.0).reshape(4, 2))
        d = diff(m)
        u = undiff(d, anchor=m)
        # m anchored at firstdate(d)-1 = qq(2020,1), value row = [1.0, 2.0].
        assert np.allclose(u[qq(2020, 1)], m[qq(2020, 1)])

    def test_undiff_anchor_wrong_length_raises(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.arange(1.0, 9.0).reshape(4, 2))
        d = diff(m)
        with pytest.raises(ValueError, match="length"):
            _ = undiff(d, anchor=np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------


class TestRepr:
    def test_repr_smoke(self) -> None:
        m = MVTSeries(qq(2020, 1), ("a", "b"), np.zeros((4, 2)))
        r = repr(m)
        assert "MVTSeries" in r
        assert "Quarterly" in r

    def test_repr_empty(self) -> None:
        m = MVTSeries(qq(2020, 1))
        r = repr(m)
        assert "no variables" in r
