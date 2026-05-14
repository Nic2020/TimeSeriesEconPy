# SPDX-License-Identifier: MIT
"""Tests for TSeries, ported from ``TimeSeriesEcon.jl/test/test_tseries.jl``.

Scope: the core TSeries surface — construction, indexing (int / slice / MIT /
MITRange / Bool TSeries / array-of-MIT), arithmetic alignment, NumPy protocol
dispatch (``__array_ufunc__`` / ``__array_function__`` / ``__array__``),
resize-on-assign, comparisons, and pretty-printing. Out of scope for this
session: ``shift`` / ``lag`` / ``lead`` / ``pct`` / ``apct`` / ``ytypct``
(tsmath.jl), and ``BDaily`` ``skip_holidays`` variants.
"""

from __future__ import annotations

import copy as _copy

import numpy as np
import pytest

from tsecon import (
    MIT,
    TSeries,
    mitrange,
    mm,
    qq,
    typenan,
    yy,
)
from tsecon.frequencies import Quarterly, Unit, Yearly

# ---------------------------------------------------------------------------
# typenan
# ---------------------------------------------------------------------------


def test_typenan_float() -> None:
    val = typenan(np.float64)
    assert np.isnan(val)


def test_typenan_int() -> None:
    val = typenan(np.int32)
    assert val == np.iinfo(np.int32).max


def test_typenan_bool() -> None:
    assert typenan(np.bool_) is np.bool_(False)


def test_typenan_unsupported() -> None:
    with pytest.raises(TypeError):
        typenan("U8")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construct_from_mit_and_array() -> None:
    s = TSeries(qq(2020, 1), [11.0 + i for i in range(12)])
    assert s.firstdate == qq(2020, 1)
    assert s.lastdate == qq(2022, 4)
    assert len(s) == 12
    assert list(s.values) == [11.0 + i for i in range(12)]
    assert s.shape == (12,)
    assert s.ndim == 1


def test_construct_from_mit_array_copies() -> None:
    # External mutation must not reach into the series.
    data = np.array([1.0, 2.0, 3.0])
    s = TSeries(yy(2000), data)
    data[0] = 999.0
    assert s.values[0] == 1.0


def test_construct_from_mit_dtype_conversion() -> None:
    s = TSeries(yy(2000), [1, 2, 3], dtype=np.int32)
    assert s.dtype == np.int32
    assert list(s.values) == [1, 2, 3]


def test_construct_from_mit_no_values_makes_empty() -> None:
    s = TSeries(qq(2020, 1))
    assert len(s) == 0
    assert s.firstdate == qq(2020, 1)
    assert s.is_empty()


def test_construct_from_mit_scalar_raises() -> None:
    # MIT + scalar is ambiguous; must use MITRange for the fill semantics.
    with pytest.raises(TypeError, match="ambiguous"):
        TSeries(qq(2020, 1), 0.5)


def test_construct_from_mit_2d_raises() -> None:
    with pytest.raises(ValueError, match="1-D"):
        TSeries(yy(2000), np.zeros((3, 2)))


def test_construct_from_range_no_values_fills_nan() -> None:
    rng = mitrange(qq(2020, 1), qq(2020, 4))
    s = TSeries(rng)
    assert len(s) == 4
    assert s.firstdate == qq(2020, 1)
    assert np.all(np.isnan(s.values))


def test_construct_from_range_int_dtype_fills_typenan() -> None:
    rng = mitrange(yy(2000), yy(2002))
    s = TSeries(rng, dtype=np.int32)
    assert s.dtype == np.int32
    assert all(v == np.iinfo(np.int32).max for v in s.values)


def test_construct_from_range_scalar_fills() -> None:
    rng = mitrange(yy(2000), yy(2002))
    s = TSeries(rng, 7)
    assert list(s.values) == [7, 7, 7]


def test_construct_from_range_array_matches_length() -> None:
    rng = mitrange(qq(2020, 1), qq(2020, 4))
    s = TSeries(rng, [10.0, 20.0, 30.0, 40.0])
    assert list(s.values) == [10.0, 20.0, 30.0, 40.0]


def test_construct_from_range_array_length_mismatch_raises() -> None:
    rng = mitrange(qq(2020, 1), qq(2020, 4))
    with pytest.raises(ValueError, match="lengths mismatch"):
        TSeries(rng, [1.0, 2.0])


def test_construct_rejects_non_mit_first_arg() -> None:
    with pytest.raises(TypeError, match="MIT or MITRange"):
        TSeries("not an MIT", [1.0])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Classmethod constructors
# ---------------------------------------------------------------------------


def test_empty_classmethod() -> None:
    s = TSeries.empty(mitrange(qq(2020, 1), qq(2020, 4)))
    assert len(s) == 4
    assert np.all(np.isnan(s.values))


def test_zeros_classmethod() -> None:
    s = TSeries.zeros(mitrange(yy(2000), yy(2002)))
    assert list(s.values) == [0.0, 0.0, 0.0]


def test_ones_classmethod_with_int_dtype() -> None:
    s = TSeries.ones(mitrange(yy(2000), yy(2002)), dtype=np.int64)
    assert list(s.values) == [1, 1, 1]
    assert s.dtype == np.int64


def test_fill_classmethod() -> None:
    s = TSeries.fill(mitrange(yy(2000), yy(2002)), 3.14)
    assert list(s.values) == [3.14, 3.14, 3.14]


def test_trues_falses() -> None:
    rng = mitrange(yy(2000), yy(2002))
    t = TSeries.trues(rng)
    f = TSeries.falses(rng)
    assert t.dtype == np.bool_
    assert list(t.values) == [True, True, True]
    assert list(f.values) == [False, False, False]


# ---------------------------------------------------------------------------
# Accessors and basic protocols
# ---------------------------------------------------------------------------


def test_range_property() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0, 3.0, 4.0])
    assert s.range == mitrange(qq(2020, 1), qq(2020, 4))


def test_range_empty() -> None:
    s = TSeries(qq(2020, 1))
    assert s.range.is_empty()


def test_lastdate_empty() -> None:
    s = TSeries(qq(2020, 1))
    # Empty series have lastdate = firstdate - 1, by convention.
    assert s.lastdate == qq(2020, 1) - 1


def test_iteration() -> None:
    s = TSeries(yy(2000), [1.0, 2.0, 3.0])
    assert list(s) == [1.0, 2.0, 3.0]


def test_bool_raises_on_multi_element() -> None:
    s = TSeries(yy(2000), [1.0, 2.0])
    with pytest.raises(ValueError, match="ambiguous"):
        bool(s)


def test_any_all() -> None:
    s = TSeries(yy(2000), np.array([False, True, False]))
    assert s.any() is True
    assert s.all() is False


# ---------------------------------------------------------------------------
# Indexing — integer / slice / array
# ---------------------------------------------------------------------------


def test_int_index_returns_scalar() -> None:
    s = TSeries(qq(2020, 1), [11.0, 12.0, 13.0, 14.0])
    assert s[0] == 11.0
    assert s[-1] == 14.0


def test_int_slice_returns_plain_array() -> None:
    s = TSeries(qq(2020, 1), [11.0, 12.0, 13.0, 14.0])
    sub = s[1:3]
    assert isinstance(sub, np.ndarray)
    assert list(sub) == [12.0, 13.0]


def test_array_of_int_returns_plain_array() -> None:
    s = TSeries(qq(2020, 1), [11.0, 12.0, 13.0, 14.0])
    sub = s[[0, 2, 3]]
    assert isinstance(sub, np.ndarray)
    assert list(sub) == [11.0, 13.0, 14.0]


def test_bool_array_returns_plain_array() -> None:
    s = TSeries(qq(2020, 1), [11.0, 12.0, 13.0, 14.0])
    mask = np.array([True, False, True, False])
    sub = s[mask]
    assert isinstance(sub, np.ndarray)
    assert list(sub) == [11.0, 13.0]


def test_bare_bool_index_raises() -> None:
    s = TSeries(yy(2000), [1.0])
    with pytest.raises(TypeError, match="ambiguous"):
        _ = s[True]


# ---------------------------------------------------------------------------
# Indexing — MIT / MITRange / MIT slice
# ---------------------------------------------------------------------------


def test_mit_index_returns_scalar() -> None:
    s = TSeries(qq(2020, 1), [11.0, 12.0, 13.0, 14.0])
    assert s[qq(2020, 1)] == 11.0
    assert s[qq(2020, 4)] == 14.0


def test_mit_index_wrong_freq_raises() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0])
    with pytest.raises(TypeError, match="frequencies"):
        _ = s[MIT(Unit(), 1)]


def test_mit_index_out_of_range_raises() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0])
    with pytest.raises(IndexError):
        _ = s[qq(2021, 1)]


def test_mitrange_index_returns_tseries() -> None:
    s = TSeries(qq(2020, 1), [11.0, 12.0, 13.0, 14.0])
    sub = s[mitrange(qq(2020, 2), qq(2020, 3))]
    assert isinstance(sub, TSeries)
    assert sub.firstdate == qq(2020, 2)
    assert list(sub.values) == [12.0, 13.0]


def test_mitrange_index_step_returns_array() -> None:
    s = TSeries(qq(2020, 1), [11.0, 12.0, 13.0, 14.0])
    sub = s[mitrange(qq(2020, 1), qq(2020, 4), 2)]
    assert isinstance(sub, np.ndarray)
    assert list(sub) == [11.0, 13.0]


def test_mit_slice_sugar() -> None:
    s = TSeries(qq(2020, 1), [11.0, 12.0, 13.0, 14.0])
    sub = s[qq(2020, 2) : qq(2020, 3)]
    assert isinstance(sub, TSeries)
    assert list(sub.values) == [12.0, 13.0]


def test_mit_slice_one_sided_raises() -> None:
    s = TSeries(qq(2020, 1), [11.0, 12.0])
    with pytest.raises(TypeError, match="MIT endpoints on both sides"):
        _ = s[qq(2020, 1) : 5]  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Assignment — basic
# ---------------------------------------------------------------------------


def test_int_assign() -> None:
    s = TSeries(qq(2020, 1), [11.0, 12.0, 13.0, 14.0])
    s[1] = 99.0
    assert list(s.values) == [11.0, 99.0, 13.0, 14.0]


def test_int_slice_assign() -> None:
    s = TSeries(qq(2020, 1), [11.0, 12.0, 13.0, 14.0])
    s[1:3] = [99.0, 100.0]
    assert list(s.values) == [11.0, 99.0, 100.0, 14.0]


def test_mit_assign_in_range() -> None:
    s = TSeries(qq(2020, 1), [11.0, 12.0, 13.0, 14.0])
    s[qq(2020, 2)] = 99.0
    assert s[qq(2020, 2)] == 99.0


def test_mit_assign_wrong_freq_raises() -> None:
    s = TSeries(qq(2020, 1), [1.0])
    with pytest.raises(TypeError, match="frequencies"):
        s[MIT(Unit(), 1)] = 5.0


def test_mitrange_assign_scalar() -> None:
    s = TSeries(qq(2020, 1), [11.0, 12.0, 13.0, 14.0])
    s[mitrange(qq(2020, 2), qq(2020, 3))] = 99.0
    assert list(s.values) == [11.0, 99.0, 99.0, 14.0]


def test_mitrange_assign_array() -> None:
    s = TSeries(qq(2020, 1), [11.0, 12.0, 13.0, 14.0])
    s[mitrange(qq(2020, 2), qq(2020, 3))] = [99.0, 100.0]
    assert list(s.values) == [11.0, 99.0, 100.0, 14.0]


def test_mitrange_assign_tseries_aligns_by_mit() -> None:
    s = TSeries(qq(2020, 1), [11.0, 12.0, 13.0, 14.0])
    other = TSeries(qq(2019, 1), np.arange(10.0, 20.0))  # 2019Q1..2021Q2
    s[mitrange(qq(2020, 2), qq(2020, 3))] = other  # picks 2020Q2, 2020Q3 from other
    assert s[qq(2020, 2)] == other[qq(2020, 2)]
    assert s[qq(2020, 3)] == other[qq(2020, 3)]


def test_mit_assign_step_range() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0, 3.0, 4.0])
    s[mitrange(qq(2020, 1), qq(2020, 4), 2)] = 99.0
    assert list(s.values) == [99.0, 2.0, 99.0, 4.0]


# ---------------------------------------------------------------------------
# Assignment — resize-on-out-of-range
# ---------------------------------------------------------------------------


def test_mit_assign_extends_left() -> None:
    s = TSeries(qq(2020, 1), [11.0, 12.0, 13.0, 14.0])
    s[qq(2019, 4)] = 99.0
    assert s.firstdate == qq(2019, 4)
    assert s[qq(2019, 4)] == 99.0
    # original values preserved
    assert s[qq(2020, 1)] == 11.0
    assert s[qq(2020, 4)] == 14.0


def test_mit_assign_extends_right_with_nan_fill() -> None:
    s = TSeries(qq(2020, 1), [11.0, 12.0, 13.0, 14.0])
    s[qq(2021, 1)] = 99.0
    assert s.lastdate == qq(2021, 1)
    assert s[qq(2021, 1)] == 99.0
    # Cell at 2020Q4 is preserved; 2021 cells filled with NaN before the new one.
    assert s[qq(2020, 4)] == 14.0


def test_mit_assign_extends_int_with_typenan() -> None:
    s = TSeries(yy(2020), [1, 2, 3], dtype=np.int32)
    s[yy(2017)] = -1
    assert s.firstdate == yy(2017)
    assert s[yy(2017)] == -1
    assert s[yy(2018)] == np.iinfo(np.int32).max
    assert s[yy(2019)] == np.iinfo(np.int32).max


def test_mitrange_assign_extends_in_both_directions() -> None:
    s = TSeries(mm(2018, 1), [float(i) for i in range(1, 13)])
    s[mitrange(mm(2017, 10), mm(2017, 11))] = 1.0
    assert s.firstdate == mm(2017, 10)
    assert s[mm(2017, 10)] == 1.0
    assert s[mm(2017, 11)] == 1.0
    # Original Jan 2018 still present.
    assert s[mm(2018, 1)] == 1.0


def test_mit_assign_empty_series_takes_that_mit_as_first() -> None:
    s: TSeries = TSeries(qq(2020, 1))
    s[qq(2021, 2)] = 7.0
    assert s.firstdate == qq(2021, 2)
    assert s.lastdate == qq(2021, 2)
    assert s[qq(2021, 2)] == 7.0


# ---------------------------------------------------------------------------
# Boolean-TSeries indexing
# ---------------------------------------------------------------------------


def test_bool_tseries_index_returns_values() -> None:
    s = TSeries(qq(2020, 1), [11.0, 12.0, 13.0, 14.0])
    mask = s < 13.0
    assert isinstance(mask, TSeries)
    assert mask.dtype == np.bool_
    assert list(mask.values) == [True, True, False, False]
    assert list(s[mask]) == [11.0, 12.0]


def test_bool_tseries_index_assigns_values() -> None:
    s = TSeries(qq(2020, 1), [11.0, 12.0, 13.0, 14.0])
    s[s < 13.0] = -1.0
    assert list(s.values) == [-1.0, -1.0, 13.0, 14.0]


def test_bool_tseries_index_wrong_range_raises() -> None:
    s = TSeries(qq(2020, 1), [11.0, 12.0, 13.0, 14.0])
    mask = TSeries(qq(2020, 2), np.array([True, False]))
    with pytest.raises(IndexError):
        _ = s[mask]


# ---------------------------------------------------------------------------
# Arithmetic — scalar, vector, TSeries alignment
# ---------------------------------------------------------------------------


def test_scalar_add() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
    out = s + 5
    assert isinstance(out, TSeries)
    assert list(out.values) == [6.0, 7.0, 8.0]
    assert out.firstdate == s.firstdate


def test_scalar_radd() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
    out = 5 + s
    assert isinstance(out, TSeries)
    assert list(out.values) == [6.0, 7.0, 8.0]


def test_scalar_multiply() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
    assert list((3 * s).values) == [3.0, 6.0, 9.0]
    assert list((s * 3).values) == [3.0, 6.0, 9.0]


def test_negate() -> None:
    s = TSeries(qq(2020, 1), [1.0, -2.0, 3.0])
    out = -s
    assert list(out.values) == [-1.0, 2.0, -3.0]


def test_abs() -> None:
    s = TSeries(qq(2020, 1), [1.0, -2.0, 3.0])
    out = abs(s)
    assert list(out.values) == [1.0, 2.0, 3.0]


def test_pow() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
    out = s**2
    assert list(out.values) == [1.0, 4.0, 9.0]


def test_truediv() -> None:
    s = TSeries(qq(2020, 1), [2.0, 4.0, 8.0])
    assert list((s / 2).values) == [1.0, 2.0, 4.0]


def test_tseries_add_same_range() -> None:
    a = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
    b = TSeries(qq(2020, 1), [10.0, 20.0, 30.0])
    out = a + b
    assert isinstance(out, TSeries)
    assert list(out.values) == [11.0, 22.0, 33.0]
    assert out.range == a.range


def test_tseries_add_intersection_range() -> None:
    a = TSeries(MIT(Unit(), 1), [7.0, 7.0, 7.0])  # 1U..3U
    b = TSeries(MIT(Unit(), 3), [2.0, 4.0, 5.0])  # 3U..5U
    out = a + b
    assert isinstance(out, TSeries)
    assert out.firstdate == MIT(Unit(), 3)
    assert list(out.values) == [9.0]


def test_tseries_add_disjoint_returns_empty() -> None:
    a = TSeries(MIT(Unit(), 1), [1.0, 2.0])
    b = TSeries(MIT(Unit(), 10), [10.0, 20.0])
    out = a + b
    assert isinstance(out, TSeries)
    assert len(out) == 0


def test_tseries_add_mixed_freq_raises() -> None:
    a = TSeries(qq(2020, 1), [1.0, 2.0])
    b = TSeries(mm(2020, 1), [1.0, 2.0])
    with pytest.raises(TypeError, match="frequencies"):
        _ = a + b


def test_tseries_add_array_same_length() -> None:
    a = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
    out = a + np.array([10.0, 20.0, 30.0])
    assert isinstance(out, TSeries)
    assert list(out.values) == [11.0, 22.0, 33.0]


def test_tseries_add_array_wrong_length_raises() -> None:
    a = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="broadcast"):
        _ = a + np.array([1.0, 2.0])


# ---------------------------------------------------------------------------
# NumPy ufunc dispatch
# ---------------------------------------------------------------------------


def test_np_sin_returns_tseries() -> None:
    s = TSeries(qq(2020, 1), [0.0, np.pi / 2, np.pi])
    out = np.sin(s)
    assert isinstance(out, TSeries)
    assert out.firstdate == qq(2020, 1)
    assert np.allclose(out.values, [0.0, 1.0, 0.0], atol=1e-12)


def test_np_exp_log_roundtrip() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
    assert isinstance(np.exp(s), TSeries)
    assert np.allclose(np.log(np.exp(s)).values, s.values)


def test_np_add_reduce_returns_scalar() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0, 3.0, 4.0])
    total = np.add.reduce(s)
    assert isinstance(total, np.floating)
    assert total == 10.0


def test_np_add_accumulate_returns_array() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0, 3.0, 4.0])
    acc = np.add.accumulate(s)
    # Reductions return bare arrays (frequency dropped).
    assert isinstance(acc, np.ndarray)
    assert list(acc) == [1.0, 3.0, 6.0, 10.0]


# ---------------------------------------------------------------------------
# __array_function__ dispatch
# ---------------------------------------------------------------------------


def test_np_concatenate_two_tseries() -> None:
    a = TSeries(qq(2020, 1), [1.0, 2.0])
    b = TSeries(qq(2030, 1), [3.0, 4.0])
    out = np.concatenate([a, b])
    assert isinstance(out, TSeries)
    # Keeps first arg's firstdate, per Julia's vcat.
    assert out.firstdate == qq(2020, 1)
    assert list(out.values) == [1.0, 2.0, 3.0, 4.0]


def test_np_concatenate_tseries_and_array() -> None:
    a = TSeries(qq(2020, 1), [1.0, 2.0])
    out = np.concatenate([a, np.array([3.0, 4.0])])
    assert isinstance(out, TSeries)
    assert list(out.values) == [1.0, 2.0, 3.0, 4.0]


def test_np_concatenate_mixed_freq_raises() -> None:
    a = TSeries(qq(2020, 1), [1.0])
    b = TSeries(mm(2020, 1), [1.0])
    with pytest.raises(TypeError, match="frequencies"):
        _ = np.concatenate([a, b])


def test_np_array_equal_handler() -> None:
    a = TSeries(qq(2020, 1), [1.0, 2.0])
    b = TSeries(qq(2020, 1), [1.0, 2.0])
    assert np.array_equal(a, b)


def test_np_allclose_handler() -> None:
    a = TSeries(qq(2020, 1), [1.0, 2.0])
    b = TSeries(qq(2020, 1), [1.0 + 1e-12, 2.0])
    assert np.allclose(a, b)


def test_np_mean_falls_back_to_array() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0, 3.0, 4.0])
    # No registered handler — fallback unwraps to ndarray.
    assert np.mean(s) == 2.5


# ---------------------------------------------------------------------------
# __array__ (escape hatch)
# ---------------------------------------------------------------------------


def test_np_asarray_returns_underlying_values() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
    arr = np.asarray(s)
    assert isinstance(arr, np.ndarray)
    assert list(arr) == [1.0, 2.0, 3.0]


def test_np_array_with_copy_true_copies() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
    arr = np.array(s, copy=True)
    arr[0] = 99.0
    # Mutating the copy must not touch the series.
    assert s.values[0] == 1.0


def test_np_asarray_with_dtype_conversion() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
    arr = np.asarray(s, dtype=np.int32)
    assert arr.dtype == np.int32
    assert list(arr) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Whole-object comparison
# ---------------------------------------------------------------------------


def test_equals_true() -> None:
    a = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
    b = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
    assert a.equals(b)


def test_equals_false_on_value_diff() -> None:
    a = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
    b = TSeries(qq(2020, 1), [1.0, 2.0, 4.0])
    assert not a.equals(b)


def test_equals_false_on_firstdate_diff() -> None:
    a = TSeries(qq(2020, 1), [1.0, 2.0])
    b = TSeries(qq(2021, 1), [1.0, 2.0])
    assert not a.equals(b)


def test_equals_false_on_freq_diff() -> None:
    a = TSeries(qq(2020, 1), [1.0, 2.0])
    b = TSeries(mm(2020, 1), [1.0, 2.0])
    assert not a.equals(b)


def test_allclose_true_within_tol() -> None:
    a = TSeries(qq(2020, 1), [1.0, 2.0])
    b = TSeries(qq(2020, 1), [1.0 + 1e-12, 2.0])
    assert a.allclose(b)


def test_allclose_false_outside_tol() -> None:
    a = TSeries(qq(2020, 1), [1.0, 2.0])
    b = TSeries(qq(2020, 1), [1.5, 2.0])
    assert not a.allclose(b)


# ---------------------------------------------------------------------------
# Elementwise comparison via Python operators
# ---------------------------------------------------------------------------


def test_lt_scalar_returns_bool_tseries() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0, 3.0, 4.0])
    out = s < 3
    assert isinstance(out, TSeries)
    assert out.dtype == np.bool_
    assert list(out.values) == [True, True, False, False]


def test_eq_tseries_returns_bool_tseries() -> None:
    a = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
    b = TSeries(qq(2020, 1), [1.0, 5.0, 3.0])
    out = a == b
    assert isinstance(out, TSeries)
    assert list(out.values) == [True, False, True]


def test_unhashable() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0])
    with pytest.raises(TypeError):
        hash(s)


# ---------------------------------------------------------------------------
# Copy / similar / resize
# ---------------------------------------------------------------------------


def test_copy_independence() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
    c = s.copy()
    c.values[0] = 99.0
    assert s.values[0] == 1.0


def test_copy_module_copy() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
    c = _copy.copy(s)
    assert c.equals(s)
    assert c is not s


def test_deepcopy() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
    c = _copy.deepcopy(s)
    c.values[0] = 99.0
    assert s.values[0] == 1.0


def test_similar_preserves_dtype_and_range() -> None:
    s = TSeries(qq(2020, 1), [1, 2, 3], dtype=np.int32)
    sim = s.similar()
    assert sim.firstdate == s.firstdate
    assert len(sim) == len(s)
    assert sim.dtype == np.int32


def test_resize_grows_with_nan_fill() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0])
    s.resize(mitrange(qq(2020, 1), qq(2020, 4)))
    assert len(s) == 4
    assert list(s.values[:2]) == [1.0, 2.0]
    assert np.isnan(s.values[2])
    assert np.isnan(s.values[3])


def test_resize_shrinks_from_front() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0, 3.0, 4.0])
    s.resize(mitrange(qq(2020, 3), qq(2020, 4)))
    assert s.firstdate == qq(2020, 3)
    assert list(s.values) == [3.0, 4.0]


def test_resize_rejects_mixed_freq() -> None:
    s = TSeries(qq(2020, 1), [1.0])
    with pytest.raises(TypeError, match="frequencies"):
        s.resize(mitrange(mm(2020, 1), mm(2020, 3)))


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def test_repr_short_lists_every_row() -> None:
    s = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
    out = repr(s)
    assert "TSeries" in out
    assert "Quarterly" in out
    assert "2020Q1" in out
    assert "1.0" in out
    assert "2020Q3" in out


def test_repr_long_truncates() -> None:
    s = TSeries(qq(2000, 1), np.arange(50.0))
    out = repr(s)
    assert "⋮" in out


def test_repr_empty() -> None:
    s = TSeries(qq(2020, 1))
    out = repr(s)
    assert out.startswith("Empty")
    assert "Quarterly" in out


def test_repr_int_dtype_includes_dtype() -> None:
    s = TSeries(yy(2000), [1, 2, 3], dtype=np.int32)
    out = repr(s)
    assert "int32" in out


# ---------------------------------------------------------------------------
# Frequency-coverage smoke (Monthly / Yearly / Unit)
# ---------------------------------------------------------------------------


def test_monthly_indexing() -> None:
    ts_m = TSeries(mm(2018, 1), [float(i) for i in range(1, 13)])
    sub = ts_m[mitrange(mm(2018, 1), mm(2018, 12))]
    assert isinstance(sub, TSeries)
    assert sub.equals(ts_m)


def test_yearly_indexing() -> None:
    ts_y = TSeries(yy(2018), [float(i) for i in range(1, 13)])
    assert isinstance(ts_y.frequency, Yearly)
    assert ts_y[yy(2020)] == 3.0


def test_unit_indexing() -> None:
    ts_u = TSeries(MIT(Unit(), 1), [1.0, 2.0, 3.0, 4.0, 5.0])
    assert ts_u.firstdate == MIT(Unit(), 1)
    assert ts_u[MIT(Unit(), 3)] == 3.0


def test_quarterly_partial_out_of_range_raises() -> None:
    ts_q = TSeries(qq(2018, 1), [float(i) for i in range(1, 13)])
    with pytest.raises(IndexError):
        _ = ts_q[mitrange(qq(2017, 1), qq(2018, 4))]


def test_frequency_property_is_singleton() -> None:
    s = TSeries(qq(2020, 1), [1.0])
    # Cached singletons from frequencies.py — see decision 15.
    assert s.frequency is Quarterly()


# ---------------------------------------------------------------------------
# Iris-style assignment cases from the Julia test
# ---------------------------------------------------------------------------


def test_iris_partial_range_assign_from_tseries() -> None:
    x = TSeries(qq(2020, 1), [0.0, 0.0, 0.0])
    y = TSeries(qq(2020, 1), [1.0, 1.0, 1.0])
    x[mitrange(qq(2020, 1), qq(2020, 2))] = y
    assert list(x.values) == [1.0, 1.0, 0.0]


def test_iris_in_range_partial_with_array() -> None:
    s = TSeries(qq(2018, 1), [float(i) for i in range(1, 9)])
    s[mitrange(qq(2018, 3), qq(2019, 2))] = [0.0, 1.0, 2.0, 3.0]
    assert list(s.values) == [1.0, 2.0, 0.0, 1.0, 2.0, 3.0, 7.0, 8.0]


def test_setindex_step_range_of_mit() -> None:
    s = TSeries(qq(2018, 1), [float(i) for i in range(1, 9)])
    s[mitrange(qq(2018, 1), qq(2019, 4), 2)] = [0.0, 1.0, 2.0, 3.0]
    assert list(s.values) == [0.0, 2.0, 1.0, 4.0, 2.0, 6.0, 3.0, 8.0]
