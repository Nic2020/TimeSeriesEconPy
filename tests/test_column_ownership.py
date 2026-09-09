# SPDX-License-Identifier: MIT
"""A table column stays attached; copies and serialized columns own their data."""

import copy
import gc
import pickle
import weakref
from typing import Any

import numpy as np
import pytest

from tsecon import MITRange, MVTSeries, qq, rec, rec_linear, rename_columns_inplace, shift_inplace
from tsecon.fconvert import strip_tseries_inplace


def table() -> MVTSeries:
    return MVTSeries(qq(2020, 1), a=[1.0, 2.0, 3.0], b=[4.0, 5.0, 6.0])


@pytest.mark.parametrize(
    "operation", ["right", "left", "range", "resize", "shrink", "shift", "strip", "rec", "linear"]
)
def test_column_structure_rejection_is_atomic(operation: str) -> None:
    parent = table()
    col = parent.a
    start = parent.firstdate
    before = parent.values.copy()
    calls = []

    def callback(date: Any) -> float:
        calls.append(date)
        return 99.0

    def mutate() -> None:
        if operation == "right":
            col[start + 3] = 99.0
        elif operation == "left":
            col[start - 1] = 99.0
        elif operation == "range":
            col[MITRange(start + 3, start, -1)] = 99.0
        elif operation == "resize":
            col.resize(MITRange(start - 1, start + 3))
        elif operation == "shrink":
            col.resize(MITRange(start + 1, start + 2))
        elif operation == "shift":
            shift_inplace(col, 1)
        elif operation == "strip":
            col[start] = np.nan
            before[0, 0] = np.nan
            strip_tseries_inplace(col)
        elif operation == "rec":
            rec(MITRange(start + 1, start + 3), col, callback)
        else:
            rec_linear(col, [0.9], [1], MITRange(start + 1, start + 3))

    with pytest.raises(ValueError, match="column"):
        mutate()
    assert calls == []
    assert col is parent.a
    assert col.range == parent.range
    np.testing.assert_array_equal(parent.values, before)
    col[start + 1] = 123.0
    assert parent.values[1, 0] == 123.0


def test_parent_shift_and_rename_preserve_retained_handle() -> None:
    parent = table()
    col = parent.a
    shift_inplace(parent, 2)
    assert col.range == parent.range
    rename_columns_inplace(parent, ["renamed", "b"])
    assert parent.renamed is col
    col[parent.firstdate] = 100.0
    assert parent.values[0, 0] == 100.0
    with pytest.raises(ValueError, match="column"):
        col[parent.lastdate + 1] = 0.0


@pytest.mark.parametrize(
    "clone", [copy.copy, copy.deepcopy, lambda x: x.copy(), lambda x: pickle.loads(pickle.dumps(x))]
)
def test_column_copies_and_pickle_own_storage(clone: Any) -> None:
    parent = table()
    out = clone(parent.a)
    out[out.lastdate + 1] = 99.0
    out[out.firstdate] = 100.0
    np.testing.assert_array_equal(parent.a.values, [1.0, 2.0, 3.0])
    assert len(out) == 4


@pytest.mark.parametrize(
    "clone", [copy.copy, copy.deepcopy, lambda x: x.copy(), lambda x: pickle.loads(pickle.dumps(x))]
)
def test_table_copy_and_pickle_restore_ownership(clone: Any) -> None:
    parent = table()
    out = clone(parent)
    out.a[out.firstdate] = 100.0
    assert out.values[0, 0] == 100.0
    assert parent.values[0, 0] == 1.0
    with pytest.raises(ValueError, match="column"):
        out.a[out.lastdate + 1] = 0.0


def test_retained_column_keeps_parent_alive_and_cycle_is_collected() -> None:
    class WeakTable(MVTSeries):
        pass

    parent = WeakTable(qq(2020, 1), a=[1.0, 2.0])
    reference = weakref.ref(parent)
    col = parent.a
    del parent
    gc.collect()
    assert reference() is not None
    col[col.firstdate] = 5.0
    assert reference().values[0, 0] == 5.0
    del col
    gc.collect()
    assert reference() is None


def test_in_range_recurrences_and_noop_resize_work() -> None:
    parent = table()
    col = parent.a
    assert col.resize(col.range) is col
    rec(MITRange(col.firstdate + 1, col.lastdate), col, lambda date: 2 * col[date - 1])
    np.testing.assert_array_equal(parent.values[:, 0], [1.0, 2.0, 4.0])
    rec_linear(col, [3.0], [1], MITRange(col.firstdate + 1, col.lastdate))
    np.testing.assert_array_equal(parent.values[:, 0], [1.0, 3.0, 9.0])


def test_empty_column_cannot_grow() -> None:
    parent = MVTSeries(qq(2020, 1), a=[])
    with pytest.raises(ValueError, match="column"):
        parent.a[parent.firstdate] = 1.0
    assert len(parent) == len(parent.a) == 0


def test_backward_recurrence_rejects_before_callback() -> None:
    parent = table()
    before = parent.values.copy()
    calls = []

    def callback(date: Any) -> float:
        calls.append(date)
        return 9.0

    with pytest.raises(ValueError, match="column"):
        rec(MITRange(parent.lastdate, parent.firstdate - 1, -1), parent.a, callback)
    assert calls == []
    np.testing.assert_array_equal(parent.values, before)
