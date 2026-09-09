# SPDX-License-Identifier: MIT
"""Date-range operations agree with enumeration of individual dates."""

import numpy as np
import pytest

from tsecon import MITRange, MVTSeries, TSeries, mm, qq

RANGES = [
    (0, 5, 1),
    (0, 5, 2),
    (5, 0, -1),
    (4, 0, -2),
    (5, 0, -2),
    (3, 3, -1),
    (0, 0, -3),
    (4, 2, 1),
    (2, 4, -1),
    (9, 8, 1),
]


def make_series(table: bool) -> TSeries | MVTSeries:
    return (
        MVTSeries(qq(2020, 1), a=np.arange(6.0), b=np.arange(6.0) + 10)
        if table
        else TSeries(qq(2020, 1), np.arange(6.0))
    )


@pytest.mark.parametrize("bounds", RANGES)
@pytest.mark.parametrize("mode", ["series", "table", "column", "columns"])
def test_enumerated_read_and_write(bounds: tuple[int, int, int], mode: str) -> None:
    obj = make_series(mode != "series")
    start = obj.firstdate
    lo, hi, step = bounds
    dates = MITRange(start + lo, start + hi, step)
    positions = [date.value - start.value for date in dates]
    key = (dates, "a") if mode == "column" else (dates, ["b", "a"]) if mode == "columns" else dates
    expected = obj.values[positions].copy()
    if mode == "column":
        expected = expected[:, 0]
    elif mode == "columns":
        expected = expected[:, [1, 0]]
    result = obj[key]
    np.testing.assert_array_equal(np.asarray(result), expected)
    if step == 1:
        assert isinstance(result, (TSeries, MVTSeries))
        assert result.firstdate == dates.start
    else:
        assert isinstance(result, np.ndarray)
    assert not np.shares_memory(np.asarray(result), obj.values)
    replacement = np.arange(expected.size).reshape(expected.shape) + 100
    baseline = obj.values.copy()
    obj[key] = replacement
    if mode == "column":
        baseline[positions, 0] = replacement
    elif mode == "columns":
        baseline[np.ix_(positions, [1, 0])] = replacement
    else:
        baseline[positions] = replacement
    np.testing.assert_array_equal(obj.values, baseline)
    assert obj.firstdate == start


@pytest.mark.parametrize("table", [False, True])
@pytest.mark.parametrize("bounds", [(6, 0, -1), (4, -1, -1), (-1, -2, -1), (7, 8, 1)])
def test_invalid_read_bounds(table: bool, bounds: tuple[int, int, int]) -> None:
    obj = make_series(table)
    a, b, step = bounds
    with pytest.raises(IndexError):
        _ = obj[MITRange(obj.firstdate + a, obj.firstdate + b, step)]


@pytest.mark.parametrize("table", [False, True])
def test_date_slice_sugar_and_aligned_rhs(table: bool) -> None:
    obj = make_series(table)
    start = obj.firstdate
    rhs = make_series(table)
    rhs.values[:] += 20
    dates = MITRange(start + 5, start, -2)
    obj[start + 5 : start : -2] = rhs
    positions = [date.value - start.value for date in dates]
    np.testing.assert_array_equal(np.asarray(obj[dates]), rhs.values[positions])


@pytest.mark.parametrize("columns", ["a", ["b", "a"]])
def test_table_aligned_rhs(columns: str | list[str]) -> None:
    obj = make_series(True)
    rhs = make_series(True)
    rhs.values[:] += 20
    dates = MITRange(obj.firstdate + 5, obj.firstdate, -1)
    obj[dates, columns] = rhs["a"] if isinstance(columns, str) else rhs
    np.testing.assert_array_equal(obj.values[:, 0], rhs.values[:, 0])
    if isinstance(columns, list):
        np.testing.assert_array_equal(obj.values, rhs.values)


def test_reverse_assignment_grows_owner_and_preserves_gaps() -> None:
    obj = TSeries(qq(2020, 1), [1.0, 2.0])
    start = obj.firstdate
    obj[MITRange(start + 4, start - 2, -2)] = [40.0, 20.0, 0.0, -20.0]
    assert obj.firstdate == start - 2
    np.testing.assert_array_equal(obj.values, [-20.0, np.nan, 0.0, 2.0, 20.0, np.nan, 40.0])


@pytest.mark.parametrize(
    "rhs", [[1.0, 2.0], ["bad", "data", "here"], TSeries(mm(2020, 1), [1.0, 2.0, 3.0])]
)
def test_invalid_assignment_does_not_extend(rhs: object) -> None:
    obj = TSeries(qq(2020, 1), [1.0, 2.0])
    start = obj.firstdate
    with pytest.raises((ValueError, TypeError)):
        obj[MITRange(start + 4, start, -2)] = rhs
    assert obj.firstdate == start
    np.testing.assert_array_equal(obj.values, [1.0, 2.0])


@pytest.mark.parametrize("table", [False, True])
def test_empty_wrong_frequency_still_rejected(table: bool) -> None:
    obj = make_series(table)
    dates = MITRange(mm(2020, 2), mm(2020, 1))
    with pytest.raises(TypeError):
        _ = obj[dates]
    with pytest.raises(TypeError):
        obj[dates] = 0


@pytest.mark.parametrize("mode", ["table", "column", "columns"])
@pytest.mark.parametrize("bad", ["bounds", "shape", "casting", "source"])
def test_rejected_table_range_assignment_is_atomic(mode: str, bad: str) -> None:
    obj = make_series(True)
    before = obj.values.copy()
    dates = MITRange(obj.firstdate + (6 if bad == "bounds" else 5), obj.firstdate, -1)
    key = (dates, "a") if mode == "column" else (dates, ["a", "b"]) if mode == "columns" else dates
    value = {"bounds": 0, "shape": np.ones((3, 3)), "casting": "bad"}.get(bad)
    if bad == "source":
        value = (
            TSeries(mm(2020, 1), [1.0, 2.0])
            if mode == "column"
            else MVTSeries(mm(2020, 1), a=[1.0, 2.0])
        )
    with pytest.raises((IndexError, TypeError, ValueError)):
        obj[key] = value
    np.testing.assert_array_equal(obj.values, before)


@pytest.mark.parametrize("table", [False, True])
def test_stride_stop_outside_storage_but_selected_dates_inside(table: bool) -> None:
    obj = make_series(table)
    dates = MITRange(obj.firstdate + 4, obj.firstdate - 1, -2)
    np.testing.assert_array_equal(np.asarray(obj[dates]), obj.values[[4, 2, 0]])


@pytest.mark.parametrize("table", [False, True])
def test_empty_storage_selection_and_scalar_assignment(table: bool) -> None:
    obj = MVTSeries(qq(2020, 1), a=[]) if table else TSeries(qq(2020, 1), [])
    dates = MITRange(obj.firstdate + 3, obj.firstdate + 2)
    assert len(obj[dates]) == 0
    obj[dates] = 7
    assert len(obj) == 0
    assert obj.firstdate == qq(2020, 1)
