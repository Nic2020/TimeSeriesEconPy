# SPDX-License-Identifier: MIT
"""Empty-range comparisons remain Boolean and support empty-owner growth."""

import numpy as np
import pytest

from tsecon import MITRange, TSeries, mm, qq


@pytest.mark.parametrize("step", [1, 2, -1, -2])
def test_empty_vs_nonempty_both_orders(step: int) -> None:
    start = qq(2020, 1)
    empty = MITRange(start, start - step, step)
    populated = MITRange(start, start + 2 * step, step)
    assert (empty == populated) is False
    assert (populated == empty) is False
    assert empty != populated
    assert populated != empty


def test_empty_equality_ignores_anchor_and_step_but_preserves_frequency() -> None:
    start = qq(2020, 1)
    a = MITRange(start, start - 1)
    b = MITRange(start + 5, start + 6, -2)
    other = MITRange(mm(2020, 1), mm(2019, 12))
    assert a == b
    assert hash(a) == hash(b)
    assert {a: "empty"}[b] == "empty"
    assert a != other
    assert other != a


@pytest.mark.parametrize("step", [2, -2])
def test_nonlanding_stops_keep_equality_and_hash(step: int) -> None:
    start = qq(2020, 1)
    a = MITRange(start, start + 2 * step, step)
    b = MITRange(start, start + 2 * step + (1 if step > 0 else -1), step)
    assert a == b
    assert hash(a) == hash(b)
    assert a != MITRange(start, start + 2 * step, 1 if step > 0 else -1)


@pytest.mark.parametrize("assignment", ["scalar", "forward", "reverse"])
def test_empty_owner_grows_at_its_anchor(assignment: str) -> None:
    start = qq(2020, 1)
    series = TSeries(start, np.array([], dtype=np.int32))
    if assignment == "scalar":
        series[start] = 7
        expected = [7]
    elif assignment == "forward":
        series[MITRange(start, start + 2)] = [7, 8, 9]
        expected = [7, 8, 9]
    else:
        series[MITRange(start + 2, start, -1)] = [9, 8, 7]
        expected = [7, 8, 9]
    assert series.firstdate == start
    assert series.dtype == np.int32
    np.testing.assert_array_equal(series.values, expected)
