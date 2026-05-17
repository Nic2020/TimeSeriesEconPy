# SPDX-License-Identifier: MIT
"""Tests for ``MITRange`` — inclusive ranges of frequency-tagged moments."""

from __future__ import annotations

import pytest

from tsecon.frequencies import Monthly, Quarterly, Unit
from tsecon.mit import MIT, Duration, mm, qq
from tsecon.mitrange import MITRange, mitrange, rangeof_span


def test_basic_range_length_and_step() -> None:
    rng = qq(2020, 1).to(qq(2020, 4))
    assert isinstance(rng, MITRange)
    assert len(rng) == 4
    assert rng.step == 1
    assert rng.first() == qq(2020, 1)
    assert rng.last() == qq(2020, 4)


def test_empty_range() -> None:
    rng = MITRange(qq(2020, 1), qq(2019, 1))
    assert rng.is_empty()
    assert len(rng) == 0
    assert not rng


def test_iteration() -> None:
    rng = qq(2020, 1).to(qq(2020, 4))
    items = list(rng)
    assert items == [qq(2020, 1), qq(2020, 2), qq(2020, 3), qq(2020, 4)]
    for m in rng:
        assert m.frequency == Quarterly()
        assert rng.first() <= m <= rng.last()


def test_indexing_positive_and_negative() -> None:
    rng = qq(2020, 1).to(qq(2020, 4))
    assert rng[0] == qq(2020, 1)
    assert rng[3] == qq(2020, 4)
    assert rng[-1] == qq(2020, 4)
    assert rng[-4] == qq(2020, 1)


def test_index_out_of_range() -> None:
    rng = qq(2020, 1).to(qq(2020, 4))
    with pytest.raises(IndexError):
        _ = rng[4]
    with pytest.raises(IndexError):
        _ = rng[-5]


def test_index_rejects_bool() -> None:
    rng = qq(2020, 1).to(qq(2020, 4))
    with pytest.raises(TypeError):
        _ = rng[True]  # type: ignore[index]


def test_membership() -> None:
    rng = qq(2020, 1).to(qq(2020, 4))
    assert qq(2020, 1) in rng
    assert qq(2020, 4) in rng
    assert qq(2019, 4) not in rng
    assert qq(2021, 1) not in rng
    assert mm(2020, 1) not in rng  # different frequency
    assert "anything" not in rng


def test_slice_returns_subrange() -> None:
    rng = qq(2020, 1).to(qq(2020, 4))
    sub = rng[1:3]
    assert isinstance(sub, MITRange)
    assert list(sub) == [qq(2020, 2), qq(2020, 3)]


def test_mixed_freq_endpoints_raise() -> None:
    with pytest.raises(TypeError):
        MITRange(qq(2020, 1), mm(2020, 1))


def test_step_range_via_int() -> None:
    rng = mitrange(qq(2020, 1), qq(2021, 4), step=2)
    assert rng.step == 2
    assert list(rng) == [qq(2020, 1), qq(2020, 3), qq(2021, 1), qq(2021, 3)]


def test_step_range_via_duration() -> None:
    rng = mitrange(qq(2020, 1), qq(2021, 4), step=Duration(Quarterly(), 2))
    assert rng.step == 2
    assert list(rng) == [qq(2020, 1), qq(2020, 3), qq(2021, 1), qq(2021, 3)]


def test_step_range_via_duration_mixed_freq_raises() -> None:
    with pytest.raises(TypeError):
        mitrange(qq(2020, 1), qq(2021, 4), step=Duration(Monthly(), 2))


def test_step_zero_raises() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        MITRange(qq(2020, 1), qq(2020, 4), step=0)


# -- negative-step ranges (M1.6.1) ----------------------------------------


def test_negative_step_basic_iteration() -> None:
    rng = MITRange(MIT(Unit(), 10), MIT(Unit(), 1), step=-1)
    items = list(rng)
    assert items == [MIT(Unit(), v) for v in (10, 9, 8, 7, 6, 5, 4, 3, 2, 1)]
    assert len(rng) == 10
    assert rng.first() == MIT(Unit(), 10)
    assert rng.last() == MIT(Unit(), 1)


def test_negative_step_quarterly() -> None:
    rng = MITRange(qq(2021, 4), qq(2020, 1), step=-1)
    items = list(rng)
    assert items == [qq(2021, 4), qq(2021, 3), qq(2021, 2), qq(2021, 1),
                     qq(2020, 4), qq(2020, 3), qq(2020, 2), qq(2020, 1)]
    assert len(rng) == 8


def test_negative_step_multistep() -> None:
    # 10, 7, 4 — three elements; stop=2 is not reached because step skips it.
    rng = MITRange(MIT(Unit(), 10), MIT(Unit(), 2), step=-3)
    assert len(rng) == 3
    assert list(rng) == [MIT(Unit(), 10), MIT(Unit(), 7), MIT(Unit(), 4)]
    assert rng.last() == MIT(Unit(), 4)


def test_negative_step_empty_when_sign_mismatch() -> None:
    # start < stop with negative step is empty (the symmetric Julia behaviour).
    rng = MITRange(MIT(Unit(), 1), MIT(Unit(), 10), step=-1)
    assert rng.is_empty()
    assert len(rng) == 0
    assert list(rng) == []


def test_positive_step_empty_when_sign_mismatch() -> None:
    # start > stop with positive step is empty (existing behaviour, retained).
    rng = MITRange(MIT(Unit(), 10), MIT(Unit(), 1), step=1)
    assert rng.is_empty()
    assert len(rng) == 0


def test_negative_step_single_element() -> None:
    rng = MITRange(qq(2020, 2), qq(2020, 2), step=-1)
    assert len(rng) == 1
    assert list(rng) == [qq(2020, 2)]
    assert rng.first() == rng.last() == qq(2020, 2)


def test_negative_step_membership() -> None:
    rng = MITRange(MIT(Unit(), 10), MIT(Unit(), 1), step=-1)
    assert MIT(Unit(), 10) in rng
    assert MIT(Unit(), 1) in rng
    assert MIT(Unit(), 5) in rng
    assert MIT(Unit(), 11) not in rng
    assert MIT(Unit(), 0) not in rng
    # multi-step: 10, 7, 4 — 8 is between 4 and 10 but not on the stride.
    rng2 = MITRange(MIT(Unit(), 10), MIT(Unit(), 2), step=-3)
    assert MIT(Unit(), 7) in rng2
    assert MIT(Unit(), 4) in rng2
    assert MIT(Unit(), 8) not in rng2


def test_negative_step_indexing() -> None:
    rng = MITRange(MIT(Unit(), 10), MIT(Unit(), 1), step=-1)
    assert rng[0] == MIT(Unit(), 10)
    assert rng[1] == MIT(Unit(), 9)
    assert rng[-1] == MIT(Unit(), 1)
    assert rng[-2] == MIT(Unit(), 2)
    with pytest.raises(IndexError):
        _ = rng[10]


def test_negative_step_slice_subrange() -> None:
    rng = MITRange(MIT(Unit(), 10), MIT(Unit(), 1), step=-1)
    sub = rng[1:4]
    # Slice indices 1..3 in iteration order are MITs 9, 8, 7.
    assert isinstance(sub, MITRange)
    assert list(sub) == [MIT(Unit(), 9), MIT(Unit(), 8), MIT(Unit(), 7)]
    assert sub.step == -1


def test_negative_step_repr() -> None:
    rng = MITRange(MIT(Unit(), 10), MIT(Unit(), 1), step=-1)
    assert repr(rng) == "10U:-1:1U"


def test_negative_step_via_mitrange_helper() -> None:
    rng = mitrange(qq(2021, 4), qq(2020, 1), step=-1)
    assert rng.step == -1
    assert rng.first() == qq(2021, 4)
    assert rng.last() == qq(2020, 1)


def test_negative_step_via_duration() -> None:
    rng = mitrange(qq(2021, 4), qq(2020, 1), step=Duration(Quarterly(), -1))
    assert rng.step == -1
    assert len(rng) == 8


def test_negative_step_equality() -> None:
    a = MITRange(MIT(Unit(), 10), MIT(Unit(), 1), step=-1)
    b = MITRange(MIT(Unit(), 10), MIT(Unit(), 1), step=-1)
    assert a == b
    # Different direction → not equal even though the value set is the same.
    fwd = MITRange(MIT(Unit(), 1), MIT(Unit(), 10), step=1)
    assert a != fwd


def test_equality() -> None:
    a = qq(2020, 1).to(qq(2020, 4))
    b = MITRange(qq(2020, 1), qq(2020, 4))
    assert a == b
    c = qq(2020, 1).to(qq(2020, 3))
    assert a != c


def test_hashable() -> None:
    a = qq(2020, 1).to(qq(2020, 4))
    b = MITRange(qq(2020, 1), qq(2020, 4))
    d = {a: "first"}
    assert d[b] == "first"


def test_repr_unit_step() -> None:
    assert repr(qq(2020, 1).to(qq(2020, 4))) == "2020Q1:2020Q4"


def test_repr_step_range() -> None:
    rng = mitrange(qq(2020, 1), qq(2021, 4), step=2)
    assert repr(rng) == "2020Q1:2:2021Q4"


def test_rangeof_span_disjoint() -> None:
    a = MITRange(MIT(Unit(), 3), MIT(Unit(), 5))
    b = MITRange(MIT(Unit(), 4), MIT(Unit(), 6))
    out = rangeof_span(a, b)
    assert out == MITRange(MIT(Unit(), 3), MIT(Unit(), 6))


def test_rangeof_span_mixed_freq_raises() -> None:
    a = MITRange(MIT(Unit(), 3), MIT(Unit(), 5))
    b = mitrange(qq(2020, 1), qq(2020, 4))
    with pytest.raises(TypeError):
        rangeof_span(a, b)


def test_rangeof_span_empty() -> None:
    out = rangeof_span()
    assert out.is_empty()
