# SPDX-License-Identifier: MIT
"""Tests for MIT and Duration, ported from ``TimeSeriesEcon.jl/test/test_mit.jl``."""

from __future__ import annotations

import datetime as _dt

import pytest

from tsecon.frequencies import (
    BDaily,
    Daily,
    HalfYearly,
    Monthly,
    Quarterly,
    Unit,
    Weekly,
    Yearly,
)
from tsecon.mit import (
    MIT,
    Duration,
    bdaily,
    daily,
    frequency_of,
    mit2yp,
    mit_to_date,
    mm,
    period,
    qq,
    weekly,
    weekly_from_iso,
    year,
    yy,
)

# ---------------------------------------------------------------------------
# mit2yp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (5, (1, 2)),
        (4, (1, 1)),
        (3, (0, 4)),
        (2, (0, 3)),
        (1, (0, 2)),
        (0, (0, 1)),
        (-1, (-1, 4)),
        (-2, (-1, 3)),
        (-3, (-1, 2)),
        (-4, (-1, 1)),
        (-5, (-2, 4)),
        (-6, (-2, 3)),
    ],
)
def test_mit2yp_quarterly(value: int, expected: tuple[int, int]) -> None:
    assert mit2yp(MIT(Quarterly(), value)) == expected


def test_mit2yp_monthly() -> None:
    assert mit2yp(mm(2020, 1)) == (2020, 1)
    assert mit2yp(mm(2020, 12)) == (2020, 12)


def test_mit2yp_yearly() -> None:
    assert mit2yp(yy(2020)) == (2020, 1)


# ---------------------------------------------------------------------------
# year / period
# ---------------------------------------------------------------------------


def test_year_period_yp() -> None:
    val = qq(2020, 2)
    assert year(val) == 2020
    assert period(val) == 2
    assert mm(2020, 12).frequency == Monthly()
    assert year(mm(2020, 12)) == 2020
    assert period(mm(2020, 12)) == 12


def test_year_period_unit_raises() -> None:
    with pytest.raises(TypeError):
        year(MIT(Unit(), 1))


# ---------------------------------------------------------------------------
# Subtraction
# ---------------------------------------------------------------------------


def test_subtraction_returns_duration() -> None:
    diff = qq(2020, 1) - qq(2019, 2)
    assert isinstance(diff, Duration)
    assert diff.frequency == Quarterly()
    assert int(diff) == 3


def test_mit_minus_int_returns_mit() -> None:
    out = qq(2020, 1) - 2
    assert isinstance(out, MIT)
    assert out.frequency == Quarterly()
    assert int(out) == int(qq(2020, 1)) - 2


def test_mit_minus_duration_returns_mit() -> None:
    out = qq(2020, 1) - Duration(Quarterly(), 2)
    assert isinstance(out, MIT)
    assert out == qq(2019, 3)


def test_duration_minus_int_returns_duration() -> None:
    out = Duration(Quarterly(), 5) - 2
    assert isinstance(out, Duration)
    assert int(out) == 3


def test_duration_minus_duration() -> None:
    out = Duration(Quarterly(), 5) - Duration(Quarterly(), 2)
    assert isinstance(out, Duration)
    assert int(out) == 3


def test_mixed_freq_subtraction_raises() -> None:
    with pytest.raises(TypeError):
        qq(2020, 1) - mm(2019, 2)
    with pytest.raises(TypeError):
        qq(2020, 1) - Duration(Monthly(), 5)
    with pytest.raises(TypeError):
        Duration(Quarterly(), 8) - Duration(Monthly(), 5)


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------


def test_equality() -> None:
    assert qq(2020, 1) == qq(2020, 1)
    assert qq(2020, 1) != qq(2020, 2)
    assert qq(2020, 1) != mm(2020, 1)


def test_mit_vs_duration_not_equal() -> None:
    """MIT and Duration are never equal, even with the same value/frequency."""
    assert MIT(Quarterly(), 5) != Duration(Quarterly(), 5)


def test_durations_different_freqs_not_equal() -> None:
    assert Duration(Quarterly(), 5) != Duration(Monthly(), 5)


def test_zero_mits_different_freqs_not_equal() -> None:
    """Unlike Julia, ``MIT == int`` is not defined in Python.

    Two zero-value MITs with different frequencies are unequal because
    frequency is part of identity.
    """
    assert qq(0, 1) != mm(0, 1)
    assert int(qq(0, 1)) == 0
    assert int(mm(0, 1)) == 0


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_ordering_same_freq() -> None:
    assert qq(2000, 1) < qq(2000, 2)
    assert qq(2000, 1) <= qq(2000, 1)
    assert qq(2000, 2) > qq(2000, 1)
    assert qq(2000, 1) >= qq(2000, 1)


def test_ordering_mixed_freq_raises() -> None:
    with pytest.raises(TypeError):
        _ = qq(2000, 1) < mm(2000, 1)
    with pytest.raises(TypeError):
        _ = qq(0, 1) <= mm(0, 1)


def test_duration_int_ordering() -> None:
    assert Duration(Quarterly(), 5) < Duration(Quarterly(), 6)
    assert not Duration(Quarterly(), 5) < Duration(Quarterly(), 5)
    assert Duration(Quarterly(), 5) <= Duration(Quarterly(), 5)


def test_mit_vs_duration_ordering_raises() -> None:
    with pytest.raises(TypeError):
        _ = MIT(Quarterly(), 5) < Duration(Quarterly(), 5)
    with pytest.raises(TypeError):
        _ = Duration(Quarterly(), 5) < MIT(Quarterly(), 5)


# ---------------------------------------------------------------------------
# Addition
# ---------------------------------------------------------------------------


def test_addition_int() -> None:
    assert qq(2020, 1) + 4 == qq(2021, 1)
    assert 4 + qq(2020, 1) == qq(2021, 1)


def test_addition_yearly() -> None:
    assert yy(2001) + 5 == yy(2006)


def test_addition_crosses_year() -> None:
    assert 6 + qq(2001, 3) == qq(2003, 1)
    assert qq(2003, 1) - qq(2001, 3) == Duration(Quarterly(), 6)


def test_addition_two_mits_raises() -> None:
    with pytest.raises(TypeError):
        qq(2020, 1) + qq(1, 1)
    with pytest.raises(TypeError):
        qq(2020, 1) + mm(1, 1)


def test_mit_plus_duration() -> None:
    assert qq(2020, 1) + Duration(Quarterly(), 4) == qq(2021, 1)
    with pytest.raises(TypeError):
        qq(2020, 1) + Duration(Monthly(), 2)


def test_int_minus_mit_raises() -> None:
    with pytest.raises(TypeError):
        _ = 6 - qq(2003, 1)


def test_duration_plus_duration_same_freq() -> None:
    out = Duration(Quarterly(), 5) + Duration(Quarterly(), 2)
    assert isinstance(out, Duration)
    assert int(out) == 7


def test_duration_plus_duration_mixed_raises() -> None:
    with pytest.raises(TypeError):
        Duration(Quarterly(), 5) + Duration(Monthly(), 2)


# ---------------------------------------------------------------------------
# Duration div / mod
# ---------------------------------------------------------------------------


def test_duration_div_mod() -> None:
    d1 = Duration(Quarterly(), 10)
    d2 = Duration(Quarterly(), 4)
    assert int(d1 - d2) == 6
    assert int(d1 // d2) == 2
    assert int(d1 % d2) == 2
    assert int(d2 // d1) == 0


def test_duration_div_mod_mixed_raises() -> None:
    d_q = Duration(Quarterly(), 4)
    d_m = Duration(Monthly(), 4)
    with pytest.raises(TypeError):
        d_q // d_m
    with pytest.raises(TypeError):
        d_q % d_m


# ---------------------------------------------------------------------------
# Float conversion (for plotting)
# ---------------------------------------------------------------------------


def test_float_yp() -> None:
    assert float(qq(2000, 1)) == 2000.0
    assert float(qq(2000, 2)) == pytest.approx(2000.25)


def test_float_unit() -> None:
    assert float(MIT(Unit(), 5)) == 5.0


def test_mit_plus_float() -> None:
    assert qq(2000, 1) + 1 == qq(2000, 2)
    assert qq(2000, 1) + 1.0 == pytest.approx(2001.0)
    assert qq(2000, 1) + 1.2 == pytest.approx(2001.2)
    assert 1.2 + MIT(Unit(), 5) == pytest.approx(6.2)
    assert MIT(Unit(), 5) + 1.2 == pytest.approx(6.2)


# ---------------------------------------------------------------------------
# Hash / dict keys
# ---------------------------------------------------------------------------


def test_mit_hashable_distinct_keys() -> None:
    zq = MIT(Quarterly(), 0)
    zm = MIT(Monthly(), 0)
    zy = MIT(Yearly(), 0)
    d = {zq: 4, zm: 12, zy: 1}
    assert len(d) == 3
    assert d[zq] == 4
    assert d[zm] == 12
    assert d[zy] == 1


# ---------------------------------------------------------------------------
# frequency_of
# ---------------------------------------------------------------------------


def test_frequency_of() -> None:
    assert frequency_of(qq(2000, 1)) == Quarterly()
    assert frequency_of(mm(2000, 1)) == Monthly()
    assert frequency_of(yy(2000)) == Yearly()
    assert frequency_of(MIT(Unit(), 1)) == Unit()
    assert frequency_of(qq(2000, 1) - qq(2000, 1)) == Quarterly()
    with pytest.raises(TypeError):
        frequency_of(1)


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def test_qq_mm_yy() -> None:
    assert mm(2020, 1) == MIT(Monthly(), 2020 * 12)
    assert qq(2020, 1) == MIT(Quarterly(), 2020 * 4)
    assert yy(2020) == MIT(Yearly(), 2020)


def test_qq_singleton() -> None:
    """1999 Q1..Q4, 1988 M1..M12 — all identifier-cached."""
    assert qq(1999, 1) == qq(1999, 1)
    assert qq(1999, 1) != qq(1999, 2)
    for m in range(1, 13):
        assert mm(1988, m) == MIT(Monthly(), 1988 * 12 + m - 1)


def test_constructor_period_validation() -> None:
    with pytest.raises(ValueError):
        qq(2020, 0)
    with pytest.raises(ValueError):
        qq(2020, 5)
    with pytest.raises(ValueError):
        mm(2020, 13)


# ---------------------------------------------------------------------------
# Daily / BDaily
# ---------------------------------------------------------------------------


def test_daily_construction() -> None:
    d1 = daily(_dt.date(2022, 1, 1))
    d2 = daily("2022-01-01")
    assert d1 == d2
    assert d1.frequency == Daily()
    assert mit_to_date(d1) == _dt.date(2022, 1, 1)


def test_bdaily_construction_strict_weekday() -> None:
    bd = bdaily("2022-01-03")  # Monday
    assert bd.frequency == BDaily()
    assert mit_to_date(bd) == _dt.date(2022, 1, 3)


def test_bdaily_strict_on_weekend_raises() -> None:
    with pytest.raises(ValueError):
        bdaily("2022-01-01")  # Saturday
    with pytest.raises(ValueError):
        bdaily("2022-01-02")  # Sunday


def test_bdaily_bias_previous_and_next() -> None:
    assert bdaily("2022-01-02", bias="previous") == bdaily("2021-12-31")
    assert bdaily("2022-01-02", bias="next") == bdaily("2022-01-03")
    assert bdaily("2022-01-01", bias="nearest") == bdaily("2021-12-31")  # Sat -> Fri
    assert bdaily("2022-01-02", bias="nearest") == bdaily("2022-01-03")  # Sun -> Mon


def test_bdaily_invalid_bias() -> None:
    with pytest.raises(ValueError):
        bdaily("2022-01-03", bias="bogus")


def test_bdaily_yp() -> None:
    bd = bdaily("2022-01-03")
    assert mit2yp(bd) == (2022, 1)
    bd2 = bdaily("2022-01-04")
    assert mit2yp(bd2) == (2022, 2)


# ---------------------------------------------------------------------------
# Weekly
# ---------------------------------------------------------------------------


def test_weekly_default_end_day_sunday() -> None:
    w = weekly("2022-01-01")
    assert w.frequency == Weekly()
    assert mit_to_date(w) == _dt.date(2022, 1, 2)  # Sunday


def test_weekly_custom_end_day() -> None:
    w = weekly("2022-01-01", end_day=6)
    assert w.frequency == Weekly(end_day=6)
    assert mit_to_date(w) == _dt.date(2022, 1, 1)


def test_weekly_from_iso() -> None:
    w = weekly_from_iso(2021, 52)
    assert mit_to_date(w) == _dt.date(2022, 1, 2)
    w2 = weekly_from_iso(2022, 1)
    assert mit_to_date(w2) == _dt.date(2022, 1, 9)


def test_weekly_from_iso_invalid_week() -> None:
    with pytest.raises(ValueError):
        weekly_from_iso(2022, 54)
    with pytest.raises(ValueError):
        weekly_from_iso(2022, 0)


# ---------------------------------------------------------------------------
# MIT -> date conversion
# ---------------------------------------------------------------------------


def test_mit_to_date_yearly() -> None:
    assert mit_to_date(yy(2022)) == _dt.date(2022, 12, 31)
    assert mit_to_date(yy(2022), ref="begin") == _dt.date(2022, 1, 1)


def test_mit_to_date_quarterly() -> None:
    assert mit_to_date(qq(2022, 1)) == _dt.date(2022, 3, 31)
    assert mit_to_date(qq(2022, 1), ref="begin") == _dt.date(2022, 1, 1)


def test_mit_to_date_halfyearly() -> None:
    h1 = MIT.from_yp(HalfYearly(), 2022, 1)
    assert mit_to_date(h1) == _dt.date(2022, 6, 30)
    assert mit_to_date(h1, ref="begin") == _dt.date(2022, 1, 1)


def test_mit_to_date_monthly() -> None:
    assert mit_to_date(mm(2022, 1)) == _dt.date(2022, 1, 31)
    assert mit_to_date(mm(2022, 1), ref="begin") == _dt.date(2022, 1, 1)


def test_mit_to_date_weekly() -> None:
    assert mit_to_date(weekly("2022-01-01")) == _dt.date(2022, 1, 2)
    assert mit_to_date(weekly("2022-01-01"), ref="begin") == _dt.date(2021, 12, 27)


def test_mit_to_date_daily_bdaily() -> None:
    assert mit_to_date(daily("2022-01-01")) == _dt.date(2022, 1, 1)
    assert mit_to_date(bdaily("2022-01-03")) == _dt.date(2022, 1, 3)


# ---------------------------------------------------------------------------
# Show / repr
# ---------------------------------------------------------------------------


def test_repr_yp() -> None:
    assert repr(qq(2022, 1)) == "2022Q1"
    assert repr(mm(2020, 12)) == "2020M12"
    assert repr(yy(2022)) == "2022Y"


def test_repr_yp_nondefault_endperiod() -> None:
    q = MIT.from_yp(Quarterly(end_month=2), 2022, 1)
    assert repr(q) == "2022Q1{2}"


def test_repr_unit() -> None:
    assert repr(MIT(Unit(), 5)) == "5U"


def test_repr_daily() -> None:
    assert repr(daily("2022-01-01")) == "2022-01-01"


def test_duration_repr() -> None:
    assert repr(Duration(Quarterly(), 5)) == "5"
