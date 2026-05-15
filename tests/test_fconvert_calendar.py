# SPDX-License-Identifier: MIT
"""Tests for the calendar-frequency fconvert paths.

Mirrors the calendar-relevant blocks from
``TimeSeriesEcon.jl/test/test_fconvert.jl``:

* ``@testset "fconvert, Weekly to lower"``
* ``@testset "fconvert, Weekly to daily"``
* ``@testset "fconvert, Weekly to BDaily"``
* ``@testset "fconvert, Daily to Weekly"``
* ``@testset "fconvert, Daily to Monthly"``
* ``@testset "fconvert, Daily to Quarterly"``
* ``@testset "fconvert, Daily to Yearly"``
* ``@testset "fconvert, BDaily to Monthly"``
* ``@testset "fconvert, BDaily to Weekly"``
* ``@testset "fconvert, BDaily to Daily"``
* ``@testset "fconvert, YPFrequency to Daily and BDaily"``
* ``@testset "fconvert, YPFrequency to Weekly"``

The BDaily-specific kwarg variants (``skip_holidays`` / ``skip_all_nans`` /
``holidays_map``) and the BDaily extend_series branch are still ⬜ (block on
``options.jl``).
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pytest

from tsecon import (
    MIT,
    BDaily,
    Daily,
    HalfYearly,
    MITRange,
    Monthly,
    Quarterly,
    TSeries,
    Weekly,
    Yearly,
    bdaily,
    daily,
    fconvert,
    fconvert_mit,
    fconvert_range,
    mm,
    qq,
    trim_series,
    weekly,
    yy,
)
from tsecon.fconvert import repeat_uneven


def wk(value: int, end_day: int = 7) -> MIT:
    """Build a raw ``MIT{Weekly(end_day)}`` from its integer offset."""
    return MIT(Weekly(end_day), value)


def qq_raw(value: int, end_month: int) -> MIT:
    """Build a non-default-Quarterly MIT from a raw integer (mirrors ``MIT{Quarterly{N}}(v)``)."""
    return MIT(Quarterly(end_month), value)


def yy_raw(value: int, end_month: int) -> MIT:
    """Build a non-default-Yearly MIT from a raw integer."""
    return MIT(Yearly(end_month), value)


def hh(year: int, half: int) -> MIT:
    return MIT.from_yp(HalfYearly(), year, half)


# ---------------------------------------------------------------------------
# Calendar MIT conversions (single-MIT round-trips, BDaily round_to policy)
# ---------------------------------------------------------------------------


class TestCalendarMIT:
    def test_yp_to_daily_end(self):
        assert fconvert_mit(Daily, qq(2022, 1)) == daily("2022-03-31")

    def test_yp_to_daily_begin(self):
        assert fconvert_mit(Daily, qq(2022, 1), ref="begin") == daily("2022-01-01")

    def test_yp_to_weekly_end(self):
        # End of Q1 2022 = 2022-03-31; week ending Sunday containing it = 2022-04-03
        assert fconvert_mit(Weekly, qq(2022, 1)) == weekly("2022-04-03")

    def test_yp_to_bdaily_current_on_business_day(self):
        # Q1 2022 ends on 2022-03-31 (Thursday); current=ok.
        assert fconvert_mit(BDaily, qq(2022, 1)) == bdaily("2022-03-31")

    def test_yp_to_bdaily_round_next(self):
        # Q1 2022 begin = 2022-01-01 (Saturday) → next Monday = 2022-01-03
        assert fconvert_mit(BDaily, qq(2022, 1), ref="begin", round_to="next") == bdaily(
            "2022-01-03"
        )

    def test_yp_to_bdaily_round_previous(self):
        # 2022-01-01 (Saturday) → previous Friday = 2021-12-31
        assert fconvert_mit(BDaily, qq(2022, 1), ref="begin", round_to="previous") == bdaily(
            "2021-12-31"
        )

    def test_yp_to_bdaily_round_current_on_weekend_raises(self):
        with pytest.raises(ValueError, match="business day"):
            fconvert_mit(BDaily, qq(2022, 1), ref="begin", round_to="current")

    def test_yp_to_bdaily_invalid_round_to(self):
        with pytest.raises(ValueError, match="round_to"):
            fconvert_mit(BDaily, qq(2022, 1), round_to="somethingelse")  # type: ignore[arg-type]

    def test_bdaily_to_daily_uses_modulo_formula(self):
        # bdaily ordinal 1 → 0001-01-01 (Monday) → Daily ordinal 1
        assert fconvert_mit(Daily, bdaily("2022-07-06")) == daily("2022-07-06")
        # Friday of week N + 1 → next Monday three days later
        assert fconvert_mit(Daily, bdaily("2022-07-08")) == daily("2022-07-08")
        assert fconvert_mit(Daily, bdaily("2022-07-11")) == daily("2022-07-11")

    def test_daily_to_yearly(self):
        assert fconvert_mit(Yearly, daily("2022-06-15")) == yy(2022)

    def test_daily_to_quarterly(self):
        assert fconvert_mit(Quarterly, daily("2022-06-15")) == qq(2022, 2)

    def test_daily_to_monthly(self):
        assert fconvert_mit(Monthly, daily("2022-06-15")) == mm(2022, 6)


# ---------------------------------------------------------------------------
# Calendar MITRange conversions
# ---------------------------------------------------------------------------


class TestCalendarMITRange:
    def test_yp_to_daily_range(self):
        assert fconvert_range(Daily, mitrange_yp(yy(2022), yy(2022))) == MITRange(
            daily("2022-01-01"), daily("2022-12-31")
        )

    def test_yp_to_bdaily_range(self):
        assert fconvert_range(BDaily, mitrange_yp(yy(2022), yy(2022))) == MITRange(
            bdaily("2022-01-03"), bdaily("2022-12-30")
        )

    def test_daily_to_monthly_trim_both(self):
        rng = MITRange(daily("2022-01-15"), daily("2022-05-15"))
        assert fconvert_range(Monthly, rng) == MITRange(mm(2022, 2), mm(2022, 4))

    def test_daily_to_monthly_trim_begin(self):
        rng = MITRange(daily("2022-01-15"), daily("2022-05-15"))
        assert fconvert_range(Monthly, rng, trim="begin") == MITRange(mm(2022, 2), mm(2022, 5))

    def test_daily_to_monthly_trim_end(self):
        rng = MITRange(daily("2022-01-15"), daily("2022-05-15"))
        assert fconvert_range(Monthly, rng, trim="end") == MITRange(mm(2022, 1), mm(2022, 4))

    def test_yp_to_weekly_range_trim_both(self):
        # Q1 2022 spans 2022-01-01 (Sat) to 2022-03-31 (Thu). The week
        # containing the start ends 2022-01-02 (begin 2021-12-27) which is
        # before Q1 — trim_start. The week containing the end ends
        # 2022-04-03 which is after Q1 — trim_end. So inner range is
        # 2022-01-09 to 2022-03-27.
        rng = MITRange(qq(2022, 1), qq(2022, 1))
        assert fconvert_range(Weekly, rng) == MITRange(weekly("2022-01-09"), weekly("2022-03-27"))

    def test_yp_to_weekly_range_trim_end(self):
        rng = MITRange(qq(2022, 1), qq(2022, 1))
        # With trim="end" only the trailing edge is truncated; head stays.
        out = fconvert_range(Weekly, rng, trim="end")
        assert out.start == weekly("2022-01-02")


def mitrange_yp(start: MIT, stop: MIT) -> MITRange:
    """Helper because ``from tsecon.mitrange import mitrange`` isn't re-exported."""
    return MITRange(start, stop)


# ---------------------------------------------------------------------------
# YP → Daily / BDaily TSeries
# (mirrors Julia @testset "fconvert, YPFrequency to Daily and BDaily")
# ---------------------------------------------------------------------------


class TestYPToDailyBDaily:
    def test_monthly_to_daily_const(self):
        t1 = TSeries(mm(2022, 1), np.arange(1, 13, dtype=np.float64))
        d1 = fconvert(Daily, t1)
        assert d1[daily("2022-01-31")] == 1.0
        assert d1[daily("2022-02-01")] == 2.0
        assert d1[daily("2022-04-01")] == 4.0
        assert len(d1) == 365

    def test_quarterly_to_daily_const(self):
        t2 = TSeries(qq(2022, 1), np.arange(1, 5, dtype=np.float64))
        d2 = fconvert(Daily, t2)
        assert d2[daily("2022-01-31")] == 1.0
        assert d2[daily("2022-02-01")] == 1.0
        assert d2[daily("2022-04-01")] == 2.0

    def test_yearly_to_daily_const(self):
        t3 = TSeries(yy(2022), np.arange(1, 3, dtype=np.float64))
        d3 = fconvert(Daily, t3)
        # First year all 1s, second year all 2s; both are 365-day non-leap years.
        assert (d3.values[:365] == 1.0).all()
        assert (d3.values[365:] == 2.0).all()
        assert len(d3) == 365 * 2

    def test_monthly_to_bdaily_const(self):
        t1 = TSeries(mm(2022, 1), np.arange(1, 13, dtype=np.float64))
        bd1 = fconvert(BDaily, t1)
        assert bd1[bdaily("2022-01-31")] == 1.0
        assert bd1[bdaily("2022-02-01")] == 2.0
        assert bd1[bdaily("2022-04-01")] == 4.0
        assert len(bd1) == 260

    def test_yearly_to_bdaily_const(self):
        t3 = TSeries(yy(2022), np.arange(1, 3, dtype=np.float64))
        bd3 = fconvert(BDaily, t3)
        assert (bd3.values[:260] == 1.0).all()
        assert (bd3.values[260:] == 2.0).all()
        assert len(bd3) == 260 * 2

    def test_monthly_to_daily_linear_end(self):
        t1 = TSeries(mm(2022, 1), np.arange(1, 13, dtype=np.float64))
        d1_lin = fconvert(Daily, t1, method="linear")
        # February: linear from 1 to 2 across 29 days (28 day-stride from Jan to Feb)
        expected = np.linspace(1, 2, 29)[1:]
        actual = d1_lin[MITRange(daily("2022-02-01"), daily("2022-02-28"))].values
        np.testing.assert_allclose(actual, expected)
        # April: 3 → 4 across 31 days
        expected_apr = np.linspace(3, 4, 31)[1:]
        actual_apr = d1_lin[MITRange(daily("2022-04-01"), daily("2022-04-30"))].values
        np.testing.assert_allclose(actual_apr, expected_apr)
        assert len(d1_lin) == 365

    def test_monthly_to_daily_linear_begin(self):
        t1 = TSeries(mm(2022, 1), np.arange(1, 13, dtype=np.float64))
        d1_lin = fconvert(Daily, t1, method="linear", ref="begin")
        # January: linear 1 → 2 across 32 days, take first 31
        expected_jan = np.linspace(1, 2, 32)[:31]
        actual_jan = d1_lin[MITRange(daily("2022-01-01"), daily("2022-01-31"))].values
        np.testing.assert_allclose(actual_jan, expected_jan)

    def test_quarterly_to_daily_linear_end(self):
        t2 = TSeries(qq(2022, 1), np.arange(1, 5, dtype=np.float64))
        d2_lin = fconvert(Daily, t2, method="linear")
        # Q2 (Apr-Jun): linear 1 → 2 across 91 days, drop first
        expected = np.linspace(1, 2, 91 + 1)[1:]
        actual = d2_lin[MITRange(daily("2022-04-01"), daily("2022-06-30"))].values
        np.testing.assert_allclose(actual, expected)
        assert len(d2_lin) == 365

    def test_yearly_to_daily_linear_end(self):
        t3 = TSeries(yy(2022), np.arange(1, 3, dtype=np.float64))
        d3_lin = fconvert(Daily, t3, method="linear")
        # Year 1 (2022, 365 days): the slope is 1 unit per year so day stride is 1/365.
        # linear from t1=1 with stride = (2-1)/365: 365 values increasing from 1 - 1/365 to 1
        # The full 2022 takes linspace(0, 1, 366)[1:366]
        expected_2022 = np.linspace(0, 1, 366)[1:366]
        actual_2022 = d3_lin[MITRange(daily("2022-01-01"), daily("2022-12-31"))].values
        np.testing.assert_allclose(actual_2022, expected_2022)
        assert len(d3_lin) == 365 * 2

    def test_monthly_to_bdaily_linear_end(self):
        t1 = TSeries(mm(2022, 1), np.arange(1, 13, dtype=np.float64))
        bd1_lin = fconvert(BDaily, t1, method="linear")
        # February: linear 1 → 2 across 21 BDays (Mon 1, Tue 1, ..., last Fri 28).
        expected_feb = np.linspace(1, 2, 21)[1:]
        actual_feb = bd1_lin[MITRange(bdaily("2022-02-01"), bdaily("2022-02-28"))].values
        np.testing.assert_allclose(actual_feb, expected_feb)
        assert len(bd1_lin) == 260


# ---------------------------------------------------------------------------
# YP → Weekly TSeries
# (mirrors Julia @testset "fconvert, YPFrequency to Weekly")
# ---------------------------------------------------------------------------


class TestYPToWeekly:
    def test_monthly_to_weekly_first_35(self):
        t1 = TSeries(mm(2022, 1), np.arange(1, 13, dtype=np.float64))
        w1 = fconvert(Weekly, t1)
        expected = [
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            5,
            5,
            5,
            5,
            5,
            6,
            6,
            6,
            6,
            7,
            7,
            7,
            7,
            7,
            8,
            8,
            8,
            8,
        ]
        assert list(w1.values[:35]) == expected
        assert len(w1) == 52
        assert w1.range == MITRange(weekly("2022-01-01"), weekly("2022-12-25"))

    def test_monthly_to_weekly_first_35_year_2017(self):
        t = TSeries(mm(2017, 1), np.arange(1, 13, dtype=np.float64))
        w = fconvert(Weekly, t)
        expected = [
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            4,
            5,
            5,
            5,
            5,
            6,
            6,
            6,
            6,
            7,
            7,
            7,
            7,
            7,
            8,
            8,
            8,
            8,
        ]
        assert list(w.values[:35]) == expected
        assert w.range == MITRange(weekly("2017-01-01"), weekly("2017-12-31"))

    def test_quarterly_to_weekly(self):
        t2 = TSeries(qq(2022, 1), np.arange(1, 5, dtype=np.float64))
        w2 = fconvert(Weekly, t2)
        expected = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
        assert list(w2.values[:20]) == expected
        assert len(w2) == 52
        assert w2.range == MITRange(weekly("2022-01-01"), weekly("2022-12-25"))

    def test_yearly_to_weekly(self):
        t3 = TSeries(yy(2022), np.arange(1, 3, dtype=np.float64))
        w3 = fconvert(Weekly, t3)
        assert (w3.values[:52] == 1.0).all()
        assert (w3.values[52:105] == 2.0).all()
        assert len(w3) == 105
        assert w3.range == MITRange(weekly("2022-01-01"), weekly("2023-12-31"))

    def test_monthly_to_weekly_ref_begin(self):
        t1 = TSeries(mm(2022, 1), np.arange(1, 13, dtype=np.float64))
        w1_beg = fconvert(Weekly, t1, ref="begin")
        expected = [
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            5,
            5,
            5,
            5,
            5,
            6,
            6,
            6,
            6,
            7,
            7,
            7,
            7,
            8,
            8,
            8,
            8,
            8,
            9,
        ]
        assert list(w1_beg.values[:36]) == expected
        assert len(w1_beg) == 52
        assert w1_beg.range == MITRange(weekly("2022-01-08"), weekly("2022-12-31"))

    def test_yearly_to_weekly_ref_begin(self):
        t3 = TSeries(yy(2022), np.arange(1, 3, dtype=np.float64))
        w3_beg = fconvert(Weekly, t3, ref="begin")
        assert (w3_beg.values[:52] == 1.0).all()
        assert (w3_beg.values[52:104] == 2.0).all()
        assert len(w3_beg) == 104
        assert w3_beg.range == MITRange(weekly("2022-01-08"), weekly("2023-12-31"))


# ---------------------------------------------------------------------------
# Weekly → Daily / BDaily
# (mirrors Julia @testset "fconvert, Weekly to daily" / "Weekly to BDaily")
# ---------------------------------------------------------------------------


class TestWeeklyToDailyBDaily:
    def test_weekly_to_daily_const_default_end_day(self):
        t1 = TSeries(wk(1, 7), np.arange(1, 21, dtype=np.float64))
        r1 = fconvert(Daily, t1)
        expected = np.repeat(np.arange(1, 21), 7)
        np.testing.assert_array_equal(r1.values, expected)
        assert len(r1) == 140

    def test_weekly_thursday_to_daily_const(self):
        t2 = TSeries(wk(2, 4), np.arange(1, 21, dtype=np.float64))
        r2 = fconvert(Daily, t2)
        expected = np.repeat(np.arange(1, 21), 7)
        np.testing.assert_array_equal(r2.values, expected)

    def test_weekly_to_bdaily_const(self):
        t1 = TSeries(wk(1, 7), np.arange(1, 21, dtype=np.float64))
        r1 = fconvert(BDaily, t1)
        expected = np.repeat(np.arange(1, 21), 5)
        np.testing.assert_array_equal(r1.values, expected)
        assert len(r1) == 100

    def test_weekly_thursday_to_bdaily_const(self):
        t2 = TSeries(wk(2, 4), np.arange(1, 21, dtype=np.float64))
        r2 = fconvert(BDaily, t2)
        expected = np.repeat(np.arange(1, 21), 5)
        np.testing.assert_array_equal(r2.values, expected)


# ---------------------------------------------------------------------------
# Daily → BDaily / Weekly / Monthly / Quarterly / Yearly
# ---------------------------------------------------------------------------


class TestDailyToBDaily:
    def test_daily_to_bdaily_samples_business_days(self):
        # Daily(1) = 0001-01-01 (Monday) — start week is Mon-Sun.
        t1 = TSeries(daily(_dt.date(1, 1, 1)), np.arange(1, 101, dtype=np.float64))
        r1 = fconvert(BDaily, t1)
        # The first 5 days are Mon..Fri (1-5), then skip Sat-Sun (6-7), Mon-Fri (8-12), etc.
        # Indices kept: 0-4, 7-11, 14-18, ...
        kept_mask = np.tile(
            np.array([True, True, True, True, True, False, False], dtype=bool),
            (len(t1) + 6) // 7,
        )[: len(t1)]
        expected = np.arange(1, 101, dtype=np.float64)[kept_mask]
        np.testing.assert_array_equal(r1.values, expected)


class TestDailyToWeekly:
    def test_daily_to_weekly_mean(self):
        t1 = TSeries(daily(_dt.date(1, 1, 1)), np.arange(1, 101, dtype=np.float64))
        r1 = fconvert(Weekly, t1, method="mean")
        np.testing.assert_array_equal(r1.values, np.arange(4, 96, 7, dtype=np.float64))
        assert r1.range == MITRange(wk(1, 7), wk(14, 7))

    def test_daily_to_weekly_point_end(self):
        t1 = TSeries(daily(_dt.date(1, 1, 1)), np.arange(1, 101, dtype=np.float64))
        r2 = fconvert(Weekly, t1, method="point", ref="end")
        np.testing.assert_array_equal(r2.values, np.arange(7, 99, 7, dtype=np.float64))
        assert r2.range == MITRange(wk(1, 7), wk(14, 7))

    def test_daily_to_weekly_point_begin(self):
        t1 = TSeries(daily(_dt.date(1, 1, 1)), np.arange(1, 101, dtype=np.float64))
        r3 = fconvert(Weekly, t1, method="point", ref="begin")
        np.testing.assert_array_equal(r3.values, np.arange(1, 101, 7, dtype=np.float64))
        assert r3.range == MITRange(wk(1, 7), wk(15, 7))

    def test_daily_to_weekly_sum(self):
        t1 = TSeries(daily(_dt.date(1, 1, 1)), np.arange(1, 101, dtype=np.float64))
        r4 = fconvert(Weekly, t1, method="sum")
        np.testing.assert_array_equal(r4.values, np.arange(28, 666, 49, dtype=np.float64))
        assert r4.range == MITRange(wk(1, 7), wk(14, 7))

    def test_daily_to_weekly_thursday_mean(self):
        t1 = TSeries(daily(_dt.date(1, 1, 1)), np.arange(1, 101, dtype=np.float64))
        r5 = fconvert(Weekly(4), t1, method="mean")
        np.testing.assert_array_equal(r5.values, np.arange(8, 93, 7, dtype=np.float64))


class TestDailyToMonthly:
    def test_daily_to_monthly_mean_starting_jan(self):
        # 100 days starting 0001-01-01: Jan (31), Feb (28), Mar (31), Apr (10)
        t1 = TSeries(daily(_dt.date(1, 1, 1)), np.arange(1, 101, dtype=np.float64))
        r1 = fconvert(Monthly, t1, method="mean")
        np.testing.assert_array_equal(r1.values, np.array([16.0, 45.5, 75.0]))
        assert r1.range == MITRange(mm(1, 1), mm(1, 3))

    def test_daily_to_monthly_point_end(self):
        t1 = TSeries(daily(_dt.date(1, 1, 1)), np.arange(1, 101, dtype=np.float64))
        r2 = fconvert(Monthly, t1, method="point", ref="end")
        np.testing.assert_array_equal(r2.values, np.array([31.0, 59.0, 90.0]))
        assert r2.range == MITRange(mm(1, 1), mm(1, 3))

    def test_daily_to_monthly_point_begin(self):
        t1 = TSeries(daily(_dt.date(1, 1, 1)), np.arange(1, 101, dtype=np.float64))
        r3 = fconvert(Monthly, t1, method="point", ref="begin")
        np.testing.assert_array_equal(r3.values, np.array([1.0, 32.0, 60.0, 91.0]))
        assert r3.range == MITRange(mm(1, 1), mm(1, 4))

    def test_daily_to_monthly_sum(self):
        t1 = TSeries(daily(_dt.date(1, 1, 1)), np.arange(1, 101, dtype=np.float64))
        r4 = fconvert(Monthly, t1, method="sum")
        expected = np.array(
            [
                np.arange(1, 32).sum(),
                np.arange(32, 32 + 28).sum(),
                np.arange(32 + 28, 32 + 28 + 31).sum(),
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(r4.values, expected)
        assert r4.range == MITRange(mm(1, 1), mm(1, 3))


class TestDailyToQuarterly:
    def test_daily_to_quarterly_point_end(self):
        t1 = TSeries(daily(_dt.date(1, 1, 1)), np.arange(1, 401, dtype=np.float64))
        r2 = fconvert(Quarterly, t1, method="point", ref="end")
        np.testing.assert_array_equal(r2.values, np.array([90.0, 181.0, 273.0, 365.0]))

    def test_daily_to_quarterly_point_begin(self):
        t1 = TSeries(daily(_dt.date(1, 1, 1)), np.arange(1, 401, dtype=np.float64))
        r3 = fconvert(Quarterly, t1, method="point", ref="begin")
        np.testing.assert_array_equal(r3.values, np.array([1.0, 91.0, 182.0, 274.0, 366.0]))


class TestDailyToYearly:
    def test_daily_to_yearly_mean(self):
        t1 = TSeries(daily(_dt.date(1, 1, 1)), np.arange(1, 2001, dtype=np.float64))
        r1 = fconvert(Yearly, t1, method="mean")
        # Year 1-3 are 365 days each (1-365, 366-730, 731-1095); year 4 is leap (1096-1461);
        # year 5 (1462-1826).
        expected = np.array(
            [
                np.arange(1, 366).mean(),
                np.arange(1 * 365 + 1, 2 * 365 + 1).mean(),
                np.arange(2 * 365 + 1, 3 * 365 + 1).mean(),
                np.arange(3 * 365 + 1, 4 * 365 + 1 + 1).mean(),
                np.arange(4 * 365 + 1 + 1, 5 * 365 + 1 + 1).mean(),
            ]
        )
        np.testing.assert_array_equal(r1.values, expected)
        assert r1.range == MITRange(yy(1), yy(5))


# ---------------------------------------------------------------------------
# BDaily → Daily TSeries
# (mirrors Julia @testset "fconvert, BDaily to Daily")
# ---------------------------------------------------------------------------


class TestBDailyToDaily:
    def test_bdaily_to_daily_const_ref_end(self):
        t1 = TSeries(bdaily("2022-07-06"), np.arange(1, 14, dtype=np.float64))
        r1 = fconvert(Daily, t1)
        expected = np.array(
            [1, 2, 3, 4, 4, 4, 5, 6, 7, 8, 9, 9, 9, 10, 11, 12, 13], dtype=np.float64
        )
        np.testing.assert_array_equal(r1.values, expected)
        assert r1.range == MITRange(daily("2022-07-06"), daily("2022-07-22"))

    def test_bdaily_to_daily_even_leaves_nans(self):
        t1 = TSeries(bdaily("2022-07-06"), np.arange(1, 14, dtype=np.float64))
        r2 = fconvert(Daily, t1, method="even")
        # NaN gaps for weekends because we never write into them.
        vals = r2.values
        nan_mask = np.isnan(vals)
        np.testing.assert_array_equal(
            nan_mask,
            np.array(
                [
                    False,
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                ]
            ),
        )

    def test_bdaily_to_daily_const_ref_begin(self):
        t1 = TSeries(bdaily("2022-07-06"), np.arange(1, 14, dtype=np.float64))
        r2 = fconvert(Daily, t1, ref="begin")
        expected = np.array(
            [1, 2, 3, 3, 3, 4, 5, 6, 7, 8, 8, 8, 9, 10, 11, 12, 13], dtype=np.float64
        )
        np.testing.assert_array_equal(r2.values, expected)

    def test_bdaily_to_daily_linear(self):
        t1 = TSeries(bdaily("2022-07-06"), np.arange(1, 14, dtype=np.float64))
        r4 = fconvert(Daily, t1, method="linear")
        expected = np.array(
            [1, 2, 3, 3 + 1 / 3, 3 + 2 / 3, 4, 5, 6, 7, 8, 8 + 1 / 3, 8 + 2 / 3, 9, 10, 11, 12, 13],
            dtype=np.float64,
        )
        np.testing.assert_allclose(r4.values, expected)


# ---------------------------------------------------------------------------
# BDaily → Monthly / Weekly TSeries
# ---------------------------------------------------------------------------


class TestBDailyToMonthlyAndWeekly:
    def test_bdaily_to_monthly_first_business_day_of_may(self):
        # First BD in May 2022 (Mon May 2) through late June, 42 BDs.
        t1 = TSeries(bdaily("2022-05-02"), np.arange(1, 43, dtype=np.float64))
        r1 = fconvert(Monthly, t1)
        assert r1.range == MITRange(mm(2022, 5), mm(2022, 5))
        np.testing.assert_array_equal(r1.values, np.array([11.5]))

    def test_bdaily_to_monthly_partial_july_only_empties(self):
        # Early July to late July — no complete month.
        t2 = TSeries(bdaily("2022-07-06"), np.arange(1, 14, dtype=np.float64))
        r2 = fconvert(Monthly, t2)
        assert r2.range.is_empty()
        assert list(r2.values) == []

    def test_bdaily_to_monthly_aug_through_sep(self):
        # Early August (Tuesday) through last business day of September.
        t3 = TSeries(bdaily("2022-08-02"), np.arange(1, 45, dtype=np.float64))
        r3 = fconvert(Monthly, t3)
        assert r3.range == MITRange(mm(2022, 9), mm(2022, 9))
        np.testing.assert_array_equal(r3.values, np.array([33.5]))

    def test_bdaily_to_monthly_nov_through_dec(self):
        # First day in November (Tuesday) to last BD in December (Friday).
        t4 = TSeries(bdaily("2022-11-01"), np.arange(1, 45, dtype=np.float64))
        r4 = fconvert(Monthly, t4)
        assert r4.range == MITRange(mm(2022, 11), mm(2022, 12))
        np.testing.assert_array_equal(r4.values, np.array([11.5, 33.5]))

    def test_bdaily_to_weekly(self):
        # Two whole weeks of business days.
        t1 = TSeries(bdaily("2000-01-03"), np.arange(1, 11, dtype=np.float64))
        r1 = fconvert(Weekly, t1)
        np.testing.assert_array_equal(r1.values, np.array([3.0, 8.0]))
        # Verify range round-trip with MITRange conversion.
        assert r1.range == fconvert(Weekly, t1.range)

    def test_bdaily_to_weekly_alt_end_day_aligned(self):
        # Weekly{5}=Fri, Weekly{6}=Sat, Weekly{7}=Sun all align with the
        # Mon-Fri business-day pattern (each week's BDays exactly cover one
        # of the input's two BDay-quintets). Mirrors Julia's r4/r5/r6 cases.
        t1 = TSeries(bdaily("2000-01-03"), np.arange(1, 11, dtype=np.float64))
        for end_day in (5, 6, 7):
            r = fconvert(Weekly(end_day), t1)
            assert list(r.values) == [3.0, 8.0]

    def test_bdaily_to_weekly_thursday_aligned_input(self):
        # t2 begins on Thursday (2000-01-06), so Weekly{3} (Wed-ending) groups
        # the input into two clean weeks (3 and 8 are the group means).
        t2 = TSeries(bdaily("2000-01-06"), np.arange(1, 11, dtype=np.float64))
        r2 = fconvert(Weekly(3), t2)
        assert list(r2.values) == [3.0, 8.0]


# ---------------------------------------------------------------------------
# Weekly → Monthly / Quarterly / Yearly TSeries
# (a thin port of the most stable Julia cases — the Weekly→Monthly block
# focuses on default-end_day Weekly source and Monthly target with the four
# aggregator methods; the parameter sweep over Weekly{4} mirrors Julia.)
# ---------------------------------------------------------------------------


class TestWeeklyToLower:
    def test_weekly_to_monthly_mean(self):
        t1 = TSeries(wk(1, 7), np.arange(1, 21, dtype=np.float64))
        r5 = fconvert(Monthly, t1, method="mean")
        np.testing.assert_allclose(r5.values, [2.5, 6.5, 10.5, 15.0], atol=0.01)
        assert r5.range == MITRange(mm(1, 1), mm(1, 4))

    def test_weekly_to_monthly_point_end(self):
        t1 = TSeries(wk(1, 7), np.arange(1, 21, dtype=np.float64))
        r7 = fconvert(Monthly, t1, method="point", ref="end")
        np.testing.assert_array_equal(r7.values, [4.0, 8.0, 12.0, 17.0])
        assert r7.range == MITRange(mm(1, 1), mm(1, 4))

    def test_weekly_to_monthly_point_begin(self):
        t1 = TSeries(wk(1, 7), np.arange(1, 21, dtype=np.float64))
        r6 = fconvert(Monthly, t1, method="point", ref="begin")
        np.testing.assert_array_equal(r6.values, [1.0, 5.0, 9.0, 13.0, 18.0])
        assert r6.range == MITRange(mm(1, 1), mm(1, 5))

    def test_weekly_to_monthly_sum(self):
        t1 = TSeries(wk(1, 7), np.arange(1, 21, dtype=np.float64))
        r8 = fconvert(Monthly, t1, method="sum")
        np.testing.assert_array_equal(r8.values, [10.0, 26.0, 42.0, 75.0])
        assert r8.range == MITRange(mm(1, 1), mm(1, 4))

    def test_weekly_thursday_to_monthly_mean(self):
        t2 = TSeries(wk(2, 4), np.arange(1, 21, dtype=np.float64))
        r13 = fconvert(Monthly, t2, method="mean")
        np.testing.assert_allclose(r13.values, [5.5, 10.0, 14.5], atol=0.01)
        assert r13.range == MITRange(mm(1, 2), mm(1, 4))

    def test_weekly_to_quarterly_mean(self):
        t3 = TSeries(wk(1, 7), np.arange(1, 61, dtype=np.float64))
        r1 = fconvert(Quarterly, t3, method="mean")
        np.testing.assert_allclose(r1.values, [6.5, 19.0, 32.43, 46.0], atol=0.1)
        assert r1.range == MITRange(qq(1, 1), qq(1, 4))

    def test_weekly_to_quarterly_sum(self):
        t3 = TSeries(wk(1, 7), np.arange(1, 61, dtype=np.float64))
        r4 = fconvert(Quarterly, t3, method="sum")
        np.testing.assert_allclose(r4.values, [78.0, 247.0, 454.0, 598.0], atol=1.0)
        assert r4.range == MITRange(qq(1, 1), qq(1, 4))

    def test_weekly_to_yearly_mean(self):
        t5 = TSeries(wk(1, 7), np.arange(1, 201, dtype=np.float64))
        r1 = fconvert(Yearly, t5, method="mean")
        np.testing.assert_allclose(r1.values, [26.5, 78.5, 130.5], atol=0.01)
        assert r1.range == MITRange(yy(1), yy(3))

    def test_weekly_to_yearly_sum(self):
        t5 = TSeries(wk(1, 7), np.arange(1, 201, dtype=np.float64))
        r4 = fconvert(Yearly, t5, method="sum")
        np.testing.assert_allclose(r4.values, [1378.0, 4082.0, 6786.0], atol=0.1)
        assert r4.range == MITRange(yy(1), yy(3))

    def test_weekly_to_quarterly_jan_ending(self):
        t3 = TSeries(wk(1, 7), np.arange(1, 61, dtype=np.float64))
        r1 = fconvert(Quarterly(1), t3, method="mean")
        np.testing.assert_allclose(r1.values, [11.0, 24.0, 36.92, 50.0], atol=0.1)
        assert r1.range == MITRange(qq_raw(5, 1), qq_raw(8, 1))


# ---------------------------------------------------------------------------
# Pass-function dispatch over calendar combinations (mirrors Julia
# @testset "fconvert, pass function" for the calendar legs).
# ---------------------------------------------------------------------------


class TestCalendarPassFunction:
    def test_yearly_to_weekly_repeat_uneven_matches_const(self):
        t = TSeries(yy(2022), np.array([10.0, 20.0]))
        a = fconvert(repeat_uneven, Weekly, t, ref="end")
        b = fconvert(Weekly, t, method="const", ref="end")
        assert a.equals(b)

    def test_daily_to_monthly_mean_via_function_matches_named_method(self):
        t = TSeries(daily(_dt.date(1, 1, 1)), np.arange(1, 101, dtype=np.float64))
        a = fconvert(np.mean, Monthly, t)
        b = fconvert(Monthly, t, method="mean")
        # ``np.mean`` returns float64 which matches the named-method aggregator.
        np.testing.assert_allclose(a.values, b.values)


# ---------------------------------------------------------------------------
# YP-target with calendar source — trim_series round-trip should preserve YP slice
# ---------------------------------------------------------------------------


class TestCalendarTrimSeries:
    def test_daily_to_monthly_trim_round_trip(self):
        t = TSeries(daily(_dt.date(1, 1, 1)), np.arange(1, 101, dtype=np.float64))
        trimmed = trim_series(Monthly, t)
        # Inner Daily range after trimming should still be Daily — but only the
        # days that align with complete months.
        assert trimmed.firstdate.frequency == Daily()
