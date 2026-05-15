# SPDX-License-Identifier: MIT
"""Tests for ``tsecon.interop.pandas`` — TSeries / MVTSeries / Workspace ↔ pandas.

pandas is an optional dependency; this whole module is skipped at collection
time when pandas is not installed.
"""

from __future__ import annotations

import importlib.util
from datetime import date

import numpy as np
import pytest

pytest.importorskip("pandas")

import pandas as pd

import tsecon as ts
from tsecon import (
    MIT,
    MITRange,
    MVTSeries,
    TSeries,
    Workspace,
    bdaily,
    daily,
    mm,
    qq,
    weekly,
    yy,
)
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
from tsecon.interop import pandas as _interop_pandas
from tsecon.interop._common import (
    freq_from_pandas_freqstr,
    freq_to_pandas_freqstr,
    mit_from_date,
    supports_pandas_period,
)
from tsecon.interop.pandas import _INSTALL_HINT

# ---------------------------------------------------------------------------
# Frequency ↔ pandas freq-alias mapping
# ---------------------------------------------------------------------------


class TestFreqAliasRoundTrip:
    @pytest.mark.parametrize(
        "freq",
        [
            Yearly(end_month=12),
            Yearly(end_month=3),
            Yearly(end_month=6),
            Quarterly(end_month=1),
            Quarterly(end_month=2),
            Quarterly(end_month=3),
            Monthly(),
            Weekly(end_day=7),
            Weekly(end_day=1),
            Weekly(end_day=4),
            Daily(),
            BDaily(),
        ],
    )
    def test_round_trip(self, freq):
        alias = freq_to_pandas_freqstr(freq)
        recovered = freq_from_pandas_freqstr(alias)
        assert recovered == freq

    def test_legacy_year_alias_accepted(self):
        assert freq_from_pandas_freqstr("A-DEC") == Yearly(end_month=12)

    def test_quarterly_q_dec_matches_calendar(self):
        # tsecon's default Quarterly(end_month=3) puts Jan-Mar 2020 in Q1.
        # That's pandas Q-DEC (fiscal year ends December).
        assert freq_to_pandas_freqstr(Quarterly(end_month=3)) == "Q-DEC"
        # em=1 → Q1 ends Jan → fiscal year ends Oct.
        assert freq_to_pandas_freqstr(Quarterly(end_month=1)) == "Q-OCT"
        # em=2 → Q1 ends Feb → fiscal year ends Nov.
        assert freq_to_pandas_freqstr(Quarterly(end_month=2)) == "Q-NOV"

    def test_unsupported_pandas_quarter_anchor_raises(self):
        # Pandas Q-MAR (fiscal year ending March) implies em=6 in tsecon's
        # extended notation, but our Quarterly only accepts em ∈ {1, 2, 3}
        # (calendar-aligned phases). Confirm we surface a helpful error rather
        # than silently producing a wrong frequency.
        with pytest.raises(ValueError, match="end_month"):
            freq_from_pandas_freqstr("Q-MAR")

    def test_halfyearly_raises(self):
        # pandas has no half-yearly period; the alias is undefined.
        with pytest.raises(TypeError, match="HalfYearly"):
            freq_to_pandas_freqstr(HalfYearly())

    def test_unit_raises(self):
        with pytest.raises(TypeError, match="Unit"):
            freq_to_pandas_freqstr(Unit())

    def test_supports_pandas_period(self):
        assert supports_pandas_period(Monthly()) is True
        assert supports_pandas_period(Quarterly()) is True
        assert supports_pandas_period(Yearly()) is True
        assert supports_pandas_period(Weekly()) is True
        # Daily/BDaily go through DatetimeIndex, not PeriodIndex.
        assert supports_pandas_period(Daily()) is False
        assert supports_pandas_period(BDaily()) is False
        assert supports_pandas_period(HalfYearly()) is False
        assert supports_pandas_period(Unit()) is False


# ---------------------------------------------------------------------------
# mit_from_date — inverse of mit_to_date for every supported frequency
# ---------------------------------------------------------------------------


class TestMitFromDate:
    def test_daily(self):
        d = date(2020, 6, 15)
        assert mit_from_date(d, Daily()) == daily(d)

    def test_bdaily_business_day(self):
        d = date(2020, 6, 15)  # Monday
        assert mit_from_date(d, BDaily()) == bdaily(d)

    def test_bdaily_weekend_snaps_nearest(self):
        sat = date(2020, 6, 13)  # Saturday → nearest is Friday Jun 12
        assert mit_from_date(sat, BDaily()) == bdaily(date(2020, 6, 12))

    def test_weekly(self):
        d = date(2020, 1, 5)  # Sunday
        assert mit_from_date(d, Weekly(end_day=7)) == weekly(d, end_day=7)

    def test_monthly(self):
        # End-of-month date should map to that month.
        m = mit_from_date(date(2020, 3, 31), Monthly())
        assert m == mm(2020, 3)
        # Middle-of-month also.
        assert mit_from_date(date(2020, 3, 15), Monthly()) == mm(2020, 3)

    def test_quarterly_default_endmonth(self):
        assert mit_from_date(date(2020, 1, 1), Quarterly()) == qq(2020, 1)
        assert mit_from_date(date(2020, 3, 31), Quarterly()) == qq(2020, 1)
        assert mit_from_date(date(2020, 4, 1), Quarterly()) == qq(2020, 2)
        assert mit_from_date(date(2020, 12, 31), Quarterly()) == qq(2020, 4)

    def test_quarterly_non_default_endmonth(self):
        # end_month=1: Q1 ends Jan, Q2 ends Apr, Q3 ends Jul, Q4 ends Oct.
        q = Quarterly(end_month=1)
        assert mit_from_date(date(2020, 1, 31), q) == MIT.from_yp(q, 2020, 1)
        assert mit_from_date(date(2020, 4, 30), q) == MIT.from_yp(q, 2020, 2)
        assert mit_from_date(date(2020, 10, 31), q) == MIT.from_yp(q, 2020, 4)
        # em=1 means Q1 ends Jan and Q4 ends Oct of the SAME calendar year as Q1.
        # 2019-12-01 falls between Q4 (ends Oct 2019) and Q1 (ends Jan 2020),
        # i.e. into the period ending Jan 2020 → Q1 of year 2020 (Nov 2019 - Jan 2020).
        assert mit_from_date(date(2019, 12, 1), q) == MIT.from_yp(q, 2020, 1)

    def test_yearly_default(self):
        assert mit_from_date(date(2020, 6, 15), Yearly()) == yy(2020)

    def test_halfyearly(self):
        h = HalfYearly()
        assert mit_from_date(date(2020, 1, 15), h) == MIT.from_yp(h, 2020, 1)
        assert mit_from_date(date(2020, 7, 15), h) == MIT.from_yp(h, 2020, 2)

    def test_unit_raises(self):
        with pytest.raises(TypeError, match="Unit"):
            mit_from_date(date(2020, 1, 1), Unit())


# ---------------------------------------------------------------------------
# to_pandas — output shape per kind
# ---------------------------------------------------------------------------


class TestToPandasTSeries:
    def test_quarterly_periodindex(self):
        t = TSeries(MITRange(qq(2020, 1), qq(2021, 4)), np.arange(8.0))
        s = ts.to_pandas(t)
        assert isinstance(s, pd.Series)
        assert isinstance(s.index, pd.PeriodIndex)
        assert s.index.freqstr == "Q-DEC"
        # Labels match tsecon convention (calendar Q1 = 2020Q1).
        assert str(s.index[0]) == "2020Q1"
        np.testing.assert_array_equal(s.to_numpy(), np.arange(8.0))

    def test_monthly_periodindex(self):
        t = TSeries(MITRange(mm(2020, 1), mm(2020, 6)), np.arange(6.0))
        s = ts.to_pandas(t)
        assert isinstance(s.index, pd.PeriodIndex)
        assert s.index.freqstr == "M"

    def test_yearly_periodindex(self):
        t = TSeries(MITRange(yy(2020), yy(2022)), np.arange(3.0))
        s = ts.to_pandas(t)
        assert isinstance(s.index, pd.PeriodIndex)
        assert s.index.freqstr == "Y-DEC"

    def test_weekly_periodindex(self):
        t = TSeries(MITRange(weekly("2020-01-05"), weekly("2020-01-26")), np.arange(4.0))
        s = ts.to_pandas(t)
        assert isinstance(s.index, pd.PeriodIndex)
        assert s.index.freqstr == "W-SUN"

    def test_daily_datetimeindex(self):
        t = TSeries(MITRange(daily("2020-01-01"), daily("2020-01-05")), np.arange(5.0))
        s = ts.to_pandas(t)
        assert isinstance(s.index, pd.DatetimeIndex)
        assert s.index[0].date() == date(2020, 1, 1)

    def test_bdaily_datetimeindex(self):
        t = TSeries(MITRange(bdaily("2020-01-02"), bdaily("2020-01-08")), np.arange(5.0))
        s = ts.to_pandas(t)
        assert isinstance(s.index, pd.DatetimeIndex)
        # No weekend dates.
        assert all(d.weekday() < 5 for d in s.index)

    def test_unit_rangeindex(self):
        t = TSeries(MITRange(MIT(Unit(), 0), MIT(Unit(), 4)), np.arange(5.0))
        s = ts.to_pandas(t)
        assert isinstance(s.index, pd.RangeIndex)
        assert list(s.index) == [0, 1, 2, 3, 4]

    def test_halfyearly_mit_fallback(self):
        h1 = MIT(HalfYearly(), 2020 * 2 + 0)
        h3 = MIT(HalfYearly(), 2020 * 2 + 2)
        t = TSeries(MITRange(h1, h3), [1.0, 2.0, 3.0])
        s = ts.to_pandas(t)
        # No pandas period analogue; fall back to object index of MIT.
        assert s.index.dtype == object
        assert isinstance(s.index[0], MIT)
        assert s.index[0].frequency == HalfYearly()

    def test_index_mit_always_works(self):
        t = TSeries(MITRange(qq(2020, 1), qq(2020, 4)), np.arange(4.0))
        s = ts.to_pandas(t, index="mit")
        assert s.index.dtype == object
        assert all(isinstance(v, MIT) for v in s.index)

    def test_index_date_emits_datetimeindex(self):
        t = TSeries(MITRange(qq(2020, 1), qq(2020, 4)), np.arange(4.0))
        s = ts.to_pandas(t, index="date")
        assert isinstance(s.index, pd.DatetimeIndex)
        # ref="end" by default → last day of each quarter.
        assert s.index[0].date() == date(2020, 3, 31)

    def test_index_date_begin(self):
        t = TSeries(MITRange(qq(2020, 1), qq(2020, 4)), np.arange(4.0))
        s = ts.to_pandas(t, index="date", date_ref="begin")
        assert s.index[0].date() == date(2020, 1, 1)

    def test_index_invalid_raises(self):
        t = TSeries(MITRange(qq(2020, 1), qq(2020, 4)), np.arange(4.0))
        with pytest.raises(ValueError, match="index must be"):
            ts.to_pandas(t, index="bogus")  # type: ignore[arg-type]

    def test_name_parameter(self):
        t = TSeries(MITRange(qq(2020, 1), qq(2020, 4)), np.arange(4.0))
        s = ts.to_pandas(t, name="gdp")
        assert s.name == "gdp"


class TestToPandasMVTSeries:
    def test_basic_dataframe(self):
        m = MVTSeries(MITRange(qq(2020, 1), qq(2020, 4)), ["a", "b"], np.arange(8.0).reshape(4, 2))
        df = ts.to_pandas(m)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a", "b"]
        assert isinstance(df.index, pd.PeriodIndex)
        np.testing.assert_array_equal(df.to_numpy(), np.arange(8.0).reshape(4, 2))


class TestToPandasWorkspace:
    def test_single_freq(self):
        ty = TSeries(MITRange(yy(2020), yy(2022)), [1.0, 2.0, 3.0])
        w = Workspace(a=ty, b=ty * 2.0)
        df = ts.to_pandas(w)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a", "b"]
        assert isinstance(df.index, pd.PeriodIndex)

    def test_mixed_freq_raises(self):
        ty = TSeries(MITRange(yy(2020), yy(2022)), [1.0, 2.0, 3.0])
        tm = TSeries(MITRange(mm(2020, 1), mm(2020, 3)), [4.0, 5.0, 6.0])
        w = Workspace(a=ty, b=tm)
        with pytest.raises(TypeError, match="single frequency"):
            ts.to_pandas(w)

    def test_empty_workspace_raises(self):
        w = Workspace()
        with pytest.raises(ValueError, match="at least one"):
            ts.to_pandas(w)

    def test_nan_pads_misaligned_members(self):
        # 'a' over 2020..2021; 'b' over 2021..2022. Spanned range is 2020..2022,
        # 'a' should have NaN at 2022 and 'b' should have NaN at 2020.
        ta = TSeries(MITRange(yy(2020), yy(2021)), [1.0, 2.0])
        tb = TSeries(MITRange(yy(2021), yy(2022)), [3.0, 4.0])
        w = Workspace(a=ta, b=tb)
        df = ts.to_pandas(w)
        assert len(df) == 3
        assert np.isnan(df["a"].iloc[-1])
        assert np.isnan(df["b"].iloc[0])
        assert df["a"].iloc[0] == 1.0
        assert df["b"].iloc[-1] == 4.0

    def test_includes_mvtseries_members(self):
        ty = TSeries(MITRange(yy(2020), yy(2022)), [1.0, 2.0, 3.0])
        m = MVTSeries(MITRange(yy(2020), yy(2022)), ["x", "y"], np.arange(6.0).reshape(3, 2))
        w = Workspace(t=ty, m=m)
        df = ts.to_pandas(w)
        # MVTSeries members expand to dotted-name columns.
        assert "m.x" in df.columns
        assert "m.y" in df.columns
        assert "t" in df.columns

    def test_unsupported_type_raises(self):
        # An int is not a TSeries or MVTSeries or container — to_pandas only
        # accepts the three containers.
        with pytest.raises(TypeError, match="to_pandas accepts"):
            ts.to_pandas(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# from_pandas — round-trip and explicit-freq paths
# ---------------------------------------------------------------------------


class TestFromPandasRoundTrip:
    @pytest.mark.parametrize(
        ("rng", "values"),
        [
            (MITRange(qq(2020, 1), qq(2021, 4)), np.arange(8.0)),
            (MITRange(mm(2020, 1), mm(2020, 6)), np.arange(6.0)),
            (MITRange(yy(2020), yy(2024)), np.arange(5.0)),
            (MITRange(weekly("2020-01-05"), weekly("2020-01-26")), np.arange(4.0)),
        ],
    )
    def test_period_round_trip(self, rng, values):
        t = TSeries(rng, values)
        t2 = ts.from_pandas(ts.to_pandas(t))
        assert t2.range == t.range
        np.testing.assert_array_equal(t2.values, t.values)

    def test_daily_round_trip_requires_freq(self):
        t = TSeries(MITRange(daily("2020-01-01"), daily("2020-01-05")), np.arange(5.0))
        s = ts.to_pandas(t)
        # DatetimeIndex without freq= must raise.
        with pytest.raises(ValueError, match="DatetimeIndex"):
            ts.from_pandas(s)
        t2 = ts.from_pandas(s, freq=Daily())
        assert t2.range == t.range

    def test_bdaily_round_trip(self):
        t = TSeries(MITRange(bdaily("2020-01-02"), bdaily("2020-01-08")), np.arange(5.0))
        t2 = ts.from_pandas(ts.to_pandas(t), freq=BDaily())
        assert t2.range == t.range

    def test_unit_round_trip(self):
        t = TSeries(MITRange(MIT(Unit(), 0), MIT(Unit(), 4)), np.arange(5.0))
        t2 = ts.from_pandas(ts.to_pandas(t))
        assert t2.range == t.range

    def test_halfyearly_round_trip_via_mit_index(self):
        h0 = MIT(HalfYearly(), 2020 * 2 + 0)
        h2 = MIT(HalfYearly(), 2020 * 2 + 2)
        t = TSeries(MITRange(h0, h2), [1.0, 2.0, 3.0])
        t2 = ts.from_pandas(ts.to_pandas(t))
        assert t2.range == t.range

    def test_mvtseries_round_trip(self):
        m = MVTSeries(MITRange(qq(2020, 1), qq(2020, 4)), ["a", "b"], np.arange(8.0).reshape(4, 2))
        m2 = ts.from_pandas(ts.to_pandas(m))
        assert isinstance(m2, MVTSeries)
        assert m2.column_names == m.column_names
        np.testing.assert_array_equal(m2.values, m.values)


class TestFromPandasToWorkspace:
    """Round-trip Workspace through pandas with ``to_workspace=True``."""

    def test_workspace_to_workspace_round_trip_via_dataframe(self):
        # Build a Workspace, send it through to_pandas, recover it.
        ty_a = TSeries(MITRange(yy(2020), yy(2023)), [1.0, 2.0, 3.0, 4.0])
        ty_b = TSeries(MITRange(yy(2020), yy(2023)), [10.0, 20.0, 30.0, 40.0])
        w = Workspace(a=ty_a, b=ty_b)
        df = ts.to_pandas(w)
        w2 = ts.from_pandas(df, to_workspace=True)
        assert isinstance(w2, Workspace)
        assert list(w2.keys()) == ["a", "b"]
        assert w2.a.range == ty_a.range
        np.testing.assert_array_equal(w2.a.values, ty_a.values)
        np.testing.assert_array_equal(w2.b.values, ty_b.values)

    def test_dataframe_to_workspace_to_dataframe_round_trip(self):
        df = pd.DataFrame(
            {
                "gdp": [1.0, 2.0, 3.0, 4.0],
                "cpi": [10.0, 11.0, 12.0, 13.0],
            },
            index=pd.PeriodIndex(
                ["2020Q1", "2020Q2", "2020Q3", "2020Q4"], freq="Q-DEC", name="period"
            ),
        )
        w = ts.from_pandas(df, to_workspace=True)
        assert isinstance(w, Workspace)
        df2 = ts.to_pandas(w)
        # Columns and values match; index round-trips through PeriodIndex.
        assert list(df2.columns) == list(df.columns)
        np.testing.assert_array_equal(df2.to_numpy(), df.to_numpy())
        assert df2.index.equals(df.index)

    def test_quarterly_workspace_round_trip(self):
        rng = MITRange(qq(2020, 1), qq(2021, 4))
        w = Workspace(x=TSeries(rng, np.arange(8.0)), y=TSeries(rng, np.arange(8.0) * 2))
        df = ts.to_pandas(w)
        w2 = ts.from_pandas(df, to_workspace=True)
        assert isinstance(w2, Workspace)
        assert w2.x.range == rng
        np.testing.assert_array_equal(w2.x.values, w.x.values)
        np.testing.assert_array_equal(w2.y.values, w.y.values)

    def test_daily_workspace_round_trip(self):
        rng = MITRange(daily("2020-01-01"), daily("2020-01-05"))
        w = Workspace(
            a=TSeries(rng, np.arange(5.0)),
            b=TSeries(rng, np.arange(5.0) + 10),
        )
        df = ts.to_pandas(w)
        # Daily emits a DatetimeIndex, so we need freq= on the way back.
        w2 = ts.from_pandas(df, to_workspace=True, freq=Daily())
        assert isinstance(w2, Workspace)
        assert w2.a.range == rng
        np.testing.assert_array_equal(w2.a.values, w.a.values)

    def test_to_workspace_rejects_series(self):
        s = pd.Series([1.0, 2.0], index=pd.PeriodIndex(["2020", "2021"], freq="Y-DEC"))
        with pytest.raises(TypeError, match="requires a DataFrame"):
            ts.from_pandas(s, to_workspace=True)

    def test_to_workspace_preserves_column_order(self):
        # Decision 05 mentions order matters. Verify with an out-of-alphabetical order.
        df = pd.DataFrame(
            {"z": [1.0, 2.0], "a": [3.0, 4.0], "m": [5.0, 6.0]},
            index=pd.PeriodIndex(["2020", "2021"], freq="Y-DEC"),
        )
        w = ts.from_pandas(df, to_workspace=True)
        assert list(w.keys()) == ["z", "a", "m"]


class TestFromPandasIndexHandling:
    def test_mit_object_index(self):
        idx = pd.Index([qq(2020, 1), qq(2020, 2), qq(2020, 3)], dtype=object)
        s = pd.Series([1.0, 2.0, 3.0], index=idx)
        t = ts.from_pandas(s)
        assert t.frequency == Quarterly()
        assert t.range.start == qq(2020, 1)

    def test_mit_object_index_mixed_freq_raises(self):
        idx = pd.Index([qq(2020, 1), mm(2020, 2)], dtype=object)
        s = pd.Series([1.0, 2.0], index=idx)
        with pytest.raises(TypeError, match="Mixed frequencies"):
            ts.from_pandas(s)

    def test_object_index_with_non_mit_raises(self):
        s = pd.Series([1.0, 2.0], index=pd.Index(["a", "b"], dtype=object))
        with pytest.raises(TypeError, match="non-MIT"):
            ts.from_pandas(s)

    def test_unknown_index_type_raises(self):
        s = pd.Series([1.0, 2.0], index=pd.Index([1.5, 2.5]))
        with pytest.raises(TypeError, match="does not know how to interpret"):
            ts.from_pandas(s)

    def test_integer_index_requires_unit_or_default(self):
        s = pd.Series([1.0, 2.0, 3.0], index=pd.Index([10, 11, 12]))
        t = ts.from_pandas(s)
        assert isinstance(t.frequency, Unit)
        assert t.firstdate.value == 10

    def test_integer_index_rejects_nonunit_freq(self):
        s = pd.Series([1.0, 2.0], index=pd.Index([10, 11]))
        with pytest.raises(ValueError, match="Unit"):
            ts.from_pandas(s, freq=Quarterly())


class TestFromPandasTimeColumn:
    def test_explicit_time_column(self):
        df = pd.DataFrame(
            {
                "t": pd.PeriodIndex(["2020Q1", "2020Q2", "2020Q3"], freq="Q-DEC"),
                "a": [1.0, 2.0, 3.0],
                "b": [4.0, 5.0, 6.0],
            }
        )
        m = ts.from_pandas(df, time_col="t")
        assert isinstance(m, MVTSeries)
        assert m.column_names == ("a", "b")
        assert m.range.start == qq(2020, 1)

    def test_missing_time_column_raises(self):
        df = pd.DataFrame({"a": [1.0]})
        with pytest.raises(KeyError, match="t"):
            ts.from_pandas(df, time_col="t")

    def test_datetime_time_column_needs_freq(self):
        df = pd.DataFrame(
            {
                "t": [date(2020, 3, 31), date(2020, 6, 30)],
                "a": [1.0, 2.0],
            }
        )
        df["t"] = pd.to_datetime(df["t"])
        with pytest.raises(ValueError, match="datetime column"):
            ts.from_pandas(df, time_col="t")
        m = ts.from_pandas(df, time_col="t", freq=Quarterly())
        assert m.range.start == qq(2020, 1)


class TestFromPandasLongFormat:
    def test_long_to_mvtseries(self):
        df = pd.DataFrame(
            {
                "date": pd.PeriodIndex(["2020Q1", "2020Q2", "2020Q1", "2020Q2"], freq="Q-DEC"),
                "var": ["a", "a", "b", "b"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        m = ts.from_pandas(df, wide=False, time_col="date", name_col="var", value_col="value")
        assert isinstance(m, MVTSeries)
        assert m.column_names == ("a", "b")
        np.testing.assert_array_equal(m.values, [[1.0, 3.0], [2.0, 4.0]])

    def test_long_requires_three_cols(self):
        df = pd.DataFrame({"date": [], "var": [], "value": []})
        with pytest.raises(ValueError, match="requires explicit"):
            ts.from_pandas(df, wide=False)


class TestFromPandasGapDetection:
    def test_gap_in_period_index_raises(self):
        idx = pd.PeriodIndex(["2020Q1", "2020Q2", "2020Q4"], freq="Q-DEC")
        s = pd.Series([1.0, 2.0, 3.0], index=idx)
        with pytest.raises(ValueError, match="contiguous"):
            ts.from_pandas(s)

    def test_empty_series_raises(self):
        s = pd.Series([], dtype=float, index=pd.PeriodIndex([], freq="Q-DEC"))
        with pytest.raises(ValueError, match="empty"):
            ts.from_pandas(s)


# ---------------------------------------------------------------------------
# Method bindings
# ---------------------------------------------------------------------------


class TestMethodBindings:
    def test_tseries_to_pandas(self):
        t = TSeries(MITRange(qq(2020, 1), qq(2020, 4)), np.arange(4.0))
        s = t.to_pandas()
        assert isinstance(s, pd.Series)

    def test_mvtseries_to_pandas(self):
        m = MVTSeries(MITRange(qq(2020, 1), qq(2020, 4)), ["a", "b"], np.arange(8.0).reshape(4, 2))
        df = m.to_pandas()
        assert isinstance(df, pd.DataFrame)

    def test_workspace_to_pandas(self):
        t = TSeries(MITRange(qq(2020, 1), qq(2020, 4)), np.arange(4.0))
        w = Workspace(a=t)
        df = w.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["a"]


# ---------------------------------------------------------------------------
# Missing-pandas behaviour (simulated)
# ---------------------------------------------------------------------------


class TestPandasMissing:
    def test_install_hint(self, monkeypatch):
        # Simulate "pandas not installed" by monkeypatching find_spec.
        def _fake_find_spec(name):
            return None if name == "pandas" else importlib.util.find_spec(name)

        monkeypatch.setattr(_interop_pandas, "find_spec", _fake_find_spec)
        t = TSeries(MITRange(qq(2020, 1), qq(2020, 4)), np.arange(4.0))
        with pytest.raises(ImportError, match="requires pandas"):
            _interop_pandas.to_pandas(t)

    def test_install_hint_text(self):
        # Sanity check on the install-hint string that's surfaced to users.
        assert "pip install" in _INSTALL_HINT
        assert "TimeSeriesEconPy[pandas]" in _INSTALL_HINT


# ---------------------------------------------------------------------------
# Type-error guards
# ---------------------------------------------------------------------------


class TestFromPandasTypeError:
    def test_wrong_type_raises(self):
        with pytest.raises(TypeError, match="from_pandas accepts"):
            ts.from_pandas([1, 2, 3])  # type: ignore[arg-type]
