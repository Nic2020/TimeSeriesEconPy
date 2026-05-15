# SPDX-License-Identifier: MIT
"""Tests for ``tsecon.interop.polars`` — TSeries / MVTSeries / Workspace ↔ polars.

polars is an optional dependency; this whole module is skipped at collection
time when polars is not installed.
"""

from __future__ import annotations

import importlib.util
from datetime import date

import numpy as np
import pytest

pytest.importorskip("polars")

import polars as pl

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
from tsecon.interop import polars as _interop_polars
from tsecon.interop.polars import _INSTALL_HINT

# ---------------------------------------------------------------------------
# to_polars — shape and dtypes
# ---------------------------------------------------------------------------


class TestToPolarsTSeries:
    def test_quarterly_date_column(self):
        t = TSeries(MITRange(qq(2020, 1), qq(2021, 4)), np.arange(8.0))
        df = ts.to_polars(t)
        assert isinstance(df, pl.DataFrame)
        assert df.columns == ["time", "value"]
        assert df.schema["time"] == pl.Date
        # End-of-period dates by default.
        assert df["time"][0] == date(2020, 3, 31)

    def test_date_ref_begin(self):
        t = TSeries(MITRange(qq(2020, 1), qq(2020, 4)), np.arange(4.0))
        df = ts.to_polars(t, date_ref="begin")
        assert df["time"][0] == date(2020, 1, 1)

    def test_daily_date_column(self):
        t = TSeries(MITRange(daily("2020-01-01"), daily("2020-01-05")), np.arange(5.0))
        df = ts.to_polars(t)
        assert df.schema["time"] == pl.Date
        assert df["time"][0] == date(2020, 1, 1)

    def test_bdaily_skips_weekends(self):
        t = TSeries(MITRange(bdaily("2020-01-02"), bdaily("2020-01-08")), np.arange(5.0))
        df = ts.to_polars(t)
        # Every date is a weekday.
        weekdays = [d.weekday() for d in df["time"]]
        assert all(w < 5 for w in weekdays)

    def test_unit_int_column(self):
        t = TSeries(MITRange(MIT(Unit(), 10), MIT(Unit(), 12)), [1.0, 2.0, 3.0])
        df = ts.to_polars(t)
        assert df.schema["time"] == pl.Int64
        assert df["time"].to_list() == [10, 11, 12]

    def test_custom_time_col_and_value_col(self):
        t = TSeries(MITRange(qq(2020, 1), qq(2020, 4)), np.arange(4.0))
        df = ts.to_polars(t, time_col="when", value_col="gdp")
        assert df.columns == ["when", "gdp"]


class TestToPolarsMVTSeries:
    def test_basic(self):
        m = MVTSeries(MITRange(qq(2020, 1), qq(2020, 4)), ["a", "b"], np.arange(8.0).reshape(4, 2))
        df = ts.to_polars(m)
        assert df.columns == ["time", "a", "b"]
        assert df.shape == (4, 3)

    def test_column_name_collision_raises(self):
        m = MVTSeries(
            MITRange(qq(2020, 1), qq(2020, 4)), ["time", "b"], np.arange(8.0).reshape(4, 2)
        )
        with pytest.raises(ValueError, match="collides"):
            ts.to_polars(m)


class TestToPolarsWorkspace:
    def test_single_freq(self):
        ty = TSeries(MITRange(yy(2020), yy(2022)), [1.0, 2.0, 3.0])
        w = Workspace(a=ty, b=ty * 2.0)
        df = ts.to_polars(w)
        assert df.columns == ["time", "a", "b"]

    def test_mixed_freq_raises(self):
        ty = TSeries(MITRange(yy(2020), yy(2022)), [1.0, 2.0, 3.0])
        tm = TSeries(MITRange(mm(2020, 1), mm(2020, 3)), [4.0, 5.0, 6.0])
        w = Workspace(a=ty, b=tm)
        with pytest.raises(TypeError, match="single frequency"):
            ts.to_polars(w)

    def test_nan_pads_misaligned(self):
        ta = TSeries(MITRange(yy(2020), yy(2021)), [1.0, 2.0])
        tb = TSeries(MITRange(yy(2021), yy(2022)), [3.0, 4.0])
        w = Workspace(a=ta, b=tb)
        df = ts.to_polars(w)
        assert df["a"][-1] is None or np.isnan(df["a"][-1])
        assert df["b"][0] is None or np.isnan(df["b"][0])

    def test_empty_raises(self):
        w = Workspace()
        with pytest.raises(ValueError, match="at least one"):
            ts.to_polars(w)

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="to_polars accepts"):
            ts.to_polars(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# from_polars
# ---------------------------------------------------------------------------


class TestFromPolarsRoundTrip:
    @pytest.mark.parametrize(
        ("rng", "values", "freq"),
        [
            (MITRange(qq(2020, 1), qq(2021, 4)), np.arange(8.0), Quarterly()),
            (MITRange(mm(2020, 1), mm(2020, 6)), np.arange(6.0), Monthly()),
            (MITRange(yy(2020), yy(2024)), np.arange(5.0), Yearly()),
            (
                MITRange(weekly("2020-01-05"), weekly("2020-01-26")),
                np.arange(4.0),
                Weekly(),
            ),
            (
                MITRange(daily("2020-01-01"), daily("2020-01-05")),
                np.arange(5.0),
                Daily(),
            ),
            (
                MITRange(bdaily("2020-01-02"), bdaily("2020-01-08")),
                np.arange(5.0),
                BDaily(),
            ),
        ],
    )
    def test_tseries_round_trip(self, rng, values, freq):
        t = TSeries(rng, values)
        t2 = ts.from_polars(ts.to_polars(t), freq=freq)
        assert t2.range == t.range
        np.testing.assert_array_equal(t2.values, t.values)

    def test_unit_round_trip(self):
        t = TSeries(MITRange(MIT(Unit(), 0), MIT(Unit(), 4)), np.arange(5.0))
        # Default freq=None resolves to Unit() for integer time column.
        t2 = ts.from_polars(ts.to_polars(t))
        assert t2.range == t.range

    def test_halfyearly_round_trip(self):
        # HalfYearly emits Date column (period-end), round-trippable with freq.
        h0 = MIT(HalfYearly(), 2020 * 2 + 0)
        h2 = MIT(HalfYearly(), 2020 * 2 + 2)
        t = TSeries(MITRange(h0, h2), [1.0, 2.0, 3.0])
        t2 = ts.from_polars(ts.to_polars(t), freq=HalfYearly())
        assert t2.range == t.range

    def test_mvtseries_round_trip(self):
        m = MVTSeries(MITRange(qq(2020, 1), qq(2020, 4)), ["a", "b"], np.arange(8.0).reshape(4, 2))
        m2 = ts.from_polars(ts.to_polars(m), freq=Quarterly())
        assert isinstance(m2, MVTSeries)
        assert m2.column_names == m.column_names
        np.testing.assert_array_equal(m2.values, m.values)


class TestFromPolarsToWorkspace:
    """Round-trip Workspace through polars with ``to_workspace=True``."""

    def test_workspace_to_workspace_round_trip_via_dataframe(self):
        ty_a = TSeries(MITRange(yy(2020), yy(2023)), [1.0, 2.0, 3.0, 4.0])
        ty_b = TSeries(MITRange(yy(2020), yy(2023)), [10.0, 20.0, 30.0, 40.0])
        w = Workspace(a=ty_a, b=ty_b)
        df = ts.to_polars(w)
        w2 = ts.from_polars(df, to_workspace=True, freq=Yearly())
        assert isinstance(w2, Workspace)
        assert list(w2.keys()) == ["a", "b"]
        assert w2.a.range == ty_a.range
        np.testing.assert_array_equal(w2.a.values, ty_a.values)
        np.testing.assert_array_equal(w2.b.values, ty_b.values)

    def test_dataframe_to_workspace_to_dataframe_round_trip(self):
        df = pl.DataFrame(
            {
                "time": [
                    date(2020, 3, 31),
                    date(2020, 6, 30),
                    date(2020, 9, 30),
                    date(2020, 12, 31),
                ],
                "gdp": [1.0, 2.0, 3.0, 4.0],
                "cpi": [10.0, 11.0, 12.0, 13.0],
            }
        )
        w = ts.from_polars(df, to_workspace=True, freq=Quarterly())
        assert isinstance(w, Workspace)
        df2 = ts.to_polars(w)
        # Columns match; values match; time column round-trips.
        assert df2.columns == df.columns
        assert df2["time"].to_list() == df["time"].to_list()
        np.testing.assert_array_equal(df2["gdp"].to_numpy(), df["gdp"].to_numpy())
        np.testing.assert_array_equal(df2["cpi"].to_numpy(), df["cpi"].to_numpy())

    def test_single_value_column_workspace(self):
        # Even with a single non-time column, to_workspace=True wins over the
        # default TSeries shortcut and returns a Workspace.
        t = TSeries(MITRange(qq(2020, 1), qq(2020, 4)), np.arange(4.0))
        df = ts.to_polars(t)
        w = ts.from_polars(df, to_workspace=True, freq=Quarterly())
        assert isinstance(w, Workspace)
        assert list(w.keys()) == ["value"]
        np.testing.assert_array_equal(w["value"].values, t.values)

    def test_to_workspace_preserves_column_order(self):
        df = pl.DataFrame(
            {
                "time": [date(2020, 12, 31), date(2021, 12, 31)],
                "z": [1.0, 2.0],
                "a": [3.0, 4.0],
                "m": [5.0, 6.0],
            }
        )
        w = ts.from_polars(df, to_workspace=True, freq=Yearly())
        assert list(w.keys()) == ["z", "a", "m"]


class TestFromPolarsErrors:
    def test_date_column_requires_freq(self):
        t = TSeries(MITRange(qq(2020, 1), qq(2020, 4)), np.arange(4.0))
        df = ts.to_polars(t)
        with pytest.raises(ValueError, match="Date / Datetime"):
            ts.from_polars(df)

    def test_integer_column_rejects_non_unit_freq(self):
        df = pl.DataFrame({"time": [0, 1, 2], "value": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="Unit"):
            ts.from_polars(df, freq=Quarterly())

    def test_unsupported_time_dtype_raises(self):
        df = pl.DataFrame({"time": [0.5, 1.5, 2.5], "value": [1.0, 2.0, 3.0]})
        with pytest.raises(TypeError, match="dtype"):
            ts.from_polars(df)

    def test_missing_time_col_raises(self):
        df = pl.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(KeyError, match="not found"):
            ts.from_polars(df, time_col="time")

    def test_wrong_input_type_raises(self):
        with pytest.raises(TypeError, match="from_polars accepts"):
            ts.from_polars([1, 2, 3])  # type: ignore[arg-type]

    def test_gap_in_time_raises(self):
        df = pl.DataFrame(
            {
                "time": [date(2020, 3, 31), date(2020, 6, 30), date(2020, 12, 31)],
                "value": [1.0, 2.0, 3.0],
            }
        )
        with pytest.raises(ValueError, match="contiguous"):
            ts.from_polars(df, freq=Quarterly())


class TestFromPolarsLongFormat:
    def test_long_to_mvtseries(self):
        df = pl.DataFrame(
            {
                "time": [
                    date(2020, 3, 31),
                    date(2020, 6, 30),
                    date(2020, 3, 31),
                    date(2020, 6, 30),
                ],
                "var": ["a", "a", "b", "b"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )
        m = ts.from_polars(
            df,
            wide=False,
            name_col="var",
            value_col="value",
            freq=Quarterly(),
        )
        assert isinstance(m, MVTSeries)
        assert m.column_names == ("a", "b")
        np.testing.assert_array_equal(m.values, [[1.0, 3.0], [2.0, 4.0]])

    def test_long_requires_three_cols(self):
        df = pl.DataFrame({"time": []})
        with pytest.raises(ValueError, match="requires explicit"):
            ts.from_polars(df, wide=False)


class TestFromPolarsCustomTimeCol:
    def test_custom_time_col_name(self):
        t = TSeries(MITRange(qq(2020, 1), qq(2020, 4)), np.arange(4.0))
        df = ts.to_polars(t, time_col="when")
        t2 = ts.from_polars(df, time_col="when", freq=Quarterly())
        assert t2.range == t.range


class TestFromPolarsReturnsTSeriesWhenSingleValueColumn:
    def test_single_value_col_returns_tseries(self):
        t = TSeries(MITRange(qq(2020, 1), qq(2020, 4)), np.arange(4.0))
        df = ts.to_polars(t)
        t2 = ts.from_polars(df, freq=Quarterly())
        assert isinstance(t2, TSeries)

    def test_multi_value_col_returns_mvtseries(self):
        m = MVTSeries(MITRange(qq(2020, 1), qq(2020, 4)), ["a", "b"], np.arange(8.0).reshape(4, 2))
        m2 = ts.from_polars(ts.to_polars(m), freq=Quarterly())
        assert isinstance(m2, MVTSeries)


# ---------------------------------------------------------------------------
# Method bindings
# ---------------------------------------------------------------------------


class TestMethodBindings:
    def test_tseries_to_polars(self):
        t = TSeries(MITRange(qq(2020, 1), qq(2020, 4)), np.arange(4.0))
        df = t.to_polars()
        assert isinstance(df, pl.DataFrame)

    def test_mvtseries_to_polars(self):
        m = MVTSeries(MITRange(qq(2020, 1), qq(2020, 4)), ["a", "b"], np.arange(8.0).reshape(4, 2))
        df = m.to_polars()
        assert isinstance(df, pl.DataFrame)

    def test_workspace_to_polars(self):
        t = TSeries(MITRange(qq(2020, 1), qq(2020, 4)), np.arange(4.0))
        w = Workspace(a=t)
        df = w.to_polars()
        assert isinstance(df, pl.DataFrame)


# ---------------------------------------------------------------------------
# Missing-polars simulation
# ---------------------------------------------------------------------------


class TestPolarsMissing:
    def test_install_hint(self, monkeypatch):
        def _fake_find_spec(name):
            return None if name == "polars" else importlib.util.find_spec(name)

        monkeypatch.setattr(_interop_polars, "find_spec", _fake_find_spec)
        t = TSeries(MITRange(qq(2020, 1), qq(2020, 4)), np.arange(4.0))
        with pytest.raises(ImportError, match="requires polars"):
            _interop_polars.to_polars(t)

    def test_install_hint_text(self):
        assert "pip install" in _INSTALL_HINT
        assert "TimeSeriesEconPy[polars]" in _INSTALL_HINT
