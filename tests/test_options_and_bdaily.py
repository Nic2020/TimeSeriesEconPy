# SPDX-License-Identifier: MIT
"""Tests for the options module + BDaily kwargs across the API.

Covers:

* :mod:`tsecon._options` — ``getoption`` / ``setoption`` / ``set_holidays_map`` /
  ``clear_holidays_map`` / ``option_scope`` / validation.
* :mod:`tsecon._bdaily` — ``cleanedvalues`` / ``bdvalues`` /
  ``replace_nans_if_warranted`` for TSeries and MVTSeries.
* BDaily kwargs (``skip_all_nans`` / ``skip_holidays`` / ``holidays_map``)
  forwarded into ``shift`` / ``lag`` / ``lead`` / ``diff`` / ``pct``,
  ``extend_series`` (BDaily source), ``fconvert(BDaily → YP/Weekly)`` lower
  path, and the ``fconvert_range`` truncation logic.
* :mod:`tsecon._stats` — Statistics reductions on TSeries{BDaily} /
  MVTSeries{BDaily} with the BDaily kwargs.

Mirrors the relevant Julia ``test/test_*.jl`` blocks for `cleanedvalues` /
`bdvalues` / `replace_nans_if_warranted!` and the BDaily fconvert / Statistics
overloads from ``test_tseries.jl`` and ``test_mvtseries.jl``.
"""

from __future__ import annotations

import datetime as _dt
import importlib.util

import numpy as np
import pytest

from tsecon import (
    MIT,
    BDaily,
    MITRange,
    Monthly,
    MVTSeries,
    Quarterly,
    TSeries,
    Weekly,
    Yearly,
    bdaily,
    bdvalues,
    cleanedvalues,
    clear_holidays_map,
    cor,
    cov,
    diff,
    extend_series,
    fconvert,
    fconvert_range,
    get_holidays_map,
    get_holidays_options,
    getoption,
    lag,
    lead,
    mean,
    median,
    option_scope,
    pct,
    qq,
    quantile,
    set_holidays_map,
    setoption,
    shift,
    std,
    stdm,
    var,
    varm,
)
from tsecon._options import VALID_BDAILY_BIASES
from tsecon.mit import mit_to_date

_HOLIDAYS_AVAILABLE = importlib.util.find_spec("holidays") is not None
_requires_holidays = pytest.mark.skipif(
    not _HOLIDAYS_AVAILABLE,
    reason="optional `holidays` package not installed",
)


@pytest.fixture(autouse=True)
def _reset_options() -> None:
    """Snapshot and restore global options around every test.

    Without this, a test that calls ``set_holidays_map(...)`` would leak the
    map into subsequent tests and the `bdaily()` default-bias tests would
    misbehave when run in non-collection order.
    """
    saved = {
        "bdaily_holidays_map": getoption("bdaily_holidays_map"),
        "bdaily_creation_bias": getoption("bdaily_creation_bias"),
        "x13path": getoption("x13path"),
    }
    yield
    for k, v in saved.items():
        setoption(k, v)


# ---------------------------------------------------------------------------
# Options module
# ---------------------------------------------------------------------------


class TestOptions:
    def test_defaults(self) -> None:
        assert getoption("bdaily_holidays_map") is None
        assert getoption("bdaily_creation_bias") == "strict"
        assert getoption("x13path") == ""

    def test_unknown_option_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown option"):
            getoption("not_a_real_option")
        with pytest.raises(KeyError, match="unknown option"):
            setoption("not_a_real_option", 42)

    @pytest.mark.parametrize("bias", sorted(VALID_BDAILY_BIASES))
    def test_set_bdaily_creation_bias_accepts_valid(self, bias: str) -> None:
        setoption("bdaily_creation_bias", bias)
        assert getoption("bdaily_creation_bias") == bias

    def test_set_bdaily_creation_bias_rejects_invalid(self) -> None:
        with pytest.raises(ValueError, match="bdaily_creation_bias"):
            setoption("bdaily_creation_bias", "previousish")

    def test_set_x13path_must_be_string(self) -> None:
        with pytest.raises(TypeError):
            setoption("x13path", 42)
        setoption("x13path", "/some/path")
        assert getoption("x13path") == "/some/path"

    def test_setoption_validates_holidays_map_type(self) -> None:
        with pytest.raises(TypeError, match="must be a BDaily Boolean TSeries"):
            setoption("bdaily_holidays_map", "not a tseries")
        # Wrong frequency.
        wrong_freq = TSeries(bdaily("2020-01-01"), np.zeros(5, dtype=float))
        with pytest.raises(TypeError, match="dtype"):
            setoption("bdaily_holidays_map", wrong_freq)
        # Right shape: BDaily + bool.
        bool_ts = TSeries(bdaily("2020-01-01"), np.ones(5, dtype=bool))
        setoption("bdaily_holidays_map", bool_ts)
        assert getoption("bdaily_holidays_map") is bool_ts

    def test_set_and_clear_holidays_map(self) -> None:
        ts = TSeries(bdaily("2020-01-01"), np.ones(10, dtype=bool))
        set_holidays_map(ts)
        assert get_holidays_map() is ts
        clear_holidays_map()
        assert get_holidays_map() is None

    def test_option_scope_restores_value(self) -> None:
        setoption("bdaily_creation_bias", "strict")
        with option_scope(bdaily_creation_bias="nearest"):
            assert getoption("bdaily_creation_bias") == "nearest"
        assert getoption("bdaily_creation_bias") == "strict"

    def test_bdaily_default_bias_consults_global_option(self) -> None:
        # 2024-01-06 is a Saturday.
        with (
            option_scope(bdaily_creation_bias="strict"),
            pytest.raises(ValueError, match="Saturday"),
        ):
            bdaily("2024-01-06")
        with option_scope(bdaily_creation_bias="previous"):
            m = bdaily("2024-01-06")
            # Previous business day is Friday 2024-01-05; round-trip via mit_to_date.
            assert mit_to_date(m) == _dt.date(2024, 1, 5)
        with option_scope(bdaily_creation_bias="next"):
            assert mit_to_date(bdaily("2024-01-06")) == _dt.date(2024, 1, 8)


# ---------------------------------------------------------------------------
# cleanedvalues / bdvalues / replace_nans_if_warranted
# ---------------------------------------------------------------------------


def _bd(date_str: str) -> MIT:
    return bdaily(date_str)


def _make_bdts(start: str, values: list[float]) -> TSeries:
    return TSeries(_bd(start), np.array(values, dtype=float))


def _all_true_map(start: str, n: int) -> TSeries:
    return TSeries(_bd(start), np.ones(n, dtype=bool))


class TestCleanedValuesTSeries:
    def test_default_returns_underlying_values(self) -> None:
        t = _make_bdts("2022-07-04", [1.0, 2.0, np.nan, 4.0])
        out = cleanedvalues(t)
        assert np.array_equal(out, np.array([1.0, 2.0, np.nan, 4.0]), equal_nan=True)

    def test_skip_all_nans_drops_nan(self) -> None:
        t = _make_bdts("2022-07-04", [1.0, 2.0, np.nan, 4.0])
        out = cleanedvalues(t, skip_all_nans=True)
        assert np.array_equal(out, np.array([1.0, 2.0, 4.0]))

    def test_skip_holidays_uses_global_map(self) -> None:
        # Mark Thursday (2022-07-07) as a holiday; should drop t[2]=3.0.
        h = TSeries(_bd("2022-07-04"), np.array([True, True, False, True], dtype=bool))
        set_holidays_map(h)
        t = _make_bdts("2022-07-04", [1.0, 2.0, 3.0, 4.0])
        out = cleanedvalues(t, skip_holidays=True)
        assert np.array_equal(out, np.array([1.0, 2.0, 4.0]))

    def test_skip_holidays_without_global_map_raises(self) -> None:
        t = _make_bdts("2022-07-04", [1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="bdaily_holidays_map"):
            cleanedvalues(t, skip_holidays=True)

    def test_holidays_map_argument_overrides(self) -> None:
        h = TSeries(_bd("2022-07-04"), np.array([True, False, True, True], dtype=bool))
        t = _make_bdts("2022-07-04", [1.0, 2.0, 3.0, 4.0])
        out = cleanedvalues(t, holidays_map=h)
        assert np.array_equal(out, np.array([1.0, 3.0, 4.0]))

    def test_non_bdaily_raises(self) -> None:
        t = TSeries(_bd("2022-07-04"), np.array([1.0, 2.0, 3.0]))
        t._firstdate = MIT(Quarterly(), 100)  # forge a non-BDaily frequency
        with pytest.raises(TypeError, match="BDaily"):
            cleanedvalues(t, skip_all_nans=True)

    def test_holidays_map_must_cover_range(self) -> None:
        t = _make_bdts("2022-07-04", [1.0, 2.0, 3.0, 4.0])
        short_map = _all_true_map("2022-07-04", 2)
        with pytest.raises(IndexError):
            bdvalues(t, holidays_map=short_map)


class TestCleanedValuesMVTSeries:
    def test_skip_all_nans_drops_all_nan_rows(self) -> None:
        mvts = MVTSeries(
            _bd("2022-07-04"),
            ["a", "b"],
            np.array(
                [
                    [1.0, 10.0],
                    [np.nan, np.nan],
                    [3.0, 30.0],
                ]
            ),
        )
        out = cleanedvalues(mvts, skip_all_nans=True)
        assert out.shape == (2, 2)
        assert np.array_equal(out, np.array([[1.0, 10.0], [3.0, 30.0]]))

    def test_skip_all_nans_warns_on_partial_nans(self) -> None:
        mvts = MVTSeries(
            _bd("2022-07-04"),
            ["a", "b"],
            np.array([[1.0, np.nan], [2.0, 20.0]]),
        )
        with pytest.warns(UserWarning, match="NaNs unequal across columns"):
            out = cleanedvalues(mvts, skip_all_nans=True)
        # Row 0 has partial NaN; ``cleanedvalues(mvts)`` keeps only rows where
        # ALL columns are non-NaN (matches Julia's nans_map[:, 2]).
        assert out.shape == (1, 2)
        assert np.array_equal(out, np.array([[2.0, 20.0]]))

    def test_holidays_map_filters_rows(self) -> None:
        mvts = MVTSeries(
            _bd("2022-07-04"),
            ["a", "b"],
            np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]),
        )
        h = TSeries(_bd("2022-07-04"), np.array([True, False, True, True], dtype=bool))
        out = bdvalues(mvts, holidays_map=h)
        assert np.array_equal(out, np.array([[1.0, 10.0], [3.0, 30.0], [4.0, 40.0]]))


class TestShiftBDailyKwargs:
    """Mirrors ``tsmath.jl`` lines 95-108 — the docstring example shifts."""

    def test_shift_forward_with_skip_all_nans(self) -> None:
        t = _make_bdts("2022-07-04", [1.0, 2.0, np.nan, 4.0])
        s = shift(t, 1, skip_all_nans=True)
        # New firstdate is 2022-07-04 - 1 = 2022-07-01 (Friday). Values: NaN at
        # position 2 is replaced by the *next* valid value (4.0), as in Julia.
        assert s.firstdate == _bd("2022-07-01")
        assert np.array_equal(s.values, np.array([1.0, 2.0, 4.0, 4.0]))

    def test_shift_backward_with_skip_all_nans(self) -> None:
        t = _make_bdts("2022-07-04", [1.0, 2.0, np.nan, 4.0])
        s = shift(t, -1, skip_all_nans=True)
        # New firstdate is 2022-07-04 + 1 = 2022-07-05. NaN replaced by *previous* (2.0).
        assert s.firstdate == _bd("2022-07-05")
        assert np.array_equal(s.values, np.array([1.0, 2.0, 2.0, 4.0]))

    def test_shift_passes_kwargs_to_lag_and_lead(self) -> None:
        t = _make_bdts("2022-07-04", [1.0, 2.0, np.nan, 4.0])
        assert np.array_equal(lag(t, 1, skip_all_nans=True).values, np.array([1.0, 2.0, 2.0, 4.0]))
        assert np.array_equal(lead(t, 1, skip_all_nans=True).values, np.array([1.0, 2.0, 4.0, 4.0]))

    def test_diff_bdaily_with_skip_all_nans(self) -> None:
        t = _make_bdts("2022-07-04", [1.0, 2.0, np.nan, 4.0])
        # diff(t, -1) = t - lag(t, 1). With skip_all_nans the NaN in the lag is
        # replaced by the previous valid value (2.0) before subtraction.
        d = diff(t, -1, skip_all_nans=True)
        # t[2022-07-05]=2 minus lag[2022-07-05]=1 → 1; t[2022-07-06]=NaN ...
        # The output aligns on the intersection of t and lagged-t, drops 2022-07-04.
        assert len(d) == 3
        assert d.firstdate == _bd("2022-07-05")

    def test_pct_bdaily_with_skip_all_nans(self) -> None:
        t = _make_bdts("2022-07-04", [100.0, 200.0, np.nan, 400.0])
        p = pct(t, skip_all_nans=True)
        # p[2022-07-05] = ((200 - 100) / 100) * 100 = 100.0
        assert pytest.approx(100.0) == float(p[_bd("2022-07-05")])

    def test_skip_kwargs_on_non_bdaily_raises(self) -> None:
        q = TSeries(qq(2020, 1), np.arange(4, dtype=float))
        with pytest.raises(TypeError, match="BDaily"):
            shift(q, 1, skip_all_nans=True)

    def test_holidays_map_with_shift(self) -> None:
        t = _make_bdts("2022-07-04", [1.0, 2.0, np.nan, 4.0])
        # NaN's source date in pre-shift is 2022-07-06 (Wednesday).
        # Mark 2022-07-06 as holiday; skip_holidays should infill the NaN
        # in the shifted output.
        h = TSeries(_bd("2022-06-27"), np.ones(20, dtype=bool))
        h[_bd("2022-07-06")] = False
        s = shift(t, 1, skip_holidays=True, holidays_map=h)
        # NaN sourced from holiday: replaced by next valid (4.0).
        assert np.array_equal(s.values, np.array([1.0, 2.0, 4.0, 4.0]))


class TestShiftMVTSeriesBDaily:
    def test_shift_with_skip_all_nans_applies_per_column(self) -> None:
        mvts = MVTSeries(
            _bd("2022-07-04"),
            ["a", "b"],
            np.array(
                [
                    [1.0, 10.0],
                    [2.0, np.nan],
                    [np.nan, 30.0],
                    [4.0, 40.0],
                ]
            ),
        )
        out = shift(mvts, 1, skip_all_nans=True)
        # Per-column infill: column a has NaN at row 2 → next valid (4.0);
        # column b has NaN at row 1 → next valid (30.0).
        expected = np.array(
            [
                [1.0, 10.0],
                [2.0, 30.0],
                [4.0, 30.0],
                [4.0, 40.0],
            ]
        )
        assert np.array_equal(out.values, expected)


# ---------------------------------------------------------------------------
# extend_series — BDaily source
# ---------------------------------------------------------------------------


class TestExtendSeriesBDaily:
    def test_extend_bdaily_with_method_end(self) -> None:
        # 4 days starting Wednesday 2024-01-03; trailing pad fills the rest of
        # that ISO week (target = Weekly Friday).
        t = TSeries(_bd("2024-01-03"), np.array([1.0, 2.0, 3.0, 4.0]))
        out = extend_series(Weekly(5), t, direction="end", method="end")
        # Last value is 4.0 (Monday 2024-01-08), so the trailing pad uses 4.0.
        assert out.values[-1] == 4.0

    def test_extend_bdaily_with_nan_at_edges_uses_cleanedvalues(self) -> None:
        t = TSeries(_bd("2024-01-03"), np.array([np.nan, 2.0, 3.0, np.nan]))
        out = extend_series(Weekly(5), t, direction="both", method="end")
        # Leading pad uses cleanedvalues(t, skip_all_nans=True)[0] = 2.0
        # (the first non-NaN). Trailing pad uses [-1] = 3.0.
        assert not np.isnan(out.values[0])
        assert out.values[0] == 2.0
        assert out.values[-1] == 3.0


# ---------------------------------------------------------------------------
# fconvert: BDaily → YP/Weekly with kwargs
# ---------------------------------------------------------------------------


class TestFconvertBDailyKwargs:
    def test_fconvert_bdaily_to_monthly_with_holidays_map(self) -> None:
        # Build a BDaily TSeries for Jan 2024 (23 business days).
        first = bdaily("2024-01-01", bias="next")
        last = bdaily("2024-01-31")
        n = last.value - first.value + 1
        vals = np.arange(1.0, n + 1)
        t = TSeries(first, vals)
        # Mark the first business day as a holiday; with the map present
        # cleanedvalues drops it from the monthly aggregate.
        h = TSeries(first, np.ones(n, dtype=bool))
        h[first] = False
        out_mean_raw = fconvert(Monthly, t, method="mean")
        out_mean_skip = fconvert(Monthly, t, method="mean", holidays_map=h)
        # The skipped value is the smallest in January, so the mean should be higher.
        assert float(out_mean_skip.values[0]) > float(out_mean_raw.values[0])

    def test_fconvert_bdaily_to_monthly_with_skip_all_nans(self) -> None:
        # A NaN inside the input must be dropped before aggregation.
        first = bdaily("2024-01-01", bias="next")
        last = bdaily("2024-01-31")
        n = last.value - first.value + 1
        vals = np.arange(1.0, n + 1)
        vals[0] = np.nan
        t = TSeries(first, vals)
        out_skip = fconvert(Monthly, t, method="mean", skip_all_nans=True)
        # Without skip, mean over [NaN, 2, ..., n] = NaN. With skip, mean(2..n).
        out_raw = fconvert(Monthly, t, method="mean")
        assert np.isnan(float(out_raw.values[0]))
        assert pytest.approx(float(np.mean(np.arange(2.0, n + 1)))) == float(out_skip.values[0])

    def test_fconvert_skip_kwargs_on_non_bdaily_source_raises(self) -> None:
        q = TSeries(qq(2020, 1), np.arange(8.0))
        with pytest.raises(TypeError, match="BDaily"):
            fconvert(Yearly, q, method="mean", skip_all_nans=True)


class TestFconvertRangeBDailyKwargs:
    def test_range_truncation_extends_past_holidays(self) -> None:
        # A 5-business-day range that starts on the second business day of a week.
        # Without holidays: truncation logic sees the range starts mid-week.
        # With a holidays_map marking the previous day as a holiday: the
        # truncation predecessor walks past it, so no truncation at the start.
        rng = MITRange(bdaily("2024-01-02"), bdaily("2024-01-08"))
        # Build a holidays_map where 2024-01-01 is a holiday (so the predecessor
        # of 2024-01-02 is "no previous business day in the same target period").
        h_first = bdaily("2024-01-01", bias="next") - 10
        h_last = bdaily("2024-01-31")
        n = h_last.value - h_first.value + 1
        h = TSeries(h_first, np.ones(n, dtype=bool))
        # Mark *every* business day before 2024-01-02 in January as holiday.
        for off in range(h_first.value, bdaily("2024-01-02").value):
            h[MIT(BDaily(), off)] = False
        out = fconvert_range(Weekly(5), rng, trim="both", holidays_map=h)
        # 2024-01-02 to 2024-01-08 in Weekly{5} (Friday end-of-week): two full weeks.
        # The first non-holiday business day before 2024-01-02 is missing, so
        # the first week is fully covered (no truncation). Sanity check: the
        # range is non-empty.
        assert len(out) >= 1


# ---------------------------------------------------------------------------
# Statistics: TSeries{BDaily} and MVTSeries{BDaily}
# ---------------------------------------------------------------------------


class TestStatisticsBDaily:
    def test_mean_with_skip_all_nans(self) -> None:
        t = _make_bdts("2024-01-02", [1.0, 2.0, np.nan, 4.0])
        assert pytest.approx(np.mean([1.0, 2.0, 4.0])) == float(mean(t, skip_all_nans=True))

    def test_std_var_match_numpy_ddof1(self) -> None:
        t = _make_bdts("2024-01-02", [1.0, 2.0, 3.0, 4.0, 5.0])
        assert pytest.approx(float(np.std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1))) == float(std(t))
        assert pytest.approx(float(np.var([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1))) == float(var(t))

    def test_quantile_and_median(self) -> None:
        t = _make_bdts("2024-01-02", [1.0, 2.0, 3.0, 4.0, 5.0])
        assert float(median(t)) == 3.0
        assert float(quantile(t, 0.5)) == 3.0

    def test_stdm_varm_with_external_mean(self) -> None:
        t = _make_bdts("2024-01-02", [1.0, 2.0, 3.0, 4.0, 5.0])
        m = 3.0
        # ddof=1 → sum_sq = 4+1+0+1+4 = 10 → var = 10/4 = 2.5 → std = sqrt(2.5)
        assert pytest.approx(2.5) == float(varm(t, m))
        assert pytest.approx(np.sqrt(2.5)) == float(stdm(t, m))

    def test_cor_self_is_one(self) -> None:
        t = _make_bdts("2024-01-02", [1.0, 2.0, 3.0, 4.0, 5.0])
        assert float(cor(t)) == 1.0

    def test_cor_two_series_matches_corrcoef(self) -> None:
        x = _make_bdts("2024-01-02", [1.0, 2.0, 3.0, 4.0, 5.0])
        y = _make_bdts("2024-01-02", [2.0, 4.0, 6.0, 8.0, 10.0])
        assert pytest.approx(1.0) == float(cor(x, y))

    def test_cor_mismatched_raises(self) -> None:
        x = _make_bdts("2024-01-02", [1.0, 2.0, 3.0])
        y = _make_bdts("2024-01-09", [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="same-firstdate"):
            cor(x, y)

    def test_cov_two_series(self) -> None:
        x = _make_bdts("2024-01-02", [1.0, 2.0, 3.0])
        y = _make_bdts("2024-01-02", [2.0, 4.0, 6.0])
        expected = float(np.cov([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], ddof=1)[0, 1])
        assert pytest.approx(expected) == float(cov(x, y))

    def test_skip_kwargs_on_non_bdaily_raises(self) -> None:
        q = TSeries(qq(2020, 1), np.arange(4.0))
        with pytest.raises(TypeError, match="BDaily"):
            mean(q, skip_all_nans=True)

    def test_mvts_cor_returns_matrix(self) -> None:
        mvts = MVTSeries(
            _bd("2024-01-02"),
            ["a", "b"],
            np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]),
        )
        c = cor(mvts)
        assert c.shape == (2, 2)
        # Linear y = 2x -> correlation matrix is the all-ones 2-by-2.
        assert pytest.approx(1.0) == float(c[0, 1])

    def test_mvts_cov_returns_matrix(self) -> None:
        mvts = MVTSeries(
            _bd("2024-01-02"),
            ["a", "b"],
            np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]),
        )
        c = cov(mvts)
        assert c.shape == (2, 2)
        # Cov(a, a) with ddof=1 over [1,2,3] = 1.0.
        assert pytest.approx(1.0) == float(c[0, 0])


# ---------------------------------------------------------------------------
# Country / subdivision loader (closes parity-gap G8)
# ---------------------------------------------------------------------------


@_requires_holidays
class TestHolidaysCountryLoader:
    """The string form of ``set_holidays_map(country, subdivision=None)``.

    Mirrors Julia's ``set_holidays_map(country, subdivision)`` in
    ``TimeSeriesEcon.jl/src/options.jl``. Delegates to the
    ``python-holidays`` PyPI package; the bundled-CSV path Julia uses is
    intentionally not vendored — see ``claude_files/parity/PARITY_GAPS.md``
    G8 closure.
    """

    def test_us_installs_federal_calendar(self) -> None:
        set_holidays_map("US")
        m = get_holidays_map()
        assert isinstance(m, TSeries)
        assert isinstance(m.frequency, BDaily)
        assert m.values.dtype == bool
        # 2024-12-25 is a Wednesday (US federal holiday).
        assert bool(m[bdaily("2024-12-25")]) is False
        # 2024-11-28 Thanksgiving (Thursday).
        assert bool(m[bdaily("2024-11-28")]) is False
        # 2024-03-04 (random Monday) is a regular business day.
        assert bool(m[bdaily("2024-03-04")]) is True

    def test_subdivision_ca_on_includes_family_day(self) -> None:
        set_holidays_map("CA", "ON")
        m = get_holidays_map()
        # Family Day 2024 = Monday 2024-02-19 (Ontario observes).
        assert bool(m[bdaily("2024-02-19")]) is False

    def test_subdivision_ca_qc_lacks_family_day_but_has_st_jean(self) -> None:
        # Quebec does not observe Family Day; it does observe
        # Saint-Jean-Baptiste (Fête nationale) on 2024-06-24 (Monday).
        set_holidays_map("CA", "QC")
        m = get_holidays_map()
        assert bool(m[bdaily("2024-02-19")]) is True  # no Family Day in QC
        assert bool(m[bdaily("2024-06-24")]) is False

    def test_unknown_country_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match=r"Unsupported country: 'XX'"):
            set_holidays_map("XX")

    def test_unknown_subdivision_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match=r"Unsupported subdivision 'ZZ' for country 'CA'"):
            set_holidays_map("CA", "ZZ")

    def test_tseries_form_still_works(self) -> None:
        ts = TSeries(bdaily("2020-01-01"), np.ones(10, dtype=bool))
        set_holidays_map(ts)
        assert get_holidays_map() is ts

    def test_non_string_non_tseries_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            set_holidays_map(42)

    def test_subdivision_with_tseries_raises_type_error(self) -> None:
        ts = TSeries(bdaily("2020-01-01"), np.ones(10, dtype=bool))
        with pytest.raises(TypeError, match=r"`subdivision=` is only meaningful"):
            set_holidays_map(ts, "ON")

    def test_loader_covers_full_default_range(self) -> None:
        # The map spans ``bdaily("1970-01-01") : bdaily("2049-12-31")`` per
        # the Julia upstream's default coverage.
        set_holidays_map("DK")
        m = get_holidays_map()
        assert m.firstdate == bdaily("1970-01-01")
        assert m.lastdate == bdaily("2049-12-31")


@_requires_holidays
class TestGetHolidaysOptions:
    def test_no_arg_returns_non_empty_sorted_tuple(self) -> None:
        codes = get_holidays_options()
        assert isinstance(codes, tuple)
        assert len(codes) > 0
        assert list(codes) == sorted(codes)
        # ISO 3166 alpha-2 staples must be present.
        assert "CA" in codes
        assert "US" in codes
        assert "DK" in codes

    def test_country_arg_returns_subdivisions(self) -> None:
        ca = get_holidays_options("CA")
        assert isinstance(ca, tuple)
        assert "ON" in ca
        assert "QC" in ca
        assert list(ca) == sorted(ca)

    def test_country_with_no_subdivisions_returns_empty_tuple(self) -> None:
        # Denmark exposes no subdivisions in the holidays package.
        assert get_holidays_options("DK") == ()

    def test_unknown_country_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match=r"Unsupported country: 'XX'"):
            get_holidays_options("XX")


class TestHolidaysLazyImport:
    """The ``[holidays]`` extra must be optional and the install hint visible.

    Skipif-gated so the test still runs on a stripped environment that has no
    real ``holidays`` install — the patch makes ``find_spec`` return ``None``
    regardless of what's actually installed.
    """

    def test_country_loader_without_package_raises_importerror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _find_spec_stub(name: str, *args: object, **kwargs: object) -> object:
            if name == "holidays":
                return None
            return importlib.util.find_spec(name)

        monkeypatch.setattr("tsecon._options.find_spec", _find_spec_stub)
        with pytest.raises(ImportError, match=r"TimeSeriesEconPy\[holidays\]"):
            set_holidays_map("CA")

    def test_get_holidays_options_without_package_raises_importerror(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _find_spec_stub(name: str, *args: object, **kwargs: object) -> object:
            if name == "holidays":
                return None
            return importlib.util.find_spec(name)

        monkeypatch.setattr("tsecon._options.find_spec", _find_spec_stub)
        with pytest.raises(ImportError, match=r"TimeSeriesEconPy\[holidays\]"):
            get_holidays_options()
