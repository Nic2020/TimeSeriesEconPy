# SPDX-License-Identifier: MIT
"""Tests for the M2.2 high-traffic X-13ARIMA-SEATS spec builders.

Covers :class:`Span`, :class:`X13series`, and the eight builder functions
(:func:`tsecon.x13.series`, :func:`~tsecon.x13.arima`,
:func:`~tsecon.x13.automdl`, :func:`~tsecon.x13.transform`,
:func:`~tsecon.x13.regression`, :func:`~tsecon.x13.forecast`,
:func:`~tsecon.x13.seats`, :func:`~tsecon.x13.x11`) plus their dataclass
containers.

Test strategy is dataclass-level (no ``.spc`` round-trip): each builder is
exercised with success cases (default-omission, kwarg pass-through,
derivation) and failure cases (validation raises / warns). The ``.spc``
emission and Julia-fixture round-trip land in M2.4 alongside the
:mod:`tsecon.x13._write` serializer.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsecon import MVTSeries, Quarterly, TSeries
from tsecon.frequencies import Monthly, Yearly
from tsecon.mit import MIT
from tsecon.mitrange import MITRange
from tsecon.x13 import (
    ArimaModel,
    ArimaSpec,
    Span,
    X13arima,
    X13automdl,
    X13check,
    X13default,
    X13estimate,
    X13force,
    X13forecast,
    X13history,
    X13identify,
    X13metadata,
    X13outlier,
    X13pickmdl,
    X13regression,
    X13seats,
    X13series,
    X13slidingspans,
    X13spectrum,
    X13transform,
    X13x11,
    X13x11regression,
    ao,
    aos,
    arima,
    automdl,
    check,
    easter,
    estimate,
    force,
    forecast,
    history,
    identify,
    labor,
    lss,
    metadata,
    outlier,
    pickmdl,
    regression,
    sceaster,
    seats,
    series,
    slidingspans,
    spectrum,
    tdstock,
    thank,
    transform,
    x11,
    x11regression,
)
from tsecon.x13._spec import _X13DEFAULT, easterstock, td

_Q = Quarterly()
_M = Monthly()
_Y = Yearly(end_month=12)


def _q(year_: int, period_: int) -> MIT:
    return MIT.from_yp(_Q, year_, period_)


def _m(year_: int, period_: int) -> MIT:
    return MIT.from_yp(_M, year_, period_)


def _quarterly_tseries(n: int = 20, start_year: int = 2010) -> TSeries:
    return TSeries(_q(start_year, 1), np.arange(n, dtype=float))


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------


class TestSpan:
    def test_default_both_open(self) -> None:
        s = Span()
        assert s.b is None
        assert s.e is None

    def test_two_mit(self) -> None:
        s = Span(_q(2010, 1), _q(2015, 4))
        assert s.b == _q(2010, 1)
        assert s.e == _q(2015, 4)

    def test_b_only(self) -> None:
        s = Span(_q(2010, 1))
        assert s.b == _q(2010, 1)
        assert s.e is None

    def test_e_only(self) -> None:
        s = Span(None, _q(2015, 4))
        assert s.b is None
        assert s.e == _q(2015, 4)

    def test_from_range(self) -> None:
        mr = MITRange(_q(2010, 1), _q(2015, 4))
        s = Span.from_range(mr)
        assert s.b == _q(2010, 1)
        assert s.e == _q(2015, 4)

    def test_frozen(self) -> None:
        s = Span(_q(2010, 1), _q(2015, 4))
        with pytest.raises(AttributeError):
            s.b = _q(2011, 1)  # type: ignore[misc]

    def test_equality(self) -> None:
        assert Span(_q(2010, 1), _q(2015, 4)) == Span(_q(2010, 1), _q(2015, 4))
        assert Span() == Span()


# ---------------------------------------------------------------------------
# series()
# ---------------------------------------------------------------------------


class TestSeriesBuilder:
    def test_returns_x13series(self) -> None:
        t = _quarterly_tseries()
        result = series(t)
        assert isinstance(result, X13series)
        assert result.start == t.range.first()

    def test_defaults_are_sentinels(self) -> None:
        result = series(_quarterly_tseries())
        # Field-by-field check: every optional kwarg defaults to _X13DEFAULT.
        for field in (
            "appendbcst",
            "appendfcst",
            "comptype",
            "compwt",
            "decimals",
            "file",
            "format",
            "modelspan",
            "name",
            "precision",
            "print",
            "save",
            "span",
            "title",
            "type",
            "divpower",
            "missingcode",
            "missingval",
            "saveprecision",
            "trimzero",
        ):
            assert isinstance(getattr(result, field), X13default), field

    def test_data_is_copy(self) -> None:
        t = _quarterly_tseries()
        result = series(t)
        assert result.data is not t
        result.data.values[0] = -999.0
        assert t.values[0] == 0.0

    def test_period_auto_quarterly(self) -> None:
        # Per Julia upstream: non-Monthly / non-Yearly auto-sets period = ppy(t).
        result = series(_quarterly_tseries())
        assert result.period == 4

    def test_period_kept_monthly(self) -> None:
        t = TSeries(_m(2020, 1), np.arange(24.0))
        # Monthly inputs leave period at the user-provided value (or _X13DEFAULT).
        result = series(t)
        assert isinstance(result.period, X13default)

    def test_start_crops_data(self) -> None:
        t = _quarterly_tseries(n=20)  # 2010Q1..2014Q4
        result = series(t, start=_q(2012, 1))
        assert result.start == _q(2012, 1)
        assert result.data.range.first() == _q(2012, 1)
        assert result.data.range.last() == _q(2014, 4)
        assert len(result.data) == 12

    def test_start_out_of_range(self) -> None:
        t = _quarterly_tseries()
        with pytest.raises(ValueError, match="must be within"):
            series(t, start=_q(2099, 1))

    def test_name_truncates_with_warning(self) -> None:
        long_name = "A" * 70
        with pytest.warns(UserWarning, match="64 characters"):
            result = series(_quarterly_tseries(), name=long_name)
        assert result.name == "A" * 64

    def test_title_truncates_with_warning(self) -> None:
        long_title = "B" * 90
        with pytest.warns(UserWarning, match="79 characters"):
            result = series(_quarterly_tseries(), title=long_title)
        assert result.title == "B" * 79

    def test_short_name_untruncated(self) -> None:
        result = series(_quarterly_tseries(), name="short")
        assert result.name == "short"

    def test_span_must_be_within_range(self) -> None:
        t = _quarterly_tseries()
        with pytest.raises(ValueError, match="must be contained"):
            series(t, span=MITRange(_q(1990, 1), _q(1999, 4)))

    def test_span_object_within_range(self) -> None:
        t = _quarterly_tseries(n=20)
        span = Span(_q(2011, 1), _q(2013, 4))
        result = series(t, span=span)
        assert result.span is span

    def test_span_object_b_too_early(self) -> None:
        t = _quarterly_tseries(n=20)
        with pytest.raises(ValueError, match="start of the specified span"):
            series(t, span=Span(_q(1990, 1), _q(2013, 4)))

    def test_span_object_e_too_late(self) -> None:
        t = _quarterly_tseries(n=20)
        with pytest.raises(ValueError, match="end of the specified span"):
            series(t, span=Span(_q(2011, 1), _q(2099, 4)))

    def test_modelspan_validation(self) -> None:
        t = _quarterly_tseries()
        with pytest.raises(ValueError, match="modelspan"):
            series(t, modelspan=MITRange(_q(1990, 1), _q(1999, 4)))

    def test_divpower_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="between -9 and 9"):
            series(_quarterly_tseries(), divpower=10)
        with pytest.raises(ValueError, match="between -9 and 9"):
            series(_quarterly_tseries(), divpower=-10)

    def test_divpower_valid(self) -> None:
        result = series(_quarterly_tseries(), divpower=2)
        assert result.divpower == 2

    def test_nan_without_missingcode(self) -> None:
        values = np.arange(10.0)
        values[3] = np.nan
        t = TSeries(_q(2020, 1), values)
        with pytest.raises(ValueError, match="missingcode"):
            series(t)

    def test_nan_with_missingcode_replaces(self) -> None:
        values = np.arange(10.0)
        values[3] = np.nan
        t = TSeries(_q(2020, 1), values)
        result = series(t, missingcode=-99999.0)
        assert result.data.values[3] == -99999.0
        assert not np.isnan(result.data.values).any()

    def test_print_all_expansion(self) -> None:
        result = series(_quarterly_tseries(), print="all")
        assert isinstance(result.print, list)
        assert "default" in result.print
        assert "seriesplot" in result.print

    def test_save_all_expansion(self) -> None:
        result = series(_quarterly_tseries(), save=["all"])
        assert isinstance(result.save, list)
        assert "span" in result.save
        assert "specfile" in result.save

    def test_non_tseries_raises(self) -> None:
        with pytest.raises(TypeError, match="TSeries"):
            series([1, 2, 3])  # type: ignore[arg-type]

    def test_frozen(self) -> None:
        result = series(_quarterly_tseries())
        with pytest.raises(AttributeError):
            result.name = "foo"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# arima()
# ---------------------------------------------------------------------------


class TestArimaBuilder:
    def test_from_arimaspec(self) -> None:
        result = arima(ArimaSpec(1, 1, 1))
        assert isinstance(result, X13arima)
        assert isinstance(result.model, ArimaModel)
        assert len(result.model.specs) == 1

    def test_from_arimamodel(self) -> None:
        model = ArimaModel.from_pdq(1, 1, 1)
        result = arima(model)
        assert result.model is model

    def test_from_tuple(self) -> None:
        pair = ArimaSpec.two_seasonal(0, 1, 1, 0, 1, 1)
        result = arima(pair)
        assert len(result.model.specs) == 2

    def test_two_seasonal_via_arima(self) -> None:
        # Mirrors Julia: arima(ArimaSpec(0,1,1,0,1,1)...).
        result = arima(ArimaSpec.two_seasonal(0, 1, 1, 0, 1, 1))
        assert len(result.model.specs) == 2

    def test_ar_initial_values(self) -> None:
        result = arima(ArimaSpec(1, 0, 0), ar=[0.7])
        assert result.ar == [0.7]

    def test_ar_with_none(self) -> None:
        result = arima(ArimaSpec(2, 0, 0), ar=[0.7, None])
        assert result.ar == [0.7, None]

    def test_fixar_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="fixar must have"):
            arima(ArimaSpec(1, 0, 0), ar=[0.7, 0.3], fixar=[True])

    def test_fixma_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="fixma must have"):
            arima(ArimaSpec(0, 0, 1), ma=[0.5], fixma=[True, False])

    def test_title_truncates(self) -> None:
        long = "T" * 90
        with pytest.warns(UserWarning, match="79 characters"):
            result = arima(ArimaSpec(1, 1, 1), title=long)
        assert result.title == "T" * 79

    def test_invalid_model_type_raises(self) -> None:
        with pytest.raises(TypeError, match="model must be"):
            arima("not a model")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# automdl()
# ---------------------------------------------------------------------------


class TestAutomdlBuilder:
    def test_returns_x13automdl(self) -> None:
        result = automdl()
        assert isinstance(result, X13automdl)

    def test_default_print_is_list(self) -> None:
        result = automdl()
        assert isinstance(result.print, list)
        assert "autochoice" in result.print

    def test_default_savelog(self) -> None:
        result = automdl()
        assert result.savelog == "alldiagnostics"

    def test_diff_valid(self) -> None:
        result = automdl(diff=[1, 1])
        assert result.diff == [1, 1]

    def test_diff_length_error(self) -> None:
        with pytest.raises(ValueError, match="exactly two values"):
            automdl(diff=[1])
        with pytest.raises(ValueError, match="exactly two values"):
            automdl(diff=[1, 1, 1])

    def test_diff_regular_value_error(self) -> None:
        with pytest.raises(ValueError, match="regular differencing"):
            automdl(diff=[3, 1])

    def test_diff_seasonal_value_error(self) -> None:
        with pytest.raises(ValueError, match="seasonal differencing"):
            automdl(diff=[1, 2])

    def test_diff_and_maxdiff_warns(self) -> None:
        with pytest.warns(UserWarning, match="diff argument.*ignored"):
            automdl(diff=[1, 1], maxdiff=[2, 1])

    def test_maxdiff_length_error(self) -> None:
        with pytest.raises(ValueError, match="exactly two values"):
            automdl(maxdiff=[1])

    def test_maxdiff_regular_value_error(self) -> None:
        with pytest.raises(ValueError, match="regular maximum"):
            automdl(maxdiff=[5, 1])

    def test_maxdiff_seasonal_value_error(self) -> None:
        with pytest.raises(ValueError, match="seasonal maximum"):
            automdl(maxdiff=[2, 2])

    def test_maxdiff_with_none_accepted(self) -> None:
        result = automdl(maxdiff=[None, 1])
        assert result.maxdiff == [None, 1]

    def test_maxorder_length_error(self) -> None:
        with pytest.raises(ValueError, match="exactly two values"):
            automdl(maxorder=[1])

    def test_maxorder_regular_value_error(self) -> None:
        with pytest.raises(ValueError, match="regular ARMA"):
            automdl(maxorder=[5, 1])

    def test_maxorder_seasonal_value_error(self) -> None:
        with pytest.raises(ValueError, match="seasonal ARMA"):
            automdl(maxorder=[2, 3])

    def test_maxorder_with_none_accepted(self) -> None:
        result = automdl(maxorder=[None, 1])
        assert result.maxorder == [None, 1]


# ---------------------------------------------------------------------------
# transform()
# ---------------------------------------------------------------------------


class TestTransformBuilder:
    def test_returns_x13transform(self) -> None:
        result = transform()
        assert isinstance(result, X13transform)

    def test_default_savelog(self) -> None:
        result = transform()
        assert result.savelog == "autotransform"

    def test_power_only(self) -> None:
        result = transform(power=0.0)
        assert result.power == 0.0
        assert isinstance(result.func, X13default)

    def test_func_only(self) -> None:
        result = transform(func="log")
        assert result.func == "log"
        assert isinstance(result.power, X13default)

    def test_power_and_func_mutex(self) -> None:
        with pytest.raises(ValueError, match="power or func"):
            transform(power=0.5, func="log")

    def test_lpyear_requires_log_power(self) -> None:
        # adjust=lpyear + power=1.0 → reject
        with pytest.raises(ValueError, match="lpyear"):
            transform(adjust="lpyear", power=1.0)

    def test_lpyear_requires_log_func(self) -> None:
        # adjust=lpyear + func=sqrt → reject
        with pytest.raises(ValueError, match="lpyear"):
            transform(adjust="lpyear", func="sqrt")

    def test_lpyear_with_log_power(self) -> None:
        result = transform(adjust="lpyear", power=0.0)
        assert result.adjust == "lpyear"

    def test_lpyear_with_log_func(self) -> None:
        result = transform(adjust="lpyear", func="log")
        assert result.func == "log"

    def test_mode_too_long(self) -> None:
        with pytest.raises(ValueError, match="up to two values"):
            transform(mode=["diff", "ratio", "percent"])

    def test_mode_diff_ratio_incompatible(self) -> None:
        with pytest.raises(ValueError, match=r"diff.*ratio"):
            transform(mode=["diff", "ratio"])

    def test_mode_diff_percent_incompatible(self) -> None:
        with pytest.raises(ValueError, match=r"diff.*percent"):
            transform(mode=["diff", "percent"])

    def test_title_truncates(self) -> None:
        long = "X" * 90
        with pytest.warns(UserWarning, match="79 characters"):
            result = transform(title=long)
        assert result.title == "X" * 79

    def test_type_without_data_raises(self) -> None:
        with pytest.raises(ValueError, match="no data has been provided"):
            transform(type="user")

    def test_type_list_length_mismatch_tseries(self) -> None:
        t = _quarterly_tseries()
        with pytest.raises(ValueError, match="must match"):
            transform(data=t, type=["a", "b"])

    def test_derive_start_from_data(self) -> None:
        t = _quarterly_tseries(n=8, start_year=2015)
        result = transform(data=t)
        assert result.start == _q(2015, 1)

    def test_derive_name_from_mvts_single_col(self) -> None:
        mvts = MVTSeries(_q(2020, 1), x=np.arange(4.0))
        result = transform(data=mvts)
        assert result.name == "x"

    def test_derive_name_from_mvts_multi_col(self) -> None:
        mvts = MVTSeries(
            _q(2020, 1),
            a=np.arange(4.0),
            b=np.arange(4.0) + 10,
        )
        result = transform(data=mvts)
        assert result.name == ["a", "b"]

    def test_print_all_expansion(self) -> None:
        result = transform(print="all")
        assert isinstance(result.print, list)
        assert "aictransform" in result.print


# ---------------------------------------------------------------------------
# regression()
# ---------------------------------------------------------------------------


class TestRegressionBuilder:
    def test_returns_x13regression(self) -> None:
        result = regression()
        assert isinstance(result, X13regression)

    def test_default_savelog_is_list(self) -> None:
        result = regression()
        assert result.savelog == ["aictest", "chi2test"]

    def test_aicdiff_pvaictest_mutex(self) -> None:
        with pytest.raises(ValueError, match="aicdiff argument cannot"):
            regression(aicdiff=[0.5], pvaictest=0.05)

    def test_usertype_invalid_str(self) -> None:
        with pytest.raises(ValueError, match="usertype argument"):
            regression(usertype="bogus")

    def test_usertype_invalid_in_list(self) -> None:
        with pytest.raises(ValueError, match="usertype argument"):
            regression(usertype=["td", "bogus"])

    def test_usertype_valid(self) -> None:
        result = regression(usertype="td")
        assert result.usertype == "td"

    def test_aictest_invalid_str(self) -> None:
        with pytest.raises(ValueError, match="aictest"):
            regression(aictest="bogus")

    def test_aictest_invalid_in_list(self) -> None:
        with pytest.raises(ValueError, match="aictest"):
            regression(aictest=["td", "bogus"])

    def test_aictest_valid(self) -> None:
        result = regression(aictest=["td", "easter"])
        assert result.aictest == ["td", "easter"]

    def test_variables_passthrough(self) -> None:
        var = ao(_q(2020, 2))
        result = regression(variables=[var])
        assert result.variables == [var]

    def test_tdstock_bounds(self) -> None:
        with pytest.raises(ValueError, match="tdstock"):
            regression(variables=[tdstock(0)])
        with pytest.raises(ValueError, match="tdstock"):
            regression(variables=[tdstock(32)])

    def test_easter_bounds(self) -> None:
        with pytest.raises(ValueError, match="easter"):
            regression(variables=[easter(-1)])
        with pytest.raises(ValueError, match="easter"):
            regression(variables=[easter(26)])

    def test_labor_bounds(self) -> None:
        with pytest.raises(ValueError, match="labor"):
            regression(variables=[labor(26)])

    def test_thank_bounds(self) -> None:
        with pytest.raises(ValueError, match="thank"):
            regression(variables=[thank(-9)])
        with pytest.raises(ValueError, match="thank"):
            regression(variables=[thank(18)])

    def test_sceaster_bounds(self) -> None:
        with pytest.raises(ValueError, match="sceaster"):
            regression(variables=[sceaster(25)])

    def test_easterstock_bounds(self) -> None:
        with pytest.raises(ValueError, match="easterstock"):
            regression(variables=[easterstock(26)])

    def test_overlapping_aos_warns(self) -> None:
        with pytest.warns(UserWarning, match="overlapping aos"):
            regression(
                variables=[
                    aos(_q(2020, 1), _q(2020, 4)),
                    aos(_q(2020, 3), _q(2021, 2)),
                ]
            )

    def test_overlapping_lss_warns(self) -> None:
        with pytest.warns(UserWarning, match="overlapping lss"):
            regression(
                variables=[
                    lss(_q(2020, 1), _q(2020, 4)),
                    lss(_q(2020, 3), _q(2021, 2)),
                ]
            )

    def test_non_overlapping_aos_silent(self) -> None:
        # Adjacent (non-overlapping) ranges should not warn. The project-level
        # filterwarnings=error::UserWarning would turn any UserWarning here
        # into a raise, so a clean call is enough to lock the silent path.
        regression(
            variables=[
                aos(_q(2020, 1), _q(2020, 4)),
                aos(_q(2021, 1), _q(2021, 4)),
            ]
        )

    def test_derive_start_user_from_data(self) -> None:
        mvts = MVTSeries(_q(2020, 1), x=np.arange(4.0))
        result = regression(data=mvts)
        assert result.start == _q(2020, 1)
        assert result.user == "x"

    def test_derive_user_multi_col(self) -> None:
        mvts = MVTSeries(
            _q(2020, 1),
            a=np.arange(4.0),
            b=np.arange(4.0),
        )
        result = regression(data=mvts)
        assert result.user == ["a", "b"]

    def test_print_all_expansion(self) -> None:
        result = regression(print="all")
        assert isinstance(result.print, list)
        assert "regressionmatrix" in result.print


# ---------------------------------------------------------------------------
# forecast()
# ---------------------------------------------------------------------------


class TestForecastBuilder:
    def test_returns_x13forecast(self) -> None:
        result = forecast()
        assert isinstance(result, X13forecast)

    def test_defaults_are_sentinels(self) -> None:
        result = forecast()
        for field in (
            "exclude",
            "lognormal",
            "maxback",
            "maxlead",
            "print",
            "save",
            "probability",
        ):
            assert isinstance(getattr(result, field), X13default), field

    def test_maxlead_passthrough(self) -> None:
        result = forecast(maxlead=24)
        assert result.maxlead == 24

    def test_print_all_expansion(self) -> None:
        result = forecast(print="all")
        assert isinstance(result.print, list)
        assert "forecasts" in result.print

    def test_save_all_expansion(self) -> None:
        result = forecast(save=["all"])
        assert result.save == [
            "transformed",
            "variances",
            "forecasts",
            "transformedbcst",
            "backcasts",
        ]


# ---------------------------------------------------------------------------
# seats()
# ---------------------------------------------------------------------------


class TestSeatsBuilder:
    def test_returns_x13seats(self) -> None:
        result = seats()
        assert isinstance(result, X13seats)
        assert result.out == 0  # explicit default per Julia upstream

    def test_savelog_is_sentinel(self) -> None:
        # Julia upstream sets savelog=_X13default at the end of seats();
        # Python mirror keeps that behaviour.
        result = seats()
        assert isinstance(result.savelog, X13default)

    def test_epsiv_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="epsiv"):
            seats(epsiv=0.0)
        with pytest.raises(ValueError, match="epsiv"):
            seats(epsiv=-0.5)

    def test_epsiv_positive_ok(self) -> None:
        result = seats(epsiv=0.001)
        assert result.epsiv == 0.001

    def test_hplan_without_hpcycle_warns(self) -> None:
        with pytest.warns(UserWarning, match="Hodrick-Prescott"):
            seats(hpcycle=False, hplan=1000)

    def test_hplan_with_hpcycle_silent(self) -> None:
        # The project's error::UserWarning filter turns any UserWarning here
        # into a raise; a clean call locks the silent path.
        seats(hpcycle=True, hplan=1000)

    def test_print_all_rejected(self) -> None:
        with pytest.raises(ValueError, match="print='all'"):
            seats(print="all")
        with pytest.raises(ValueError, match="print='all'"):
            seats(print=["all"])

    def test_save_all_expands_and_zeros_out(self) -> None:
        result = seats(save="all", out=5)
        assert isinstance(result.save, list)
        assert "trend" in result.save
        assert "seasonaladj" in result.save
        assert result.out == 0  # forced reset

    def test_qmax_passthrough(self) -> None:
        result = seats(qmax=75)
        assert result.qmax == 75


# ---------------------------------------------------------------------------
# x11()
# ---------------------------------------------------------------------------


class TestX11Builder:
    def test_returns_x13x11(self) -> None:
        result = x11()
        assert isinstance(result, X13x11)

    def test_default_savelog(self) -> None:
        result = x11()
        assert result.savelog == "alldiagnostics"

    def test_trendma_must_be_odd(self) -> None:
        with pytest.raises(ValueError, match="odd number"):
            x11(trendma=12)

    def test_trendma_too_small(self) -> None:
        with pytest.raises(ValueError, match="between 3 and 101"):
            x11(trendma=1)

    def test_trendma_too_large(self) -> None:
        with pytest.raises(ValueError, match="between 3 and 101"):
            x11(trendma=103)

    def test_trendma_valid(self) -> None:
        result = x11(trendma=13)
        assert result.trendma == 13

    def test_sigmavec_requires_calendarsigma_select(self) -> None:
        with pytest.raises(ValueError, match="sigmavec"):
            x11(sigmavec=["jan", "feb"])
        with pytest.raises(ValueError, match="sigmavec"):
            x11(sigmavec=["jan"], calendarsigma="all")

    def test_sigmavec_with_calendarsigma_select(self) -> None:
        result = x11(sigmavec=["jan", "feb"], calendarsigma="select")
        assert result.sigmavec == ["jan", "feb"]

    def test_print_all_expansion(self) -> None:
        result = x11(print="all")
        assert isinstance(result.print, list)
        assert "seasadj" in result.print

    def test_save_all_expansion(self) -> None:
        result = x11(save=["all"])
        assert isinstance(result.save, list)
        assert "seasonal" in result.save
        assert "trend" in result.save


# ---------------------------------------------------------------------------
# Cross-builder integration: frozen + sentinel-default behaviour
# ---------------------------------------------------------------------------


class TestM22SurfaceShape:
    """Lock the M2.2 surface as a class of frozen dataclasses, all sharing the
    :class:`X13default` sentinel pattern.
    """

    @pytest.mark.parametrize(
        ("builder", "factory"),
        [
            (series, lambda: series(_quarterly_tseries())),
            (arima, lambda: arima(ArimaSpec(1, 1, 1))),
            (automdl, lambda: automdl()),
            (transform, lambda: transform()),
            (regression, lambda: regression()),
            (forecast, lambda: forecast()),
            (seats, lambda: seats()),
            (x11, lambda: x11()),
        ],
    )
    def test_frozen(self, builder, factory) -> None:
        result = factory()
        # All M2.2 containers are frozen dataclasses; any attribute assignment
        # raises FrozenInstanceError (a subclass of AttributeError).
        fields = [f for f in result.__slots__ if not f.startswith("_")]
        with pytest.raises(AttributeError):
            setattr(result, fields[0], None)

    def test_sentinel_singleton(self) -> None:
        assert _X13DEFAULT is X13default()
        assert _X13DEFAULT is X13default()


# ===========================================================================
# M2.3 — Rare spec builders
# ===========================================================================


# ---------------------------------------------------------------------------
# check()
# ---------------------------------------------------------------------------


class TestCheckBuilder:
    def test_returns_x13check(self) -> None:
        assert isinstance(check(), X13check)

    def test_default_savelog(self) -> None:
        assert check().savelog == "alldiagnostics"

    def test_kwargs_passthrough(self) -> None:
        result = check(maxlag=24, qtype="ljungbox", acflimit=2.0, qlimit=0.01)
        assert result.maxlag == 24
        assert result.qtype == "ljungbox"
        assert result.acflimit == 2.0
        assert result.qlimit == 0.01

    def test_print_all_expansion(self) -> None:
        result = check(print="all")
        assert isinstance(result.print, list)
        assert "acf" in result.print
        assert "normalitytest" in result.print

    def test_save_all_expansion(self) -> None:
        result = check(save=["all"])
        assert result.save == ["acf", "pacf", "acfsquared"]

    def test_explicit_print_list_passthrough(self) -> None:
        result = check(print=["acf", "pacf"])
        assert result.print == ["acf", "pacf"]


# ---------------------------------------------------------------------------
# estimate()
# ---------------------------------------------------------------------------


class TestEstimateBuilder:
    def test_returns_x13estimate(self) -> None:
        assert isinstance(estimate(), X13estimate)

    def test_default_savelog(self) -> None:
        assert estimate().savelog == "alldiagnostics"

    def test_kwargs_passthrough(self) -> None:
        result = estimate(
            exact="arma",
            maxiter=2000,
            outofsample=True,
            tol=1e-7,
            file="my.mdl",
            fix="all",
        )
        assert result.exact == "arma"
        assert result.maxiter == 2000
        assert result.outofsample is True
        assert result.tol == 1e-7
        assert result.file == "my.mdl"
        assert result.fix == "all"

    def test_print_all_expansion(self) -> None:
        result = estimate(print="all")
        assert isinstance(result.print, list)
        assert "model" in result.print
        assert "residuals" in result.print

    def test_save_all_expansion(self) -> None:
        result = estimate(save="all")
        assert isinstance(result.save, list)
        assert "estimates" in result.save


# ---------------------------------------------------------------------------
# force()
# ---------------------------------------------------------------------------


class TestForceBuilder:
    def test_returns_x13force(self) -> None:
        assert isinstance(force(), X13force)

    def test_kwargs_passthrough(self) -> None:
        result = force(
            lambda_=1.0,
            mode="ratio",
            rho=0.7,
            round=True,
            target="original",
            type="denton",
            usefcst=False,
            indforce=False,
        )
        assert result.lambda_ == 1.0
        assert result.mode == "ratio"
        assert result.rho == 0.7
        assert result.round is True
        assert result.target == "original"
        assert result.type == "denton"
        assert result.usefcst is False
        assert result.indforce is False

    def test_rho_too_low(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            force(rho=-0.1)

    def test_rho_too_high(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 1"):
            force(rho=1.1)

    def test_rho_boundary_values(self) -> None:
        assert force(rho=0.0).rho == 0.0
        assert force(rho=1.0).rho == 1.0

    def test_print_all_expansion(self) -> None:
        result = force(print="all")
        assert result.print == _FORCE_PRINT_ALL_EXPECTED

    def test_save_all_expansion(self) -> None:
        result = force(save=["all"])
        assert "rndsachangespct" in result.save


_FORCE_PRINT_ALL_EXPECTED = [
    "seasadjtot",
    "saround",
    "revsachanges",
    "rndsachanges",
]


# ---------------------------------------------------------------------------
# history()
# ---------------------------------------------------------------------------


class TestHistoryBuilder:
    def test_returns_x13history(self) -> None:
        assert isinstance(history(), X13history)

    def test_default_savelog(self) -> None:
        assert history().savelog == ["alldiagnostics"]

    def test_fstep_scalar_ok(self) -> None:
        assert history(fstep=12).fstep == 12

    def test_fstep_list_ok(self) -> None:
        assert history(fstep=[1, 2, 12]).fstep == [1, 2, 12]

    def test_fstep_list_too_long(self) -> None:
        with pytest.raises(ValueError, match="up to four"):
            history(fstep=[1, 2, 3, 4, 5])

    def test_fstep_zero_in_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="less than one"):
            history(fstep=[1, 0, 12])

    def test_fstep_negative_scalar_rejected(self) -> None:
        with pytest.raises(ValueError, match="less than one"):
            history(fstep=0)

    def test_sadjlags_scalar_ok(self) -> None:
        assert history(sadjlags=3).sadjlags == 3

    def test_sadjlags_list_ok(self) -> None:
        assert history(sadjlags=[1, 2, 3]).sadjlags == [1, 2, 3]

    def test_sadjlags_too_long(self) -> None:
        with pytest.raises(ValueError, match="up to five"):
            history(sadjlags=[1, 2, 3, 4, 5, 6])

    def test_sadjlags_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="less than one"):
            history(sadjlags=[0, 1])

    def test_sadjlags_negative_scalar_rejected(self) -> None:
        with pytest.raises(ValueError, match="less than one"):
            history(sadjlags=0)

    def test_kwargs_passthrough(self) -> None:
        result = history(
            estimates=["sadj", "aic"],
            fixmdl=True,
            fixreg=False,
            target="final",
            outlier="keep",
            outlierwin=12,
            refresh=False,
            transformfcst=False,
            x11outlier=True,
        )
        assert result.estimates == ["sadj", "aic"]
        assert result.fixmdl is True
        assert result.target == "final"
        assert result.outlier == "keep"

    def test_print_all_expansion(self) -> None:
        result = history(print="all")
        assert isinstance(result.print, list)
        assert "fcsthistory" in result.print


# ---------------------------------------------------------------------------
# identify()
# ---------------------------------------------------------------------------


class TestIdentifyBuilder:
    def test_returns_x13identify(self) -> None:
        assert isinstance(identify(), X13identify)

    def test_kwargs_passthrough(self) -> None:
        result = identify(diff=[0, 1], sdiff=[0, 1], maxlag=24)
        assert result.diff == [0, 1]
        assert result.sdiff == [0, 1]
        assert result.maxlag == 24

    def test_print_all_expansion(self) -> None:
        result = identify(print="all")
        assert isinstance(result.print, list)
        assert "acf" in result.print
        assert "regcoefficients" in result.print

    def test_save_all_expansion(self) -> None:
        result = identify(save=["all"])
        assert result.save == ["acf", "pacf"]


# ---------------------------------------------------------------------------
# metadata()
# ---------------------------------------------------------------------------


class TestMetadataBuilder:
    def test_returns_x13metadata(self) -> None:
        assert isinstance(metadata({"key": "value"}), X13metadata)

    def test_single_pair_tuple(self) -> None:
        result = metadata(("key", "value"))
        assert result.entries == (("key", "value"),)

    def test_list_of_pairs(self) -> None:
        result = metadata([("a", "1"), ("b", "2")])
        assert result.entries == (("a", "1"), ("b", "2"))

    def test_dict_input_preserves_order(self) -> None:
        result = metadata({"a": "1", "b": "2", "c": "3"})
        assert result.entries == (("a", "1"), ("b", "2"), ("c", "3"))

    def test_too_many_entries(self) -> None:
        with pytest.raises(ValueError, match="maximum of 20"):
            metadata({f"k{i}": str(i) for i in range(21)})

    def test_exactly_20_entries_ok(self) -> None:
        entries = {f"k{i}": str(i) for i in range(20)}
        assert len(metadata(entries).entries) == 20

    def test_key_too_long(self) -> None:
        with pytest.raises(ValueError, match="maximum length of 132"):
            metadata({"k" * 133: "v"})

    def test_value_too_long(self) -> None:
        with pytest.raises(ValueError, match=r"Values .* maximum length of 132"):
            metadata({"k": "v" * 133})

    def test_keys_combined_too_long(self) -> None:
        # 17 keys * 120 chars = 2040 > 2000. Each key 120 chars fits the
        # per-key 132 limit, so the per-key check passes and the
        # combined check fires.
        long = "k" * 117
        with pytest.raises(ValueError, match="maximum combined length of 2000"):
            metadata({f"{i:03d}{long}": "v" for i in range(17)})

    def test_frozen(self) -> None:
        m = metadata({"a": "1"})
        with pytest.raises(AttributeError):
            m.entries = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# outlier()
# ---------------------------------------------------------------------------


class TestOutlierBuilder:
    def test_returns_x13outlier(self) -> None:
        assert isinstance(outlier(), X13outlier)

    def test_default_savelog(self) -> None:
        assert outlier().savelog == "identified"

    def test_critical_scalar(self) -> None:
        assert outlier(critical=3.5).critical == 3.5

    def test_critical_list_ok(self) -> None:
        result = outlier(critical=[3.5, 4.0, 4.0])
        assert result.critical == [3.5, 4.0, 4.0]

    def test_critical_list_too_long(self) -> None:
        with pytest.raises(ValueError, match="up to 3"):
            outlier(critical=[3.5, 4.0, 4.0, 3.5])

    def test_critical_list_with_none(self) -> None:
        # Mirrors Julia's missing-as-default-value-marker.
        result = outlier(critical=[3.25, None, 3.25])
        assert result.critical == [3.25, None, 3.25]

    def test_lsrun_in_range(self) -> None:
        assert outlier(lsrun=5).lsrun == 5
        assert outlier(lsrun=0).lsrun == 0

    def test_lsrun_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="0 to 5"):
            outlier(lsrun=-1)

    def test_lsrun_too_high(self) -> None:
        with pytest.raises(ValueError, match="0 to 5"):
            outlier(lsrun=6)

    def test_almost_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="greater than zero"):
            outlier(almost=-0.1)

    def test_almost_zero_ok(self) -> None:
        assert outlier(almost=0.0).almost == 0.0

    def test_tcrate_in_range(self) -> None:
        assert outlier(tcrate=0.5).tcrate == 0.5

    def test_tcrate_at_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="greater than zero and less than one"):
            outlier(tcrate=0.0)

    def test_tcrate_at_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="greater than zero and less than one"):
            outlier(tcrate=1.0)

    def test_span_as_mitrange(self) -> None:
        mr = MITRange(_q(2010, 1), _q(2015, 4))
        result = outlier(span=mr)
        assert result.span == mr

    def test_span_as_span(self) -> None:
        s = Span(_q(2010, 1), _q(2015, 4))
        result = outlier(span=s)
        assert result.span == s

    def test_print_all_expansion(self) -> None:
        result = outlier(print="all")
        assert isinstance(result.print, list)
        assert "finaltests" in result.print

    def test_save_all_expansion(self) -> None:
        assert outlier(save="all").save == ["iterations", "finaltests"]

    def test_types_passthrough(self) -> None:
        assert outlier(types=["ao", "ls", "tc"]).types == ["ao", "ls", "tc"]

    def test_method_passthrough(self) -> None:
        assert outlier(method="addall").method == "addall"


# ---------------------------------------------------------------------------
# pickmdl()
# ---------------------------------------------------------------------------


class TestPickmdlBuilder:
    def _two_models(self) -> tuple[ArimaModel, ArimaModel]:
        m1 = ArimaModel.from_specs(ArimaSpec(0, 1, 1))
        m2 = ArimaModel.from_specs(ArimaSpec(1, 1, 0))
        return m1, m2

    def test_returns_x13pickmdl_from_file(self) -> None:
        result = pickmdl(file="models.mdl")
        assert isinstance(result, X13pickmdl)
        assert result.file == "models.mdl"
        assert isinstance(result.models, X13default)

    def test_returns_x13pickmdl_from_varargs(self) -> None:
        m1, m2 = self._two_models()
        result = pickmdl(m1, m2)
        assert isinstance(result, X13pickmdl)
        assert result.models == [m1, m2]

    def test_returns_x13pickmdl_from_unpacked_list(self) -> None:
        m1, m2 = self._two_models()
        result = pickmdl(*[m1, m2])
        assert result.models == [m1, m2]

    def test_default_savelog(self) -> None:
        assert pickmdl(file="x.mdl").savelog == "automodel"

    def test_no_models_no_file_rejected(self) -> None:
        with pytest.raises(ValueError, match="one or more ArimaModels"):
            pickmdl()

    def test_single_model_rejected(self) -> None:
        m1, _ = self._two_models()
        with pytest.raises(ValueError, match=r"at least 2"):
            pickmdl(m1)

    def test_two_models_with_two_defaults_rejected(self) -> None:
        m1 = ArimaModel.from_specs(ArimaSpec(0, 1, 1), default=True)
        m2 = ArimaModel.from_specs(ArimaSpec(1, 1, 0), default=True)
        with pytest.raises(ValueError, match="only have one model"):
            pickmdl(m1, m2)

    def test_two_models_with_one_default_ok(self) -> None:
        m1 = ArimaModel.from_specs(ArimaSpec(0, 1, 1), default=True)
        m2 = ArimaModel.from_specs(ArimaSpec(1, 1, 0))
        result = pickmdl(m1, m2)
        assert result.models is not None

    def test_percent_kwargs_in_range(self) -> None:
        result = pickmdl(file="x.mdl", bcstlim=20, fcstlim=15, qlim=5, overdiff=0.9)
        assert result.bcstlim == 20
        assert result.fcstlim == 15
        assert result.qlim == 5
        assert result.overdiff == 0.9

    def test_bcstlim_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 100"):
            pickmdl(file="x.mdl", bcstlim=101)
        with pytest.raises(ValueError, match="between 0 and 100"):
            pickmdl(file="x.mdl", bcstlim=-1)

    def test_fcstlim_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 100"):
            pickmdl(file="x.mdl", fcstlim=101)

    def test_qlim_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 100"):
            pickmdl(file="x.mdl", qlim=101)

    def test_overdiff_too_high(self) -> None:
        with pytest.raises(ValueError, match="not be greater than 1"):
            pickmdl(file="x.mdl", overdiff=1.1)

    def test_overdiff_too_low(self) -> None:
        with pytest.raises(ValueError, match=r"not be less than 0\.9"):
            pickmdl(file="x.mdl", overdiff=0.5)

    def test_identify_kwarg_does_not_shadow(self) -> None:
        # The kwarg ``identify`` shadows the spec-builder ``identify`` at
        # module level only inside the function body; from outside, the
        # builder is still importable.
        result = pickmdl(file="x.mdl", identify="first")
        assert result.identify == "first"

    def test_print_all_expansion(self) -> None:
        result = pickmdl(file="x.mdl", print="all")
        assert isinstance(result.print, list)
        assert "header" in result.print


# ---------------------------------------------------------------------------
# slidingspans()
# ---------------------------------------------------------------------------


class TestSlidingspansBuilder:
    def test_returns_x13slidingspans(self) -> None:
        assert isinstance(slidingspans(), X13slidingspans)

    def test_default_savelog(self) -> None:
        assert slidingspans().savelog == "percents"

    def test_kwargs_passthrough(self) -> None:
        result = slidingspans(
            cutchng=5.0,
            cutseas=5.0,
            cuttd=1.0,
            length=96,
            numspans=4,
            outlier="keep",
            additivesa="difference",
        )
        assert result.cutchng == 5.0
        assert result.cutseas == 5.0
        assert result.cuttd == 1.0
        assert result.length == 96
        assert result.numspans == 4
        assert result.outlier == "keep"

    def test_fixmdl_bool(self) -> None:
        assert slidingspans(fixmdl=True).fixmdl is True

    def test_fixmdl_symbol_clear(self) -> None:
        assert slidingspans(fixmdl="clear").fixmdl == "clear"

    def test_fixmdl_true_with_fixreg_warns(self) -> None:
        with pytest.warns(UserWarning, match="fixreg will be ignored"):
            slidingspans(fixmdl=True, fixreg=["td"])

    def test_fixmdl_false_with_fixreg_silent(self) -> None:
        # Quiet when fixmdl=False (project's error::UserWarning filter
        # turns an unexpected warning into a raise).
        result = slidingspans(fixmdl=False, fixreg=["td"])
        assert result.fixreg == ["td"]

    def test_print_all_expansion(self) -> None:
        result = slidingspans(print="all")
        assert isinstance(result.print, list)
        assert "tdspans" in result.print

    def test_save_all_expansion(self) -> None:
        result = slidingspans(save="all")
        assert isinstance(result.save, list)
        assert "saspans" in result.save


# ---------------------------------------------------------------------------
# spectrum()
# ---------------------------------------------------------------------------


class TestSpectrumBuilder:
    def test_returns_x13spectrum(self) -> None:
        assert isinstance(spectrum(), X13spectrum)

    def test_default_savelog(self) -> None:
        assert spectrum().savelog == "alldiagnostics"

    def test_kwargs_passthrough(self) -> None:
        result = spectrum(
            logqs=True,
            qcheck=True,
            tukey120=False,
            decibel=False,
            difference="first",
            maxar=20,
            peakwidth=2,
            series="original",
            siglevel=5,
            type="periodogram",
        )
        assert result.logqs is True
        assert result.qcheck is True
        assert result.tukey120 is False
        assert result.decibel is False
        assert result.difference == "first"
        assert result.maxar == 20
        assert result.peakwidth == 2
        assert result.series == "original"
        assert result.siglevel == 5
        assert result.type == "periodogram"

    def test_print_all_expansion(self) -> None:
        result = spectrum(print="all")
        assert isinstance(result.print, list)
        assert "tukeypeaks" in result.print

    def test_save_all_expansion(self) -> None:
        result = spectrum(save=["all"])
        assert isinstance(result.save, list)
        assert "specorig" in result.save


# ---------------------------------------------------------------------------
# x11regression()
# ---------------------------------------------------------------------------


class TestX11RegressionBuilder:
    def test_returns_x13x11regression(self) -> None:
        assert isinstance(x11regression(), X13x11regression)

    def test_default_savelog(self) -> None:
        assert x11regression().savelog == "aictest"

    def test_sigma_positive(self) -> None:
        assert x11regression(sigma=3.0).sigma == 3.0

    def test_sigma_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="greater than 0"):
            x11regression(sigma=0.0)

    def test_sigma_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="greater than 0"):
            x11regression(sigma=-1.0)

    def test_tdprior_correct_length(self) -> None:
        result = x11regression(tdprior=[0.7, 0.7, 0.7, 1.05, 1.4, 1.4, 1.05])
        assert result.tdprior == [0.7, 0.7, 0.7, 1.05, 1.4, 1.4, 1.05]

    def test_tdprior_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="length of exactly 7"):
            x11regression(tdprior=[1.0, 1.0, 1.0])

    def test_tdprior_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            x11regression(tdprior=[0.7, -0.1, 0.7, 1.05, 1.4, 1.4, 1.05])

    def test_aictest_scalar_allowed(self) -> None:
        assert x11regression(aictest="td").aictest == "td"

    def test_aictest_scalar_rejected(self) -> None:
        with pytest.raises(ValueError, match="can only contain"):
            x11regression(aictest="invalid")

    def test_aictest_list_allowed(self) -> None:
        assert x11regression(aictest=["td", "easter"]).aictest == ["td", "easter"]

    def test_aictest_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="can only contain"):
            x11regression(aictest=["td", "invalid"])

    def test_usertype_scalar_allowed(self) -> None:
        assert x11regression(usertype="td").usertype == "td"

    def test_usertype_scalar_rejected(self) -> None:
        with pytest.raises(ValueError, match="can only have"):
            x11regression(usertype="constant")

    def test_usertype_list_allowed(self) -> None:
        assert x11regression(usertype=["td", "holiday"]).usertype == ["td", "holiday"]

    def test_usertype_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="can only have"):
            x11regression(usertype=["td", "constant"])

    def test_data_derives_start_and_user(self) -> None:
        # Two columns → user becomes a list.
        m = MVTSeries(_q(2010, 1), ("x", "y"), np.zeros((20, 2)))
        result = x11regression(data=m)
        assert result.start == _q(2010, 1)
        assert result.user == ["x", "y"]

    def test_data_single_column_user_string(self) -> None:
        m = MVTSeries(_q(2010, 1), ("x",), np.zeros((20, 1)))
        result = x11regression(data=m)
        assert result.user == "x"

    def test_umdata_derives_umstart_and_umname(self) -> None:
        m = MVTSeries(_q(2010, 1), ("u1", "u2"), np.zeros((20, 2)))
        result = x11regression(umdata=m)
        assert result.umstart == _q(2010, 1)
        assert result.umname == ["u1", "u2"]

    def test_td_aictest_must_match_variables(self) -> None:
        # tdstock in aictest but td in variables → mismatch within the
        # TD family triggers the upstream check.
        with pytest.raises(ValueError, match="must correspond"):
            x11regression(aictest=["tdstock"], variables=[td()])

    def test_td_aictest_matches_variables_ok(self) -> None:
        result = x11regression(aictest=["td"], variables=[td()])
        assert result.aictest == ["td"]

    def test_span_as_span(self) -> None:
        s = Span(_q(2010, 1), _q(2015, 4))
        result = x11regression(span=s)
        assert result.span == s

    def test_outlierspan_as_mitrange(self) -> None:
        mr = MITRange(_q(2010, 1), _q(2015, 4))
        result = x11regression(outlierspan=mr)
        assert result.outlierspan == mr

    def test_print_all_expansion(self) -> None:
        result = x11regression(print="all")
        assert isinstance(result.print, list)
        assert "tradingday" in result.print

    def test_save_all_expansion(self) -> None:
        result = x11regression(save=["all"])
        assert isinstance(result.save, list)
        assert "tradingday" in result.save


# ---------------------------------------------------------------------------
# Cross-builder integration: frozen + sentinel-default behaviour (M2.3)
# ---------------------------------------------------------------------------


class TestM23SurfaceShape:
    """Lock the M2.3 surface as a class of frozen dataclasses, all sharing the
    :class:`X13default` sentinel pattern.
    """

    @pytest.mark.parametrize(
        ("builder", "factory"),
        [
            (check, lambda: check()),
            (estimate, lambda: estimate()),
            (force, lambda: force()),
            (history, lambda: history()),
            (identify, lambda: identify()),
            (metadata, lambda: metadata({"a": "1"})),
            (outlier, lambda: outlier()),
            (
                pickmdl,
                lambda: pickmdl(
                    ArimaModel.from_specs(ArimaSpec(0, 1, 1)),
                    ArimaModel.from_specs(ArimaSpec(1, 1, 0)),
                ),
            ),
            (slidingspans, lambda: slidingspans()),
            (spectrum, lambda: spectrum()),
            (x11regression, lambda: x11regression()),
        ],
    )
    def test_frozen(self, builder, factory) -> None:
        result = factory()
        fields = [f for f in result.__slots__ if not f.startswith("_")]
        with pytest.raises(AttributeError):
            setattr(result, fields[0], None)
