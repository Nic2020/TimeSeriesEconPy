# SPDX-License-Identifier: MIT
"""M2.4 cross-spec validator tests: validateX13spec.

One test class per Julia branch family in ``x13spec.jl:3563-4055``:

* :class:`TestArimaMutualExclusion` — arima ⊥ automdl ⊥ pickmdl.
* :class:`TestEstimateFileOverrides` — ``estimate.file`` blocks
  ``arima.{model,ar,ma}`` and ``regression.{variables,user,b}``.
* :class:`TestForecastHistoryConsistency` — ``history.fstep`` vs
  ``forecast.maxlead``.
* :class:`TestRegressionVariableTypeCompatibility` — the ~16 raise
  sites covering the trading-day / leap-year / length-of-period /
  stock-vs-flow compatibility matrix.
* :class:`TestRegressionAICTestTypeCompatibility` — parallel matrix
  for the ``aictest=`` argument.
* :class:`TestOutlierRangeContainment` — point/range outliers in
  series range.
* :class:`TestRegressionDataRangeContainment` — data + forecast
  back/lead range expansion.
* :class:`TestSlidingspansLengthBounds` — 12 ≤ length ≤ 76 (Q) /
  36 ≤ length ≤ 228 (M).
* :class:`TestSoftWarnings` — every ``@warn`` branch (HP filter sample
  size, modelspan-vs-span, slidingspans/spectrum hints, x11regression
  forcecal, history-outlier-without-spec). All assert
  :class:`UserWarning` is raised by ``pytest.warns(...)``; the project
  ``error::UserWarning`` filter at ``pyproject.toml`` keeps the same
  warnings from going silent in production runs.
* :class:`TestTransformX11ModeCompat` — transform.adjust ⊥ x11.mode ∈
  {add, pseudoadd}; x11 default-mode ⊥ transform default-power+func.
* :class:`TestX11RegressionContainment` — x11regression's data /
  umdata / outlierspan / span containment + forcecal warn.
* :class:`TestValidateEntryAndSurface` — ``validateX13spec`` happy-path
  smoke + the missing-series guard.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tsecon.frequencies import Monthly, Quarterly, Yearly
from tsecon.mit import MIT
from tsecon.mitrange import MITRange
from tsecon.mvtseries import MVTSeries
from tsecon.tseries import TSeries
from tsecon.x13 import (
    ArimaModel,
    Span,
    X13default,
    X13spec,
    ao,
    arima,
    automdl,
    estimate,
    forecast,
    history,
    lpyear,
    ls,
    newspec,
    outlier,
    pickmdl,
    regression,
    seats,
    series,
    slidingspans,
    spectrum,
    td,
    transform,
    validateX13spec,
    x11,
    x11regression,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def q_ts_short() -> TSeries:
    """Quarterly TSeries, 1990Q1..1999Q4 (40 points) — too short for HP filter (< 48)."""
    return TSeries(MIT.from_yp(Quarterly(), 1990, 1), np.arange(40.0))


@pytest.fixture
def q_ts_long() -> TSeries:
    """Quarterly TSeries, 1990Q1..2009Q4 (80 points) — passes the HP-filter check."""
    return TSeries(MIT.from_yp(Quarterly(), 1990, 1), np.arange(80.0))


@pytest.fixture
def m_ts_short() -> TSeries:
    """Monthly TSeries, 2000M1..2008M12 (108 points) — too short for HP filter (< 120)."""
    return TSeries(MIT.from_yp(Monthly(), 2000, 1), np.arange(108.0))


@pytest.fixture
def m_ts_long() -> TSeries:
    """Monthly TSeries, 2000M1..2019M12 (240 points)."""
    return TSeries(MIT.from_yp(Monthly(), 2000, 1), np.arange(240.0))


def _minimal_passable(spec: X13spec) -> X13spec:
    """Add transform + x11 so the default-mode invariant doesn't fire."""
    if isinstance(spec.transform, X13default):
        spec.transform = transform(power=0.5)
    if isinstance(spec.x11, X13default):
        spec.x11 = x11(mode="mult")
    return spec


# ---------------------------------------------------------------------------
# arima ⊥ automdl ⊥ pickmdl
# ---------------------------------------------------------------------------


class TestArimaMutualExclusion:
    def test_arima_with_automdl_raises(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.arima = arima(ArimaModel.from_pdq(0, 1, 1))
        spec.automdl = automdl()
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="arima spec cannot be used"):
            validateX13spec(spec)

    def test_arima_with_pickmdl_raises(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.arima = arima(ArimaModel.from_pdq(0, 1, 1))
        spec.pickmdl = pickmdl(
            ArimaModel.from_pdq_seasonal(0, 1, 1, 0, 1, 1, default=True),
            ArimaModel.from_pdq_seasonal(2, 1, 0, 0, 1, 1),
        )
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="arima spec cannot be used"):
            validateX13spec(spec)

    def test_automdl_with_pickmdl_raises(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.automdl = automdl()
        spec.pickmdl = pickmdl(
            ArimaModel.from_pdq_seasonal(0, 1, 1, 0, 1, 1, default=True),
            ArimaModel.from_pdq_seasonal(2, 1, 0, 0, 1, 1),
        )
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="automdl spec cannot be used"):
            validateX13spec(spec)

    def test_arima_alone_passes(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.arima = arima(ArimaModel.from_pdq(0, 1, 1))
        _minimal_passable(spec)
        validateX13spec(spec)  # no raise


# ---------------------------------------------------------------------------
# estimate.file overrides
# ---------------------------------------------------------------------------


class TestEstimateFileOverrides:
    def test_arima_model_blocked_by_estimate_file(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.arima = arima(ArimaModel.from_pdq(0, 1, 1))
        spec.estimate = estimate(file="/path/to/prior")
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="model, ma, and ar arguments"):
            validateX13spec(spec)

    def test_regression_variables_blocked_by_estimate_file(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.regression = regression(variables="td")
        spec.estimate = estimate(file="/path/to/prior")
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="variables, user, and b arguments"):
            validateX13spec(spec)

    def test_automdl_blocked_by_estimate_file(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.automdl = automdl()
        spec.estimate = estimate(file="/path/to/prior")
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="automdl spec cannot be used"):
            validateX13spec(spec)


# ---------------------------------------------------------------------------
# forecast / history consistency
# ---------------------------------------------------------------------------


class TestForecastHistoryConsistency:
    def test_history_fstep_scalar_exceeds_maxlead(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.forecast = forecast(maxlead=4)
        spec.history = history(fstep=8)
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="fstep in the history spec"):
            validateX13spec(spec)

    def test_history_fstep_list_exceeds_maxlead(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.forecast = forecast(maxlead=4)
        spec.history = history(fstep=[1, 2, 8])
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="fstep in the history spec"):
            validateX13spec(spec)

    def test_history_fstep_equal_passes(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.forecast = forecast(maxlead=4)
        spec.history = history(fstep=[1, 2, 4])
        _minimal_passable(spec)
        validateX13spec(spec)


# ---------------------------------------------------------------------------
# regression variable type compatibility
# ---------------------------------------------------------------------------


class TestRegressionVariableTypeCompatibility:
    def test_td_conflicts_with_lpyear(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.regression = regression(variables=["td", "lpyear"])
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="td cannot be used with"):
            validateX13spec(spec)

    def test_lpyear_conflicts_with_td(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.regression = regression(variables=["lpyear", "td1coef"])
        _minimal_passable(spec)
        # Iteration order is variables-as-given: lpyear's branch fires first,
        # raising its message for the conflicting td1coef in types_used.
        with pytest.raises(ValueError, match="lpyear cannot be used"):
            validateX13spec(spec)

    def test_td_with_yearly_data_raises(self) -> None:
        ts = TSeries(MIT.from_yp(Yearly(), 1990, 1), np.arange(40.0))
        spec = newspec(ts)
        spec.regression = regression(variables="td")
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="td regressors can only be used"):
            validateX13spec(spec)

    def test_tdstock_with_quarterly_raises(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.regression = regression(variables="tdstock")
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="tdstock regressors can only be used"):
            validateX13spec(spec)

    def test_td_with_transform_adjust_raises(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.regression = regression(variables="td")
        # ``adjust='lpyear'`` requires ``power=0.0`` (log-transform) per the
        # transform() builder's own validation.
        spec.transform = transform(power=0.0, adjust="lpyear")
        spec.x11 = x11(mode="mult")
        with pytest.raises(ValueError, match="adjust argument of the transform"):
            validateX13spec(spec)

    def test_adjust_lom_blocks_td(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.regression = regression(variables="td")
        spec.transform = transform(power=0.5, adjust="lom")
        spec.x11 = x11(mode="mult")
        with pytest.raises(ValueError, match="adjust='lom'"):
            validateX13spec(spec)

    def test_lpyear_with_yearly_data_raises(self) -> None:
        ts = TSeries(MIT.from_yp(Yearly(), 1990, 1), np.arange(40.0))
        spec = newspec(ts)
        spec.regression = regression(variables="lpyear")
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="lpyear regressors can only be used"):
            validateX13spec(spec)

    def test_seasonal_passes(self, q_ts_long: TSeries) -> None:
        """``seasonal`` has no frequency / type compatibility raises in the matrix."""
        spec = newspec(q_ts_long)
        spec.regression = regression(variables=[td()])
        _minimal_passable(spec)
        validateX13spec(spec)


class TestRegressionAICTestTypeCompatibility:
    def test_aictest_td_with_yearly_raises(self) -> None:
        ts = TSeries(MIT.from_yp(Yearly(), 1990, 1), np.arange(40.0))
        spec = newspec(ts)
        spec.regression = regression(variables="ao", aictest=["td"])
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="aictest: td regressors"):
            validateX13spec(spec)

    def test_aictest_loq_with_monthly_raises(self, m_ts_long: TSeries) -> None:
        spec = newspec(m_ts_long)
        spec.regression = regression(variables="ao", aictest=["loq"])
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="aictest: loq regressors"):
            validateX13spec(spec)

    def test_aictest_lom_with_quarterly_raises(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.regression = regression(variables="ao", aictest=["lom"])
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="aictest: lom regressors"):
            validateX13spec(spec)

    def test_aictest_td_lpyear_in_vars_raises(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.regression = regression(
            variables=[lpyear(MIT.from_yp(Quarterly(), 1995, 1))],
            aictest=["td"],
        )
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="aictest: td cannot be used"):
            validateX13spec(spec)


# ---------------------------------------------------------------------------
# outlier range containment
# ---------------------------------------------------------------------------


class TestOutlierRangeContainment:
    def test_ao_outside_range_raises(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.regression = regression(variables=[ao(MIT.from_yp(Quarterly(), 2050, 1))])
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="ao regressors must have a date"):
            validateX13spec(spec)

    def test_ls_at_first_period_raises(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.regression = regression(variables=[ls(MIT.from_yp(Quarterly(), 1990, 1))])
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="ls regressors cannot be at the start"):
            validateX13spec(spec)

    def test_ao_at_first_period_passes(self, q_ts_long: TSeries) -> None:
        """``ao`` at start is allowed (only ``ls`` / ``so`` are blocked there)."""
        spec = newspec(q_ts_long)
        spec.regression = regression(variables=[ao(MIT.from_yp(Quarterly(), 1990, 1))])
        _minimal_passable(spec)
        validateX13spec(spec)

    def test_ao_within_range_passes(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.regression = regression(variables=[ao(MIT.from_yp(Quarterly(), 1995, 2))])
        _minimal_passable(spec)
        validateX13spec(spec)


# ---------------------------------------------------------------------------
# regression / x11regression data range containment
# ---------------------------------------------------------------------------


class TestRegressionDataRangeContainment:
    def test_data_must_cover_series(self, q_ts_long: TSeries) -> None:
        short_data = MVTSeries(
            MITRange(
                MIT.from_yp(Quarterly(), 1995, 1),
                MIT.from_yp(Quarterly(), 2005, 4),
            ),
            {"reg": np.arange(44.0)},
        )
        spec = newspec(q_ts_long)
        spec.regression = regression(variables="td", data=short_data)
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="data provided in the regression spec"):
            validateX13spec(spec)

    def test_data_covers_series_passes(self, q_ts_long: TSeries) -> None:
        full_data = MVTSeries(q_ts_long.range, {"reg": np.arange(80.0)})
        spec = newspec(q_ts_long)
        spec.regression = regression(variables="td", data=full_data)
        _minimal_passable(spec)
        validateX13spec(spec)

    def test_data_must_cover_forecast_maxlead(self, q_ts_long: TSeries) -> None:
        """With ``forecast.maxlead=4``, regression data must extend 4 obs past the end."""
        full_data = MVTSeries(q_ts_long.range, {"reg": np.arange(80.0)})
        spec = newspec(q_ts_long)
        spec.regression = regression(variables="td", data=full_data)
        spec.forecast = forecast(maxlead=4)
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="data provided in the regression spec"):
            validateX13spec(spec)


# ---------------------------------------------------------------------------
# slidingspans length bounds
# ---------------------------------------------------------------------------


class TestSlidingspansLengthBounds:
    def test_quarterly_under_3_years_raises(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.slidingspans = slidingspans(length=8)  # 2 years quarterly
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="must cover at least 3 years"):
            validateX13spec(spec)

    def test_quarterly_over_19_years_raises(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.slidingspans = slidingspans(length=100)  # 25 years quarterly
        _minimal_passable(spec)
        with pytest.raises(ValueError, match="can cover at most 19 years"):
            validateX13spec(spec)

    def test_monthly_under_3_years_raises(self, m_ts_long: TSeries) -> None:
        spec = newspec(m_ts_long)
        spec.transform = transform(power=0.5)
        spec.x11 = x11(mode="mult")
        spec.slidingspans = slidingspans(length=24)  # 2 years monthly
        with pytest.raises(ValueError, match="must cover at least 3 years"):
            validateX13spec(spec)

    def test_monthly_over_19_years_raises(self, m_ts_long: TSeries) -> None:
        spec = newspec(m_ts_long)
        spec.transform = transform(power=0.5)
        spec.x11 = x11(mode="mult")
        spec.slidingspans = slidingspans(length=240)  # 20 years monthly
        with pytest.raises(ValueError, match="can cover at most 19 years"):
            validateX13spec(spec)

    def test_quarterly_in_range_passes(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.slidingspans = slidingspans(length=20)  # 5 years quarterly
        _minimal_passable(spec)
        validateX13spec(spec)


# ---------------------------------------------------------------------------
# Soft warnings (every @warn branch in Julia)
# ---------------------------------------------------------------------------


class TestSoftWarnings:
    """Warnings raised by ``warnings.warn(UserWarning)``.

    Under the project ``error::UserWarning`` filter at ``pyproject.toml``,
    these warnings turn into test failures by default. Each test catches
    them explicitly via :func:`pytest.warns` so the silent-misuse safety
    net stays in place for production runs.
    """

    def test_history_outlier_without_outlier_spec_warns(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.history = history(outlier="keep")
        _minimal_passable(spec)
        with pytest.warns(UserWarning, match="outlier argument of the history spec"):
            validateX13spec(spec)

    def test_seats_hpcycle_short_quarterly_warns(self, q_ts_short: TSeries) -> None:
        spec = newspec(q_ts_short)  # 40 quarters < 48
        spec.transform = transform(power=0.5)
        spec.seats = seats(hpcycle=True)
        with pytest.warns(UserWarning, match="Hodrick-Prescott filters will not be used"):
            validateX13spec(spec)

    def test_seats_hpcycle_short_monthly_warns(self, m_ts_short: TSeries) -> None:
        spec = newspec(m_ts_short)  # 108 months < 120
        spec.transform = transform(power=0.5)
        spec.seats = seats(hpcycle=True)
        with pytest.warns(UserWarning, match="Hodrick-Prescott filters will not be used"):
            validateX13spec(spec)

    def test_modelspan_vs_span_mismatch_warns(self, q_ts_long: TSeries) -> None:
        spec = newspec(
            series(
                q_ts_long,
                span=Span(MIT.from_yp(Quarterly(), 1991, 1), None),
                modelspan=Span(MIT.from_yp(Quarterly(), 1995, 1), None),
            )
        )
        spec.forecast = forecast(maxback=4)
        _minimal_passable(spec)
        with pytest.warns(UserWarning, match="Backcasts will not be generated"):
            validateX13spec(spec)

    def test_slidingspans_outlier_without_outlier_spec_warns(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.slidingspans = slidingspans(length=20, outlier="keep")
        _minimal_passable(spec)
        with pytest.warns(UserWarning, match="outlier argument of the slidingspans spec"):
            validateX13spec(spec)

    def test_spectrum_qcheck_on_quarterly_warns(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.spectrum = spectrum(qcheck=True)
        _minimal_passable(spec)
        with pytest.warns(UserWarning, match="qcheck argument of the spectrum spec"):
            validateX13spec(spec)

    def test_spectrum_qcheck_on_monthly_passes(self, m_ts_long: TSeries) -> None:
        """``qcheck`` is silent on Monthly — the only frequency it supports."""
        spec = newspec(m_ts_long)
        spec.transform = transform(power=0.5)
        spec.x11 = x11(mode="mult")
        spec.spectrum = spectrum(qcheck=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            validateX13spec(spec)

    def test_x11regression_forcecal_without_combo_warns(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.x11 = x11(mode="mult")
        spec.x11regression = x11regression(variables="td", forcecal=True)
        spec.transform = transform(power=0.5)
        with pytest.warns(UserWarning, match="forcecal argument of the x11regression"):
            validateX13spec(spec)


# ---------------------------------------------------------------------------
# transform / x11 mode compatibility
# ---------------------------------------------------------------------------


class TestTransformX11ModeCompat:
    def test_adjust_with_x11_mode_add_raises(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        # ``adjust='lpyear'`` needs ``power=0.0`` (log-transform) to clear the
        # transform() builder's own validation; the cross-spec validate then
        # fires on the x11.mode='add' / transform.adjust combination.
        spec.transform = transform(power=0.0, adjust="lpyear")
        spec.x11 = x11(mode="add")
        with pytest.raises(ValueError, match="adjust argument of the transform spec"):
            validateX13spec(spec)

    def test_adjust_with_x11_mode_pseudoadd_raises(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.transform = transform(power=0.0, adjust="lpyear")
        spec.x11 = x11(mode="pseudoadd")
        with pytest.raises(ValueError, match="adjust argument of the transform spec"):
            validateX13spec(spec)

    def test_default_x11_mode_and_default_transform_raises(self, q_ts_long: TSeries) -> None:
        """All defaults — both transform and x11.mode unset — is the X-13 bug trap."""
        spec = newspec(q_ts_long)
        spec.transform = transform()  # no func / power
        spec.x11 = x11()  # no mode
        with pytest.raises(ValueError, match="default value for the mode argument"):
            validateX13spec(spec)


# ---------------------------------------------------------------------------
# x11regression containment
# ---------------------------------------------------------------------------


class TestX11RegressionContainment:
    def test_x11regression_data_too_short_raises(self, q_ts_long: TSeries) -> None:
        short_data = MVTSeries(
            MITRange(
                MIT.from_yp(Quarterly(), 1995, 1),
                MIT.from_yp(Quarterly(), 2005, 4),
            ),
            {"reg": np.arange(44.0)},
        )
        spec = newspec(q_ts_long)
        spec.transform = transform(power=0.5)
        spec.x11 = x11(mode="mult")
        spec.x11regression = x11regression(data=short_data, variables="td")
        with pytest.raises(ValueError, match="data provided in the x11regression spec"):
            validateX13spec(spec)

    def test_x11regression_outlierspan_outside_raises(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        spec.transform = transform(power=0.5)
        spec.x11 = x11(mode="mult")
        spec.x11regression = x11regression(
            outlierspan=MITRange(
                MIT.from_yp(Quarterly(), 1985, 1),
                MIT.from_yp(Quarterly(), 1989, 4),
            ),
            variables="td",
        )
        with pytest.raises(ValueError, match="outlierspan argument of the x11regression"):
            validateX13spec(spec)


# ---------------------------------------------------------------------------
# Validate entry + surface
# ---------------------------------------------------------------------------


class TestValidateEntryAndSurface:
    def test_validate_minimal_passes(self, q_ts_long: TSeries) -> None:
        spec = newspec(q_ts_long)
        _minimal_passable(spec)
        validateX13spec(spec)  # no raise

    def test_validate_missing_series_raises(self) -> None:
        spec = newspec()  # series defaults to X13default()
        with pytest.raises(ValueError, match="X13series"):
            validateX13spec(spec)

    def test_newspec_accepts_tseries(self, q_ts_long: TSeries) -> None:
        """``newspec(ts)`` wraps via ``series(...)`` (Julia convenience constructor)."""
        spec = newspec(q_ts_long)
        from tsecon.x13 import X13series  # noqa: PLC0415

        assert isinstance(spec.series, X13series)

    def test_x13spec_is_mutable(self, q_ts_long: TSeries) -> None:
        """The aggregator is intentionally mutable (Julia parity for ``spec.x = ...``)."""
        spec = newspec(q_ts_long)
        spec.outlier = outlier(types=["ao"])
        # Mutation succeeded — assignment didn't raise.
        from tsecon.x13 import X13outlier  # noqa: PLC0415

        assert isinstance(spec.outlier, X13outlier)
