# SPDX-License-Identifier: MIT
"""Polars-native benchmark scenarios for the 4-column comparison harness.

Sibling of ``scenarios_pandas.py``. Each scenario uses the polars idiom a
competent user would write today: ``pl.Series`` for univariate value ops
(mean / std / shift / diff / rolling / pct_change) and ``pl.DataFrame``
with an explicit ``time`` column for ops that need a time axis (joins,
``group_by_dynamic`` resampling).

Polars has no row-index and no period type, so the time axis is always a
``pl.Date`` column representing the *period-start* (Q1 2020 → 2020-01-01,
not 2020-03-31). This is the documented polars convention; it's also the
shape ``tsecon.interop.to_polars`` writes out, so the benchmark exercises
the same wire format users will encounter.

Scenarios intentionally omitted (and therefore reported as ``n/a``):

* kernel-direct ``*_numpy`` / ``*_cython`` rows — these time tsecon's
  internal kernels and have no polars analogue;
* ``workspace_*`` — polars has no Workspace concept;
* ``construct_mvts_*``, ``indexing_mvts_column``, ``cor_mvts_*`` — covered
  by the corresponding ``construct_mvts_qq_100x5`` row (polars stores a
  multivariate panel as a single ``pl.DataFrame``);
* ``undiff_quarterly`` — polars has no inverse of ``.diff()`` and we
  decline to invent one.

The absence of a scenario in this file is itself a paper finding: each
``n/a`` cell is one place where a polars user would have to either drop
out of polars or write the conversion themselves.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Shared shapes — period-start Dates throughout to match polars's
# date-as-the-period-start convention. ``every="1q"`` in group_by_dynamic
# anchors aggregated buckets at the period-start of each quarter.
# ---------------------------------------------------------------------------

_QQ_DATES_100 = pl.date_range(date(2020, 1, 1), date(2044, 10, 1), interval="1q", eager=True)
_QQ_DATES_100_LATER = pl.date_range(date(2032, 1, 1), date(2056, 10, 1), interval="1q", eager=True)
_MM_DATES_300 = pl.date_range(date(2020, 1, 1), date(2044, 12, 1), interval="1mo", eager=True)
_MM_DATES_120 = pl.date_range(date(2020, 1, 1), date(2029, 12, 1), interval="1mo", eager=True)
_YY_DATES_25 = pl.date_range(date(2020, 1, 1), date(2044, 1, 1), interval="1y", eager=True)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _setup_construct_tseries_qq_100() -> dict[str, Any]:
    return {"dates": _QQ_DATES_100, "values": np.arange(100, dtype=np.float64)}


def _run_construct_tseries_qq_100(state: dict[str, Any]) -> pl.DataFrame:
    return pl.DataFrame({"time": state["dates"], "value": state["values"]})


def _setup_construct_mvts_qq_100x5() -> dict[str, Any]:
    arr = np.arange(500, dtype=np.float64).reshape(100, 5)
    return {
        "dates": _QQ_DATES_100,
        "cols": {"a": arr[:, 0], "b": arr[:, 1], "c": arr[:, 2], "d": arr[:, 3], "e": arr[:, 4]},
    }


def _run_construct_mvts_qq_100x5(state: dict[str, Any]) -> pl.DataFrame:
    return pl.DataFrame({"time": state["dates"], **state["cols"]})


# ---------------------------------------------------------------------------
# Indexing — sum 100 ``.filter(time == k)`` reads over a DataFrame. This is
# the polars-natural idiom for per-period lookup; there is no row-label
# accessor analogous to pandas's ``.loc``. The cost will be high; that's
# the point — polars filters scan the whole column for each lookup.
# ---------------------------------------------------------------------------


def _setup_indexing_mit_lookup_100() -> dict[str, Any]:
    df = pl.DataFrame({"time": _QQ_DATES_100, "value": np.arange(100, dtype=np.float64)})
    keys = list(_QQ_DATES_100.to_list())
    return {"df": df, "keys": keys}


def _run_indexing_mit_lookup_100(state: dict[str, Any]) -> float:
    df = state["df"]
    total = 0.0
    for k in state["keys"]:
        total += float(df.filter(pl.col("time") == k)["value"][0])
    return total


# ---------------------------------------------------------------------------
# Arithmetic — misaligned addition forces an explicit outer join in polars
# (no row-index alignment to fall back on). The result DataFrame has 148
# rows (100 + 100 - 52 overlap), same as pandas's auto-aligned Series.
# ---------------------------------------------------------------------------


def _setup_arith_add_misaligned() -> dict[str, Any]:
    a = pl.DataFrame({"time": _QQ_DATES_100, "a": np.arange(100, dtype=np.float64)})
    b = pl.DataFrame({"time": _QQ_DATES_100_LATER, "b": np.arange(100, dtype=np.float64) * 0.5})
    return {"a": a, "b": b}


def _run_arith_add_misaligned(state: dict[str, Any]) -> pl.DataFrame:
    return (
        state["a"]
        .join(state["b"], on="time", how="full", coalesce=True)
        .with_columns((pl.col("a").fill_null(0.0) + pl.col("b").fill_null(0.0)).alias("sum"))
        .sort("time")
    )


def _setup_arith_add_aligned() -> dict[str, Any]:
    a = pl.Series("a", np.arange(100, dtype=np.float64))
    b = pl.Series("b", np.arange(100, dtype=np.float64) * 0.5)
    return {"a": a, "b": b}


def _run_arith_add_aligned(state: dict[str, Any]) -> pl.Series:
    return state["a"] + state["b"]


def _setup_arith_mul_scalar() -> dict[str, Any]:
    return {"s": pl.Series("v", np.arange(100, dtype=np.float64))}


def _run_arith_mul_scalar(state: dict[str, Any]) -> pl.Series:
    return state["s"] * 2.5


# ---------------------------------------------------------------------------
# Shift / diff / pct_change — polars Series carries the same surface as
# pandas Series for these ops, in the same order, with the same defaults.
# ---------------------------------------------------------------------------


def _setup_shift_quarterly_lag1() -> dict[str, Any]:
    return {"s": pl.Series("v", np.arange(100, dtype=np.float64))}


def _run_shift_quarterly_lag1(state: dict[str, Any]) -> pl.Series:
    return state["s"].shift(-1)


def _setup_lead_quarterly_lag1() -> dict[str, Any]:
    return {"s": pl.Series("v", np.arange(100, dtype=np.float64))}


def _run_lead_quarterly_lag1(state: dict[str, Any]) -> pl.Series:
    # tsecon's lead(t, 1) == shift(t, +1); polars's analogue is .shift(+1).
    return state["s"].shift(1)


def _setup_diff_quarterly() -> dict[str, Any]:
    return {"s": pl.Series("v", np.arange(100, dtype=np.float64))}


def _run_diff_quarterly(state: dict[str, Any]) -> pl.Series:
    return state["s"].diff()


def _setup_pct_quarterly() -> dict[str, Any]:
    return {"s": pl.Series("v", np.arange(1.0, 101.0))}


def _run_pct_quarterly(state: dict[str, Any]) -> pl.Series:
    return state["s"].pct_change()


def _setup_ytypct_quarterly_100() -> dict[str, Any]:
    return {"s": pl.Series("v", np.arange(1.0, 101.0))}


def _run_ytypct_quarterly_100(state: dict[str, Any]) -> pl.Series:
    # Quarterly year-on-year: ppy=4, mirrors tsecon's ytypct.
    return state["s"].pct_change(n=4) * 100.0


# ---------------------------------------------------------------------------
# Stats — mean / std / cor.
# ---------------------------------------------------------------------------


def _setup_mean_quarterly_100() -> dict[str, Any]:
    return {"s": pl.Series("v", np.arange(100, dtype=np.float64))}


def _run_mean_quarterly_100(state: dict[str, Any]) -> float:
    return float(state["s"].mean())  # type: ignore[arg-type]


def _setup_std_quarterly_100() -> dict[str, Any]:
    return {"s": pl.Series("v", np.arange(100, dtype=np.float64))}


def _run_std_quarterly_100(state: dict[str, Any]) -> float:
    return float(state["s"].std())  # type: ignore[arg-type]


def _setup_cor_two_tseries() -> dict[str, Any]:
    rng = np.random.default_rng(seed=20260515)
    return {
        "a": pl.Series("a", rng.standard_normal(100)),
        "b": pl.Series("b", rng.standard_normal(100)),
    }


def _run_cor_two_tseries(state: dict[str, Any]) -> float:
    df = pl.DataFrame({"a": state["a"], "b": state["b"]})
    return float(df.select(pl.corr("a", "b")).item())


def _setup_cov_two_tseries() -> dict[str, Any]:
    rng = np.random.default_rng(seed=20260515)
    return {
        "a": pl.Series("a", rng.standard_normal(100)),
        "b": pl.Series("b", rng.standard_normal(100)),
    }


def _run_cov_two_tseries(state: dict[str, Any]) -> float:
    df = pl.DataFrame({"a": state["a"], "b": state["b"]})
    return float(df.select(pl.cov("a", "b")).item())


def _setup_quantile_quarterly_100() -> dict[str, Any]:
    rng = np.random.default_rng(seed=20260515)
    return {"s": pl.Series("v", rng.standard_normal(100))}


def _run_quantile_quarterly_100(state: dict[str, Any]) -> float:
    return float(state["s"].quantile(0.5))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Moving — ``.rolling_mean(N)`` / ``.rolling_sum(N)``. These are the
# polars Series methods (not the DataFrame ``.rolling(...)`` window
# context, which is heavier).
# ---------------------------------------------------------------------------


def _setup_moving_average_quarterly_4() -> dict[str, Any]:
    return {"s": pl.Series("v", np.arange(100, dtype=np.float64))}


def _run_moving_average_quarterly_4(state: dict[str, Any]) -> pl.Series:
    return state["s"].rolling_mean(4)


def _setup_moving_sum_quarterly_4() -> dict[str, Any]:
    return {"s": pl.Series("v", np.arange(100, dtype=np.float64))}


def _run_moving_sum_quarterly_4(state: dict[str, Any]) -> pl.Series:
    return state["s"].rolling_sum(4)


# ---------------------------------------------------------------------------
# Frequency conversion — polars uses ``group_by_dynamic`` for the
# lower-frequency direction (aggregation) and ``upsample`` for the
# higher-frequency direction. Neither carries a period concept; we name
# the bucket size in calendar terms (``every="1q"`` / ``every="1y"``).
# ---------------------------------------------------------------------------


def _setup_fconvert_qq_to_yy_mean() -> dict[str, Any]:
    return {"df": pl.DataFrame({"time": _QQ_DATES_100, "value": np.arange(100, dtype=np.float64)})}


def _run_fconvert_qq_to_yy_mean(state: dict[str, Any]) -> pl.DataFrame:
    return state["df"].group_by_dynamic("time", every="1y").agg(pl.col("value").mean())


def _setup_fconvert_qq_to_yy_sum() -> dict[str, Any]:
    return {"df": pl.DataFrame({"time": _QQ_DATES_100, "value": np.arange(100, dtype=np.float64)})}


def _run_fconvert_qq_to_yy_sum(state: dict[str, Any]) -> pl.DataFrame:
    return state["df"].group_by_dynamic("time", every="1y").agg(pl.col("value").sum())


def _setup_fconvert_yy_to_qq_const() -> dict[str, Any]:
    return {"df": pl.DataFrame({"time": _YY_DATES_25, "value": np.arange(25, dtype=np.float64)})}


def _run_fconvert_yy_to_qq_const(state: dict[str, Any]) -> pl.DataFrame:
    return state["df"].upsample("time", every="1q").fill_null(strategy="forward")


def _setup_fconvert_yy_to_qq_linear() -> dict[str, Any]:
    return {"df": pl.DataFrame({"time": _YY_DATES_25, "value": np.arange(25, dtype=np.float64)})}


def _run_fconvert_yy_to_qq_linear(state: dict[str, Any]) -> pl.DataFrame:
    # Polars upsample produces null gaps between yearly anchors; linear
    # interpolation is the standard fill, mirroring tsecon's method="linear".
    return state["df"].upsample("time", every="1q").interpolate()


def _setup_fconvert_yy_to_qq_even() -> dict[str, Any]:
    return {"df": pl.DataFrame({"time": _YY_DATES_25, "value": np.arange(25, dtype=np.float64)})}


def _run_fconvert_yy_to_qq_even(state: dict[str, Any]) -> pl.DataFrame:
    # tsecon's "even" divides each yearly value across its 4 quarters; the
    # closest natural polars idiom is forward-fill then divide-by-ppy.
    return (
        state["df"]
        .upsample("time", every="1q")
        .fill_null(strategy="forward")
        .with_columns((pl.col("value") / 4.0).alias("value"))
    )


def _setup_fconvert_mm_to_qq_mean() -> dict[str, Any]:
    return {"df": pl.DataFrame({"time": _MM_DATES_120, "value": np.arange(120, dtype=np.float64)})}


def _run_fconvert_mm_to_qq_mean(state: dict[str, Any]) -> pl.DataFrame:
    return state["df"].group_by_dynamic("time", every="1q").agg(pl.col("value").mean())


# ---------------------------------------------------------------------------
# Recursion — AR(2) over 100 quarters. Polars has no in-place per-row
# write idiom; the documented "recurrence" pattern is to drop to numpy
# and loop there (which is the same kernel the tsecon
# ``rec_linear_ar2_100_numpy`` scenario already measures). We mirror that
# pattern here so the cost is honest about *what polars-native code
# costs*. The ``.values()``-style pull and loop is the polars community's
# standard answer for non-vectorisable sequential dependencies.
# ---------------------------------------------------------------------------


def _setup_rec_ar2_100() -> dict[str, Any]:
    s = pl.Series("v", np.zeros(102, dtype=np.float64))
    return {"s": s}


def _run_rec_ar2_100(state: dict[str, Any]) -> pl.Series:
    values = state["s"].to_numpy().copy()
    values[0] = 1.0
    values[1] = 1.0
    for i in range(2, 102):
        values[i] = 0.5 * values[i - 1] + 0.3 * values[i - 2]
    return pl.Series("v", values)


# ---------------------------------------------------------------------------
# Mixed-frequency demonstrators — the headline rows. Polars has no
# period-aware arithmetic, so each conversion is an explicit
# ``group_by_dynamic`` / ``upsample`` step followed by a ``.join`` on the
# time column. The pipeline reads as more LOC than the pandas / tsecon
# counterparts; the harness measures whether the polars implementation
# (Rust core, vectorised joins) makes up the verbosity in speed.
# ---------------------------------------------------------------------------


def _setup_mixed_freq_qq_minus_mm_mean() -> dict[str, Any]:
    return {
        "gdp": pl.DataFrame({"time": _QQ_DATES_100, "gdp": np.arange(100, dtype=np.float64)}),
        "cpi": pl.DataFrame({"time": _MM_DATES_300, "cpi": np.arange(300, dtype=np.float64)}),
    }


def _run_mixed_freq_qq_minus_mm_mean(state: dict[str, Any]) -> pl.DataFrame:
    cpi_q = state["cpi"].group_by_dynamic("time", every="1q").agg(pl.col("cpi").mean())
    return (
        state["gdp"]
        .join(cpi_q, on="time", how="left")
        .with_columns((pl.col("gdp") - pl.col("cpi")).alias("diff"))
    )


def _setup_mixed_freq_pipeline_three_freq() -> dict[str, Any]:
    return {
        "unemp": pl.DataFrame({"time": _YY_DATES_25, "unemp": np.arange(25, dtype=np.float64)}),
        "gdp": pl.DataFrame({"time": _QQ_DATES_100, "gdp": np.arange(100, dtype=np.float64)}),
        "cpi": pl.DataFrame({"time": _MM_DATES_300, "cpi": np.arange(300, dtype=np.float64)}),
    }


def _run_mixed_freq_pipeline_three_freq(state: dict[str, Any]) -> pl.DataFrame:
    cpi_q = state["cpi"].group_by_dynamic("time", every="1q").agg(pl.col("cpi").mean())
    unemp_q = state["unemp"].upsample("time", every="1q").fill_null(strategy="forward")
    return (
        state["gdp"]
        .join(unemp_q, on="time", how="left")
        .join(cpi_q, on="time", how="left")
        .with_columns((pl.col("gdp") + pl.col("unemp") + pl.col("cpi")).alias("sum"))
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SETUP: dict[str, Callable[[], Any]] = {
    "construct_tseries_qq_100": _setup_construct_tseries_qq_100,
    "construct_mvts_qq_100x5": _setup_construct_mvts_qq_100x5,
    "indexing_mit_lookup_100": _setup_indexing_mit_lookup_100,
    "arith_add_misaligned": _setup_arith_add_misaligned,
    "arith_add_aligned": _setup_arith_add_aligned,
    "arith_mul_scalar": _setup_arith_mul_scalar,
    "shift_quarterly_lag1": _setup_shift_quarterly_lag1,
    "lead_quarterly_lag1": _setup_lead_quarterly_lag1,
    "diff_quarterly": _setup_diff_quarterly,
    "pct_quarterly": _setup_pct_quarterly,
    "ytypct_quarterly_100": _setup_ytypct_quarterly_100,
    "mean_quarterly_100": _setup_mean_quarterly_100,
    "std_quarterly_100": _setup_std_quarterly_100,
    "quantile_quarterly_100": _setup_quantile_quarterly_100,
    "cor_two_tseries": _setup_cor_two_tseries,
    "cov_two_tseries": _setup_cov_two_tseries,
    "moving_average_quarterly_4": _setup_moving_average_quarterly_4,
    "moving_sum_quarterly_4": _setup_moving_sum_quarterly_4,
    "fconvert_qq_to_yy_mean": _setup_fconvert_qq_to_yy_mean,
    "fconvert_qq_to_yy_sum": _setup_fconvert_qq_to_yy_sum,
    "fconvert_yy_to_qq_const": _setup_fconvert_yy_to_qq_const,
    "fconvert_yy_to_qq_linear": _setup_fconvert_yy_to_qq_linear,
    "fconvert_yy_to_qq_even": _setup_fconvert_yy_to_qq_even,
    "fconvert_mm_to_qq_mean": _setup_fconvert_mm_to_qq_mean,
    "rec_ar2_100": _setup_rec_ar2_100,
    "mixed_freq_qq_minus_mm_mean": _setup_mixed_freq_qq_minus_mm_mean,
    "mixed_freq_pipeline_three_freq": _setup_mixed_freq_pipeline_three_freq,
}

RUN: dict[str, Callable[[Any], Any]] = {
    "construct_tseries_qq_100": _run_construct_tseries_qq_100,
    "construct_mvts_qq_100x5": _run_construct_mvts_qq_100x5,
    "indexing_mit_lookup_100": _run_indexing_mit_lookup_100,
    "arith_add_misaligned": _run_arith_add_misaligned,
    "arith_add_aligned": _run_arith_add_aligned,
    "arith_mul_scalar": _run_arith_mul_scalar,
    "shift_quarterly_lag1": _run_shift_quarterly_lag1,
    "lead_quarterly_lag1": _run_lead_quarterly_lag1,
    "diff_quarterly": _run_diff_quarterly,
    "pct_quarterly": _run_pct_quarterly,
    "ytypct_quarterly_100": _run_ytypct_quarterly_100,
    "mean_quarterly_100": _run_mean_quarterly_100,
    "std_quarterly_100": _run_std_quarterly_100,
    "quantile_quarterly_100": _run_quantile_quarterly_100,
    "cor_two_tseries": _run_cor_two_tseries,
    "cov_two_tseries": _run_cov_two_tseries,
    "moving_average_quarterly_4": _run_moving_average_quarterly_4,
    "moving_sum_quarterly_4": _run_moving_sum_quarterly_4,
    "fconvert_qq_to_yy_mean": _run_fconvert_qq_to_yy_mean,
    "fconvert_qq_to_yy_sum": _run_fconvert_qq_to_yy_sum,
    "fconvert_yy_to_qq_const": _run_fconvert_yy_to_qq_const,
    "fconvert_yy_to_qq_linear": _run_fconvert_yy_to_qq_linear,
    "fconvert_yy_to_qq_even": _run_fconvert_yy_to_qq_even,
    "fconvert_mm_to_qq_mean": _run_fconvert_mm_to_qq_mean,
    "rec_ar2_100": _run_rec_ar2_100,
    "mixed_freq_qq_minus_mm_mean": _run_mixed_freq_qq_minus_mm_mean,
    "mixed_freq_pipeline_three_freq": _run_mixed_freq_pipeline_three_freq,
}

assert SETUP.keys() == RUN.keys(), "polars scenario registries must agree"
