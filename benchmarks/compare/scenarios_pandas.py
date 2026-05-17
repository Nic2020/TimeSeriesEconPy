# SPDX-License-Identifier: MIT
"""Pandas-native benchmark scenarios for the 4-column comparison harness.

Mirrors a paper-strategic subset of the scenarios in ``scenarios.py``,
re-expressed as a competent pandas user would write them today. Each
scenario uses the natural pandas idiom — ``pd.Series`` with ``PeriodIndex``
for quarterly / monthly / yearly data, ``.shift`` / ``.diff`` / ``.rolling``
for the obvious ops, ``.resample`` for frequency conversion, and an
``iloc`` Python-loop for the AR(2) recurrence (the polars sibling does the
same thing; the recurrence is the JSS Section 5 exhibit for "tabular
abstractions do not help with this operation").

Scenarios intentionally omitted from this file (and therefore reported as
``n/a`` by the harness):

* kernel-direct ``*_numpy`` / ``*_cython`` rows — these time tsecon's
  internal kernels and have no pandas analogue;
* ``workspace_*`` — pandas has no Workspace concept;
* ``construct_mvts_*``, ``indexing_mvts_column``, ``cor_mvts_*`` — the
  multivariate equivalent is a ``pd.DataFrame``; the relevant rows here
  are the univariate ones plus ``construct_df_qq_100x5`` keyed under the
  same scenario name (``construct_mvts_qq_100x5``);
* ``undiff_quarterly`` — pandas has no inverse of ``.diff()`` and we
  decline to invent one;
* ``cor_two_tseries_*`` cython rows — covered by ``cor_two_tseries``.

The harness joins on scenario name across this file, ``scenarios.py``,
``scenarios_polars.py``, and ``julia/scenarios.jl``. A missing key on this
side becomes a literal ``n/a`` cell — that absence is itself a paper
finding, not a gap.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared shapes — keep these aligned with the tsecon-side setups so the
# four columns measure semantically-equivalent work. Period anchors:
# Q-DEC = calendar Q1 ends March (tsecon Quarterly(end_month=3) ≡ pandas
# Q-DEC, mirroring the rule baked into tsecon/interop/pandas.py).
# ---------------------------------------------------------------------------

_QQ_IDX_100 = pd.period_range(start="2020Q1", periods=100, freq="Q-DEC")
_QQ_IDX_100_LATER = pd.period_range(start="2032Q1", periods=100, freq="Q-DEC")
_MM_IDX_300 = pd.period_range(start="2020-01", periods=300, freq="M")
_YY_IDX_25 = pd.period_range(start="2020", periods=25, freq="Y-DEC")


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _setup_construct_tseries_qq_100() -> dict[str, Any]:
    return {"idx": _QQ_IDX_100, "values": np.arange(100, dtype=np.float64)}


def _run_construct_tseries_qq_100(state: dict[str, Any]) -> pd.Series:
    return pd.Series(state["values"], index=state["idx"])


def _setup_construct_mvts_qq_100x5() -> dict[str, Any]:
    return {
        "idx": _QQ_IDX_100,
        "cols": ["a", "b", "c", "d", "e"],
        "values": np.arange(500, dtype=np.float64).reshape(100, 5),
    }


def _run_construct_mvts_qq_100x5(state: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(state["values"], index=state["idx"], columns=state["cols"])


# ---------------------------------------------------------------------------
# Indexing — sum 100 .loc[period] reads. Mirrors the tsecon MIT-lookup
# scenario; what the harness measures is the cost of pandas's per-period
# attribute access against PeriodIndex.
# ---------------------------------------------------------------------------


def _setup_indexing_mit_lookup_100() -> dict[str, Any]:
    s = pd.Series(np.arange(100, dtype=np.float64), index=_QQ_IDX_100)
    keys = list(_QQ_IDX_100)
    return {"s": s, "keys": keys}


def _run_indexing_mit_lookup_100(state: dict[str, Any]) -> float:
    s = state["s"]
    total = 0.0
    for k in state["keys"]:
        total += float(s.loc[k])
    return total


# ---------------------------------------------------------------------------
# Arithmetic — index alignment is the headline pandas feature here. ``s + s2``
# performs an outer-index union and NaN-pads the gap; the resulting Series
# is length 148 (100 + 100 - 52 overlap).
# ---------------------------------------------------------------------------


def _setup_arith_add_misaligned() -> dict[str, Any]:
    a = pd.Series(np.arange(100, dtype=np.float64), index=_QQ_IDX_100)
    b = pd.Series(np.arange(100, dtype=np.float64) * 0.5, index=_QQ_IDX_100_LATER)
    return {"a": a, "b": b}


def _run_arith_add_misaligned(state: dict[str, Any]) -> pd.Series:
    return state["a"] + state["b"]


def _setup_arith_add_aligned() -> dict[str, Any]:
    a = pd.Series(np.arange(100, dtype=np.float64), index=_QQ_IDX_100)
    b = pd.Series(np.arange(100, dtype=np.float64) * 0.5, index=_QQ_IDX_100)
    return {"a": a, "b": b}


def _run_arith_add_aligned(state: dict[str, Any]) -> pd.Series:
    return state["a"] + state["b"]


def _setup_arith_mul_scalar() -> dict[str, Any]:
    return {"s": pd.Series(np.arange(100, dtype=np.float64), index=_QQ_IDX_100)}


def _run_arith_mul_scalar(state: dict[str, Any]) -> pd.Series:
    return state["s"] * 2.5


# ---------------------------------------------------------------------------
# Shift / diff / pct_change — direct pandas methods. pct_change starts at
# 1.0 to avoid divide-by-zero on the first ratio (mirrors the tsecon
# scenario which starts at 1.0 to dodge the same RuntimeWarning).
# ---------------------------------------------------------------------------


def _setup_shift_quarterly_lag1() -> dict[str, Any]:
    return {"s": pd.Series(np.arange(100, dtype=np.float64), index=_QQ_IDX_100)}


def _run_shift_quarterly_lag1(state: dict[str, Any]) -> pd.Series:
    return state["s"].shift(-1)


def _setup_lead_quarterly_lag1() -> dict[str, Any]:
    return {"s": pd.Series(np.arange(100, dtype=np.float64), index=_QQ_IDX_100)}


def _run_lead_quarterly_lag1(state: dict[str, Any]) -> pd.Series:
    # tsecon's lead(t, 1) == shift(t, +1); pandas's analogue is .shift(+1).
    return state["s"].shift(1)


def _setup_diff_quarterly() -> dict[str, Any]:
    return {"s": pd.Series(np.arange(100, dtype=np.float64), index=_QQ_IDX_100)}


def _run_diff_quarterly(state: dict[str, Any]) -> pd.Series:
    return state["s"].diff()


def _setup_pct_quarterly() -> dict[str, Any]:
    return {"s": pd.Series(np.arange(1.0, 101.0), index=_QQ_IDX_100)}


def _run_pct_quarterly(state: dict[str, Any]) -> pd.Series:
    return state["s"].pct_change()


def _setup_ytypct_quarterly_100() -> dict[str, Any]:
    # Quarterly year-on-year: ppy(Quarterly)=4, mirrors tsecon's ytypct.
    return {"s": pd.Series(np.arange(1.0, 101.0), index=_QQ_IDX_100)}


def _run_ytypct_quarterly_100(state: dict[str, Any]) -> pd.Series:
    return state["s"].pct_change(periods=4) * 100.0


# ---------------------------------------------------------------------------
# Stats — mean / std / cor. ``.corr()`` on two same-length Series is the
# direct counterpart of tsecon's ``cor(a, b)``.
# ---------------------------------------------------------------------------


def _setup_mean_quarterly_100() -> dict[str, Any]:
    return {"s": pd.Series(np.arange(100, dtype=np.float64), index=_QQ_IDX_100)}


def _run_mean_quarterly_100(state: dict[str, Any]) -> float:
    return float(state["s"].mean())


def _setup_std_quarterly_100() -> dict[str, Any]:
    return {"s": pd.Series(np.arange(100, dtype=np.float64), index=_QQ_IDX_100)}


def _run_std_quarterly_100(state: dict[str, Any]) -> float:
    return float(state["s"].std())


def _setup_cor_two_tseries() -> dict[str, Any]:
    rng = np.random.default_rng(seed=20260515)
    a = pd.Series(rng.standard_normal(100), index=_QQ_IDX_100)
    b = pd.Series(rng.standard_normal(100), index=_QQ_IDX_100)
    return {"a": a, "b": b}


def _run_cor_two_tseries(state: dict[str, Any]) -> float:
    return float(state["a"].corr(state["b"]))


def _setup_cov_two_tseries() -> dict[str, Any]:
    rng = np.random.default_rng(seed=20260515)
    a = pd.Series(rng.standard_normal(100), index=_QQ_IDX_100)
    b = pd.Series(rng.standard_normal(100), index=_QQ_IDX_100)
    return {"a": a, "b": b}


def _run_cov_two_tseries(state: dict[str, Any]) -> float:
    return float(state["a"].cov(state["b"]))


def _setup_quantile_quarterly_100() -> dict[str, Any]:
    rng = np.random.default_rng(seed=20260515)
    return {"s": pd.Series(rng.standard_normal(100), index=_QQ_IDX_100)}


def _run_quantile_quarterly_100(state: dict[str, Any]) -> float:
    return float(state["s"].quantile(0.5))


# ---------------------------------------------------------------------------
# Moving — rolling-window reductions. ``.rolling(N).mean()`` is the
# canonical pandas idiom; internally Cythonised in pandas's own extension
# module, so this should land close to tsecon's NumPy ``cumsum`` form.
# ---------------------------------------------------------------------------


def _setup_moving_average_quarterly_4() -> dict[str, Any]:
    return {"s": pd.Series(np.arange(100, dtype=np.float64), index=_QQ_IDX_100)}


def _run_moving_average_quarterly_4(state: dict[str, Any]) -> pd.Series:
    return state["s"].rolling(4).mean()


def _setup_moving_sum_quarterly_4() -> dict[str, Any]:
    return {"s": pd.Series(np.arange(100, dtype=np.float64), index=_QQ_IDX_100)}


def _run_moving_sum_quarterly_4(state: dict[str, Any]) -> pd.Series:
    return state["s"].rolling(4).sum()


# ---------------------------------------------------------------------------
# Frequency conversion — pandas's ``.resample(...)`` is the closest match
# to tsecon's ``fconvert``. Lower-frequency direction uses an aggregator;
# higher-frequency direction uses ``.ffill()`` / ``.asfreq()`` to broadcast.
# Q-DEC = calendar quarters; Y-DEC = calendar years.
# ---------------------------------------------------------------------------


def _setup_fconvert_qq_to_yy_mean() -> dict[str, Any]:
    return {"s": pd.Series(np.arange(100, dtype=np.float64), index=_QQ_IDX_100)}


def _run_fconvert_qq_to_yy_mean(state: dict[str, Any]) -> pd.Series:
    return state["s"].resample("Y-DEC").mean()


def _setup_fconvert_qq_to_yy_sum() -> dict[str, Any]:
    return {"s": pd.Series(np.arange(100, dtype=np.float64), index=_QQ_IDX_100)}


def _run_fconvert_qq_to_yy_sum(state: dict[str, Any]) -> pd.Series:
    return state["s"].resample("Y-DEC").sum()


def _setup_fconvert_yy_to_qq_const() -> dict[str, Any]:
    return {"s": pd.Series(np.arange(25, dtype=np.float64), index=_YY_IDX_25)}


def _run_fconvert_yy_to_qq_const(state: dict[str, Any]) -> pd.Series:
    return state["s"].resample("Q-DEC").ffill()


def _setup_fconvert_yy_to_qq_linear() -> dict[str, Any]:
    return {"s": pd.Series(np.arange(25, dtype=np.float64), index=_YY_IDX_25)}


def _run_fconvert_yy_to_qq_linear(state: dict[str, Any]) -> pd.Series:
    # Mirrors tsecon's higher-freq linear method: upsample then linearly
    # interpolate between yearly anchors.
    return state["s"].resample("Q-DEC").interpolate(method="linear")


def _setup_fconvert_yy_to_qq_even() -> dict[str, Any]:
    return {"s": pd.Series(np.arange(25, dtype=np.float64), index=_YY_IDX_25)}


def _run_fconvert_yy_to_qq_even(state: dict[str, Any]) -> pd.Series:
    # tsecon's "even" divides each yearly value across its 4 quarters; the
    # closest natural pandas idiom is ffill then divide-by-ppy.
    return state["s"].resample("Q-DEC").ffill() / 4.0


def _setup_fconvert_mm_to_qq_mean() -> dict[str, Any]:
    idx = pd.period_range("2020-01", periods=120, freq="M")
    return {"s": pd.Series(np.arange(120, dtype=np.float64), index=idx)}


def _run_fconvert_mm_to_qq_mean(state: dict[str, Any]) -> pd.Series:
    return state["s"].resample("Q-DEC").mean()


# ---------------------------------------------------------------------------
# Recursion — AR(2) over 100 quarters via ``.iloc`` loop. This is the
# pandas-native form a competent user would write; the alternative
# ("drop to .values and loop in numpy") leaks out of pandas entirely and
# is already measured under ``rec_linear_ar2_100_numpy`` on the tsecon
# side. The point of this row is to show what staying *inside* pandas
# costs for a sequentially-dependent operation.
# ---------------------------------------------------------------------------


def _setup_rec_ar2_100() -> dict[str, Any]:
    idx = pd.period_range(start="2020Q1", periods=102, freq="Q-DEC")
    target = pd.Series(np.zeros(102), index=idx)
    target.iloc[0] = 1.0
    target.iloc[1] = 1.0
    return {"target": target}


def _run_rec_ar2_100(state: dict[str, Any]) -> pd.Series:
    target = state["target"]
    for i in range(2, 102):
        target.iloc[i] = 0.5 * target.iloc[i - 1] + 0.3 * target.iloc[i - 2]
    return target


# ---------------------------------------------------------------------------
# Mixed-frequency demonstrators — these are the paper-headline rows
# because they show the cost of the conversion step the user has to write
# explicitly. tsecon hides the alignment cost behind ``fconvert`` + the
# usual arithmetic; pandas hides it behind ``.resample(...).mean()`` + the
# usual arithmetic. The two paths look very similar in source; the
# benchmark surfaces what they actually cost.
# ---------------------------------------------------------------------------


def _setup_mixed_freq_qq_minus_mm_mean() -> dict[str, Any]:
    return {
        "gdp": pd.Series(np.arange(100, dtype=np.float64), index=_QQ_IDX_100),
        "cpi": pd.Series(np.arange(300, dtype=np.float64), index=_MM_IDX_300),
    }


def _run_mixed_freq_qq_minus_mm_mean(state: dict[str, Any]) -> pd.Series:
    return state["gdp"] - state["cpi"].resample("Q-DEC").mean()


def _setup_mixed_freq_pipeline_three_freq() -> dict[str, Any]:
    return {
        "unemp": pd.Series(np.arange(25, dtype=np.float64), index=_YY_IDX_25),
        "gdp": pd.Series(np.arange(100, dtype=np.float64), index=_QQ_IDX_100),
        "cpi": pd.Series(np.arange(300, dtype=np.float64), index=_MM_IDX_300),
    }


def _run_mixed_freq_pipeline_three_freq(state: dict[str, Any]) -> pd.Series:
    return (
        state["unemp"].resample("Q-DEC").ffill()
        + state["gdp"]
        + state["cpi"].resample("Q-DEC").mean()
    )


# ---------------------------------------------------------------------------
# reindex — only the label-shift operation from `various.jl` has a clean
# pandas analogue. `overlay` and `compare` get no row here on purpose;
# pandas has neither a recursive Workspace structure nor a "first-non-NaN
# wins, range-union" merge primitive. The closest cousins —
# ``pd.Series.combine_first`` for overlay-of-two and
# ``pd.testing.assert_series_equal`` for compare — diverge enough in
# semantics that a side-by-side row would mislead more than inform.
# ---------------------------------------------------------------------------


def _setup_reindex_tseries_100() -> dict[str, Any]:
    return {
        "t": pd.Series(np.arange(100, dtype=np.float64), index=_QQ_IDX_100),
        # New labels = original index shifted; PeriodIndex doesn't relabel
        # to Unit, so the closest pandas form is "give it a fresh index with
        # the same length". ``set_axis(copy=False)`` is the no-allocation path.
        "new_idx": pd.RangeIndex(start=1, stop=101),
    }


def _run_reindex_tseries_100(state: dict[str, Any]) -> pd.Series:
    return state["t"].set_axis(state["new_idx"], copy=False)


# ---------------------------------------------------------------------------
# Registry — same shape as scenarios.py. Only scenarios with a natural
# pandas form appear; the rest are intentionally absent and become ``n/a``
# cells in the comparison table.
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
    "reindex_tseries_100": _setup_reindex_tseries_100,
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
    "reindex_tseries_100": _run_reindex_tseries_100,
}

assert SETUP.keys() == RUN.keys(), "pandas scenario registries must agree"
