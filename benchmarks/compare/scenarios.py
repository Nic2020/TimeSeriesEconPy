# SPDX-License-Identifier: MIT
"""Benchmark scenarios for the Julia ↔ Python comparison harness.

Each scenario exposes:

* ``SETUP[name]() -> state`` — builds whatever fixed inputs the run needs.
  The setup cost is **not** measured; only ``RUN[name](state)`` is timed.
* ``RUN[name](state) -> result`` — the operation being benchmarked. The return
  value exists only to defeat dead-code elimination; the driver discards it.

The matching Julia implementations live in ``julia/scenarios.jl``; the two
files are intentionally kept side-by-side so a reviewer can diff them.

The scenarios cover representative operations across the M1 surface, plus
the M1.5 three-flavor kernel pair (see decision 17):

================================  ==============================================
``construct_tseries_qq_100``      ``TSeries(qq(2020,1), arr)`` from a length-100 array.
``indexing_mit_lookup_100``       Read all 100 MIT positions and sum.
``arith_add_misaligned``          Add two 100-period TSeries with a 50-period overlap.
``shift_quarterly_lag1``          ``shift(t, -1)`` over a 100-period quarterly TSeries.
``moving_average_quarterly_4``    4-period moving average over the same input.
``fconvert_qq_to_yy_mean``        Quarterly → Yearly with ``method="mean"``.
``rec_ar2_100``                   100-step AR(2) via general ``rec`` + lambda.
``rec_linear_ar2_100_numpy``      Same AR(2) via the NumPy kernel direct.
``rec_linear_ar2_100_cython``     Same AR(2) via the Cython kernel direct (registered
                                  only when the compiled extension is importable).
``workspace_merge_5_series``      Merge two Workspaces each holding five TSeries.
================================  ==============================================

The Python entries here import only the public ``tsecon`` surface (plus the
two kernel modules, addressed directly to skip dispatch overhead) — the
benchmark must exercise the same API users will. The Julia counterpart only
imports the upstream ``TimeSeriesEcon`` package.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from tsecon import (
    MITRange,
    MVTSeries,
    Quarterly,
    TSeries,
    Workspace,
    Yearly,
    cor,
    cov,
    diff,
    fconvert,
    mean,
    mm,
    moving_average,
    moving_sum,
    pct,
    qq,
    rec,
    shift,
    std,
    undiff,
    yy,
)
from tsecon._rec_kernels import rec_linear_numpy

try:
    from tsecon._rec_kernels_cy import rec_linear_cython  # type: ignore[import-not-found]

    _CYTHON_KERNEL_AVAILABLE = True
except ImportError:  # pragma: no cover — depends on wheel-build state
    _CYTHON_KERNEL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _setup_construct_tseries_qq_100() -> dict[str, Any]:
    return {"start": qq(2020, 1), "values": np.arange(100, dtype=np.float64)}


def _run_construct_tseries_qq_100(state: dict[str, Any]) -> TSeries:
    return TSeries(state["start"], state["values"])


# ---------------------------------------------------------------------------
# Indexing (sum 100 MIT positions)
# ---------------------------------------------------------------------------


def _setup_indexing_mit_lookup_100() -> dict[str, Any]:
    start = qq(2020, 1)
    t = TSeries(start, np.arange(100, dtype=np.float64))
    keys = [start + i for i in range(100)]
    return {"t": t, "keys": keys}


def _run_indexing_mit_lookup_100(state: dict[str, Any]) -> float:
    t = state["t"]
    keys = state["keys"]
    s = 0.0
    for k in keys:
        s += float(t[k])
    return s


# ---------------------------------------------------------------------------
# Arithmetic with misalignment (50-period overlap)
# ---------------------------------------------------------------------------


def _setup_arith_add_misaligned() -> dict[str, Any]:
    a = TSeries(qq(2020, 1), np.arange(100, dtype=np.float64))
    b = TSeries(qq(2032, 1), np.arange(100, dtype=np.float64) * 0.5)
    return {"a": a, "b": b}


def _run_arith_add_misaligned(state: dict[str, Any]) -> TSeries:
    return state["a"] + state["b"]


# ---------------------------------------------------------------------------
# Shift
# ---------------------------------------------------------------------------


def _setup_shift_quarterly_lag1() -> dict[str, Any]:
    return {"t": TSeries(qq(2020, 1), np.arange(100, dtype=np.float64))}


def _run_shift_quarterly_lag1(state: dict[str, Any]) -> TSeries:
    return shift(state["t"], -1)


# ---------------------------------------------------------------------------
# Moving average (n = 4)
# ---------------------------------------------------------------------------


def _setup_moving_average_quarterly_4() -> dict[str, Any]:
    return {"t": TSeries(qq(2020, 1), np.arange(100, dtype=np.float64))}


def _run_moving_average_quarterly_4(state: dict[str, Any]) -> TSeries:
    return moving_average(state["t"], 4)


# ---------------------------------------------------------------------------
# Frequency conversion (qq → yy, method="mean")
# ---------------------------------------------------------------------------


def _setup_fconvert_qq_to_yy_mean() -> dict[str, Any]:
    return {
        "target": Yearly(),
        "t": TSeries(qq(2020, 1), np.arange(100, dtype=np.float64)),
    }


def _run_fconvert_qq_to_yy_mean(state: dict[str, Any]) -> TSeries:
    return fconvert(state["target"], state["t"], method="mean")


# ---------------------------------------------------------------------------
# rec — AR(2) over a 100-period quarterly range (M1.5 Cython candidate)
# ---------------------------------------------------------------------------


def _setup_rec_ar2_100() -> dict[str, Any]:
    start = qq(2020, 1)
    target = TSeries(start, np.zeros(102, dtype=np.float64))
    target[start] = 1.0
    target[start + 1] = 1.0
    rng = MITRange(start + 2, start + 101)
    return {"target": target, "rng": rng}


def _run_rec_ar2_100(state: dict[str, Any]) -> TSeries:
    target = state["target"]
    rec(state["rng"], target, lambda t: 0.5 * target[t - 1] + 0.3 * target[t - 2])
    return target


# ---------------------------------------------------------------------------
# rec_linear — three-flavor AR(2) recurrence (decision 17 § option β).
#
# Two adjacent scenarios time the NumPy reference and the Cython compiled
# kernel directly (skipping the public rec_linear dispatcher) so the
# benchmark sees both *kernels* honestly without dispatch tax biasing
# either column. The matching Julia scenario `rec_linear_ar2_100` runs
# the same AR(2) recurrence via `@rec` — Julia has no equivalent
# "kernel split" because the macro inlines into native code at compile
# time, so the single Julia number stands for both Python rows in the
# three-flavor table the JSS paper will render.
# ---------------------------------------------------------------------------

# Shared shape: 100-step AR(2) `target[t] = 0.5*t[-1] + 0.3*t[-2]`. The
# kernels expect the values buffer pre-allocated with NaN padding past
# the initial conditions; the setup also pre-computes the kernel inputs
# (offset, count, coeffs, lags) so the timed window is exactly the
# numerical loop.
_REC_LINEAR_COEFFS = np.array([0.5, 0.3], dtype=np.float64)
_REC_LINEAR_LAGS = np.array([1, 2], dtype=np.int64)
_REC_LINEAR_OFFSET = 2  # First write is values[2]; values[0] and values[1] seed
_REC_LINEAR_COUNT = 100


def _setup_rec_linear_ar2_100() -> dict[str, Any]:
    # Fresh buffer per call so a prior run's NaN-overwrites don't leak.
    values = np.zeros(102, dtype=np.float64)
    values[0] = 1.0
    values[1] = 1.0
    return {
        "values": values,
        "offset": _REC_LINEAR_OFFSET,
        "count": _REC_LINEAR_COUNT,
        "coeffs": _REC_LINEAR_COEFFS,
        "lags": _REC_LINEAR_LAGS,
    }


def _run_rec_linear_ar2_100_numpy(state: dict[str, Any]) -> np.ndarray:
    rec_linear_numpy(
        state["values"], state["offset"], state["count"], state["coeffs"], state["lags"]
    )
    return state["values"]


def _run_rec_linear_ar2_100_cython(state: dict[str, Any]) -> np.ndarray:
    rec_linear_cython(
        state["values"], state["offset"], state["count"], state["coeffs"], state["lags"]
    )
    return state["values"]


# ---------------------------------------------------------------------------
# rec_linear_ar2_100_pylist — the four-column-shape "Python native" baseline.
# Decomposes the gap into (data structure: list vs ndarray) and (loop:
# interpreted vs compiled). See claude_files/paper/NOTES.md § "M1.5 first
# Cython port" for the framing.
# ---------------------------------------------------------------------------


def _setup_rec_linear_ar2_100_pylist() -> dict[str, Any]:
    values = [0.0] * 102
    values[0] = 1.0
    values[1] = 1.0
    return {"values": values, "offset": 2, "count": 100}


def _run_rec_linear_ar2_100_pylist(state: dict[str, Any]) -> list[float]:
    values = state["values"]
    offset = state["offset"]
    for i in range(state["count"]):
        out_idx = offset + i
        values[out_idx] = 0.5 * values[out_idx - 1] + 0.3 * values[out_idx - 2]
    return values


# ---------------------------------------------------------------------------
# Inventory expansion (session 18). The next blocks add coverage across the
# rest of the M1 public surface so the harness produces a comprehensive
# ratio table for [decision 18](decisions/18_cython_port_plan.md). Each new
# scenario follows the same setup/run shape as the originals; the matching
# Julia scenarios live in julia/scenarios.jl.
# ---------------------------------------------------------------------------


def _setup_construct_mvts_qq_100x5() -> dict[str, Any]:
    return {
        "start": qq(2020, 1),
        "cols": ["a", "b", "c", "d", "e"],
        "values": np.arange(500, dtype=np.float64).reshape(100, 5),
    }


def _run_construct_mvts_qq_100x5(state: dict[str, Any]) -> MVTSeries:
    return MVTSeries(state["start"], state["cols"], state["values"])


def _setup_indexing_int_lookup_100() -> dict[str, Any]:
    t = TSeries(qq(2020, 1), np.arange(100, dtype=np.float64))
    return {"t": t, "keys": list(range(100))}


def _run_indexing_int_lookup_100(state: dict[str, Any]) -> float:
    t = state["t"]
    s = 0.0
    for k in state["keys"]:
        s += float(t[k])
    return s


def _setup_indexing_mitrange_slice() -> dict[str, Any]:
    start = qq(2020, 1)
    t = TSeries(start, np.arange(100, dtype=np.float64))
    return {"t": t, "rng": MITRange(start + 20, start + 79)}


def _run_indexing_mitrange_slice(state: dict[str, Any]) -> TSeries:
    return state["t"][state["rng"]]


def _setup_indexing_mvts_column() -> dict[str, Any]:
    mvts = MVTSeries(
        qq(2020, 1),
        ["a", "b", "c", "d", "e"],
        np.arange(500, dtype=np.float64).reshape(100, 5),
    )
    return {"mvts": mvts}


def _run_indexing_mvts_column(state: dict[str, Any]) -> TSeries:
    return state["mvts"]["c"]


def _setup_arith_add_aligned() -> dict[str, Any]:
    start = qq(2020, 1)
    a = TSeries(start, np.arange(100, dtype=np.float64))
    b = TSeries(start, np.arange(100, dtype=np.float64) * 0.5)
    return {"a": a, "b": b}


def _run_arith_add_aligned(state: dict[str, Any]) -> TSeries:
    return state["a"] + state["b"]


def _setup_arith_mul_scalar() -> dict[str, Any]:
    return {"t": TSeries(qq(2020, 1), np.arange(100, dtype=np.float64))}


def _run_arith_mul_scalar(state: dict[str, Any]) -> TSeries:
    return state["t"] * 2.5


def _setup_diff_quarterly() -> dict[str, Any]:
    return {"t": TSeries(qq(2020, 1), np.arange(100, dtype=np.float64))}


def _run_diff_quarterly(state: dict[str, Any]) -> TSeries:
    return diff(state["t"])


def _setup_pct_quarterly() -> dict[str, Any]:
    # pct emits a RuntimeWarning on zero values; start at 1.0 to avoid it
    # (sister test in test_math.py mirrors this — see SESSION_LOG session 13).
    return {"t": TSeries(qq(2020, 1), np.arange(1.0, 101.0))}


def _run_pct_quarterly(state: dict[str, Any]) -> TSeries:
    return pct(state["t"])


def _setup_mean_quarterly_100() -> dict[str, Any]:
    return {"t": TSeries(qq(2020, 1), np.arange(100, dtype=np.float64))}


def _run_mean_quarterly_100(state: dict[str, Any]) -> float:
    return float(mean(state["t"]))


def _setup_std_quarterly_100() -> dict[str, Any]:
    return {"t": TSeries(qq(2020, 1), np.arange(100, dtype=np.float64))}


def _run_std_quarterly_100(state: dict[str, Any]) -> float:
    return float(std(state["t"]))


def _setup_cor_two_tseries() -> dict[str, Any]:
    start = qq(2020, 1)
    rng = np.random.default_rng(seed=20260515)
    a = TSeries(start, rng.standard_normal(100))
    b = TSeries(start, rng.standard_normal(100))
    return {"a": a, "b": b}


def _run_cor_two_tseries(state: dict[str, Any]) -> float:
    return float(cor(state["a"], state["b"]))


def _setup_cor_mvts_5_columns() -> dict[str, Any]:
    rng = np.random.default_rng(seed=20260515)
    mvts = MVTSeries(
        qq(2020, 1),
        ["a", "b", "c", "d", "e"],
        rng.standard_normal((100, 5)),
    )
    return {"mvts": mvts}


def _run_cor_mvts_5_columns(state: dict[str, Any]) -> np.ndarray:
    return cor(state["mvts"])


def _setup_cov_mvts_5_columns() -> dict[str, Any]:
    rng = np.random.default_rng(seed=20260515)
    mvts = MVTSeries(
        qq(2020, 1),
        ["a", "b", "c", "d", "e"],
        rng.standard_normal((100, 5)),
    )
    return {"mvts": mvts}


def _run_cov_mvts_5_columns(state: dict[str, Any]) -> np.ndarray:
    return cov(state["mvts"])


def _setup_moving_sum_quarterly_4() -> dict[str, Any]:
    return {"t": TSeries(qq(2020, 1), np.arange(100, dtype=np.float64))}


def _run_moving_sum_quarterly_4(state: dict[str, Any]) -> TSeries:
    return moving_sum(state["t"], 4)


def _setup_undiff_quarterly() -> dict[str, Any]:
    # Differenced series of length 100; undiff produces a length-101
    # cumulative integral anchored at value 0 the period before firstdate.
    return {"t": TSeries(qq(2020, 1), np.arange(100, dtype=np.float64))}


def _run_undiff_quarterly(state: dict[str, Any]) -> TSeries:
    return undiff(state["t"])


def _setup_fconvert_qq_to_yy_sum() -> dict[str, Any]:
    return {
        "target": Yearly(),
        "t": TSeries(qq(2020, 1), np.arange(100, dtype=np.float64)),
    }


def _run_fconvert_qq_to_yy_sum(state: dict[str, Any]) -> TSeries:
    return fconvert(state["target"], state["t"], method="sum")


def _setup_fconvert_yy_to_qq_const() -> dict[str, Any]:
    return {
        "target": Quarterly(),
        "t": TSeries(yy(2020), np.arange(25, dtype=np.float64)),
    }


def _run_fconvert_yy_to_qq_const(state: dict[str, Any]) -> TSeries:
    # Higher-freq direction (yearly → quarterly); default method is "const"
    # which broadcasts each year's value across its four quarters.
    return fconvert(state["target"], state["t"], method="const")


def _setup_fconvert_mm_to_qq_mean() -> dict[str, Any]:
    return {
        "target": Quarterly(),
        "t": TSeries(mm(2020, 1), np.arange(120, dtype=np.float64)),
    }


def _run_fconvert_mm_to_qq_mean(state: dict[str, Any]) -> TSeries:
    return fconvert(state["target"], state["t"], method="mean")


def _setup_workspace_filter_5_series() -> dict[str, Any]:
    start = qq(2020, 1)
    arr = np.arange(40, dtype=np.float64)
    w = Workspace()
    for name in ("a", "b", "c", "d", "e", "f", "g", "h", "i", "j"):
        w[name] = TSeries(start, arr.copy())
    return {"w": w, "keep": frozenset({"a", "b", "c", "d", "e"})}


def _run_workspace_filter_5_series(state: dict[str, Any]) -> Workspace:
    keep = state["keep"]
    return state["w"].filter(lambda k, v: k in keep)


# ---------------------------------------------------------------------------
# Workspace merge (5 series each)
# ---------------------------------------------------------------------------


def _setup_workspace_merge_5_series() -> dict[str, Any]:
    start = qq(2020, 1)
    arr = np.arange(40, dtype=np.float64)
    w1 = Workspace()
    w2 = Workspace()
    for name in ("a", "b", "c", "d", "e"):
        w1[name] = TSeries(start, arr.copy())
    for name in ("f", "g", "h", "i", "j"):
        w2[name] = TSeries(start, arr.copy())
    return {"w1": w1, "w2": w2}


def _run_workspace_merge_5_series(state: dict[str, Any]) -> Workspace:
    return state["w1"].merge(state["w2"])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SETUP: dict[str, Callable[[], Any]] = {
    # Construction
    "construct_tseries_qq_100": _setup_construct_tseries_qq_100,
    "construct_mvts_qq_100x5": _setup_construct_mvts_qq_100x5,
    # Indexing
    "indexing_mit_lookup_100": _setup_indexing_mit_lookup_100,
    "indexing_int_lookup_100": _setup_indexing_int_lookup_100,
    "indexing_mitrange_slice": _setup_indexing_mitrange_slice,
    "indexing_mvts_column": _setup_indexing_mvts_column,
    # Arithmetic
    "arith_add_misaligned": _setup_arith_add_misaligned,
    "arith_add_aligned": _setup_arith_add_aligned,
    "arith_mul_scalar": _setup_arith_mul_scalar,
    # Shift family
    "shift_quarterly_lag1": _setup_shift_quarterly_lag1,
    "diff_quarterly": _setup_diff_quarterly,
    "pct_quarterly": _setup_pct_quarterly,
    # Stats
    "mean_quarterly_100": _setup_mean_quarterly_100,
    "std_quarterly_100": _setup_std_quarterly_100,
    "cor_two_tseries": _setup_cor_two_tseries,
    "cor_mvts_5_columns": _setup_cor_mvts_5_columns,
    "cov_mvts_5_columns": _setup_cov_mvts_5_columns,
    # Moving / undiff
    "moving_average_quarterly_4": _setup_moving_average_quarterly_4,
    "moving_sum_quarterly_4": _setup_moving_sum_quarterly_4,
    "undiff_quarterly": _setup_undiff_quarterly,
    # fconvert
    "fconvert_qq_to_yy_mean": _setup_fconvert_qq_to_yy_mean,
    "fconvert_qq_to_yy_sum": _setup_fconvert_qq_to_yy_sum,
    "fconvert_yy_to_qq_const": _setup_fconvert_yy_to_qq_const,
    "fconvert_mm_to_qq_mean": _setup_fconvert_mm_to_qq_mean,
    # Recursion (general)
    "rec_ar2_100": _setup_rec_ar2_100,
    # Recursion (kernel-direct, three / four flavor)
    "rec_linear_ar2_100_pylist": _setup_rec_linear_ar2_100_pylist,
    "rec_linear_ar2_100_numpy": _setup_rec_linear_ar2_100,
    # Workspace
    "workspace_merge_5_series": _setup_workspace_merge_5_series,
    "workspace_filter_5_series": _setup_workspace_filter_5_series,
}

RUN: dict[str, Callable[[Any], Any]] = {
    # Construction
    "construct_tseries_qq_100": _run_construct_tseries_qq_100,
    "construct_mvts_qq_100x5": _run_construct_mvts_qq_100x5,
    # Indexing
    "indexing_mit_lookup_100": _run_indexing_mit_lookup_100,
    "indexing_int_lookup_100": _run_indexing_int_lookup_100,
    "indexing_mitrange_slice": _run_indexing_mitrange_slice,
    "indexing_mvts_column": _run_indexing_mvts_column,
    # Arithmetic
    "arith_add_misaligned": _run_arith_add_misaligned,
    "arith_add_aligned": _run_arith_add_aligned,
    "arith_mul_scalar": _run_arith_mul_scalar,
    # Shift family
    "shift_quarterly_lag1": _run_shift_quarterly_lag1,
    "diff_quarterly": _run_diff_quarterly,
    "pct_quarterly": _run_pct_quarterly,
    # Stats
    "mean_quarterly_100": _run_mean_quarterly_100,
    "std_quarterly_100": _run_std_quarterly_100,
    "cor_two_tseries": _run_cor_two_tseries,
    "cor_mvts_5_columns": _run_cor_mvts_5_columns,
    "cov_mvts_5_columns": _run_cov_mvts_5_columns,
    # Moving / undiff
    "moving_average_quarterly_4": _run_moving_average_quarterly_4,
    "moving_sum_quarterly_4": _run_moving_sum_quarterly_4,
    "undiff_quarterly": _run_undiff_quarterly,
    # fconvert
    "fconvert_qq_to_yy_mean": _run_fconvert_qq_to_yy_mean,
    "fconvert_qq_to_yy_sum": _run_fconvert_qq_to_yy_sum,
    "fconvert_yy_to_qq_const": _run_fconvert_yy_to_qq_const,
    "fconvert_mm_to_qq_mean": _run_fconvert_mm_to_qq_mean,
    # Recursion (general)
    "rec_ar2_100": _run_rec_ar2_100,
    # Recursion (kernel-direct, three / four flavor)
    "rec_linear_ar2_100_pylist": _run_rec_linear_ar2_100_pylist,
    "rec_linear_ar2_100_numpy": _run_rec_linear_ar2_100_numpy,
    # Workspace
    "workspace_merge_5_series": _run_workspace_merge_5_series,
    "workspace_filter_5_series": _run_workspace_filter_5_series,
}

# Description rendered into the comparison table; keep terse, the scenario
# name itself carries most of the meaning.
DESCRIPTION: dict[str, str] = {
    "construct_tseries_qq_100": "TSeries(qq, arr) from length-100 ndarray",
    "construct_mvts_qq_100x5": "MVTSeries(qq, 5 cols, 100x5 ndarray)",
    "indexing_mit_lookup_100": "Sum t[mit] over 100 keys",
    "indexing_int_lookup_100": "Sum t[int] over 100 keys",
    "indexing_mitrange_slice": "t[MITRange] — single 60-period slice",
    "indexing_mvts_column": "mvts['c'] — column access",
    "arith_add_misaligned": "100Q + 100Q with 50Q overlap",
    "arith_add_aligned": "100Q + 100Q same range",
    "arith_mul_scalar": "t * 2.5",
    "shift_quarterly_lag1": "shift(t, -1)",
    "diff_quarterly": "diff(t)",
    "pct_quarterly": "pct(t)",
    "mean_quarterly_100": "mean(t)",
    "std_quarterly_100": "std(t)",
    "cor_two_tseries": "cor(a, b) on two TSeries",
    "cor_mvts_5_columns": "cor(mvts) — 5x5 corr matrix",
    "cov_mvts_5_columns": "cov(mvts) — 5x5 cov matrix",
    "moving_average_quarterly_4": "moving_average(t, 4)",
    "moving_sum_quarterly_4": "moving_sum(t, 4)",
    "undiff_quarterly": "undiff(t)",
    "fconvert_qq_to_yy_mean": "fconvert(Yearly, t, method='mean')",
    "fconvert_qq_to_yy_sum": "fconvert(Yearly, t, method='sum')",
    "fconvert_yy_to_qq_const": "fconvert(Quarterly, t, method='const') (higher)",
    "fconvert_mm_to_qq_mean": "fconvert(Quarterly, monthly_t, method='mean')",
    "rec_ar2_100": "AR(2) over 100 quarters — general rec + lambda",
    "rec_linear_ar2_100_pylist": "AR(2) over 100 — rec_linear, pure-Python list",
    "rec_linear_ar2_100_numpy": "AR(2) over 100 — rec_linear NumPy kernel",
    "workspace_merge_5_series": "Workspace merge: 5 + 5 series",
    "workspace_filter_5_series": "Workspace filter: 10 down to 5 series",
}

# rec_linear's Cython kernel is conditionally registered: when the wheel
# was built without the C toolchain (Windows installs without the SDK,
# editable installs that skipped the build hook), the Cython column
# legitimately reads `n/a`. Adding a scenario only when the kernel is
# importable keeps the "missing data" honest in the table — see
# decision 17 on three-flavor reporting.
if _CYTHON_KERNEL_AVAILABLE:
    SETUP["rec_linear_ar2_100_cython"] = _setup_rec_linear_ar2_100
    RUN["rec_linear_ar2_100_cython"] = _run_rec_linear_ar2_100_cython
    DESCRIPTION["rec_linear_ar2_100_cython"] = "AR(2) over 100 quarters — rec_linear Cython kernel"

assert SETUP.keys() == RUN.keys() == DESCRIPTION.keys(), "scenario registries must agree"
