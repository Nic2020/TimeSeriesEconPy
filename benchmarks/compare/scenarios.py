# SPDX-License-Identifier: MIT
"""Benchmark scenarios for the Julia ↔ Python comparison harness.

Each scenario exposes:

* ``SETUP[name]() -> state`` — builds whatever fixed inputs the run needs.
  The setup cost is **not** measured; only ``RUN[name](state)`` is timed.
* ``RUN[name](state) -> result`` — the operation being benchmarked. The return
  value exists only to defeat dead-code elimination; the driver discards it.

The matching Julia implementations live in ``julia/scenarios.jl``; the two
files are intentionally kept side-by-side so a reviewer can diff them.

The eight scenarios cover representative operations across the M1 surface:

================================  ==============================================
``construct_tseries_qq_100``      ``TSeries(qq(2020,1), arr)`` from a length-100 array.
``indexing_mit_lookup_100``       Read all 100 MIT positions and sum.
``arith_add_misaligned``          Add two 100-period TSeries with a 50-period overlap.
``shift_quarterly_lag1``          ``shift(t, -1)`` over a 100-period quarterly TSeries.
``moving_average_quarterly_4``    4-period moving average over the same input.
``fconvert_qq_to_yy_mean``        Quarterly → Yearly with ``method="mean"``.
``rec_ar2_100``                   100-step AR(2) recurrence (M1.5 Cython candidate).
``workspace_merge_5_series``      Merge two Workspaces each holding five TSeries.
================================  ==============================================

The Python entries here import only the public ``tsecon`` surface — the
benchmark must exercise the same API users will. The Julia counterpart only
imports the upstream ``TimeSeriesEcon`` package.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from tsecon import (
    MITRange,
    TSeries,
    Workspace,
    Yearly,
    fconvert,
    moving_average,
    qq,
    rec,
    shift,
)

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
    "construct_tseries_qq_100": _setup_construct_tseries_qq_100,
    "indexing_mit_lookup_100": _setup_indexing_mit_lookup_100,
    "arith_add_misaligned": _setup_arith_add_misaligned,
    "shift_quarterly_lag1": _setup_shift_quarterly_lag1,
    "moving_average_quarterly_4": _setup_moving_average_quarterly_4,
    "fconvert_qq_to_yy_mean": _setup_fconvert_qq_to_yy_mean,
    "rec_ar2_100": _setup_rec_ar2_100,
    "workspace_merge_5_series": _setup_workspace_merge_5_series,
}

RUN: dict[str, Callable[[Any], Any]] = {
    "construct_tseries_qq_100": _run_construct_tseries_qq_100,
    "indexing_mit_lookup_100": _run_indexing_mit_lookup_100,
    "arith_add_misaligned": _run_arith_add_misaligned,
    "shift_quarterly_lag1": _run_shift_quarterly_lag1,
    "moving_average_quarterly_4": _run_moving_average_quarterly_4,
    "fconvert_qq_to_yy_mean": _run_fconvert_qq_to_yy_mean,
    "rec_ar2_100": _run_rec_ar2_100,
    "workspace_merge_5_series": _run_workspace_merge_5_series,
}

# Description rendered into the comparison table; keep terse, the scenario
# name itself carries most of the meaning.
DESCRIPTION: dict[str, str] = {
    "construct_tseries_qq_100": "TSeries(qq, arr) from length-100 ndarray",
    "indexing_mit_lookup_100": "Sum t[mit] over 100 keys",
    "arith_add_misaligned": "100Q + 100Q with 50Q overlap",
    "shift_quarterly_lag1": "shift(t, -1)",
    "moving_average_quarterly_4": "moving_average(t, 4)",
    "fconvert_qq_to_yy_mean": "fconvert(Yearly, t, method='mean')",
    "rec_ar2_100": "AR(2) recurrence over 100 quarters",
    "workspace_merge_5_series": "Workspace merge: 5 + 5 series",
}

assert SETUP.keys() == RUN.keys() == DESCRIPTION.keys(), "scenario registries must agree"
