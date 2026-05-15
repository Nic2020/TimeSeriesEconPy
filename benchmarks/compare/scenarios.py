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
    TSeries,
    Workspace,
    Yearly,
    fconvert,
    moving_average,
    qq,
    rec,
    shift,
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
    "rec_linear_ar2_100_numpy": _setup_rec_linear_ar2_100,
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
    "rec_linear_ar2_100_numpy": _run_rec_linear_ar2_100_numpy,
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
    "rec_linear_ar2_100_numpy": "AR(2) over 100 quarters — rec_linear NumPy kernel",
    "workspace_merge_5_series": "Workspace merge: 5 + 5 series",
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
