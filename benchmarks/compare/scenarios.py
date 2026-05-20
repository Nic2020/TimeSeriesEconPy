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
``rec_backcasting_via_lambda``    100-step backcast via reversed ``MITRange`` + ``rec`` lambda
                                  (M1.6.1).
``rec_linear_ar2_100_numpy``      Same AR(2) via the NumPy kernel direct.
``rec_linear_ar2_100_cython``     Same AR(2) via the Cython kernel direct (registered
                                  only when the compiled extension is importable).
``undiff_quarterly_numpy``        Anchored cumsum over length 101 via the NumPy kernel direct
                                  (M1.6.2).
``undiff_quarterly_cython``       Same workload via the Cython kernel direct (M1.6.2;
                                  registered only when the compiled extension is importable).
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
    MIT,
    MITRange,
    MVTSeries,
    Quarterly,
    TSeries,
    Unit,
    Workspace,
    Yearly,
    compare,
    copyto,
    cor,
    cov,
    diff,
    fconvert,
    lead,
    lookup,
    mean,
    mm,
    moving_average,
    moving_sum,
    overlay,
    pct,
    qq,
    quantile,
    rangeof,
    rec,
    reindex,
    shift,
    std,
    undiff,
    ytypct,
    yy,
)
from tsecon._fconvert_kernels import (
    METHOD_MEAN,
    METHOD_SUM,
    aggregate_groups_numpy,
)
from tsecon._indexing_kernels import gather_numpy
from tsecon._math_kernels import cumsum_anchored_numpy
from tsecon._rec_kernels import rec_linear_numpy
from tsecon._stats_kernels import cor_numpy, mean_numpy, std_numpy

try:
    from tsecon._rec_kernels_cy import rec_linear_cython  # type: ignore[import-not-found]

    _REC_CYTHON_AVAILABLE = True
except ImportError:  # pragma: no cover — depends on wheel-build state
    _REC_CYTHON_AVAILABLE = False

try:
    from tsecon._indexing_kernels_cy import gather_cython  # type: ignore[import-not-found]

    _INDEXING_CYTHON_AVAILABLE = True
except ImportError:  # pragma: no cover — depends on wheel-build state
    _INDEXING_CYTHON_AVAILABLE = False

try:
    from tsecon._stats_kernels_cy import (  # type: ignore[import-not-found]
        cor_cython,
        mean_cython,
        std_cython,
    )

    _STATS_CYTHON_AVAILABLE = True
except ImportError:  # pragma: no cover — depends on wheel-build state
    _STATS_CYTHON_AVAILABLE = False

try:
    from tsecon._fconvert_kernels_cy import (  # type: ignore[import-not-found]
        aggregate_groups_cython,
    )

    _FCONVERT_CYTHON_AVAILABLE = True
except ImportError:  # pragma: no cover — depends on wheel-build state
    _FCONVERT_CYTHON_AVAILABLE = False

try:
    from tsecon._math_kernels_cy import (  # type: ignore[import-not-found]
        cumsum_anchored_cython,
    )

    _MATH_CYTHON_AVAILABLE = True
except ImportError:  # pragma: no cover — depends on wheel-build state
    _MATH_CYTHON_AVAILABLE = False

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
# Vectorised lookup — the M1.5 second Cython port (session 19).
#
# Three new scenarios mirror the rec_linear multi-flavor shape:
#
# * ``indexing_lookup_100_api``   — full public ``lookup(t, keys)`` including
#                                   MIT-to-offset translation (the realistic
#                                   user-facing operation).
# * ``indexing_lookup_100_numpy`` — kernel-direct ``gather_numpy(values, ix)``
#                                   skipping public-API validation (option β).
# * ``indexing_lookup_100_cython`` — kernel-direct Cython gather (conditional
#                                   on the compiled extension).
#
# The existing ``indexing_mit_lookup_100`` (Python for-loop with ``t[k]`` per
# element) stays as the "naive user pattern" baseline — the 935x row.
#
# Unlike rec_linear, the NumPy reference here is *already* vectorised
# (``np.take`` runs in C). So the kernel-direct numpy/cython columns
# should land within a small factor of each other; the *big* win is the
# gap between the per-element loop (935x) and the vectorised public API.
# The framing: vectorised API is the win, not the Cython kernel itself.
# ---------------------------------------------------------------------------


def _setup_indexing_lookup_100_api() -> dict[str, Any]:
    start = qq(2020, 1)
    t = TSeries(start, np.arange(100, dtype=np.float64))
    keys = [start + i for i in range(100)]
    return {"t": t, "keys": keys}


def _run_indexing_lookup_100_api(state: dict[str, Any]) -> np.ndarray:
    return lookup(state["t"], state["keys"])


# Kernel-direct scenarios share a single pre-built (values, indices) pair so
# the timed window is exactly the gather call. Mirrors the rec_linear
# kernel-direct setup style (decision 17 § option β).
def _setup_indexing_gather_100_kernel() -> dict[str, Any]:
    return {
        "values": np.arange(100, dtype=np.float64),
        "indices": np.arange(100, dtype=np.int64),
    }


def _run_indexing_gather_100_numpy(state: dict[str, Any]) -> np.ndarray:
    return gather_numpy(state["values"], state["indices"])


def _run_indexing_gather_100_cython(state: dict[str, Any]) -> np.ndarray:
    return gather_cython(state["values"], state["indices"])


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
# rec — backcasting via lambda over a reversed range (M1.6.1).
#
# Mirrors Julia's `@rec t=lastQ:-1:firstQ s[t] = s[t+1] - g`. The Python
# port goes through the same general `rec` higher-order entry as the
# forward AR(2) above; the reversed iteration order is encoded in the
# MITRange `step=-1`. Times the per-iteration cost of: reversed-range
# iterator, lambda dispatch, `target[t + 1]` getitem, `target[t]` setitem.
# Pairs name-for-name with the Julia scenario `rec_backcasting_via_lambda`.
# ---------------------------------------------------------------------------


def _setup_rec_backcasting_via_lambda() -> dict[str, Any]:
    # 100-step backcast: target[lastQ] = 100, walk backward applying
    # target[t] = target[t+1] - 0.5 each step.
    start = qq(2020, 1)
    n = 100
    target = TSeries(start, np.zeros(n, dtype=np.float64))
    target[start + (n - 1)] = 100.0
    # Range: penultimate down to first.
    rng = MITRange(start + (n - 2), start, step=-1)
    return {"target": target, "rng": rng}


def _run_rec_backcasting_via_lambda(state: dict[str, Any]) -> TSeries:
    target = state["target"]
    rec(state["rng"], target, lambda t: target[t + 1] - 0.5)
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
        "step": 1,
        "coeffs": _REC_LINEAR_COEFFS,
        "lags": _REC_LINEAR_LAGS,
    }


def _run_rec_linear_ar2_100_numpy(state: dict[str, Any]) -> np.ndarray:
    rec_linear_numpy(
        state["values"],
        state["offset"],
        state["count"],
        state["step"],
        state["coeffs"],
        state["lags"],
    )
    return state["values"]


def _run_rec_linear_ar2_100_cython(state: dict[str, Any]) -> np.ndarray:
    rec_linear_cython(
        state["values"],
        state["offset"],
        state["count"],
        state["step"],
        state["coeffs"],
        state["lags"],
    )
    return state["values"]


# ---------------------------------------------------------------------------
# rec_linear_ar2_100_pylist — the four-column-shape "Python native" baseline.
# Decomposes the gap into (data structure: list vs ndarray) and (loop:
# interpreted vs compiled). The pure-Python list+loop column is the
# rec_linear M1.5 framing baseline.
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
# Inventory expansion. The next blocks add coverage across the rest of
# the M1 public surface so the harness produces a comprehensive ratio
# table for the Cython port plan (see docs/design/decisions.md #18).
# Each new scenario follows the same setup/run shape as the originals;
# the matching Julia scenarios live in julia/scenarios.jl.
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
    # (sister test in test_math.py mirrors this).
    return {"t": TSeries(qq(2020, 1), np.arange(1.0, 101.0))}


def _run_pct_quarterly(state: dict[str, Any]) -> TSeries:
    return pct(state["t"])


def _setup_lead_quarterly_lag1() -> dict[str, Any]:
    return {"t": TSeries(qq(2020, 1), np.arange(100, dtype=np.float64))}


def _run_lead_quarterly_lag1(state: dict[str, Any]) -> TSeries:
    return lead(state["t"], 1)


def _setup_ytypct_quarterly_100() -> dict[str, Any]:
    # ytypct divides by shift(t, -ppy) — start at 1.0 to avoid zero-division.
    return {"t": TSeries(qq(2020, 1), np.arange(1.0, 101.0))}


def _run_ytypct_quarterly_100(state: dict[str, Any]) -> TSeries:
    return ytypct(state["t"])


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


# ---------------------------------------------------------------------------
# Stats kernel-direct scenarios — the M1.5 third Cython port (session 20).
#
# Three new scenario triples (mean / std / cor) mirror the rec_linear and
# indexing multi-flavor shape:
#
# * ``mean_quarterly_100``         — public ``mean(t)`` (now Cython-backed
#                                    when the kernel is available; falls back
#                                    to ``np.mean`` otherwise).
# * ``mean_quarterly_100_numpy``   — kernel-direct ``mean_numpy(values)``
#                                    = ``float(np.mean(values))`` skipping
#                                    public-API dispatch and TSeries
#                                    resolution (option β).
# * ``mean_quarterly_100_cython``  — kernel-direct ``mean_cython(values)``
#                                    (conditional on the compiled extension).
#
# The same shape applies for ``std`` and ``cor``. The kernel-direct numpy
# rows skip the ``_resolve_values`` overhead that the public path pays;
# the kernel-direct cython rows additionally skip NumPy's per-call dispatch
# and 0-D scalar boxing tax. The Julia side has no kernel split (Julia's ``mean(t)`` inlines
# directly), so all three Python flavors share a single Julia counterpart.
# ---------------------------------------------------------------------------


# Shared kernel-input setups — pre-extract the values ndarray so the timed
# window is exactly the kernel call (skipping TSeries resolution).
def _setup_stats_scalar_kernel_100() -> dict[str, Any]:
    return {"values": np.arange(100, dtype=np.float64)}


def _run_mean_quarterly_100_numpy(state: dict[str, Any]) -> float:
    return mean_numpy(state["values"])


def _run_mean_quarterly_100_cython(state: dict[str, Any]) -> float:
    return mean_cython(state["values"])


def _run_std_quarterly_100_numpy(state: dict[str, Any]) -> float:
    return std_numpy(state["values"], 1)


def _run_std_quarterly_100_cython(state: dict[str, Any]) -> float:
    return std_cython(state["values"], 1)


def _setup_cor_two_kernel() -> dict[str, Any]:
    rng = np.random.default_rng(seed=20260515)
    return {
        "x": np.ascontiguousarray(rng.standard_normal(100), dtype=np.float64),
        "y": np.ascontiguousarray(rng.standard_normal(100), dtype=np.float64),
    }


def _run_cor_two_tseries_numpy(state: dict[str, Any]) -> float:
    return cor_numpy(state["x"], state["y"])


def _run_cor_two_tseries_cython(state: dict[str, Any]) -> float:
    return cor_cython(state["x"], state["y"])


def _setup_linalg_matrix_tseries_100() -> dict[str, Any]:
    """100x100 coefficient matrix times a length-100 TSeries (``A @ t``).

    M1.6.3g — closes G12. Matches Julia's ``A * t`` precedent exactly:
    matmul strips frequency / range labels and returns a plain ndarray
    (see ``src/tsecon/linalg.py``). The square shape matches the
    canonical "transition matrix on a length-N trajectory" VAR-style op;
    the length-100 trajectory aligns with the project's ``_100``
    benchmark-naming convention.
    """
    rng = np.random.default_rng(seed=20260518)
    matrix = rng.standard_normal((100, 100))
    t = TSeries(MITRange(qq(2020, 1), qq(2020, 1) + 99), rng.standard_normal(100))
    return {"a": matrix, "t": t}


def _run_linalg_matrix_tseries_100(state: dict[str, Any]) -> Any:
    return state["a"] @ state["t"]


def _setup_mean_mvts_axis0_5cols() -> dict[str, Any]:
    """100x5 MVTSeries for the per-column axis=0 reduction (G11)."""
    rng = np.random.default_rng(seed=20260518)
    mvts = MVTSeries(
        qq(2020, 1),
        ["a", "b", "c", "d", "e"],
        rng.standard_normal((100, 5)),
    )
    return {"mvts": mvts}


def _run_mean_mvts_axis0_5cols(state: dict[str, Any]) -> Any:
    return mean(state["mvts"], axis=0)


def _setup_mean_mvts_axis1_100rows() -> dict[str, Any]:
    """Same 100x5 MVTSeries — the per-row axis=1 reduction (G11)."""
    rng = np.random.default_rng(seed=20260518)
    mvts = MVTSeries(
        qq(2020, 1),
        ["a", "b", "c", "d", "e"],
        rng.standard_normal((100, 5)),
    )
    return {"mvts": mvts}


def _run_mean_mvts_axis1_100rows(state: dict[str, Any]) -> Any:
    return mean(state["mvts"], axis=1)


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


# ---------------------------------------------------------------------------
# M1.6 coverage expansion — quantile / cov(x,y) / ytypct / lead, plus the
# two missing higher-freq fconvert methods (linear, even).
# ---------------------------------------------------------------------------


def _setup_quantile_quarterly_100() -> dict[str, Any]:
    rng = np.random.default_rng(seed=20260515)
    return {"t": TSeries(qq(2020, 1), rng.standard_normal(100))}


def _run_quantile_quarterly_100(state: dict[str, Any]) -> float:
    return float(quantile(state["t"], 0.5))


def _setup_cov_two_tseries() -> dict[str, Any]:
    start = qq(2020, 1)
    rng = np.random.default_rng(seed=20260515)
    a = TSeries(start, rng.standard_normal(100))
    b = TSeries(start, rng.standard_normal(100))
    return {"a": a, "b": b}


def _run_cov_two_tseries(state: dict[str, Any]) -> float:
    return float(cov(state["a"], state["b"]))


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


# ---------------------------------------------------------------------------
# undiff kernel-direct scenarios — the M1.6.2 fifth Cython port.
#
# Three new scenarios mirror the rec_linear / indexing / stats / fconvert
# multi-flavor shape:
#
# * ``undiff_quarterly``           — public ``undiff(t)`` including anchor
#                                    resolution + dvar extension (kept
#                                    unchanged from before this port; now
#                                    Cython-backed when the kernel is
#                                    available).
# * ``undiff_quarterly_numpy``     — kernel-direct
#                                    ``cumsum_anchored_numpy(buffer, 0,
#                                    101, 0.0, 0)`` skipping the public-API
#                                    overhead (anchor resolution, dvar
#                                    extension, TSeries construction).
# * ``undiff_quarterly_cython``    — kernel-direct Cython
#                                    (conditional on the compiled
#                                    extension).
#
# The kernel pair mutates its input in place — the run function reinitialises
# the chunk per call from a frozen reference so subsequent calls don't
# compound the cumsum into infinity. The reset cost is small and identical
# across the NumPy and Cython rows, so it doesn't bias the comparison.
#
# The Julia side has no kernel split (Julia's ``undiff`` inlines directly),
# so all three Python flavors share a single Julia counterpart.
# ---------------------------------------------------------------------------


def _setup_undiff_kernel_quarterly() -> dict[str, Any]:
    # The dvar is arange(0, 100); the public undiff extends with one zero
    # at the front so the default anchor (at firstdate-1) falls inside.
    # The kernel works on the post-extension length-101 buffer with anchor
    # at index 0, anchor_value=0.
    initial = np.concatenate([[0.0], np.arange(100, dtype=np.float64)])
    buffer = initial.copy()
    return {
        "initial": initial,
        "buffer": buffer,
        "offset": 0,
        "count": 101,
        "anchor_value": 0.0,
        "anchor_relative_idx": 0,
    }


def _run_undiff_quarterly_numpy(state: dict[str, Any]) -> np.ndarray:
    # Reset the chunk each call — the kernel mutates in place and a
    # repeated cumsum would compound to overflow within a benchmark window.
    state["buffer"][:] = state["initial"]
    cumsum_anchored_numpy(
        state["buffer"],
        state["offset"],
        state["count"],
        state["anchor_value"],
        state["anchor_relative_idx"],
    )
    return state["buffer"]


def _run_undiff_quarterly_cython(state: dict[str, Any]) -> np.ndarray:
    state["buffer"][:] = state["initial"]
    cumsum_anchored_cython(
        state["buffer"],
        state["offset"],
        state["count"],
        state["anchor_value"],
        state["anchor_relative_idx"],
    )
    return state["buffer"]


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


def _setup_fconvert_yy_to_qq_linear() -> dict[str, Any]:
    return {
        "target": Quarterly(),
        "t": TSeries(yy(2020), np.arange(25, dtype=np.float64)),
    }


def _run_fconvert_yy_to_qq_linear(state: dict[str, Any]) -> TSeries:
    # Higher-freq direction with linear interpolation between yearly values.
    return fconvert(state["target"], state["t"], method="linear")


def _setup_fconvert_yy_to_qq_even() -> dict[str, Any]:
    return {
        "target": Quarterly(),
        "t": TSeries(yy(2020), np.arange(25, dtype=np.float64)),
    }


def _run_fconvert_yy_to_qq_even(state: dict[str, Any]) -> TSeries:
    # Higher-freq direction; each year's value divided evenly across 4 quarters.
    return fconvert(state["target"], state["t"], method="even")


def _setup_fconvert_mm_to_qq_mean() -> dict[str, Any]:
    return {
        "target": Quarterly(),
        "t": TSeries(mm(2020, 1), np.arange(120, dtype=np.float64)),
    }


def _run_fconvert_mm_to_qq_mean(state: dict[str, Any]) -> TSeries:
    return fconvert(state["target"], state["t"], method="mean")


# ---------------------------------------------------------------------------
# fconvert kernel-direct scenarios — the M1.5 fourth Cython port (session 21).
#
# Three new scenario triples (qq→yy mean / qq→yy sum / mm→qq mean) mirror
# the rec_linear, indexing, and stats multi-flavor shape:
#
# * ``fconvert_qq_to_yy_mean``         — public ``fconvert(Yearly, t,
#                                        method='mean')`` (now Cython-backed
#                                        when the kernel is available;
#                                        falls back to NumPy otherwise).
# * ``fconvert_qq_to_yy_mean_numpy``   — kernel-direct
#                                        ``aggregate_groups_numpy(values,
#                                        starts, lengths, METHOD_MEAN)``
#                                        skipping the public-API range
#                                        computation and TSeries
#                                        construction (option β).
# * ``fconvert_qq_to_yy_mean_cython``  — kernel-direct Cython
#                                        (conditional on the compiled
#                                        extension).
#
# The kernel-direct numpy rows skip the truncation arithmetic and
# TSeries-wrap that the public path pays; the kernel-direct cython rows
# additionally fuse the outer per-group dispatch loop into C. The Julia
# side has no kernel split
# (Julia's frequency conversion inlines directly), so all three Python
# flavors share a single Julia counterpart.
# ---------------------------------------------------------------------------


def _setup_fconvert_qq_to_yy_kernel() -> dict[str, Any]:
    # 100 quarters → 25 yearly groups of 4 quarters each.
    return {
        "values": np.arange(100, dtype=np.float64),
        "group_starts": np.arange(0, 100, 4, dtype=np.int64),
        "group_lengths": np.full(25, 4, dtype=np.int64),
    }


def _run_fconvert_qq_to_yy_mean_numpy(state: dict[str, Any]) -> np.ndarray:
    return aggregate_groups_numpy(
        state["values"], state["group_starts"], state["group_lengths"], METHOD_MEAN
    )


def _run_fconvert_qq_to_yy_mean_cython(state: dict[str, Any]) -> np.ndarray:
    return aggregate_groups_cython(
        state["values"], state["group_starts"], state["group_lengths"], METHOD_MEAN
    )


def _run_fconvert_qq_to_yy_sum_numpy(state: dict[str, Any]) -> np.ndarray:
    return aggregate_groups_numpy(
        state["values"], state["group_starts"], state["group_lengths"], METHOD_SUM
    )


def _run_fconvert_qq_to_yy_sum_cython(state: dict[str, Any]) -> np.ndarray:
    return aggregate_groups_cython(
        state["values"], state["group_starts"], state["group_lengths"], METHOD_SUM
    )


def _setup_fconvert_mm_to_qq_kernel() -> dict[str, Any]:
    # 120 months → 40 quarterly groups of 3 months each.
    return {
        "values": np.arange(120, dtype=np.float64),
        "group_starts": np.arange(0, 120, 3, dtype=np.int64),
        "group_lengths": np.full(40, 3, dtype=np.int64),
    }


def _run_fconvert_mm_to_qq_mean_numpy(state: dict[str, Any]) -> np.ndarray:
    return aggregate_groups_numpy(
        state["values"], state["group_starts"], state["group_lengths"], METHOD_MEAN
    )


def _run_fconvert_mm_to_qq_mean_cython(state: dict[str, Any]) -> np.ndarray:
    return aggregate_groups_cython(
        state["values"], state["group_starts"], state["group_lengths"], METHOD_MEAN
    )


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
# Mixed-frequency scenarios — added 2026-05-17 (this session) for the
# pandas/polars 4-column comparison. These scenarios exist specifically to
# expose the friction that DataFrame-based pipelines hit when the data
# spans multiple frequencies. tsecon represents the result of mixed-freq
# work explicitly (every series carries its own ``frequency``); pandas and
# polars represent it implicitly through index alignment / time-column
# joins, and the user pays for the conversion. The two scenarios below are
# the smallest non-toy macro patterns that surface the cost.
#
# Result types are identical across tsecon / Julia / pandas / polars (a
# single quarterly time series); only the *path* to the result differs
# per backend.
# ---------------------------------------------------------------------------


def _setup_mixed_freq_qq_minus_mm_mean() -> dict[str, Any]:
    return {
        "target": Quarterly(),
        "gdp": TSeries(qq(2020, 1), np.arange(100, dtype=np.float64)),
        "cpi": TSeries(mm(2020, 1), np.arange(300, dtype=np.float64)),
    }


def _run_mixed_freq_qq_minus_mm_mean(state: dict[str, Any]) -> TSeries:
    return state["gdp"] - fconvert(state["target"], state["cpi"], method="mean")


def _setup_mixed_freq_pipeline_three_freq() -> dict[str, Any]:
    return {
        "target": Quarterly(),
        "unemp": TSeries(yy(2020), np.arange(25, dtype=np.float64)),
        "gdp": TSeries(qq(2020, 1), np.arange(100, dtype=np.float64)),
        "cpi": TSeries(mm(2020, 1), np.arange(300, dtype=np.float64)),
    }


def _run_mixed_freq_pipeline_three_freq(state: dict[str, Any]) -> TSeries:
    q = state["target"]
    return (
        fconvert(q, state["unemp"], method="const")
        + state["gdp"]
        + fconvert(q, state["cpi"], method="mean")
    )


# ---------------------------------------------------------------------------
# overlay / compare / reindex — M1.6.3b (`various.jl` pull-forward)
# ---------------------------------------------------------------------------


def _setup_overlay_three_tseries() -> dict[str, Any]:
    # Three 100-period quarterly TSeries with partial overlap and scattered NaN
    # so the per-position "first-non-typenan wins" logic actually does work.
    arr = np.arange(100, dtype=np.float64)
    a = TSeries(qq(2020, 1), arr.copy())
    a.values[::7] = np.nan
    b = TSeries(qq(2019, 1), np.full(100, 100.0))
    b.values[::5] = np.nan
    c = TSeries(qq(2021, 1), np.full(100, 200.0))
    return {"a": a, "b": b, "c": c}


def _run_overlay_three_tseries(state: dict[str, Any]) -> TSeries:
    return overlay(state["a"], state["b"], state["c"])


def _setup_compare_workspaces_equal_5_keys() -> dict[str, Any]:
    start = qq(2020, 1)
    arr = np.arange(100, dtype=np.float64)
    w1 = Workspace()
    w2 = Workspace()
    for name in ("a", "b", "c", "d", "e"):
        w1[name] = TSeries(start, arr.copy())
        w2[name] = TSeries(start, arr.copy())
    return {"w1": w1, "w2": w2}


def _run_compare_workspaces_equal_5_keys(state: dict[str, Any]) -> bool:
    # `quiet=True` removes stdout I/O from the measured time (the harness's
    # min-of-many-runs would otherwise be dominated by terminal flushes).
    return compare(state["w1"], state["w2"], quiet=True).equal


def _setup_compare_workspaces_differ_5_keys() -> dict[str, Any]:
    # Differs at one element of one TSeries — exercises the recursive walk
    # *and* the diff-collection path (positive case stops at TSeries-level
    # `isapprox`; this one builds CompareDifference instances).
    start = qq(2020, 1)
    arr = np.arange(100, dtype=np.float64)
    w1 = Workspace()
    w2 = Workspace()
    for name in ("a", "b", "c", "d", "e"):
        w1[name] = TSeries(start, arr.copy())
        w2[name] = TSeries(start, arr.copy())
    # Position 50 lands at qq(2032, 3) on a 100-period quarterly series starting 2020Q1.
    w2["c"][50] = -999.0
    return {"w1": w1, "w2": w2}


def _run_compare_workspaces_differ_5_keys(state: dict[str, Any]) -> bool:
    return compare(state["w1"], state["w2"], quiet=True).equal


def _setup_reindex_tseries_100() -> dict[str, Any]:
    return {
        "t": TSeries(qq(2020, 1), np.arange(100, dtype=np.float64)),
        "pair": (qq(2020, 1), MIT(Unit(), 1)),
    }


def _run_reindex_tseries_100(state: dict[str, Any]) -> TSeries:
    return reindex(state["t"], state["pair"])


# ---------------------------------------------------------------------------
# rangeof(t, drop=1) — the tutorial-1 @rec idiom (M1.6.3c, G5 closure).
#
# Times the per-call overhead of the new public `rangeof` free function on
# a 100-period quarterly TSeries with `drop=1`. The matching Julia scenario
# `rangeof_tseries_drop1` calls `rangeof(t; drop=1)` (a closure call —
# expected to be very fast on the Julia side; the Python column is meant
# to characterise the kwarg-dispatch tax against the documented baseline).
# ---------------------------------------------------------------------------


def _setup_rangeof_tseries_drop1() -> dict[str, Any]:
    return {"t": TSeries(qq(2020, 1), np.arange(100, dtype=np.float64))}


def _run_rangeof_tseries_drop1(state: dict[str, Any]) -> MITRange:
    return rangeof(state["t"], drop=1)


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
# copyto — Workspace → MVTSeries in-place materialiser (M1.6.3h, closes G13)
# ---------------------------------------------------------------------------


def _setup_workspace_to_mvts_copyto_5cols() -> dict[str, Any]:
    start = qq(2020, 1)
    stop = start + 99  # 100 periods
    rng = MITRange(start, stop)
    names = ["a", "b", "c", "d", "e"]
    arr = np.arange(100, dtype=np.float64)
    w = Workspace(**{n: TSeries(start, arr.copy() + i) for i, n in enumerate(names)})
    dst = MVTSeries(rng, names)
    return {"dst": dst, "w": w}


def _run_workspace_to_mvts_copyto_5cols(state: dict[str, Any]) -> MVTSeries:
    return copyto(state["dst"], state["w"])


# ---------------------------------------------------------------------------
# X-13ARIMA-SEATS deseasonalisation (M2.6).
#
# The first benchmark row where the wrapper is *not* the dominant cost —
# X-13 invokes the bundled Fortran binary, which dwarfs Python-side spec
# construction + result parsing. The point of this scenario is **not** to
# rank the wrapper against alternatives; it is to confirm Python and
# Julia wrappers spend the same per-call time on top of the same binary,
# i.e. that neither wrapper imposes a measurable extra cost.
#
# Two-flavor (Python + Julia; no Cython): both wrappers shell out to the
# same x13as.exe in identical CWDs; the .spc-write + subprocess-spawn +
# output-parse pipeline is what each row measures. The matching Julia
# row lives in benchmarks/compare/julia/scenarios.jl as
# `deseasonalize_quarterly_50y`.
#
# Conditional registration: registered only when `_resolve_binary()` is
# not None. The setup builds a 200-quarter macro series (50 years
# quarterly = 200 points; the realistic span for X-11 to estimate
# seasonal factors).
# ---------------------------------------------------------------------------


def _setup_deseasonalize_quarterly_50y() -> dict[str, Any]:
    n = 200  # 50 years × 4 quarters.
    start = qq(2000, 1)
    end = start + (n - 1)
    rng = MITRange(start, end)
    i = np.arange(n, dtype=np.float64)
    # Realistic macro-style series: log-linear trend + sinusoidal seasonal +
    # mild Gaussian noise (rng seeded for reproducibility).
    trend = 100.0 + 0.5 * i + 0.001 * i * i
    seasonal = 5.0 * np.sin(2 * np.pi * i / 4.0)
    rng_seed = np.random.default_rng(seed=20260520)
    noise = rng_seed.standard_normal(n) * 0.5
    return {"ts": TSeries(rng, trend + seasonal + noise)}


def _run_deseasonalize_quarterly_50y(state: dict[str, Any]) -> Any:
    from tsecon.x13 import deseasonalize  # noqa: PLC0415 - optional surface

    return deseasonalize(state["ts"])


def _x13_binary_available() -> bool:
    """True iff a usable x13as binary is reachable on this machine."""
    from tsecon.x13._result import _resolve_binary  # noqa: PLC0415

    return _resolve_binary() is not None


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
    # Indexing (M1.5 second Cython port — kernel-direct and public API)
    "indexing_lookup_100_api": _setup_indexing_lookup_100_api,
    "indexing_lookup_100_numpy": _setup_indexing_gather_100_kernel,
    # Arithmetic
    "arith_add_misaligned": _setup_arith_add_misaligned,
    "arith_add_aligned": _setup_arith_add_aligned,
    "arith_mul_scalar": _setup_arith_mul_scalar,
    # Shift family
    "shift_quarterly_lag1": _setup_shift_quarterly_lag1,
    "lead_quarterly_lag1": _setup_lead_quarterly_lag1,
    "diff_quarterly": _setup_diff_quarterly,
    "pct_quarterly": _setup_pct_quarterly,
    "ytypct_quarterly_100": _setup_ytypct_quarterly_100,
    # Stats
    "mean_quarterly_100": _setup_mean_quarterly_100,
    "std_quarterly_100": _setup_std_quarterly_100,
    "quantile_quarterly_100": _setup_quantile_quarterly_100,
    "cor_two_tseries": _setup_cor_two_tseries,
    "cov_two_tseries": _setup_cov_two_tseries,
    "cor_mvts_5_columns": _setup_cor_mvts_5_columns,
    "cov_mvts_5_columns": _setup_cov_mvts_5_columns,
    # MVTSeries axis= reductions (M1.6.3f — closes G11)
    "mean_mvts_axis0_5cols": _setup_mean_mvts_axis0_5cols,
    "mean_mvts_axis1_100rows": _setup_mean_mvts_axis1_100rows,
    # Stats (M1.5 third Cython port — kernel-direct + public API)
    "mean_quarterly_100_numpy": _setup_stats_scalar_kernel_100,
    "std_quarterly_100_numpy": _setup_stats_scalar_kernel_100,
    "cor_two_tseries_numpy": _setup_cor_two_kernel,
    # Moving / undiff
    "moving_average_quarterly_4": _setup_moving_average_quarterly_4,
    "moving_sum_quarterly_4": _setup_moving_sum_quarterly_4,
    "undiff_quarterly": _setup_undiff_quarterly,
    # undiff (M1.6.2 fifth Cython port — kernel-direct + public API)
    "undiff_quarterly_numpy": _setup_undiff_kernel_quarterly,
    # fconvert
    "fconvert_qq_to_yy_mean": _setup_fconvert_qq_to_yy_mean,
    "fconvert_qq_to_yy_sum": _setup_fconvert_qq_to_yy_sum,
    "fconvert_yy_to_qq_const": _setup_fconvert_yy_to_qq_const,
    "fconvert_yy_to_qq_linear": _setup_fconvert_yy_to_qq_linear,
    "fconvert_yy_to_qq_even": _setup_fconvert_yy_to_qq_even,
    "fconvert_mm_to_qq_mean": _setup_fconvert_mm_to_qq_mean,
    # fconvert (M1.5 fourth Cython port — kernel-direct + public API)
    "fconvert_qq_to_yy_mean_numpy": _setup_fconvert_qq_to_yy_kernel,
    "fconvert_qq_to_yy_sum_numpy": _setup_fconvert_qq_to_yy_kernel,
    "fconvert_mm_to_qq_mean_numpy": _setup_fconvert_mm_to_qq_kernel,
    # Recursion (general)
    "rec_ar2_100": _setup_rec_ar2_100,
    "rec_backcasting_via_lambda": _setup_rec_backcasting_via_lambda,
    # Recursion (kernel-direct, three / four flavor)
    "rec_linear_ar2_100_pylist": _setup_rec_linear_ar2_100_pylist,
    "rec_linear_ar2_100_numpy": _setup_rec_linear_ar2_100,
    # Workspace
    "workspace_merge_5_series": _setup_workspace_merge_5_series,
    "workspace_filter_5_series": _setup_workspace_filter_5_series,
    # copyto (M1.6.3h — closes G13)
    "workspace_to_mvts_copyto_5cols": _setup_workspace_to_mvts_copyto_5cols,
    # Mixed-frequency (pandas/polars friction demonstrators)
    "mixed_freq_qq_minus_mm_mean": _setup_mixed_freq_qq_minus_mm_mean,
    "mixed_freq_pipeline_three_freq": _setup_mixed_freq_pipeline_three_freq,
    # various.jl helpers (M1.6.3b)
    "overlay_three_tseries": _setup_overlay_three_tseries,
    "compare_workspaces_equal_5_keys": _setup_compare_workspaces_equal_5_keys,
    "compare_workspaces_differ_5_keys": _setup_compare_workspaces_differ_5_keys,
    "reindex_tseries_100": _setup_reindex_tseries_100,
    # rangeof (M1.6.3c — closes G5)
    "rangeof_tseries_drop1": _setup_rangeof_tseries_drop1,
    # linalg (M1.6.3g — closes G12)
    "linalg_matrix_tseries_100": _setup_linalg_matrix_tseries_100,
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
    # Indexing (M1.5 second Cython port — kernel-direct and public API)
    "indexing_lookup_100_api": _run_indexing_lookup_100_api,
    "indexing_lookup_100_numpy": _run_indexing_gather_100_numpy,
    # Arithmetic
    "arith_add_misaligned": _run_arith_add_misaligned,
    "arith_add_aligned": _run_arith_add_aligned,
    "arith_mul_scalar": _run_arith_mul_scalar,
    # Shift family
    "shift_quarterly_lag1": _run_shift_quarterly_lag1,
    "lead_quarterly_lag1": _run_lead_quarterly_lag1,
    "diff_quarterly": _run_diff_quarterly,
    "pct_quarterly": _run_pct_quarterly,
    "ytypct_quarterly_100": _run_ytypct_quarterly_100,
    # Stats
    "mean_quarterly_100": _run_mean_quarterly_100,
    "std_quarterly_100": _run_std_quarterly_100,
    "quantile_quarterly_100": _run_quantile_quarterly_100,
    "cor_two_tseries": _run_cor_two_tseries,
    "cov_two_tseries": _run_cov_two_tseries,
    "cor_mvts_5_columns": _run_cor_mvts_5_columns,
    "cov_mvts_5_columns": _run_cov_mvts_5_columns,
    # MVTSeries axis= reductions (M1.6.3f — closes G11)
    "mean_mvts_axis0_5cols": _run_mean_mvts_axis0_5cols,
    "mean_mvts_axis1_100rows": _run_mean_mvts_axis1_100rows,
    # Stats (M1.5 third Cython port — kernel-direct + public API)
    "mean_quarterly_100_numpy": _run_mean_quarterly_100_numpy,
    "std_quarterly_100_numpy": _run_std_quarterly_100_numpy,
    "cor_two_tseries_numpy": _run_cor_two_tseries_numpy,
    # Moving / undiff
    "moving_average_quarterly_4": _run_moving_average_quarterly_4,
    "moving_sum_quarterly_4": _run_moving_sum_quarterly_4,
    "undiff_quarterly": _run_undiff_quarterly,
    # undiff (M1.6.2 fifth Cython port — kernel-direct + public API)
    "undiff_quarterly_numpy": _run_undiff_quarterly_numpy,
    # fconvert
    "fconvert_qq_to_yy_mean": _run_fconvert_qq_to_yy_mean,
    "fconvert_qq_to_yy_sum": _run_fconvert_qq_to_yy_sum,
    "fconvert_yy_to_qq_const": _run_fconvert_yy_to_qq_const,
    "fconvert_yy_to_qq_linear": _run_fconvert_yy_to_qq_linear,
    "fconvert_yy_to_qq_even": _run_fconvert_yy_to_qq_even,
    "fconvert_mm_to_qq_mean": _run_fconvert_mm_to_qq_mean,
    # fconvert (M1.5 fourth Cython port — kernel-direct + public API)
    "fconvert_qq_to_yy_mean_numpy": _run_fconvert_qq_to_yy_mean_numpy,
    "fconvert_qq_to_yy_sum_numpy": _run_fconvert_qq_to_yy_sum_numpy,
    "fconvert_mm_to_qq_mean_numpy": _run_fconvert_mm_to_qq_mean_numpy,
    # Recursion (general)
    "rec_ar2_100": _run_rec_ar2_100,
    "rec_backcasting_via_lambda": _run_rec_backcasting_via_lambda,
    # Recursion (kernel-direct, three / four flavor)
    "rec_linear_ar2_100_pylist": _run_rec_linear_ar2_100_pylist,
    "rec_linear_ar2_100_numpy": _run_rec_linear_ar2_100_numpy,
    # Workspace
    "workspace_merge_5_series": _run_workspace_merge_5_series,
    "workspace_filter_5_series": _run_workspace_filter_5_series,
    # copyto (M1.6.3h — closes G13)
    "workspace_to_mvts_copyto_5cols": _run_workspace_to_mvts_copyto_5cols,
    # Mixed-frequency (pandas/polars friction demonstrators)
    "mixed_freq_qq_minus_mm_mean": _run_mixed_freq_qq_minus_mm_mean,
    "mixed_freq_pipeline_three_freq": _run_mixed_freq_pipeline_three_freq,
    # various.jl helpers (M1.6.3b)
    "overlay_three_tseries": _run_overlay_three_tseries,
    "compare_workspaces_equal_5_keys": _run_compare_workspaces_equal_5_keys,
    "compare_workspaces_differ_5_keys": _run_compare_workspaces_differ_5_keys,
    "reindex_tseries_100": _run_reindex_tseries_100,
    # rangeof (M1.6.3c — closes G5)
    "rangeof_tseries_drop1": _run_rangeof_tseries_drop1,
    # linalg (M1.6.3g — closes G12)
    "linalg_matrix_tseries_100": _run_linalg_matrix_tseries_100,
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
    "indexing_lookup_100_api": "lookup(t, mit_keys) — public vectorised API",
    "indexing_lookup_100_numpy": "gather_numpy(values, ix) — NumPy kernel",
    "arith_add_misaligned": "100Q + 100Q with 50Q overlap",
    "arith_add_aligned": "100Q + 100Q same range",
    "arith_mul_scalar": "t * 2.5",
    "shift_quarterly_lag1": "shift(t, -1)",
    "lead_quarterly_lag1": "lead(t, 1)",
    "diff_quarterly": "diff(t)",
    "pct_quarterly": "pct(t)",
    "ytypct_quarterly_100": "ytypct(t) — year-on-year %",
    "mean_quarterly_100": "mean(t)",
    "std_quarterly_100": "std(t)",
    "quantile_quarterly_100": "quantile(t, 0.5) — median",
    "cor_two_tseries": "cor(a, b) on two TSeries",
    "cov_two_tseries": "cov(a, b) on two TSeries",
    "cor_mvts_5_columns": "cor(mvts) — 5x5 corr matrix",
    "cov_mvts_5_columns": "cov(mvts) — 5x5 cov matrix",
    "mean_mvts_axis0_5cols": "mean(mvts, axis=0) — per-column → 1-row MVTSeries",
    "mean_mvts_axis1_100rows": "mean(mvts, axis=1) — per-row → 100-row TSeries",
    "mean_quarterly_100_numpy": "mean_numpy(values) — NumPy kernel",
    "std_quarterly_100_numpy": "std_numpy(values, 1) — NumPy kernel",
    "cor_two_tseries_numpy": "cor_numpy(x, y) — NumPy kernel",
    "moving_average_quarterly_4": "moving_average(t, 4)",
    "moving_sum_quarterly_4": "moving_sum(t, 4)",
    "undiff_quarterly": "undiff(t)",
    "undiff_quarterly_numpy": "cumsum_anchored_numpy 101 anchored at 0 — NumPy kernel",
    "fconvert_qq_to_yy_mean": "fconvert(Yearly, t, method='mean')",
    "fconvert_qq_to_yy_sum": "fconvert(Yearly, t, method='sum')",
    "fconvert_yy_to_qq_const": "fconvert(Quarterly, t, method='const') (higher)",
    "fconvert_yy_to_qq_linear": "fconvert(Quarterly, t, method='linear') (higher)",
    "fconvert_yy_to_qq_even": "fconvert(Quarterly, t, method='even') (higher)",
    "fconvert_mm_to_qq_mean": "fconvert(Quarterly, monthly_t, method='mean')",
    "fconvert_qq_to_yy_mean_numpy": "aggregate_groups_numpy 25x4 mean - NumPy kernel",
    "fconvert_qq_to_yy_sum_numpy": "aggregate_groups_numpy 25x4 sum - NumPy kernel",
    "fconvert_mm_to_qq_mean_numpy": "aggregate_groups_numpy 40x3 mean - NumPy kernel",
    "rec_ar2_100": "AR(2) over 100 quarters — general rec + lambda",
    "rec_backcasting_via_lambda": "Backcast over 100 quarters — reversed range + rec lambda",
    "rec_linear_ar2_100_pylist": "AR(2) over 100 — rec_linear, pure-Python list",
    "rec_linear_ar2_100_numpy": "AR(2) over 100 — rec_linear NumPy kernel",
    "workspace_merge_5_series": "Workspace merge: 5 + 5 series",
    "workspace_filter_5_series": "Workspace filter: 10 down to 5 series",
    "workspace_to_mvts_copyto_5cols": "copyto(MVTSeries, Workspace) — 100Q × 5 cols, in-place",
    "mixed_freq_qq_minus_mm_mean": "qq_gdp - fconvert(Q, mm_cpi, mean) — mixed freq",
    "mixed_freq_pipeline_three_freq": "Y+Q+M → quarterly via fconvert — mixed freq",
    "overlay_three_tseries": "overlay(a, b, c) — 100Q three-way first-non-NaN",
    "compare_workspaces_equal_5_keys": "compare(w1, w2) — 5×TSeries, equal",
    "compare_workspaces_differ_5_keys": "compare(w1, w2) — 5×TSeries, one diff",
    "reindex_tseries_100": "reindex(t, qq=>1U) — 100Q label shift",
    "rangeof_tseries_drop1": "rangeof(t, drop=1) — 100Q tutorial-1 @rec idiom",
    "linalg_matrix_tseries_100": "A @ t — 100x100 matrix × length-100 TSeries (strips labels)",
}

# Cython kernel scenarios are conditionally registered: when the wheel
# was built without the C toolchain (Windows installs without the SDK,
# editable installs that skipped the build hook), the Cython column
# legitimately reads `n/a`. Adding a scenario only when the kernel is
# importable keeps the "missing data" honest in the table — see
# decision 17 on three-flavor reporting.
if _REC_CYTHON_AVAILABLE:
    SETUP["rec_linear_ar2_100_cython"] = _setup_rec_linear_ar2_100
    RUN["rec_linear_ar2_100_cython"] = _run_rec_linear_ar2_100_cython
    DESCRIPTION["rec_linear_ar2_100_cython"] = "AR(2) over 100 quarters — rec_linear Cython kernel"

if _INDEXING_CYTHON_AVAILABLE:
    SETUP["indexing_lookup_100_cython"] = _setup_indexing_gather_100_kernel
    RUN["indexing_lookup_100_cython"] = _run_indexing_gather_100_cython
    DESCRIPTION["indexing_lookup_100_cython"] = "gather_cython(values, ix) — Cython kernel"

if _STATS_CYTHON_AVAILABLE:
    SETUP["mean_quarterly_100_cython"] = _setup_stats_scalar_kernel_100
    RUN["mean_quarterly_100_cython"] = _run_mean_quarterly_100_cython
    DESCRIPTION["mean_quarterly_100_cython"] = "mean_cython(values) — Cython kernel"

    SETUP["std_quarterly_100_cython"] = _setup_stats_scalar_kernel_100
    RUN["std_quarterly_100_cython"] = _run_std_quarterly_100_cython
    DESCRIPTION["std_quarterly_100_cython"] = "std_cython(values, 1) — Cython kernel"

    SETUP["cor_two_tseries_cython"] = _setup_cor_two_kernel
    RUN["cor_two_tseries_cython"] = _run_cor_two_tseries_cython
    DESCRIPTION["cor_two_tseries_cython"] = "cor_cython(x, y) — Cython kernel"

if _FCONVERT_CYTHON_AVAILABLE:
    SETUP["fconvert_qq_to_yy_mean_cython"] = _setup_fconvert_qq_to_yy_kernel
    RUN["fconvert_qq_to_yy_mean_cython"] = _run_fconvert_qq_to_yy_mean_cython
    DESCRIPTION["fconvert_qq_to_yy_mean_cython"] = (
        "aggregate_groups_cython 25x4 mean - Cython kernel"
    )

    SETUP["fconvert_qq_to_yy_sum_cython"] = _setup_fconvert_qq_to_yy_kernel
    RUN["fconvert_qq_to_yy_sum_cython"] = _run_fconvert_qq_to_yy_sum_cython
    DESCRIPTION["fconvert_qq_to_yy_sum_cython"] = "aggregate_groups_cython 25x4 sum - Cython kernel"

    SETUP["fconvert_mm_to_qq_mean_cython"] = _setup_fconvert_mm_to_qq_kernel
    RUN["fconvert_mm_to_qq_mean_cython"] = _run_fconvert_mm_to_qq_mean_cython
    DESCRIPTION["fconvert_mm_to_qq_mean_cython"] = (
        "aggregate_groups_cython 40x3 mean - Cython kernel"
    )

if _MATH_CYTHON_AVAILABLE:
    SETUP["undiff_quarterly_cython"] = _setup_undiff_kernel_quarterly
    RUN["undiff_quarterly_cython"] = _run_undiff_quarterly_cython
    DESCRIPTION["undiff_quarterly_cython"] = (
        "cumsum_anchored_cython 101 anchored at 0 — Cython kernel"
    )

# X-13 deseasonalisation depends on the vendored x13as binary; register the
# scenario only when one is reachable so that wheel-build matrices without
# the binary (or local checkouts that have not run scripts/fetch_x13as_local.py)
# do not produce a misleading n/a row. The Julia side mirrors this gate.
if _x13_binary_available():
    SETUP["deseasonalize_quarterly_50y"] = _setup_deseasonalize_quarterly_50y
    RUN["deseasonalize_quarterly_50y"] = _run_deseasonalize_quarterly_50y
    DESCRIPTION["deseasonalize_quarterly_50y"] = "deseasonalize(t) — 200Q via x13as binary (M2.6)"

assert SETUP.keys() == RUN.keys() == DESCRIPTION.keys(), "scenario registries must agree"
