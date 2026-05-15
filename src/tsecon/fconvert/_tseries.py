# SPDX-License-Identifier: MIT
"""Frequency conversion for :class:`TSeries` (YP only).

Mirrors ``TimeSeriesEcon.jl/src/fconvert/fconvert_tseries.jl`` for the YP-only
subset:

* :func:`fconvert_tseries` — the public entry. Accepts an optional first
  positional callable that replaces the built-in aggregator / spreader.
* Higher-frequency conversion (e.g. Yearly → Monthly) supports
  ``method="const" / "even" / "linear"``.
* Lower-frequency conversion (e.g. Monthly → Quarterly) supports
  ``method="mean" / "sum" / "min" / "max" / "point" / "begin" / "end"``.
* :func:`extend_series` / :func:`trim_series` — helpers from
  ``fconvert_helpers.jl`` that operate against a target YP frequency.

The Calendar / BDaily / Weekly variants are deferred to the follow-up session.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np

from tsecon.fconvert._helpers import (
    Ref,
    _get_end_truncation_yp,
    _get_start_truncation_yp,
    linear_uneven,
)
from tsecon.fconvert._mit import (
    fconvert_mit,
    fconvert_parts,
    fconvert_range,
)
from tsecon.frequencies import (
    Frequency,
    FrequencyLike,
    Unit,
    YPFrequency,
    ppy,
    sanitize_frequency,
)
from tsecon.mit import MIT
from tsecon.mitrange import MITRange
from tsecon.tseries import TSeries

__all__ = ["extend_series", "fconvert_tseries", "trim_series"]


_LowerMethod = Literal["mean", "sum", "min", "max", "point", "begin", "end"]
_HigherMethod = Literal["const", "even", "linear"]
_AnyMethod = Literal[
    "mean", "sum", "min", "max", "point", "begin", "end", "const", "even", "linear"
]
ExtendDirection = Literal["both", "begin", "end"]
ExtendMethod = Literal["mean", "end"]


_LOWER_AGGREGATORS: dict[str, Callable[[np.ndarray], Any]] = {
    "mean": lambda col: float(np.mean(col)),
    "sum": lambda col: col.sum(),
    "min": lambda col: col.min(),
    "max": lambda col: col.max(),
}


def _require_yp(f: Frequency, *, what: str) -> YPFrequency:
    if isinstance(f, Unit):
        msg = f"Cannot fconvert {what} to or from Unit."
        raise ValueError(msg)
    if not isinstance(f, YPFrequency):
        msg = (
            f"YP-only fconvert: {what} must be a YPFrequency (Yearly / HalfYearly / "
            f"Quarterly / Monthly), got {type(f).__name__}. The Calendar-frequency "
            "variants (Daily / BDaily / Weekly) are not yet ported."
        )
        raise NotImplementedError(msg)
    return f


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def fconvert_tseries(
    arg1: FrequencyLike | Callable[..., Any],
    arg2: TSeries | FrequencyLike,
    arg3: TSeries | None = None,
    *,
    method: _AnyMethod | None = None,
    ref: Ref | None = None,
    **kwargs: Any,
) -> TSeries:
    """Convert ``t`` to ``target`` frequency.

    Two call shapes mirror Julia's two ``fconvert`` overloads:

    * ``fconvert_tseries(target, t, *, method=..., ref=...)`` — built-in
      aggregator / spreader. ``method`` is one of ``"mean" / "sum" / "min" /
      "max" / "point" / "begin" / "end"`` for lower-frequency conversions and
      ``"const" / "even" / "linear"`` for higher-frequency conversions.
    * ``fconvert_tseries(f, target, t, *, ref=..., **kwargs)`` — pass a custom
      callable. For lower-frequency conversions ``f`` must accept a single
      vector and return a scalar; for higher-frequency conversions it must
      accept ``(values, output_lengths)`` plus the keyword arguments
      ``ref=...`` and ``outrange=...``.

    Examples
    --------
    >>> from tsecon import qq, mm, TSeries, Quarterly, Monthly, Yearly
    >>> import numpy as np
    >>> q = TSeries(qq(5, 1), np.arange(1, 11, dtype=float))
    >>> fconvert_tseries(Monthly, q, method="const").firstdate
    5M1
    """
    target, t, f = _normalise_call(arg1, arg2, arg3)

    f_to = sanitize_frequency(target)
    f_from = t.frequency
    if f_to == f_from:
        return t  # mirror Julia's identity overload

    f_to = _require_yp(f_to, what="target")
    f_from = _require_yp(f_from, what="source")

    n_to = ppy(f_to)
    n_from = ppy(f_from)

    if n_to > n_from:  # → higher frequency
        if method is not None and method not in ("const", "even", "linear"):
            msg = (
                f"fconvert: method must be 'const', 'even', or 'linear' when converting "
                f"to a higher frequency. Received: {method!r}."
            )
            raise ValueError(msg)
        higher_ref: Ref = ref if ref in ("begin", "end") else "end"
        if f is not None:
            return _fconvert_higher_with_function(f_to, t, f, ref=higher_ref, **kwargs)
        higher_method: _HigherMethod = method if method is not None else "const"
        return _fconvert_higher_yp_to_yp(f_to, t, method=higher_method, ref=higher_ref)

    # → lower frequency
    if method is not None and method not in ("mean", "sum", "min", "max", "point", "begin", "end"):
        msg = (
            f"fconvert: method must be 'mean', 'sum', 'min', 'max', 'point', 'begin', or 'end' "
            f"when converting to a lower frequency. Received: {method!r}."
        )
        raise ValueError(msg)
    lower_ref: Ref = ref if ref in ("begin", "end") else "end"
    if f is not None:
        return _fconvert_lower_with_function(f_to, t, f, ref=lower_ref)
    lower_method: _LowerMethod = method if method is not None else "mean"
    return _fconvert_lower_yp_to_yp(f_to, t, method=lower_method, ref=lower_ref)


def _normalise_call(
    arg1: FrequencyLike | Callable[..., Any],
    arg2: TSeries | FrequencyLike,
    arg3: TSeries | None,
) -> tuple[FrequencyLike, TSeries, Callable[..., Any] | None]:
    """Normalise the two Julia call shapes into ``(target, t, optional_f)``."""
    if arg3 is not None:
        # fconvert(f, target, t) — arg3's TSeries-ness is enforced at the
        # public ``fconvert`` entry; here ``arg3`` is already typed ``TSeries``.
        if not callable(arg1):
            msg = (
                "fconvert: when called with three positional arguments the first must be a "
                f"callable (custom aggregator / spreader). Got: {type(arg1).__name__}."
            )
            raise TypeError(msg)
        return cast("FrequencyLike", arg2), arg3, arg1
    # fconvert(target, t)
    if not isinstance(arg2, TSeries):
        msg = (
            "fconvert: second positional argument must be a TSeries (or pass a custom function "
            f"as the first of three positional arguments). Got: {type(arg2).__name__}."
        )
        raise TypeError(msg)
    if callable(arg1) and not isinstance(arg1, (type, Frequency)):
        msg = (
            "fconvert: when passing a custom function as the first argument, the target "
            "frequency must be the second positional argument and the TSeries the third."
        )
        raise TypeError(msg)
    return cast("FrequencyLike", arg1), arg2, None


# ---------------------------------------------------------------------------
# Higher-frequency conversion (YP → YP)
# ---------------------------------------------------------------------------


def _fconvert_higher_get_fi(f_to: YPFrequency, first_mit: MIT, *, ref: Ref) -> MIT:
    fi_to_period, fi_from_start_month, fi_to_start_month = fconvert_parts(
        f_to, first_mit, ref="begin"
    )
    if ref == "end":
        mpp_to = 12 // ppy(f_to)
        fi_to_end_month = fi_to_start_month + mpp_to - 1
        trunc_start = 1 if fi_to_end_month < fi_from_start_month else 0
    else:
        trunc_start = 1 if fi_to_start_month < fi_from_start_month else 0
    return MIT(f_to, fi_to_period + trunc_start)


def _fconvert_higher_yp_to_yp(
    f_to: YPFrequency,
    t: TSeries,
    *,
    method: _HigherMethod,
    ref: Ref,
) -> TSeries:
    f_from = cast("YPFrequency", t.frequency)
    np_count = ppy(f_to) // ppy(f_from)
    fi = _fconvert_higher_get_fi(f_to, t.firstdate, ref=ref)
    if method == "const":
        out = np.repeat(t.values, np_count)
    elif method == "even":
        out = np.repeat(np.asarray(t.values, dtype=np.float64) / np_count, np_count)
    else:  # method == "linear"
        out = linear_uneven(t.values, np.full(len(t), np_count, dtype=np.int64), ref=ref)
    return TSeries(fi, out)


def _fconvert_higher_with_function(
    f_to: YPFrequency,
    t: TSeries,
    f: Callable[..., Any],
    *,
    ref: Ref,
    **kwargs: Any,
) -> TSeries:
    f_from = cast("YPFrequency", t.frequency)
    np_count = ppy(f_to) // ppy(f_from)
    fi = _fconvert_higher_get_fi(f_to, t.firstdate, ref=ref)
    output_lengths = np.full(len(t), np_count, dtype=np.int64)
    outrange = MITRange(fi, MIT(f_to, int(fi) + np_count * len(t) - 1))
    out = np.asarray(f(t.values, output_lengths, ref=ref, outrange=outrange, **kwargs))
    return TSeries(fi, out)


# ---------------------------------------------------------------------------
# Lower-frequency conversion (YP → YP)
# ---------------------------------------------------------------------------


def _fconvert_lower_yp_to_yp(
    f_to: YPFrequency,
    t: TSeries,
    *,
    method: _LowerMethod,
    ref: Ref,
) -> TSeries:
    if method == "begin":
        return _fconvert_lower_point_yp(f_to, t, ref="begin")
    if method == "end":
        return _fconvert_lower_point_yp(f_to, t, ref="end")
    if method == "point":
        return _fconvert_lower_point_yp(f_to, t, ref=ref)
    aggregator = _LOWER_AGGREGATORS[method]
    return _fconvert_lower_aggregate_yp(f_to, t, aggregator, ref=ref)


def _lower_out_range_with_truncation(
    f_to: YPFrequency,
    t: TSeries,
    *,
    require: Literal["single", "all"],
    ref: Ref,
) -> tuple[
    MITRange,  # output range
    int,  # mpp_from
    int,  # mpp_to
    int,  # np_count
    int,  # trunc_start
    int,  # fi_from_start_month
    int,  # fi_to_start_month
]:
    f_from = cast("YPFrequency", t.frequency)
    mpp_from = 12 // ppy(f_from)
    mpp_to = 12 // ppy(f_to)
    n_from = ppy(f_from)
    n_to = ppy(f_to)
    np_count = n_from // n_to
    parts = fconvert_range(f_to, t.range, trim="both", parts=True)
    (
        fi_to_period,
        fi_from_start_month,
        fi_to_start_month,
        li_to_period,
        li_from_end_month,
        li_to_end_month,
    ) = parts
    trunc_start = _get_start_truncation_yp(
        ref, require, fi_from_start_month, fi_to_start_month, mpp_from, mpp_to
    )
    trunc_end = _get_end_truncation_yp(
        ref, require, li_from_end_month, li_to_end_month, mpp_from, mpp_to
    )
    fi = MIT(f_to, fi_to_period + trunc_start)
    li = MIT(f_to, li_to_period - trunc_end)
    return (
        MITRange(fi, li),
        mpp_from,
        mpp_to,
        np_count,
        trunc_start,
        fi_from_start_month,
        fi_to_start_month,
    )


def _fconvert_lower_aggregate_yp(
    f_to: YPFrequency,
    t: TSeries,
    aggregator: Callable[[np.ndarray], Any],
    *,
    ref: Ref,
) -> TSeries:
    (
        out_range,
        mpp_from,
        mpp_to,
        np_count,
        trunc_start,
        fi_from_start_month,
        fi_to_start_month,
    ) = _lower_out_range_with_truncation(f_to, t, require="all", ref=ref)
    fi_truncation_adjustment = mpp_to if trunc_start == 1 else 0
    begin_adjustment = mpp_from - 1 if ref == "begin" else 0
    months_of_misalignment = (
        fi_to_start_month - fi_from_start_month + fi_truncation_adjustment + begin_adjustment
    )
    periods_of_misalignment = math.floor(months_of_misalignment / mpp_from)
    start_index = periods_of_misalignment
    end_index = start_index + np_count * len(out_range)
    chunk = np.asarray(t.values[start_index:end_index])
    # Julia is column-major: ``reshape(vec, np, :)`` groups consecutive values
    # into columns; aggregating across each column. NumPy's row-major
    # ``reshape(-1, np)`` puts those same groups into rows, so we aggregate
    # across each row (axis=1).
    if len(out_range) == 0:
        return TSeries(out_range.start, np.empty(0, dtype=t.dtype))
    grouped = chunk.reshape(len(out_range), np_count)
    out = np.array([aggregator(row) for row in grouped])
    return TSeries(out_range.start, out)


def _fconvert_lower_point_yp(f_to: YPFrequency, t: TSeries, *, ref: Ref) -> TSeries:
    (
        out_range,
        mpp_from,
        mpp_to,
        np_count,
        trunc_start,
        fi_from_start_month,
        fi_to_start_month,
    ) = _lower_out_range_with_truncation(f_to, t, require="single", ref=ref)
    fi_truncation_adjustment = mpp_to if trunc_start == 1 else 0
    if ref == "end":
        fi_from_end_month = fi_from_start_month + mpp_from - 1
        fi_to_end_month = fi_to_start_month + mpp_to - 1
        months_of_misalignment = fi_to_end_month - fi_from_end_month + fi_truncation_adjustment
    else:
        months_of_misalignment = fi_to_start_month - fi_from_start_month + fi_truncation_adjustment
    periods_of_misalignment = math.floor(months_of_misalignment / mpp_from)
    candidates = range(periods_of_misalignment, len(t.values), np_count)
    indices = [i for i in candidates if i >= 0][: len(out_range)]
    out = np.asarray(t.values)[indices]
    return TSeries(out_range.start, out)


def _fconvert_lower_with_function(
    f_to: YPFrequency,
    t: TSeries,
    f: Callable[..., Any],
    *,
    ref: Ref,
) -> TSeries:
    """Custom-function lower-frequency conversion: groups inputs by output MIT, applies ``f``."""
    (
        out_range,
        mpp_from,
        mpp_to,
        np_count,
        trunc_start,
        fi_from_start_month,
        fi_to_start_month,
    ) = _lower_out_range_with_truncation(f_to, t, require="all", ref=ref)
    fi_truncation_adjustment = mpp_to if trunc_start == 1 else 0
    begin_adjustment = mpp_from - 1 if ref == "begin" else 0
    months_of_misalignment = (
        fi_to_start_month - fi_from_start_month + fi_truncation_adjustment + begin_adjustment
    )
    periods_of_misalignment = math.floor(months_of_misalignment / mpp_from)
    start_index = periods_of_misalignment
    end_index = start_index + np_count * len(out_range)
    chunk = np.asarray(t.values[start_index:end_index])
    if len(out_range) == 0:
        return TSeries(out_range.start, np.empty(0, dtype=t.dtype))
    grouped = chunk.reshape(len(out_range), np_count)
    out = np.array([f(row) for row in grouped])
    return TSeries(out_range.start, out)


# ---------------------------------------------------------------------------
# extend_series / trim_series
# ---------------------------------------------------------------------------


def extend_series(
    target: FrequencyLike,
    t: TSeries,
    *,
    direction: ExtendDirection = "both",
    method: ExtendMethod = "mean",
) -> TSeries:
    """Pad a TSeries to align with target-frequency period boundaries.

    Returns a new TSeries (the input is not mutated). ``method="mean"`` fills
    each padded section with the mean of the existing values that fall within
    the spanned target period; ``method="end"`` repeats the nearest in-range
    value (last value for the trailing pad, first in-range value for the
    leading pad — matching Julia's behaviour when there is exactly one value
    in the spanning period).

    Raises ``NotImplementedError`` if either frequency is BDaily / Daily /
    Weekly (those code paths land with the calendar-frequency port).
    """
    if direction not in ("both", "begin", "end"):
        msg = f"direction must be 'both', 'begin', or 'end'. Received: {direction!r}."
        raise ValueError(msg)
    if method not in ("mean", "end"):
        msg = f"method must be 'mean' or 'end'. Received: {method!r}."
        raise ValueError(msg)
    f_to = _require_yp(sanitize_frequency(target), what="target")
    _require_yp(t.frequency, what="source")
    out = t.copy()
    if direction in ("begin", "both"):
        _extend_begin(f_to, out, method=method)
    if direction in ("end", "both"):
        _extend_end(f_to, out, method=method)
    return out


def _extend_begin(f_to: YPFrequency, t: TSeries, *, method: ExtendMethod) -> None:
    f_from = cast("YPFrequency", t.frequency)
    first_mit_in_output_freq = fconvert_mit(f_to, t.firstdate)  # ref="end"
    desired_first_mit = fconvert_mit(f_from, first_mit_in_output_freq, ref="begin")
    if desired_first_mit.value >= t.firstdate.value:
        return
    affected = MITRange(desired_first_mit, MIT(f_from, t.firstdate.value - 1))
    if method == "end":
        # Repeat the source's first in-range value. Julia uses
        # ``ts[first(affected_range)+1]`` but that only lands inside the
        # original series when the gap is exactly one step; the semantic
        # value is the first existing entry of t.
        fill = t.values[0]
    else:  # method == "mean"
        single = MITRange(first_mit_in_output_freq, first_mit_in_output_freq)
        data_basis = fconvert_range(f_from, single)
        lo = max(data_basis.first().value, t.firstdate.value)
        hi = min(data_basis.last().value, t.lastdate.value)
        if lo > hi:
            return
        sub = MITRange(MIT(f_from, lo), MIT(f_from, hi))
        fill = float(np.mean(t[sub].values))
    t[affected] = fill


def _extend_end(f_to: YPFrequency, t: TSeries, *, method: ExtendMethod) -> None:
    f_from = cast("YPFrequency", t.frequency)
    last_mit_in_output_freq = fconvert_mit(f_to, t.lastdate)  # ref="end"
    desired_last_mit = fconvert_mit(f_from, last_mit_in_output_freq, ref="end")
    if desired_last_mit.value <= t.lastdate.value:
        return
    affected = MITRange(MIT(f_from, t.lastdate.value + 1), desired_last_mit)
    if method == "end":
        fill = t.values[-1]
    else:  # method == "mean"
        single = MITRange(last_mit_in_output_freq, last_mit_in_output_freq)
        data_basis = fconvert_range(f_from, single)
        lo = max(data_basis.first().value, t.firstdate.value)
        hi = min(data_basis.last().value, t.lastdate.value)
        if lo > hi:
            return
        sub = MITRange(MIT(f_from, lo), MIT(f_from, hi))
        fill = float(np.mean(t[sub].values))
    t[affected] = fill


def trim_series(
    target: FrequencyLike,
    t: TSeries,
    *,
    direction: ExtendDirection = "both",
) -> TSeries:
    """Trim a TSeries to ranges that align with target-frequency boundaries.

    Returns a new TSeries restricted to the inner range produced by
    ``fconvert(F_from, fconvert(F_to, rangeof(t), trim=direction))``. Mirrors
    Julia's ``trim_series``.
    """
    if direction not in ("both", "begin", "end"):
        msg = f"direction must be 'both', 'begin', or 'end'. Received: {direction!r}."
        raise ValueError(msg)
    f_to = _require_yp(sanitize_frequency(target), what="target")
    f_from = _require_yp(t.frequency, what="source")
    target_range = fconvert_range(f_to, t.range, trim=direction)
    back = fconvert_range(f_from, target_range)
    return cast("TSeries", t[back])
