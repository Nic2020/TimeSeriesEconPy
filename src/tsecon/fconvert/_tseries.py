# SPDX-License-Identifier: MIT
"""Frequency conversion for :class:`TSeries`.

Mirrors ``TimeSeriesEcon.jl/src/fconvert/fconvert_tseries.jl``:

* :func:`fconvert_tseries` — the public entry. Accepts an optional first
  positional callable that replaces the built-in aggregator / spreader.
* Higher-frequency conversion (e.g. Yearly → Monthly / Weekly / Daily / BDaily)
  supports ``method="const" / "even" / "linear"``.
* Lower-frequency conversion (e.g. Monthly → Quarterly, Daily → Monthly)
  supports ``method="mean" / "sum" / "min" / "max" / "point" / "begin" / "end"``.
* :func:`extend_series` / :func:`trim_series` — helpers from
  ``fconvert_helpers.jl`` that operate against a target frequency.

The BDaily-specific kwarg variants (``skip_holidays`` / ``skip_all_nans`` /
``holidays_map``) block on the ``options.jl`` port and are still ⬜.
"""

from __future__ import annotations

import datetime as _dt
import math
from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np

from tsecon.fconvert._helpers import (
    Ref,
    _fconvert_using_dates_parts,
    _freq_is_higher,
    _get_end_truncation_yp,
    _get_out_indices,
    _get_start_truncation_yp,
    linear_uneven,
)
from tsecon.fconvert._mit import (
    fconvert_mit,
    fconvert_parts,
    fconvert_range,
)
from tsecon.frequencies import (
    BDaily,
    CalendarFrequency,
    Daily,
    Frequency,
    FrequencyLike,
    Unit,
    Weekly,
    YPFrequency,
    ppy,
    sanitize_frequency,
)
from tsecon.mit import MIT, bdaily, daily, mit_to_date
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


def _agg_first(col: np.ndarray) -> Any:
    return col[0]


def _agg_last(col: np.ndarray) -> Any:
    return col[-1]


_LOWER_AGGREGATORS: dict[str, Callable[[np.ndarray], Any]] = {
    "mean": lambda col: float(np.mean(col)),
    "sum": lambda col: col.sum(),
    "min": lambda col: col.min(),
    "max": lambda col: col.max(),
}


def _reject_unit(f: Frequency, *, what: str) -> None:
    if isinstance(f, Unit):
        msg = f"Cannot fconvert {what} to or from Unit."
        raise ValueError(msg)


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
        return t  # mirror Julia's identity overload (works for Unit too)
    _reject_unit(f_to, what="target")
    _reject_unit(f_from, what="source")

    direction = _dispatch_direction(f_to, f_from)
    if direction == "higher":
        return _dispatch_higher(f_to, t, f, method=method, ref=ref, **kwargs)
    return _dispatch_lower(f_to, t, f, method=method, ref=ref, **kwargs)


def _normalise_call(
    arg1: FrequencyLike | Callable[..., Any],
    arg2: TSeries | FrequencyLike,
    arg3: TSeries | None,
) -> tuple[FrequencyLike, TSeries, Callable[..., Any] | None]:
    """Normalise the two Julia call shapes into ``(target, t, optional_f)``."""
    if arg3 is not None:
        if not callable(arg1):
            msg = (
                "fconvert: when called with three positional arguments the first must be a "
                f"callable (custom aggregator / spreader). Got: {type(arg1).__name__}."
            )
            raise TypeError(msg)
        return cast("FrequencyLike", arg2), arg3, arg1
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


def _dispatch_direction(f_to: Frequency, f_from: Frequency) -> Literal["higher", "lower"]:
    """Pick the higher/lower branch.

    Special case: Daily → BDaily and BDaily → Daily are siblings; Julia routes
    Daily → BDaily through the lower-frequency dispatcher and BDaily → Daily
    through the higher-frequency one, because Daily has 365 ppy and BDaily 260.
    The generic ``ppy``-based check picks the right branch.
    """
    return "higher" if _freq_is_higher(f_to, f_from) else "lower"


# ---------------------------------------------------------------------------
# Higher-frequency dispatch
# ---------------------------------------------------------------------------


def _dispatch_higher(
    f_to: Frequency,
    t: TSeries,
    f: Callable[..., Any] | None,
    *,
    method: _AnyMethod | None,
    ref: Ref | None,
    **kwargs: Any,
) -> TSeries:
    f_from = t.frequency
    # Validate method when no custom function is supplied.
    if f is None and method is not None and method not in ("const", "even", "linear"):
        msg = (
            f"fconvert: method must be 'const', 'even', or 'linear' when converting "
            f"to a higher frequency. Received: {method!r}."
        )
        raise ValueError(msg)
    eff_ref: Ref = ref if ref in ("begin", "end") else "end"
    higher_method: _HigherMethod = method if method in ("const", "even", "linear") else "const"

    # YP → YP: closed-form np_count repeats.
    if isinstance(f_to, YPFrequency) and isinstance(f_from, YPFrequency):
        if f is not None:
            return _fconvert_higher_yp_to_yp_with_function(f_to, t, f, ref=eff_ref, **kwargs)
        return _fconvert_higher_yp_to_yp(f_to, t, method=higher_method, ref=eff_ref)

    # BDaily → Daily: dedicated path with weekend infill.
    if isinstance(f_to, Daily) and isinstance(f_from, BDaily):
        if f is not None:
            msg = "fconvert: custom-function dispatch is not supported for BDaily → Daily."
            raise NotImplementedError(msg)
        return _fconvert_higher_bdaily_to_daily(t, method=higher_method, ref=eff_ref)

    # YP / Weekly → Weekly: uneven spreader using date boundaries.
    if isinstance(f_to, Weekly) and isinstance(f_from, YPFrequency):
        if f is not None:
            return _fconvert_higher_yp_to_weekly_with_function(f_to, t, f, ref=eff_ref, **kwargs)
        return _fconvert_higher_yp_to_weekly(f_to, t, method=higher_method, ref=eff_ref)

    # YP / Weekly → Daily / BDaily: per-period output-length spreader.
    if isinstance(f_to, (Daily, BDaily)) and isinstance(f_from, (YPFrequency, Weekly)):
        if f is not None:
            return _fconvert_higher_to_daily_or_bdaily_with_function(
                f_to, t, f, ref=eff_ref, **kwargs
            )
        return _fconvert_higher_to_daily_or_bdaily(f_to, t, method=higher_method, ref=eff_ref)

    msg = (
        f"fconvert: higher-frequency conversion from {type(f_from).__name__} to "
        f"{type(f_to).__name__} is not implemented."
    )
    raise NotImplementedError(msg)


# YP → YP higher ------------------------------------------------------------


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


def _fconvert_higher_yp_to_yp_with_function(
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


# YP → Weekly higher --------------------------------------------------------


def _fconvert_higher_yp_to_weekly_dates(f_to: Weekly, t: TSeries, *, ref: Ref) -> list[_dt.date]:
    """Build the anchor dates between input YP periods for a Weekly target."""
    rng = t.range
    if ref == "end":
        # End-of-period dates, with first dates shifted by +1 day to land on
        # the day after each period's end (= first day of next period). The
        # final end-date stays put unless it happens to land exactly on the
        # weekly end_day (then bump by 1 so the boundary is right of the cell).
        dates = [mit_to_date(m) for m in rng]
        for i in range(len(dates) - 1):
            dates[i] = dates[i] + _dt.timedelta(days=1)
        if dates and dates[-1].isoweekday() == f_to.end_day:
            dates[-1] = dates[-1] + _dt.timedelta(days=1)
        # Prepend the very first day-of-period of the first input period.
        dates.insert(0, mit_to_date(rng.first(), ref="begin"))
        return dates
    # ref == "begin": end-of-period dates of each input MIT, with the
    # previous week prepended to the start.
    dates_begin = [mit_to_date(m) for m in rng]
    first_date = mit_to_date(rng.first(), ref="begin")
    dates_begin.insert(0, first_date - _dt.timedelta(days=7))
    return dates_begin


def _fconvert_higher_yp_to_weekly(
    f_to: Weekly,
    t: TSeries,
    *,
    method: _HigherMethod,
    ref: Ref,
) -> TSeries:
    spreader = _higher_spreader_for(method)
    return _fconvert_higher_yp_to_weekly_with_function(f_to, t, spreader, ref=ref)


def _fconvert_higher_yp_to_weekly_with_function(
    f_to: Weekly,
    t: TSeries,
    f: Callable[..., Any],
    *,
    ref: Ref,
    **kwargs: Any,
) -> TSeries:
    fi, li, trunc_start, trunc_end = _fconvert_using_dates_parts(f_to, t.range, trim=ref)
    dates = _fconvert_higher_yp_to_weekly_dates(f_to, t, ref=ref)
    out_indices = _get_out_indices(f_to, dates)
    output_periods_per_input_period = np.array(
        [out_indices[i + 1].value - out_indices[i].value for i in range(len(out_indices) - 1)],
        dtype=np.int64,
    )
    out_range = MITRange(MIT(f_to, fi.value + trunc_start), MIT(f_to, li.value - trunc_end))
    ret = np.asarray(
        f(t.values, output_periods_per_input_period, ref=ref, outrange=out_range, **kwargs)
    )
    # Julia's "ret[begin+trunc_start:end]" drops the leading trunc_start
    # entries. The right edge is already bounded by ``output_periods_per_input_period``.
    ret = ret[trunc_start : trunc_start + len(out_range)]
    return TSeries(out_range.start, ret)


# YP / Weekly → Daily / BDaily higher --------------------------------------


def _fconvert_higher_to_daily_or_bdaily(
    f_to: Daily | BDaily,
    t: TSeries,
    *,
    method: _HigherMethod,
    ref: Ref,
) -> TSeries:
    spreader = _higher_spreader_for(method)
    return _fconvert_higher_to_daily_or_bdaily_with_function(f_to, t, spreader, ref=ref)


def _fconvert_higher_to_daily_or_bdaily_with_function(
    f_to: Daily | BDaily,
    t: TSeries,
    f: Callable[..., Any],
    *,
    ref: Ref,
    **kwargs: Any,
) -> TSeries:
    if ref not in ("begin", "end"):
        msg = f"fconvert: ref must be 'begin' or 'end'. Received: {ref!r}."
        raise ValueError(msg)
    first_date = mit_to_date(t.firstdate, ref="begin")
    last_date = mit_to_date(t.lastdate)
    if isinstance(f_to, BDaily):
        fi = bdaily(first_date, bias="next")
        li = bdaily(last_date, bias="previous")
    else:
        fi = daily(first_date)
        li = daily(last_date)
    output_lengths = np.array(
        [_output_periods_per_input(m, f_to) for m in t.range],
        dtype=np.int64,
    )
    outrange = MITRange(fi, li)
    out = np.asarray(f(t.values, output_lengths, ref=ref, outrange=outrange, **kwargs))
    return TSeries(fi, out)


def _output_periods_per_input(m: MIT, f_to: Daily | BDaily) -> int:
    """Return the count of Daily / BDaily output periods inside input ``m``."""
    begin_date = mit_to_date(m, ref="begin")
    end_date = mit_to_date(m, ref="end")
    if isinstance(f_to, BDaily):
        return (
            int(bdaily(end_date, bias="previous").value - bdaily(begin_date, bias="next").value) + 1
        )
    return int(daily(end_date).value - daily(begin_date).value) + 1


# BDaily → Daily higher -----------------------------------------------------


def _fconvert_higher_bdaily_to_daily(
    t: TSeries,
    *,
    method: _HigherMethod,
    ref: Ref,
) -> TSeries:
    f_to = Daily()
    fi = fconvert_mit(f_to, t.firstdate)
    li = fconvert_mit(f_to, t.lastdate)
    out_length = li.value - fi.value + 1
    out_arr = np.full(out_length, np.nan, dtype=np.float64)
    # Place the business-day values at their Daily ordinal positions.
    bd_dates = [mit_to_date(m) for m in t.range]
    bd_daysofweek = [d.isoweekday() for d in bd_dates]
    out_dates = [daily(d) for d in bd_dates]
    for od, val in zip(out_dates, np.asarray(t.values, dtype=np.float64), strict=True):
        out_arr[od.value - fi.value] = val
    ts = TSeries(fi, out_arr)
    if method == "even":
        return ts
    if method == "const":
        if ref == "end":
            # Carry each Monday's value backwards into Sat & Sun.
            monday_indices = [
                d.value for d, dow in zip(out_dates, bd_daysofweek, strict=True) if dow == 1
            ]
            if monday_indices and monday_indices[0] == fi.value:
                monday_indices = monday_indices[1:]
            for mi in monday_indices:
                out_arr[mi - 2 - fi.value] = out_arr[mi - fi.value]
                out_arr[mi - 1 - fi.value] = out_arr[mi - fi.value]
        elif ref == "begin":
            # Carry each Friday's value forwards into Sat & Sun.
            friday_indices = [
                d.value for d, dow in zip(out_dates, bd_daysofweek, strict=True) if dow == 5
            ]
            if friday_indices and friday_indices[-1] == li.value:
                friday_indices = friday_indices[:-1]
            for fi_idx in friday_indices:
                out_arr[fi_idx + 1 - fi.value] = out_arr[fi_idx - fi.value]
                out_arr[fi_idx + 2 - fi.value] = out_arr[fi_idx - fi.value]
        return ts
    if method == "linear":
        monday_indices = [
            d.value for d, dow in zip(out_dates, bd_daysofweek, strict=True) if dow == 1
        ]
        friday_indices = [
            d.value for d, dow in zip(out_dates, bd_daysofweek, strict=True) if dow == 5
        ]
        if monday_indices and monday_indices[0] == fi.value:
            monday_indices = monday_indices[1:]
        if friday_indices and friday_indices[-1] == li.value:
            friday_indices = friday_indices[:-1]
        if len(monday_indices) != len(friday_indices):
            msg = (
                f"fconvert: BDaily → Daily linear interpolation needs balanced Monday/Friday "
                f"pairs (got {len(monday_indices)} Mondays vs {len(friday_indices)} Fridays)."
            )
            raise ValueError(msg)
        for fri, mon in zip(friday_indices, monday_indices, strict=True):
            diff = out_arr[mon - fi.value] - out_arr[fri - fi.value]
            out_arr[fri + 1 - fi.value] = out_arr[fri - fi.value] + diff / 3
            out_arr[fri + 2 - fi.value] = out_arr[fri - fi.value] + 2 * diff / 3
    return ts


# Higher-frequency spreader helpers ----------------------------------------


def _higher_spreader_for(method: _HigherMethod) -> Callable[..., Any]:
    """Return the callable that ``method=...`` maps to in the custom-function form."""
    if method == "const":
        from tsecon.fconvert._helpers import repeat_uneven  # noqa: PLC0415  (avoid cycle)

        return repeat_uneven
    if method == "even":
        from tsecon.fconvert._helpers import divide_uneven  # noqa: PLC0415

        return divide_uneven
    return linear_uneven


# ---------------------------------------------------------------------------
# Lower-frequency dispatch
# ---------------------------------------------------------------------------


def _dispatch_lower(
    f_to: Frequency,
    t: TSeries,
    f: Callable[..., Any] | None,
    *,
    method: _AnyMethod | None,
    ref: Ref | None,
    **kwargs: Any,
) -> TSeries:
    if kwargs:
        msg = (
            f"fconvert: unsupported keyword arguments for lower-frequency conversion: "
            f"{sorted(kwargs)}."
        )
        raise TypeError(msg)
    f_from = t.frequency
    if (
        f is None
        and method is not None
        and method
        not in (
            "mean",
            "sum",
            "min",
            "max",
            "point",
            "begin",
            "end",
        )
    ):
        msg = (
            f"fconvert: method must be 'mean', 'sum', 'min', 'max', 'point', 'begin', or 'end' "
            f"when converting to a lower frequency. Received: {method!r}."
        )
        raise ValueError(msg)
    eff_ref: Ref = ref if ref in ("begin", "end") else "end"

    # YP → YP: closed-form.
    if isinstance(f_to, YPFrequency) and isinstance(f_from, YPFrequency):
        if f is not None:
            return _fconvert_lower_with_function(f_to, t, f, ref=eff_ref)
        lower_method: _LowerMethod = (
            method if method in ("mean", "sum", "min", "max", "point", "begin", "end") else "mean"
        )
        return _fconvert_lower_yp_to_yp(f_to, t, method=lower_method, ref=eff_ref)

    # Daily → BDaily: ignore method/aggregator; sample on weekdays.
    if isinstance(f_to, BDaily) and isinstance(f_from, Daily):
        return _fconvert_lower_daily_to_bdaily(t)

    # Calendar → YP / Weekly: aggregate per output MIT.
    if isinstance(f_to, (YPFrequency, Weekly)) and isinstance(f_from, CalendarFrequency):
        if f is not None:
            return _fconvert_lower_calendar_to_yp_or_weekly(f_to, t, f, ref=eff_ref)
        lower_method2: _LowerMethod = (
            method if method in ("mean", "sum", "min", "max", "point", "begin", "end") else "mean"
        )
        agg = _lower_aggregator_for(lower_method2, eff_ref)
        return _fconvert_lower_calendar_to_yp_or_weekly(f_to, t, agg.func, ref=agg.ref)

    msg = (
        f"fconvert: lower-frequency conversion from {type(f_from).__name__} to "
        f"{type(f_to).__name__} is not implemented."
    )
    raise NotImplementedError(msg)


# YP → YP lower -------------------------------------------------------------


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
) -> tuple[MITRange, int, int, int, int, int, int]:
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


# Calendar → YP / Weekly lower ---------------------------------------------


class _AggregatorBundle:
    """Bundle of ``(aggregator_func, effective_ref)`` produced by the lower-method translator."""

    __slots__ = ("func", "ref")

    def __init__(self, func: Callable[[np.ndarray], Any], ref: Ref) -> None:
        self.func = func
        self.ref = ref


def _lower_aggregator_for(method: _LowerMethod, ref: Ref) -> _AggregatorBundle:
    """Translate ``(method, ref)`` into the aggregator + effective ref pair.

    Julia's ``:point`` / ``:begin`` / ``:end`` methods route to ``first`` /
    ``last`` aggregators with a forced ref. The other methods preserve ref.
    """
    if method == "begin":
        return _AggregatorBundle(_agg_first, "begin")
    if method == "end":
        return _AggregatorBundle(_agg_last, "end")
    if method == "point":
        return _AggregatorBundle(_agg_last if ref == "end" else _agg_first, ref)
    return _AggregatorBundle(_LOWER_AGGREGATORS[method], ref)


def _fconvert_lower_calendar_to_yp_or_weekly(
    f_to: Frequency,
    t: TSeries,
    aggregator: Callable[[np.ndarray], Any],
    *,
    ref: Ref,
) -> TSeries:
    f_from = t.frequency
    rng_from = t.range
    # Compute date for each input MIT (the "anchor" used to map each input to
    # an output period). Mirrors Julia's dispatch on F_from and aggregator.
    if isinstance(f_from, Daily):
        # All days in the range, generated by stepping the ordinal.
        first_d = mit_to_date(rng_from.first())
        last_d = mit_to_date(rng_from.last())
        dates = [
            _dt.date.fromordinal(first_d.toordinal() + i)
            for i in range(last_d.toordinal() - first_d.toordinal() + 1)
        ]
    elif aggregator in (_agg_first, _agg_last):
        # Julia: for method=:point/:begin/:end, anchor on the *end* of each
        # input period (regardless of ref) so an input that overlaps the start
        # of an output period maps into that output period.
        dates = [mit_to_date(m) for m in rng_from]
    else:
        dates = [mit_to_date(m, ref=ref) for m in rng_from]

    trim: Literal["both", "begin", "end"] = ref if aggregator in (_agg_first, _agg_last) else "both"
    if isinstance(f_from, BDaily):
        # BDaily aggregate path: group rng_from by out_index and aggregate the
        # underlying values per group; truncation comes from
        # _fconvert_using_dates_parts. Mirrors Julia line 416-450 but with
        # cleanedvalues replaced by raw slicing (no skip_holidays/skip_all_nans
        # in this milestone).
        return _fconvert_lower_bdaily_to_yp_or_weekly(f_to, t, aggregator, ref=ref, trim=trim)

    fi, li, trunc_start, trunc_end = _fconvert_using_dates_parts(f_to, rng_from, trim=trim)
    out_index = _get_out_indices(f_to, dates)
    if isinstance(f_from, Weekly):
        fi = out_index[0]
        li = out_index[-1]
    # Group input values by output MIT and aggregate.
    vals = np.asarray(t.values)
    out_index_values = np.array([m.value for m in out_index], dtype=np.int64)
    unique_targets, first_idx = np.unique(out_index_values, return_index=True)
    order = np.argsort(first_idx)
    unique_in_order = unique_targets[order]
    aggregated: list[Any] = []
    for target in unique_in_order:
        mask = out_index_values == target
        aggregated.append(aggregator(vals[mask]))
    # Apply truncation: Julia's "ret[begin+trunc_start:end-trunc_end]".
    end = len(aggregated) - trunc_end if trunc_end > 0 else len(aggregated)
    aggregated_trimmed = aggregated[trunc_start:end]
    out_range = MITRange(
        MIT(fi.frequency, fi.value + trunc_start), MIT(li.frequency, li.value - trunc_end)
    )
    if len(aggregated_trimmed) == 0 or out_range.is_empty():
        return TSeries(out_range.start, np.empty(0, dtype=vals.dtype))
    out_arr = np.array(aggregated_trimmed)
    return TSeries(out_range.start, out_arr)


def _fconvert_lower_bdaily_to_yp_or_weekly(
    f_to: Frequency,
    t: TSeries,
    aggregator: Callable[[np.ndarray], Any],
    *,
    ref: Ref,
    trim: Literal["both", "begin", "end"],
) -> TSeries:
    rng_from = t.range
    dates = [mit_to_date(m) for m in rng_from]
    out_index = _get_out_indices(f_to, dates)
    fi, li, trunc_start, trunc_end = _fconvert_using_dates_parts(f_to, rng_from, trim=trim)
    # Group consecutive input MITs by target output MIT; aggregate per group.
    vals = np.asarray(t.values)
    out_index_values = np.array([m.value for m in out_index], dtype=np.int64)
    unique_targets, first_idx = np.unique(out_index_values, return_index=True)
    order = np.argsort(first_idx)
    unique_in_order = unique_targets[order]
    aggregated: list[Any] = []
    rng_from_starts = np.array([m.value for m in rng_from], dtype=np.int64)
    for target in unique_in_order:
        mask = out_index_values == target
        target_indices = rng_from_starts[mask]
        # Julia's BDaily lower path slices ``t[target_range[begin]:target_range[end]]`` —
        # the *contiguous* range from first to last input MIT in this group.
        # Without holidays/skip_all_nans, this matches the plain mask slice.
        lo_idx = int(target_indices[0] - rng_from_starts[0])
        hi_idx = int(target_indices[-1] - rng_from_starts[0]) + 1
        aggregated.append(aggregator(vals[lo_idx:hi_idx]))
    end = len(aggregated) - trunc_end if trunc_end > 0 else len(aggregated)
    aggregated_trimmed = aggregated[trunc_start:end]
    out_range = MITRange(
        MIT(fi.frequency, fi.value + trunc_start), MIT(li.frequency, li.value - trunc_end)
    )
    if len(aggregated_trimmed) == 0 or out_range.is_empty():
        return TSeries(out_range.start, np.empty(0, dtype=vals.dtype))
    out_arr = np.array(aggregated_trimmed)
    return TSeries(out_range.start, out_arr)


# Daily → BDaily lower ------------------------------------------------------


def _fconvert_lower_daily_to_bdaily(t: TSeries) -> TSeries:
    """Sample a Daily series on business days only.

    ``method`` and ``ref`` are ignored — Julia drops them too. The first
    business day at or after ``firstdate`` becomes the output's firstdate.
    """
    f_to = BDaily()
    fi = fconvert_mit(f_to, t.firstdate, round_to="next")
    first_date = mit_to_date(t.firstdate)
    first_day = first_date.isoweekday()
    # Build a 7-day repeating mask where Sat/Sun = False relative to the start.
    week_mask = [True] * 7
    sat = 7 if first_day == 7 else 7 - first_day
    sun = 1 if first_day == 7 else sat + 1
    week_mask[sat - 1] = False
    week_mask[sun - 1] = False
    mask = np.array(week_mask, dtype=bool)
    repeats = -(-len(t) // 7)
    full_mask = np.tile(mask, repeats)[: len(t)]
    return TSeries(fi, np.asarray(t.values)[full_mask])


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

    Source / target may be any non-Unit frequency. The BDaily-source variant
    that needs ``cleanedvalues(...; skip_all_nans=true)`` is still ⬜ (blocks
    on ``options.jl``) and raises :class:`NotImplementedError`.
    """
    if direction not in ("both", "begin", "end"):
        msg = f"direction must be 'both', 'begin', or 'end'. Received: {direction!r}."
        raise ValueError(msg)
    if method not in ("mean", "end"):
        msg = f"method must be 'mean' or 'end'. Received: {method!r}."
        raise ValueError(msg)
    f_to = sanitize_frequency(target)
    f_from = t.frequency
    _reject_unit(f_to, what="target")
    _reject_unit(f_from, what="source")
    if isinstance(f_from, BDaily):
        msg = (
            "extend_series: BDaily source requires the cleanedvalues / skip_all_nans path, "
            "which blocks on the options.jl port. Deferred."
        )
        raise NotImplementedError(msg)
    out = t.copy()
    if direction in ("begin", "both"):
        _extend_begin(f_to, out, method=method)
    if direction in ("end", "both"):
        _extend_end(f_to, out, method=method)
    return out


def _extend_begin(f_to: Frequency, t: TSeries, *, method: ExtendMethod) -> None:
    f_from = t.frequency
    first_mit_in_output_freq = fconvert_mit(f_to, t.firstdate)  # ref="end"
    desired_first_mit = fconvert_mit(f_from, first_mit_in_output_freq, ref="begin")
    if desired_first_mit.value >= t.firstdate.value:
        return
    affected = MITRange(desired_first_mit, MIT(f_from, t.firstdate.value - 1))
    if method == "end":
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


def _extend_end(f_to: Frequency, t: TSeries, *, method: ExtendMethod) -> None:
    f_from = t.frequency
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
    ``fconvert(F_from, fconvert(F_to, rangeof(t), trim=direction))``.
    """
    if direction not in ("both", "begin", "end"):
        msg = f"direction must be 'both', 'begin', or 'end'. Received: {direction!r}."
        raise ValueError(msg)
    f_to = sanitize_frequency(target)
    f_from = t.frequency
    _reject_unit(f_to, what="target")
    _reject_unit(f_from, what="source")
    target_range = fconvert_range(f_to, t.range, trim=direction)
    back = fconvert_range(f_from, target_range)
    return cast("TSeries", t[back])
