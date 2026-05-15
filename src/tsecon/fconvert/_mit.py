# SPDX-License-Identifier: MIT
"""Frequency conversion for :class:`MIT` and :class:`MITRange`.

Mirrors ``TimeSeriesEcon.jl/src/fconvert/fconvert_mit.jl``:

* :func:`fconvert_mit` — convert an :class:`MIT` to a target frequency.
* :func:`fconvert_range` — convert an :class:`MITRange` to a target frequency.
* :func:`fconvert_parts` — internal helper exposing the
  ``(period, source_month, target_month)`` triple used by the YP→YP TSeries
  conversion path.

``Unit`` is intentionally excluded from ``fconvert``: any attempt to convert
to or from ``Unit`` raises (matching Julia's ``ErrorException``).
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Literal, overload

from tsecon.fconvert._helpers import (
    Ref,
    Trim,
    _fconvert_using_dates_parts,
    _get_out_indices,
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
    endperiod,
    ppy,
    sanitize_frequency,
)
from tsecon.mit import MIT, bdaily, daily, mit_to_date
from tsecon.mitrange import MITRange

__all__ = ["fconvert_mit", "fconvert_parts", "fconvert_range"]


RoundTo = Literal["current", "previous", "next"]


def _reject_unit(f: Frequency, *, what: str) -> None:
    if isinstance(f, Unit):
        msg = f"Cannot fconvert {what} to or from Unit."
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# fconvert(MIT)
# ---------------------------------------------------------------------------


def fconvert_mit(
    target: FrequencyLike,
    mit_from: MIT,
    *,
    ref: Ref = "end",
    round_to: RoundTo = "current",
) -> MIT:
    """Convert ``mit_from`` to ``target``.

    For YP→YP and any Calendar-involving conversion ``ref`` selects whether to
    align to the end of the source period (``"end"``, the default) or the start
    (``"begin"``). Mirrors Julia's ``fconvert(F_to, MIT_from; ref=:end)``.

    When the target is :class:`BDaily`, ``round_to`` controls how a date that
    lands on a weekend is resolved: ``"current"`` (default) raises on
    weekends, ``"previous"`` snaps to the preceding business day, ``"next"``
    snaps forward.

    Examples
    --------
    >>> from tsecon import yy, Quarterly
    >>> fconvert_mit(Quarterly, yy(22), ref="end")
    22Q4
    >>> fconvert_mit(Quarterly, yy(22), ref="begin")
    22Q1
    """
    f_to = sanitize_frequency(target)
    f_from = mit_from.frequency
    if ref not in ("begin", "end"):
        msg = f"ref argument must be 'begin' or 'end'. Received: {ref!r}."
        raise ValueError(msg)
    # Same-frequency identity (works for Unit too).
    if f_to == f_from:
        return mit_from
    _reject_unit(f_to, what="target")
    _reject_unit(f_from, what="source")
    # YP → YP — closed-form formula for performance.
    if isinstance(f_to, YPFrequency) and isinstance(f_from, YPFrequency):
        return _fconvert_mit_yp_to_yp(f_to, mit_from, ref=ref)
    # BDaily target — date-based with rounding policy.
    if isinstance(f_to, BDaily):
        return _fconvert_mit_to_bdaily(mit_from, ref=ref, round_to=round_to)
    # Daily target.
    if isinstance(f_to, Daily):
        return _fconvert_mit_to_daily(mit_from, ref=ref)
    # Any remaining target (YP from Calendar, Weekly from anywhere) — go via
    # ``_get_out_indices`` over the relevant boundary date.
    if isinstance(f_to, (YPFrequency, Weekly)):
        date_ref = mit_to_date(mit_from, ref=ref)
        return _get_out_indices(f_to, [date_ref])[0]
    msg = (
        f"fconvert: conversion from {type(f_from).__name__} "
        f"to {type(f_to).__name__} not implemented."
    )
    raise NotImplementedError(msg)


def _fconvert_mit_yp_to_yp(f_to: YPFrequency, mit_from: MIT, *, ref: Ref) -> MIT:
    f_from = mit_from.frequency
    n_from = ppy(f_from)
    n_to = ppy(f_to)
    ep_from = endperiod(f_from)
    ref_adjust = 1 if ref == "end" else 0
    rounder = math.ceil if ref == "end" else math.floor
    # The Julia formula uses real division throughout; we keep doubles for parity.
    from_month = (int(mit_from) + ref_adjust) * 12 / n_from - ref_adjust
    from_month -= (12 / n_from) - ep_from
    out_period = (from_month + ref_adjust) / (12 / n_to) - ref_adjust
    return MIT(f_to, rounder(out_period))


def _fconvert_mit_to_bdaily(mit_from: MIT, *, ref: Ref, round_to: RoundTo) -> MIT:
    f_from = mit_from.frequency
    if not isinstance(f_from, (Weekly, YPFrequency, Daily)):
        msg = (
            f"fconvert: BDaily target only supports Daily / Weekly / YPFrequency "
            f"sources, got {type(f_from).__name__}."
        )
        raise NotImplementedError(msg)
    if round_to not in ("current", "previous", "next"):
        msg = f"round_to argument must be 'current', 'previous', or 'next'. Received: {round_to!r}."
        raise ValueError(msg)
    date_ref = mit_to_date(mit_from, ref=ref)
    if round_to == "previous":
        return bdaily(date_ref, bias="previous")
    if round_to == "next":
        return bdaily(date_ref, bias="next")
    # round_to == "current"
    return bdaily(date_ref, bias="strict")


def _fconvert_mit_to_daily(mit_from: MIT, *, ref: Ref) -> MIT:
    f_from = mit_from.frequency
    if isinstance(f_from, BDaily):
        # BDaily → Daily uses the closed-form mod-5 formula in Julia (ignoring ref).
        v = mit_from.value
        rem = v % 5
        if rem == 0:
            rem = 5
        ordinal = ((v - 1) // 5) * 7 + rem
        return daily(_dt.date.fromordinal(ordinal))
    if isinstance(f_from, (Weekly, YPFrequency)):
        return daily(mit_to_date(mit_from, ref=ref))
    msg = (
        f"fconvert: Daily target only supports BDaily / Weekly / YPFrequency "
        f"sources, got {type(f_from).__name__}."
    )
    raise NotImplementedError(msg)


# ---------------------------------------------------------------------------
# fconvert_parts (internal helper used by the YP→YP TSeries conversion path)
# ---------------------------------------------------------------------------


def fconvert_parts(
    target: FrequencyLike,
    mit_from: MIT,
    *,
    ref: Ref = "end",
) -> tuple[int, int, int]:
    """Return ``(to_period, from_month, to_month)`` for the YP→YP boundary.

    With ``ref="end"`` the second / third elements are the end-month-of-year
    of the source / target periods; with ``ref="begin"`` they are the start
    months. Mirrors Julia's ``fconvert_parts``. YP-only; raises for any
    Calendar source or target.
    """
    f_to = sanitize_frequency(target)
    f_from = mit_from.frequency
    if not isinstance(f_to, YPFrequency) or not isinstance(f_from, YPFrequency):
        msg = "fconvert_parts is YP-only; pass YP source and YP target."
        raise TypeError(msg)
    mpp_from = 12 // ppy(f_from)
    mpp_to = 12 // ppy(f_to)
    from_month_adjustment = endperiod(f_from) - mpp_from
    to_month_adjustment = endperiod(f_to) - mpp_to
    if ref == "begin":
        from_start_month = int(mit_from) * mpp_from + 1 + from_month_adjustment
        to_period = (from_start_month - to_month_adjustment - 1) // mpp_to
        to_start_month = to_period * mpp_to + 1 + to_month_adjustment
        return to_period, from_start_month, to_start_month
    # ref == "end" (the Literal narrowing rules out anything else)
    from_end_month = (int(mit_from) + 1) * mpp_from + from_month_adjustment
    to_period = (from_end_month - to_month_adjustment - 1) // mpp_to
    to_end_month = (to_period + 1) * mpp_to + to_month_adjustment
    return to_period, from_end_month, to_end_month


# ---------------------------------------------------------------------------
# fconvert(MITRange)
# ---------------------------------------------------------------------------


@overload
def fconvert_range(
    target: FrequencyLike,
    range_from: MITRange,
    *,
    trim: Trim = "both",
    parts: Literal[False] = False,
) -> MITRange: ...


@overload
def fconvert_range(
    target: FrequencyLike,
    range_from: MITRange,
    *,
    trim: Trim = "both",
    parts: Literal[True],
) -> tuple[int, int, int, int, int, int]: ...


def fconvert_range(
    target: FrequencyLike,
    range_from: MITRange,
    *,
    trim: Trim = "both",
    parts: bool = False,
) -> MITRange | tuple[int, int, int, int, int, int]:
    """Convert ``range_from`` to ``target``.

    ``trim`` controls whether the start and / or end of the output range are
    truncated when the source range begins or ends partway through a target
    period. With ``parts=True`` (YP→YP only) returns the six-tuple
    ``(fi_to_period, fi_from_start_month, fi_to_start_month, li_to_period,
    li_from_end_month, li_to_end_month)`` used by the TSeries conversion path.

    Examples
    --------
    >>> from tsecon import yy, mitrange, Quarterly
    >>> fconvert_range(Quarterly, mitrange(yy(22), yy(24)))
    22Q1:24Q4
    """
    f_to = sanitize_frequency(target)
    f_from = range_from.frequency
    if trim not in ("both", "begin", "end"):
        msg = f"trim argument must be 'both', 'begin', or 'end'. Received: {trim!r}."
        raise ValueError(msg)
    # Same-frequency identity (mirrors the ``F_to == F_from`` Julia overload; works for Unit).
    if f_to == f_from:
        if parts:
            msg = "parts=True is only meaningful for cross-frequency conversions."
            raise ValueError(msg)
        return range_from
    _reject_unit(f_to, what="target")
    _reject_unit(f_from, what="source")
    # YP → YP: closed-form formula + `parts` introspection support.
    if isinstance(f_to, YPFrequency) and isinstance(f_from, YPFrequency):
        return _fconvert_range_yp_to_yp(f_to, range_from, trim=trim, parts=parts)
    if parts:
        msg = "parts=True is only supported for YP → YP conversions."
        raise ValueError(msg)
    # Daily target — date-based.
    if isinstance(f_to, Daily):
        return _fconvert_range_to_daily(range_from)
    # BDaily target — date-based.
    if isinstance(f_to, BDaily):
        return _fconvert_range_to_bdaily(range_from)
    # YP-or-Weekly target with anything else (Calendar source, or YP→Weekly):
    # use the date-based path.
    if isinstance(f_to, (YPFrequency, Weekly)) and isinstance(
        f_from, (Daily, BDaily, Weekly, YPFrequency)
    ):
        fi, li, ts, te = _fconvert_using_dates_parts(f_to, range_from, trim=trim)
        # The truncation arithmetic differs slightly by direction (higher vs
        # lower). For higher: just add/subtract. For lower: same — `fi` and
        # `li` are already the *inner* indices from _fconvert_using_dates_parts.
        return MITRange(MIT(f_to, fi.value + ts), MIT(f_to, li.value - te))
    msg = (
        f"fconvert_range: conversion from {type(f_from).__name__} "
        f"to {type(f_to).__name__} not implemented."
    )
    raise NotImplementedError(msg)


def _fconvert_range_yp_to_yp(
    f_to: YPFrequency,
    range_from: MITRange,
    *,
    trim: Trim,
    parts: bool,
) -> MITRange | tuple[int, int, int, int, int, int]:
    fi_to_period, fi_from_start_month, fi_to_start_month = fconvert_parts(
        f_to, range_from.first(), ref="begin"
    )
    li_to_period, li_from_end_month, li_to_end_month = fconvert_parts(
        f_to, range_from.last(), ref="end"
    )
    if parts:
        return (
            fi_to_period,
            fi_from_start_month,
            fi_to_start_month,
            li_to_period,
            li_from_end_month,
            li_to_end_month,
        )
    f_from = range_from.frequency
    assert isinstance(f_from, YPFrequency)
    mpp_from = 12 // ppy(f_from)
    mpp_to = 12 // ppy(f_to)
    trunc_start = 0
    trunc_end = 0
    if mpp_from > mpp_to:  # to higher frequency
        if trim != "end" and fi_to_start_month < fi_from_start_month:
            trunc_start = 1
        if trim != "begin" and li_to_end_month > li_from_end_month:
            trunc_end = 1
    else:  # to lower (or similar) frequency
        if trim in ("begin", "both") and (
            fi_to_start_month < fi_from_start_month
            and fi_to_start_month <= fi_from_start_month - (mpp_from - 1)
        ):
            trunc_start = 1
        if trim in ("end", "both") and (
            li_to_end_month > li_from_end_month
            and li_to_end_month >= li_from_end_month + mpp_from - 1
        ):
            trunc_end = 1
    fi = MIT(f_to, fi_to_period + trunc_start)
    li = MIT(f_to, li_to_period - trunc_end)
    return MITRange(fi, li)


def _fconvert_range_to_daily(range_from: MITRange) -> MITRange:
    """Daily target — span from the source range's start to its end (in days).

    Equivalent to Julia's ``daily(Dates.Date(range_from[begin] - 1) + Day(1))``
    formula but implemented as ``mit_to_date(first, ref="begin")`` to avoid
    computing pre-0001 predecessor dates for YP sources anchored at year 1.
    """
    f_from = range_from.frequency
    if not isinstance(f_from, (CalendarFrequency, YPFrequency)):
        msg = (
            f"fconvert: Daily target only supports Calendar (Daily/BDaily/Weekly) "
            f"or YPFrequency sources, got {type(f_from).__name__}."
        )
        raise NotImplementedError(msg)
    if isinstance(f_from, BDaily):
        # BDaily: source is already a single date; the "predecessor + 1" formula
        # would land on a non-BDay weekend or the previous BD. Julia uses the
        # plain start date.
        first_date = mit_to_date(range_from.first())
    else:
        first_date = mit_to_date(range_from.first(), ref="begin")
    last_date = mit_to_date(range_from.last())
    return MITRange(daily(first_date), daily(last_date))


def _fconvert_range_to_bdaily(range_from: MITRange) -> MITRange:
    """BDaily target — first business day at/after source start to last business day at/before source end."""  # noqa: E501
    f_from = range_from.frequency
    if not isinstance(f_from, (CalendarFrequency, YPFrequency)):
        msg = (
            f"fconvert: BDaily target only supports Calendar (Daily/BDaily/Weekly) "
            f"or YPFrequency sources, got {type(f_from).__name__}."
        )
        raise NotImplementedError(msg)
    first_date = mit_to_date(range_from.first(), ref="begin")
    last_date = mit_to_date(range_from.last())
    return MITRange(
        bdaily(first_date, bias="next"),
        bdaily(last_date, bias="previous"),
    )
