# SPDX-License-Identifier: MIT
"""Frequency conversion for :class:`MIT` and :class:`MITRange` (YP only).

Mirrors ``TimeSeriesEcon.jl/src/fconvert/fconvert_mit.jl`` for the YP-only
subset:

* :func:`fconvert_mit` — convert an :class:`MIT` to a target YPFrequency.
* :func:`fconvert_range` — convert an :class:`MITRange` to a target YPFrequency.
* :func:`fconvert_parts` — internal helper exposing the
  ``(period, source_month, target_month)`` triple used by the TSeries
  conversion path.

The Calendar (Daily / BDaily / Weekly) variants are deferred to the follow-up
session that closes out fconvert. ``Unit`` is intentionally excluded from
``fconvert``: any attempt to convert to or from ``Unit`` raises (matching
Julia's ``ErrorException``).
"""

from __future__ import annotations

import math
from typing import Literal, overload

from tsecon.frequencies import (
    Frequency,
    FrequencyLike,
    Unit,
    YPFrequency,
    endperiod,
    ppy,
    sanitize_frequency,
)
from tsecon.mit import MIT
from tsecon.mitrange import MITRange

__all__ = ["fconvert_mit", "fconvert_parts", "fconvert_range"]


Ref = Literal["begin", "end"]
Trim = Literal["both", "begin", "end"]


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


def _sanitized_yp(target: FrequencyLike, *, what: str) -> YPFrequency:
    f = sanitize_frequency(target)
    return _require_yp(f, what=what)


# ---------------------------------------------------------------------------
# fconvert(MIT)
# ---------------------------------------------------------------------------


def fconvert_mit(target: FrequencyLike, mit_from: MIT, *, ref: Ref = "end") -> MIT:
    """Convert ``mit_from`` to ``target``.

    For YP→YP conversions ``ref`` selects whether to align to the end of the
    source period (``"end"``, the default) or the start (``"begin"``). Mirrors
    Julia's ``fconvert(F_to, MIT_from; ref=:end)``.

    Examples
    --------
    >>> from tsecon import qq, yy, Quarterly, Yearly
    >>> fconvert_mit(Quarterly, yy(22), ref="end")
    22Q4
    >>> fconvert_mit(Quarterly, yy(22), ref="begin")
    22Q1
    """
    f_to = sanitize_frequency(target)
    f_from = mit_from.frequency
    # Same-frequency identity (mirrors the ``F_to == F_from`` Julia overload).
    if f_to == f_from:
        return mit_from
    f_to = _require_yp(f_to, what="target")
    f_from = _require_yp(f_from, what="source")
    if ref not in ("begin", "end"):
        msg = f"ref argument must be 'begin' or 'end'. Received: {ref!r}."
        raise ValueError(msg)
    return _fconvert_mit_yp_to_yp(f_to, mit_from, ref=ref)


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


# ---------------------------------------------------------------------------
# fconvert_parts (internal helper used by the TSeries conversion path)
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
    months. Mirrors Julia's ``fconvert_parts``.
    """
    f_to = _sanitized_yp(target, what="target")
    f_from = _require_yp(mit_from.frequency, what="source")
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
    period. With ``parts=True`` returns the six-tuple
    ``(fi_to_period, fi_from_start_month, fi_to_start_month, li_to_period,
    li_from_end_month, li_to_end_month)`` used by the TSeries conversion path.

    Examples
    --------
    >>> from tsecon import qq, yy, mitrange, Quarterly, Yearly
    >>> fconvert_range(Quarterly, mitrange(yy(22), yy(24)))
    22Q1:24Q4
    """
    f_to = sanitize_frequency(target)
    f_from = range_from.frequency
    if f_to == f_from:
        if parts:
            msg = "parts=True is only meaningful for cross-frequency conversions."
            raise ValueError(msg)
        return range_from
    if trim not in ("both", "begin", "end"):
        msg = f"trim argument must be 'both', 'begin', or 'end'. Received: {trim!r}."
        raise ValueError(msg)
    f_to = _require_yp(f_to, what="target")
    f_from = _require_yp(f_from, what="source")
    return _fconvert_range_yp_to_yp(f_to, range_from, trim=trim, parts=parts)


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
