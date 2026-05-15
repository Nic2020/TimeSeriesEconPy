# SPDX-License-Identifier: MIT
"""Numerical and bookkeeping helpers shared by the fconvert subsystem.

Mirrors ``TimeSeriesEcon.jl/src/fconvert/fconvert_helpers.jl``. Public surface
re-exported from ``tsecon.fconvert``:

* :func:`repeat_uneven` — broadcast each input value across a per-input output
  count (``np.repeat`` with a vector of repeats).
* :func:`divide_uneven` — like :func:`repeat_uneven` but each input is divided
  evenly across its output cells.
* :func:`linear_uneven` — linearly interpolate across input values according to
  per-input output counts. Used by the ``method="linear"`` higher-frequency
  conversion.
* :func:`strip_tseries` / :func:`strip_tseries_inplace` — trim leading and
  trailing typenan entries from a :class:`~tsecon.tseries.TSeries`.

Internal helpers (used by ``fconvert._mit`` and ``fconvert._tseries``):

* :func:`_valid_range` — inner range of non-typenan entries in a TSeries.
* :func:`_get_start_truncation_yp` / :func:`_get_end_truncation_yp` —
  alignment-truncation deciders for the YP→YP lower-frequency conversion.
* :func:`_get_out_indices` — list of dates → list of MITs in a target
  YP-or-Weekly frequency (the period each date falls into).
* :func:`_fconvert_using_dates_parts` — date-based variant of
  :func:`fconvert_range`, used for any conversion where the source or target is
  Calendar (Daily / BDaily / Weekly) or for Weekly targets from YP sources.
"""

from __future__ import annotations

import datetime as _dt
from typing import Literal

import numpy as np
import numpy.typing as npt

from tsecon.frequencies import (
    BDaily,
    Daily,
    Frequency,
    HalfYearly,
    Monthly,
    Quarterly,
    Weekly,
    Yearly,
    YPFrequency,
    ppy,
)
from tsecon.mit import MIT, bdaily, mit_to_date, weekly
from tsecon.mitrange import MITRange
from tsecon.tseries import TSeries, typenan

__all__ = [
    "divide_uneven",
    "linear_uneven",
    "repeat_uneven",
    "strip_tseries",
    "strip_tseries_inplace",
]


Ref = Literal["begin", "end"]
Require = Literal["single", "all"]
Trim = Literal["both", "begin", "end"]


# ---------------------------------------------------------------------------
# Vector helpers used by the higher-frequency conversion
# ---------------------------------------------------------------------------


def repeat_uneven(x: npt.ArrayLike, inner: npt.ArrayLike, **_kwargs: object) -> np.ndarray:
    """Repeat each element of ``x`` according to the matching count in ``inner``.

    Returns a vector of length ``sum(inner)``. Mirrors the Julia
    ``repeat_uneven``: ``repeat_uneven([1, 2, 4], [2, 1, 4]) ==
    [1, 1, 2, 4, 4, 4, 4]``. Extra keyword arguments are accepted and
    ignored — required so this helper is drop-in callable from
    :func:`tsecon.fconvert.fconvert` (which always forwards ``ref`` and
    ``outrange``).
    """
    arr = np.asarray(x)
    counts = np.asarray(inner, dtype=np.int64)
    if arr.shape[0] != counts.shape[0]:
        msg = (
            f"repeat_uneven: x and inner must have the same length "
            f"({arr.shape[0]} vs {counts.shape[0]})."
        )
        raise ValueError(msg)
    return np.repeat(arr, counts)


def divide_uneven(x: npt.ArrayLike, inner: npt.ArrayLike, **_kwargs: object) -> np.ndarray:
    """Divide each element of ``x`` by its inner count, then repeat.

    Returns a Float64 vector of length ``sum(inner)``. Mirrors the Julia
    ``divide_uneven``: ``divide_uneven([1, 2, 4], [2, 1, 4]) ==
    [0.5, 0.5, 2.0, 1.0, 1.0, 1.0, 1.0]``. Extra keyword arguments are
    accepted and ignored.
    """
    arr = np.asarray(x, dtype=np.float64)
    counts = np.asarray(inner, dtype=np.int64)
    if arr.shape[0] != counts.shape[0]:
        msg = (
            f"divide_uneven: x and inner must have the same length "
            f"({arr.shape[0]} vs {counts.shape[0]})."
        )
        raise ValueError(msg)
    return np.repeat(arr / counts, counts)


def linear_uneven(
    x: npt.ArrayLike,
    output_lengths: npt.ArrayLike,
    *,
    ref: Ref = "end",
    **_kwargs: object,
) -> np.ndarray:
    """Linearly interpolate ``x`` across per-input output counts.

    Returns a Float64 vector of length ``sum(output_lengths)``. ``ref="end"``
    treats each input value as the end-point of its segment; ``ref="begin"``
    treats it as the start-point. Tail-end segments extrapolate with the slope
    of the adjacent segment.
    """
    if ref not in ("begin", "end"):
        msg = f"linear_uneven: ref must be 'begin' or 'end'. Received: {ref!r}."
        raise ValueError(msg)
    arr = np.asarray(x, dtype=np.float64)
    counts = np.asarray(output_lengths, dtype=np.int64)
    if arr.shape[0] != counts.shape[0]:
        msg = (
            f"linear_uneven: x and output_lengths must have the same length "
            f"({arr.shape[0]} vs {counts.shape[0]})."
        )
        raise ValueError(msg)
    out = np.empty(int(counts.sum()), dtype=np.float64)
    pos = 0
    n = arr.shape[0]
    if ref == "end":
        for i in range(n):
            length = int(counts[i])
            if i == 0:
                step = (arr[1] - arr[0]) / int(counts[1])
                vals = np.linspace(arr[0] - length * step, arr[0], length + 1)
            else:
                vals = np.linspace(arr[i - 1], arr[i], length + 1)
            out[pos : pos + length] = vals[1:]
            pos += length
    else:
        for i in range(n):
            length = int(counts[i])
            if i == n - 1:
                step = (arr[i] - arr[i - 1]) / int(counts[i - 1])
                vals = np.linspace(arr[i], arr[i] + length * step, length + 1)
            else:
                vals = np.linspace(arr[i], arr[i + 1], length + 1)
            out[pos : pos + length] = vals[:-1]
            pos += length
    return out


# ---------------------------------------------------------------------------
# strip
# ---------------------------------------------------------------------------


def _valid_range(t: TSeries) -> MITRange:
    """Return the inner range of non-typenan entries in ``t``.

    Mirrors Julia's private ``_valid_range``. For an all-typenan series the
    returned range is empty, anchored at the original ``firstdate``. Boolean
    dtypes are not stripped (their typenan is ``False``, which would erase
    everything); the original range is returned unchanged.
    """
    if t.is_empty():
        return t.range
    arr = t.values
    dt = arr.dtype
    if dt == np.bool_:
        return t.range
    sentinel = typenan(dt)
    valid = ~np.isnan(arr) if np.issubdtype(dt, np.floating) else arr != sentinel
    if not valid.any():
        empty_stop = MIT(t.firstdate.frequency, t.firstdate.value - 1)
        return MITRange(t.firstdate, empty_stop)
    first_valid = int(np.argmax(valid))
    last_valid = len(arr) - 1 - int(np.argmax(valid[::-1]))
    new_start = MIT(t.firstdate.frequency, t.firstdate.value + first_valid)
    new_stop = MIT(t.firstdate.frequency, t.firstdate.value + last_valid)
    return MITRange(new_start, new_stop)


def strip_tseries(t: TSeries) -> TSeries:
    """Return a new :class:`TSeries` with leading and trailing typenan trimmed.

    Mirrors Julia's ``Base.strip(t::TSeries)``. The original is unchanged.
    An all-typenan input returns an empty TSeries anchored at ``firstdate``.
    """
    rng = _valid_range(t)
    if rng.is_empty():
        return TSeries(t.firstdate, np.empty(0, dtype=t.dtype))
    return t[rng]  # type: ignore[no-any-return]


def strip_tseries_inplace(t: TSeries) -> TSeries:
    """Trim leading and trailing typenan entries in place; return ``t``.

    Mirrors Julia's ``strip!(t::TSeries)``.
    """
    t.resize(_valid_range(t))
    return t


# ---------------------------------------------------------------------------
# Truncation deciders for YP → YP lower-frequency conversion
# ---------------------------------------------------------------------------


def _get_start_truncation_yp(
    ref: Ref,
    require: Require,
    fi_from_start_month: int,
    fi_to_start_month: int,
    mpp_from: int,
    mpp_to: int,
) -> int:
    """Decide whether to drop the first output period at the start of a range.

    Mirrors Julia's four ``get_start_truncation_yp(Val{ref}, Val{require}, ...)``
    overloads. Returns 1 to truncate, 0 to keep.
    """
    if ref == "end" and require == "single":
        if fi_from_start_month + (mpp_from - 1) <= fi_to_start_month + (mpp_to - 1):
            return 0
        return 1
    if ref == "end" and require == "all":
        if fi_from_start_month == fi_to_start_month:
            return 0
        if (
            fi_from_start_month < fi_to_start_month
            and fi_from_start_month + (mpp_from - 1) >= fi_to_start_month
        ):
            return 0
        return 1
    if ref == "begin" and require == "single":
        if fi_from_start_month == fi_to_start_month:
            return 0
        if (
            fi_from_start_month < fi_to_start_month
            and fi_from_start_month + (mpp_from - 1) >= fi_to_start_month
        ):
            return 0
        return 1
    # ref == "begin" and require == "all"
    if fi_from_start_month == fi_to_start_month:
        return 0
    if (
        fi_from_start_month > fi_to_start_month
        and fi_from_start_month - (mpp_from - 1) <= fi_to_start_month
    ):
        return 0
    return 1


def _freq_is_higher(a: Frequency, b: Frequency) -> bool:
    """Return True if ``a`` has more periods per year than ``b``.

    Mirrors Julia's ``Frequency`` ``<``/``>`` ordering (defined by ``ppy``).
    Used by the calendar-aware fconvert paths to pick the higher/lower branch.
    """
    return ppy(a) > ppy(b)


def _get_out_indices(f_to: Frequency, dates: list[_dt.date]) -> list[MIT]:
    """Return the list of ``f_to``-frequency MITs that contain each ``dates[i]``.

    Mirrors Julia's ``_get_out_indices`` (and ``_get_out_indices_actual``).
    ``f_to`` must be a :class:`Monthly`, :class:`Quarterly`, :class:`HalfYearly`,
    :class:`Yearly`, or :class:`Weekly` instance.
    """
    if isinstance(f_to, Weekly):
        end_day = f_to.end_day
        return [weekly(d, end_day) for d in dates]
    if isinstance(f_to, Monthly):
        return [MIT.from_yp(f_to, d.year, d.month) for d in dates]
    if isinstance(f_to, Quarterly):
        end_month = f_to.end_month
        # Julia adjusts months by (3 - NtQ) so that the quarter boundary
        # aligned with end_month becomes the boundary between quarters.
        shift = 3 - end_month
        out: list[MIT] = []
        for d in dates:
            m = d.month + shift
            y = d.year
            if m > 12:
                m -= 12
                y += 1
            quarter = -(-m // 3)  # ceil(m / 3)
            out.append(MIT.from_yp(f_to, y, quarter))
        return out
    if isinstance(f_to, HalfYearly):
        end_month = f_to.end_month
        shift = 6 - end_month
        out_h: list[MIT] = []
        for d in dates:
            m = d.month + shift
            y = d.year
            if m > 12:
                m -= 12
                y += 1
            half = -(-m // 6)  # ceil(m / 6)
            out_h.append(MIT.from_yp(f_to, y, half))
        return out_h
    if isinstance(f_to, Yearly):
        end_month = f_to.end_month
        shift = 12 - end_month
        out_y: list[MIT] = []
        for d in dates:
            m = d.month + shift
            y = d.year
            if m > 12:
                y += 1
            out_y.append(MIT.from_yp(f_to, y, 1))
        return out_y
    msg = (
        f"_get_out_indices: target must be Monthly / Quarterly / HalfYearly / "
        f"Yearly / Weekly, got {type(f_to).__name__}."
    )
    raise TypeError(msg)


def _target_to_source_mit(source_freq: Frequency, target_mit: MIT, *, ref: Ref) -> MIT:
    """Convert ``target_mit`` back to ``source_freq``, aligned to ``ref`` of the target period.

    Used inside :func:`_fconvert_using_dates_parts` to ask "is the source MIT
    that anchors the target period's begin/end equal to the actual range
    endpoint?". Mirrors the Julia ``fconvert(F_from, fi, ref=:begin)`` /
    ``fconvert(F_from, li, ref=:end)`` calls.

    For Daily / BDaily / Weekly sources we work in integer ordinal space, so
    we can produce ``MIT(Daily(), -2)`` etc. without hitting Python's
    pre-0001 date limit. For pre-0001 YP-target periods (which would otherwise
    crash :func:`mit_to_date`) we return a sentinel "very small" source MIT —
    the truncation comparisons in :func:`_fconvert_using_dates_parts` only care
    about ordering against the actual range first/last source MIT, both of
    which are ≥ 1 in our model.
    """
    if isinstance(source_freq, Daily):
        return MIT(Daily(), _target_period_ordinal(target_mit, ref=ref))
    if isinstance(source_freq, BDaily):
        ordinal = _target_period_ordinal(target_mit, ref=ref)
        return _bdaily_from_ordinal(ordinal, ref=ref)
    try:
        target_date = mit_to_date(target_mit, ref=ref)
    except ValueError:
        # Pre-0001 target period — return a sentinel "very small" source MIT.
        # Comparisons against the source range's actual first/last (≥ 1) all
        # come out as `< first_mit`, which is the correct semantic answer
        # (the target period extends before the source range, so truncation
        # is warranted by the lower-frequency direction's logic).
        return MIT(source_freq, -(10**9))
    if isinstance(source_freq, Weekly):
        return weekly(target_date, source_freq.end_day)
    if isinstance(source_freq, YPFrequency):
        return _get_out_indices(source_freq, [target_date])[0]
    msg = f"_target_to_source_mit: unsupported source frequency {type(source_freq).__name__}."
    raise TypeError(msg)


def _target_period_ordinal(target_mit: MIT, *, ref: Ref) -> int:
    """Return the proleptic-Gregorian day ordinal of target_mit's begin/end.

    Uses integer arithmetic so callers can produce ``MIT(Daily(), ord)`` for
    ordinals ≤ 0 (which would otherwise crash :func:`datetime.date.fromordinal`).
    """
    f = target_mit.frequency
    v = target_mit.value
    if isinstance(f, Daily):
        return v
    if isinstance(f, BDaily):
        rem = v % 5
        if rem == 0:
            rem = 5
        return ((v - 1) // 5) * 7 + rem
    if isinstance(f, Weekly):
        ed = f.end_day
        return v * 7 - (7 - ed) - (6 if ref == "begin" else 0)
    return mit_to_date(target_mit, ref=ref).toordinal()


def _bdaily_from_ordinal(ordinal: int, *, ref: Ref) -> MIT:
    """Construct a BDaily MIT from a day ordinal, applying the begin/end bias.

    For non-business-day ordinals the bias snaps to previous (for ref="end")
    or next (for ref="begin"). Pre-0001-01-01 ordinals are clamped to
    ``MIT(BDaily(), 0)`` or ``MIT(BDaily(), 1)`` — sentinel anchors that
    suffice for the truncation comparisons performed by
    :func:`_fconvert_using_dates_parts`.
    """
    if ordinal >= 1:
        return bdaily(_dt.date.fromordinal(ordinal), bias="previous" if ref == "end" else "next")
    return MIT(BDaily(), 1 if ref == "begin" else 0)


def _fconvert_using_dates_parts(
    f_to: Frequency,
    range_from: MITRange,
    *,
    trim: Trim,
) -> tuple[MIT, MIT, int, int]:
    """Return ``(fi, li, trunc_start, trunc_end)`` for a date-based range conversion.

    ``f_to`` is a YPFrequency or Weekly target; ``range_from`` is any calendar
    or YP source range. ``trim`` is one of ``"both"``, ``"begin"``, ``"end"``.

    Mirrors Julia's ``_fconvert_using_dates_parts`` for the non-BDaily-kwarg
    case (``skip_holidays=False`` and ``holidays_map=None`` only). We replace
    the Julia "build predecessor / successor dates and check that the target
    periods match" idiom with an equivalent MIT-space check: a target period
    bounds the source range exactly iff the source MIT anchoring the target
    period's begin/end equals the source range endpoint. This avoids needing
    to represent dates before 0001-01-01 for ``MIT(Daily, 1)`` and friends.
    """
    if trim not in ("both", "begin", "end"):
        msg = f"trim argument must be 'both', 'begin', or 'end'. Received: {trim!r}."
        raise ValueError(msg)
    f_from = range_from.frequency
    first_mit = range_from.first()
    last_mit = range_from.last()
    if _freq_is_higher(f_to, f_from):
        # to higher frequency (e.g., YP → Weekly): the target period containing
        # the start-date of the first source MIT, through the period containing
        # the end-date of the last source MIT.
        first_date = mit_to_date(first_mit, ref="begin")
        last_date = mit_to_date(last_mit, ref="end")
    else:
        # to lower (or similar) frequency (e.g., Calendar → YP, Calendar → Weekly):
        # the target period containing the single date of the first/last source MIT.
        first_date = mit_to_date(first_mit)
        last_date = mit_to_date(last_mit)
    out_index = _get_out_indices(f_to, [first_date, last_date])
    fi = out_index[0]
    li = out_index[1]
    trunc_start = 0
    if trim != "end":
        src_anchor_start = _target_to_source_mit(f_from, fi, ref="begin")
        if _freq_is_higher(f_to, f_from):
            # Higher: truncate if the target period doesn't start exactly at the source range start
            if src_anchor_start != first_mit:
                trunc_start = 1
        # Lower: truncate if the target period begins before the source range
        # (i.e., the source MIT anchoring the target period's begin is before
        # the actual first source MIT, meaning we're missing input for the
        # start of the target period).
        elif src_anchor_start < first_mit:
            trunc_start = 1
    trunc_end = 0
    if trim != "begin":
        src_anchor_end = _target_to_source_mit(f_from, li, ref="end")
        if _freq_is_higher(f_to, f_from):
            if src_anchor_end != last_mit:
                trunc_end = 1
        elif src_anchor_end > last_mit:
            trunc_end = 1
    return fi, li, trunc_start, trunc_end


def _get_end_truncation_yp(
    ref: Ref,
    require: Require,
    li_from_end_month: int,
    li_to_end_month: int,
    mpp_from: int,
    mpp_to: int,
) -> int:
    """Decide whether to drop the last output period at the end of a range.

    Mirrors Julia's four ``get_end_truncation_yp(Val{ref}, Val{require}, ...)``
    overloads. Returns 1 to truncate, 0 to keep.
    """
    if ref == "end" and require in ("single", "all"):
        # The :single and :all variants share an implementation upstream.
        if li_from_end_month == li_to_end_month:
            return 0
        if (
            li_from_end_month < li_to_end_month
            and li_from_end_month + (mpp_from - 1) >= li_to_end_month
        ):
            return 0
        return 1
    if ref == "begin" and require == "single":
        if li_from_end_month - (mpp_from - 1) >= li_to_end_month - (mpp_to - 1):
            return 0
        if li_from_end_month - (mpp_from - 1) < li_to_end_month - (
            mpp_to - 1
        ) and li_from_end_month >= li_to_end_month - (mpp_to - 1):
            return 0
        return 1
    # ref == "begin" and require == "all"
    if (
        li_from_end_month > li_to_end_month
        and li_from_end_month - (mpp_from - 1) <= li_to_end_month
    ):
        return 0
    return 1
