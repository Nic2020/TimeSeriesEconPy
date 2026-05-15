# SPDX-License-Identifier: MIT
"""Numerical and bookkeeping helpers shared by the fconvert subsystem.

Mirrors ``TimeSeriesEcon.jl/src/fconvert/fconvert_helpers.jl`` (the YP-only
subset; BDaily / Daily / Weekly date helpers are deferred to the follow-up
session that lands the calendar-frequency conversions).

Public surface re-exported from ``tsecon.fconvert``:

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
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

from tsecon.mit import MIT
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
