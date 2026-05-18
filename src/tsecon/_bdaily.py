# SPDX-License-Identifier: MIT
"""BDaily-specific helpers.

The ``cleanedvalues`` family and the ``replace_nans_if_warranted`` infill
used by the BDaily ``shift`` overload.

Mirrors the BDaily branches of ``TimeSeriesEcon.jl``'s ``tseries.jl``
(``cleanedvalues(TSeries{BDaily}; ...)``, ``bdvalues``), ``mvtseries.jl``
(``cleanedvalues(MVTSeries{BDaily}; ...)``, ``bdvalues`` MVTSeries variant,
``nans_map``), and ``tsmath.jl`` (``replace_nans_if_warranted!``).

The holidays-map convention is the Julia one: ``True`` marks a regular
business day, ``False`` marks a holiday. ``skip_holidays=True`` (without a
``holidays_map`` kwarg) reads the calendar from :func:`tsecon.getoption`
(option name ``"bdaily_holidays_map"``); a missing or wrong-shape global map
raises :class:`ValueError` to match Julia's ``ArgumentError``.

These functions are BDaily-only; they raise :class:`TypeError` when called
on a non-BDaily series.
"""

from __future__ import annotations

import warnings

import numpy as np

from tsecon._options import getoption
from tsecon.frequencies import BDaily
from tsecon.mit import MIT
from tsecon.mitrange import MITRange
from tsecon.mvtseries import MVTSeries
from tsecon.tseries import TSeries

__all__ = [
    "bdaily_row_keep_mask",
    "bdvalues",
    "cleanedvalues",
    "nans_map",
    "replace_nans_if_warranted",
]


def cleanedvalues(
    t: TSeries | MVTSeries,
    *,
    skip_all_nans: bool = False,
    skip_holidays: bool = False,
    holidays_map: TSeries | None = None,
) -> np.ndarray:
    """Return a filtered view of the underlying values of a BDaily series.

    Behaviour mirrors Julia's ``cleanedvalues`` overloads:

    * ``holidays_map`` is given → use that map; entries where the map is
      ``False`` are dropped.
    * ``holidays_map is None`` and ``skip_all_nans=True`` → drop NaN entries.
      For an :class:`~tsecon.mvtseries.MVTSeries`, only rows where *all*
      columns are NaN are dropped; rows where some columns are NaN and some
      are not emit a :class:`UserWarning` and are also dropped.
    * ``holidays_map is None`` and ``skip_holidays=True`` → fetch the global
      map from ``getoption("bdaily_holidays_map")``; raise if missing.
    * No kwargs → return the underlying values unchanged.

    The 1-D return is ``t.values`` for a :class:`~tsecon.tseries.TSeries`; the
    2-D return is ``mvts.values`` (an ``(rows, cols)`` view) for an
    :class:`~tsecon.mvtseries.MVTSeries`.
    """
    _require_bdaily(t)
    if holidays_map is not None:
        return bdvalues(t, holidays_map=holidays_map)
    if skip_all_nans:
        return _drop_all_nan_rows(t)
    if skip_holidays:
        h_map = getoption("bdaily_holidays_map")
        if not isinstance(h_map, TSeries) or not isinstance(h_map.frequency, BDaily):
            msg = (
                "skip_holidays=True requires a BDaily TSeries stored under "
                "getoption('bdaily_holidays_map'); none is currently set. "
                "Use tsecon.set_holidays_map(...) to install one."
            )
            raise ValueError(msg)
        return bdvalues(t, holidays_map=h_map)
    return np.asarray(t.values)


def bdvalues(t: TSeries | MVTSeries, *, holidays_map: TSeries | None = None) -> np.ndarray:
    """Return values of ``t`` masked by ``holidays_map``.

    ``holidays_map`` must be a BDaily Boolean :class:`~tsecon.tseries.TSeries`
    that spans ``t.range``. Entries where the map is ``False`` are dropped;
    the result is the underlying ndarray (1-D for TSeries, 2-D for MVTSeries)
    indexed by the Boolean mask.
    """
    _require_bdaily(t)
    if holidays_map is None:
        return np.asarray(t.values)
    _validate_holidays_arg(holidays_map)
    rng = t.range
    if rng.first() < holidays_map.firstdate or rng.last() > holidays_map.lastdate:
        msg = (
            "holidays_map does not cover the full range of the series: "
            f"series range {rng}, map range "
            f"{MITRange(holidays_map.firstdate, holidays_map.lastdate)}."
        )
        raise IndexError(msg)
    slice_ = holidays_map[rng]
    mask = np.asarray(slice_.values, dtype=bool)
    vals = np.asarray(t.values)
    if isinstance(t, MVTSeries):
        return np.asarray(vals[mask, :])
    return np.asarray(vals[mask])


def bdaily_row_keep_mask(
    t: MVTSeries,
    *,
    skip_all_nans: bool = False,
    skip_holidays: bool = False,
    holidays_map: TSeries | None = None,
) -> np.ndarray | None:
    """Return a ``(nrows,)`` bool keep-mask, or ``None`` when no filter is requested.

    ``True`` flags rows that survive the BDaily filter; ``False`` flags rows
    dropped by ``cleanedvalues``. The companion to :func:`cleanedvalues` —
    where ``cleanedvalues`` returns the filtered matrix (rows dropped), this
    returns the boolean mask itself so callers that need the original
    positions (e.g. ``axis=1`` per-row reductions producing an output TSeries
    aligned with the input range) can mask the output instead of dropping.

    Returns ``None`` iff none of the BDaily kwargs is set; this lets callers
    write ``mask := bdaily_row_keep_mask(...); if mask is None: skip``
    without re-doing the kwarg check.
    """
    if not (skip_all_nans or skip_holidays or holidays_map is not None):
        return None
    _require_bdaily(t)
    rng = t.range
    if holidays_map is not None:
        _validate_holidays_arg(holidays_map)
        if rng.first() < holidays_map.firstdate or rng.last() > holidays_map.lastdate:
            msg = (
                "holidays_map does not cover the full range of the series: "
                f"series range {rng}, map range "
                f"{MITRange(holidays_map.firstdate, holidays_map.lastdate)}."
            )
            raise IndexError(msg)
        return np.asarray(holidays_map[rng].values, dtype=bool)
    if skip_holidays:
        h_map = getoption("bdaily_holidays_map")
        if not isinstance(h_map, TSeries) or not isinstance(h_map.frequency, BDaily):
            msg = (
                "skip_holidays=True requires a BDaily TSeries stored under "
                "getoption('bdaily_holidays_map'); none is currently set. "
                "Use tsecon.set_holidays_map(...) to install one."
            )
            raise ValueError(msg)
        return np.asarray(h_map[rng].values, dtype=bool)
    keep = ~np.all(np.isnan(np.asarray(t.values, dtype=float)), axis=1)
    return np.asarray(keep, dtype=bool)


def nans_map(values: np.ndarray) -> np.ndarray:
    """Return a two-column Boolean mask matrix for an MVTSeries' values.

    Column 0 is ``True`` whenever *some* of the entries in that row are not
    NaN; column 1 is ``True`` whenever *all* of the entries in that row are
    not NaN. Mirrors Julia's ``nans_map`` in ``mvtseries.jl``.
    """
    arr = np.asarray(values, dtype=float)
    not_nan = ~np.isnan(arr)
    any_valid = not_nan.any(axis=1)
    all_valid = not_nan.all(axis=1)
    return np.column_stack([any_valid, all_valid])


def replace_nans_if_warranted(
    ts: TSeries,
    k: int,
    *,
    skip_all_nans: bool = False,
    skip_holidays: bool = False,
    holidays_map: TSeries | None = None,
) -> None:
    """In-place NaN infill on a (shifted) BDaily TSeries.

    Mirrors Julia's ``replace_nans_if_warranted!``. Called by the BDaily
    overload of :func:`tsecon.shift` after the firstdate has been adjusted by
    ``-k``. With ``skip_all_nans=True`` every NaN in ``ts.values`` is
    replaced by the nearest valid forward (for ``k > 0``) or backward
    (for ``k < 0``) value. With ``skip_holidays=True`` or ``holidays_map=...``
    only NaNs whose *source* date (in the pre-shift series) was a holiday are
    infilled.

    When neither ``skip_all_nans`` nor ``skip_holidays`` is set and no map is
    given, the function returns without mutating ``ts``.
    """
    _require_bdaily(ts)
    if not (skip_all_nans or skip_holidays or holidays_map is not None):
        return
    ts_range = ts.range
    holidays: np.ndarray | None = None
    if holidays_map is not None:
        _validate_holidays_arg(holidays_map)
        skip_holidays = True  # holidays_map subsumes/overrides the global flag
        holidays = _slice_holidays_with_pad(holidays_map, ts_range, k)
    elif skip_holidays:
        h_map = getoption("bdaily_holidays_map")
        if not isinstance(h_map, TSeries) or not isinstance(h_map.frequency, BDaily):
            msg = (
                "skip_holidays=True requires a BDaily TSeries stored under "
                "getoption('bdaily_holidays_map'); none is currently set."
            )
            raise ValueError(msg)
        holidays = _slice_holidays_with_pad(h_map, ts_range, k)
    direction_next = k > 0
    values = ts.values
    n = len(values)
    if direction_next:
        # Walk right-to-left so we pick up "the nearest valid value to the right".
        last_valid = np.nan
        for j in range(n):
            i = n - 1 - j
            input_is_holiday = (
                (holidays is not None) and skip_holidays and not holidays[i + k + abs(k)]
            )
            val = values[i]
            if np.isnan(val):
                if skip_all_nans or (skip_holidays and input_is_holiday):
                    values[i] = last_valid
                elif skip_holidays and not input_is_holiday:
                    last_valid = val
            elif skip_all_nans or (skip_holidays and not input_is_holiday):
                last_valid = val
    else:
        # Walk left-to-right for "the nearest valid value to the left".
        last_valid = np.nan
        for i in range(n):
            input_is_holiday = (
                (holidays is not None) and skip_holidays and not holidays[i + k + abs(k)]
            )
            val = values[i]
            if np.isnan(val):
                if skip_all_nans or (skip_holidays and input_is_holiday):
                    values[i] = last_valid
                elif skip_holidays and not input_is_holiday:
                    last_valid = val
            elif skip_all_nans or (skip_holidays and not input_is_holiday):
                last_valid = val


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _drop_all_nan_rows(t: TSeries | MVTSeries) -> np.ndarray:
    if isinstance(t, MVTSeries):
        mask_matrix = nans_map(np.asarray(t.values))
        # Column 0: any-non-nan in row. Column 1: all-non-nan in row.
        any_mask = mask_matrix[:, 0]
        all_mask = mask_matrix[:, 1]
        # Emit Julia's "NaNs unequal across columns" warning if some rows have
        # *some* valid values but not *all*. We compare on the rows we'd keep
        # under the any-valid mask to detect partial-NaN rows that survive the
        # row-level "drop only if entirely NaN" decision.
        if bool(np.any(np.isnan(np.asarray(t.values)[any_mask & ~all_mask, :]))):
            warnings.warn(
                "NaNs unequal across columns. Rows with some valid values removed.",
                UserWarning,
                stacklevel=3,
            )
        return np.asarray(np.asarray(t.values)[all_mask, :])
    arr = np.asarray(t.values, dtype=float)
    mask = ~np.isnan(arr)
    return np.asarray(arr[mask])


def _slice_holidays_with_pad(holidays_map: TSeries, rng: MITRange, k: int) -> np.ndarray:
    """Return a padded slice of ``holidays_map`` covering ``rng`` ± ``|k|`` periods.

    Mirrors Julia's
    ``holidays_map[ts_range[begin]-abs(k):ts_range[end]+abs(k)]``. The slice
    is indexed in :func:`replace_nans_if_warranted` by ``i + abs(k)``, where
    ``i`` is the zero-based index inside the shifted series.
    """
    _validate_holidays_arg(holidays_map)
    ak = abs(int(k))
    pad_start = MIT(rng.frequency, rng.first().value - ak)
    pad_end = MIT(rng.frequency, rng.last().value + ak)
    if pad_start < holidays_map.firstdate or pad_end > holidays_map.lastdate:
        msg = (
            "holidays_map does not cover the padded range required by "
            f"shift(k={k}): need {MITRange(pad_start, pad_end)}, map range "
            f"{MITRange(holidays_map.firstdate, holidays_map.lastdate)}."
        )
        raise IndexError(msg)
    return np.asarray(holidays_map[MITRange(pad_start, pad_end)].values, dtype=bool)


def _require_bdaily(t: TSeries | MVTSeries) -> None:
    if not isinstance(t.frequency, BDaily):
        msg = (
            f"This operation is only defined for BDaily series; got frequency "
            f"{type(t.frequency).__name__}."
        )
        raise TypeError(msg)


def _validate_holidays_arg(holidays_map: object) -> None:
    if not isinstance(holidays_map, TSeries):
        msg = f"holidays_map must be a BDaily Boolean TSeries; got {type(holidays_map).__name__}."
        raise TypeError(msg)
    if not isinstance(holidays_map.frequency, BDaily):
        msg = (
            "holidays_map must be a BDaily TSeries; got frequency "
            f"{type(holidays_map.frequency).__name__}."
        )
        raise TypeError(msg)
    if holidays_map.values.dtype != bool:
        msg = f"holidays_map must be a Boolean TSeries; got dtype {holidays_map.values.dtype}."
        raise TypeError(msg)
