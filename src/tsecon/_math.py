# SPDX-License-Identifier: MIT
"""Time-series math helpers.

Implements ``shift`` / ``lag`` / ``lead`` / ``diff`` / ``pct`` / ``apct`` /
``ytypct`` / ``moving`` / ``moving_sum`` / ``moving_average`` / ``undiff`` /
``undiff_inplace``, ported from ``TimeSeriesEcon.jl``'s ``tsmath.jl`` (plus
``Base.diff(::TSeries)`` and the ``moving`` / ``undiff`` block from
``mvtseries.jl``; the latter is grouped here because both helpers work on
``TSeries`` and `MVTSeries` alike — only the ``TSeries`` overloads are ported
in M1, with MVTSeries overloads added when MVTSeries lands).

Convention follows the Julia source: positive ``k`` produces the lead (the
value at ``t+k`` is read at ``t``), negative ``k`` produces the lag. So
``shift(x, +1)`` moves the firstdate **earlier** by one period (the value
previously at ``2020Q1`` is now at ``2019Q4``).

In-place variants (``shift_inplace`` / ``lag_inplace`` / ``lead_inplace`` /
``undiff_inplace``) mirror the Julia ``shift!`` / ``lag!`` / ``lead!`` /
``undiff!`` family: they only mutate their first argument and reuse its
existing buffer (resizing as needed).

BDaily ``skip_all_nans`` / ``skip_holidays`` / ``holidays_map`` keyword
variants (``tsmath.jl`` lines 111+, ``tseries.jl`` line 525+) live on
``shift`` / ``shift_inplace`` / ``lag`` / ``lag_inplace`` / ``lead`` /
``lead_inplace`` / ``diff`` / ``pct``. They require a BDaily series and call
:func:`tsecon._bdaily.replace_nans_if_warranted` after the firstdate shift to
infill NaN entries. Passing any of these kwargs on a non-BDaily series
raises :class:`TypeError` to mirror Julia's type-dispatch separation.
"""

from __future__ import annotations

from typing import Any, Final, cast

import numpy as np

from tsecon._bdaily import replace_nans_if_warranted
from tsecon._kernel_dispatch import _dispatch_kernel, _is_kernel_eligible
from tsecon._math_kernels import cumsum_anchored_numpy
from tsecon.frequencies import BDaily, Frequency, YPFrequency, ppy, prettyprint_frequency
from tsecon.mit import MIT
from tsecon.mitrange import MITRange, rangeof_span
from tsecon.mvtseries import MVTSeries
from tsecon.tseries import TSeries

# Try to load the optional Cython-compiled accelerator. When the wheel was
# built without the C toolchain (or for editable installs that skipped the
# build hook), the import fails silently and undiff / undiff_inplace fall
# back to the NumPy reference. The public surface is otherwise unchanged.
try:
    from tsecon._math_kernels_cy import cumsum_anchored_cython  # type: ignore[import-not-found]

    _CYTHON_AVAILABLE: Final[bool] = True
except ImportError:
    _CYTHON_AVAILABLE = False  # type: ignore[misc]
    # Stub matches the cython name so `_dispatch_kernel` resolves cleanly
    # when the extension isn't built (see `_kernel_dispatch.py` docstring).
    cumsum_anchored_cython = cumsum_anchored_numpy

__all__ = [
    "apct",
    "diff",
    "lag",
    "lag_inplace",
    "lead",
    "lead_inplace",
    "moving",
    "moving_average",
    "moving_sum",
    "pct",
    "shift",
    "shift_inplace",
    "undiff",
    "undiff_inplace",
    "undiff_is_cython",
    "ytypct",
]


def undiff_is_cython() -> bool:
    """Return True iff the Cython-compiled cumsum-anchored kernel was importable.

    Useful for tests, benchmarks, and diagnostic prints — the public
    :func:`undiff` and :func:`undiff_inplace` are implementation-
    agnostic. When this returns ``False`` the same calls go through
    the pure-NumPy kernel in ``_math_kernels.py``; behaviour is
    identical, only speed differs.

    The fast path also requires the integrated buffer to be ``float64``
    + C-contiguous (the canonical kernel-eligibility contract); when
    either condition fails (e.g. integer ``dvar`` with integer
    anchor), the call routes through ``np.cumsum`` even when the
    extension is importable.
    """
    return _CYTHON_AVAILABLE


# An anchor passed to ``undiff`` can be (a) a scalar number; (b) a TSeries that
# the value is read from at ``firstdate(dvar)-1``; (c) a 2-tuple ``(date, value)``
# where ``value`` is a number; or (d) a 2-tuple ``(date, anchor_series)`` where
# the value is read from ``anchor_series[date]``.
_Number = float | int | bool | np.generic
_Anchor = _Number | TSeries | tuple[MIT, _Number] | tuple[MIT, TSeries]
# Anchor accepted by ``undiff`` on an MVTSeries — broader than the TSeries
# overload (scalar / vector / matrix / TSeries / MVTSeries / (MIT, *)-pair).
_MVAnchor = (
    _Number
    | TSeries
    | MVTSeries
    | np.ndarray
    | tuple[MIT, _Number]
    | tuple[MIT, np.ndarray]
    | tuple[MIT, TSeries]
    | tuple[MIT, MVTSeries]
)


def shift(
    t: TSeries | MVTSeries,
    k: int,
    *,
    skip_all_nans: bool = False,
    skip_holidays: bool = False,
    holidays_map: TSeries | None = None,
) -> TSeries | MVTSeries:
    """Return a copy of ``t`` with its dates shifted by ``k`` periods.

    By convention, positive ``k`` produces the *lead* (the value at ``t+k`` is
    read at ``t``); negative ``k`` produces the *lag*. The returned series has
    the same values but ``firstdate`` moved by ``-k`` periods.

    Accepts a :class:`~tsecon.tseries.TSeries` or
    :class:`~tsecon.mvtseries.MVTSeries`.

    BDaily kwargs (``skip_all_nans`` / ``skip_holidays`` / ``holidays_map``)
    are accepted only when ``t.frequency`` is :class:`BDaily`; on the shifted
    result they run :func:`tsecon._bdaily.replace_nans_if_warranted` to
    infill NaN entries. For an MVTSeries{BDaily} the infill is applied
    per-column (this extends Julia's TSeries-only overload — Julia's
    MVTSeries shift has no kwarg path, so a Python user passing kwargs would
    otherwise just lose them).
    """
    _check_bdaily_kwargs(t, skip_all_nans, skip_holidays, holidays_map)
    if isinstance(t, MVTSeries):
        return _shift_mvts_copy(
            t,
            int(k),
            skip_all_nans=skip_all_nans,
            skip_holidays=skip_holidays,
            holidays_map=holidays_map,
        )
    new_start = MIT(t.frequency, t.firstdate.value - int(k))
    new_values = t.values.copy()
    if _has_bdaily_kwargs(skip_all_nans, skip_holidays, holidays_map):
        if not np.issubdtype(new_values.dtype, np.floating):
            new_values = new_values.astype(np.float64)
        new_ts = TSeries(new_start, new_values)
        replace_nans_if_warranted(
            new_ts,
            int(k),
            skip_all_nans=skip_all_nans,
            skip_holidays=skip_holidays,
            holidays_map=holidays_map,
        )
        return new_ts
    return TSeries(new_start, new_values)


def shift_inplace(
    t: TSeries | MVTSeries,
    k: int,
    *,
    skip_all_nans: bool = False,
    skip_holidays: bool = False,
    holidays_map: TSeries | None = None,
) -> TSeries | MVTSeries:
    """In-place version of :func:`shift`. Mutates the firstdate(s) and returns the same object."""
    _check_bdaily_kwargs(t, skip_all_nans, skip_holidays, holidays_map)
    if isinstance(t, MVTSeries):
        new_start = MIT(t.frequency, t.firstdate.value - int(k))
        t._firstdate = new_start
        for col in t._columns.values():
            col._firstdate = MIT(col.frequency, col.firstdate.value - int(k))
        if _has_bdaily_kwargs(skip_all_nans, skip_holidays, holidays_map):
            _bdaily_infill_mvts_inplace(
                t,
                int(k),
                skip_all_nans=skip_all_nans,
                skip_holidays=skip_holidays,
                holidays_map=holidays_map,
            )
        return t
    t._firstdate = MIT(t.frequency, t.firstdate.value - int(k))
    if _has_bdaily_kwargs(skip_all_nans, skip_holidays, holidays_map):
        if not np.issubdtype(t.values.dtype, np.floating):
            msg = (
                "shift_inplace with BDaily kwargs requires a float-dtype series; "
                f"got dtype {t.values.dtype}. Use shift(...) for a fresh allocation."
            )
            raise TypeError(msg)
        replace_nans_if_warranted(
            t,
            int(k),
            skip_all_nans=skip_all_nans,
            skip_holidays=skip_holidays,
            holidays_map=holidays_map,
        )
    return t


def lag(
    t: TSeries | MVTSeries,
    k: int = 1,
    *,
    skip_all_nans: bool = False,
    skip_holidays: bool = False,
    holidays_map: TSeries | None = None,
) -> TSeries | MVTSeries:
    """Return the ``k``-th lag. Same as ``shift(t, -k)``."""
    return shift(
        t,
        -int(k),
        skip_all_nans=skip_all_nans,
        skip_holidays=skip_holidays,
        holidays_map=holidays_map,
    )


def lag_inplace(
    t: TSeries | MVTSeries,
    k: int = 1,
    *,
    skip_all_nans: bool = False,
    skip_holidays: bool = False,
    holidays_map: TSeries | None = None,
) -> TSeries | MVTSeries:
    """In-place version of :func:`lag`."""
    return shift_inplace(
        t,
        -int(k),
        skip_all_nans=skip_all_nans,
        skip_holidays=skip_holidays,
        holidays_map=holidays_map,
    )


def lead(
    t: TSeries | MVTSeries,
    k: int = 1,
    *,
    skip_all_nans: bool = False,
    skip_holidays: bool = False,
    holidays_map: TSeries | None = None,
) -> TSeries | MVTSeries:
    """Return the ``k``-th lead. Same as ``shift(t, k)``."""
    return shift(
        t,
        int(k),
        skip_all_nans=skip_all_nans,
        skip_holidays=skip_holidays,
        holidays_map=holidays_map,
    )


def lead_inplace(
    t: TSeries | MVTSeries,
    k: int = 1,
    *,
    skip_all_nans: bool = False,
    skip_holidays: bool = False,
    holidays_map: TSeries | None = None,
) -> TSeries | MVTSeries:
    """In-place version of :func:`lead`."""
    return shift_inplace(
        t,
        int(k),
        skip_all_nans=skip_all_nans,
        skip_holidays=skip_holidays,
        holidays_map=holidays_map,
    )


def diff(
    t: TSeries | MVTSeries,
    k: int = -1,
    *,
    skip_all_nans: bool = False,
    skip_holidays: bool = False,
    holidays_map: TSeries | None = None,
) -> TSeries | MVTSeries:
    """First (or ``k``-th) difference of ``t``.

    Defined as ``t - shift(t, k)``. With the default ``k=-1`` (subtract the
    lag), this matches the standard first-difference operator.

    For an :class:`~tsecon.mvtseries.MVTSeries` the operation is performed
    column-wise; the result is a new MVTSeries one row shorter (for
    ``|k|=1``). BDaily kwargs are forwarded to the inner ``lag`` call so
    NaN/holiday infill is applied to the lag side of the subtraction
    (mirrors ``Base.diff(::TSeries{BDaily})`` in ``tseries.jl``).
    """
    _check_bdaily_kwargs(t, skip_all_nans, skip_holidays, holidays_map)
    if isinstance(t, MVTSeries):
        if not _has_bdaily_kwargs(skip_all_nans, skip_holidays, holidays_map):
            return _diff_mvts(t, int(k))
        # BDaily kwargs path: build the lag (with infill) and subtract via MVTSeries arithmetic.
        lag_t = cast(
            "MVTSeries",
            lag(
                t,
                -int(k),
                skip_all_nans=skip_all_nans,
                skip_holidays=skip_holidays,
                holidays_map=holidays_map,
            ),
        )
        return cast("MVTSeries", t - lag_t)
    lag_ts = lag(
        t,
        -int(k),
        skip_all_nans=skip_all_nans,
        skip_holidays=skip_holidays,
        holidays_map=holidays_map,
    )
    return _as_tseries(t - lag_ts, reference=t)


def _diff_mvts(t: MVTSeries, k: int) -> MVTSeries:
    """Column-wise difference for an MVTSeries (avoids needing MVTSeries arithmetic)."""
    nrows = t.shape[0]
    ak = abs(k)
    if ak >= nrows:
        msg = f"diff window |k|={ak} is not smaller than the series length {nrows}."
        raise ValueError(msg)
    if k < 0:
        diffs = t.values[ak:, :] - t.values[: nrows - ak, :]
        new_first = MIT(t.frequency, t.firstdate.value + ak)
    else:
        diffs = t.values[: nrows - ak, :] - t.values[ak:, :]
        new_first = t.firstdate
    return MVTSeries(new_first, list(t._columns.keys()), diffs)


def pct(
    t: TSeries | MVTSeries,
    shift_value: int = -1,
    *,
    islog: bool = False,
    skip_all_nans: bool = False,
    skip_holidays: bool = False,
    holidays_map: TSeries | None = None,
) -> TSeries | MVTSeries:
    """Observation-to-observation percent rate of change.

    When ``islog`` is ``True``, ``t`` is interpreted as a log-series — values
    are exponentiated before differencing.

    For BDaily series, accepts ``skip_all_nans`` / ``skip_holidays`` /
    ``holidays_map`` kwargs. The kwargs are forwarded to the inner
    :func:`shift` call (the value that ends up in the denominator), so NaN
    / holiday infill on the shift result propagates into the resulting
    percent series.
    """
    _check_bdaily_kwargs(t, skip_all_nans, skip_holidays, holidays_map)
    if islog:
        if isinstance(t, MVTSeries):
            exp_arr = np.exp(np.asarray(t.values, dtype=np.float64))
            a: TSeries | MVTSeries = MVTSeries(t.firstdate, list(t._columns.keys()), exp_arr)
        else:
            a = TSeries(t.firstdate, np.exp(np.asarray(t.values, dtype=np.float64)))
        b = shift(
            a,
            int(shift_value),
            skip_all_nans=skip_all_nans,
            skip_holidays=skip_holidays,
            holidays_map=holidays_map,
        )
    else:
        a = t
        b = shift(
            t,
            int(shift_value),
            skip_all_nans=skip_all_nans,
            skip_holidays=skip_holidays,
            holidays_map=holidays_map,
        )
    if isinstance(t, MVTSeries):
        return cast("MVTSeries", ((a - b) / b) * 100)
    return _as_tseries(((a - b) / b) * 100, reference=t)


def apct(t: TSeries, islog: bool = False) -> TSeries:
    """Annualised percent rate of change.

    Requires a :class:`~tsecon.frequencies.YPFrequency` (Yearly / HalfYearly /
    Quarterly / Monthly). The annualisation exponent is ``ppy(t.frequency)``.
    """
    if not isinstance(t.frequency, YPFrequency):
        msg = f"apct is only defined for YPFrequency series; got {type(t.frequency).__name__}."
        raise TypeError(msg)
    n = ppy(t.frequency)
    if islog:
        exp_t = TSeries(t.firstdate, np.exp(t.values))
        a = exp_t
        b = shift(exp_t, -1)
    else:
        a = t
        b = shift(t, -1)
    return _as_tseries(((a / b) ** n - 1) * 100, reference=t)


def ytypct(t: TSeries) -> TSeries:
    """Year-to-year percent change.

    Requires a :class:`~tsecon.frequencies.YPFrequency`. Implemented as
    ``100 * (t / shift(t, -ppy(t.frequency)) - 1)``.
    """
    if not isinstance(t.frequency, YPFrequency):
        msg = f"ytypct is only defined for YPFrequency series; got {type(t.frequency).__name__}."
        raise TypeError(msg)
    return _as_tseries(100 * (t / shift(t, -ppy(t.frequency)) - 1), reference=t)


def _as_tseries(value: object, *, reference: TSeries) -> TSeries:
    """Ensure ``value`` is a TSeries; wrap a bare ndarray fallback.

    All callers in this module produce TSeries outputs through the
    ``__array_ufunc__`` machinery, but the static type of arithmetic on a
    TSeries is ``Any``. This thin helper narrows it back so callers and mypy
    both see a TSeries return.
    """
    if isinstance(value, TSeries):
        return value
    if isinstance(value, np.ndarray) and value.ndim == 1 and value.shape[0] == len(reference):
        return TSeries(reference.firstdate, value)
    msg = f"Expected TSeries result, got {type(value).__name__}."
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Moving sum / average / general moving (= average)
# ---------------------------------------------------------------------------


def moving(t: TSeries | MVTSeries, n: int) -> TSeries | MVTSeries:
    """Compute the moving average of ``t`` over a window of ``n`` periods.

    If ``n > 0`` the window is backward-looking ``(-n+1 .. 0)`` (the result at
    period ``p`` is the mean of ``t[p-n+1 .. p]``). If ``n < 0`` the window is
    forward-looking ``(0 .. -n-1)`` (the result at ``p`` is the mean of
    ``t[p .. p+|n|-1]``).

    The returned series has length ``len(t) - |n| + 1``. For ``n > 0`` its
    ``firstdate`` is ``t.firstdate + n - 1``; for ``n < 0`` it is
    ``t.firstdate``.

    Identical to :func:`moving_average`; the bare name matches the Julia
    convention. Accepts a :class:`~tsecon.tseries.TSeries` or
    :class:`~tsecon.mvtseries.MVTSeries`.
    """
    return moving_average(t, n)


def moving_average(t: TSeries | MVTSeries, n: int) -> TSeries | MVTSeries:
    """Compute the moving average of ``t`` over a window of ``n`` periods. See :func:`moving`."""
    if isinstance(t, MVTSeries):
        return _moving_sum_mvts(t, int(n), avg=True)
    return _moving_sum(t, int(n), avg=True)


def moving_sum(t: TSeries | MVTSeries, n: int) -> TSeries | MVTSeries:
    """Compute the rolling sum of ``t`` over a window of ``n`` periods.

    Identical to :func:`moving_average` but without dividing by ``|n|``.
    """
    if isinstance(t, MVTSeries):
        return _moving_sum_mvts(t, int(n), avg=False)
    return _moving_sum(t, int(n), avg=False)


def _moving_sum(t: TSeries, n: int, *, avg: bool) -> TSeries:
    if n == 0:
        msg = "moving window size n must be nonzero."
        raise ValueError(msg)
    an = abs(n)
    series_len = len(t)
    if an > series_len:
        msg = f"moving window size |n|={an} exceeds series length {series_len}."
        raise ValueError(msg)
    out_len = series_len - an + 1
    # Match Julia's `zeros(len + 1)` default: accumulate in float64 even if the
    # source is integer (or float32).
    arr = np.zeros(out_len, dtype=np.float64)
    src = t.values
    for i in range(an):
        arr += src[i : i + out_len]
    if avg:
        arr /= an
    offset = n - 1 if n > 0 else 0
    new_first = MIT(t.frequency, t.firstdate.value + offset)
    return TSeries(new_first, arr)


def _moving_sum_mvts(t: MVTSeries, n: int, *, avg: bool) -> MVTSeries:
    if n == 0:
        msg = "moving window size n must be nonzero."
        raise ValueError(msg)
    an = abs(n)
    nrows = t.shape[0]
    if an > nrows:
        msg = f"moving window size |n|={an} exceeds series length {nrows}."
        raise ValueError(msg)
    out_len = nrows - an + 1
    arr = np.zeros((out_len, t.shape[1]), dtype=np.float64)
    src = t.values
    for i in range(an):
        arr += src[i : i + out_len, :]
    if avg:
        arr /= an
    offset = n - 1 if n > 0 else 0
    new_first = MIT(t.frequency, t.firstdate.value + offset)
    return MVTSeries(new_first, list(t._columns.keys()), arr)


# ---------------------------------------------------------------------------
# undiff / undiff_inplace
# ---------------------------------------------------------------------------


def undiff(
    dvar: TSeries | MVTSeries,
    anchor: _Anchor | _MVAnchor = 0,
) -> TSeries | MVTSeries:
    """Inverse of :func:`diff` (cumulative sum, anchored at a known value).

    ``dvar`` is the differenced series. ``anchor`` says what the *integrated*
    series equals at some anchor date; the result is the cumulative sum of
    ``dvar`` shifted so that ``result[date] == value``. Forms accepted:

    * ``anchor=v`` (a number, default ``0``) — anchor date defaults to
      ``firstdate(dvar) - 1`` (the period just before the differencing window).
      If that date is outside ``dvar.range``, ``dvar`` is extended with zeros
      so that the anchor falls inside.
    * ``anchor=(date, value)`` — both date and value given explicitly.
    * ``anchor=other_tseries`` — anchor date defaults to ``firstdate(dvar)-1``;
      the value is read from ``other_tseries`` at that date.
    * ``anchor=(date, other_tseries)`` — value is read from
      ``other_tseries[date]``.

    Forward vs. backward integration
    --------------------------------
    The math is direction-agnostic: ``undiff`` computes a cumulative sum and
    chooses the constant of integration so ``result[anchor_date] == anchor_value``.
    The anchor's position determines the user-visible direction:

    * **Forward** — anchor at or before ``firstdate(dvar) - 1`` (the default
      ``anchor=v`` form). Values past the anchor are accumulated forward.
    * **Backward (backcasting)** — anchor at ``lastdate(dvar)`` or later,
      e.g. ``undiff(dvar, (dvar.lastdate, terminal_value))``. Values
      before the anchor are recovered by walking the same cumulative-sum
      curve backward; result spans ``dvar.range``.

    An anchor strictly *after* ``lastdate(dvar)`` extends ``dvar`` with zeros
    so the anchor falls inside the new range — the tail (positions strictly
    between ``lastdate(dvar)`` and the anchor) is flat at the anchor value.

    Note on a mid-range anchor: when ``date`` falls inside ``dvar.range``, the
    cumulative sum is still computed over the full range; the result is then
    shifted by a constant so ``result[date] == value`` — every other period
    moves by the same constant. See ``undiff_inplace`` for the alternative
    semantics where ``dvar`` values at and before ``fromdate`` are ignored.

    On an :class:`~tsecon.mvtseries.MVTSeries`, ``anchor`` may additionally
    be a vector / matrix (one entry per column) or another MVTSeries; the
    cumulative sum is performed column-wise.

    Dispatch
    --------
    The TSeries path's hot loop (cumsum over the integrated chunk +
    constant-shift correction) routes through the
    :func:`tsecon._math_kernels.cumsum_anchored_numpy` /
    :func:`tsecon._math_kernels_cy.cumsum_anchored_cython` kernel pair
    when the integrated buffer is ``float64`` + C-contiguous; use
    :func:`undiff_is_cython` to check which path is active. Integer
    ``dvar`` + integer anchor combinations (which would broaden the
    result dtype away from ``float64``) stay on the pure-NumPy path
    so the integer return type is preserved. The MVTSeries path stays
    pure NumPy in M1.6 — ``np.cumsum(..., axis=0)`` already runs the
    column-wise reduction in C, and the outer benchmark ratio sits at
    14× rather than the 25-80× band the kernel addresses.
    """
    if isinstance(dvar, MVTSeries):
        return _undiff_mvts(dvar, anchor)
    ad, av = _resolve_undiff_anchor(dvar, anchor)
    # Promote dtype so that an integer dvar + float anchor produces a float
    # series, matching Julia's `Base.promote_eltype(dvar, value)`.
    av_arr = np.asarray(av)
    et = np.result_type(dvar.values.dtype, av_arr.dtype)
    rng = dvar.range
    if not (rng.start <= ad <= rng.stop):
        # Extend dvar with zeros so the anchor period is inside.
        new_range = rangeof_span(MITRange(ad, ad), rng)
        extended = TSeries(new_range, 0, dtype=et)
        if not rng.is_empty():
            extended[rng] = dvar
        dvar = extended
        rng = new_range
    ad_idx = ad.value - rng.start.value
    # Copy dvar values into a fresh float64 buffer; the kernel mutates it
    # in place (cumsum + correction). For non-float64 result dtypes (e.g.
    # int dvar + int anchor) stay on the NumPy path so the integer return
    # type is preserved.
    if et == np.float64:
        result_arr = dvar.values.astype(np.float64, copy=True)
        if _is_kernel_eligible(result_arr):
            _dispatch_kernel(
                _CYTHON_AVAILABLE,
                cumsum_anchored_cython,
                cumsum_anchored_numpy,
                result_arr,
                0,
                result_arr.shape[0],
                float(av),
                ad_idx,
            )
            return TSeries(rng.start, result_arr)
    # Non-float64 fallback (preserves integer return dtype):
    result_arr = np.cumsum(dvar.values).astype(et, copy=True)
    correction = av - result_arr[ad_idx]
    return TSeries(rng.start, result_arr + correction)


def undiff_inplace(
    var: TSeries,
    dvar: TSeries,
    *,
    fromdate: MIT | None = None,
) -> TSeries:
    """Anchor-based undiff that writes into ``var`` (mirrors Julia ``undiff!``).

    Reads the anchor value from ``var[fromdate]`` and writes the integrated
    series into ``var`` at ``fromdate+1 .. lastdate(dvar)``. Values of ``var``
    at and before ``fromdate`` are left untouched, and values of ``dvar`` at
    and before ``fromdate`` are ignored (treated as zero).

    If ``var`` does not yet extend to ``lastdate(dvar)``, it is resized
    (extending the end). ``fromdate`` defaults to ``firstdate(dvar) - 1`` and
    must satisfy ``fromdate >= firstdate(var)``.

    .. note::

       The semantics here differ from :func:`undiff` when ``fromdate`` falls
       in the middle of ``rangeof(dvar)``. There, :func:`undiff` shifts the
       *whole* cumulative result so ``result[fromdate] == value``;
       :func:`undiff_inplace` zeros out the earlier ``dvar`` entries instead,
       so the integrated tail starts cleanly from the existing anchor.

    Dispatch
    --------
    Shares the cumsum + anchor-shift kernel with :func:`undiff` — the
    inplace variant calls into the same
    :func:`tsecon._math_kernels.cumsum_anchored_numpy` /
    :func:`tsecon._math_kernels_cy.cumsum_anchored_cython` pair with
    ``anchor_relative_idx = -1`` (the anchor sits one position before
    the integrated chunk, so the correction collapses to a single
    fused add per element). Use :func:`undiff_is_cython` to check
    which path is active. The kernel only engages when ``var.values``
    is ``float64`` + C-contiguous; other dtypes fall back to the
    pre-existing ``np.cumsum`` path.
    """
    if var.frequency != dvar.frequency:
        raise _mixed_freq(var.frequency, dvar.frequency)
    fd = MIT(dvar.frequency, dvar.firstdate.value - 1) if fromdate is None else fromdate
    if fd.frequency != dvar.frequency:
        raise _mixed_freq(fd.frequency, dvar.frequency)
    if fd < var.firstdate:
        msg = f"Range mismatch: fromdate {fd!s} < firstdate(var) {var.firstdate!s}."
        raise ValueError(msg)
    if var.lastdate < dvar.lastdate:
        var.resize(MITRange(var.firstdate, dvar.lastdate))
    if fd >= dvar.lastdate:
        return var
    # Compute var[fd+1 .. lastdate(dvar)] = var[fd] + cumsum(dvar[fd+1 .. lastdate]).
    start_val = fd.value + 1
    stop_val = dvar.lastdate.value
    n = stop_val - start_val + 1
    dvar_off = start_val - dvar.firstdate.value
    if dvar_off < 0:
        msg = (
            f"fromdate {fd!s} is too far before firstdate(dvar) "
            f"{dvar.firstdate!s}; dvar lookups would be out of range."
        )
        raise ValueError(msg)
    var_anchor_idx = fd.value - var.firstdate.value
    var_write_start = var_anchor_idx + 1
    if (
        var.values.dtype == np.float64
        and dvar.values.dtype == np.float64
        and _is_kernel_eligible(var.values)
    ):
        # Copy the dvar chunk into the destination slot, then run the
        # in-place cumsum kernel anchored at the position just before
        # the chunk (anchor_relative_idx = -1, reference cumsum = 0).
        var.values[var_write_start : var_write_start + n] = dvar.values[dvar_off : dvar_off + n]
        _dispatch_kernel(
            _CYTHON_AVAILABLE,
            cumsum_anchored_cython,
            cumsum_anchored_numpy,
            var.values,
            var_write_start,
            n,
            float(var.values[var_anchor_idx]),
            -1,
        )
        return var
    # Non-float64 fallback (preserves the original dtype-promoting cumsum path):
    dvar_chunk = dvar.values[dvar_off : dvar_off + n]
    var.values[var_write_start : var_write_start + n] = var.values[var_anchor_idx] + np.cumsum(
        dvar_chunk
    )
    return var


# ---------------------------------------------------------------------------
# undiff anchor resolution + frequency-mismatch helper
# ---------------------------------------------------------------------------


def _undiff_mvts(dvar: MVTSeries, anchor: object) -> MVTSeries:
    """``undiff`` for an MVTSeries: column-wise integration with a vector anchor."""
    ad, av_row = _resolve_undiff_anchor_mvts(dvar, anchor)
    ncols = dvar.shape[1]
    # Promote element type to accommodate the anchor.
    et = np.result_type(dvar.values.dtype, av_row.dtype)
    rng = dvar.range
    if not (rng.start <= ad <= rng.stop):
        # Extend dvar with zeros so the anchor period is inside the range.
        new_range = rangeof_span(MITRange(ad, ad), rng)
        new_n = len(new_range)
        new_arr = np.zeros((new_n, ncols), dtype=et)
        if not rng.is_empty():
            off = rng.start.value - new_range.start.value
            new_arr[off : off + len(rng), :] = dvar.values
        dvar_values = new_arr
        rng = new_range
    else:
        dvar_values = dvar.values.astype(et, copy=True)
    result = np.cumsum(dvar_values, axis=0)
    ad_idx = ad.value - rng.start.value
    correction = av_row.astype(et, copy=False) - result[ad_idx, :]
    result = result + correction
    return MVTSeries(rng.start, list(dvar._columns.keys()), result)


def _resolve_undiff_anchor_mvts(dvar: MVTSeries, anchor: object) -> tuple[MIT, np.ndarray]:
    """Normalize ``anchor`` into a ``(MIT, row-vector)`` pair for ``undiff(MVTSeries)``."""
    default_date = MIT(dvar.frequency, dvar.firstdate.value - 1)
    ncols = dvar.shape[1]

    def _broadcast(value: object) -> np.ndarray:
        if _is_number(value):
            return np.full(ncols, value)
        arr = np.asarray(value)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr.reshape(-1)
        if arr.ndim != 1 or arr.shape[0] != ncols:
            msg = (
                f"undiff anchor vector length {arr.shape!r} does not match MVTSeries "
                f"column count {ncols}."
            )
            raise ValueError(msg)
        return arr

    if isinstance(anchor, MVTSeries):
        if anchor.frequency != dvar.frequency:
            raise _mixed_freq(anchor.frequency, dvar.frequency)
        return default_date, np.asarray(anchor[default_date])
    if isinstance(anchor, TSeries):
        if anchor.frequency != dvar.frequency:
            raise _mixed_freq(anchor.frequency, dvar.frequency)
        return default_date, np.full(ncols, anchor[default_date])
    if isinstance(anchor, tuple):
        if len(anchor) != 2 or not isinstance(anchor[0], MIT):
            msg = "MVTSeries undiff anchor tuple must start with an MIT."
            raise TypeError(msg)
        date, val = anchor
        if date.frequency != dvar.frequency:
            raise _mixed_freq(date.frequency, dvar.frequency)
        if isinstance(val, MVTSeries):
            if val.frequency != dvar.frequency:
                raise _mixed_freq(val.frequency, dvar.frequency)
            return date, np.asarray(val[date])
        if isinstance(val, TSeries):
            if val.frequency != dvar.frequency:
                raise _mixed_freq(val.frequency, dvar.frequency)
            return date, np.full(ncols, val[date])
        if _is_number(val):
            return date, np.full(ncols, val)
        return date, _broadcast(val)
    if isinstance(anchor, MIT):
        msg = "MVTSeries undiff anchor must be paired with a value when given as an MIT."
        raise TypeError(msg)
    if _is_number(anchor) or isinstance(anchor, np.ndarray):
        return default_date, _broadcast(anchor)
    msg = f"MVTSeries undiff anchor type not supported: {type(anchor).__name__}."
    raise TypeError(msg)


def _is_number(x: object) -> bool:
    return isinstance(x, (bool, int, float, np.generic))


def _resolve_undiff_anchor(dvar: TSeries, anchor: object) -> tuple[MIT, Any]:
    """Normalize ``anchor`` into a ``(MIT, scalar)`` pair for :func:`undiff`."""
    default_date = MIT(dvar.frequency, dvar.firstdate.value - 1)
    if isinstance(anchor, MIT):
        msg = (
            "undiff anchor must be a number, a TSeries, or a (MIT, value) pair; "
            "got a bare MIT (did you forget to pair it with a value?)."
        )
        raise TypeError(msg)
    if isinstance(anchor, TSeries):
        if anchor.frequency != dvar.frequency:
            raise _mixed_freq(anchor.frequency, dvar.frequency)
        return default_date, anchor[default_date]
    if isinstance(anchor, tuple):
        if len(anchor) != 2 or not isinstance(anchor[0], MIT):
            msg = (
                "undiff anchor tuple must be (MIT, value) or (MIT, TSeries); "
                f"got tuple of len {len(anchor)} starting with {type(anchor[0]).__name__}."
            )
            raise TypeError(msg)
        date, val = anchor
        if date.frequency != dvar.frequency:
            raise _mixed_freq(date.frequency, dvar.frequency)
        if isinstance(val, TSeries):
            if val.frequency != dvar.frequency:
                raise _mixed_freq(val.frequency, dvar.frequency)
            return date, val[date]
        if isinstance(val, (bool, int, float, np.generic)):
            return date, val
        msg = f"undiff anchor value must be a number or TSeries; got {type(val).__name__}."
        raise TypeError(msg)
    if isinstance(anchor, (bool, int, float, np.generic)):
        return default_date, anchor
    msg = (
        "undiff anchor must be a number, TSeries, or (MIT, value) pair; "
        f"got {type(anchor).__name__}."
    )
    raise TypeError(msg)


def _mixed_freq(left: object, right: object) -> TypeError:
    left_label = prettyprint_frequency(left) if isinstance(left, Frequency) else type(left).__name__
    right_label = (
        prettyprint_frequency(right) if isinstance(right, Frequency) else type(right).__name__
    )
    return TypeError(f"Mixing frequencies not allowed: {left_label} and {right_label}.")


# ---------------------------------------------------------------------------
# BDaily kwarg plumbing
# ---------------------------------------------------------------------------


def _has_bdaily_kwargs(
    skip_all_nans: bool, skip_holidays: bool, holidays_map: TSeries | None
) -> bool:
    return bool(skip_all_nans) or bool(skip_holidays) or holidays_map is not None


def _check_bdaily_kwargs(
    t: TSeries | MVTSeries,
    skip_all_nans: bool,
    skip_holidays: bool,
    holidays_map: TSeries | None,
) -> None:
    """Reject BDaily kwargs on non-BDaily series.

    Mirrors Julia's type-dispatch separation: the kwargs only exist on the
    ``TSeries{BDaily}`` overloads of ``shift`` / ``lag`` / ``lead`` /
    ``diff`` / ``pct``.
    """
    if not _has_bdaily_kwargs(skip_all_nans, skip_holidays, holidays_map):
        return
    if not isinstance(t.frequency, BDaily):
        msg = (
            "skip_all_nans / skip_holidays / holidays_map keywords are only "
            f"valid on BDaily series; got frequency {type(t.frequency).__name__}."
        )
        raise TypeError(msg)


def _shift_mvts_copy(
    t: MVTSeries,
    k: int,
    *,
    skip_all_nans: bool,
    skip_holidays: bool,
    holidays_map: TSeries | None,
) -> MVTSeries:
    """Shift an MVTSeries by ``k`` periods, applying BDaily infill if requested.

    Without BDaily kwargs this is the original copy + firstdate-shift; with
    kwargs, the result's underlying values are float-promoted and each column
    independently gets ``replace_nans_if_warranted`` applied (per-column to
    mirror what the user would get by calling ``shift`` on each column
    separately).
    """
    out = t.copy()
    new_start = MIT(t.frequency, t.firstdate.value - int(k))
    out._firstdate = new_start
    for col in out._columns.values():
        col._firstdate = MIT(col.frequency, col.firstdate.value - int(k))
    if _has_bdaily_kwargs(skip_all_nans, skip_holidays, holidays_map):
        _bdaily_infill_mvts_inplace(
            out,
            int(k),
            skip_all_nans=skip_all_nans,
            skip_holidays=skip_holidays,
            holidays_map=holidays_map,
        )
    return out


def _bdaily_infill_mvts_inplace(
    t: MVTSeries,
    k: int,
    *,
    skip_all_nans: bool,
    skip_holidays: bool,
    holidays_map: TSeries | None,
) -> None:
    """Apply :func:`replace_nans_if_warranted` to each column of an MVTSeries{BDaily}."""
    if not np.issubdtype(t.values.dtype, np.floating):
        # MVTSeries inplace currently shares the underlying float array between
        # the matrix and column views; the column shift path already preserves
        # the float assumption. Mirror that here by requiring float-dtype input.
        msg = (
            "shift with BDaily kwargs on an MVTSeries requires a float-dtype "
            f"underlying array; got dtype {t.values.dtype}."
        )
        raise TypeError(msg)
    for col in t._columns.values():
        replace_nans_if_warranted(
            col,
            int(k),
            skip_all_nans=skip_all_nans,
            skip_holidays=skip_holidays,
            holidays_map=holidays_map,
        )
