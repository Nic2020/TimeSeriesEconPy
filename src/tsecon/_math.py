# SPDX-License-Identifier: MIT
"""Time-series math helpers.

Implements ``shift`` / ``lag`` / ``lead`` / ``diff`` / ``pct`` / ``apct`` /
``ytypct``, ported from ``TimeSeriesEcon.jl``'s ``tsmath.jl`` (and
``Base.diff(::TSeries)`` from ``tseries.jl``).

Convention follows the Julia source: positive ``k`` produces the lead (the
value at ``t+k`` is read at ``t``), negative ``k`` produces the lag. So
``shift(x, +1)`` moves the firstdate **earlier** by one period (the value
previously at ``2020Q1`` is now at ``2019Q4``).

In-place variants (``shift_inplace`` / ``lag_inplace`` / ``lead_inplace``) mirror
the Julia ``shift!`` / ``lag!`` / ``lead!`` family: they only mutate the
``firstdate`` and reuse the existing buffer.

BDaily-specific ``skip_all_nans`` / ``skip_holidays`` keyword variants are
deferred until ``options.jl`` is ported (see ``parity/PARITY.md``).
"""

from __future__ import annotations

import numpy as np

from tsecon.frequencies import YPFrequency, ppy
from tsecon.mit import MIT
from tsecon.mitrange import MITRange
from tsecon.tseries import TSeries

__all__ = [
    "apct",
    "diff",
    "lag",
    "lag_inplace",
    "lead",
    "lead_inplace",
    "pct",
    "shift",
    "shift_inplace",
    "ytypct",
]


def shift(t: TSeries, k: int) -> TSeries:
    """Return a copy of ``t`` with its dates shifted by ``k`` periods.

    By convention, positive ``k`` produces the *lead* (the value at ``t+k`` is
    read at ``t``); negative ``k`` produces the *lag*. The returned TSeries has
    the same values but ``firstdate`` moved by ``-k`` periods.
    """
    new_start = MIT(t.frequency, t.firstdate.value - int(k))
    return TSeries(new_start, t.values.copy())


def shift_inplace(t: TSeries, k: int) -> TSeries:
    """In-place version of :func:`shift`. Mutates ``t.firstdate`` and returns ``t``."""
    t._firstdate = MIT(t.frequency, t.firstdate.value - int(k))
    return t


def lag(t: TSeries, k: int = 1) -> TSeries:
    """Return the ``k``-th lag of ``t``. Same as ``shift(t, -k)``."""
    return shift(t, -int(k))


def lag_inplace(t: TSeries, k: int = 1) -> TSeries:
    """In-place version of :func:`lag`."""
    return shift_inplace(t, -int(k))


def lead(t: TSeries, k: int = 1) -> TSeries:
    """Return the ``k``-th lead of ``t``. Same as ``shift(t, k)``."""
    return shift(t, int(k))


def lead_inplace(t: TSeries, k: int = 1) -> TSeries:
    """In-place version of :func:`lead`."""
    return shift_inplace(t, int(k))


def diff(t: TSeries, k: int = -1) -> TSeries:
    """First (or ``k``-th) difference of ``t``.

    Defined as ``t - shift(t, k)``. With the default ``k=-1`` (i.e. subtract
    the lag), this matches the standard first-difference operator. Positive
    ``k`` subtracts a lead; negative ``k`` subtracts a lag.

    The result is a new TSeries whose range is the intersection of ``t.range``
    and ``shift(t, k).range`` — one period shorter than ``t``.
    """
    return _as_tseries(t - shift(t, int(k)), reference=t)


def pct(t: TSeries, shift_value: int = -1, *, islog: bool = False) -> TSeries:
    """Observation-to-observation percent rate of change.

    When ``islog`` is ``True``, ``t`` is interpreted as a log-series — values
    are exponentiated before differencing.
    """
    if islog:
        exp_t = TSeries(t.firstdate, np.exp(t.values))
        a = exp_t
        b = shift(exp_t, int(shift_value))
    else:
        a = t
        b = shift(t, int(shift_value))
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


# Keep import alive so static checkers see it as used (MITRange is used by
# tests directly and may show up here in future fconvert work).
_ = MITRange
