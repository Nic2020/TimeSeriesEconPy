# SPDX-License-Identifier: MIT
"""Shared helpers for the pandas / polars interop adapters.

These are intentionally library-agnostic: they convert between tsecon's
:class:`~tsecon.frequencies.Frequency` and the strings / dates used by both
DataFrame backends. Neither pandas nor polars is imported here.
"""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING

from tsecon.frequencies import (
    BDaily,
    Daily,
    Frequency,
    HalfYearly,
    Monthly,
    Quarterly,
    Unit,
    Weekly,
    Yearly,
)
from tsecon.mit import MIT, bdaily, daily, mit_to_date, weekly
from tsecon.mitrange import MITRange

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "freq_from_pandas_freqstr",
    "freq_to_pandas_freqstr",
    "mit_from_date",
    "mitrange_to_dates",
    "mits_to_dates",
    "supports_pandas_period",
]


_MONTH_ABBRS = (
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
)

# pandas Weekly period anchors: end-of-week day. ISO day numbering aligns with
# the tsecon Weekly(end_day=N) convention (1 = Monday, ..., 7 = Sunday), so the
# index here is 1-based.
_DAY_ABBRS = ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")


def supports_pandas_period(freq: Frequency) -> bool:
    """Return True when ``freq`` has a pandas ``PeriodIndex`` analogue.

    pandas supports periodic indexes for Yearly, Quarterly, Monthly, and
    Weekly (with arbitrary anchor). It has no built-in half-yearly period;
    Daily / BDaily are best represented as ``DatetimeIndex`` instead, since
    pandas treats those as regular date ranges rather than periods.
    """
    return isinstance(freq, (Yearly, Quarterly, Monthly, Weekly))


def freq_to_pandas_freqstr(freq: Frequency) -> str:
    """Map a tsecon :class:`Frequency` to a pandas frequency-alias string.

    Used to construct ``pd.PeriodIndex`` instances and (for Daily / BDaily)
    ``pd.date_range`` calls. Raises :class:`TypeError` for frequencies
    pandas cannot represent (currently :class:`HalfYearly` and :class:`Unit`).

    Notes
    -----
    For Quarterly, pandas's ``Q-X`` anchor is the month in which Q4 ends
    (the fiscal year's last month), whereas tsecon's ``end_month`` is the
    month in which Q1 ends. Q4 ends 9 months after Q1, so the pandas
    anchor is ``((end_month + 8) mod 12) + 1``. For ``Quarterly(end_month=3)``
    (calendar quarters) this yields ``"Q-DEC"`` — pandas labels Jan-Mar 2020
    as ``"2020Q1"`` under that anchor, matching tsecon's ``qq(2020, 1)``.

    Note that tsecon's Quarterly restricts ``end_month`` to ``{1, 2, 3}``
    (the three calendar-aligned phases), so the only pandas anchors emitted
    here are ``Q-OCT``, ``Q-NOV``, and ``Q-DEC``. Fiscal-year conventions
    that anchor elsewhere (``Q-MAR``, ``Q-JUN``, ``Q-SEP``) are not
    representable in tsecon and raise on parse.

    For Yearly / Monthly / Weekly the conventions coincide directly.
    """
    if isinstance(freq, Yearly):
        return f"Y-{_MONTH_ABBRS[freq.end_month - 1]}"
    if isinstance(freq, Quarterly):
        anchor = ((freq.end_month + 8) % 12) + 1
        return f"Q-{_MONTH_ABBRS[anchor - 1]}"
    if isinstance(freq, Monthly):
        return "M"
    if isinstance(freq, Weekly):
        return f"W-{_DAY_ABBRS[freq.end_day - 1]}"
    if isinstance(freq, Daily):
        return "D"
    if isinstance(freq, BDaily):
        return "B"
    msg = f"No pandas frequency alias for {type(freq).__name__}."
    raise TypeError(msg)


def freq_from_pandas_freqstr(freqstr: str) -> Frequency:
    """Inverse of :func:`freq_to_pandas_freqstr`.

    Accepts both pandas 2.x aliases (``Y-DEC``, ``Q-DEC``, ``M``) and the
    pandas 3.x ``E``/``S`` (period-end / period-start) suffixed variants
    (``YE-DEC``, ``QE-DEC``, ``ME``, ``YS``, ``QS-DEC``). The legacy ``A-``
    annual alias is also accepted. Raises :class:`ValueError` if the string
    is not recognised.
    """
    s = freqstr.upper()
    head, _, tail = s.partition("-")
    # Strip the optional E (end-of-period) / S (start-of-period) suffix that
    # pandas 3.x adds to single-letter frequency codes. We always work in
    # end-of-period semantics, so the suffix is informational only.
    if len(head) >= 2 and head[-1] in ("E", "S") and head[:-1] in ("Y", "A", "Q", "M", "W"):
        head = head[:-1]
    if head in ("Y", "A"):
        em = _month_from_abbr(tail) if tail else 12
        return Yearly(end_month=em)
    if head == "Q":
        anchor = _month_from_abbr(tail) if tail else 12
        # Inverse of (em + 8) mod 12 + 1: em = ((anchor - 10) mod 12) + 1.
        em = ((anchor - 10) % 12) + 1
        return Quarterly(end_month=em)
    if head == "M":
        return Monthly()
    if head == "W":
        ed = _day_from_abbr(tail) if tail else 7
        return Weekly(end_day=ed)
    if head == "D":
        return Daily()
    if head == "B":
        return BDaily()
    msg = f"Unrecognised pandas frequency alias: {freqstr!r}."
    raise ValueError(msg)


def _month_from_abbr(abbr: str) -> int:
    try:
        return _MONTH_ABBRS.index(abbr) + 1
    except ValueError as e:
        msg = f"Unknown month abbreviation in pandas freq: {abbr!r}."
        raise ValueError(msg) from e


def _day_from_abbr(abbr: str) -> int:
    try:
        return _DAY_ABBRS.index(abbr) + 1
    except ValueError as e:
        msg = f"Unknown day-of-week abbreviation in pandas freq: {abbr!r}."
        raise ValueError(msg) from e


def mits_to_dates(mits: Iterable[MIT], *, ref: str = "end") -> list[_dt.date]:
    """Convert an iterable of MITs to a list of period-end (or -begin) dates."""
    return [mit_to_date(m, ref=ref) for m in mits]


def mitrange_to_dates(rng: MITRange, *, ref: str = "end") -> list[_dt.date]:
    """Convert every MIT in ``rng`` to a date via :func:`mit_to_date`."""
    return [mit_to_date(m, ref=ref) for m in rng]


def mit_from_date(d: _dt.date, freq: Frequency) -> MIT:
    """Recover an MIT from a date, given the target frequency.

    Inverse of :func:`mit_to_date` (with ``ref="end"``) for calendar
    frequencies. For YP frequencies, the date is mapped to the *period that
    contains it* — the same behaviour as ``daily(d)`` followed by an
    fconvert from Daily, but computed directly without temporary TSeries.
    """
    if isinstance(freq, Daily):
        return daily(d)
    if isinstance(freq, BDaily):
        return bdaily(d, bias="nearest")
    if isinstance(freq, Weekly):
        return weekly(d, end_day=freq.end_day)
    if isinstance(freq, Monthly):
        return MIT.from_yp(freq, d.year, d.month)
    if isinstance(freq, (Quarterly, HalfYearly, Yearly)):
        # YP frequencies: scan candidate periods by trying mit_to_date on a
        # small window around the expected year. Robust to arbitrary
        # ``end_month`` anchors at year boundaries without per-frequency
        # arithmetic.
        return _yp_period_containing(d, freq)
    if isinstance(freq, Unit):
        msg = "Cannot recover a Unit-frequency MIT from a date; pass an int instead."
        raise TypeError(msg)
    msg = f"Cannot recover an MIT from a date for frequency {type(freq).__name__}."
    raise TypeError(msg)


def _yp_period_containing(d: _dt.date, freq: Frequency) -> MIT:
    """Return the MIT for the YP period of ``freq`` that contains date ``d``.

    Linear search across a 3-period window centred on the date's calendar
    year ± 1 — robust to arbitrary ``end_month`` anchors at year boundaries
    without per-frequency arithmetic.
    """
    if not isinstance(freq, (Yearly, HalfYearly, Quarterly)):
        msg = f"_yp_period_containing only supports YP frequencies, got {type(freq).__name__}."
        raise TypeError(msg)
    ppy = freq.periods_per_year
    # Try (year-1) periods, (year) periods, (year+1) periods. For each MIT,
    # check whether d falls in [begin, end].
    for y_off in (-1, 0, 1):
        y = d.year + y_off
        for p in range(1, ppy + 1):
            m = MIT.from_yp(freq, y, p)
            begin = mit_to_date(m, ref="begin")
            end = mit_to_date(m, ref="end")
            if begin <= d <= end:
                return m
    msg = f"No {type(freq).__name__} period contains {d.isoformat()}; this should be unreachable."
    raise RuntimeError(msg)
