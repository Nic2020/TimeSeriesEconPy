# SPDX-License-Identifier: MIT
"""Frequency types — calendar and unit frequencies for time-series MITs.

This module mirrors the ``Frequency`` hierarchy from ``TimeSeriesEcon.jl``'s
``momentintime.jl``. The Julia source uses parametric types (e.g.
``Yearly{end_month}``); Python has no parametric primitive types, so we use
frozen, cached-singleton dataclasses. Two calls with the same arguments return
the *same* object, so both ``==`` and ``is`` comparisons work:

>>> Yearly() is Yearly(end_month=12)
True
>>> Quarterly(end_month=3) is Quarterly()
True

The hierarchy:

* :class:`Frequency` — abstract supertype.
* :class:`Unit` — dimensionless (no calendar).
* :class:`CalendarFrequency` — abstract; all calendar-aware frequencies.
* :class:`YPFrequency` — abstract; calendar frequencies defined by a fixed
  number of periods per year (Yearly / HalfYearly / Quarterly / Monthly).
* :class:`Daily`, :class:`BDaily`, :class:`Weekly` — non-YP calendar frequencies.

See ``claude_files/decisions/15_frequency_model.md`` for the rationale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final, TypeAlias

__all__ = [
    "BDaily",
    "CalendarFrequency",
    "Daily",
    "Frequency",
    "FrequencyLike",
    "HalfYearly",
    "Monthly",
    "Quarterly",
    "Unit",
    "Weekly",
    "YPFrequency",
    "Yearly",
    "endperiod",
    "is_bdaily",
    "is_daily",
    "is_halfyearly",
    "is_monthly",
    "is_quarterly",
    "is_weekly",
    "is_yearly",
    "ppy",
    "prettyprint_frequency",
    "sanitize_frequency",
]


# ---------------------------------------------------------------------------
# Base hierarchy
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Frequency:
    """Abstract supertype for all frequencies.

    Concrete subclasses use cached singletons via ``__new__``; do not
    instantiate :class:`Frequency` directly.
    """


@dataclass(frozen=True, slots=True)
class CalendarFrequency(Frequency):
    """Abstract supertype for calendar-aware frequencies."""


@dataclass(frozen=True, slots=True)
class YPFrequency(CalendarFrequency):
    """Abstract supertype for fixed-periods-per-year calendar frequencies.

    Subclasses provide a class-level :attr:`periods_per_year` integer N. An
    ``MIT{F}`` value for ``F <: YPFrequency`` decomposes into a (year, period)
    pair via ``int(mit) = N * year + (period - 1)``.
    """

    periods_per_year: ClassVar[int]


# ---------------------------------------------------------------------------
# Unit
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Unit(Frequency):
    """Non-dimensional frequency (no calendar association)."""

    _instance: ClassVar[Unit | None] = None

    def __new__(cls) -> Unit:
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "Unit()"


# ---------------------------------------------------------------------------
# YP frequencies
# ---------------------------------------------------------------------------


def _validate_end_month(name: str, end_month: int, lo: int, hi: int) -> None:
    if not isinstance(end_month, int) or isinstance(end_month, bool):
        msg = f"The end_month for a {name} frequency must be an integer. Received: {end_month!r}"
        raise TypeError(msg)
    if not lo <= end_month <= hi:
        msg = (
            f"The end_month for a {name} frequency must be between {lo} and {hi}. "
            f"Received: {end_month}"
        )
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Yearly(YPFrequency):
    """Yearly frequency: 1 period per year. Default end-month is December (12)."""

    end_month: int = 12

    periods_per_year: ClassVar[int] = 1
    _cache: ClassVar[dict[int, Yearly]] = {}

    def __new__(cls, end_month: int = 12) -> Yearly:
        _validate_end_month("Yearly", end_month, 1, 12)
        cached = cls._cache.get(end_month)
        if cached is not None:
            return cached
        obj = object.__new__(cls)
        cls._cache[end_month] = obj
        return obj

    def __repr__(self) -> str:
        return "Yearly()" if self.end_month == 12 else f"Yearly(end_month={self.end_month})"


@dataclass(frozen=True, slots=True)
class HalfYearly(YPFrequency):
    """Half-yearly frequency: 2 periods per year. Default end-month is June (6)."""

    end_month: int = 6

    periods_per_year: ClassVar[int] = 2
    _cache: ClassVar[dict[int, HalfYearly]] = {}

    def __new__(cls, end_month: int = 6) -> HalfYearly:
        _validate_end_month("HalfYearly", end_month, 1, 6)
        cached = cls._cache.get(end_month)
        if cached is not None:
            return cached
        obj = object.__new__(cls)
        cls._cache[end_month] = obj
        return obj

    def __repr__(self) -> str:
        return "HalfYearly()" if self.end_month == 6 else f"HalfYearly(end_month={self.end_month})"


@dataclass(frozen=True, slots=True)
class Quarterly(YPFrequency):
    """Quarterly frequency: 4 periods per year. Default end-month is March (3)."""

    end_month: int = 3

    periods_per_year: ClassVar[int] = 4
    _cache: ClassVar[dict[int, Quarterly]] = {}

    def __new__(cls, end_month: int = 3) -> Quarterly:
        _validate_end_month("Quarterly", end_month, 1, 3)
        cached = cls._cache.get(end_month)
        if cached is not None:
            return cached
        obj = object.__new__(cls)
        cls._cache[end_month] = obj
        return obj

    def __repr__(self) -> str:
        return "Quarterly()" if self.end_month == 3 else f"Quarterly(end_month={self.end_month})"


@dataclass(frozen=True, slots=True)
class Monthly(YPFrequency):
    """Monthly frequency: 12 periods per year."""

    periods_per_year: ClassVar[int] = 12
    _instance: ClassVar[Monthly | None] = None

    def __new__(cls) -> Monthly:
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "Monthly()"


# ---------------------------------------------------------------------------
# Non-YP calendar frequencies
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Daily(CalendarFrequency):
    """Daily frequency (every calendar day)."""

    _instance: ClassVar[Daily | None] = None

    def __new__(cls) -> Daily:
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "Daily()"


@dataclass(frozen=True, slots=True)
class BDaily(CalendarFrequency):
    """Business-daily frequency (Monday-Friday, no weekends)."""

    _instance: ClassVar[BDaily | None] = None

    def __new__(cls) -> BDaily:
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "BDaily()"


@dataclass(frozen=True, slots=True)
class Weekly(CalendarFrequency):
    """Weekly frequency. ``end_day`` is the ISO weekday the week ends on (Mon=1..Sun=7).

    Default is ``end_day=7`` (Sunday).
    """

    end_day: int = 7

    _cache: ClassVar[dict[int, Weekly]] = {}

    def __new__(cls, end_day: int = 7) -> Weekly:
        if not isinstance(end_day, int) or isinstance(end_day, bool):
            msg = f"The end_day for a Weekly frequency must be an integer. Received: {end_day!r}"
            raise TypeError(msg)
        if not 1 <= end_day <= 7:
            msg = f"The end_day for a Weekly frequency must be between 1 and 7. Received: {end_day}"
            raise ValueError(msg)
        cached = cls._cache.get(end_day)
        if cached is not None:
            return cached
        obj = object.__new__(cls)
        cls._cache[end_day] = obj
        return obj

    def __repr__(self) -> str:
        return "Weekly()" if self.end_day == 7 else f"Weekly(end_day={self.end_day})"


# ---------------------------------------------------------------------------
# Type aliases and helpers
# ---------------------------------------------------------------------------


FrequencyLike: TypeAlias = Frequency | type[Frequency]
"""Either a frequency instance or the class of one.

Some helpers (``ppy``, ``sanitize_frequency``, ``is_yearly``) accept either form.
Mirrors Julia's ability to call helpers on both values and types.
"""


_PPY_DEFAULTS: Final[dict[type[Frequency], int]] = {
    Daily: 365,
    BDaily: 260,
    Weekly: 52,
}


def ppy(x: FrequencyLike) -> int:
    """Return the (approximate) periods-per-year for the given frequency.

    Exact for ``YPFrequency`` subclasses. Approximate for ``Daily`` (365),
    ``BDaily`` (260), and ``Weekly`` (52). Raises ``ValueError`` for ``Unit``.
    """
    cls = x if isinstance(x, type) else type(x)
    if issubclass(cls, YPFrequency):
        return cls.periods_per_year
    default = _PPY_DEFAULTS.get(cls)
    if default is not None:
        return default
    msg = f"Frequency {cls.__name__} does not have periods per year"
    raise ValueError(msg)


def endperiod(x: FrequencyLike) -> int:
    """Return the end-of-period offset for a frequency.

    For ``Yearly`` / ``HalfYearly`` / ``Quarterly`` returns ``end_month``; for
    ``Weekly`` returns ``end_day``; for ``Monthly`` / ``Daily`` / ``BDaily`` /
    ``Unit`` returns 1.
    """
    if isinstance(x, type):
        if issubclass(x, (Yearly, HalfYearly, Quarterly, Weekly)):
            return endperiod(x())
        return 1
    if isinstance(x, (Yearly, HalfYearly, Quarterly)):
        return x.end_month
    if isinstance(x, Weekly):
        return x.end_day
    return 1


def sanitize_frequency(x: FrequencyLike) -> Frequency:
    """Return a concrete singleton frequency instance.

    If ``x`` is already an instance, return it. If ``x`` is a frequency *class*
    with no required arguments (or all defaults), return its default singleton.
    """
    if isinstance(x, Frequency):
        return x
    if isinstance(x, type) and issubclass(x, Frequency):
        return x()
    msg = f"Cannot sanitize {x!r} into a Frequency"  # type: ignore[unreachable]
    raise TypeError(msg)


def prettyprint_frequency(x: FrequencyLike) -> str:
    """Return a short display name for a frequency.

    Default-parameter frequencies print without their parameter
    (``Quarterly`` rather than ``Quarterly(end_month=3)``); non-default ones
    include the parameter in braces, matching the Julia ``show`` convention.
    """
    f = sanitize_frequency(x)
    if isinstance(f, Yearly):
        return "Yearly" if f.end_month == 12 else f"Yearly{{{f.end_month}}}"
    if isinstance(f, HalfYearly):
        return "HalfYearly" if f.end_month == 6 else f"HalfYearly{{{f.end_month}}}"
    if isinstance(f, Quarterly):
        return "Quarterly" if f.end_month == 3 else f"Quarterly{{{f.end_month}}}"
    if isinstance(f, Weekly):
        return "Weekly" if f.end_day == 7 else f"Weekly{{{f.end_day}}}"
    return type(f).__name__


# ---------------------------------------------------------------------------
# is_* predicates
# ---------------------------------------------------------------------------


def _is_kind(x: object, cls: type[Frequency]) -> bool:
    if isinstance(x, cls):
        return True
    if isinstance(x, type):
        return issubclass(x, cls)
    return False


def is_yearly(x: object) -> bool:
    """Return True if ``x`` is (or is an instance of) :class:`Yearly`."""
    return _is_kind(x, Yearly)


def is_halfyearly(x: object) -> bool:
    """Return True if ``x`` is (or is an instance of) :class:`HalfYearly`."""
    return _is_kind(x, HalfYearly)


def is_quarterly(x: object) -> bool:
    """Return True if ``x`` is (or is an instance of) :class:`Quarterly`."""
    return _is_kind(x, Quarterly)


def is_monthly(x: object) -> bool:
    """Return True if ``x`` is (or is an instance of) :class:`Monthly`."""
    return _is_kind(x, Monthly)


def is_weekly(x: object) -> bool:
    """Return True if ``x`` is (or is an instance of) :class:`Weekly`."""
    return _is_kind(x, Weekly)


def is_daily(x: object) -> bool:
    """Return True if ``x`` is (or is an instance of) :class:`Daily`."""
    return _is_kind(x, Daily)


def is_bdaily(x: object) -> bool:
    """Return True if ``x`` is (or is an instance of) :class:`BDaily`."""
    return _is_kind(x, BDaily)
