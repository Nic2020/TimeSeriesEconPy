# SPDX-License-Identifier: MIT
"""MIT (moment-in-time) and Duration types.

Mirrors ``TimeSeriesEcon.jl``'s ``momentintime.jl``. An :class:`MIT` carries a
frequency and an integer value; :class:`Duration` is the same shape but
represents a *distance* between two MITs of the same frequency.

* ``MIT - MIT`` → ``Duration`` (same frequency required).
* ``MIT - Duration`` → ``MIT``.
* ``MIT + Duration`` → ``MIT``.
* ``MIT + MIT`` raises (semantically meaningless).
* Adding/subtracting a plain ``int`` is allowed and interpreted as a duration
  in the MIT's own frequency.

Constructor functions :func:`qq`, :func:`mm`, :func:`yy`, :func:`daily`,
:func:`bdaily`, :func:`weekly` are the Pythonic equivalent of the Julia
literal-suffix syntax ``2020Q1``, ``2020M3``, ``2020Y``, etc.

Equality with plain integers is **not** supported (unlike Julia, where
``MIT(Quarterly, 5) == 5`` is true). The Julia idiom violates Python's
``__hash__``/``__eq__`` invariant; use ``int(mit)`` or ``mit.value`` to
extract the underlying integer instead.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Union, overload

from tsecon._options import getoption
from tsecon.frequencies import (
    BDaily,
    CalendarFrequency,
    Daily,
    Frequency,
    HalfYearly,
    Monthly,
    Quarterly,
    Unit,
    Weekly,
    Yearly,
    YPFrequency,
    prettyprint_frequency,
)

if TYPE_CHECKING:
    from tsecon.mitrange import MITRange

__all__ = [
    "MIT",
    "BDailyBias",
    "Duration",
    "bdaily",
    "daily",
    "frequency_of",
    "mit2yp",
    "mm",
    "period",
    "qq",
    "weekly",
    "weekly_from_iso",
    "year",
    "yy",
]


BDailyBias = Union["str"]
"""One of ``"strict"``, ``"previous"``, ``"next"``, ``"nearest"``.

Determines how :func:`bdaily` resolves a date that falls on a weekend.
``"strict"`` (the default, matching Julia) raises rather than guessing.
"""

_VALID_BDAILY_BIASES: Final[frozenset[str]] = frozenset({"strict", "previous", "next", "nearest"})


def _mixed_freq_error(left: object, right: object) -> TypeError:
    lf = _freq_name(left)
    rf = _freq_name(right)
    return TypeError(f"Mixing frequencies not allowed: {lf} and {rf}.")


def _freq_name(x: object) -> str:
    if isinstance(x, (MIT, Duration)):
        return prettyprint_frequency(x.frequency)
    return type(x).__name__


# ---------------------------------------------------------------------------
# MIT
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MIT:
    """A frequency-tagged moment in time.

    Construct via :meth:`from_yp` for year/period frequencies, or via the
    convenience constructors :func:`qq`, :func:`mm`, :func:`yy`, :func:`daily`,
    :func:`bdaily`, :func:`weekly`. Direct ``MIT(frequency, value)``
    construction takes a raw integer offset.
    """

    frequency: Frequency
    value: int

    def __post_init__(self) -> None:
        if not isinstance(self.frequency, Frequency):
            msg = f"frequency must be a Frequency instance, got {type(self.frequency).__name__}"  # type: ignore[unreachable]
            raise TypeError(msg)
        if not isinstance(self.value, int) or isinstance(self.value, bool):
            msg = f"value must be int, got {type(self.value).__name__}"
            raise TypeError(msg)

    # -- alternate constructors --------------------------------------------

    @classmethod
    def from_yp(cls, frequency: YPFrequency, year_: int, period_: int) -> MIT:
        """Construct an MIT from year+period (only for YP frequencies)."""
        if not isinstance(frequency, YPFrequency):
            msg = f"from_yp requires a YPFrequency, got {type(frequency).__name__}"  # type: ignore[unreachable]
            raise TypeError(msg)
        n = frequency.periods_per_year
        if not 1 <= period_ <= n:
            msg = f"period must be 1..{n} for {type(frequency).__name__}, got {period_}"
            raise ValueError(msg)
        return cls(frequency, n * int(year_) + int(period_) - 1)

    # -- conversion / introspection ----------------------------------------

    def __int__(self) -> int:
        return self.value

    def __index__(self) -> int:
        return self.value

    def __float__(self) -> float:
        f = self.frequency
        if isinstance(f, YPFrequency):
            y, p = mit2yp(self)
            return y + (p - 1) / f.periods_per_year
        return float(self.value)

    # -- equality / hash ---------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MIT):
            return self.frequency == other.frequency and self.value == other.value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("MIT", self.frequency, self.value))

    # -- arithmetic --------------------------------------------------------

    def __add__(self, other: object) -> MIT | float:
        if isinstance(other, MIT):
            raise TypeError("Illegal addition of two MIT values.")
        if isinstance(other, Duration):
            if self.frequency != other.frequency:
                raise _mixed_freq_error(self, other)
            return MIT(self.frequency, self.value + other.value)
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int):
            return MIT(self.frequency, self.value + other)
        if isinstance(other, float):
            return float(self) + other
        return NotImplemented

    def __radd__(self, other: object) -> MIT | float:
        return self.__add__(other)

    def __sub__(self, other: object) -> MIT | Duration | float:
        if isinstance(other, MIT):
            if self.frequency != other.frequency:
                raise _mixed_freq_error(self, other)
            return Duration(self.frequency, self.value - other.value)
        if isinstance(other, Duration):
            if self.frequency != other.frequency:
                raise _mixed_freq_error(self, other)
            return MIT(self.frequency, self.value - other.value)
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int):
            return MIT(self.frequency, self.value - other)
        if isinstance(other, float):
            return float(self) - other
        return NotImplemented

    # int - MIT is illegal (matches Julia)
    def __rsub__(self, other: object) -> Duration | float:
        if isinstance(other, float):
            return other - float(self)
        if isinstance(other, (int, bool)):
            raise TypeError("Cannot subtract MIT from int; use MIT - MIT instead.")
        return NotImplemented

    # -- ordering ----------------------------------------------------------

    def __lt__(self, other: object) -> bool:
        if isinstance(other, MIT):
            if self.frequency != other.frequency:
                raise _mixed_freq_error(self, other)
            return self.value < other.value
        if isinstance(other, Duration):
            raise TypeError(
                f"Illegal comparison of {type(self).__name__} and {type(other).__name__}."
            )
        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, MIT):
            if self.frequency != other.frequency:
                raise _mixed_freq_error(self, other)
            return self.value <= other.value
        if isinstance(other, Duration):
            raise TypeError(
                f"Illegal comparison of {type(self).__name__} and {type(other).__name__}."
            )
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, MIT):
            if self.frequency != other.frequency:
                raise _mixed_freq_error(self, other)
            return self.value > other.value
        if isinstance(other, Duration):
            raise TypeError(
                f"Illegal comparison of {type(self).__name__} and {type(other).__name__}."
            )
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        if isinstance(other, MIT):
            if self.frequency != other.frequency:
                raise _mixed_freq_error(self, other)
            return self.value >= other.value
        if isinstance(other, Duration):
            raise TypeError(
                f"Illegal comparison of {type(self).__name__} and {type(other).__name__}."
            )
        return NotImplemented

    # -- range syntax ------------------------------------------------------

    def to(self, stop: MIT) -> MITRange:
        """Return the ``MITRange`` from ``self`` through ``stop`` (inclusive).

        Mirrors Julia's ``start:stop`` syntax for MITs. Python's slice/colon
        cannot be overloaded outside of indexing, so this method is the
        idiomatic spelling. The free function :func:`tsecon.mitrange.mitrange`
        also works.
        """
        from tsecon.mitrange import MITRange  # noqa: PLC0415  (avoid circular import)

        return MITRange(self, stop)

    # -- repr / str --------------------------------------------------------

    def __repr__(self) -> str:
        return _format_mit(self)

    def __str__(self) -> str:
        return _format_mit(self)


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Duration:
    """A frequency-tagged distance between two MITs."""

    frequency: Frequency
    value: int

    def __post_init__(self) -> None:
        if not isinstance(self.frequency, Frequency):
            msg = f"frequency must be a Frequency instance, got {type(self.frequency).__name__}"  # type: ignore[unreachable]
            raise TypeError(msg)
        if not isinstance(self.value, int) or isinstance(self.value, bool):
            msg = f"value must be int, got {type(self.value).__name__}"
            raise TypeError(msg)

    # -- conversion --------------------------------------------------------

    def __int__(self) -> int:
        return self.value

    def __index__(self) -> int:
        return self.value

    def __float__(self) -> float:
        f = self.frequency
        if isinstance(f, YPFrequency):
            return self.value / f.periods_per_year
        return float(self.value)

    def __bool__(self) -> bool:
        return self.value != 0

    # -- equality ----------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Duration):
            return self.frequency == other.frequency and self.value == other.value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(("Duration", self.frequency, self.value))

    # -- unary -------------------------------------------------------------

    def __neg__(self) -> Duration:
        return Duration(self.frequency, -self.value)

    def __pos__(self) -> Duration:
        return self

    def __abs__(self) -> Duration:
        return Duration(self.frequency, abs(self.value))

    # -- arithmetic --------------------------------------------------------

    def __add__(self, other: object) -> Duration:
        if isinstance(other, MIT):
            raise TypeError("Illegal addition of Duration and MIT. Try MIT + Duration.")
        if isinstance(other, Duration):
            if self.frequency != other.frequency:
                raise _mixed_freq_error(self, other)
            return Duration(self.frequency, self.value + other.value)
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int):
            return Duration(self.frequency, self.value + other)
        return NotImplemented

    def __radd__(self, other: object) -> Duration:
        return self.__add__(other)

    def __sub__(self, other: object) -> Duration:
        if isinstance(other, MIT):
            raise TypeError("Cannot subtract MIT from Duration.")
        if isinstance(other, Duration):
            if self.frequency != other.frequency:
                raise _mixed_freq_error(self, other)
            return Duration(self.frequency, self.value - other.value)
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int):
            return Duration(self.frequency, self.value - other)
        return NotImplemented

    def __rsub__(self, other: object) -> Duration:
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int):
            return Duration(self.frequency, other - self.value)
        return NotImplemented

    def __mul__(self, other: object) -> Duration:
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int):
            return Duration(self.frequency, self.value * other)
        return NotImplemented

    def __rmul__(self, other: object) -> Duration:
        return self.__mul__(other)

    def __floordiv__(self, other: object) -> Duration:
        if isinstance(other, Duration):
            if self.frequency != other.frequency:
                raise _mixed_freq_error(self, other)
            return Duration(self.frequency, self.value // other.value)
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int):
            return Duration(self.frequency, self.value // other)
        return NotImplemented

    def __mod__(self, other: object) -> Duration:
        if isinstance(other, Duration):
            if self.frequency != other.frequency:
                raise _mixed_freq_error(self, other)
            return Duration(self.frequency, self.value % other.value)
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int):
            return Duration(self.frequency, self.value % other)
        return NotImplemented

    # -- comparison --------------------------------------------------------

    def __lt__(self, other: object) -> bool:
        if isinstance(other, Duration):
            if self.frequency != other.frequency:
                raise _mixed_freq_error(self, other)
            return self.value < other.value
        if isinstance(other, MIT):
            raise TypeError(
                f"Illegal comparison of {type(self).__name__} and {type(other).__name__}."
            )
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int):
            return self.value < other
        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, Duration):
            if self.frequency != other.frequency:
                raise _mixed_freq_error(self, other)
            return self.value <= other.value
        if isinstance(other, MIT):
            raise TypeError(
                f"Illegal comparison of {type(self).__name__} and {type(other).__name__}."
            )
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int):
            return self.value <= other
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, Duration):
            if self.frequency != other.frequency:
                raise _mixed_freq_error(self, other)
            return self.value > other.value
        if isinstance(other, MIT):
            raise TypeError(
                f"Illegal comparison of {type(self).__name__} and {type(other).__name__}."
            )
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int):
            return self.value > other
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        if isinstance(other, Duration):
            if self.frequency != other.frequency:
                raise _mixed_freq_error(self, other)
            return self.value >= other.value
        if isinstance(other, MIT):
            raise TypeError(
                f"Illegal comparison of {type(self).__name__} and {type(other).__name__}."
            )
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int):
            return self.value >= other
        return NotImplemented

    # -- repr / str --------------------------------------------------------

    def __repr__(self) -> str:
        return str(self.value)

    def __str__(self) -> str:
        return str(self.value)


# ---------------------------------------------------------------------------
# Year/period helpers
# ---------------------------------------------------------------------------


def mit2yp(x: MIT) -> tuple[int, int]:
    """Recover ``(year, period)`` from a YP- or Daily-/BDaily-frequency MIT.

    For ``YPFrequency`` subclasses this is the canonical decomposition
    ``int(mit) == N * year + (period - 1)``. For :class:`Daily` and
    :class:`BDaily` it returns the year and the 1-based index of the day
    within that year.
    """
    f = x.frequency
    if isinstance(f, YPFrequency):
        n = f.periods_per_year
        y, r = divmod(x.value, n)
        return (y, r + 1)
    if isinstance(f, Daily):
        date = _date_from_daily(x.value)
        return (date.year, date.timetuple().tm_yday)
    if isinstance(f, BDaily):
        date = _date_from_bdaily(x.value)
        y = date.year
        first = _dt.date(y, 1, 1)
        first_wd = first.isoweekday()
        days_diff = 8 - first_wd if first_wd > 5 else 0
        start = first + _dt.timedelta(days=days_diff)
        start_mit = bdaily(start).value
        return (y, x.value - start_mit + 1)
    msg = f"Value of type {type(f).__name__} cannot be represented as (year, period)."
    raise TypeError(msg)


def year(x: MIT) -> int:
    """Return the year component of a YP-/Daily-/BDaily-frequency MIT."""
    return mit2yp(x)[0]


def period(x: MIT) -> int:
    """Return the period component of a YP-/Daily-/BDaily-frequency MIT."""
    return mit2yp(x)[1]


def frequency_of(x: object) -> Frequency:
    """Return the frequency of an MIT, Duration, or other frequency-bearing value.

    Mirrors Julia's ``frequencyof``. Raises ``TypeError`` for values that
    don't carry a frequency.
    """
    if isinstance(x, (MIT, Duration)):
        return x.frequency
    if isinstance(x, Frequency):
        return x
    # MITRange: imported lazily to avoid circular import.
    from tsecon.mitrange import MITRange  # noqa: PLC0415

    if isinstance(x, MITRange):
        return x.frequency
    msg = f"{type(x).__name__} does not have a frequency."
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Convenience constructor functions (Pythonic equivalents of Julia 2020Q1 etc.)
# ---------------------------------------------------------------------------


def qq(year_: int, period_: int) -> MIT:
    """Construct an ``MIT`` at quarter ``period_`` of year ``year_`` (Q1..Q4)."""
    return MIT.from_yp(Quarterly(), year_, period_)


def mm(year_: int, period_: int) -> MIT:
    """Construct an ``MIT`` at month ``period_`` of year ``year_`` (1..12)."""
    return MIT.from_yp(Monthly(), year_, period_)


def yy(year_: int, period_: int = 1) -> MIT:
    """Construct an ``MIT`` at year ``year_``. ``period_`` must be 1 (default)."""
    return MIT.from_yp(Yearly(), year_, period_)


# ---------------------------------------------------------------------------
# Daily / BDaily / Weekly construction and Date conversion
# ---------------------------------------------------------------------------


# Julia uses _d0 = Date(1, 1, 1) - Day(1), so MIT{Daily}(v) corresponds to
# the date `_d0 + Day(v)`. In Python, `date.fromordinal(1) == date(1, 1, 1)`,
# so an MIT{Daily}(v) maps directly to `date.fromordinal(v)`. The two
# conventions agree.


def _parse_date(d: _dt.date | str) -> _dt.date:
    if isinstance(d, _dt.date) and not isinstance(d, _dt.datetime):
        return d
    if isinstance(d, _dt.datetime):
        return d.date()
    if isinstance(d, str):
        return _dt.date.fromisoformat(d)
    msg = f"Expected date or ISO date string, got {type(d).__name__}"  # type: ignore[unreachable]
    raise TypeError(msg)


@overload
def daily(d: _dt.date) -> MIT: ...
@overload
def daily(d: str) -> MIT: ...
def daily(d: _dt.date | str) -> MIT:
    """Construct an ``MIT{Daily}`` from a :class:`datetime.date` or ISO string."""
    date = _parse_date(d)
    return MIT(Daily(), date.toordinal())


def _date_from_daily(value: int) -> _dt.date:
    return _dt.date.fromordinal(value)


@overload
def bdaily(d: _dt.date, *, bias: str | None = ...) -> MIT: ...
@overload
def bdaily(d: str, *, bias: str | None = ...) -> MIT: ...
def bdaily(d: _dt.date | str, *, bias: str | None = None) -> MIT:
    """Construct an ``MIT{BDaily}`` from a date or ISO string.

    Business-daily MITs index Monday-Friday only. ``bias`` controls how a
    date that lands on a weekend is resolved:

    * ``"strict"`` — raise ``ValueError``.
    * ``"previous"`` — return the preceding Friday.
    * ``"next"`` — return the following Monday.
    * ``"nearest"`` — Saturday → Friday, Sunday → Monday.

    When ``bias`` is not supplied, the value falls back to the global option
    ``bdaily_creation_bias`` (default ``"strict"``). Mirrors Julia's
    ``bdaily(d::Date; bias::Symbol=getoption(:bdaily_creation_bias))``.
    """
    if bias is None:
        bias = getoption("bdaily_creation_bias")
    if bias not in _VALID_BDAILY_BIASES:
        msg = f"bias must be one of {sorted(_VALID_BDAILY_BIASES)}, got {bias!r}"
        raise ValueError(msg)
    date = _parse_date(d)
    ord_ = date.toordinal()
    num_weekends, rem = divmod(ord_, 7)
    # `rem` is in [0, 6]: 0 = Sunday, 1 = Monday, ..., 6 = Saturday (since
    # date.fromordinal(1) = 0001-01-01 is a Monday).
    adjustment = 0
    if rem == 0:  # Sunday
        if bias in ("next", "nearest"):
            adjustment = -1
        elif bias == "strict":
            msg = f"{date.isoformat()} is not a valid business day (Sunday)."
            raise ValueError(msg)
    elif rem == 6:  # Saturday
        if bias in ("previous", "nearest"):
            adjustment = 1
        elif bias == "strict":
            msg = f"{date.isoformat()} is not a valid business day (Saturday)."
            raise ValueError(msg)
    return MIT(BDaily(), ord_ - num_weekends * 2 - adjustment)


def _date_from_bdaily(value: int) -> _dt.date:
    return _dt.date.fromordinal(value + 2 * ((value - 1) // 5))


@overload
def weekly(d: _dt.date, end_day: int = ...) -> MIT: ...
@overload
def weekly(d: str, end_day: int = ...) -> MIT: ...
def weekly(d: _dt.date | str, end_day: int = 7) -> MIT:
    """Construct an ``MIT{Weekly(end_day)}`` from a date or ISO string.

    The resulting MIT identifies the week (ending on the given ``end_day``)
    that contains the provided date.
    """
    date = _parse_date(d)
    base = -(-date.toordinal() // 7)  # ceil(d / 7)
    bump = max(0, min(1, date.isoweekday() - end_day))
    return MIT(Weekly(end_day), base + bump)


def weekly_from_iso(year_: int, week_: int) -> MIT:
    """Construct an ``MIT{Weekly(7)}`` from an ISO year + week number (1..53)."""
    if not 1 <= week_ <= 53:
        msg = f"week must be between 1 and 53 (inclusive). Received: {week_}"
        raise ValueError(msg)
    first = _dt.date(year_, 1, 1)
    week_of_first = first.isocalendar().week
    padding = 1 if week_of_first != 1 else 0
    candidate_date = first + _dt.timedelta(days=((week_ - 1) + padding) * 7)
    mit = weekly(candidate_date)
    derived = _date_from_weekly(mit.value, end_day=7)
    if derived.year != year_ and first.isocalendar().week < 52:
        msg = f"The year {year_} does not have a week {week_}."
        raise ValueError(msg)
    return mit


def _date_from_weekly(value: int, *, end_day: int) -> _dt.date:
    return _dt.date.fromordinal(value * 7 - (7 - end_day))


# ---------------------------------------------------------------------------
# MIT -> date conversion
# ---------------------------------------------------------------------------


def _add_months(d: _dt.date, months: int) -> _dt.date:
    total = d.year * 12 + (d.month - 1) + months
    y, m = divmod(total, 12)
    return _dt.date(y, m + 1, min(d.day, _days_in_month(y, m + 1)))


def _days_in_month(year_: int, month_: int) -> int:
    if month_ == 12:
        return 31
    return (_dt.date(year_, month_ + 1, 1) - _dt.timedelta(days=1)).day


def mit_to_date(m: MIT, *, ref: str = "end") -> _dt.date:
    """Convert an MIT to a :class:`datetime.date`.

    For multi-day frequencies (Yearly, HalfYearly, Quarterly, Monthly,
    Weekly) ``ref`` controls whether the *end* of the period (default) or
    its *beginning* (``ref="begin"``) is returned.
    """
    if ref not in ("begin", "end"):
        msg = f"ref must be 'begin' or 'end', got {ref!r}"
        raise ValueError(msg)
    f = m.frequency
    v = m.value
    if isinstance(f, Daily):
        return _date_from_daily(v)
    if isinstance(f, BDaily):
        return _date_from_bdaily(v)
    if isinstance(f, Weekly):
        if ref == "begin":
            return _dt.date.fromordinal(v * 7 - 6 - (7 - f.end_day))
        return _date_from_weekly(v, end_day=f.end_day)
    if isinstance(f, Monthly):
        y, mo = divmod(v, 12)
        if ref == "begin":
            return _add_months(_dt.date(y, 1, 1), mo)
        return _add_months(_dt.date(y, 1, 1), mo + 1) - _dt.timedelta(days=1)
    if isinstance(f, Quarterly):
        em = f.end_month
        y, q = divmod(v, 4)
        if ref == "begin":
            return _add_months(_dt.date(y, 1, 1), q * 3 - (3 - em))
        return _add_months(_dt.date(y, 1, 1), (q + 1) * 3 - (3 - em)) - _dt.timedelta(days=1)
    if isinstance(f, HalfYearly):
        em = f.end_month
        y, h = divmod(v, 2)
        if ref == "begin":
            return _add_months(_dt.date(y, 1, 1), h * 6 - (6 - em))
        return _add_months(_dt.date(y, 1, 1), (h + 1) * 6 - (6 - em)) - _dt.timedelta(days=1)
    if isinstance(f, Yearly):
        em = f.end_month
        if ref == "begin":
            return _add_months(_dt.date(v, 1, 1), -(12 - em))
        return _add_months(_dt.date(v + 1, 1, 1), -(12 - em)) - _dt.timedelta(days=1)
    msg = f"Cannot convert MIT of frequency {type(f).__name__} to a date."
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


_YP_LETTER = {
    Yearly: "Y",
    HalfYearly: "H",
    Quarterly: "Q",
    Monthly: "M",
}


def _format_mit(m: MIT) -> str:
    f = m.frequency
    if isinstance(f, Unit):
        return f"{m.value}U"
    if isinstance(f, (Daily, BDaily)):
        return mit_to_date(m).isoformat()
    if isinstance(f, Weekly):
        return mit_to_date(m).isoformat()
    if isinstance(f, YPFrequency):
        y, p = mit2yp(m)
        letter = _YP_LETTER.get(type(f), "P")
        out = f"{y}{letter}"
        if f.periods_per_year > 1:
            out += str(p)
        # append non-default end_period in braces, matching Julia
        if (
            (isinstance(f, Yearly) and f.end_month != 12)
            or (isinstance(f, HalfYearly) and f.end_month != 6)
            or (isinstance(f, Quarterly) and f.end_month != 3)
        ):
            out += f"{{{f.end_month}}}"
        return out
    return f"MIT({prettyprint_frequency(f)}, {m.value})"


# Public alias for callers who prefer attribute-style import.
_ = (CalendarFrequency,)  # keep import alive (subclass check helpers)
