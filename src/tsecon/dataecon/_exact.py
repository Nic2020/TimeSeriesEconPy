# SPDX-License-Identifier: MIT
"""Exact arithmetic behind Julia's marker-mapped scalar reconstructions.

Julia's DataEcon loader rebuilds ``Rational{T}``, ``Complex{T}`` (integer
``T``), ``Date`` and ``DateTime`` scalars from their Float64/Int64 payloads
with ``convert``, ``Base.rationalize`` and ``Dates.unix2datetime``, and stores
``Int128``/``UInt128``/``ComplexF16`` scalars at their own width. The functions
here reproduce those constructions in Python integers and IEEE floats so a
reader can state exactly what the pinned Julia would load, or that it would
raise, without evaluating any marker text: every parameter comes from the
finite tables below and every value is a plain Python integer, float or tuple.
Which Python objects such values become in the public API is a separate choice;
this module makes none and touches no native code; :mod:`._scalars` builds the
public ``StoredScalar`` on top of it.

Julia's own rules, transcribed from ``base/rational.jl``, ``base/float.jl``
and ``stdlib/Dates`` at the pinned version:

- ``Rational{T}(x::Float64)`` is ``rationalize(T, x, tol=0)`` — the
  continued-fraction convergents of ``x`` with checked ``T`` arithmetic, the
  last convergent before an overflow being the result — followed by
  ``x == Float64(result)`` (``InexactError`` otherwise). So ``1/3`` becomes
  ``6004799503160661//18014398509481984`` under ``Int64`` but ``1//3`` under
  ``Int32``, ``Inf`` becomes ``1//0``, ``NaN`` and a negative value under an
  unsigned parameter fail. An integer payload converts as ``T(n)//1``.
- ``Complex{T}`` takes ``T(real)`` and ``T(imag)``: each component must be
  finite, integral and within ``T`` (``typemin`` included), with no cap at
  ``2**53`` beyond what Float64 itself can hold.
- ``unix2datetime(x)`` is ``UNIXEPOCH + trunc(Int64, 1000 * x)`` in Rata
  Die milliseconds: the multiply is a float product in the payload's width
  for a float payload and a wrapping product in the integer's promoted
  width for an integer one, ``trunc`` is Julia's checked conversion, and
  the final Int64 addition wraps. Every wrap is Julia's defined modular
  arithmetic and is reproduced; the one undefined step, ``trunc(Int64,
  -Inf16)`` (Julia's Float16 range check admits it), is refused.
  ``datetime2unix`` is ``Float64(ms - UNIXEPOCH) / 1000.0``. The proleptic
  Gregorian calendar has a year zero and no range limit.
"""

from __future__ import annotations

import math
import struct
from decimal import Decimal
from typing import Any

import numpy as np

from ._metadata import MAX_INT64, MIN_INT64

# IEEE float classes by payload width (the arithmetic width Julia uses).
FLOAT_TYPES: dict[int, type[np.floating[Any]]] = {2: np.float16, 4: np.float32, 8: np.float64}

# ---- finite parameter tables ----------------------------------------------

# Julia's bit-integer types by exact spelling: (typemin, typemax).
INTEGER_PARAMETERS: dict[str, tuple[int, int]] = {
    **{f"Int{bits}": (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) for bits in (8, 16, 32, 64, 128)},
    **{f"UInt{bits}": (0, (1 << bits) - 1) for bits in (8, 16, 32, 64, 128)},
}
# ``Complex{Bool}`` is a Julia type its writer stores and its loader rebuilds;
# ``Rational{Bool}`` fails Julia's own type assertion, so Bool is a complex
# parameter only.
COMPLEX_PARAMETERS: dict[str, tuple[int, int]] = {**INTEGER_PARAMETERS, "Bool": (0, 1)}
RATIONAL_PARAMETERS: dict[str, tuple[int, int]] = dict(INTEGER_PARAMETERS)


def rational_parameter(token: str) -> str | None:
    """Return the parameter of an exactly spelled ``Rational{T}`` token, else None."""
    return _parameter(token, "Rational", RATIONAL_PARAMETERS)


def integer_complex_parameter(token: str) -> str | None:
    """Return the parameter of an exactly spelled integer ``Complex{T}`` token, else None."""
    return _parameter(token, "Complex", COMPLEX_PARAMETERS)


def rational_complex_parameter(token: str) -> str | None:
    """Return ``T`` of an exactly spelled ``Complex{Rational{T}}`` token, else None."""
    if token.startswith("Complex{") and token.endswith("}"):
        return rational_parameter(token[len("Complex{") : -1])
    return None


def _parameter(token: str, outer: str, table: dict[str, tuple[int, int]]) -> str | None:
    prefix = f"{outer}{{"
    if token.startswith(prefix) and token.endswith("}"):
        inner = token[len(prefix) : -1]
        if inner in table:
            return inner
    return None


# ---- Julia's integer constructors ------------------------------------------


def _failure(kind: str, target: str, value: Any) -> ValueError:
    return ValueError(f"Julia raises {kind} rebuilding {target} from {value!r}.")


def integer_from_float(value: float, parameter: str) -> int:
    """Julia's ``T(x::Float64)``: finite, integral and within ``T``, else ``InexactError``.

    Signed zero converts to ``0``. ``Bool`` accepts exactly ``0.0`` and ``1.0``.
    """
    low, high = COMPLEX_PARAMETERS[parameter]
    if not math.isfinite(value) or value != math.floor(value):
        raise _failure("InexactError", parameter, value)
    number = int(value)
    if not low <= number <= high:
        raise _failure("InexactError", parameter, value)
    return number


def integer_from_integer(value: int, parameter: str) -> int:
    """Julia's ``T(n::Integer)``: within ``T`` (``typemin`` included), else ``InexactError``."""
    low, high = COMPLEX_PARAMETERS[parameter]
    if not low <= value <= high:
        raise _failure("InexactError", parameter, value)
    return value


# ---- Rational{T} ----------------------------------------------------------


def float_as_ratio(value: float) -> tuple[int, int]:
    """Return the exact dyadic ratio of a finite float in lowest terms (``-0.0`` is ``0/1``)."""
    if not math.isfinite(value):
        raise ValueError("Only finite floats have an exact ratio.")
    return value.as_integer_ratio()


def _push(
    a: float, p: int, q: int, pp: int, qq: int, low: int, high: int
) -> tuple[int, int] | None:
    """``convert(T, a)`` and the checked convergent recurrence; None where Julia's ``try`` fails."""
    a = float(a)
    if not (math.isfinite(a) and low <= a <= high):
        return None
    ia = int(a)
    np_, nq = ia * p + pp, ia * q + qq
    if not (low <= np_ <= high and low <= nq <= high):
        return None
    return np_, nq


def _convergents(value: float, low: int, high: int, width: int = 8) -> tuple[int, int]:
    """Julia's ``rationalize(T, x, tol=0)`` for a finite ``x``, operation for operation.

    Julia keeps the remainders exact by carrying the pair ``x/y`` and taking
    ``divrem(x, y)`` in the payload's own float width (an exact ``fmod`` and,
    for Float64, ``round((x - r) / y)``; Float32 and Float16 truncate the
    quotient computed one width up and round it back), and pushes each partial
    quotient through the checked ``T`` arithmetic of the convergent recurrence;
    the first quotient or product outside ``T`` ends the expansion with the
    previous convergent. The quotient itself is a float, so a partial quotient
    above ``2**53`` that is not exactly representable rounds, and Julia then
    builds a *nearby* rational that its final ``x == T(r)`` check still
    accepts: ``Rational{Int64}(3 * 2.0^-62)`` is ``3//4611686018427387649``,
    not ``3//2^62``. That is the value Julia rebuilds, so it is the value
    reproduced here; the exact dyadic decomposition (:func:`float_as_ratio`)
    is a different number.
    """
    cls = FLOAT_TYPES[width]
    with np.errstate(all="ignore"):
        x = cls(value)
        p, q = (-1 if x < 0 else 1), 0
        pp, qq = 0, 1
        x = abs(x)
        a = np.trunc(x)
        r = x - a
        y = cls(1)
        zero = cls(0)
        while r > zero:
            if (pushed := _push(a, p, q, pp, qq, low, high)) is None:
                return p, q
            p, pp = pushed[0], p
            q, qq = pushed[1], q
            x, y = y, r
            r = np.fmod(x, y)
            if width == 8:
                quotient = (x - r) / y
                a = np.rint(quotient) if np.isfinite(quotient) else quotient
            else:
                wider = np.float64 if width == 4 else np.float32
                a = cls(np.trunc(wider(x) / wider(y)))
        # The final semiconvergent ``cld(x, y)`` is this same quotient once r == 0.
        pushed = _push(a, p, q, pp, qq, low, high)
        return (p, q) if pushed is None else pushed


def julia_float(num: int, den: int) -> float:
    """Return Julia's ``Float64(::Rational)``: ``Float64(num) / Float64(den)``.

    This is also the Float64 Julia's writer stores for a ``Rational`` scalar
    (two roundings, not the correctly rounded quotient). A zero denominator
    gives the signed infinity; ``Float64(n)`` of a bit integer rounds to
    nearest-even, as ``float(int)`` does.
    """
    if den == 0:
        return math.copysign(math.inf, num)
    return float(num) / float(den)


def round_to_precision(number: int, precision: int) -> float:
    """Nearest-even rounding of an integer to ``precision`` significant bits.

    The result is returned as a Python float, which represents it exactly for
    the 24-bit (Float32) precision used here; ``float(number)`` already
    performs the 53-bit rounding for Float64.
    """
    if number == 0:
        return 0.0
    sign, magnitude = (-1, -number) if number < 0 else (1, number)
    shift = magnitude.bit_length() - precision
    if shift <= 0:
        return float(sign * magnitude)
    quotient, remainder = divmod(magnitude, 1 << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and quotient & 1):
        quotient += 1
    return sign * math.ldexp(float(quotient), shift)


def _float32_of_int(number: int) -> np.float32:
    """Julia's ``Float32(n::Integer)``: one correctly rounded conversion (``sitofp``)."""
    with np.errstate(all="ignore"):
        return np.float32(round_to_precision(number, 24))


def julia_narrow_float(num: int, den: int, width: int, parameter: str) -> np.floating[Any]:
    """Julia's ``Float32(::Rational{T})`` or ``Float16(::Rational{T})`` (``base/rational.jl``).

    ``T(x::Rational{S})`` divides ``P(num)`` by ``P(den)`` in ``P =
    promote_type(T, S)``, which is ``T`` itself for every bit integer ``S``;
    the exceptions avoid spurious overflow: ``Float16`` of a 16- to 64-bit
    parameter goes through ``Float32``, and ``Float32``/``Float16`` of a
    128-bit parameter through ``Float64``.
    """
    cls = FLOAT_TYPES[width]
    bits = int(parameter.lstrip("UInt"))
    with np.errstate(all="ignore"):
        if den == 0:
            return cls(math.copysign(math.inf, num))
        if bits == 128:
            return cls(np.float64(float(num)) / np.float64(float(den)))
        if width == 4:
            return _float32_of_int(num) / _float32_of_int(den)
        if bits <= 8:
            return np.float16(_float32_of_int(num)) / np.float16(_float32_of_int(den))
        return np.float16(_float32_of_int(num) / _float32_of_int(den))


def rationalize(value: float, parameter: str, width: int = 8) -> tuple[int, int]:
    """Julia's ``Rational{T}(x)`` of a float ``x``, as ``(numerator, denominator)``.

    ``width`` is the payload's byte width (8, 4 or 2), which selects the float
    arithmetic Julia runs. Raises ``ValueError`` naming Julia's failure:
    ``OverflowError`` for a negative value under an unsigned ``T``,
    ``InexactError`` for NaN and for any value whose best ``T``-representable
    convergent does not convert back to ``x`` in ``x``'s own width. ``±Inf``
    gives ``(±1, 0)``, exactly as Julia does (``1//0``).
    """
    low, high = RATIONAL_PARAMETERS[parameter]
    x = float(value)
    if low == 0 and x < 0:
        raise _failure("OverflowError", f"Rational{{{parameter}}}", x)
    if math.isnan(x):
        raise _failure("InexactError", parameter, x)
    if math.isinf(x):
        return (-1 if x < 0 else 1), 0
    num, den = _convergents(x, low, high, width)
    rebuilt: Any = (
        julia_float(num, den) if width == 8 else julia_narrow_float(num, den, width, parameter)
    )
    if rebuilt != FLOAT_TYPES[width](x):
        raise _failure("InexactError", f"Rational{{{parameter}}}", x)
    return num, den


def rational_from_integer(value: int, parameter: str) -> tuple[int, int]:
    """Julia's ``convert(Rational{T}, n::Integer)``: ``T(n)//1``."""
    return integer_from_integer(value, parameter), 1


# ---- Complex{T} for integer T --------------------------------------------


def integer_complex_from_floats(real: float, imag: float, parameter: str) -> tuple[int, int]:
    """Julia's ``convert(Complex{T}, z)`` for a Float64 or ComplexF64 payload.

    A Float64 payload is the pair ``(value, 0.0)``. Each component follows
    ``T(x::Float64)`` (:func:`integer_from_float`); ``Complex(-0.0, -0.0)``
    is ``0 + 0im``.
    """
    return integer_from_float(real, parameter), integer_from_float(imag, parameter)


def integer_complex_from_integer(value: int, parameter: str) -> tuple[int, int]:
    """Julia's ``convert(Complex{T}, n::Integer)``: ``T(n) + 0im``."""
    return integer_from_integer(value, parameter), 0


# ---- Dates: proleptic Gregorian calendar with a year zero -----------------

# Rata Die milliseconds of 1970-01-01T00:00:00 (``Dates.UNIXEPOCH``); the
# Rata Die day count is Python's ``date.toordinal`` extended to every year.
UNIX_EPOCH_MS = 62135683200000
MS_PER_DAY = 86400000
_SHIFTED_MONTH_DAYS = (306, 337, 0, 31, 61, 92, 122, 153, 184, 214, 245, 275)


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def days_in_month(year: int, month: int) -> int:
    if month == 2:
        return 29 if is_leap_year(year) else 28
    return 30 if month in (4, 6, 9, 11) else 31


def total_days(year: int, month: int, day: int) -> int:
    """Julia's ``Dates.totaldays``: Rata Die days of a validated calendar date."""
    if not 1 <= month <= 12:
        raise ValueError(f"Month {month} is outside 1..12.")
    if not 1 <= day <= days_in_month(year, month):
        raise ValueError(
            f"Day {day} is outside the {days_in_month(year, month)} days of month {month}."
        )
    z = year - 1 if month < 3 else year
    return day + _SHIFTED_MONTH_DAYS[month - 1] + 365 * z + z // 4 - z // 100 + z // 400 - 306


def year_month_day(days: int) -> tuple[int, int, int]:
    """Julia's ``Dates.yearmonthday``: the calendar date of a Rata Die day count."""
    z = days + 306
    h = 100 * z - 25
    a = h // 3652425
    b = a - a // 4
    y = (100 * b + h) // 36525
    c = b + z - 365 * y - y // 4
    m = (5 * c + 456) // 153
    d = c - (153 * m - 457) // 5
    return (y + 1, m - 12, d) if m > 12 else (y, m, d)


def wrap_int64(value: int) -> int:
    """Reduce an integer to Julia's wrapping Int64 (its defined modular arithmetic)."""
    return (value + (1 << 63)) % (1 << 64) - (1 << 63)


def rata_die_ms_from_unix_seconds(seconds: float) -> int:
    """``Dates.unix2datetime`` for a Float64 payload, as Rata Die milliseconds.

    ``trunc(Int64, 1000.0 * x)`` toward zero, so a stored ``-0.0015`` is
    ``-1`` ms; a product that is not finite or lies outside Int64 is Julia's
    ``InexactError``. ``UNIXEPOCH + ms`` then wraps like Julia's Int64
    addition (a product within about ``6.2e10`` of ``typemax`` lands in the
    year -292 million).
    """
    product = 1000.0 * seconds
    if not math.isfinite(product):
        raise _failure("InexactError", "Int64", product)
    truncated = math.trunc(product)
    if not MIN_INT64 <= truncated <= MAX_INT64:
        raise _failure("InexactError", "Int64", product)
    return wrap_int64(UNIX_EPOCH_MS + truncated)


def rata_die_ms_from_unix_narrow(seconds: Any, width: int) -> int:
    """``Dates.unix2datetime`` for a Float32 or Float16 payload.

    ``Int64(1000) * x`` promotes to the payload's own float type, so the
    product rounds in that width (``Float32(1000) * x`` or ``Float16(1000) *
    x``) before ``trunc(Int64, ...)``. A product that is not finite is Julia's
    ``InexactError`` for Float32 and for a Float16 ``+Inf``/``NaN``; a Float16
    product of ``-Inf`` passes Julia's range check (``Float16(typemin(Int64))
    <= x``) into ``unsafe_trunc``, whose result for an infinite value is
    undefined, so it is refused rather than reproduced.
    """
    cls = FLOAT_TYPES[width]
    with np.errstate(all="ignore"):
        product = cls(1000) * cls(seconds)
    if not np.isfinite(product):
        if width == 2 and product < 0:
            raise ValueError(
                "The Float16 product of the unix time is -Inf; Julia's trunc(Int64, ::Float16) "
                "passes it to an undefined conversion, so no defined value exists to reproduce."
            )
        raise _failure("InexactError", "Int64", float(product))
    return rata_die_ms_from_unix_seconds(float(product) / 1000.0)


def rata_die_ms_from_unix_uint64(seconds: int) -> int:
    """``Dates.unix2datetime`` for a UInt64 payload: a wrapping UInt64 product, then ``trunc``.

    ``Int64(1000) * x`` promotes to UInt64 and wraps modulo ``2**64``;
    ``trunc(Int64, ::UInt64)`` then raises ``InexactError`` above
    ``typemax(Int64)``, and ``UNIXEPOCH + n`` wraps in Int64.
    """
    product = (1000 * seconds) % (1 << 64)
    if product > MAX_INT64:
        raise _failure("InexactError", "Int64", product)
    return wrap_int64(UNIX_EPOCH_MS + product)


def rata_die_ms_from_unix_wide_integer(seconds: int, *, signed: bool) -> int:
    """``Dates.unix2datetime`` for an Int128/UInt128 payload: a 128-bit product, then ``trunc``.

    ``Int64(1000) * x`` wraps modulo ``2**128`` in the payload's own
    signedness; ``trunc(Int64, ...)`` raises ``InexactError`` outside Int64,
    and ``UNIXEPOCH + n`` wraps in Int64.
    """
    product = 1000 * seconds
    if signed:
        product = (product + (1 << 127)) % (1 << 128) - (1 << 127)
    else:
        product %= 1 << 128
    if not MIN_INT64 <= product <= MAX_INT64:
        raise _failure("InexactError", "Int64", product)
    return wrap_int64(UNIX_EPOCH_MS + product)


def rata_die_ms_from_unix_integer(seconds: int) -> int:
    """``Dates.unix2datetime`` for an Int8..Int64 or UInt8..UInt32 payload.

    The payload promotes to Int64, ``1000 * n`` wraps in Int64 and so does
    ``UNIXEPOCH + n``: Julia's defined modular arithmetic, reproduced as is.
    """
    return wrap_int64(UNIX_EPOCH_MS + wrap_int64(1000 * seconds))


def civil_from_rata_die_ms(rata: int) -> tuple[int, int, int, int, int, int, int]:
    """``(year, month, day, hour, minute, second, millisecond)`` of a DateTime value."""
    days, ms_of_day = divmod(rata, MS_PER_DAY)
    year, month, day = year_month_day(days)
    hour, rest = divmod(ms_of_day, 3600000)
    minute, rest = divmod(rest, 60000)
    second, millisecond = divmod(rest, 1000)
    return year, month, day, hour, minute, second, millisecond


def date_from_rata_die_ms(rata: int) -> tuple[int, int, int]:
    """``Date(dt::DateTime)``: the calendar date of ``fld(value, 86400000)``."""
    return year_month_day(rata // MS_PER_DAY)


def rata_die_ms_from_civil(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
    second: int = 0,
    millisecond: int = 0,
) -> int:
    """``Dates.DateTime(y, m, d, h, mi, s, ms)`` as Rata Die milliseconds."""
    if not (0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60 and 0 <= millisecond < 1000):
        raise ValueError("Time of day components are outside their ranges.")
    return millisecond + 1000 * (
        second + 60 * minute + 3600 * hour + 86400 * total_days(year, month, day)
    )


def unix_seconds_from_rata_die_ms(rata: int) -> float:
    """``Dates.datetime2unix``: ``Float64(rata - UNIXEPOCH) / 1000.0``.

    This is the Float64 Julia's writer stores for a ``Date`` or ``DateTime``;
    beyond ``2**53`` milliseconds (about year ±285,000) the conversion to
    Float64 rounds, which is Julia's own precision loss.
    """
    return float(rata - UNIX_EPOCH_MS) / 1000.0


# ---- Julia's printed forms (base/ryu/shortest.jl) ---------------------------


def _shortest_digits(value: Any, width: int) -> tuple[str, int]:
    """Return ``(digits, exponent)`` with ``value == int(digits) * 10**exponent``.

    The digit string is the shortest one that round-trips in the value's own
    width (Ryu's ``reduce_shortest``), without trailing zeros: Python's
    ``repr`` provides it for Float64 and NumPy's Dragon4 (``unique=True``) for
    Float32 and Float16; both pick the candidate closest to the true value,
    as Ryu does.
    """
    if width == 8:
        text = repr(float(value))
        mantissa, _, exp = text.partition("e")
        exponent = int(exp) if exp else 0
    else:
        text = np.format_float_scientific(FLOAT_TYPES[width](value), unique=True, trim="-")
        mantissa, _, exp = text.partition("e")
        exponent = int(exp)
    whole, _, frac = mantissa.lstrip("-").partition(".")
    digits = (whole + frac).lstrip("0")
    exponent -= len(frac)
    stripped = digits.rstrip("0")
    return stripped, exponent + len(digits) - len(stripped)


def julia_float_string(value: Any, width: int, typed: bool = False, *, show: bool = False) -> str:
    """Julia's ``string(x)`` of a Float64/Float32/Float16 (``Ryu.writeshortest``).

    ``typed=False`` is ``print``/``string`` (``1.5``, ``1.0e10``, ``NaN``);
    ``typed=True`` is ``show`` inside a complex number (``1.5f0``, ``1.0f10``,
    ``NaN32``, ``Float16(1.5)``); ``show=True`` is ``show`` where the context
    already knows the type (an element of a ``Float32[...]`` array): no
    suffix, but Julia keeps ``f`` as the Float32 exponent marker
    (``1.0f10``). Fixed notation is used when the decimal point lands in
    ``-4 < pt <= 6`` (``<= 3`` for Float16) and exponent notation otherwise,
    with Julia's bare exponent (``e-5``, ``e15``).
    """
    x = float(value)
    suffix = {4: "32", 2: "16"}.get(width, "") if typed else ""
    if math.isnan(x):
        return "NaN" + suffix
    if math.isinf(x):
        return ("-" if x < 0 else "") + "Inf" + suffix
    sign = "-" if math.copysign(1.0, x) < 0 else ""
    if x == 0:
        body = sign + "0.0"
    else:
        digits, nexp = _shortest_digits(value, width)
        olength = len(digits)
        pt = nexp + olength
        limit = 3 if width == 2 else 6
        fixed = -4 < pt <= limit and not (
            pt >= olength and abs((x + 0.05) % 10 ** (pt - olength) - 0.05) > 0.05
        )
        if fixed:
            if pt <= 0:
                body = "0." + "0" * (-pt) + digits
            elif pt >= olength:
                body = digits + "0" * nexp + ".0"
            else:
                body = digits[:pt] + "." + digits[pt:]
            body = sign + body
        else:
            mantissa = digits[0] + "." + (digits[1:] if olength > 1 else "0")
            expchar = "f" if (typed or show) and width == 4 else "e"
            body = sign + mantissa + expchar + str(pt - 1)
            return f"Float16({body})" if typed and width == 2 else body
    if typed and width == 4:
        body += "f0"
    if typed and width == 2:
        return f"Float16({body})"
    return body


def julia_complex_string(real: Any, imag: Any, width: int) -> str:
    """Julia's ``string(z)`` of a ComplexF64/ComplexF32/ComplexF16 (``show(io, z::Complex)``).

    Components print typed; a negative finite imaginary part prints as ``" - "``
    plus its magnitude, otherwise ``" + "``; a non-finite imaginary part gets
    ``*`` before ``im``. NaN components never carry a sign.
    """
    i = float(imag)
    text = julia_float_string(real, width, typed=True)
    if math.copysign(1.0, i) < 0 and not math.isnan(i):
        text += " - " + julia_float_string(-i, width, typed=True)
    else:
        text += " + " + julia_float_string(i, width, typed=True)
    if not math.isfinite(i):
        text += "*"
    return text + "im"


def exact_decimal(value: Any) -> Decimal:
    """Return the exact value of the ``BigFloat`` Julia builds from an integer or float payload.

    ``BigFloat(x)`` of a Float64/Float32/Float16 or of any bit integer up to
    128 bits is exact at Julia's default 256-bit precision, so the loaded
    number is exactly the payload's value; :class:`decimal.Decimal` holds it
    exactly (``Decimal(0.1)`` prints Julia's ``0.1000000000000000055511151231257827...``).
    NaN and the infinities map to Decimal's own ``NaN``/``Infinity``.
    """
    if isinstance(value, (int, np.integer)):
        return Decimal(int(value))
    x = float(value)
    if math.isnan(x):
        return Decimal("NaN")
    return Decimal(x)


# ---- width-preserving scalar bytes ---------------------------------------


def int128_from_bytes(payload: bytes, *, signed: bool) -> int:
    """Return the Python integer of a sixteen-byte little-endian Int128/UInt128 payload."""
    if len(payload) != 16:
        raise ValueError("Int128 and UInt128 scalars are exactly sixteen bytes.")
    return int.from_bytes(payload, "little", signed=signed)


def int128_to_bytes(value: int, *, signed: bool) -> bytes:
    """Return the sixteen little-endian bytes of an integer within Int128 or UInt128."""
    low, high = INTEGER_PARAMETERS["Int128" if signed else "UInt128"]
    if type(value) is not int or not low <= value <= high:
        raise ValueError(
            f"{value!r} is not an exact integer within {'Int128' if signed else 'UInt128'}."
        )
    return value.to_bytes(16, "little", signed=signed)


def complexf16_from_bytes(payload: bytes) -> tuple[np.float16, np.float16]:
    """Return the two Float16 components of a four-byte ComplexF16 payload, bits preserved."""
    if len(payload) != 4:
        raise ValueError("ComplexF16 scalars are exactly four bytes.")
    real, imag = np.frombuffer(payload, dtype="<f2")
    return np.float16(real), np.float16(imag)


def complexf16_to_bytes(real: np.float16, imag: np.float16) -> bytes:
    """Return the four bytes of two exact Float16 components (NaN payloads, signed zeros kept)."""
    if type(real) is not np.float16 or type(imag) is not np.float16:
        raise TypeError("ComplexF16 components must be exact numpy.float16 values.")
    return bytes(real.astype("<f2").tobytes()) + bytes(imag.astype("<f2").tobytes())


def float64_bits(value: float) -> bytes:
    """Return the eight little-endian bytes of a Python float (the Float64 scalar payload)."""
    return struct.pack("<d", value)


def float64_from_bits(payload: bytes) -> float:
    if len(payload) != 8:
        raise ValueError("Float64 scalars are exactly eight bytes.")
    return float(struct.unpack("<d", payload)[0])
