# SPDX-License-Identifier: MIT
"""Stored representation of DataEcon scalars that familiar Python values cannot hold.

Ordinary scalars (every numeric width with a NumPy type, strings, MIT dates and
Durations) travel as plain Python or NumPy values. Julia also stores scalars
that those values cannot carry losslessly: Int128/UInt128 and ComplexF16
widths, strings that are not valid UTF-8 or hold an interior NUL, and any
scalar whose ``jtype`` attribute asks the loader to rebuild another type
(``Symbol``, ``Date``, ``Rational{Int64}``, ``Complex{Int8}``, a custom
type, ...). :class:`StoredScalar` keeps such a scalar exactly as stored: the
payload bytes, the native type code, the frequency code of a date or
duration and the marker text. Reading never converts and never evaluates;
writing a ``StoredScalar`` back reproduces the stored object byte for byte,
including markers Julia's own writer cannot reproduce.

Interpretation is explicit. :meth:`StoredScalar.to_interpreted` returns the
value the pinned Julia loader would build, through a finite table of marker
routes backed by :mod:`._exact` (Julia's own ``Rational{T}``, integer
``Complex{T}`` and ``Dates`` arithmetic) and the numeric conversion kernels of
:mod:`._interpret` (the same routes the series containers use). Values Julia
cannot load raise ``ValueError``; markers with no supported interpretation
raise ``TypeError`` and remain preserved. Explicit accessors (``to_int``,
``to_float``, ``to_fraction``, ``to_date``, ``to_datetime64``, ...) expose one
family each. The constructors build the storage Julia's own writer produces
for a Fraction, a calendar date, an integer complex value, a Symbol or a
128-bit integer, checking that Julia reconstructs the requested value.
"""

from __future__ import annotations

import datetime as dt
import math
import struct
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np

from tsecon.frequencies import Frequency
from tsecon.mit import MIT, Duration

from . import _exact, _interpret, _printed
from ._interpret import Target
from ._metadata import (
    _CALENDAR_RANGES,
    _SCALAR_FREQUENCIES,
    _SCALAR_MIN_DATES,
    JULIA_NUMERIC_TYPES,
    KIND_COMPLEX,
    KIND_DATE,
    KIND_FLOAT,
    KIND_INTEGER,
    KIND_STRING,
    KIND_UNSIGNED,
    MAX_BYTES,
    MAX_DATE,
    MAX_INT64,
    MIN_INT64,
    UNIT_FREQUENCY,
)
from ._represented import (
    COMPLEXF16,
    INT128,
    UINT128,
    DatedComplex,
    IntegerComplex,
    RationalComplex,
    StoredElement,
    _exact_float,
    pack_complexf16,
)

__all__ = [
    "IntegerComplex",
    "RationalComplex",
    "StoredScalar",
    "scalar_frequency",
    "scalar_frequency_code",
    "validate_date_code",
    "validate_scalar_metadata",
]

# ---- native scalar metadata ------------------------------------------------

_SCALAR_SUPPORT = (
    "DataEcon scalar support covers Float16/32/64, Int8/16/32/64, UInt8/16/32/64, "
    "Complex64/128, Booleans stored as Int8, strings, and MIT dates or Durations "
    "over the Unit, Daily, BDaily, "
    "Weekly, Monthly, Quarterly, HalfYearly and Yearly frequencies."
)
# Payload widths per numeric type code. Int128/UInt128 (16 bytes) and ComplexF16
# (4 bytes) are Julia-written widths with no NumPy scalar type; they read as a
# StoredScalar. Every other width has no Julia loader method either.
NUMERIC_WIDTHS: dict[int, frozenset[int]] = {
    KIND_INTEGER: frozenset({1, 2, 4, 8, 16}),
    KIND_UNSIGNED: frozenset({1, 2, 4, 8, 16}),
    KIND_FLOAT: frozenset({2, 4, 8}),
    KIND_COMPLEX: frozenset({4, 8, 16}),
}
WIDE_WIDTHS: frozenset[tuple[int, int]] = frozenset(
    {(KIND_INTEGER, 16), (KIND_UNSIGNED, 16), (KIND_COMPLEX, 4)}
)
# Ordinary carrier dtypes by (type code, width); Bool is not a stored width.
_SCALAR_DTYPES: dict[tuple[int, int], np.dtype[Any]] = {
    (kind, dtype.itemsize): dtype
    for kind, dtype in JULIA_NUMERIC_TYPES.values()
    if dtype.kind != "b"
}
_WIDE_ELEMENTS: dict[tuple[int, int], StoredElement] = {
    (KIND_INTEGER, 16): INT128,
    (KIND_UNSIGNED, 16): UINT128,
    (KIND_COMPLEX, 4): COMPLEXF16,
}


def scalar_frequency(code: int) -> Frequency:
    """Resolve a supported native frequency code for a date or duration scalar."""
    try:
        return _SCALAR_FREQUENCIES[code]
    except KeyError:
        raise TypeError(_SCALAR_SUPPORT) from None


def scalar_frequency_code(frequency: object) -> int:
    """Return the canonical native code for a supported core frequency object."""
    code = next((code for code, freq in _SCALAR_FREQUENCIES.items() if freq == frequency), None)
    if code is None:
        raise TypeError(_SCALAR_SUPPORT)
    return code


def validate_date_code(frequency: int, code: int) -> None:
    """Require a date code the native codec decodes exactly for this frequency.

    Unit codes are plain signed 64-bit integers with no date bound. Calendar and
    year/period codes must lie inside their verified native windows; the backend
    then confirms the native round trip for those frequencies.
    """
    scalar_frequency(frequency)
    if frequency == UNIT_FREQUENCY:
        if not MIN_INT64 <= code <= MAX_INT64:
            raise ValueError("Unit date codes must fit the signed 64-bit range.")
        return
    if frequency in _CALENDAR_RANGES:
        minimum, maximum = _CALENDAR_RANGES[frequency]
    else:
        minimum, maximum = _SCALAR_MIN_DATES[frequency], MAX_DATE
    if not minimum <= code <= maximum:
        raise ValueError("Date is outside the reliable native date range for its frequency.")


def validate_scalar_metadata(metadata: tuple[int, int, int, int]) -> None:
    """Validate scalar class, type, frequency and length before dereferencing native memory."""
    cls, kind, frequency, nbytes = metadata
    if cls != 1:
        raise TypeError(_SCALAR_SUPPORT)
    if kind == KIND_STRING:
        if frequency != 0:
            raise TypeError("DataEcon string scalars carry no frequency.")
        if not 1 <= nbytes <= MAX_BYTES:
            raise ValueError(
                "DataEcon string scalars require a NUL-terminated payload within the size limit."
            )
        return
    if kind == KIND_DATE or (kind == KIND_INTEGER and frequency != 0):
        scalar_frequency(frequency)
        if nbytes != 8:
            raise ValueError("DataEcon date and duration scalars require exactly eight bytes.")
        return
    if kind not in NUMERIC_WIDTHS:
        raise TypeError(_SCALAR_SUPPORT)
    if frequency != 0:
        raise TypeError("DataEcon numeric scalars carry no frequency.")
    if nbytes not in NUMERIC_WIDTHS[kind]:
        raise ValueError(
            "DataEcon numeric scalar payload width is not supported for its type; "
            "supported widths are eight bytes for Int64/Float64, 1/2/4/16 for other "
            "integers, 2/4 for narrower floats and 4/8/16 for complex values."
        )


# ---- plain values of ordinary payloads --------------------------------------


def plain_text(payload: bytes) -> str:
    """Decode a NUL-terminated, NUL-free, valid UTF-8 text payload; ``ValueError`` otherwise."""
    if payload[-1] != 0 or b"\0" in payload[:-1]:
        raise ValueError("DataEcon string payload is not a single NUL-terminated string.")
    try:
        return payload[:-1].decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("DataEcon string payload is not valid UTF-8.") from None


def is_plain_text(payload: bytes) -> bool:
    try:
        plain_text(payload)
    except ValueError:
        return False
    return True


def plain_numeric(kind: int, payload: bytes) -> Any:
    """Return the Python/NumPy value of an ordinary frequency-free numeric payload.

    Eight-byte Int64/Float64 and sixteen-byte ComplexF64 return the exact-width
    built-ins; every other width returns the sized NumPy scalar class, copied
    out of the payload snapshot.
    """
    width = len(payload)
    if kind == KIND_INTEGER and width == 8:
        return int(struct.unpack("<q", payload)[0])
    if kind == KIND_FLOAT and width == 8:
        return float(struct.unpack("<d", payload)[0])
    if kind == KIND_COMPLEX and width == 16:
        real, imag = struct.unpack("<dd", payload)
        return complex(real, imag)
    result: np.generic = np.frombuffer(payload, dtype=_SCALAR_DTYPES[(kind, width)])[0]
    return result


# ---- finite marker vocabulary ----------------------------------------------

SPELLINGS = _interpret.SPELLINGS
TEXT_TOKENS = ("Symbol", "SubString{String}", "AbstractString", "String")
ABSTRACT_TOKENS = (
    "Any",
    "Number",
    "Real",
    "Integer",
    "Signed",
    "Unsigned",
    "AbstractFloat",
    "Union{Int64,Float64}",
    "Union{Int64, Float64}",
)
UNION_TOKENS = ("Union{Int64,Float64}", "Union{Int64, Float64}")
DATE_TOKENS = ("Date", "DateTime")
_INTEGER_NAMES: dict[tuple[int, int], str] = {
    (KIND_INTEGER, 1): "Int8",
    (KIND_INTEGER, 2): "Int16",
    (KIND_INTEGER, 4): "Int32",
    (KIND_INTEGER, 8): "Int64",
    (KIND_INTEGER, 16): "Int128",
    (KIND_UNSIGNED, 1): "UInt8",
    (KIND_UNSIGNED, 2): "UInt16",
    (KIND_UNSIGNED, 4): "UInt32",
    (KIND_UNSIGNED, 8): "UInt64",
    (KIND_UNSIGNED, 16): "UInt128",
}
_FLOAT_NAMES: dict[int, str] = {2: "Float16", 4: "Float32", 8: "Float64"}
_COMPLEX_NAMES: dict[int, str] = {4: "ComplexF16", 8: "ComplexF32", 16: "ComplexF64"}
_SIGNED_OF: dict[str, str] = {f"UInt{b}": f"Int{b}" for b in (8, 16, 32, 64, 128)}
_UNSIGNED_OF: dict[str, str] = {f"Int{b}": f"UInt{b}" for b in (8, 16, 32, 64, 128)}


resolve_spelling = _interpret.resolve_spelling


def _unsupported(what: str) -> TypeError:
    return TypeError(
        f"{what}; the stored value and its marker are preserved in the StoredScalar and the "
        "marker text is never evaluated."
    )


def _julia_refuses(kind: str, target: str, value: object) -> ValueError:
    return ValueError(f"Julia's loader raises {kind} rebuilding {target} from {value!r}.")


# ---- the stored scalar -----------------------------------------------------


@dataclass(frozen=True, slots=True)
class StoredScalar:
    """A DataEcon scalar kept exactly as stored: bytes, type code, frequency and marker.

    ``payload`` holds the stored bytes (a text payload includes its NUL
    terminator, exactly as stored); ``kind`` is the native type code (1
    integer, 2 unsigned, 3 date, 4 float, 5 complex, 6 string); ``frequency``
    is the native frequency code of a date or duration and zero otherwise;
    ``marker`` is the exact ``jtype`` text, or None. The value is immutable
    and independent of any file; equality compares all four fields, so a
    read-modify-write that preserves the object compares equal to the read.

    Ordinary supported scalars are returned as plain values by ``read_scalar``;
    a ``StoredScalar`` appears for the sixteen-byte and four-byte widths, for
    text that is not clean UTF-8, and for any marked scalar. Interpretation is
    always an explicit call.
    """

    payload: bytes
    kind: int
    frequency: int = 0
    marker: str | None = None

    def __post_init__(self) -> None:
        if type(self.payload) is not bytes:
            raise TypeError("StoredScalar payload must be a bytes object.")
        if type(self.kind) is not int or type(self.frequency) is not int:
            raise TypeError("StoredScalar kind and frequency must be native integer codes.")
        validate_scalar_metadata((1, self.kind, self.frequency, len(self.payload)))
        if self.kind == KIND_DATE:
            validate_date_code(self.frequency, self._code())
        _interpret.check_marker_text(self.marker, "scalar")

    # ---- constructors: the storage Julia's own writer produces --------------

    @classmethod
    def int128(cls, value: int) -> StoredScalar:
        """Build a sixteen-byte ``Int128`` scalar from an exact ``int`` in the signed range."""
        return cls(_exact.int128_to_bytes(value, signed=True), KIND_INTEGER)

    @classmethod
    def uint128(cls, value: int) -> StoredScalar:
        """Build a sixteen-byte ``UInt128`` scalar from an exact ``int`` in the unsigned range."""
        return cls(_exact.int128_to_bytes(value, signed=False), KIND_UNSIGNED)

    @classmethod
    def complexf16(cls, value: complex | tuple[np.float16, np.float16]) -> StoredScalar:
        """Build a four-byte ``ComplexF16`` scalar.

        Accepts a ``complex`` whose components are exactly representable in
        Float16 (NaN and infinities included; a NaN stores the quiet pattern)
        or an exact ``(float16, float16)`` pair, whose bits are stored
        unchanged. The packer is the one the series carriers use.
        """
        if type(value) is not complex and not (
            isinstance(value, tuple)
            and len(value) == 2
            and all(type(part) is np.float16 for part in value)
        ):
            raise TypeError(
                "ComplexF16 scalars are built from a Python complex or a (float16, float16) pair."
            )
        return cls(pack_complexf16([value]).tobytes(), KIND_COMPLEX)

    @classmethod
    def symbol(cls, text: str) -> StoredScalar:
        """Build a Julia ``Symbol``: the UTF-8 text plus the ``Symbol`` marker."""
        return cls.text(text, "Symbol")

    @classmethod
    def text(cls, text: str, marker: str | None = None) -> StoredScalar:
        """Build a string scalar with an optional preserved marker (``SubString{String}``, ...)."""
        if type(text) is not str:
            raise TypeError("StoredScalar.text requires a Python str.")
        return cls(encode_text(text), KIND_STRING, 0, marker)

    @classmethod
    def raw_text(
        cls, data: bytes, *, allow_nul: bool = False, marker: str | None = None
    ) -> StoredScalar:
        """Build a string scalar from arbitrary bytes (invalid UTF-8 allowed), NUL-terminated.

        An interior NUL is refused unless ``allow_nul=True``: Julia's own loader
        reads such a string back only up to the first NUL, so the opt-in
        documents that loss. The bytes must not already end with the terminator.
        """
        if type(data) is not bytes:
            raise TypeError("StoredScalar.raw_text requires a bytes object.")
        if b"\0" in data and not allow_nul:
            raise ValueError(
                "The bytes contain NUL; Julia's loader would truncate the string there. "
                "Pass allow_nul=True to store them anyway."
            )
        if len(data) + 1 > MAX_BYTES:
            raise ValueError("DataEcon string exceeds the payload size limit.")
        return cls(data + b"\0", KIND_STRING, 0, marker)

    @classmethod
    def fraction(
        cls, value: Fraction, *, parameter: str = "Int64", exact: bool = True
    ) -> StoredScalar:
        """Build the storage of a ``Rational{parameter}``: Julia's ``float(p//q)`` plus the marker.

        By default the write is accepted only when Julia's loader rebuilds
        exactly ``value`` from that Float64 (its continued-fraction
        reconstruction, not a Float64-exactness or denominator test); otherwise
        ``ValueError`` names ``exact=False``, which stores the float when Julia
        can load *some* rational from it and leaves the loss explicit. A value
        Julia cannot load at all is refused either way.
        """
        if type(value) is not Fraction:
            raise TypeError("StoredScalar.fraction requires a fractions.Fraction.")
        if parameter not in _exact.RATIONAL_PARAMETERS:
            raise TypeError(f"Unknown Rational parameter {parameter!r}.")
        num, den = value.numerator, value.denominator
        try:
            stored = _exact.julia_float(num, den)
        except OverflowError:
            raise ValueError(
                "The Fraction components exceed Julia's Float64 scalar storage range."
            ) from None
        rebuilt = _exact.rationalize(stored, parameter)  # ValueError where Julia fails
        if exact and rebuilt != (num, den):
            raise ValueError(
                f"Julia would reload {value} (stored as {stored!r}) as "
                f"{rebuilt[0]}//{rebuilt[1]} under Rational{{{parameter}}}; pass exact=False "
                "to store that float with the loss explicit."
            )
        return cls(_exact.float64_bits(stored), KIND_FLOAT, 0, f"Rational{{{parameter}}}")

    @classmethod
    def date(cls, value: dt.date) -> StoredScalar:
        """Build the storage of a Julia ``Date``: Float64 unix seconds plus the ``Date`` marker."""
        if type(value) is not dt.date:
            raise TypeError("StoredScalar.date requires an exact datetime.date (not a datetime).")
        return cls.calendar_date(value.year, value.month, value.day)

    @classmethod
    def datetime(cls, value: dt.datetime) -> StoredScalar:
        """Build the storage of a Julia ``DateTime`` from a naive millisecond-resolution datetime.

        Timezone-aware values are refused (``TypeError``): Julia's ``DateTime``
        carries no zone and the caller must convert explicitly. Sub-millisecond
        microseconds are refused (``ValueError``) rather than rounded.
        """
        if type(value) is not dt.datetime:
            raise TypeError("StoredScalar.datetime requires an exact datetime.datetime.")
        if value.tzinfo is not None:
            raise TypeError(
                "Julia DateTime values carry no time zone; convert the aware datetime explicitly."
            )
        if value.microsecond % 1000:
            raise ValueError(
                "Julia DateTime values have millisecond resolution; refusing to round."
            )
        return cls.calendar_datetime(
            value.year,
            value.month,
            value.day,
            value.hour,
            value.minute,
            value.second,
            value.microsecond // 1000,
        )

    @classmethod
    def calendar_date(cls, year: int, month: int, day: int) -> StoredScalar:
        """Build a Julia ``Date`` for any proleptic Gregorian year (year zero and negatives too)."""
        rata = _exact.MS_PER_DAY * _exact.total_days(*_ints(year, month, day))
        return cls._date_storage(rata, "Date")

    @classmethod
    def calendar_datetime(
        cls,
        year: int,
        month: int,
        day: int,
        hour: int = 0,
        minute: int = 0,
        second: int = 0,
        millisecond: int = 0,
    ) -> StoredScalar:
        """Build a Julia ``DateTime`` for any proleptic year, at millisecond resolution."""
        rata = _exact.rata_die_ms_from_civil(
            *_ints(year, month, day, hour, minute, second, millisecond)
        )
        return cls._date_storage(rata, "DateTime")

    @classmethod
    def _date_storage(cls, rata: int, marker: str) -> StoredScalar:
        # Julia's writer stores datetime2unix; the value is accepted only when
        # unix2datetime of that Float64 reproduces the same instant. Test the
        # actual multiply/truncate path: neither a year nor a 2**53 cutoff
        # characterizes every exact round trip.
        if not MIN_INT64 <= rata <= MAX_INT64:
            raise ValueError("The instant is outside Julia's Int64 millisecond range.")
        stored = _exact.unix_seconds_from_rata_die_ms(rata)
        try:
            rebuilt = _exact.rata_die_ms_from_unix_seconds(stored)
        except ValueError:
            rebuilt = None
        if rebuilt != rata:
            raise ValueError(
                "Julia stores dates as Float64 unix seconds and would not reload this "
                "instant exactly."
            )
        return cls(_exact.float64_bits(stored), KIND_FLOAT, 0, marker)

    @classmethod
    def integer_complex(
        cls, real: int | IntegerComplex, imag: int = 0, *, parameter: str = "Int64"
    ) -> StoredScalar:
        """Build the storage of an integer ``Complex{parameter}``: two Float64 plus the marker.

        Accepted exactly when each component lies within ``parameter`` and is
        exactly representable in Float64, which is what Julia's own writer
        stores and reloads (no 2**53 cutoff: ``2**54 + 1im`` round-trips,
        ``2**53 + 1`` does not).
        """
        if type(real) is IntegerComplex:
            if imag != 0:
                raise TypeError("Pass either an IntegerComplex or two integers, not both.")
            real, imag = real.real, real.imag
        if type(real) is not int or type(imag) is not int:
            raise TypeError("StoredScalar.integer_complex requires exact Python int components.")
        if parameter not in _exact.COMPLEX_PARAMETERS:
            raise TypeError(f"Unknown Complex parameter {parameter!r}.")
        stored_real = _exact_float(real, "real part")
        stored_imag = _exact_float(imag, "imaginary part")
        rebuilt = _exact.integer_complex_from_floats(stored_real, stored_imag, parameter)
        if rebuilt != (real, imag):  # pragma: no cover - the bounds check above raised
            raise ValueError("Julia would not reload the requested components.")
        return cls(
            struct.pack("<dd", stored_real, stored_imag), KIND_COMPLEX, 0, f"Complex{{{parameter}}}"
        )

    @classmethod
    def rational_complex(
        cls,
        real: Fraction | RationalComplex,
        imag: Fraction | None = None,
        *,
        parameter: str = "Int64",
        exact: bool = True,
    ) -> StoredScalar:
        """Build the storage of a ``Complex{Rational{parameter}}``: ``float(z)`` plus the marker.

        Julia's writer stores the two components as Float64 (``Float64(num) /
        Float64(den)`` each) and its loader rebuilds each with ``Rational{T}``'s
        continued-fraction reconstruction. As for :meth:`fraction`, the write is
        accepted by default only when both components reload exactly;
        ``exact=False`` stores the floats when Julia can load *some* rational
        pair from them and leaves the loss explicit.
        """
        if isinstance(real, RationalComplex):
            if imag is not None:
                raise TypeError("Pass either a RationalComplex or two Fractions, not both.")
            parts: tuple[Any, Any] = (real.real, real.imag)
        else:
            parts = (real, Fraction(0) if imag is None else imag)
        if any(type(part) is not Fraction for part in parts):
            raise TypeError("StoredScalar.rational_complex requires fractions.Fraction components.")
        if parameter not in _exact.RATIONAL_PARAMETERS:
            raise TypeError(f"Unknown Rational parameter {parameter!r}.")
        stored = []
        for part, what in zip(parts, ("real part", "imaginary part"), strict=True):
            try:
                value = _exact.julia_float(part.numerator, part.denominator)
            except OverflowError:
                raise ValueError(
                    f"The {what} components exceed Julia's Float64 scalar storage range."
                ) from None
            rebuilt = _exact.rationalize(value, parameter)  # ValueError where Julia fails
            if exact and rebuilt != (part.numerator, part.denominator):
                raise ValueError(
                    f"Julia would reload the {what} {part} (stored as {value!r}) as "
                    f"{rebuilt[0]}//{rebuilt[1]} under Rational{{{parameter}}}; pass "
                    "exact=False to store that float with the loss explicit."
                )
            stored.append(value)
        return cls(
            struct.pack("<dd", *stored), KIND_COMPLEX, 0, f"Complex{{Rational{{{parameter}}}}}"
        )

    @classmethod
    def bigfloat(cls, value: Decimal | float | int) -> StoredScalar:
        """Build a Float64 payload with the ``BigFloat`` marker, which Julia's loader accepts.

        This is a storage path this adapter provides, not a transcription of
        Julia's writer: the pinned writer fails on a ``BigFloat`` (its type
        mapping recurses), while the pinned loader rebuilds
        ``BigFloat(::Float64)`` from a Float64 payload under the marker, which
        is exact. The requested value must therefore itself be exactly a
        Float64: a ``Decimal``, ``float`` or ``int`` whose value survives
        ``float(value)`` unchanged (``Decimal("0.1")`` does not, ``Decimal(0.1)``
        does). Anything else is refused rather than rounded. This is not a
        general arbitrary-precision interface; only Float64-representable
        values can be written this way.
        """
        if type(value) not in (Decimal, float, int):
            raise TypeError("StoredScalar.bigfloat takes a Decimal, float or int.")
        try:
            stored = float(value)
        except OverflowError:
            raise ValueError(f"{value!r} is beyond the Float64 range.") from None
        if not math.isnan(stored) and _exact.exact_decimal(stored) != Decimal(value):
            raise ValueError(
                f"{value!r} is not exactly representable in Float64; the stored float would "
                "reload as a different BigFloat."
            )
        return cls(_exact.float64_bits(stored), KIND_FLOAT, 0, "BigFloat")

    # ---- description ---------------------------------------------------------

    def __repr__(self) -> str:
        shown = self.payload.hex() if len(self.payload) <= 32 else f"{len(self.payload)} bytes"
        marker = "" if self.marker is None else f", marker={self.marker!r}"
        return f"StoredScalar({self.julia_name}, {shown}{marker})"

    def _code(self) -> int:
        return int(struct.unpack("<q", self.payload)[0])

    @property
    def nbytes(self) -> int:
        return len(self.payload)

    @property
    def is_text(self) -> bool:
        """Whether the stored family is a string (of any bytes)."""
        return self.kind == KIND_STRING

    @property
    def is_wide(self) -> bool:
        """Whether this is a sixteen-byte integer or four-byte complex width."""
        return (self.kind, self.nbytes) in WIDE_WIDTHS

    @property
    def active_marker(self) -> str | None:
        """The preserved ``jtype`` text (never evaluated)."""
        return self.marker

    @property
    def element(self) -> StoredElement | None:
        """The stored family as the shared element descriptor; None for text."""
        if self.kind == KIND_STRING:
            return None
        if self.kind == KIND_DATE:
            return StoredElement.date(scalar_frequency(self.frequency))
        if self.frequency != 0:
            return StoredElement.duration(scalar_frequency(self.frequency))
        wide = _WIDE_ELEMENTS.get((self.kind, self.nbytes))
        if wide is not None:
            return wide
        return StoredElement.numeric(_SCALAR_DTYPES[(self.kind, self.nbytes)])

    @property
    def julia_name(self) -> str:
        """Julia's name of the stored family (``Float64``, ``String``, ``MIT{Monthly}``, ...)."""
        element = self.element
        return "String" if element is None else element.julia_name

    def _source(self) -> Target:
        element = self.element
        assert element is not None
        return element.target

    def _carrier(self) -> np.ndarray[Any, Any]:
        """Return the one-element carrier array of the stored family."""
        return np.frombuffer(self.payload, dtype=self._source().dtype)

    # ---- accessors of the stored family (the marker is not applied) -----------

    def to_bytes(self) -> bytes:
        return self.payload

    def to_int(self) -> int:
        """Return the exact integer of an integer, unsigned, date-code or duration payload."""
        if self.kind not in (KIND_INTEGER, KIND_UNSIGNED, KIND_DATE):
            raise TypeError(f"to_int applies to integer payloads, not {self.julia_name}.")
        return int.from_bytes(self.payload, "little", signed=self.kind != KIND_UNSIGNED)

    def to_float(self) -> float:
        """Return the exact Python float of a Float16/32/64 payload (``+/-inf`` for ``+/-1//0``)."""
        if self.kind != KIND_FLOAT:
            raise TypeError(f"to_float applies to float payloads, not {self.julia_name}.")
        return float(
            np.frombuffer(self.payload, dtype=_SCALAR_DTYPES[(KIND_FLOAT, self.nbytes)])[0]
        )

    def to_complex(self) -> complex:
        """Return the Python complex of a ComplexF16/32/64 payload (Float16/32 widen exactly)."""
        if self.kind != KIND_COMPLEX:
            raise TypeError(f"to_complex applies to complex payloads, not {self.julia_name}.")
        if self.nbytes == 4:
            real, imag = _exact.complexf16_from_bytes(self.payload)
            return complex(float(real), float(imag))
        real, imag = np.frombuffer(self.payload, dtype=f"<f{self.nbytes // 2}")
        return complex(float(real), float(imag))

    def to_complex64(self) -> np.complex64:
        """Widen a ComplexF16 payload exactly into ``numpy.complex64``."""
        if (self.kind, self.nbytes) != (KIND_COMPLEX, 4):
            raise TypeError("to_complex64 applies to ComplexF16 payloads only.")
        real, imag = _exact.complexf16_from_bytes(self.payload)
        # Assign the components separately: complex arithmetic would turn an
        # infinite imaginary part into a NaN real part and lose a signed zero.
        wide = np.empty(1, dtype=np.complex64)
        wide.real = np.float32(real)
        wide.imag = np.float32(imag)
        result: np.complex64 = wide[0]
        return result

    def to_str(self) -> str:
        """Decode a string payload strictly as UTF-8 (raw bytes or an interior NUL raise)."""
        if self.kind != KIND_STRING:
            raise TypeError(f"to_str applies to string payloads, not {self.julia_name}.")
        return plain_text(self.payload)

    def _stored_value(self) -> Any:
        """Return the value of the stored family without its marker (what ``Any`` reloads)."""
        if self.kind == KIND_STRING:
            return self.to_str()
        if self.kind == KIND_DATE:
            return MIT(scalar_frequency(self.frequency), self._code())
        if self.frequency != 0:
            return Duration(scalar_frequency(self.frequency), self._code())
        if self.is_wide:
            return self.to_int() if self.kind != KIND_COMPLEX else self.to_complex()
        return plain_numeric(self.kind, self.payload)

    # ---- marker-driven interpretation ------------------------------------------

    def to_interpreted(self) -> Any:
        """Return the value the pinned Julia loader builds from the payload and marker.

        Unmarked wide widths return ``int`` (Int128/UInt128) or ``complex``
        (ComplexF16); unmarked text returns ``str`` when it is clean UTF-8.
        Marked scalars follow a finite table: text markers give ``str``
        (``Symbol`` of a numeric or date payload is Julia's printed form),
        ``Date``/``DateTime`` give ``datetime.date``/``datetime.datetime``
        (years 1..9999 only; ``to_calendar()`` and ``to_datetime64()`` offer
        explicit alternatives with their own bounds), ``Rational{T}`` gives
        ``fractions.Fraction`` (``+/-1//0`` raises; use ``to_float()``), integer
        ``Complex{T}`` gives :class:`IntegerComplex`, ``Complex{Rational{T}}``
        gives :class:`RationalComplex`, ``BigInt`` gives ``int``, ``BigFloat``
        gives the exact ``decimal.Decimal``, and numeric element and abstract
        ``Any``/``Real``/``Number``/``Integer``/``Signed``/``Unsigned``/
        ``AbstractFloat``/``Union{Int64,Float64}`` names give the converted
        Python or NumPy value. ``ValueError`` marks a value Julia's loader
        refuses (or a Python range limit); ``TypeError`` marks a marker route
        with no supported interpretation, which stays preserved for rewriting.
        """
        if self.marker is None:
            return self._stored_value()
        token = resolve_spelling(self.marker)
        handler = getattr(self, f"_interpret_{_route(token)}")
        return handler(token)

    # -- text --

    def _interpret_text(self, token: str) -> str:
        if self.kind == KIND_STRING:
            return self.to_str()
        if token == "Symbol":
            return self.to_printed()
        raise TypeError(f"Julia has no {token} conversion from a {self.julia_name} payload.")

    def _interpret_char(self, token: str) -> str:
        """``convert(Char, x)``: ``Char(UInt32(x))`` of an integer, integral float or complex."""
        if self.kind in (KIND_STRING, KIND_DATE) or self.frequency != 0:
            raise TypeError(f"Julia has no Char conversion from a {self.julia_name} payload.")
        return self.to_char()

    def to_char(self) -> str:
        """Return the one-character ``str`` of Julia's ``Char(x)`` for a numeric payload.

        Integers of any width, integral finite floats and complexes with a
        zero imaginary part convert through ``UInt32`` (``ValueError`` where
        Julia raises ``InexactError``); ``Char`` accepts code points below
        ``0x200000`` (``CodePointError`` above). Julia builds an invalid Char
        for ``0x110000..0x1fffff``, which no Python ``str`` character
        represents: that is refused explicitly, while a surrogate is returned.
        """
        source = self._source()
        values = self._carrier()
        target = Target("char", np.dtype(object))
        _interpret.check_route(source, target)
        _interpret.check_values(values, source, target)
        return str(_interpret.convert_values(values, source, target)[0])

    def to_printed(self) -> str:
        """Return Julia's ``string(value)`` of the stored family (what a ``Symbol`` marker loads).

        Integers print in decimal, floats and complex values in Julia's
        shortest round-trip notation (``1.5``, ``1.0e10``, ``NaN``, ``1.0f0 +
        2.0f0im``, ``Float16(1.5)``), MIT dates as Julia prints them
        (``2024M1``, ``7U``, and for the calendar frequencies the Date Julia
        builds in any proleptic year: ``0000-12-31``, ``-0001-12-31``,
        ``10000-01-01``) and durations as their code.
        """
        if self.kind in (KIND_INTEGER, KIND_UNSIGNED) and self.frequency == 0:
            return str(self.to_int())
        if self.kind == KIND_FLOAT:
            return _exact.julia_float_string(self._carrier()[0], self.nbytes)
        if self.kind == KIND_COMPLEX:
            real, imag = self._narrow_components()
            return _exact.julia_complex_string(real, imag, self.nbytes // 2)
        if self.kind == KIND_STRING:
            return self.to_str()
        if self.kind == KIND_DATE:
            return _printed.mit_string(self._code(), scalar_frequency(self.frequency))
        return str(self._code())

    def _narrow_components(self) -> tuple[Any, Any]:
        """Return the two stored components of a complex payload in their own width."""
        if self.nbytes == 4:
            return _exact.complexf16_from_bytes(self.payload)
        real, imag = np.frombuffer(self.payload, dtype=f"<f{self.nbytes // 2}")
        return real, imag

    # -- calendar --

    def _rata_die_ms(self, token: str) -> int:
        """Julia's ``unix2datetime`` of the payload as Rata Die milliseconds.

        Float64 payloads multiply in Float64, Float32/Float16 payloads in their
        own width, a complex payload must have a zero imaginary part
        (``InexactError`` otherwise), Int8..Int64 and UInt8..UInt32 promote
        to the Int64 product, UInt64 wraps its unsigned product and the
        128-bit widths multiply in their own width; every Int64 step
        (``1000 * n`` and ``UNIXEPOCH + ms``) wraps as Julia's defined
        integer arithmetic does. Only ``trunc(Int64, -Inf16)`` (a Float16
        product that overflowed) is refused: Julia's range check lets it
        through to an undefined conversion.
        """
        if self.kind == KIND_FLOAT:
            if self.nbytes == 8:
                return _exact.rata_die_ms_from_unix_seconds(self.to_float())
            return _exact.rata_die_ms_from_unix_narrow(self._carrier()[0], self.nbytes)
        if self.kind == KIND_COMPLEX:
            real, imag = self._narrow_components()
            if imag != 0:
                raise _julia_refuses("InexactError", token, complex(float(real), float(imag)))
            if self.nbytes == 16:
                return _exact.rata_die_ms_from_unix_seconds(float(real))
            return _exact.rata_die_ms_from_unix_narrow(real, self.nbytes // 2)
        if self.kind in (KIND_STRING, KIND_DATE):
            raise TypeError(f"Julia has no {token} conversion from a {self.julia_name} payload.")
        # A Duration is a Signed count: Int64(1000) * d multiplies its code.
        return self._rata_die_ms_of_integer()

    def _rata_die_ms_of_integer(self) -> int:
        if self.nbytes == 16:
            return _exact.rata_die_ms_from_unix_wide_integer(
                self.to_int(), signed=self.kind == KIND_INTEGER
            )
        if (self.kind, self.nbytes) == (KIND_UNSIGNED, 8):
            return _exact.rata_die_ms_from_unix_uint64(self.to_int())
        return _exact.rata_die_ms_from_unix_integer(self.to_int())

    def _date_token(self) -> str:
        token = None if self.marker is None else resolve_spelling(self.marker)
        if token not in DATE_TOKENS:
            raise TypeError("This accessor applies to Date- or DateTime-marked scalars only.")
        return token

    def to_calendar(self) -> tuple[int, ...]:
        """Return the exact calendar components Julia rebuilds, for any proleptic year.

        A ``Date`` marker gives ``(year, month, day)``; a ``DateTime`` marker
        gives ``(year, month, day, hour, minute, second, millisecond)``.
        """
        token = self._date_token()
        rata = self._rata_die_ms(token)
        if token == "Date":
            return _exact.date_from_rata_die_ms(rata)
        return _exact.civil_from_rata_die_ms(rata)

    def _civil(self, token: str) -> dt.date | dt.datetime:
        parts = self.to_calendar()
        try:
            if token == "Date":
                return dt.date(parts[0], parts[1], parts[2])
            year, month, day, hour, minute, second, millisecond = parts
            return dt.datetime(year, month, day, hour, minute, second, millisecond * 1000)
        except ValueError:
            raise ValueError(
                f"Julia rebuilds the {token} {parts}, whose year is outside Python's "
                "datetime range 1..9999; use to_calendar() or to_datetime64()."
            ) from None

    def to_date(self) -> dt.date:
        """Return the ``datetime.date`` of a ``Date``-marked scalar (years 1..9999)."""
        if self._date_token() != "Date":
            raise TypeError("to_date applies to Date-marked scalars; use to_datetime().")
        result = self._civil("Date")
        assert type(result) is dt.date
        return result

    def to_datetime(self) -> dt.datetime:
        """Return the naive ``datetime.datetime`` of a ``DateTime`` marker (years 1..9999)."""
        if self._date_token() != "DateTime":
            raise TypeError("to_datetime applies to DateTime-marked scalars; use to_date().")
        result = self._civil("DateTime")
        assert type(result) is dt.datetime
        return result

    def to_datetime64(self) -> np.datetime64:
        """Return the exact ``numpy.datetime64`` of a calendar marker.

        The unit is ``[D]`` for ``Date`` and ``[ms]`` for ``DateTime``. NumPy's
        proleptic calendar numbers years exactly as Julia does (year zero
        included), so every Julia value converts, except an instant that would
        collide with NumPy's ``NaT`` sentinel or overflow its Int64 count.
        """
        token = self._date_token()
        rata = self._rata_die_ms(token)
        if token == "Date":
            count = rata // _exact.MS_PER_DAY - _exact.UNIX_EPOCH_MS // _exact.MS_PER_DAY
            unit = "D"
        else:
            count = rata - _exact.UNIX_EPOCH_MS
            unit = "ms"
        if not MIN_INT64 < count <= MAX_INT64:
            raise ValueError(
                "This instant is NumPy's NaT sentinel or outside its Int64 count; "
                "use to_calendar()."
            )
        result: np.datetime64 = np.datetime64(count, unit)  # type: ignore[call-overload]
        return result

    # -- rationals --

    def _rational_target(self) -> str:
        token = None if self.marker is None else resolve_spelling(self.marker)
        if token == "Rational":
            if self.kind in (KIND_INTEGER, KIND_UNSIGNED) and self.frequency == 0:
                return _INTEGER_NAMES[(self.kind, self.nbytes)]  # the payload's own type
            if (self.kind, self.nbytes) in ((KIND_FLOAT, 2), (KIND_COMPLEX, 4)):
                # Rational(x::Float16) has no method; a complex payload fails its
                # imaginary-part check first.
                if self.kind == KIND_COMPLEX and self._narrow_components()[1] != 0:
                    raise _julia_refuses("InexactError", "Rational", self.to_complex())
                raise TypeError(
                    f"Julia has no Rational conversion from a {self.julia_name} payload."
                )
            if self.kind in (KIND_FLOAT, KIND_COMPLEX):
                return "Int64"  # Rational(x::Float64) and Rational(x::Float32)
            raise TypeError(f"Julia has no Rational conversion from a {self.julia_name} payload.")
        parameter = None if token is None else _exact.rational_parameter(token)
        if parameter is None:
            raise TypeError("to_fraction applies to Rational-marked scalars only.")
        return parameter

    def _ratio(self) -> tuple[int, int]:
        """Julia's rebuilt ``(numerator, denominator)`` for the Rational marker."""
        return self._ratio_of(self._rational_target())

    def _ratio_of(self, parameter: str) -> tuple[int, int]:
        """``Rational{parameter}`` of the payload: integers as ``T(n)//1``, floats rationalized."""
        target = f"Rational{{{parameter}}}"
        if self.kind == KIND_STRING:
            raise TypeError(f"Julia has no {target} conversion from a String payload.")
        if self.kind == KIND_DATE or self.frequency != 0:
            # Rational{Int64}(x::MIT) is Int64(x)//1; every other parameter has no method.
            if parameter != "Int64":
                raise TypeError(
                    f"Julia has no {target} conversion from a {self.julia_name} payload."
                )
            return self._code(), 1
        if self.kind in (KIND_INTEGER, KIND_UNSIGNED):
            return _exact.rational_from_integer(self.to_int(), parameter)
        if self.kind == KIND_FLOAT:
            return _exact.rationalize(self._carrier()[0], parameter, self.nbytes)
        real, imag = self._narrow_components()
        if imag != 0:
            raise _julia_refuses("InexactError", target, complex(float(real), float(imag)))
        return _exact.rationalize(real, parameter, self.nbytes // 2)

    def to_fraction(self) -> Fraction:
        """Return the ``fractions.Fraction`` Julia rebuilds for a ``Rational`` marker.

        Raises ``ValueError`` where Julia's loader fails (NaN, values outside
        the parameter, negative values under an unsigned parameter) and for
        ``+/-1//0``, which no Fraction can hold (``to_float()`` gives the
        signed infinity).
        """
        num, den = self._ratio()
        if den == 0:
            raise ValueError(
                f"Julia rebuilds {num}//0, a signed infinity no Fraction can hold; "
                "to_float() returns it as a float."
            )
        return Fraction(num, den)

    def to_rational_complex(self) -> RationalComplex:
        """Return the :class:`RationalComplex` Julia rebuilds for a ``Complex{Rational{T}}`` marker.

        Each component follows the ``Rational{T}`` route of its payload width
        (an integer, MIT or Duration payload contributes ``n//1`` and a zero
        imaginary part; a float payload contributes its rationalization and
        ``0//1``). A component Julia loads as ``±1//0`` raises ``ValueError``.
        """
        token = None if self.marker is None else resolve_spelling(self.marker)
        parameter = None if token is None else _exact.rational_complex_parameter(token)
        if parameter is None:
            raise TypeError("to_rational_complex applies to Complex{Rational{T}}-marked scalars.")
        if self.kind == KIND_COMPLEX:
            components = self._narrow_components()
            real = _exact.rationalize(components[0], parameter, self.nbytes // 2)
            imag = _exact.rationalize(components[1], parameter, self.nbytes // 2)
        else:
            real, imag = self._ratio_of(parameter), (0, 1)
        for num, den in (real, imag):
            if den == 0:
                raise ValueError(
                    f"Julia rebuilds a component {num}//0, a signed infinity no Fraction can "
                    "hold; to_complex() returns the stored floats."
                )
        return RationalComplex(Fraction(*real), Fraction(*imag))

    def _interpret_rationalcomplex(self, token: str) -> RationalComplex:
        return self.to_rational_complex()

    # -- integer complex --

    def _complex_parameter(self, token: str) -> str | None:
        if token == "Complex":
            if self.kind in (KIND_INTEGER, KIND_UNSIGNED) and self.frequency == 0:
                return _INTEGER_NAMES[(self.kind, self.nbytes)]
            return None  # a float or complex payload gives ComplexF{width}
        return _exact.integer_complex_parameter(token)

    def _float_components(self, target: str) -> tuple[float, float]:
        """Return the exactly widened Float64 components of a float or complex payload."""
        if self.kind == KIND_FLOAT:
            return self.to_float(), 0.0
        if self.kind == KIND_COMPLEX:
            value = self.to_complex()
            return value.real, value.imag
        raise TypeError(f"Julia has no {target} conversion from a {self.julia_name} payload.")

    def to_integer_complex(self) -> IntegerComplex:
        """Return the exact :class:`IntegerComplex` Julia rebuilds for an integer ``Complex{T}``."""
        token = None if self.marker is None else resolve_spelling(self.marker)
        parameter = None if token is None else self._complex_parameter(token)
        if parameter is None:
            raise TypeError("to_integer_complex applies to integer Complex{T}-marked scalars only.")
        target = f"Complex{{{parameter}}}"
        if self.kind == KIND_DATE or self.frequency != 0:
            # Complex{Int64}(x::MIT) is Int64(x) + 0im and Complex{Bool} needs a 0/1 code;
            # every other parameter has no method.
            if parameter not in ("Int64", "Bool"):
                raise TypeError(
                    f"Julia has no {target} conversion from a {self.julia_name} payload."
                )
            return IntegerComplex(_exact.integer_from_integer(self._code(), parameter))
        if self.kind in (KIND_INTEGER, KIND_UNSIGNED):
            pair = _exact.integer_complex_from_integer(self.to_int(), parameter)
        else:
            pair = _exact.integer_complex_from_floats(*self._float_components(target), parameter)
        return IntegerComplex(*pair)

    def _interpret_complex(self, token: str) -> Any:
        if self._complex_parameter(token) is not None:
            return self.to_integer_complex()
        # Bare Complex on a float payload is ComplexF{width}; on a complex payload identity.
        if self.kind == KIND_FLOAT:
            name = _COMPLEX_NAMES[self.nbytes * 2]
            return self._convert_element(_interpret.ACTIVE_TOKENS[name])
        if self.kind == KIND_COMPLEX:
            return self._stored_value()
        if self.kind == KIND_DATE or self.frequency != 0:
            return self.to_dated_complex()
        raise TypeError(f"Julia has no Complex conversion from a {self.julia_name} payload.")

    def to_dated_complex(self) -> DatedComplex:
        """Return the :class:`DatedComplex` Julia rebuilds under ``Complex``/``Complex{MIT{F}}``.

        ``convert(Complex, x)`` on an MIT or Duration payload is
        ``Complex(x, zero(x))``; the explicit ``Complex{MIT{F}}`` token also
        takes any integer payload (``MIT{F}(Int64(n))``, ``InexactError``
        outside Int64) and ``Complex{Duration{F}}`` an Int64 payload; every
        other family has no method (``TypeError``).
        """
        token = None if self.marker is None else resolve_spelling(self.marker)
        if token == "Complex" and (self.kind == KIND_DATE or self.frequency != 0):
            target = _interpret.bare_complex_target(self._source())
        else:
            target = None if token is None else _interpret._dated_complex_token(token)
        if target is None:
            raise TypeError(
                "to_dated_complex applies to Complex-marked date/duration scalars and to "
                "Complex{MIT{F}}/Complex{Duration{F}}-marked scalars only."
            )
        if self.kind == KIND_STRING:
            raise TypeError(f"Julia has no {target.julia_name} conversion from a String payload.")
        source = self._source()
        _interpret.check_route(source, target)
        values = self._carrier()
        _interpret.check_values(values, source, target)
        converted = _interpret.convert_values(values, source, target)
        assert target.frequency is not None
        family = MIT if target.parameter == "MIT" else Duration
        return DatedComplex(
            family(target.frequency, int(converted["re"][0])),
            family(target.frequency, int(converted["im"][0])),
        )

    def _interpret_datedcomplex(self, token: str) -> DatedComplex:
        return self.to_dated_complex()

    # -- abstract numeric names --

    def _numeric_only(self, token: str) -> None:
        if self.kind in (KIND_STRING, KIND_DATE) or self.frequency != 0:
            raise TypeError(f"Julia has no {token} conversion from a {self.julia_name} payload.")

    def _real_part(self, token: str) -> StoredScalar:
        """Return the real component of a complex payload in its own width (imaginary part zero)."""
        real, imag = self._float_components(token)
        if imag != 0:
            raise _julia_refuses("InexactError", token, complex(real, imag))
        width = self.nbytes // 2
        # Julia extracts the real component without arithmetic. Repacking it
        # through Python float would quiet a Float32 signaling NaN.
        return StoredScalar(self.payload[:width], KIND_FLOAT)

    def _interpret_abstract(self, token: str) -> Any:
        if token == "Any":
            return self._stored_value()
        if self.kind == KIND_STRING:
            raise TypeError(f"Julia has no {token} conversion from a String payload.")
        if self.kind == KIND_DATE or self.frequency != 0:
            return self._interpret_abstract_date(token)
        if token in UNION_TOKENS:
            return self._interpret_union(token)
        if token in ("Real", "AbstractFloat"):
            return self._interpret_real_kind(token)
        if token == "Number":
            return self._stored_value()
        return self._interpret_integer_kind(token)

    def _interpret_real_kind(self, token: str) -> Any:
        """``Real`` is an identity on real payloads; ``AbstractFloat`` makes integers Float64."""
        if self.kind == KIND_COMPLEX:
            return self._real_part(token)._stored_value()
        if token == "Real" or self.kind == KIND_FLOAT:
            return self._stored_value()
        return float(self.to_int())  # Float64(n): nearest-even, as Julia's convert

    def _interpret_union(self, token: str) -> Any:
        """``Union{Int64,Float64}``: identities, plus ComplexF64 with a zero imaginary part."""
        if (self.kind, self.nbytes) in ((KIND_INTEGER, 8), (KIND_FLOAT, 8)):
            return self._stored_value()
        if (self.kind, self.nbytes) == (KIND_COMPLEX, 16) and self.to_complex().imag == 0:
            return self._real_part(token)._stored_value()
        raise TypeError(f"Julia has no {token} conversion from a {self.julia_name} payload.")

    def _interpret_abstract_date(self, token: str) -> Any:
        """``MIT``/``Duration`` are ``Signed``: identities except the two conversions Julia has."""
        if token in ("Number", "Real", "Integer", "Signed"):
            return self._stored_value()
        if token == "AbstractFloat":
            # Float64(x::MIT) is the plotting value; Float64(x::Duration) the count per year.
            return self._convert_element(_interpret.ACTIVE_TOKENS["Float64"])
        raise TypeError(f"Julia has no {token} conversion from a {self.julia_name} payload.")

    def _interpret_integer_kind(self, token: str) -> Any:
        """Apply ``Integer``, ``Signed`` or ``Unsigned``.

        Integer payloads are identities or a value-checked same-width sign
        change; float and complex payloads with a zero imaginary part give
        Int64/UInt64 when integral.
        """
        if self.kind in (KIND_INTEGER, KIND_UNSIGNED):
            name = _INTEGER_NAMES[(self.kind, self.nbytes)]
            if token == "Integer" or (token == "Signed") == (self.kind == KIND_INTEGER):
                return self._stored_value()
            name = _SIGNED_OF[name] if token == "Signed" else _UNSIGNED_OF[name]
            return self._convert_element(_interpret.ACTIVE_TOKENS[name])
        name = "UInt64" if token == "Unsigned" else "Int64"
        real, imag = self._float_components(token)
        if imag != 0:
            raise _julia_refuses("InexactError", name, complex(real, imag))
        number = _exact.integer_from_float(real, name)
        return number if name == "Int64" else np.uint64(number)

    def _interpret_rational(self, token: str) -> Fraction:
        return self.to_fraction()

    def _interpret_bigfloat(self, token: str) -> Decimal:
        return self.to_decimal()

    def to_decimal(self) -> Decimal:
        """Return the exact ``decimal.Decimal`` of the ``BigFloat`` Julia rebuilds.

        ``BigFloat(x)`` is exact for every bit-integer and IEEE float payload
        at Julia's default precision, so the loaded number is the payload's
        own value; a complex payload needs a zero imaginary part and a date or
        duration payload has no ``BigFloat`` method. Arithmetic in Julia's
        arbitrary precision is not reproduced; this is the loaded value only.
        """
        self._numeric_only("BigFloat")
        if self.kind in (KIND_INTEGER, KIND_UNSIGNED):
            return _exact.exact_decimal(self.to_int())
        if self.kind == KIND_COMPLEX:
            real, imag = self._narrow_components()
            if imag != 0:
                raise _julia_refuses("InexactError", "BigFloat", complex(float(real), float(imag)))
            return _exact.exact_decimal(float(real))
        return _exact.exact_decimal(self.to_float())

    def _interpret_date(self, token: str) -> dt.date | dt.datetime:
        return self._civil(token)

    def _interpret_bigint(self, token: str) -> int:
        self._numeric_only("BigInt")
        if self.kind in (KIND_INTEGER, KIND_UNSIGNED):
            return self.to_int()
        if self.kind == KIND_COMPLEX and self.nbytes < 16:
            real, imag = self._narrow_components()
            if imag != 0:
                raise _julia_refuses("InexactError", "BigInt", complex(float(real), float(imag)))
            return self._big_of(real)
        real, imag = self._float_components("BigInt")
        if imag != 0:
            raise _julia_refuses("InexactError", "BigInt", complex(real, imag))
        return self._big_of(real)

    @staticmethod
    def _big_of(real: Any) -> int:
        value = float(real)
        if not math.isfinite(value) or value != math.floor(value):
            raise _julia_refuses("InexactError", "BigInt", value)
        return int(value)

    # -- numeric element tokens through the shared series kernels --

    def _interpret_element(self, token: str) -> Any:
        target = _interpret.resolve_token(token)
        if target is None:
            raise _unsupported("This scalar reconstruction marker has no supported interpretation")
        if target.kind in ("date", "duration") and self.kind == KIND_COMPLEX:
            value = self.to_complex()
            if value.imag != 0:
                # Scalar construction checks the imaginary component before
                # dispatching the real value to the date/duration constructor.
                raise _julia_refuses("InexactError", target.julia_name, value)
        return self._convert_element(target)

    def _convert_element(self, target: Target) -> Any:
        if self.kind == KIND_STRING:
            raise TypeError(f"Julia has no {target.julia_name} conversion from a String payload.")
        source = self._source()
        _interpret.check_route(source, target)
        values = self._carrier()
        _interpret.check_values(values, source, target)
        converted = _interpret.convert_values(values, source, target)
        return _target_value(converted, target)


_FIXED_ROUTES: dict[str, str] = {
    **dict.fromkeys(TEXT_TOKENS, "text"),
    **dict.fromkeys(DATE_TOKENS, "date"),
    **dict.fromkeys(ABSTRACT_TOKENS, "abstract"),
    "BigInt": "bigint",
    "BigFloat": "bigfloat",
    "Rational": "rational",
    "Complex": "complex",
    "Char": "char",
}


def _route(token: str) -> str:
    """Classify an exact token into its finite scalar route (``"element"`` for the rest)."""
    route = _FIXED_ROUTES.get(token)
    if route is not None:
        return route
    if _exact.rational_parameter(token) is not None:
        return "rational"
    if _exact.integer_complex_parameter(token) is not None:
        return "complex"
    if _exact.rational_complex_parameter(token) is not None:
        return "rationalcomplex"
    if _interpret._dated_complex_token(token) is not None:
        return "datedcomplex"
    return "element"


def _target_value(values: np.ndarray[Any, Any], target: Target) -> Any:
    """Return the scalar result of a one-element carrier of ``target``'s family."""
    if target.kind == "numeric":
        if target.is_bool:
            return bool(values[0])
        return plain_numeric(target.native_kind, values.tobytes())
    if target.kind in ("int128", "uint128"):
        return _interpret.unpack_words(values, target.kind == "int128")[0]
    if target.kind == "complexf16":
        return complex(float(values["real"][0]), float(values["imag"][0]))
    code = int(values[0])
    assert target.frequency is not None
    # convert(MIT{F}, n) and convert(Duration{F}, n) reinterpret the Int64 code
    # without the native date codec, so no reliable-window check applies here;
    # writing such a date back is the scalar writer's own decision.
    if target.kind == "date":
        return MIT(target.frequency, code)
    return Duration(target.frequency, code)


def _ints(*values: object) -> tuple[int, ...]:
    for value in values:
        if type(value) is not int:
            raise TypeError("Calendar components must be exact Python int objects.")
    return values  # type: ignore[return-value]


def encode_text(value: str) -> bytes:
    """UTF-8 plus the NUL terminator; NUL, lone surrogates and oversize text are refused."""
    if "\0" in value:
        raise ValueError(
            "DataEcon strings cannot contain NUL; the native format is NUL-terminated."
        )
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(
            "DataEcon strings must be encodable as UTF-8 (no lone surrogates)."
        ) from None
    if len(encoded) >= MAX_BYTES:
        raise ValueError("DataEcon string exceeds the payload size limit.")
    return encoded + b"\0"


def is_stored_form(kind: int, payload: bytes, marker: str | None) -> bool:
    """Whether a read must return a ``StoredScalar`` rather than a plain value."""
    if marker is not None or (kind, len(payload)) in WIDE_WIDTHS:
        return True
    return kind == KIND_STRING and not is_plain_text(payload)
