# SPDX-License-Identifier: MIT
"""Scalar and series conversions between core objects and native DataEcon payloads.

Scalars: Float16/32/64, Int8/16/32/64, UInt8/16/32/64, Complex64/128, UTF-8
strings, and MIT dates or Durations over the unit, daily, business-daily,
weekly (every end day), monthly, quarterly, half-yearly and annual
frequencies. Series: Float64 values over the same frequencies except Unit
(daily, business-daily and weekly axes use the verified calendar windows).
"""

from __future__ import annotations

import struct
import sys
from typing import Any, TypeAlias

import numpy as np

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
from tsecon.mit import MIT, Duration
from tsecon.tseries import TSeries

# Native sqlite3_bind_blob takes a C int despite daec.h accepting int64_t.
# Bound both allocations and integer conversions well below that limit.
MAX_BYTES = 128 * 1024 * 1024
MIN_DATE = -(2**31)
MAX_DATE = 2**31 - 1
MIN_MONTHLY_DATE = -393600
MIN_QUARTERLY_DATE = -131200
MIN_HALFYEARLY_DATE = -65600
_FREQUENCIES: dict[int, Monthly | Quarterly | HalfYearly | Yearly] = {
    32: Monthly(),
    65: Quarterly(1),
    66: Quarterly(2),
    67: Quarterly(3),
    **{128 + month: HalfYearly(month) for month in range(1, 7)},
    **{256 + month: Yearly(month) for month in range(1, 13)},
}
# The native decoder adds EPOCH_L * periods_per_year in uint32 arithmetic and
# then divides; below these codes the wrapped sum decodes to a different year.
# Annual (division by one) preserves the whole signed 32-bit range. Series
# writes already reject monthly codes below -393600 through the native
# round-trip check; scalar dates apply the explicit table before any C call.
_MIN_DATES: dict[int, tuple[int, str]] = {
    **dict.fromkeys((65, 66, 67), (MIN_QUARTERLY_DATE, "quarterly")),
    **dict.fromkeys(range(129, 135), (MIN_HALFYEARLY_DATE, "half-yearly")),
}
_SCALAR_MIN_DATES: dict[int, int] = {
    32: MIN_MONTHLY_DATE,
    **{code: minimum for code, (minimum, _) in _MIN_DATES.items()},
    **dict.fromkeys(range(257, 269), MIN_DATE),
}
MIN_INT64 = -(2**63)
MAX_INT64 = 2**63 - 1
# Scalar-only frequencies. Unit (11) is Julia's generic pass-through: the code
# is a plain signed 64-bit integer and no native codec applies. Daily (12),
# business daily (13) and weekly (16 + ISO end day, 17..23) use the native
# calendar codec, whose encoder accepts years in [-32800, 32800] and whose
# decoders shift by fixed uint32 constants. The exact round-trip windows below
# were verified natively on every code: daily from 1 March -32800, business
# daily and weekly from the last week of December -32800, all through 31
# December 32800. Code 16 (a second Sunday alias) and 24..31 are never written
# by Julia and decode to invalid anchors, so they are rejected like bare codes.
UNIT_FREQUENCY = 11
_CALENDAR_FREQUENCIES: dict[int, Daily | BDaily | Weekly] = {
    12: Daily(),
    13: BDaily(),
    **{16 + day: Weekly(day) for day in range(1, 8)},
}
_CALENDAR_RANGES: dict[int, tuple[int, int]] = {
    12: (-11980259, 11979954),
    13: (-8557114, 8557110),
    **dict.fromkeys(range(17, 24), (-1711422, 1711422)),
}
# Series axes: the year/period families plus the calendar families. Unit series
# (code 11) are not yet supported; Julia writes them as a plain Int64 axis.
_SERIES_FREQUENCIES: dict[int, Frequency] = {**_FREQUENCIES, **_CALENDAR_FREQUENCIES}
_SCALAR_FREQUENCIES: dict[int, Frequency] = {
    **_FREQUENCIES,
    UNIT_FREQUENCY: Unit(),
    **_CALENDAR_FREQUENCIES,
}
# Native scalar type codes: type_integer (which the header also names
# type_signed) is 1, type_unsigned 2, type_date 3, type_float 4, type_complex 5
# and type_string 6. Type 1 with a supported frequency is a Duration; type 3
# always carries a frequency. Julia reloads types 1, 2, 4 and 5 by byte width
# alone (frequency zero) and stores every width without a marker.
KIND_INTEGER = 1
KIND_UNSIGNED = 2
KIND_DATE = 3
KIND_FLOAT = 4
KIND_COMPLEX = 5
KIND_STRING = 6
# Supported numeric widths per type code. Int128/UInt128 (16 bytes) and
# ComplexF16 (4 bytes) are Julia-supported but have no NumPy scalar type; they
# stay rejected until a representation is chosen.
_NUMERIC_WIDTHS: dict[int, frozenset[int]] = {
    KIND_INTEGER: frozenset({1, 2, 4, 8}),
    KIND_UNSIGNED: frozenset({1, 2, 4, 8}),
    KIND_FLOAT: frozenset({2, 4, 8}),
    KIND_COMPLEX: frozenset({8, 16}),
}
# Read results by (type, nbytes): eight-byte Int64/Float64 and sixteen-byte
# ComplexF64 return the exact-width built-ins; every other width returns the
# sized NumPy scalar class (np.dtype("<i4").type is np.int32 on every platform).
_NUMPY_RESULTS: dict[tuple[int, int], np.dtype[Any]] = {
    (KIND_INTEGER, 1): np.dtype("<i1"),
    (KIND_INTEGER, 2): np.dtype("<i2"),
    (KIND_INTEGER, 4): np.dtype("<i4"),
    (KIND_UNSIGNED, 1): np.dtype("<u1"),
    (KIND_UNSIGNED, 2): np.dtype("<u2"),
    (KIND_UNSIGNED, 4): np.dtype("<u4"),
    (KIND_UNSIGNED, 8): np.dtype("<u8"),
    (KIND_FLOAT, 2): np.dtype("<f2"),
    (KIND_FLOAT, 4): np.dtype("<f4"),
    (KIND_COMPLEX, 8): np.dtype("<c8"),
}
# Exact NumPy scalar classes accepted on write, discovered through the dtype
# type codes (signed b/h/i/l/q/p, unsigned B/H/I/L/Q/P, float e/f/d, complex
# F/D) rather than through attribute names: the C-named classes (intc, uintc,
# long, ulong, longlong, ulonglong) are distinct from the sized names on some
# platforms (Windows: intc/uintc; Linux/macOS: longlong/ulonglong) and the same
# object on others, and NumPy 1.26 does not expose the names ``np.long`` and
# ``np.ulong`` at all. Every supported NumPy version defines the type codes, so
# the table is platform- and version-independent; each class maps to its
# dtype's type code and byte width. Lookups must compare classes by identity
# (see ``_numpy_class``): a dictionary membership test would consult the
# candidate's metaclass ``__eq__``/``__hash__``, which a subclass can spoof.
_DTYPE_KINDS = {"i": KIND_INTEGER, "u": KIND_UNSIGNED, "f": KIND_FLOAT, "c": KIND_COMPLEX}
_NUMPY_TYPE_CODES = "bhilqpBHILQPefdFD"
_NUMPY_CLASSES: dict[type, tuple[int, int]] = {
    np.dtype(code).type: (_DTYPE_KINDS[np.dtype(code).kind], np.dtype(code).itemsize)
    for code in _NUMPY_TYPE_CODES
}
_NUMPY_CLASSES_BY_ID: dict[int, tuple[type, tuple[int, int]]] = {
    id(cls): (cls, entry) for cls, entry in _NUMPY_CLASSES.items()
}
if any(width not in _NUMERIC_WIDTHS[kind] for kind, width in _NUMPY_CLASSES.values()):
    raise ImportError("NumPy defines a scalar width outside the DataEcon width table.")


def _numpy_class(cls: type) -> tuple[int, int] | None:
    """Return the (type code, width) of an accepted NumPy scalar class, by identity only."""
    entry = _NUMPY_CLASSES_BY_ID.get(id(cls))
    if entry is None or entry[0] is not cls:
        return None
    return entry[1]


Metadata: TypeAlias = tuple[int, int, int, int, int, int, int, int, int]
ScalarMetadata: TypeAlias = tuple[int, int, int, int]
ScalarValue: TypeAlias = (
    float
    | np.float64
    | np.float32
    | np.float16
    | int
    | np.int64
    | np.int32
    | np.int16
    | np.int8
    | np.longlong
    | np.intc
    | np.uint64
    | np.uint32
    | np.uint16
    | np.uint8
    | np.ulonglong
    | np.uintc
    | complex
    | np.complex128
    | np.complex64
    | str
    | MIT
    | Duration
)
ScalarResult: TypeAlias = (
    float
    | np.float32
    | np.float16
    | int
    | np.int32
    | np.int16
    | np.int8
    | np.uint64
    | np.uint32
    | np.uint16
    | np.uint8
    | complex
    | np.complex64
    | str
    | MIT
    | Duration
)
_SCALAR_SUPPORT = (
    "DataEcon scalar support covers Float16/32/64, Int8/16/32/64, UInt8/16/32/64, "
    "Complex64/128, strings, and MIT dates or Durations over the Unit, Daily, BDaily, "
    "Weekly, Monthly, Quarterly, HalfYearly and Yearly frequencies."
)
_SERIES_SUPPORT = (
    "DataEcon supports only monthly, quarterly, half-yearly, annual, daily, "
    "business-daily or weekly Float64 TSeries."
)


def series_frequency(code: int) -> Frequency:
    """Resolve only canonical supported native series frequency codes."""
    try:
        return _SERIES_FREQUENCIES[code]
    except KeyError:
        raise TypeError(_SERIES_SUPPORT) from None


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


def validate_scalar_metadata(metadata: ScalarMetadata) -> None:
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
    if kind not in _NUMERIC_WIDTHS:
        raise TypeError(_SCALAR_SUPPORT)
    if frequency != 0:
        raise TypeError("DataEcon numeric scalars carry no frequency.")
    if nbytes not in _NUMERIC_WIDTHS[kind]:
        raise ValueError(
            "DataEcon numeric scalar payload width is not supported for its type; "
            "supported widths are eight bytes for Int64/Float64, 1/2/4 for narrower "
            "integers, 2/4 for narrower floats and 8/16 for complex values."
        )


def _encode_text(value: str) -> bytes:
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


def encode_scalar(value: ScalarValue) -> tuple[int, int, bytes]:
    """Return the native type code, frequency code and an owned payload snapshot.

    Accepted by exact class: Python float/NumPy float64 (Float64), NumPy
    float32/float16, Python int (range-checked)/NumPy int64 (Int64), NumPy
    int8/int16/int32, NumPy uint8/uint16/uint32/uint64, Python complex/NumPy
    complex128 (ComplexF64), NumPy complex64, Python str, core MIT and core
    Duration. NumPy scalars are stored at their own width from their own
    little-endian bytes, so nothing is widened and integers never pass through
    floating point; strings are UTF-8 plus a NUL terminator; dates must lie
    inside the reliable native range of their frequency (Unit dates are
    unbounded 64-bit codes). Subclasses, arrays, bytes, bool and every other
    type are rejected without conversion.
    """
    if sys.byteorder != "little":
        raise RuntimeError("DataEcon interchange requires a little-endian host.")
    if type(value) is float or type(value) is np.float64:
        kind, frequency, packed = KIND_FLOAT, 0, struct.pack("<d", value)
    elif type(value) is int or type(value) is np.int64:
        number = int(value)
        if not MIN_INT64 <= number <= MAX_INT64:
            raise ValueError("write_scalar integers must fit the signed 64-bit range.")
        kind, frequency, packed = KIND_INTEGER, 0, struct.pack("<q", number)
    elif type(value) is complex or type(value) is np.complex128:
        kind, frequency, packed = KIND_COMPLEX, 0, struct.pack("<dd", value.real, value.imag)
    elif isinstance(value, np.generic) and (numpy_class := _numpy_class(type(value))):
        # Identity lookup: a subclass whose metaclass compares equal to an
        # accepted class is rejected below without calling its tobytes().
        kind, width = numpy_class
        frequency, packed = 0, value.tobytes()
        if len(packed) != width:
            raise ValueError("NumPy scalar bytes do not match its declared width.")
    elif type(value) is str:
        kind, frequency, packed = KIND_STRING, 0, _encode_text(value)
    elif type(value) is MIT:
        frequency = scalar_frequency_code(value.frequency)
        validate_date_code(frequency, value.value)
        kind, packed = KIND_DATE, struct.pack("<q", value.value)
    elif type(value) is Duration:
        frequency = scalar_frequency_code(value.frequency)
        if not MIN_INT64 <= value.value <= MAX_INT64:
            raise ValueError("write_scalar durations must fit the signed 64-bit range.")
        kind, packed = KIND_INTEGER, struct.pack("<q", value.value)
    else:
        raise TypeError(
            "write_scalar requires a Python float, int, complex or str, an exact NumPy "
            "float16/32/64, int8/16/32/64, uint8/16/32/64 or complex64/128 scalar, "
            "an MIT or a Duration."
        )
    return kind, frequency, packed


def decode_scalar(kind: int, frequency: int, payload: bytes) -> ScalarResult:
    """Return an independent core value from validated metadata and a byte snapshot."""
    validate_scalar_metadata((1, kind, frequency, len(payload)))
    if sys.byteorder != "little":
        raise RuntimeError("DataEcon interchange requires a little-endian host.")
    if kind == KIND_STRING:
        if payload[-1] != 0 or b"\0" in payload[:-1]:
            raise ValueError("DataEcon string payload is not a single NUL-terminated string.")
        try:
            return payload[:-1].decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("DataEcon string payload is not valid UTF-8.") from None
    if (frequency == 0 and kind != KIND_INTEGER) or len(payload) != 8:
        # Float, complex and every narrow width; Int64 and dates/durations follow.
        return _decode_numeric(kind, payload)
    number = int(struct.unpack("<q", payload)[0])
    if frequency == 0:
        return number
    if kind == KIND_DATE:
        validate_date_code(frequency, number)
        return MIT(scalar_frequency(frequency), number)
    return Duration(scalar_frequency(frequency), number)


def _decode_numeric(kind: int, payload: bytes) -> ScalarResult:
    """Decode a validated frequency-free numeric payload other than Int64."""
    if kind == KIND_FLOAT and len(payload) == 8:
        return float(struct.unpack("<d", payload)[0])
    if kind == KIND_COMPLEX and len(payload) == 16:
        real, imag = struct.unpack("<dd", payload)
        return complex(real, imag)
    # The NumPy scalar copies its bytes out of the payload snapshot.
    result: np.generic = np.frombuffer(payload, dtype=_NUMPY_RESULTS[(kind, len(payload))])[0]
    return result  # type: ignore[return-value]


def validate_metadata(metadata: Metadata) -> None:
    """Validate native metadata before the wrapper dereferences a value pointer."""
    cls, kind, element, element_freq, axis, length, frequency, first, nbytes = metadata
    series_frequency(frequency)
    if (cls, kind, element, element_freq, axis) != (2, 12, 4, 0, 1):
        raise TypeError(_SERIES_SUPPORT)
    if length < 0:
        raise ValueError("Invalid negative DataEcon series length.")
    if nbytes != length * 8 or not 0 <= nbytes <= MAX_BYTES:
        raise ValueError("Invalid or oversized DataEcon series payload.")
    if not MIN_DATE <= first <= MAX_DATE or (length and first + length - 1 > MAX_DATE):
        raise ValueError("DataEcon dates must fit the native signed 32-bit date range.")
    if frequency in _CALENDAR_RANGES:
        # Only the stored first date must lie inside the verified calendar
        # window: Julia packs nothing else and never decodes a trailing date.
        # Later observations are implicit consecutive codes, bounded above by
        # the signed 32-bit check and the payload limit (at most 16,777,216
        # values), so the last code stays below maximum + 16,777,216 and never
        # reaches a native calendar call.
        minimum, maximum = _CALENDAR_RANGES[frequency]
        if not minimum <= first <= maximum:
            raise ValueError("Date is outside the reliable native date range for its frequency.")
    elif frequency in _MIN_DATES and first < _MIN_DATES[frequency][0]:
        raise ValueError(
            f"Date is outside the reliable native {_MIN_DATES[frequency][1]} date range."
        )


def encode_series(series: TSeries) -> tuple[int, int, bytes]:
    """Return the frequency code, first date code and an independent Float64 snapshot."""
    if not isinstance(series, TSeries):
        raise TypeError("write_series requires a TSeries.")
    if sys.byteorder != "little":
        raise RuntimeError(
            "DataEcon interchange is currently supported on little-endian hosts only."
        )
    code = next(
        (code for code, freq in _SERIES_FREQUENCIES.items() if freq == series.frequency), None
    )
    if code is None or series.values.dtype != np.dtype(np.float64):
        raise TypeError(_SERIES_SUPPORT + " Values must be native-endian float64.")
    length = len(series.values)
    first = series.firstdate.value
    validate_metadata((2, 12, 4, 0, 1, length, code, first, length * 8))
    return code, first, series.values.tobytes(order="C")


def decode_series(code: int, first: int, payload: bytes) -> TSeries:
    """Construct an owning core series; no native storage escapes the adapter."""
    if sys.byteorder != "little":
        raise RuntimeError(
            "DataEcon interchange is currently supported on little-endian hosts only."
        )
    values = np.frombuffer(payload, dtype=np.float64).copy()
    return TSeries(MIT(series_frequency(code), first), values)
