# SPDX-License-Identifier: MIT
"""Scalar and series conversions between core objects and native DataEcon payloads.

Scalars: Float16/32/64, Int8/16/32/64, UInt8/16/32/64, Complex64/128, UTF-8
strings, and MIT dates or Durations over the unit, daily, business-daily,
weekly (every end day), monthly, quarterly, half-yearly and annual
frequencies. Series also support represented MIT/Duration, Int128/UInt128 and
ComplexF16 elements, over the same axis frequencies except Unit (daily,
business-daily and weekly axes use the verified calendar windows).
"""

from __future__ import annotations

import struct
import sys
from typing import Any, NamedTuple, TypeAlias

import numpy as np

from tsecon.frequencies import (
    Frequency,
)
from tsecon.mit import MIT, Duration
from tsecon.tseries import TSeries

from ._metadata import (
    _CALENDAR_FREQUENCIES,
    _CALENDAR_RANGES,
    _FREQUENCIES,
    _MIN_DATES,
    _SCALAR_FREQUENCIES,
    _SCALAR_MIN_DATES,
    _SERIES_FREQUENCIES,
    KIND_COMPLEX,
    KIND_DATE,
    KIND_FLOAT,
    KIND_INTEGER,
    KIND_STRING,
    KIND_UNSIGNED,
    MAX_BYTES,
    MAX_DATE,
    MAX_INT64,
    MIN_DATE,
    MIN_HALFYEARLY_DATE,
    MIN_INT64,
    MIN_MONTHLY_DATE,
    MIN_QUARTERLY_DATE,
    UNIT_FREQUENCY,
)
from ._represented import (
    COMPLEXF16,
    INT128,
    UINT128,
    StoredElement,
    StoredSeries,
    _check_bool_interpretation,
)

__all__ = [
    "KIND_COMPLEX",
    "KIND_DATE",
    "KIND_FLOAT",
    "KIND_INTEGER",
    "KIND_STRING",
    "KIND_UNSIGNED",
    "MAX_BYTES",
    "MAX_DATE",
    "MAX_INT64",
    "MIN_DATE",
    "MIN_HALFYEARLY_DATE",
    "MIN_INT64",
    "MIN_MONTHLY_DATE",
    "MIN_QUARTERLY_DATE",
    "UNIT_FREQUENCY",
    "_CALENDAR_FREQUENCIES",
    "_CALENDAR_RANGES",
    "_FREQUENCIES",
    "_MIN_DATES",
    "_SCALAR_FREQUENCIES",
    "_SCALAR_MIN_DATES",
    "_SERIES_FREQUENCIES",
]

# Scalar numeric widths per type code. Int128/UInt128 (16 bytes) and
# ComplexF16 (4 bytes) remain unsupported for scalars; represented series
# accept them through the separate _SERIES_WIDTHS table below.
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
    bool
    | np.bool_
    | float
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
    "Complex64/128, Booleans stored as Int8, strings, and MIT dates or Durations "
    "over the Unit, Daily, BDaily, "
    "Weekly, Monthly, Quarterly, HalfYearly and Yearly frequencies."
)
_SERIES_SUPPORT = (
    "DataEcon supports only monthly, quarterly, half-yearly, annual, daily, "
    "business-daily or weekly numeric/Boolean TSeries or represented StoredSeries."
)

# Empty payloads have no byte width; only an exact reconstruction token or the
# native kind's wide default can establish the element dtype.
_SERIES_TYPES: dict[str, tuple[int, np.dtype[Any]]] = {
    "Int8": (1, np.dtype("i1")),
    "Int16": (1, np.dtype("<i2")),
    "Int32": (1, np.dtype("<i4")),
    "Int64": (1, np.dtype("<i8")),
    "UInt8": (2, np.dtype("u1")),
    "UInt16": (2, np.dtype("<u2")),
    "UInt32": (2, np.dtype("<u4")),
    "UInt64": (2, np.dtype("<u8")),
    "Float16": (4, np.dtype("<f2")),
    "Float32": (4, np.dtype("<f4")),
    "Float64": (4, np.dtype("<f8")),
    "ComplexF32": (5, np.dtype("<c8")),
    "ComplexF64": (5, np.dtype("<c16")),
    "Bool": (1, np.dtype("?")),
}
_SERIES_DEFAULTS = {1: "Int64", 2: "UInt64", 4: "Float64", 5: "ComplexF64"}
_SERIES_DTYPES = {
    (kind, dtype.itemsize): dtype for kind, dtype in _SERIES_TYPES.values() if dtype.kind != "b"
}


class SeriesPayload(NamedTuple):
    """Owned bytes and explicit storage information for the native boundary."""

    frequency: int
    first: int
    payload: bytes
    element: int
    element_frequency: int
    length: int
    marker: str | None


# Series widths include the represented families; scalar acceptance is unchanged.
_SERIES_WIDTHS = {
    **_NUMERIC_WIDTHS,
    KIND_INTEGER: _NUMERIC_WIDTHS[KIND_INTEGER] | {16},
    KIND_UNSIGNED: _NUMERIC_WIDTHS[KIND_UNSIGNED] | {16},
    KIND_COMPLEX: _NUMERIC_WIDTHS[KIND_COMPLEX] | {4},
}
_WIDE_ELEMENTS = {(1, 16): INT128, (2, 16): UINT128, (5, 4): COMPLEXF16}
_WIDE_NAMES = {e.julia_name: e for e in _WIDE_ELEMENTS.values()}
SeriesValue: TypeAlias = TSeries | StoredSeries


def series_dtype(
    element: int, element_frequency: int, length: int, nbytes: int, marker: str | None
) -> np.dtype[Any] | StoredElement:
    """Resolve structurally validated metadata using finite marker comparisons."""
    if marker is not None and type(marker) is not str:
        raise TypeError("Unsupported Julia reconstruction attribute.")
    if element_frequency:
        descriptor = StoredElement(
            "date" if element == KIND_DATE else "duration", scalar_frequency(element_frequency)
        )
        if marker not in (None, descriptor.julia_name):
            raise TypeError(
                "Unsupported Julia reconstruction attribute for date/duration elements."
            )
        return descriptor
    if not length:
        return _empty_series_type(element, marker)
    width = nbytes // length
    wide = _WIDE_ELEMENTS.get((element, width))
    if wide is not None:
        if marker == "Bool":
            return wide.with_bool_marker()
        if marker not in (None, wide.julia_name):
            raise TypeError("Unsupported Julia reconstruction attribute for this element encoding.")
        return wide
    if marker not in (None, "Bool"):
        raise TypeError("Unsupported Julia reconstruction attribute for this element encoding.")
    # Keep the carrier dtype until Bool values have been checked without casting.
    return _SERIES_DTYPES[element, width]


def _empty_series_type(element: int, marker: str | None) -> np.dtype[Any] | StoredElement:
    if marker == "Bool":
        return np.dtype("?")
    if marker in _WIDE_NAMES:
        descriptor = _WIDE_NAMES[marker]
        if descriptor.native_kind != element:
            raise TypeError("Unsupported Julia reconstruction attribute for this element encoding.")
        return descriptor
    return _empty_series_dtype(element, marker)


def _empty_series_dtype(element: int, marker: str | None) -> np.dtype[Any]:
    if marker is None:
        return _SERIES_TYPES[_SERIES_DEFAULTS[element]][1]
    if marker not in _SERIES_TYPES or _SERIES_TYPES[marker][0] != element:
        raise TypeError("Unsupported Julia reconstruction attribute for this element encoding.")
    return _SERIES_TYPES[marker][1]


def validate_series_payload(
    element: int, element_frequency: int, length: int, payload: bytes, marker: str | None
) -> np.dtype[Any] | StoredElement:
    """Validate interpretation of an owned, structurally checked payload."""
    resolved = series_dtype(element, element_frequency, length, len(payload), marker)
    if isinstance(resolved, StoredElement):
        if resolved.marker is not None:
            _check_bool_interpretation(np.frombuffer(payload, dtype=resolved.dtype), resolved)
    elif marker == "Bool":
        values = np.frombuffer(payload, dtype=resolved)
        if not np.all((values == 0) | (values == 1)):
            raise ValueError("Boolean series payload must contain only zero and one values.")
    return resolved


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


def _encode_int64(value: int) -> bytes:
    if not MIN_INT64 <= value <= MAX_INT64:
        raise ValueError("write_scalar integers must fit the signed 64-bit range.")
    return struct.pack("<q", value)


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
    unbounded 64-bit codes). Exact Python/NumPy Booleans store canonical Int8
    zero or one, like Julia, and read back as np.int8. Subclasses, arrays, bytes and every other
    type are rejected without conversion.
    """
    if sys.byteorder != "little":
        raise RuntimeError("DataEcon interchange requires a little-endian host.")
    if type(value) is bool or type(value) is np.bool_:
        # Store the truth value as Julia's canonical Int8 zero or one.
        kind, frequency, packed = KIND_INTEGER, 0, b"\x01" if value else b"\x00"
    elif type(value) is float or type(value) is np.float64:
        kind, frequency, packed = KIND_FLOAT, 0, struct.pack("<d", value)
    elif type(value) is int or type(value) is np.int64:
        kind, frequency, packed = KIND_INTEGER, 0, _encode_int64(int(value))
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
            "write_scalar requires a Python bool, float, int, complex or str, "
            "an exact NumPy bool_, "
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
    if (cls, kind, axis) != (2, 12, 1):
        raise TypeError(_SERIES_SUPPORT)
    widths = _series_widths(element, element_freq)
    if length < 0:
        raise ValueError("Invalid negative DataEcon series length.")
    if not 0 <= nbytes <= MAX_BYTES or (not length and nbytes):
        raise ValueError("Invalid or oversized DataEcon series payload.")
    if length and (nbytes % length or nbytes // length not in widths):
        raise ValueError("Invalid DataEcon series element width or payload length.")
    if not MIN_DATE <= first <= MAX_DATE or (length and first + length - 1 > MAX_DATE):
        raise ValueError("DataEcon dates must fit the native signed 32-bit date range.")
    if frequency in _CALENDAR_RANGES:
        # Only the stored first date must lie inside the verified calendar
        # window: Julia packs nothing else and never decodes a trailing date.
        # Later observations are implicit consecutive codes, bounded above by
        # the signed 32-bit check and the payload limit (at most 134,217,728
        # one-byte values), so the last code stays below maximum + 134,217,728 and never
        # reaches a native calendar call.
        minimum, maximum = _CALENDAR_RANGES[frequency]
        if not minimum <= first <= maximum:
            raise ValueError("Date is outside the reliable native date range for its frequency.")
    elif frequency in _MIN_DATES and first < _MIN_DATES[frequency][0]:
        raise ValueError(
            f"Date is outside the reliable native {_MIN_DATES[frequency][1]} date range."
        )


def _series_widths(element: int, element_frequency: int) -> frozenset[int]:
    if element_frequency:
        if element not in (KIND_INTEGER, KIND_DATE) or element_frequency not in _SCALAR_FREQUENCIES:
            raise TypeError("Unsupported DataEcon series element frequency or kind.")
        return frozenset({8})
    if element not in _SERIES_WIDTHS:
        raise TypeError(_SERIES_SUPPORT)
    return _SERIES_WIDTHS[element]


def encode_series(series: SeriesValue) -> SeriesPayload:
    """Return an independent snapshot with explicit element kind/frequency/length."""
    if not isinstance(series, (TSeries, StoredSeries)):
        raise TypeError("write_series requires a TSeries or StoredSeries.")
    if sys.byteorder != "little":
        raise RuntimeError(
            "DataEcon interchange is currently supported on little-endian hosts only."
        )
    code = next(
        (code for code, freq in _SERIES_FREQUENCIES.items() if freq == series.frequency), None
    )
    if code is None:
        raise TypeError(_SERIES_SUPPORT)
    if isinstance(series, StoredSeries):
        series.validate()
        descriptor = series.element
        element, element_frequency = descriptor.native_kind, descriptor.native_frequency
        marker = descriptor.written_marker(len(series))
        values = series.values
    else:
        element, marker, values = _numeric_series_values(series)
        element_frequency = 0
    length, first = len(values), series.firstdate.value
    validate_metadata((2, 12, element, element_frequency, 1, length, code, first, values.nbytes))
    # Capacity/date checks precede the Boolean normalization allocation too.
    if values.dtype.kind == "b":
        values = values.astype(np.int8)
    payload = values.tobytes(order="C")
    # Recheck the owned payload's interpretation. This is not an atomic snapshot
    # of an array concurrently resized or edited by its caller.
    validate_series_payload(element, element_frequency, length, payload, marker)
    return SeriesPayload(code, first, payload, element, element_frequency, length, marker)


def _numeric_series_values(series: TSeries) -> tuple[int, str | None, np.ndarray[Any, Any]]:
    dtype = series.values.dtype
    entry = next(((name, kind) for name, (kind, dt) in _SERIES_TYPES.items() if dt == dtype), None)
    # Windows longdouble can compare equal to float64; preserve the explicit rejection.
    if entry is None or not dtype.isnative or dtype.char in ("g", "G"):
        raise TypeError(_SERIES_SUPPORT + " Values must have a supported native-endian dtype.")
    name, element = entry
    marker = (
        name
        if name == "Bool" or (not len(series.values) and name != _SERIES_DEFAULTS[element])
        else None
    )
    return element, marker, series.values


def decode_series(
    code: int,
    first: int,
    payload: bytes,
    element: int,
    element_frequency: int,
    length: int,
    marker: str | None,
) -> SeriesValue:
    """Construct owning values; native storage never escapes the adapter."""
    if sys.byteorder != "little":
        raise RuntimeError(
            "DataEcon interchange is currently supported on little-endian hosts only."
        )
    validate_metadata((2, 12, element, element_frequency, 1, length, code, first, len(payload)))
    resolved = validate_series_payload(element, element_frequency, length, payload, marker)
    anchor = MIT(series_frequency(code), first)
    if isinstance(resolved, StoredElement):
        values = np.frombuffer(payload, dtype=resolved.dtype).copy()
        return StoredSeries(anchor, values, resolved, copy=False)
    values = np.frombuffer(payload, dtype=resolved)
    if marker == "Bool":
        return TSeries(anchor, np.array(values == 1, dtype=bool))
    return TSeries(anchor, values.copy())
