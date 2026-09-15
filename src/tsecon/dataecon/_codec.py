# SPDX-License-Identifier: MIT
"""Scalar, series and array conversions between core objects and native DataEcon payloads.

Scalars: Float16/32/64, Int8/16/32/64, UInt8/16/32/64, Complex64/128, UTF-8
strings, and MIT dates or Durations over the unit, daily, business-daily,
weekly (every end day), monthly, quarterly, half-yearly and annual
frequencies; Int128/UInt128/ComplexF16 widths, raw text and marker-bearing
scalars (``Symbol``, ``Date``, ``Rational{T}``, integer ``Complex{T}``, ...)
travel as ``StoredScalar`` with explicit interpretation, and ``Fraction``,
``datetime.date``/``datetime.datetime`` and ``IntegerComplex`` inputs are
written in the exact storage Julia's own writer produces. Series and
MVTSeries also support represented MIT/Duration,
Int128/UInt128 and ComplexF16 elements over every core axis frequency (daily,
business-daily and weekly axes use the verified calendar windows), and
preserve Julia's finite reconstruction markers (``jeltype``/``jtype``) as
stored form plus explicit interpretation. No marker text is evaluated.
"""

from __future__ import annotations

import datetime as dt
import struct
import sys
from fractions import Fraction
from typing import Any, Literal, NamedTuple, TypeAlias, cast

import numpy as np

from tsecon.frequencies import (
    Frequency,
)
from tsecon.mit import MIT, Duration
from tsecon.mitrange import MITRange
from tsecon.mvtseries import MVTSeries
from tsecon.tseries import TSeries

from . import _arrays, _interpret
from ._arrays import StoredArray, StoredText, resolve_array_interpretation
from ._metadata import (
    _CALENDAR_FREQUENCIES,
    _CALENDAR_RANGES,
    _FREQUENCIES,
    _MIN_DATES,
    _SCALAR_FREQUENCIES,
    _SCALAR_MIN_DATES,
    _SERIES_FREQUENCIES,
    JULIA_KIND_DEFAULTS,
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
    MIN_DATE,
    MIN_HALFYEARLY_DATE,
    MIN_INT64,
    MIN_MONTHLY_DATE,
    MIN_QUARTERLY_DATE,
    UNIT_FREQUENCY,
    julia_frequency_name,
)
from ._represented import (
    COMPLEXF16,
    INT128,
    UINT128,
    StoredElement,
    StoredMVTSeries,
    StoredSeries,
    check_column_names,
    resolve_interpretation,
    resolve_mvtseries_interpretation,
)
from ._scalars import (
    NUMERIC_WIDTHS,
    IntegerComplex,
    StoredScalar,
    encode_text,
    is_stored_form,
    plain_numeric,
    plain_text,
    scalar_frequency,
    scalar_frequency_code,
    validate_date_code,
    validate_scalar_metadata,
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
    "scalar_frequency",
    "scalar_frequency_code",
    "validate_date_code",
    "validate_scalar_metadata",
]

# Scalar numeric widths per type code (the wide Int128/UInt128/ComplexF16
# widths read as StoredScalar) live in _scalars.NUMERIC_WIDTHS; ordinary
# read results by (type, nbytes) come from _scalars.plain_numeric.
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
if any(width not in NUMERIC_WIDTHS[kind] for kind, width in _NUMPY_CLASSES.values()):
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
    | StoredScalar
    | Fraction
    | dt.date
    | dt.datetime
    | IntegerComplex
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
    | StoredScalar
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
_SERIES_TYPES = JULIA_NUMERIC_TYPES
_SERIES_DEFAULTS = JULIA_KIND_DEFAULTS
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
    object_marker: str | None


class ArrayPayload(NamedTuple):
    """Owned bytes and explicit storage information for a plain vector or range."""

    object_type: int
    axis_type: int
    frequency: int
    first: int
    payload: bytes
    element: int
    element_frequency: int
    length: int
    marker: str | None
    object_marker: str | None


class MatrixPayload(NamedTuple):
    """Owned column-major bytes and explicit storage information for a 2-D object."""

    object_type: int
    element: int
    element_frequency: int
    axis1_type: int
    rows: int
    frequency: int
    first: int
    columns: int
    names: str | None
    payload: bytes
    marker: str | None
    object_marker: str | None


class TensorPayload(NamedTuple):
    """Owned column-major bytes and explicit storage information for a 3-D to 5-D object."""

    object_type: int
    element: int
    element_frequency: int
    shape: tuple[int, ...]
    payload: bytes
    marker: str | None
    object_marker: str | None


# Series widths are the scalar widths, represented families included.
_SERIES_WIDTHS = dict(NUMERIC_WIDTHS)
_WIDE_ELEMENTS = {(1, 16): INT128, (2, 16): UINT128, (5, 4): COMPLEXF16}
_WIDE_NAMES = {e.julia_name: e for e in _WIDE_ELEMENTS.values()}
SeriesValue: TypeAlias = TSeries | StoredSeries | MVTSeries | StoredMVTSeries
# Text input is a flat or rectangular nested sequence of ``str`` (list/tuple),
# a NumPy ``str_`` array, or a StoredText of one to five dimensions.
ArrayValue: TypeAlias = (
    np.ndarray[Any, Any] | range | MITRange | StoredArray | StoredText | list[Any] | tuple[Any, ...]
)
MatrixMetadata: TypeAlias = tuple[int, ...]
# Six header integers (class, object type, element, element frequency, axis
# count, nbytes) followed by DE_MAX_AXES slots of (id, axis type, length,
# frequency, first), exactly as the native loader fills `ndtseries_t`.
TensorMetadata: TypeAlias = tuple[int, ...]
MAX_AXES = _interpret.MAX_AXES
_TENSOR_METADATA_LENGTH = 6 + 5 * MAX_AXES
_TENSOR_SUPPORT = (
    "DataEcon N-dimensional support covers ordinary numeric and Boolean plain arrays, "
    f"represented plain arrays and text arrays with three to {MAX_AXES} plain axes."
)
_MATRIX_SUPPORT = (
    "DataEcon two-dimensional support covers ordinary numeric and Boolean plain "
    "matrices and MVTSeries, represented plain matrices and MVTSeries, and text matrices."
)


def _require_token(marker: str) -> _interpret.Target:
    target = _interpret.resolve_token(marker)
    if target is None:
        raise TypeError(
            "Unsupported Julia reconstruction attribute for this element encoding; marker "
            "text is compared with a finite table and never evaluated."
        )
    return target


def _base_element(element: int, length: int, nbytes: int) -> np.dtype[Any] | StoredElement:
    """Return the element Julia's loader builds before any marker: dtype or wide descriptor."""
    if not length:
        return _SERIES_TYPES[_SERIES_DEFAULTS[element]][1]
    width = nbytes // length
    wide = _WIDE_ELEMENTS.get((element, width))
    return wide if wide is not None else _SERIES_DTYPES[element, width]


def series_dtype(
    element: int,
    element_frequency: int,
    length: int,
    nbytes: int,
    marker: str | None,
    object_marker: str | None = None,
) -> np.dtype[Any] | StoredElement:
    """Resolve structurally validated metadata using finite marker comparisons.

    Returns the dtype of a ``TSeries`` result (an ordinary payload, possibly
    with the normalizing ``Bool`` marker) or the ``StoredElement`` of a
    ``StoredSeries`` result, whose ``marker`` is the preserved ``jeltype``
    text. A present whole-object marker makes the element marker inactive and
    always yields a preserved container; its own token is validated against
    the axis by :func:`validate_series_payload`. Redundant tokens that name the
    stored family resolve to the unmarked family.
    """
    _interpret.check_marker_text(marker, "element")
    _interpret.check_marker_text(object_marker, "whole-object")
    if element_frequency:
        return _dated_series_type(element, element_frequency, length, marker, object_marker)
    if object_marker is not None:
        base = _base_element(element, length, nbytes)
        if isinstance(base, StoredElement):
            return base.with_marker(marker)
        return StoredElement.numeric(base, marker)
    if not length:
        return _empty_series_type(element, marker)
    return _nonempty_series_type(_base_element(element, length, nbytes), marker)


def _dated_series_type(
    element: int, element_frequency: int, length: int, marker: str | None, object_marker: str | None
) -> StoredElement:
    descriptor = StoredElement(
        "date" if element == KIND_DATE else "duration", scalar_frequency(element_frequency)
    )
    if object_marker is None and (marker is None or marker == descriptor.julia_name):
        return descriptor
    if object_marker is None and marker is not None:
        _require_token(marker)
    if not length:
        raise TypeError(
            "Julia cannot load an empty date or duration series; a reconstruction marker "
            "other than its own element type is not supported on one."
        )
    return descriptor.with_marker(marker)


def _nonempty_series_type(
    base: np.dtype[Any] | StoredElement, marker: str | None
) -> np.dtype[Any] | StoredElement:
    if isinstance(base, StoredElement):
        if marker is None or marker == base.julia_name:
            return base
        if marker == "Bool":
            return base.with_bool_marker()
        _require_token(marker)
        return base.with_marker(marker)
    if marker is None or marker == "Bool":
        # Keep the carrier dtype until Bool values have been checked without casting.
        return base
    _require_token(marker)
    # Ordinary nonempty identity markers were not accepted before this slice.
    # They are therefore new foreign encodings and retain their literal token
    # under the storage-preserving policy, even though Julia drops it on rewrite.
    return StoredElement.numeric(base, marker)


def _empty_series_type(element: int, marker: str | None) -> np.dtype[Any] | StoredElement:
    default = _SERIES_TYPES[_SERIES_DEFAULTS[element]][1]
    if marker is None:
        return default
    if marker == "Bool":
        return np.dtype("?")
    target = _require_token(marker)
    if target.kind == "numeric":
        # An exact same-kind token restores the typed empty; a foreign kind is
        # preserved on the kind default (the stored width is unrecoverable).
        if marker in _SERIES_TYPES and _SERIES_TYPES[marker][0] == element:
            return target.dtype
    elif marker in _WIDE_NAMES:
        wide = _WIDE_NAMES[marker]
        if wide.native_kind == element:
            return wide
    return StoredElement.numeric(default, marker)


def validate_series_payload(
    code: int,
    element: int,
    element_frequency: int,
    length: int,
    payload: bytes,
    marker: str | None,
    object_marker: str | None = None,
) -> np.dtype[Any] | StoredElement:
    """Validate interpretation of an owned, structurally checked payload.

    Resolves the markers, then checks that the declared conversion is possible
    for the stored values (route and value rules of the pinned Julia loader)
    without allocating a converted array. ``TypeError`` names an unsupported
    token or route, ``ValueError`` an inconvertible value.
    """
    resolved = series_dtype(element, element_frequency, length, len(payload), marker, object_marker)
    if isinstance(resolved, StoredElement):
        values = np.frombuffer(payload, dtype=resolved.dtype)
        resolve_interpretation(values, resolved, object_marker, series_frequency(code))
    elif marker == "Bool":
        _interpret.check_bool(
            np.frombuffer(payload, dtype=resolved), _interpret.numeric_target(resolved)
        )
    return resolved


def series_frequency(code: int) -> Frequency:
    """Resolve only canonical supported native series frequency codes."""
    try:
        return _SERIES_FREQUENCIES[code]
    except KeyError:
        raise TypeError(_SERIES_SUPPORT) from None


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
        kind, frequency, packed = KIND_STRING, 0, encode_text(value)
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
            "an MIT or a Duration, a fractions.Fraction, an exact datetime.date or "
            "datetime.datetime, an IntegerComplex or a StoredScalar."
        )
    return kind, frequency, packed


def decode_scalar(
    kind: int, frequency: int, payload: bytes, marker: str | None = None
) -> ScalarResult:
    """Return an independent core value from validated metadata and a byte snapshot.

    Ordinary unmarked encodings decode to plain values. A ``jtype`` marker, a
    sixteen-byte integer or four-byte complex width, or a text payload that is
    not a single NUL-terminated valid UTF-8 string returns a ``StoredScalar``
    holding the exact stored form; nothing is converted or evaluated.
    """
    validate_scalar_metadata((1, kind, frequency, len(payload)))
    if sys.byteorder != "little":
        raise RuntimeError("DataEcon interchange requires a little-endian host.")
    if is_stored_form(kind, payload, marker):
        return StoredScalar(payload, kind, frequency, marker)
    if kind == KIND_STRING:
        return plain_text(payload)
    if (frequency == 0 and kind != KIND_INTEGER) or len(payload) != 8:
        # Float, complex and every narrow width; Int64 and dates/durations follow.
        return cast(ScalarResult, plain_numeric(kind, payload))
    number = int(struct.unpack("<q", payload)[0])
    if frequency == 0:
        return number
    if kind == KIND_DATE:
        validate_date_code(frequency, number)
        return MIT(scalar_frequency(frequency), number)
    return Duration(scalar_frequency(frequency), number)


def encode_marked_scalar(value: ScalarValue) -> tuple[int, int, bytes, str | None]:
    """Return the native type code, frequency, owned payload and ``jtype`` marker.

    A ``StoredScalar`` writes its exact stored form. ``fractions.Fraction``,
    exact ``datetime.date``/``datetime.datetime`` and ``IntegerComplex``
    inputs build the storage Julia's own writer produces (``float(p//q)``
    plus ``Rational{Int64}``, unix seconds plus ``Date``/``DateTime``, two
    Float64 plus ``Complex{Int64}``), each accepted only when the pinned Julia
    loader rebuilds the requested value; the ``StoredScalar`` constructors
    expose the other parameters and the explicit lossy option. Every other
    input follows :func:`encode_scalar` unmarked.
    """
    if type(value) is StoredScalar:
        return value.kind, value.frequency, value.payload, value.marker
    if type(value) is Fraction:
        stored = StoredScalar.fraction(value)
    elif type(value) is dt.date:
        stored = StoredScalar.date(value)
    elif type(value) is dt.datetime:
        stored = StoredScalar.datetime(value)
    elif type(value) is IntegerComplex:
        stored = StoredScalar.integer_complex(value)
    else:
        kind, frequency, payload = encode_scalar(value)
        return kind, frequency, payload, None
    return stored.kind, stored.frequency, stored.payload, stored.marker


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
    # Later observations are implicit consecutive codes, bounded above by the
    # signed 32-bit check and the payload limit (at most 134,217,728 one-byte
    # values), so the last code stays below maximum + 134,217,728 and never
    # reaches a native calendar call.
    _check_range_axis(frequency, first, length)


def _series_widths(element: int, element_frequency: int) -> frozenset[int]:
    if element_frequency:
        if element not in (KIND_INTEGER, KIND_DATE) or element_frequency not in _SCALAR_FREQUENCIES:
            raise TypeError("Unsupported DataEcon series element frequency or kind.")
        return frozenset({8})
    if element not in _SERIES_WIDTHS:
        raise TypeError(_SERIES_SUPPORT)
    return _SERIES_WIDTHS[element]


def encode_series(series: SeriesValue) -> SeriesPayload | MatrixPayload:
    """Return an independent snapshot with explicit element kind/frequency/length."""
    if isinstance(series, MVTSeries):
        return encode_mvtseries(series)
    if isinstance(series, StoredMVTSeries):
        return encode_stored_mvtseries(series)
    if not isinstance(series, (TSeries, StoredSeries)):
        raise TypeError(
            "write_series requires a TSeries, StoredSeries, MVTSeries or StoredMVTSeries."
        )
    if sys.byteorder != "little":
        raise RuntimeError(
            "DataEcon interchange is currently supported on little-endian hosts only."
        )
    code = next(
        (code for code, freq in _SERIES_FREQUENCIES.items() if freq == series.frequency), None
    )
    if code is None:
        raise TypeError(_SERIES_SUPPORT)
    object_marker = None
    if isinstance(series, StoredSeries):
        series.validate()
        descriptor = series.element
        element, element_frequency = descriptor.native_kind, descriptor.native_frequency
        marker = descriptor.written_marker(len(series))
        object_marker = series.object_marker
        values = series.values
    else:
        element, marker, values = _numeric_series_values(series)
        element_frequency = 0
    length, first, nbytes = len(values), series.firstdate.value, values.nbytes
    validate_metadata((2, 12, element, element_frequency, 1, length, code, first, nbytes))
    # Capacity/date checks precede the Boolean normalization allocation too.
    if values.dtype.kind == "b":
        values = values.astype(np.int8)
    payload = values.tobytes(order="C")
    # Recheck the owned snapshot against the metadata captured above. The file
    # lock does not serialize a caller's own array mutations: an array resized
    # or edited by another thread between validation and the snapshot is
    # detected here and refused, but no atomic snapshot is promised.
    if len(payload) != nbytes:
        raise ValueError("The series values changed size during the snapshot; nothing was written.")
    resolved = validate_series_payload(
        code, element, element_frequency, length, payload, marker, object_marker
    )
    if isinstance(series, StoredSeries) and resolved != series.element:
        # An emptied "Bool"-marked wide carrier would otherwise resolve to the
        # ambiguous empty Boolean encoding and lose its stored width; the same
        # recheck covers every preserved marker and dtype.
        raise ValueError(
            "The snapshot no longer matches the container's stored element; nothing was written."
        )
    return SeriesPayload(
        code, first, payload, element, element_frequency, length, marker, object_marker
    )


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
    object_marker: str | None = None,
) -> SeriesValue:
    """Construct owning values; native storage never escapes the adapter."""
    if sys.byteorder != "little":
        raise RuntimeError(
            "DataEcon interchange is currently supported on little-endian hosts only."
        )
    validate_metadata((2, 12, element, element_frequency, 1, length, code, first, len(payload)))
    resolved = validate_series_payload(
        code, element, element_frequency, length, payload, marker, object_marker
    )
    anchor = MIT(series_frequency(code), first)
    if isinstance(resolved, StoredElement):
        values = np.frombuffer(payload, dtype=resolved.dtype).copy()
        return StoredSeries(anchor, values, resolved, copy=False, object_marker=object_marker)
    values = np.frombuffer(payload, dtype=resolved)
    if marker == "Bool":
        return TSeries(anchor, np.array(values == 1, dtype=bool))
    return TSeries(anchor, values.copy())


def _check_range_axis(frequency: int, first: int, length: int) -> None:
    """Bound a dated axis anchor and its implicit endpoint for its frequency."""
    if frequency == UNIT_FREQUENCY:
        if not MIN_INT64 <= first <= MAX_INT64 or (length and first + length - 1 > MAX_INT64):
            raise ValueError("Unit dates must fit the signed 64-bit range.")
    elif not MIN_DATE <= first <= MAX_DATE or (length and first + length - 1 > MAX_DATE):
        raise ValueError("DataEcon dates must fit the native signed 32-bit date range.")
    if frequency in _CALENDAR_RANGES:
        # Only the stored first date must lie inside the verified calendar
        # window: Julia packs nothing else and never decodes a trailing date.
        minimum, maximum = _CALENDAR_RANGES[frequency]
        if not minimum <= first <= maximum:
            raise ValueError("Date is outside the reliable native date range for its frequency.")
    elif frequency in _MIN_DATES and first < _MIN_DATES[frequency][0]:
        raise ValueError(
            f"Date is outside the reliable native {_MIN_DATES[frequency][1]} date range."
        )


def _validate_vector_metadata(metadata: Metadata) -> None:
    _, _, element, element_freq, axis, length, frequency, first, nbytes = metadata
    if (axis, frequency, first) != (0, 0, 0):
        raise TypeError("A DataEcon vector must have one plain axis.")
    if not 0 <= nbytes <= MAX_BYTES or (not length and nbytes):
        raise ValueError("Invalid or oversized DataEcon vector payload.")
    if element == KIND_STRING:
        if element_freq:
            raise TypeError("A DataEcon text vector carries no element frequency.")
        return
    widths = _series_widths(element, element_freq)
    if length and (nbytes % length or nbytes // length not in widths):
        raise ValueError("Invalid DataEcon vector element width or payload length.")


def _validate_range_metadata(metadata: Metadata) -> None:
    _, _, element, element_freq, axis, length, frequency, first, nbytes = metadata
    if (element, element_freq, nbytes) != (0, 0, 0):
        raise TypeError("A DataEcon range has no element payload.")
    if axis == 0:
        if (frequency, first) != (0, 0):
            raise TypeError("An integer DataEcon range must have a plain axis.")
        return
    if axis != 1:
        raise TypeError("A dated DataEcon range must have one range axis.")
    scalar_frequency(frequency)
    validate_date_code(frequency, first)
    if length and first + length - 1 > MAX_INT64:
        raise ValueError("The DataEcon range endpoint exceeds the signed 64-bit range.")


def validate_array_metadata(metadata: Metadata) -> None:
    """Validate a plain one-dimensional vector, text vector or unit-step range."""
    cls, obj_type, _, _, _, length, _, _, _ = metadata
    if cls != 2 or obj_type not in (10, 11):
        raise TypeError("DataEcon arrays support plain vectors and unit-step ranges only.")
    if length < 0 or length > MAX_INT64:
        raise ValueError("Invalid DataEcon array length.")
    if obj_type == 10:
        _validate_vector_metadata(metadata)
    else:
        _validate_range_metadata(metadata)


# Text markers Julia's loader resolves for a packed string vector. "String" is
# the token Julia writes on an empty String vector, where it names the stored
# element itself; the others are foreign and are preserved as stored.
TEXT_MARKERS = ("String", "Symbol", "SubString{String}", "AbstractString")


def split_text_payload(payload: bytes, length: int) -> tuple[bytes, ...]:
    """Split an owned packed text payload into exactly ``length`` elements.

    DataEcon packs text as NUL-terminated byte strings. The native unpacker
    bounds only each element's start and then scans for NUL without a limit, so
    a payload whose last element is unterminated would read past the buffer;
    this function never calls it and validates the owned bytes in Python
    instead. Julia ignores bytes after the last visited element; a payload with
    unvisited trailing bytes is refused here so a rewrite stays exact.
    """
    if length == 0:
        if payload:
            raise ValueError("An empty DataEcon text array must have an empty payload.")
        return ()
    if payload.count(b"\0") != length or not payload.endswith(b"\0"):
        raise ValueError(
            f"A DataEcon text array of {length} elements needs exactly {length} NUL "
            "terminators and no unvisited trailing bytes."
        )
    return tuple(payload.split(b"\0")[:length])


def _text_marker(marker: str | None, object_marker: str | None) -> str | None:
    if object_marker is not None:
        raise TypeError(
            "Whole-object reconstruction markers on DataEcon text arrays are not supported yet."
        )
    if marker is not None and marker not in TEXT_MARKERS:
        raise TypeError(
            f"Unsupported Julia reconstruction attribute {marker!r} for a text array; "
            "marker text is compared with a finite table and never evaluated."
        )
    return marker


def _array_dtype(
    element: int,
    element_frequency: int,
    length: int,
    nbytes: int,
    marker: str | None,
    object_marker: str | None,
) -> np.dtype[Any] | StoredElement:
    """Resolve a plain array's element the way :func:`series_dtype` resolves a series."""
    resolved = series_dtype(element, element_frequency, length, nbytes, marker, object_marker)
    canonical_bool = (
        element == KIND_INTEGER and not element_frequency and (not length or nbytes // length == 1)
    )
    # A Bool marker that resolves to an ordinary dtype becomes Boolean values,
    # which Julia writes as one signed byte. A Bool marker on a represented
    # carrier resolves to a StoredElement instead and keeps its stored width
    # under the approved wide-Bool preservation policy.
    if (
        marker == "Bool"
        and object_marker is None
        and not isinstance(resolved, StoredElement)
        and not canonical_bool
    ):
        raise TypeError("A plain Boolean array requires the canonical one-byte signed encoding.")
    return resolved


def validate_array_payload(
    metadata: Metadata, payload: bytes, marker: str | None, object_marker: str | None
) -> np.dtype[Any] | StoredElement | None:
    """Validate array metadata, finite markers and values without constructing output."""
    validate_array_metadata(metadata)
    _, obj_type, element, element_freq, axis, length, frequency, _, _ = metadata
    _interpret.check_marker_text(marker, "element")
    _interpret.check_marker_text(object_marker, "whole-object")
    if obj_type == 11:
        if object_marker is not None:
            raise TypeError("Whole-object reconstruction markers on ranges are not supported.")
        expected = (
            "Int64" if axis == 0 else f"MIT{{{julia_frequency_name(scalar_frequency(frequency))}}}"
        )
        if marker is not None and (length or marker != expected):
            raise TypeError("Unsupported Julia reconstruction attribute for this range encoding.")
        return None
    if element == KIND_STRING:
        _text_marker(marker, object_marker)
        split_text_payload(payload, length)
        return None
    resolved = _array_dtype(element, element_freq, length, len(payload), marker, object_marker)
    if isinstance(resolved, StoredElement):
        values = np.frombuffer(payload, dtype=resolved.dtype)
        resolve_array_interpretation(values, resolved, object_marker)
    elif marker == "Bool":
        _interpret.check_bool(
            np.frombuffer(payload, dtype=resolved), _interpret.numeric_target(resolved)
        )
    return resolved


def _ordinary_array_entry(dtype: np.dtype[Any]) -> tuple[str, int]:
    entry = next(
        ((name, kind) for name, (kind, candidate) in _SERIES_TYPES.items() if candidate == dtype),
        None,
    )
    # Windows longdouble can compare equal to float64; preserve the explicit rejection.
    if entry is None or not dtype.isnative or dtype.char in ("g", "G"):
        raise TypeError(
            "DataEcon plain array support covers ordinary native-endian numeric and Boolean dtypes."
        )
    return entry


def _ordinary_payload(values: np.ndarray[Any, Any], order: Literal["C", "F"]) -> bytes:
    if values.dtype.kind == "b":
        values = values.astype(np.int8)
    return values.tobytes(order=order)


def _ordinary_snapshot(
    values: np.ndarray[Any, Any], order: Literal["C", "F"]
) -> tuple[bytes, np.dtype[Any], int]:
    """Snapshot an ordinary array's bytes against the shape and dtype seen beforehand.

    The expected byte count is fixed from the size and storage width captured
    before ``tobytes`` runs, never from the live array afterwards: an input
    that shrinks during the snapshot would otherwise pass a comparison with
    its own new size, and the same native kind at a narrower width would then
    be stored under the original shape. Returns the payload, the storage dtype
    the bytes must resolve to (Int8 for Boolean values) and the expected count.
    """
    storage = np.dtype("i1") if values.dtype.kind == "b" else values.dtype
    expected = int(values.size) * storage.itemsize
    payload = _ordinary_payload(values, order)
    if len(payload) != expected:
        raise ValueError("The array changed size during the snapshot; nothing was written.")
    return payload, storage, expected


def _check_snapshot_storage(
    resolved: np.dtype[Any] | StoredElement | None, storage: np.dtype[Any], marker: str | None
) -> None:
    """Require the validated snapshot to resolve to the dtype that was snapshotted."""
    # An empty Boolean payload resolves to the bool dtype itself; every other
    # ordinary payload resolves to its own storage dtype.
    if resolved == storage or (marker == "Bool" and resolved == np.dtype("?")):
        return
    raise ValueError(
        "The snapshot no longer resolves to the array's own dtype; nothing was written."
    )


def _text_elements(value: Any) -> tuple[tuple[bytes, ...], str | None, tuple[int, ...]]:
    """Return column-major element bytes, the marker and the captured shape of a text input.

    Every path checks rank, dimensions, the Python-integer element count and
    the minimum packed size before the input is flattened or copied, and the
    cumulative UTF-8 byte total before every element is encoded.
    """
    if isinstance(value, StoredText):
        assert value.shape is not None
        _arrays.check_text_capacity(sum(len(item) for item in value.values) + len(value.values))
        return value.values, value.marker, value.shape
    if isinstance(value, np.ndarray):
        shape = _arrays.text_array_shape(value)
        # The logical values are snapshotted in column-major order, in bounded
        # chunks, only after the shape-derived bounds passed; the array is
        # never raveled as a whole, modified or kept.
        return _arrays.text_array_bytes(value, shape), None, shape
    shape, flat = _arrays.nested_text(value)
    return _arrays.column_major(_arrays.pack_strings(flat), shape), None, shape


def _encode_ordinary_vector(value: np.ndarray[Any, Any]) -> ArrayPayload:
    name, element = _ordinary_array_entry(value.dtype)
    marker = (
        name if name == "Bool" or (not len(value) and name != _SERIES_DEFAULTS[element]) else None
    )
    length = len(value)
    validate_array_metadata((2, 10, element, 0, 0, length, 0, 0, value.nbytes))
    payload, storage, nbytes = _ordinary_snapshot(value, "C")
    resolved = validate_array_payload(
        (2, 10, element, 0, 0, length, 0, 0, nbytes), payload, marker, None
    )
    _check_snapshot_storage(resolved, storage, marker)
    return ArrayPayload(10, 0, 0, 0, payload, element, 0, length, marker, None)


def _encode_stored_vector(value: StoredArray) -> ArrayPayload:
    element = value.element
    length = len(value)
    marker = element.written_marker(length)
    payload = value.values.tobytes(order="C")
    if len(payload) != length * element.itemsize:
        raise ValueError("The array changed size during the snapshot; nothing was written.")
    metadata = (2, 10, element.native_kind, element.native_frequency, 0, length, 0, 0, len(payload))
    resolved = validate_array_payload(metadata, payload, marker, value.object_marker)
    if resolved != element:
        raise ValueError(
            "The snapshot no longer matches the container's stored element; nothing was written."
        )
    return ArrayPayload(
        10,
        0,
        0,
        0,
        payload,
        element.native_kind,
        element.native_frequency,
        length,
        marker,
        value.object_marker,
    )


def _encode_text_array(value: Any) -> ArrayPayload | MatrixPayload | TensorPayload:
    """Encode a text vector, matrix or tensor as NUL-terminated column-major UTF-8 bytes."""
    elements, marker, shape = _text_elements(value)
    if marker is not None and marker not in TEXT_MARKERS:
        raise TypeError(f"Unsupported text reconstruction marker {marker!r}.")
    count = _arrays.text_count(shape)
    # The snapshot must hold exactly the elements of the captured shape at
    # every rank: an input that shrinks or grows while it is flattened would
    # otherwise be stored under the wrong length or shape.
    if len(elements) != count:
        raise ValueError("The text array changed size during the snapshot; nothing was written.")
    # Julia writes the element token on every empty array; keep byte parity.
    if marker is None and count == 0:
        marker = "String"
    payload = b"".join(item + b"\0" for item in elements)
    if len(shape) == 1:
        metadata = (2, 10, KIND_STRING, 0, 0, count, 0, 0, len(payload))
        validate_array_payload(metadata, payload, marker, None)
        return ArrayPayload(10, 0, 0, 0, payload, KIND_STRING, 0, count, marker, None)
    if len(shape) == 2:
        rows, columns = shape
        matrix = (3, 20, KIND_STRING, 0, 0, rows, 0, 0, 0, columns, 0, 0, len(payload))
        validate_matrix_payload(matrix, payload, marker, None)
        return MatrixPayload(
            20, KIND_STRING, 0, 0, rows, 0, 0, columns, None, payload, marker, None
        )
    tensor = tensor_metadata(KIND_STRING, 0, shape, len(payload))
    validate_tensor_payload(tensor, payload, marker, None)
    return TensorPayload(30, KIND_STRING, 0, shape, payload, marker, None)


def _encode_integer_range(value: range) -> ArrayPayload:
    try:
        length = len(value)
    except OverflowError:
        raise ValueError(
            "The integer range length exceeds the native signed 64-bit range."
        ) from None
    if value.start != 1 or value.step != 1 or (not length and value.stop != 1):
        raise ValueError(
            "DataEcon preserves only an integer range's length; use range(1, stop) "
            "for a lossless write."
        )
    return ArrayPayload(11, 0, 0, 0, b"", 0, 0, length, "Int64" if length == 0 else None, None)


def _encode_date_range(value: MITRange) -> ArrayPayload:
    if value.step != 1:
        raise ValueError("DataEcon supports only unit-step MITRange values.")
    try:
        length = len(value)
    except OverflowError:
        raise ValueError("The MITRange length exceeds the native signed 64-bit range.") from None
    frequency = scalar_frequency_code(value.frequency)
    first = value.start.value
    validate_array_metadata((2, 11, 0, 0, 1, length, frequency, first, 0))
    marker = f"MIT{{{julia_frequency_name(value.frequency)}}}" if length == 0 else None
    return ArrayPayload(11, 1, frequency, first, b"", 0, 0, length, marker, None)


def _encode_ndarray(value: np.ndarray[Any, Any]) -> ArrayPayload | MatrixPayload | TensorPayload:
    if value.dtype.kind in ("U", "S", "O", "T"):
        return _encode_text_array(value)
    if value.ndim == 2:
        return _encode_plain_matrix(value)
    if value.ndim == 1:
        return _encode_ordinary_vector(value)
    _check_tensor_rank(value.ndim)
    return _encode_plain_tensor(value)


def _encode_stored_array(value: StoredArray) -> ArrayPayload | MatrixPayload | TensorPayload:
    value.validate()
    if value.ndim == 2:
        return _encode_stored_matrix(value)
    if value.ndim == 1:
        return _encode_stored_vector(value)
    return _encode_stored_tensor(value)


def encode_array(value: ArrayValue) -> ArrayPayload | MatrixPayload | TensorPayload:
    """Encode a plain array, text vector or lossless unit-step range."""
    if sys.byteorder != "little":
        raise RuntimeError(
            "DataEcon interchange is currently supported on little-endian hosts only."
        )
    if isinstance(value, np.ndarray):
        return _encode_ndarray(value)
    if isinstance(value, StoredArray):
        return _encode_stored_array(value)
    if isinstance(value, (StoredText, list, tuple)):
        return _encode_text_array(value)
    if type(value) is range:
        return _encode_integer_range(value)
    if isinstance(value, MITRange):
        return _encode_date_range(value)
    raise TypeError(
        "write_array requires a NumPy array, StoredArray, text sequence (flat or nested), "
        "StoredText, range or MITRange."
    )


def decode_array(
    metadata: Metadata | MatrixMetadata | TensorMetadata,
    payload: bytes,
    marker: str | None,
    object_marker: str | None,
    names: str | None = None,
) -> ArrayValue:
    """Decode an owning plain array, text vector or lossless range representation."""
    if len(metadata) == 13:
        result = decode_matrix(metadata, payload, marker, object_marker, names)
        if isinstance(result, (MVTSeries, StoredMVTSeries)):
            raise TypeError("This object is an MVTSeries; read it with read_series.")
        return result
    if len(metadata) == _TENSOR_METADATA_LENGTH:
        return decode_tensor(metadata, payload, marker, object_marker)
    metadata = cast("Metadata", metadata)
    resolved = validate_array_payload(metadata, payload, marker, object_marker)
    _, obj_type, element, _, axis, length, frequency, first, _ = metadata
    if obj_type == 11:
        return _decode_range(axis, length, frequency, first)
    if element == KIND_STRING:
        return _decode_text(payload, marker, (length,))
    if isinstance(resolved, StoredElement):
        values = np.frombuffer(payload, dtype=resolved.dtype).copy()
        return StoredArray(values, resolved, copy=False, object_marker=object_marker)
    assert resolved is not None
    values = np.frombuffer(payload, dtype=resolved)
    return np.array(values == 1, dtype=bool) if marker == "Bool" else values.copy()


def _decode_range(axis: int, length: int, frequency: int, first: int) -> range | MITRange:
    if axis == 0:
        return range(1, length + 1)
    start = MIT(scalar_frequency(frequency), first)
    return MITRange(start, MIT(start.frequency, first + length - 1))


def _decode_text(
    payload: bytes, marker: str | None, shape: tuple[int, ...]
) -> list[str] | np.ndarray[Any, Any] | StoredText:
    """Ordinary decodable text returns plain strings; anything else keeps its bytes.

    A vector returns ``list[str]``; a matrix or tensor returns an owning NumPy
    ``str_`` array of the stored shape (empties included) unless that array
    would exceed ``MAX_UNICODE_BYTES``, in which case the shaped ``StoredText``
    is returned so nothing is allocated beyond the payload. Julia writes
    ``jeltype = "String"`` only on an *empty* text array, where it merely
    repeats the stored element, so that token is canonical and dropped. A
    nonempty array carrying the same token is a foreign marker no Julia writer
    produces, and the preservation policy keeps it literally in a StoredText.
    """
    count = _arrays.text_count(shape)
    elements = split_text_payload(payload, count)
    canonical = marker == "String" and not count
    stored = StoredText(elements, None if canonical else marker, shape)
    if stored.marker is not None:
        return stored
    try:
        decoded = stored._decoded()
    except ValueError:
        return stored
    if len(shape) == 1:
        return decoded
    # A fixed-width Unicode array is count * longest element * 4 bytes, which a
    # small packed payload can inflate without bound; above the cap the
    # lossless shaped container is returned instead of allocating.
    if _arrays.unicode_nbytes(decoded) > _arrays.MAX_UNICODE_BYTES:
        return stored
    return _arrays.unicode_array(decoded, shape)


def _decode_text_array(
    payload: bytes, marker: str | None, shape: tuple[int, ...]
) -> np.ndarray[Any, Any] | StoredText:
    """Decode a text matrix or tensor; the rank-one ``list[str]`` form never applies here."""
    decoded = _decode_text(payload, marker, shape)
    assert not isinstance(decoded, list)
    return decoded


def _owned(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return independent, writable, C-ordered storage for a decoded array.

    `np.ascontiguousarray` is not enough: a Fortran-shaped view of the read-only
    payload buffer is already C-contiguous whenever a dimension is one or zero,
    and would be handed back unchanged and immutable. Reads always own their
    result, so the copy is unconditional.
    """
    return np.array(values, copy=True, order="C")


# ---- two-dimensional objects ----------------------------------------------


def _check_text_matrix_kind(obj_type: int, element_freq: int) -> None:
    # Text elements carry no width: the payload is validated by its NUL
    # terminators once the element count is known. Julia's MVTSeries holds
    # numeric elements only, so a text MVTSeries is native-only capacity.
    if element_freq:
        raise TypeError("A DataEcon text array carries no element frequency.")
    if obj_type != 20:
        raise TypeError(
            "A DataEcon MVTSeries cannot hold text elements; only plain text matrices "
            "are supported."
        )


def validate_matrix_metadata(metadata: MatrixMetadata) -> None:
    """Validate a plain matrix or MVTSeries before any value pointer is read."""
    (
        cls,
        obj_type,
        element,
        element_freq,
        ax1_type,
        rows,
        ax1_freq,
        ax1_first,
        ax2_type,
        columns,
        ax2_freq,
        ax2_first,
        nbytes,
    ) = metadata
    if cls != 3 or obj_type not in (20, 21):
        raise TypeError(_MATRIX_SUPPORT)
    if rows < 0 or columns < 0:
        raise ValueError("Invalid negative DataEcon matrix dimension.")
    if element == KIND_STRING:
        _check_text_matrix_kind(obj_type, element_freq)
    if (ax2_freq, ax2_first) != (0, 0):
        raise TypeError("A DataEcon matrix column axis carries no frequency or first date.")
    if obj_type == 20:
        if (ax1_type, ax2_type, ax1_freq, ax1_first) != (0, 0, 0, 0):
            raise TypeError("A plain DataEcon matrix must have two plain axes.")
    else:
        if (ax1_type, ax2_type) != (1, 2):
            raise TypeError("An MVTSeries must have a dated row axis and a named column axis.")
        series_frequency(ax1_freq)
        _check_range_axis(ax1_freq, ax1_first, rows)
    if rows and columns > MAX_INT64 // rows:
        raise ValueError("The DataEcon matrix element count exceeds the signed 64-bit range.")
    size = rows * columns
    if not 0 <= nbytes <= MAX_BYTES or (not size and nbytes):
        raise ValueError("Invalid or oversized DataEcon matrix payload.")
    if element == KIND_STRING:
        return
    widths = _series_widths(element, element_freq)
    if size and (nbytes % size or nbytes // size not in widths):
        raise ValueError("Invalid DataEcon matrix element width or payload length.")


def split_names(names: str | None, columns: int) -> tuple[str, ...]:
    """Split a names axis into exactly ``columns`` distinct column names.

    DataEcon joins column names with a newline and stores them as one
    NUL-terminated C string, so a name can contain neither a newline (the
    separator) nor NUL (the terminator). Julia raises when the entry count
    disagrees with the column count; a zero-column MVTSeries is unrepresentable
    because the empty string still splits into one name. Python's MVTSeries
    keys its columns by name, so duplicates would silently collapse and are
    refused rather than read back as a shorter object.
    """
    if names is None:
        raise TypeError("An MVTSeries needs a named column axis.")
    if columns == 0:
        raise TypeError(
            "A DataEcon names axis cannot encode zero columns; the empty names string "
            "still splits into one name."
        )
    parts = tuple(names.split("\n"))
    if len(parts) != columns:
        raise ValueError(f"The DataEcon names axis holds {len(parts)} names for {columns} columns.")
    if len(set(parts)) != len(parts):
        raise ValueError(
            "The DataEcon names axis holds duplicate column names, which an MVTSeries "
            "cannot represent."
        )
    return parts


def join_names(names: Any) -> str:
    """Join validated column names into the native names axis encoding."""
    return "\n".join(check_column_names([str(name) for name in names]))


def validate_matrix_payload(
    metadata: MatrixMetadata,
    payload: bytes,
    marker: str | None,
    object_marker: str | None,
    names: str | None = None,
) -> np.dtype[Any] | StoredElement | None:
    """Validate matrix metadata, names, finite markers and values (None for text)."""
    validate_matrix_metadata(metadata)
    _, obj_type, element, element_freq, _, rows, _, _, _, columns, _, _, _ = metadata
    _interpret.check_marker_text(marker, "element")
    _interpret.check_marker_text(object_marker, "whole-object")
    if obj_type == 21:
        split_names(names, columns)
    elif names is not None:
        raise TypeError("A plain DataEcon matrix has no column names.")
    size = rows * columns
    if element == KIND_STRING:
        _text_marker(marker, object_marker)
        split_text_payload(payload, size)
        return None
    if obj_type == 21:
        # A dated matrix follows the dated-series marker policy: a Bool marker
        # on any ordinary width converts exact zero/one values (Julia's map
        # builds an MVTSeries{Bool}); plain matrices keep the one-byte rule.
        resolved = series_dtype(element, element_freq, size, len(payload), marker, object_marker)
    else:
        resolved = _array_dtype(element, element_freq, size, len(payload), marker, object_marker)
    if isinstance(resolved, StoredElement):
        values = np.frombuffer(payload, dtype=resolved.dtype).reshape((rows, columns), order="F")
        if obj_type == 21:
            # A dated matrix has its own whole-object vocabulary (identities
            # only); the element table, routes and value rules are shared.
            resolve_mvtseries_interpretation(
                values, resolved, object_marker, series_frequency(metadata[6])
            )
        else:
            resolve_array_interpretation(values, resolved, object_marker)
    elif marker == "Bool":
        _interpret.check_bool(
            np.frombuffer(payload, dtype=resolved), _interpret.numeric_target(resolved)
        )
    return resolved


def _encode_plain_matrix(value: np.ndarray[Any, Any]) -> MatrixPayload:
    name, element = _ordinary_array_entry(value.dtype)
    rows, columns = (int(n) for n in value.shape)
    marker = (
        name if name == "Bool" or (not value.size and name != _SERIES_DEFAULTS[element]) else None
    )
    metadata = (3, 20, element, 0, 0, rows, 0, 0, 0, columns, 0, 0, value.nbytes)
    validate_matrix_metadata(metadata)
    payload, storage, nbytes = _ordinary_snapshot(value, "F")
    stored = (3, 20, element, 0, 0, rows, 0, 0, 0, columns, 0, 0, nbytes)
    resolved = validate_matrix_payload(stored, payload, marker, None)
    _check_snapshot_storage(resolved, storage, marker)
    return MatrixPayload(20, element, 0, 0, rows, 0, 0, columns, None, payload, marker, None)


def _encode_stored_matrix(value: StoredArray) -> MatrixPayload:
    element = value.element
    rows, columns = value.shape
    marker = element.written_marker(rows * columns)
    payload = value.values.tobytes(order="F")
    if len(payload) != rows * columns * element.itemsize:
        raise ValueError("The matrix changed size during the snapshot; nothing was written.")
    metadata = (
        3,
        20,
        element.native_kind,
        element.native_frequency,
        0,
        rows,
        0,
        0,
        0,
        columns,
        0,
        0,
        len(payload),
    )
    resolved = validate_matrix_payload(metadata, payload, marker, value.object_marker)
    if resolved != element:
        raise ValueError(
            "The snapshot no longer matches the container's stored element; nothing was written."
        )
    return MatrixPayload(
        20,
        element.native_kind,
        element.native_frequency,
        0,
        rows,
        0,
        0,
        columns,
        None,
        payload,
        marker,
        value.object_marker,
    )


def encode_mvtseries(series: MVTSeries) -> MatrixPayload:
    """Encode an ordinary numeric or Boolean MVTSeries into its native payload."""
    if sys.byteorder != "little":
        raise RuntimeError(
            "DataEcon interchange is currently supported on little-endian hosts only."
        )
    code = next(
        (code for code, freq in _SERIES_FREQUENCIES.items() if freq == series.frequency), None
    )
    if code is None:
        raise TypeError(_MATRIX_SUPPORT)
    values = series.values
    name, element = _ordinary_array_entry(values.dtype)
    rows, columns = (int(n) for n in values.shape)
    names = join_names(series.columns)
    first = series.firstdate.value
    marker = (
        name if name == "Bool" or (not values.size and name != _SERIES_DEFAULTS[element]) else None
    )
    metadata = (3, 21, element, 0, 1, rows, code, first, 2, columns, 0, 0, values.nbytes)
    validate_matrix_metadata(metadata)
    payload, storage, nbytes = _ordinary_snapshot(values, "F")
    stored = (3, 21, element, 0, 1, rows, code, first, 2, columns, 0, 0, nbytes)
    resolved = validate_matrix_payload(stored, payload, marker, None, names)
    _check_snapshot_storage(resolved, storage, marker)
    return MatrixPayload(
        21, element, 0, 1, rows, code, first, columns, names, payload, marker, None
    )


def encode_stored_mvtseries(series: StoredMVTSeries) -> MatrixPayload:
    """Encode a represented or marker-preserving MVTSeries into its native payload.

    The live carrier and its markers are validated first, the column-major
    bytes are snapshotted against the captured shape, and the snapshot is
    validated again exactly as a read would be before anything is stored.
    """
    if sys.byteorder != "little":
        raise RuntimeError(
            "DataEcon interchange is currently supported on little-endian hosts only."
        )
    series.validate()
    code = next(
        (code for code, freq in _SERIES_FREQUENCIES.items() if freq == series.frequency), None
    )
    if code is None:
        raise TypeError(_MATRIX_SUPPORT)
    element = series.element
    rows, columns = series.shape
    names = join_names(series.columns)
    first = series.firstdate.value
    marker = element.written_marker(rows * columns)
    payload = series.values.tobytes(order="F")
    if len(payload) != rows * columns * element.itemsize:
        raise ValueError("The MVTSeries changed size during the snapshot; nothing was written.")
    metadata = (
        3,
        21,
        element.native_kind,
        element.native_frequency,
        1,
        rows,
        code,
        first,
        2,
        columns,
        0,
        0,
        len(payload),
    )
    resolved = validate_matrix_payload(metadata, payload, marker, series.object_marker, names)
    if resolved != element:
        raise ValueError(
            "The snapshot no longer matches the container's stored element; nothing was written."
        )
    return MatrixPayload(
        21,
        element.native_kind,
        element.native_frequency,
        1,
        rows,
        code,
        first,
        columns,
        names,
        payload,
        marker,
        series.object_marker,
    )


def decode_matrix(
    metadata: MatrixMetadata,
    payload: bytes,
    marker: str | None,
    object_marker: str | None,
    names: str | None = None,
) -> np.ndarray[Any, Any] | StoredArray | StoredText | MVTSeries | StoredMVTSeries:
    """Decode an owning matrix, MVTSeries or StoredMVTSeries from a column-major payload."""
    if sys.byteorder != "little":
        raise RuntimeError(
            "DataEcon interchange is currently supported on little-endian hosts only."
        )
    resolved = validate_matrix_payload(metadata, payload, marker, object_marker, names)
    _, obj_type, element, _, _, rows, frequency, first, _, columns, _, _, _ = metadata
    shape = (rows, columns)
    if element == KIND_STRING:
        return _decode_text_array(payload, marker, shape)
    if isinstance(resolved, StoredElement):
        values = np.frombuffer(payload, dtype=resolved.dtype).reshape(shape, order="F")
        if obj_type == 21:
            return StoredMVTSeries(
                MIT(series_frequency(frequency), first),
                split_names(names, columns),
                _owned(values),
                resolved,
                copy=False,
                object_marker=object_marker,
            )
        return StoredArray(_owned(values), resolved, copy=False, object_marker=object_marker)
    assert resolved is not None
    values = np.frombuffer(payload, dtype=resolved).reshape(shape, order="F")
    result = _owned(values == 1) if marker == "Bool" else _owned(values)
    if obj_type == 20:
        return result
    anchor = MIT(series_frequency(frequency), first)
    return MVTSeries(anchor, list(split_names(names, columns)), result, copy=False)


# ---- N-dimensional objects ------------------------------------------------


def _check_tensor_rank(ndim: int) -> None:
    if not 3 <= ndim <= MAX_AXES:
        raise ValueError(
            f"write_array supports NumPy arrays of one to {MAX_AXES} dimensions; DataEcon "
            f"stores at most {MAX_AXES} axes."
        )


def validate_tensor_metadata(metadata: TensorMetadata) -> tuple[int, ...]:
    """Validate a plain tensor's header and axis slots before any value pointer is read.

    Returns the stored shape. Only ``type_tensor`` objects with three to five
    plain axes are accepted: class-4 objects with fewer axes and the dated or
    "other" object types are native capacity Julia never writes. Every slot
    inside the axis count must hold a real plain axis; a gap or a dangling axis
    id (the native loader reports id -1 or 0 there) is refused because Julia's
    loader would silently drop that dimension. The element count is accumulated
    in Python integers and compared with the payload size before any copy.
    """
    if len(metadata) != _TENSOR_METADATA_LENGTH:
        raise TypeError(_TENSOR_SUPPORT)
    cls, obj_type, element, element_freq, naxes, nbytes = metadata[:6]
    slots = [tuple(metadata[6 + 5 * i : 11 + 5 * i]) for i in range(MAX_AXES)]
    if cls != 4 or obj_type != 30:
        raise TypeError(
            _TENSOR_SUPPORT
            + " Dated and other N-dimensional object types are native capacity Julia never writes."
        )
    if not 3 <= naxes <= MAX_AXES:
        raise TypeError(
            f"A DataEcon N-dimensional object with {naxes} axes is native capacity Julia never "
            f"writes; supported objects have three to {MAX_AXES} plain axes."
        )
    if element == KIND_STRING and element_freq:
        raise TypeError("A DataEcon text array carries no element frequency.")
    shape = _tensor_shape(slots, naxes)
    size = 1
    for length in shape:
        size *= length
        if size > MAX_INT64:
            raise ValueError("The DataEcon tensor element count exceeds the signed 64-bit range.")
    if not 0 <= nbytes <= MAX_BYTES or (not size and nbytes):
        raise ValueError("Invalid or oversized DataEcon tensor payload.")
    if element == KIND_STRING:
        # Text has no element width; the terminators are checked against the
        # element count once the owned payload is available.
        return shape
    widths = _series_widths(element, element_freq)
    if size and (nbytes % size or nbytes // size not in widths):
        raise ValueError("Invalid DataEcon tensor element width or payload length.")
    return shape


def _tensor_shape(slots: list[tuple[int, ...]], naxes: int) -> tuple[int, ...]:
    """Return the lengths of the used slots, refusing missing, foreign or unused-but-set slots."""
    shape: list[int] = []
    for index, (axis_id, ax_type, length, frequency, first) in enumerate(slots):
        if index >= naxes:
            if axis_id != -1:
                raise TypeError("A DataEcon tensor has axis slots beyond its axis count.")
            continue
        if axis_id <= 0:
            raise TypeError(
                f"DataEcon tensor axis {index} is missing or refers to no stored axis; Julia "
                "would silently drop that dimension."
            )
        if (ax_type, frequency, first) != (0, 0, 0):
            raise TypeError("A DataEcon tensor must have plain axes only.")
        if length < 0 or length > MAX_INT64:
            raise ValueError("Invalid DataEcon tensor dimension.")
        shape.append(length)
    return tuple(shape)


def tensor_metadata(
    element: int, element_frequency: int, shape: tuple[int, ...], nbytes: int
) -> TensorMetadata:
    """Build the native-shaped metadata tuple for a tensor Python is about to write."""
    header = [4, 30, element, element_frequency, len(shape), nbytes]
    slots: list[int] = []
    for index in range(MAX_AXES):
        if index < len(shape):
            # The axis id is unknown before the native call; any positive
            # placeholder satisfies the "real axis" rule the reader applies.
            slots.extend((1, 0, int(shape[index]), 0, 0))
        else:
            slots.extend((-1, 0, 0, 0, 0))
    return tuple(header + slots)


def validate_tensor_payload(
    metadata: TensorMetadata, payload: bytes, marker: str | None, object_marker: str | None
) -> np.dtype[Any] | StoredElement | None:
    """Validate tensor metadata, finite markers and values without constructing output.

    Returns the resolved element, or None for a text tensor.
    """
    shape = validate_tensor_metadata(metadata)
    _, _, element, element_freq, _, _ = metadata[:6]
    _interpret.check_marker_text(marker, "element")
    _interpret.check_marker_text(object_marker, "whole-object")
    size = 1
    for length in shape:
        size *= length
    if element == KIND_STRING:
        _text_marker(marker, object_marker)
        split_text_payload(payload, size)
        return None
    resolved = _array_dtype(element, element_freq, size, len(payload), marker, object_marker)
    if isinstance(resolved, StoredElement):
        values = np.frombuffer(payload, dtype=resolved.dtype)
        resolve_array_interpretation(values.reshape(shape, order="F"), resolved, object_marker)
    elif marker == "Bool":
        _interpret.check_bool(
            np.frombuffer(payload, dtype=resolved), _interpret.numeric_target(resolved)
        )
    return resolved


def _encode_plain_tensor(value: np.ndarray[Any, Any]) -> TensorPayload:
    name, element = _ordinary_array_entry(value.dtype)
    shape = tuple(int(n) for n in value.shape)
    marker = (
        name if name == "Bool" or (not value.size and name != _SERIES_DEFAULTS[element]) else None
    )
    validate_tensor_metadata(tensor_metadata(element, 0, shape, value.nbytes))
    payload, storage, nbytes = _ordinary_snapshot(value, "F")
    resolved = validate_tensor_payload(
        tensor_metadata(element, 0, shape, nbytes), payload, marker, None
    )
    _check_snapshot_storage(resolved, storage, marker)
    return TensorPayload(30, element, 0, shape, payload, marker, None)


def _encode_stored_tensor(value: StoredArray) -> TensorPayload:
    element = value.element
    shape = value.shape
    size = 1
    for length in shape:
        size *= length
    marker = element.written_marker(size)
    payload = value.values.tobytes(order="F")
    if len(payload) != size * element.itemsize:
        raise ValueError("The array changed size during the snapshot; nothing was written.")
    metadata = tensor_metadata(element.native_kind, element.native_frequency, shape, len(payload))
    resolved = validate_tensor_payload(metadata, payload, marker, value.object_marker)
    if resolved != element:
        raise ValueError(
            "The snapshot no longer matches the container's stored element; nothing was written."
        )
    return TensorPayload(
        30,
        element.native_kind,
        element.native_frequency,
        shape,
        payload,
        marker,
        value.object_marker,
    )


def decode_tensor(
    metadata: TensorMetadata, payload: bytes, marker: str | None, object_marker: str | None
) -> np.ndarray[Any, Any] | StoredArray | StoredText:
    """Decode an owning three- to five-dimensional array from a column-major payload."""
    if sys.byteorder != "little":
        raise RuntimeError(
            "DataEcon interchange is currently supported on little-endian hosts only."
        )
    resolved = validate_tensor_payload(metadata, payload, marker, object_marker)
    shape = validate_tensor_metadata(metadata)
    if metadata[2] == KIND_STRING:
        return _decode_text_array(payload, marker, shape)
    if isinstance(resolved, StoredElement):
        values = np.frombuffer(payload, dtype=resolved.dtype).reshape(shape, order="F")
        return StoredArray(_owned(values), resolved, copy=False, object_marker=object_marker)
    assert resolved is not None
    values = np.frombuffer(payload, dtype=resolved).reshape(shape, order="F")
    return _owned(values == 1) if marker == "Bool" else _owned(values)
