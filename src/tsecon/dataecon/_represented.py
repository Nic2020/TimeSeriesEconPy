# SPDX-License-Identifier: MIT
"""Stored representations for DataEcon series that NumPy or Julia's markers keep apart.

Julia's dated ``TSeries`` and ``MVTSeries`` may hold ``MIT{F}`` dates and
``Duration{F}`` spans (raw Int64 codes with their own element frequency,
independent of the axis), ``Int128``/``UInt128`` integers and ``ComplexF16``
values. NumPy has no scalar type for any of them, so :class:`StoredSeries`
keeps such a series in its stored form: a dated anchor, a contiguous carrier
array with the exact stored bytes, and a :class:`StoredElement` describing the
element family. :class:`StoredMVTSeries` is the same representation for a
dated matrix: a dated row anchor, the column names and a two-dimensional
carrier, sharing the descriptor, the packers and the marker rules.

A file may also carry Julia's reconstruction markers: ``jeltype`` names an
element type the values are converted to on load, and ``jtype`` names a
whole-object type that takes precedence. Where such a marker asks for
something other than the stored family, the container preserves the stored
kind, element frequency, bytes and the exact marker text (``element.marker``
and :attr:`StoredSeries.object_marker`); :meth:`StoredSeries.to_interpreted`
returns Julia's converted value as an independently owning object without
changing what is stored. Explicit conversions (``tolist``, ``from_list``,
``to_complex64``, ``to_bool``, ``to_interpreted``) are the only paths to
ordinary Python or NumPy values. No marker text is ever evaluated.

This module holds no native resource. File codecs preserve these carrier
bytes and their validated metadata without implicit value conversion.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from tsecon.frequencies import Frequency
from tsecon.mit import MIT, Duration
from tsecon.mvtseries import MVTSeries
from tsecon.tseries import TSeries

from . import _interpret
from ._interpret import Target
from ._metadata import (
    _SCALAR_FREQUENCIES,
    _SERIES_FREQUENCIES,
    CODE_DTYPE,
    COMPLEXF16_DTYPE,
    INT128_DTYPE,
    JULIA_KIND_DEFAULTS,
    JULIA_NUMERIC_TYPES,
    KIND_COMPLEX,
    KIND_DATE,
    KIND_INTEGER,
    KIND_UNSIGNED,
    MAX_BYTES,
    MAX_INT64,
    MIN_INT64,
    julia_frequency_name,
)

__all__ = [
    "BOOL_MARKER",
    "CODE_DTYPE",
    "COMPLEXF16",
    "COMPLEXF16_DTYPE",
    "INT128",
    "INT128_DTYPE",
    "UINT128",
    "StoredElement",
    "StoredMVTSeries",
    "StoredSeries",
    "check_column_names",
    "element_tolist",
    "julia_frequency_name",
    "pack_complexf16",
    "pack_elements",
]

ElementKind = Literal["date", "duration", "int128", "uint128", "complexf16", "numeric"]

BOOL_MARKER = "Bool"
_WIDE_KINDS: frozenset[str] = frozenset({"int128", "uint128", "complexf16"})
_DATE_KINDS: frozenset[str] = frozenset({"date", "duration"})
_KINDS: frozenset[str] = _WIDE_KINDS | _DATE_KINDS | {"numeric"}
_DTYPES: dict[str, np.dtype[Any]] = {
    "date": CODE_DTYPE,
    "duration": CODE_DTYPE,
    "int128": INT128_DTYPE,
    "uint128": INT128_DTYPE,
    "complexf16": COMPLEXF16_DTYPE,
}
_NATIVE_KINDS: dict[str, int] = {
    "date": KIND_DATE,
    "duration": KIND_INTEGER,
    "int128": KIND_INTEGER,
    "uint128": KIND_UNSIGNED,
    "complexf16": KIND_COMPLEX,
}
_JULIA_NAMES: dict[str, str] = {
    "int128": "Int128",
    "uint128": "UInt128",
    "complexf16": "ComplexF16",
}
_ELEMENT_FREQUENCY_CODES: dict[Frequency, int] = {
    frequency: code for code, frequency in _SCALAR_FREQUENCIES.items()
}
# Ordinary numeric carrier dtypes (Boolean values belong in a TSeries).
_ORDINARY_DTYPES: dict[np.dtype[Any], str] = {
    dtype: name for name, (_, dtype) in JULIA_NUMERIC_TYPES.items() if dtype.kind != "b"
}
_MASK64 = (1 << 64) - 1
_FLOAT16_QUIET_NAN = np.uint16(0x7E00)
_FLOAT16_MAX = 65504.0


@dataclass(frozen=True, slots=True)
class StoredElement:
    """Finite description of a stored element family and its reconstruction marker.

    ``kind`` names the family; ``frequency`` is the element frequency of a
    date or duration element (independent of the series axis) and ``None``
    otherwise; ``marker`` is the preserved ``jeltype`` text, kept in its exact
    spelling and never evaluated; ``numeric_dtype`` is the carrier dtype of an
    ordinary numeric family kept in stored form because of a marker. Whether a
    marker can be interpreted is decided against the values by
    :class:`StoredSeries`; the descriptor itself only records it.
    """

    kind: ElementKind
    frequency: Frequency | None = None
    marker: str | None = None
    numeric_dtype: np.dtype[Any] | None = None

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise TypeError(f"Unsupported stored element kind: {self.kind!r}")
        if self.kind in _DATE_KINDS:
            if self.frequency not in _ELEMENT_FREQUENCY_CODES:
                raise TypeError(
                    "Date and duration elements need a supported element frequency "
                    "(Unit, Daily, BDaily, Weekly, Monthly, Quarterly, HalfYearly or Yearly)."
                )
        elif self.frequency is not None:
            raise TypeError(f"{self.kind} elements carry no element frequency.")
        if self.kind == "numeric":
            if (
                self.numeric_dtype is None
                or self.numeric_dtype.char in ("g", "G")
                or self.numeric_dtype not in _ORDINARY_DTYPES
            ):
                raise TypeError(
                    "Numeric stored elements need one of the ordinary native-endian series "
                    "dtypes (int8..int64, uint8..uint64, float16/32/64, complex64/128); "
                    "Boolean values belong in a TSeries."
                )
        elif self.numeric_dtype is not None:
            raise TypeError(f"{self.kind} elements have a fixed carrier dtype.")
        _interpret.check_marker_text(self.marker, "element")

    @classmethod
    def date(cls, frequency: Frequency) -> StoredElement:
        """Describe ``MIT{F}`` elements with element frequency ``F``."""
        return cls("date", frequency)

    @classmethod
    def duration(cls, frequency: Frequency) -> StoredElement:
        """Describe ``Duration{F}`` elements with element frequency ``F``."""
        return cls("duration", frequency)

    @classmethod
    def numeric(cls, dtype: Any, marker: str | None = None) -> StoredElement:
        """Describe an ordinary numeric family kept in stored form under a marker."""
        return cls("numeric", None, marker, np.dtype(dtype))

    def with_marker(self, marker: str | None) -> StoredElement:
        """Return this descriptor with another preserved ``jeltype`` text."""
        return StoredElement(self.kind, self.frequency, marker, self.numeric_dtype)

    def with_bool_marker(self) -> StoredElement:
        """Return this descriptor with the preserved ``"Bool"`` marker."""
        return self.with_marker(BOOL_MARKER)

    @property
    def is_wide(self) -> bool:
        """Whether this is a 128-bit integer or ComplexF16 carrier."""
        return self.kind in _WIDE_KINDS

    @property
    def is_numeric(self) -> bool:
        """Whether this is an ordinary numeric family kept in stored form."""
        return self.kind == "numeric"

    @property
    def dtype(self) -> np.dtype[Any]:
        """The carrier dtype holding the exact stored bytes."""
        if self.numeric_dtype is not None:
            return self.numeric_dtype
        return _DTYPES[self.kind]

    @property
    def itemsize(self) -> int:
        """Stored bytes per observation."""
        return self.dtype.itemsize

    @property
    def native_kind(self) -> int:
        """The native element type code (3 date, 1 integer, 2 unsigned, 4 float, 5 complex)."""
        if self.numeric_dtype is not None:
            return JULIA_NUMERIC_TYPES[_ORDINARY_DTYPES[self.numeric_dtype]][0]
        return _NATIVE_KINDS[self.kind]

    @property
    def native_frequency(self) -> int:
        """The native element frequency code, zero except for dates and durations."""
        if self.frequency is None:
            return 0
        return _ELEMENT_FREQUENCY_CODES[self.frequency]

    @property
    def julia_name(self) -> str:
        """Julia's name for the stored carrier type, used as a marker token only."""
        if self.frequency is not None:
            outer = "MIT" if self.kind == "date" else "Duration"
            return f"{outer}{{{julia_frequency_name(self.frequency)}}}"
        if self.numeric_dtype is not None:
            return _ORDINARY_DTYPES[self.numeric_dtype]
        return _JULIA_NAMES[self.kind]

    @property
    def target(self) -> Target:
        """The stored family as an interpretation target (no marker)."""
        if self.kind == "numeric":
            return _interpret.numeric_target(self.dtype)
        if self.kind in _WIDE_KINDS:
            return _interpret.wide_target(self.kind)
        assert self.frequency is not None
        return _interpret.date_target(self.kind, self.frequency)

    def written_marker(self, length: int) -> str | None:
        """Return the ``jeltype`` token a writer stores for a series of this length.

        A preserved marker is always written back in its exact spelling.
        Otherwise Julia's exact token is written only for empty series: an
        unmarked empty payload reads back as the native kind's default element
        (the element frequency and kind are stored independently of the token,
        so an empty date or duration series is recoverable without it and the
        token is written for byte parity with Julia's writer). Nonempty
        unmarked series carry no token.
        """
        if self.marker is not None:
            return self.marker
        if length:
            return None
        if self.kind == "numeric" and self.julia_name == JULIA_KIND_DEFAULTS[self.native_kind]:
            return None
        return self.julia_name


INT128 = StoredElement("int128")
UINT128 = StoredElement("uint128")
COMPLEXF16 = StoredElement("complexf16")


def _element_from_target(target: Target) -> StoredElement:
    if target.kind == "numeric":
        return StoredElement.numeric(target.dtype)
    if target.kind in _WIDE_KINDS:
        return StoredElement(target.kind)  # type: ignore[arg-type]
    assert target.frequency is not None
    return StoredElement(target.kind, target.frequency)  # type: ignore[arg-type]


def _pack_128(items: Iterable[object], signed: bool) -> np.ndarray[Any, Any]:
    low, high = (-(1 << 127), (1 << 127) - 1) if signed else (0, (1 << 128) - 1)
    numbers: list[int] = []
    for item in items:
        if type(item) is not int:
            raise TypeError("128-bit values must be Python int objects.")
        if not low <= item <= high:
            raise ValueError(f"{item} is outside the {'Int128' if signed else 'UInt128'} range.")
        numbers.append(item)
    return _interpret.pack_words(numbers)


def _float16_bits(component: float) -> np.uint16:
    """Return the float16 bit pattern of an exactly representable Python float."""
    if math.isnan(component):
        return _FLOAT16_QUIET_NAN
    if not math.isinf(component) and abs(component) > _FLOAT16_MAX:
        raise ValueError(f"{component!r} overflows float16 (largest finite value 65504).")
    narrowed = np.float16(component)
    if not math.isinf(component) and float(narrowed) != component:
        raise ValueError(f"{component!r} is not exactly representable as float16.")
    return narrowed.view(np.uint16)


def pack_complexf16(items: Iterable[object]) -> np.ndarray[Any, Any]:
    """Pack exact ``complex`` values or ``(float16, float16)`` pairs into a ComplexF16 carrier."""
    real_bits: list[np.uint16] = []
    imag_bits: list[np.uint16] = []
    for item in items:
        if type(item) is complex:
            real_bits.append(_float16_bits(item.real))
            imag_bits.append(_float16_bits(item.imag))
        elif (
            isinstance(item, tuple)
            and len(item) == 2
            and all(type(part) is np.float16 for part in item)
        ):
            real_bits.append(item[0].view(np.uint16))
            imag_bits.append(item[1].view(np.uint16))
        else:
            raise TypeError(
                "ComplexF16 values must be Python complex objects or (float16, float16) pairs."
            )
    out = np.empty(len(real_bits), dtype=COMPLEXF16_DTYPE)
    out["real"].view("<u2")[:] = np.array(real_bits, dtype="<u2")
    out["imag"].view("<u2")[:] = np.array(imag_bits, dtype="<u2")
    return out


class StoredSeries:
    """A dated series kept in its stored representation.

    ``values`` is the live carrier array of exactly ``element.dtype``. By
    default the constructor copies, so the container owns its carrier. With
    ``copy=False`` a 1-D, C-contiguous, writable ``ndarray`` of the exact
    dtype is shared with the caller (edits are visible through both names);
    any other compatible input, including read-only or strided arrays, is
    copied into a fresh owning writable array. Input of another dtype is
    refused rather than converted; build the carrier with :meth:`from_list`
    or an explicit ``np.array(..., dtype=element.dtype)``.

    ``object_marker`` is a preserved whole-object ``jtype`` text. When present
    it takes precedence: the element marker is kept as inactive data and is
    neither validated nor interpreted, exactly as Julia ignores it. The anchor,
    descriptor and object marker are read-only. The carrier's contents may be
    edited in place; its shape, dtype and layout are revalidated before every
    operation, and an active marker is validated against the current contents
    by :meth:`validate` and every conversion.
    """

    __slots__ = ("_element", "_firstdate", "_object_marker", "_values")

    def __init__(
        self,
        firstdate: MIT,
        values: np.ndarray[Any, Any],
        element: StoredElement,
        *,
        copy: bool = True,
        object_marker: str | None = None,
    ) -> None:
        if not isinstance(firstdate, MIT):
            raise TypeError("firstdate must be a core MIT anchor.")
        if firstdate.frequency not in _SERIES_FREQUENCIES.values():
            raise TypeError(
                "The series axis must be monthly, quarterly, half-yearly, annual, daily, "
                "business-daily or weekly."
            )
        if not isinstance(element, StoredElement):
            raise TypeError("element must be a StoredElement.")
        if not isinstance(values, np.ndarray):
            raise TypeError(
                "values must be a NumPy array of the element's carrier dtype; "
                "use StoredSeries.from_list for Python values."
            )
        if values.ndim != 1:
            raise ValueError("The carrier must be one-dimensional.")
        _interpret.check_marker_text(object_marker, "whole-object")
        element = canonical_element(element, object_marker, int(values.shape[0]))
        _check_input(values, element)
        if copy or not (values.flags.writeable and values.flags.c_contiguous):
            values = np.array(values, dtype=element.dtype, copy=True, order="C")
        self._firstdate = firstdate
        self._values = values
        self._element = element
        self._object_marker = object_marker
        self._check_interpretation()

    @classmethod
    def from_list(
        cls, firstdate: MIT, element: StoredElement, items: Iterable[object]
    ) -> StoredSeries:
        """Build a container from Python values with strict, explicit conversion.

        Dates and durations take core ``MIT``/``Duration`` objects of exactly
        the element frequency; 128-bit carriers take Python ``int`` values in
        range; ComplexF16 takes Python ``complex`` values whose finite
        components are exactly representable as float16 (infinities map to
        float16 infinities, NaNs to the canonical quiet NaN) or
        ``(np.float16, np.float16)`` pairs, which preserve every bit pattern.
        Ordinary numeric families are built from an explicit NumPy array of
        the carrier dtype instead, so that no implicit conversion is applied.
        """
        return cls(firstdate, pack_elements(element, items), element, copy=False)

    @property
    def firstdate(self) -> MIT:
        """The dated anchor of the axis."""
        return self._firstdate

    @property
    def element(self) -> StoredElement:
        """The stored element family, including any preserved element marker."""
        return self._element

    @property
    def object_marker(self) -> str | None:
        """The preserved whole-object ``jtype`` text, or ``None``."""
        return self._object_marker

    @property
    def values(self) -> np.ndarray[Any, Any]:
        """The live carrier array; its contents may be edited in place."""
        return self._values

    @property
    def frequency(self) -> Frequency:
        """The axis frequency, taken from the anchor."""
        return self._firstdate.frequency

    @property
    def lastdate(self) -> MIT:
        """The axis endpoint ``firstdate + len - 1``.

        For an empty series this is the synthetic date one period before the
        anchor, exactly as ``TSeries.lastdate``; it never names a stored
        observation.
        """
        return MIT(self._firstdate.frequency, self._firstdate.value + len(self) - 1)

    @property
    def active_marker(self) -> str | None:
        """The element marker Julia would act on: ``None`` under a whole-object marker."""
        return None if self._object_marker is not None else self._element.marker

    def __len__(self) -> int:
        _check_carrier(self._values, self._element)
        return int(self._values.shape[0])

    def __repr__(self) -> str:
        outer = "" if self._object_marker is None else f", object_marker={self._object_marker!r}"
        return (
            f"StoredSeries(firstdate={self._firstdate!r}, element={self._element!r}, "
            f"length={self._values.shape[0]}{outer})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StoredSeries):
            return NotImplemented
        _check_carrier(self._values, self._element)
        _check_carrier(other._values, other._element)
        return (
            self._element == other._element
            and self._object_marker == other._object_marker
            and self._firstdate == other._firstdate
            and self._values.tobytes(order="C") == other._values.tobytes(order="C")
        )

    __hash__ = None  # type: ignore[assignment]

    # ---- validation ------------------------------------------------------

    def _interpretation(self) -> tuple[str, Target]:
        """Resolve precedence for the live carrier; see :func:`resolve_interpretation`."""
        return resolve_interpretation(
            self._values, self._element, self._object_marker, self._firstdate.frequency
        )

    def _check_interpretation(self) -> None:
        _check_carrier(self._values, self._element)
        self._interpretation()

    def validate(self) -> None:
        """Check the live carrier and any preserved marker against its contents.

        Raises ``TypeError``/``ValueError`` if the array was changed in place
        to another shape, dtype or layout (it must stay one-dimensional and
        C-contiguous), exceeds the payload capacity, or if an active marker no
        longer describes a possible conversion of the current values (a
        ``"Bool"``-marked carrier holding a two, an ``Int8``-marked Int64
        carrier holding 300, an emptied marked wide carrier). Clearing the
        array's writeable flag is allowed and does not affect validity.
        """
        self._check_interpretation()

    # ---- explicit conversions --------------------------------------------

    def tolist(self) -> list[Any]:
        """Convert every stored observation to a Python object.

        Dates and durations become core ``MIT``/``Duration`` objects with the
        element frequency; 128-bit values become Python ``int``; ComplexF16
        values become Python ``complex`` (a NaN component loses its payload);
        ordinary numeric carriers give NumPy's ``tolist`` values. A preserved
        marker does not change this: the stored values are returned; use
        :meth:`to_interpreted` for Julia's converted values.
        """
        _check_carrier(self._values, self._element)
        return element_tolist(self._values, self._element)

    def to_complex64(self) -> TSeries:
        """Widen a ComplexF16 carrier exactly into a ``complex64`` ``TSeries``."""
        if self._element.kind != "complexf16":
            raise TypeError("to_complex64 applies to ComplexF16 carriers only.")
        values = self._conversion_snapshot()
        _interpret.check_output_capacity(
            int(values.shape[0]), _interpret.numeric_target(np.dtype("<c8"))
        )
        wide = np.empty(values.shape[0], dtype=np.complex64)
        wide.real = values["real"].astype(np.float32)
        wide.imag = values["imag"].astype(np.float32)
        return TSeries(self._firstdate, wide)

    def to_bool(self) -> TSeries:
        """Convert a carrier with an active ``"Bool"`` marker into a Boolean ``TSeries``.

        Every stored value must be exactly zero or one (a zero imaginary part
        and either signed zero are accepted, as in the Julia reference);
        anything else raises ``ValueError``. Carriers without an active Bool
        marker are refused.
        """
        if self.active_marker != BOOL_MARKER:
            raise TypeError('to_bool applies to carriers with an active "Bool" marker only.')
        values = self._conversion_snapshot()
        resolve_interpretation(
            values, self._element, self._object_marker, self._firstdate.frequency
        )
        return TSeries(self._firstdate, _interpret.bool_flags(values, self._element.target))

    def to_interpreted(self) -> TSeries | StoredSeries | np.ndarray[Any, Any]:
        """Return the value Julia's loader would build, without changing what is stored.

        The result is independently owning: a ``TSeries`` for ordinary numeric
        or Boolean targets, an unmarked ``StoredSeries`` for date, duration or
        wide targets (and for whole-object identity markers on such carriers),
        or an empty NumPy array for the supported ``Vector`` object markers.
        Floating targets reproduce Julia's rounding and finite overflow;
        integer, Boolean and date targets are exact or raise ``ValueError``.
        The output size is checked against the payload limit before the
        converted result array is allocated.
        """
        values = self._conversion_snapshot()
        kind, target = resolve_interpretation(
            values, self._element, self._object_marker, self._firstdate.frequency
        )
        if kind == "vector":
            assert self._object_marker is not None
            return np.empty(0, dtype=_interpret.vector_dtype(self._object_marker, target))
        if kind in ("identity", "stored"):
            if self._element.kind == "numeric":
                return TSeries(self._firstdate, values.copy())
            return StoredSeries(self._firstdate, values, _element_from_target(target))
        _interpret.check_output_capacity(int(values.shape[0]), target)
        converted = _interpret.convert_values(values, self._element.target, target)
        if target.kind == "numeric":
            return TSeries(self._firstdate, converted)
        return StoredSeries(self._firstdate, converted, _element_from_target(target), copy=False)

    def _conversion_snapshot(self) -> np.ndarray[Any, Any]:
        """Own one carrier snapshot used for both validation and conversion.

        Arbitrary concurrent mutation is outside the thread-safety contract. A
        conversion nevertheless validates and converts the same owned bytes, so
        a later edit of the caller-visible carrier cannot enter the result
        without validation.
        """
        _check_carrier(self._values, self._element)
        length = int(self._values.shape[0])
        payload = self._values.tobytes(order="C")
        if len(payload) != length * self._element.itemsize:
            raise ValueError("The series values changed size during the conversion snapshot.")
        return np.frombuffer(payload, dtype=self._element.dtype).copy()


def element_tolist(values: np.ndarray[Any, Any], element: StoredElement) -> list[Any]:
    """Convert one contiguous run of stored values into Python objects.

    Shared by the dated and plain-array containers so both report identical
    Python values for the same stored bytes.
    """
    if element.kind == "date":
        return [MIT(element.frequency, int(v)) for v in values.tolist()]  # type: ignore[arg-type]
    if element.kind == "duration":
        return [Duration(element.frequency, int(v)) for v in values.tolist()]  # type: ignore[arg-type]
    if element.kind == "complexf16":
        return [
            complex(float(r), float(i)) for r, i in zip(values["real"], values["imag"], strict=True)
        ]
    if element.kind == "numeric":
        return list(values.tolist())
    return _interpret.unpack_words(values, element.kind == "int128")


def pack_elements(element: StoredElement, items: Iterable[object]) -> np.ndarray[Any, Any]:
    """Pack Python values into a fresh one-dimensional carrier of ``element.dtype``.

    The strict rules of :meth:`StoredSeries.from_list` apply: core
    ``MIT``/``Duration`` objects of exactly the element frequency, Python
    ``int`` values in the 128-bit range, and Python ``complex`` values or
    ``(np.float16, np.float16)`` pairs for ComplexF16. Ordinary numeric
    families are refused; they are built from an explicit NumPy array.
    """
    if not isinstance(element, StoredElement):
        raise TypeError("element must be a StoredElement.")
    if element.kind in _DATE_KINDS:
        return _pack_codes(items, element)
    if element.kind == "complexf16":
        return pack_complexf16(items)
    if element.kind == "numeric":
        raise TypeError(
            "Build ordinary numeric stored values from an explicit NumPy array of the "
            "carrier dtype; from_list applies no implicit numeric conversion."
        )
    return _pack_128(items, element.kind == "int128")


def _pack_codes(items: Iterable[object], element: StoredElement) -> np.ndarray[Any, Any]:
    expected: type[MIT] | type[Duration] = MIT if element.kind == "date" else Duration
    codes: list[int] = []
    for item in items:
        if type(item) is not expected or item.frequency != element.frequency:
            raise TypeError(
                f"{element.kind} elements must be core {expected.__name__} objects "
                f"with element frequency {element.frequency!r}."
            )
        if not MIN_INT64 <= item.value <= MAX_INT64:
            raise ValueError(f"{item!r} is outside the Int64 element code range.")
        codes.append(item.value)
    return np.array(codes, dtype=CODE_DTYPE)


def _check_input(values: np.ndarray[Any, Any], element: StoredElement) -> None:
    """Validate constructor input before any copy; any layout is acceptable here."""
    # On Windows, NumPy longdouble/clongdouble compare equal to float64/
    # complex128 even though their explicit dtype chars remain g/G. Keep the
    # same cross-platform rejection used by the ordinary TSeries codec.
    if values.dtype.char in ("g", "G") or values.dtype != element.dtype:
        raise TypeError(
            f"values must have the {element.kind} carrier dtype {element.dtype!r}, "
            f"got {values.dtype!r}; no implicit conversion is applied."
        )
    if values.ndim != 1:
        raise ValueError("The carrier must be one-dimensional.")
    if values.shape[0] * element.itemsize > MAX_BYTES:
        raise ValueError(f"The series payload exceeds the {MAX_BYTES} byte limit.")


def _check_carrier(values: np.ndarray[Any, Any], element: StoredElement) -> None:
    """Validate the live carrier: input rules plus the promised contiguous layout."""
    _check_input(values, element)
    if not values.flags.c_contiguous:
        raise ValueError("The carrier must remain C-contiguous; its strides were changed.")


def canonical_element(
    element: StoredElement, object_marker: str | None, length: int
) -> StoredElement:
    """Drop only identity markers covered by the adapter before this slice.

    Exact represented/date markers and canonical empty numeric markers keep
    their established result and rewrite. Newly accepted ordinary nonempty
    identity markers and aliases remain preserved. An inactive marker under a
    whole-object marker is also kept verbatim.
    """
    if object_marker is not None or element.marker is None:
        return element
    target = _interpret.resolve_token(element.marker)
    represented_exact = element.kind != "numeric" and element.marker == element.julia_name
    empty_numeric_exact = (
        element.kind == "numeric"
        and length == 0
        and element.marker in JULIA_NUMERIC_TYPES
        and target == element.target
    )
    if represented_exact or empty_numeric_exact:
        return element.with_marker(None)
    return element


def resolve_interpretation(
    values: np.ndarray[Any, Any],
    element: StoredElement,
    object_marker: str | None,
    axis: Frequency,
) -> tuple[str, Target]:
    """Resolve marker precedence and check that the declared interpretation is possible.

    Returns ``("identity"|"vector"|"element"|"stored", target)``. A present
    whole-object marker wins and leaves the element marker inactive, as in
    Julia. ``TypeError`` names an unsupported token or route; ``ValueError`` a
    value the declared conversion cannot represent. Validation may allocate
    temporary masks, but it does not allocate the converted result array.
    """
    length = int(values.shape[0])
    base = element.target
    if object_marker is not None:
        if element.kind in _DATE_KINDS and not length:
            raise TypeError(
                "Julia cannot load an empty date or duration series; a whole-object "
                "marker on one is not supported."
            )
        return _interpret.object_interpretation(object_marker, axis, base, length), base
    if element.marker is None and element.kind == "numeric":
        raise TypeError(
            "Ordinary numeric values without a reconstruction marker belong in a "
            "TSeries, not a StoredSeries."
        )
    return element_interpretation(values, element, "series")


def element_interpretation(
    values: np.ndarray[Any, Any], element: StoredElement, noun: str
) -> tuple[str, Target]:
    """Resolve an active element marker against the flat stored values.

    Shared by every stored container (dated series, plain arrays and dated
    matrices), so one finite table and one set of route/value rules decide
    what Julia's ``map``-based element conversion would build. ``noun`` names
    the container in messages. Returns ``("stored", base)`` for an unmarked or
    redundant descriptor and ``("element", target)`` for a possible foreign
    conversion; ``TypeError`` names an unsupported token or route and
    ``ValueError`` a value the conversion cannot represent.
    """
    length = int(values.shape[0])
    base = element.target
    marker = element.marker
    if marker is None:
        return "stored", base
    target = _interpret.resolve_token(marker)
    if target is None:
        raise TypeError(
            f"Unsupported reconstruction marker {marker!r}; marker text is compared with "
            "a finite table and never evaluated."
        )
    if target == base:
        return "stored", base
    if element.kind == "numeric" and target.is_bool:
        result = "TSeries" if noun == "series" else noun
        raise TypeError(
            f'A "Bool" marker on an ordinary numeric payload reads as a Boolean {result}; '
            "build one with a bool dtype instead of a stored container."
        )
    if not length:
        if element.kind == "numeric":
            # Julia builds an empty typed vector for any target; the stored width
            # is unrecoverable, so the descriptor keeps the kind default.
            if element.julia_name != JULIA_KIND_DEFAULTS[element.native_kind]:
                raise TypeError(
                    "An empty numeric payload has no stored width; a foreign marker is "
                    f"kept on the kind default {JULIA_KIND_DEFAULTS[element.native_kind]}, "
                    f"not on {element.julia_name}."
                )
            return "element", target
        if element.is_wide:
            raise ValueError(
                f"An empty {element.julia_name} carrier with the foreign marker "
                f"{marker!r} has no storable width; use an empty numeric value or an "
                "unmarked wide carrier."
            )
        raise TypeError(
            f"Julia cannot load an empty {element.julia_name} {noun}; the foreign marker "
            f"{marker!r} on one is not supported."
        )
    _interpret.check_route(base, target)
    _interpret.check_values(values, base, target)
    return "element", target


# ---- dated matrices ---------------------------------------------------------


def check_column_names(columns: object) -> tuple[str, ...]:
    """Validate MVTSeries column names for DataEcon's names axis.

    The axis is one newline-joined, NUL-terminated string, so a name can
    contain neither a newline nor NUL; the empty string still splits into one
    name, so at least one column is required; Python's ``MVTSeries`` keys
    columns by name, so the names must be distinct. A single ``str`` is one
    name; otherwise every entry must be exactly ``str``.
    """
    if isinstance(columns, str):
        parts: tuple[str, ...] = (columns,)
    else:
        try:
            parts = tuple(columns)  # type: ignore[arg-type]
        except TypeError:
            raise TypeError("MVTSeries column names must be an iterable of str.") from None
    if not parts:
        raise ValueError(
            "DataEcon cannot store an MVTSeries with no columns; the names axis has no "
            "encoding for zero names."
        )
    for name in parts:
        if type(name) is not str:
            raise TypeError("MVTSeries column names must be plain Python strings.")
        if "\n" in name:
            raise ValueError(
                "A DataEcon column name cannot contain a newline; it separates the names."
            )
        if "\0" in name:
            raise ValueError(
                "A DataEcon column name cannot contain NUL; the names axis is stored as a "
                "NUL-terminated string and would be truncated."
            )
    if len(set(parts)) != len(parts):
        raise ValueError("DataEcon column names must be distinct.")
    return parts


def _check_matrix_input(values: np.ndarray[Any, Any], element: StoredElement, columns: int) -> None:
    """Validate a dated-matrix carrier before any copy: dtype, rank, columns, capacity."""
    if not isinstance(values, np.ndarray):
        raise TypeError(
            "values must be a NumPy array of the element's carrier dtype; "
            "use StoredMVTSeries.from_list for Python values."
        )
    if values.dtype.char in ("g", "G") or values.dtype != element.dtype:
        raise TypeError(
            f"values must have the {element.kind} carrier dtype {element.dtype!r}, "
            f"got {values.dtype!r}; no implicit conversion is applied."
        )
    if values.ndim != 2:
        raise ValueError("The carrier must be two-dimensional (rows by columns).")
    if int(values.shape[1]) != columns:
        raise ValueError(
            f"The carrier has {int(values.shape[1])} columns but {columns} column names were given."
        )
    if int(values.shape[0]) * columns * element.itemsize > MAX_BYTES:
        raise ValueError(f"The MVTSeries payload exceeds the {MAX_BYTES} byte limit.")


def _check_matrix_carrier(
    values: np.ndarray[Any, Any], element: StoredElement, columns: int
) -> None:
    _check_matrix_input(values, element, columns)
    if not values.flags.c_contiguous:
        raise ValueError("The carrier must remain C-contiguous; its strides were changed.")


def resolve_mvtseries_interpretation(
    values: np.ndarray[Any, Any],
    element: StoredElement,
    object_marker: str | None,
    axis: Frequency,
) -> tuple[str, Target]:
    """Resolve marker precedence for a dated matrix and check the declared conversion.

    Returns ``("identity"|"element"|"stored", target)``. A present
    whole-object marker wins and leaves the element marker inactive, as in
    Julia; the only supported whole-object tokens are identities (see
    :func:`_interpret.mvtseries_object_interpretation`). Element markers use
    the same finite table, routes and value rules as dated series and plain
    arrays, applied to the column-major flat values.
    """
    base = element.target
    flat = values.reshape(-1, order="F")
    if object_marker is not None:
        if element.kind in _DATE_KINDS and not flat.shape[0]:
            raise TypeError(
                "Julia cannot load an empty date or duration MVTSeries; a whole-object "
                "marker on one is not supported."
            )
        return _interpret.mvtseries_object_interpretation(object_marker, axis, base), base
    if element.marker is None and element.kind == "numeric":
        raise TypeError(
            "Ordinary numeric values without a reconstruction marker belong in an "
            "MVTSeries, not a StoredMVTSeries."
        )
    return element_interpretation(flat, element, "MVTSeries")


class StoredMVTSeries:
    """A dated matrix (Julia's ``MVTSeries``) kept in its stored representation.

    ``firstdate`` is the row anchor, ``columns`` the distinct column names and
    ``values`` the live carrier: a two-dimensional (rows by columns),
    C-contiguous NumPy array of exactly ``element.dtype`` holding the stored
    bytes (the adapter writes DataEcon's column-major payload from it). The
    constructor copies by default; with ``copy=False`` a writable, C-contiguous
    array of the exact dtype is shared with the caller and anything else is
    copied. Input of another dtype is refused rather than converted; build the
    carrier with :meth:`from_list` or an explicit ``np.array(..., dtype=...)``.

    ``object_marker`` is a preserved whole-object ``jtype`` text; when present
    it takes precedence and the element marker is inactive, as in Julia. The
    anchor, names, descriptor and object marker are read-only; the carrier's
    contents may be edited in place and its shape, dtype and layout are
    revalidated before every operation. Explicit conversions (``tolist``,
    ``to_bool``, ``to_complex64``, ``to_interpreted``) are the only paths to
    ordinary Python, NumPy or core ``MVTSeries`` values.
    """

    __slots__ = ("_columns", "_element", "_firstdate", "_object_marker", "_values")

    def __init__(
        self,
        firstdate: MIT,
        columns: object,
        values: np.ndarray[Any, Any],
        element: StoredElement,
        *,
        copy: bool = True,
        object_marker: str | None = None,
    ) -> None:
        if not isinstance(firstdate, MIT):
            raise TypeError("firstdate must be a core MIT anchor.")
        if firstdate.frequency not in _SERIES_FREQUENCIES.values():
            raise TypeError(
                "The row axis must be monthly, quarterly, half-yearly, annual, daily, "
                "business-daily, weekly or Unit."
            )
        if not isinstance(element, StoredElement):
            raise TypeError("element must be a StoredElement.")
        names = check_column_names(columns)
        _interpret.check_marker_text(object_marker, "whole-object")
        _check_matrix_input(values, element, len(names))
        element = canonical_element(element, object_marker, int(values.size))
        if copy or not (values.flags.writeable and values.flags.c_contiguous):
            values = np.array(values, dtype=element.dtype, copy=True, order="C")
        self._firstdate = firstdate
        self._columns = names
        self._values = values
        self._element = element
        self._object_marker = object_marker
        self._check_interpretation()

    @classmethod
    def from_list(
        cls,
        firstdate: MIT,
        columns: object,
        element: StoredElement,
        rows: Iterable[Iterable[object]],
    ) -> StoredMVTSeries:
        """Build a container from nested Python values, one inner sequence per row.

        Every row must hold exactly one value per column name; an empty outer
        sequence gives a zero-row matrix. Values follow the strict rules of
        :func:`pack_elements` (core ``MIT``/``Duration`` of the element
        frequency, in-range ``int`` for 128-bit carriers, ``complex`` or
        ``(np.float16, np.float16)`` pairs for ComplexF16); ordinary numeric
        families are built from an explicit NumPy array instead.
        """
        names = check_column_names(columns)
        flat: list[object] = []
        count = 0
        for row in rows:
            entries = list(row)
            if len(entries) != len(names):
                raise ValueError(
                    f"Row {count} holds {len(entries)} values for {len(names)} columns."
                )
            flat.extend(entries)
            count += 1
        # A reshaped view would not own its buffer; one owning copy of the
        # freshly packed values keeps the container independent (setting
        # `shape` in place is deprecated from NumPy 2.5).
        packed = pack_elements(element, flat).reshape((count, len(names))).copy(order="C")
        return cls(firstdate, names, packed, element, copy=False)

    # ---- accessors -------------------------------------------------------

    @property
    def firstdate(self) -> MIT:
        """The dated anchor of the row axis."""
        return self._firstdate

    @property
    def columns(self) -> tuple[str, ...]:
        """The distinct column names, in stored order."""
        return self._columns

    @property
    def element(self) -> StoredElement:
        """The stored element family, including any preserved element marker."""
        return self._element

    @property
    def object_marker(self) -> str | None:
        """The preserved whole-object ``jtype`` text, or ``None``."""
        return self._object_marker

    @property
    def values(self) -> np.ndarray[Any, Any]:
        """The live rows-by-columns carrier; its contents may be edited in place."""
        return self._values

    @property
    def frequency(self) -> Frequency:
        """The row axis frequency, taken from the anchor."""
        return self._firstdate.frequency

    @property
    def shape(self) -> tuple[int, int]:
        """``(rows, columns)`` of the carrier."""
        return (int(self._values.shape[0]), int(self._values.shape[1]))

    @property
    def lastdate(self) -> MIT:
        """The row axis endpoint ``firstdate + rows - 1`` (synthetic when empty)."""
        return MIT(self._firstdate.frequency, self._firstdate.value + len(self) - 1)

    @property
    def active_marker(self) -> str | None:
        """The element marker Julia would act on: ``None`` under a whole-object marker."""
        return None if self._object_marker is not None else self._element.marker

    def __len__(self) -> int:
        _check_matrix_carrier(self._values, self._element, len(self._columns))
        return int(self._values.shape[0])

    def __repr__(self) -> str:
        outer = "" if self._object_marker is None else f", object_marker={self._object_marker!r}"
        return (
            f"StoredMVTSeries(firstdate={self._firstdate!r}, columns={self._columns!r}, "
            f"element={self._element!r}, shape={self.shape}{outer})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StoredMVTSeries):
            return NotImplemented
        _check_matrix_carrier(self._values, self._element, len(self._columns))
        _check_matrix_carrier(other._values, other._element, len(other._columns))
        return (
            self._element == other._element
            and self._object_marker == other._object_marker
            and self._firstdate == other._firstdate
            and self._columns == other._columns
            and self.shape == other.shape
            and self._values.tobytes(order="C") == other._values.tobytes(order="C")
        )

    __hash__ = None  # type: ignore[assignment]

    # ---- validation ------------------------------------------------------

    def _check_interpretation(self) -> tuple[str, Target]:
        _check_matrix_carrier(self._values, self._element, len(self._columns))
        return resolve_mvtseries_interpretation(
            self._values, self._element, self._object_marker, self._firstdate.frequency
        )

    def validate(self) -> None:
        """Check the live carrier and any preserved marker against its contents.

        Raises ``TypeError``/``ValueError`` if the array was changed in place
        to another shape, dtype or layout (it must stay two-dimensional with
        one column per name and C-contiguous), exceeds the payload capacity,
        or if an active marker no longer describes a possible conversion of
        the current values.
        """
        self._check_interpretation()

    # ---- explicit conversions --------------------------------------------

    def tolist(self) -> list[list[Any]]:
        """Convert the stored values to Python objects, one inner list per row.

        Dates and durations become core ``MIT``/``Duration`` objects with the
        element frequency, 128-bit values Python ``int``, ComplexF16 values
        Python ``complex`` and ordinary numeric carriers NumPy's ``tolist``
        values. A preserved marker does not change this; use
        :meth:`to_interpreted` for Julia's converted values.
        """
        _check_matrix_carrier(self._values, self._element, len(self._columns))
        rows, columns = self.shape
        flat = element_tolist(np.ascontiguousarray(self._values).reshape(-1), self._element)
        return [flat[i * columns : (i + 1) * columns] for i in range(rows)]

    def to_complex64(self) -> MVTSeries:
        """Widen a ComplexF16 carrier exactly into a ``complex64`` ``MVTSeries``."""
        if self._element.kind != "complexf16":
            raise TypeError("to_complex64 applies to ComplexF16 carriers only.")
        values = self._conversion_snapshot()
        _interpret.check_output_capacity(
            int(values.size), _interpret.numeric_target(np.dtype("<c8"))
        )
        wide = np.empty(values.shape, dtype=np.complex64)
        wide.real = values["real"].astype(np.float32)
        wide.imag = values["imag"].astype(np.float32)
        return MVTSeries(self._firstdate, list(self._columns), wide)

    def to_bool(self) -> MVTSeries:
        """Convert a carrier with an active ``"Bool"`` marker into a Boolean ``MVTSeries``.

        Every stored value must be exactly zero or one (a zero imaginary part
        and either signed zero are accepted, as in the Julia reference);
        anything else raises ``ValueError``.
        """
        if self.active_marker != BOOL_MARKER:
            raise TypeError('to_bool applies to carriers with an active "Bool" marker only.')
        values = self._conversion_snapshot()
        resolve_mvtseries_interpretation(
            values, self._element, self._object_marker, self._firstdate.frequency
        )
        flags = _interpret.bool_flags(values.reshape(-1), self._element.target)
        return MVTSeries(self._firstdate, list(self._columns), flags.reshape(values.shape))

    def to_interpreted(self) -> MVTSeries | StoredMVTSeries:
        """Return the value Julia's loader would build, without changing what is stored.

        The result is independently owning: an ``MVTSeries`` for ordinary
        numeric or Boolean targets, an unmarked ``StoredMVTSeries`` for date,
        duration or wide targets and for whole-object identity markers.
        Floating targets reproduce Julia's rounding and finite overflow;
        integer, Boolean and date targets are exact or raise ``ValueError``.
        """
        values = self._conversion_snapshot()
        kind, target = resolve_mvtseries_interpretation(
            values, self._element, self._object_marker, self._firstdate.frequency
        )
        names = list(self._columns)
        if kind in ("identity", "stored"):
            if self._element.kind == "numeric":
                return MVTSeries(self._firstdate, names, values.copy())
            return StoredMVTSeries(self._firstdate, names, values, _element_from_target(target))
        _interpret.check_output_capacity(int(values.size), target)
        flat = _interpret.convert_values(values.reshape(-1), self._element.target, target)
        converted = flat.reshape(values.shape)
        if target.kind == "numeric":
            return MVTSeries(self._firstdate, names, converted)
        return StoredMVTSeries(
            self._firstdate, names, converted, _element_from_target(target), copy=False
        )

    def _conversion_snapshot(self) -> np.ndarray[Any, Any]:
        """Own one carrier snapshot used for both validation and conversion.

        The shape and expected byte count are captured before the bytes are
        taken and compared afterwards, so a carrier resized mid-call is
        refused rather than silently converted at a shape nothing validated.
        """
        _check_matrix_carrier(self._values, self._element, len(self._columns))
        shape = self.shape
        expected = shape[0] * shape[1] * self._element.itemsize
        payload = self._values.tobytes(order="C")
        if len(payload) != expected:
            raise ValueError("The MVTSeries values changed size during the conversion snapshot.")
        return np.frombuffer(payload, dtype=self._element.dtype).reshape(shape).copy()
