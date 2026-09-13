# SPDX-License-Identifier: MIT
"""Stored representations for DataEcon series elements NumPy cannot name.

Julia's dated ``TSeries`` may hold ``MIT{F}`` dates and ``Duration{F}`` spans
(raw Int64 codes with their own element frequency, independent of the axis),
``Int128``/``UInt128`` integers and ``ComplexF16`` values. NumPy has no scalar
type for any of them, so :class:`StoredSeries` keeps such a series in its
stored form: a dated anchor, a contiguous carrier array with the exact stored
bytes, and a :class:`StoredElement` describing the element family. Explicit
conversions (``tolist``, ``from_list``, ``to_complex64``, ``to_bool``) are the
only paths to ordinary Python or NumPy values.

This module holds no native resource. File codecs preserve these carrier
bytes and their validated metadata without implicit value conversion.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from tsecon.dataecon._metadata import (
    _SCALAR_FREQUENCIES,
    _SERIES_FREQUENCIES,
    KIND_COMPLEX,
    KIND_DATE,
    KIND_INTEGER,
    KIND_UNSIGNED,
    MAX_BYTES,
    MAX_INT64,
    MIN_INT64,
)
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

ElementKind = Literal["date", "duration", "int128", "uint128", "complexf16"]

# Carrier dtypes: the exact little-endian stored bytes. Int128 and UInt128 are
# the two 64-bit words with the low word first; ComplexF16 is the real
# component followed by the imaginary one, each an IEEE binary16.
CODE_DTYPE: np.dtype[Any] = np.dtype("<i8")
INT128_DTYPE: np.dtype[Any] = np.dtype([("lo", "<u8"), ("hi", "<u8")])
COMPLEXF16_DTYPE: np.dtype[Any] = np.dtype([("real", "<f2"), ("imag", "<f2")])
BOOL_MARKER = "Bool"
_WIDE_KINDS: frozenset[str] = frozenset({"int128", "uint128", "complexf16"})
_DATE_KINDS: frozenset[str] = frozenset({"date", "duration"})
_KINDS: frozenset[str] = _WIDE_KINDS | _DATE_KINDS
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
_MASK64 = (1 << 64) - 1
_FLOAT16_QUIET_NAN = np.uint16(0x7E00)
_FLOAT16_MAX = 65504.0


_PLAIN_JULIA_FREQUENCIES: dict[type[Frequency], str] = {
    Unit: "Unit",
    Daily: "Daily",
    BDaily: "BDaily",
    Monthly: "Monthly",
}
_PARAMETRIC_JULIA_FREQUENCIES: dict[type[Frequency], tuple[str, str]] = {
    Weekly: ("Weekly", "end_day"),
    Quarterly: ("Quarterly", "end_month"),
    HalfYearly: ("HalfYearly", "end_month"),
    Yearly: ("Yearly", "end_month"),
}


def julia_frequency_name(frequency: Frequency) -> str:
    """Return Julia's spelling of a supported frequency (``Weekly{7}``, ``Monthly``...)."""
    cls = type(frequency)
    if cls in _PLAIN_JULIA_FREQUENCIES:
        return _PLAIN_JULIA_FREQUENCIES[cls]
    if cls in _PARAMETRIC_JULIA_FREQUENCIES:
        name, attribute = _PARAMETRIC_JULIA_FREQUENCIES[cls]
        return f"{name}{{{getattr(frequency, attribute)}}}"
    raise TypeError(f"Unsupported element frequency: {frequency!r}")


@dataclass(frozen=True, slots=True)
class StoredElement:
    """Finite description of a stored element family.

    ``kind`` names the family; ``frequency`` is the element frequency of a
    date or duration element (independent of the series axis) and ``None``
    otherwise; ``marker`` is the preserved foreign interpretation marker,
    which is only ever ``"Bool"`` and only on the 128-bit and ComplexF16
    carriers. The descriptor is compared by value and never evaluates a
    marker string.
    """

    kind: ElementKind
    frequency: Frequency | None = None
    marker: str | None = None

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
        if self.marker is not None and (self.marker != BOOL_MARKER or self.kind not in _WIDE_KINDS):
            raise TypeError(
                'Only the "Bool" interpretation marker on int128, uint128 or complexf16 '
                "carriers is supported."
            )

    @classmethod
    def date(cls, frequency: Frequency) -> StoredElement:
        """Describe ``MIT{F}`` elements with element frequency ``F``."""
        return cls("date", frequency)

    @classmethod
    def duration(cls, frequency: Frequency) -> StoredElement:
        """Describe ``Duration{F}`` elements with element frequency ``F``."""
        return cls("duration", frequency)

    def with_bool_marker(self) -> StoredElement:
        """Return this wide carrier descriptor with the preserved ``"Bool"`` marker."""
        return StoredElement(self.kind, self.frequency, BOOL_MARKER)

    @property
    def is_wide(self) -> bool:
        """Whether this is a 128-bit integer or ComplexF16 carrier."""
        return self.kind in _WIDE_KINDS

    @property
    def dtype(self) -> np.dtype[Any]:
        """The carrier dtype holding the exact stored bytes."""
        return _DTYPES[self.kind]

    @property
    def itemsize(self) -> int:
        """Stored bytes per observation: 8, 16 or 4."""
        return self.dtype.itemsize

    @property
    def native_kind(self) -> int:
        """The native element type code (3 date, 1 integer, 2 unsigned, 5 complex)."""
        return _NATIVE_KINDS[self.kind]

    @property
    def native_frequency(self) -> int:
        """The native element frequency code, zero for the wide carriers."""
        if self.frequency is None:
            return 0
        return _ELEMENT_FREQUENCY_CODES[self.frequency]

    @property
    def julia_name(self) -> str:
        """Julia's name for the stored carrier type, used as a marker token only."""
        if self.frequency is not None:
            outer = "MIT" if self.kind == "date" else "Duration"
            return f"{outer}{{{julia_frequency_name(self.frequency)}}}"
        return _JULIA_NAMES[self.kind]

    def written_marker(self, length: int) -> str | None:
        """Return the ``jeltype`` token a writer stores for a series of this length.

        A preserved ``"Bool"`` marker is always written back. Otherwise Julia's
        exact token is written only for empty series: an unmarked empty 128-bit
        or ComplexF16 payload would read back as the native kind's wide default
        (the element frequency and kind are stored independently of the token,
        so an empty date or duration series is recoverable without it and the
        token is written for byte parity with Julia's writer). Nonempty stored
        series carry no token.
        """
        if self.marker is not None:
            return self.marker
        return self.julia_name if length == 0 else None


INT128 = StoredElement("int128")
UINT128 = StoredElement("uint128")
COMPLEXF16 = StoredElement("complexf16")


def _unpack_128(values: np.ndarray[Any, Any], signed: bool) -> list[int]:
    lo = values["lo"].astype(object)
    hi = values["hi"].astype(object)
    combined = lo | (hi << 64)
    if signed:
        combined = np.where(values["hi"] >= (1 << 63), combined - (1 << 128), combined)
    return [int(v) for v in combined]


def _pack_128(items: Iterable[object], signed: bool) -> np.ndarray[Any, Any]:
    low, high = (-(1 << 127), (1 << 127) - 1) if signed else (0, (1 << 128) - 1)
    words: list[tuple[int, int]] = []
    for item in items:
        if type(item) is not int:
            raise TypeError("128-bit values must be Python int objects.")
        if not low <= item <= high:
            raise ValueError(f"{item} is outside the {'Int128' if signed else 'UInt128'} range.")
        unsigned = item % (1 << 128)
        words.append((unsigned & _MASK64, (unsigned >> 64) & _MASK64))
    out = np.empty(len(words), dtype=INT128_DTYPE)
    for index, pair in enumerate(words):
        out[index] = pair
    return out


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


def _pack_complexf16(items: Iterable[object]) -> np.ndarray[Any, Any]:
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

    The anchor and descriptor are read-only. The carrier's contents may be
    edited in place; its shape, dtype and layout are revalidated before every
    operation, and a preserved ``"Bool"`` marker is validated against the
    current contents by :meth:`validate` and :meth:`to_bool`.
    """

    __slots__ = ("_element", "_firstdate", "_values")

    def __init__(
        self,
        firstdate: MIT,
        values: np.ndarray[Any, Any],
        element: StoredElement,
        *,
        copy: bool = True,
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
        _check_input(values, element)
        if copy or not (values.flags.writeable and values.flags.c_contiguous):
            values = np.array(values, dtype=element.dtype, copy=True, order="C")
        if element.marker is not None:
            _check_bool_interpretation(values, element)
        self._firstdate = firstdate
        self._values = values
        self._element = element

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
        """
        if not isinstance(element, StoredElement):
            raise TypeError("element must be a StoredElement.")
        if element.kind in _DATE_KINDS:
            packed = _pack_codes(items, element)
        elif element.kind == "complexf16":
            packed = _pack_complexf16(items)
        else:
            packed = _pack_128(items, element.kind == "int128")
        return cls(firstdate, packed, element, copy=False)

    @property
    def firstdate(self) -> MIT:
        """The dated anchor of the axis."""
        return self._firstdate

    @property
    def element(self) -> StoredElement:
        """The stored element family, including any preserved marker."""
        return self._element

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

    def __len__(self) -> int:
        _check_carrier(self._values, self._element)
        return int(self._values.shape[0])

    def __repr__(self) -> str:
        return (
            f"StoredSeries(firstdate={self._firstdate!r}, element={self._element!r}, "
            f"length={self._values.shape[0]})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StoredSeries):
            return NotImplemented
        _check_carrier(self._values, self._element)
        _check_carrier(other._values, other._element)
        return (
            self._element == other._element
            and self._firstdate == other._firstdate
            and self._values.tobytes(order="C") == other._values.tobytes(order="C")
        )

    __hash__ = None  # type: ignore[assignment]

    def validate(self) -> None:
        """Check the live carrier and any preserved marker against its contents.

        Raises ``TypeError``/``ValueError`` if the array was changed in place
        to another shape, dtype or layout (it must stay one-dimensional and
        C-contiguous), exceeds the payload capacity, or if a ``"Bool"``-marked
        carrier no longer holds only exact zeros and ones or was emptied (an
        empty marked payload has no storable width). Clearing the array's
        writeable flag is allowed and does not affect validity.
        """
        _check_carrier(self._values, self._element)
        if self._element.marker is not None:
            _check_bool_interpretation(self._values, self._element)

    def tolist(self) -> list[Any]:
        """Convert every stored observation to a Python object.

        Dates and durations become core ``MIT``/``Duration`` objects with the
        element frequency; 128-bit values become Python ``int``; ComplexF16
        values become Python ``complex`` (a NaN component loses its payload).
        A preserved ``"Bool"`` marker does not change this: the stored values
        are returned; use :meth:`to_bool` for Boolean output.
        """
        _check_carrier(self._values, self._element)
        element = self._element
        if element.kind == "date":
            return [MIT(element.frequency, int(v)) for v in self._values.tolist()]  # type: ignore[arg-type]
        if element.kind == "duration":
            return [Duration(element.frequency, int(v)) for v in self._values.tolist()]  # type: ignore[arg-type]
        if element.kind == "complexf16":
            return [
                complex(float(r), float(i))
                for r, i in zip(self._values["real"], self._values["imag"], strict=True)
            ]
        return _unpack_128(self._values, element.kind == "int128")

    def to_complex64(self) -> TSeries:
        """Widen a ComplexF16 carrier exactly into a ``complex64`` ``TSeries``."""
        _check_carrier(self._values, self._element)
        if self._element.kind != "complexf16":
            raise TypeError("to_complex64 applies to ComplexF16 carriers only.")
        wide = np.empty(self._values.shape[0], dtype=np.complex64)
        wide.real = self._values["real"].astype(np.float32)
        wide.imag = self._values["imag"].astype(np.float32)
        return TSeries(self._firstdate, wide)

    def to_bool(self) -> TSeries:
        """Convert a ``"Bool"``-marked wide carrier into a Boolean ``TSeries``.

        Every stored value must be exactly zero or one (a zero imaginary part
        and either signed zero are accepted, as in the Julia reference);
        anything else raises ``ValueError``. Unmarked carriers are refused.
        """
        _check_carrier(self._values, self._element)
        if self._element.marker is None:
            raise TypeError('to_bool applies to carriers with the preserved "Bool" marker only.')
        _check_bool_interpretation(self._values, self._element)
        return TSeries(self._firstdate, _bool_flags(self._values, self._element))


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
    if values.dtype != element.dtype:
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


def _bool_flags(values: np.ndarray[Any, Any], element: StoredElement) -> np.ndarray[Any, Any]:
    """Return the Boolean reading of a validated carrier."""
    if element.kind == "complexf16":
        return np.asarray(values["real"] == 1, dtype=bool)
    return np.asarray(values["lo"] == 1, dtype=bool)


def _check_bool_interpretation(values: np.ndarray[Any, Any], element: StoredElement) -> None:
    """Enforce the marker rules shared by construction, validate and to_bool."""
    if values.shape[0] == 0:
        raise ValueError(
            "An empty Bool-marked wide carrier has no storable width; "
            "use an empty Boolean TSeries instead."
        )
    if element.kind == "complexf16":
        real = values["real"]
        imag = values["imag"]
        valid = ((real == 0) | (real == 1)) & (imag == 0)
    else:
        valid = (values["hi"] == 0) & (values["lo"] <= 1)
    bad = np.flatnonzero(~valid)
    if bad.size:
        raise ValueError(
            f'A "Bool"-marked {element.julia_name} carrier must hold only exact zeros and '
            f"ones; observation {int(bad[0])} does not."
        )
