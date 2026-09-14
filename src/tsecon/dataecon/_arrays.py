# SPDX-License-Identifier: MIT
"""Stored representations for plain DataEcon arrays and text vectors.

Ordinary numeric and Boolean arrays travel as plain NumPy arrays and ordinary
text as a list of Python strings. Two small adapter containers carry what those
familiar types cannot hold losslessly:

* :class:`StoredArray` pairs a contiguous carrier with a
  :class:`~tsecon.dataecon._represented.StoredElement`, so ``MIT``/``Duration``
  codes, 128-bit integers, ComplexF16 pairs and preserved reconstruction
  markers keep their stored kind, width and bytes. It holds one- and
  two-dimensional values; :meth:`StoredArray.to_interpreted` is the explicit
  conversion to the value Julia's loader would build.
* :class:`StoredText` keeps each element's exact stored bytes together with the
  preserved ``jeltype`` text, which is how a ``Symbol`` vector and text that is
  not valid UTF-8 survive a read-modify-write.

Neither container evaluates marker text, and neither is part of the core
package: they exist only at the DataEcon adapter boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _interpret
from ._interpret import Target
from ._metadata import JULIA_KIND_DEFAULTS
from ._represented import (
    BOOL_MARKER,
    StoredElement,
    _element_from_target,
    element_tolist,
)

__all__ = ["StoredArray", "StoredText"]

# Julia writes `jeltype = "String"` on an empty String vector and omits it
# otherwise; both spell the stored element itself and are dropped on rewrite.
CANONICAL_TEXT_MARKERS = ("String",)


def _check_array_values(values: np.ndarray[Any, Any], element: StoredElement) -> None:
    """Validate an input array's type, rank, exact dtype and capacity.

    Windows `np.longdouble`/`np.clongdouble` compare equal to float64/complex128
    while keeping their own type codes, so the dtype character is checked
    explicitly, exactly as the dated container does. Nothing here converts: a
    mismatched width or a fractional value must be refused, never silently cast.
    """
    if not isinstance(values, np.ndarray):
        raise TypeError("A StoredArray carrier must be a NumPy array.")
    if values.ndim not in (1, 2):
        raise ValueError("A StoredArray carrier must be one- or two-dimensional.")
    if (
        values.dtype != element.dtype
        or values.dtype.char != element.dtype.char
        or not values.dtype.isnative
    ):
        raise TypeError(
            f"A {element.julia_name} carrier must already use the {element.dtype!r} storage "
            "dtype; values are never converted to fit a descriptor."
        )
    _interpret.check_output_capacity(int(values.size), element.target)


def _check_array_carrier(values: np.ndarray[Any, Any], element: StoredElement) -> None:
    """Validate a retained carrier, which must also stay C-contiguous."""
    _check_array_values(values, element)
    if not values.flags["C_CONTIGUOUS"]:
        raise ValueError("A StoredArray carrier must stay C-contiguous.")


def _conjugated(values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Julia's ``conj`` on a stored carrier, without leaving its storage form."""
    if values.dtype.fields is not None:
        if "imag" in values.dtype.fields:
            out = values.copy(order="C")
            out["imag"] = -out["imag"]
            return out
        # The two 128-bit integer carriers are real; conjugation is identity.
        return values
    if values.dtype.kind == "c":
        return values.conj()
    return values


def _clear_diagonal_imaginary(out: np.ndarray[Any, Any], rows: int) -> None:
    """Make the diagonal real, as ``Hermitian`` does, keeping the real part's sign."""
    index = np.arange(rows)
    if out.dtype.fields is not None:
        if "imag" in out.dtype.fields:
            out["imag"][index, index] = 0
        return
    if out.dtype.kind == "c":
        out[index, index] = out[index, index].real


def _structure_dense(values: np.ndarray[Any, Any], token: str) -> np.ndarray[Any, Any]:
    """Reproduce Julia's LinearAlgebra reconstruction of a dense square matrix.

    ``Diagonal`` keeps only the diagonal, and ``Symmetric``/``Hermitian`` mirror
    the upper triangle, which is the triangle Julia's own writer materialised
    when it stored the value. ``Hermitian`` also conjugates the mirrored entries
    and makes the diagonal real.

    The mirroring copies elements rather than adding a zeroed triangle: adding
    a structural ``+0.0`` to a stored ``-0.0`` would silently clear a sign the
    reference preserves. Copying also keeps the two-word 128-bit and ComplexF16
    carriers in range, so every supported element works here.
    """
    rows, columns = values.shape
    if rows != columns:
        raise ValueError(f"The {token} reconstruction marker applies to square matrices only.")
    if token == "Diagonal":
        out = np.zeros_like(values)
        index = np.arange(rows)
        out[index, index] = values[index, index]
        return out
    out = values.copy(order="C")
    upper_rows, upper_columns = np.triu_indices(rows, 1)
    mirrored = values[upper_rows, upper_columns]
    if token == "Hermitian":
        mirrored = _conjugated(mirrored)
    out[upper_columns, upper_rows] = mirrored
    if token == "Hermitian":
        _clear_diagonal_imaginary(out, rows)
    return out


class StoredArray:
    """A plain DataEcon array kept in its stored element representation.

    ``values`` is a one- or two-dimensional C-contiguous NumPy carrier holding
    the exact stored bytes for ``element``; a two-dimensional carrier is in the
    natural NumPy row-major order and the adapter writes DataEcon's
    column-major payload from it. ``object_marker`` is a preserved ``jtype``
    text. The container is copied by default so the file's bytes cannot be
    mutated through an alias.
    """

    __slots__ = ("_element", "_object_marker", "_values")

    def __init__(
        self,
        values: np.ndarray[Any, Any],
        element: StoredElement,
        *,
        copy: bool = True,
        object_marker: str | None = None,
    ) -> None:
        if not isinstance(element, StoredElement):
            raise TypeError("A StoredArray needs a StoredElement descriptor.")
        _interpret.check_marker_text(object_marker, "whole-object")
        # Validate the caller's actual array first; only then copy it, so a
        # wrong width or a noninteger value is refused instead of cast.
        _check_array_values(values, element)
        if copy:
            values = values.copy(order="C")
        _check_array_carrier(values, element)
        self._values = values
        self._element = element
        self._object_marker = object_marker
        self._interpretation()

    # ---- accessors -------------------------------------------------------

    @property
    def values(self) -> np.ndarray[Any, Any]:
        """The stored carrier itself; writes through it change this container."""
        return self._values

    @property
    def element(self) -> StoredElement:
        """The stored element descriptor, including any preserved marker."""
        return self._element

    @property
    def object_marker(self) -> str | None:
        """The preserved whole-object ``jtype`` text, if any."""
        return self._object_marker

    @property
    def ndim(self) -> int:
        """One for a plain vector, two for a matrix."""
        return int(self._values.ndim)

    @property
    def shape(self) -> tuple[int, ...]:
        """The stored shape."""
        return tuple(int(n) for n in self._values.shape)

    @property
    def active_marker(self) -> str | None:
        """The marker Julia would act on: the whole-object text when present."""
        return self._object_marker if self._object_marker is not None else self._element.marker

    def __len__(self) -> int:
        return int(self._values.shape[0])

    def __repr__(self) -> str:
        marker = "" if self.active_marker is None else f", marker={self.active_marker!r}"
        return (
            f"StoredArray({self._element.julia_name}, shape={self.shape}"
            f"{marker}) with {self._values.size} elements"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StoredArray):
            return NotImplemented
        return (
            self._element == other._element
            and self._object_marker == other._object_marker
            and self.shape == other.shape
            and self._values.tobytes(order="C") == other._values.tobytes(order="C")
        )

    __hash__ = None  # type: ignore[assignment]

    # ---- validation ------------------------------------------------------

    def _interpretation(self) -> tuple[str, Target]:
        return resolve_array_interpretation(self._values, self._element, self._object_marker)

    def validate(self) -> None:
        """Check the live carrier and any preserved marker against its contents.

        Raises if the carrier was changed in place to another dtype, layout or
        dimensionality, if it exceeds the payload capacity, or if an active
        marker no longer describes a possible conversion of the current values.
        """
        _check_array_carrier(self._values, self._element)
        self._interpretation()

    # ---- explicit conversions --------------------------------------------

    def tolist(self) -> list[Any]:
        """Convert the stored values to Python objects, nesting rows for a matrix.

        Dates and durations become core ``MIT``/``Duration`` objects with the
        element frequency, 128-bit values become Python ``int`` and ComplexF16
        values become Python ``complex``. A preserved marker does not change
        this; use :meth:`to_interpreted` for Julia's converted values.
        """
        _check_array_carrier(self._values, self._element)
        if self._values.ndim == 1:
            return self._row_to_list(self._values)
        return [self._row_to_list(np.ascontiguousarray(row)) for row in self._values]

    def _row_to_list(self, row: np.ndarray[Any, Any]) -> list[Any]:
        return element_tolist(row, self._element)

    def to_bool(self) -> np.ndarray[Any, Any]:
        """Convert a carrier with an active ``"Bool"`` marker into a Boolean array."""
        if self.active_marker != BOOL_MARKER:
            raise TypeError('to_bool applies to carriers with an active "Bool" marker only.')
        values = self._conversion_snapshot()
        resolve_array_interpretation(values, self._element, self._object_marker)
        flags = _interpret.bool_flags(values.reshape(-1), self._element.target)
        # The snapshot's own shape, never the live array's: a concurrent
        # reshape must not move the result away from the values just checked.
        return flags.reshape(values.shape)

    def to_interpreted(self) -> np.ndarray[Any, Any] | StoredArray:
        """Return the value Julia's loader would build, without changing what is stored.

        Ordinary numeric and Boolean targets give an owning NumPy array; date,
        duration and wide targets give an unmarked :class:`StoredArray`. A
        ``Diagonal``, ``Symmetric`` or ``Hermitian`` marker gives the dense
        matrix Julia reconstructs, which discards the entries that wrapper
        ignores. Conversions reproduce the reference's rounding and raise
        ``ValueError`` on values the target cannot hold.
        """
        values = self._conversion_snapshot()
        shape = values.shape
        kind, target = resolve_array_interpretation(values, self._element, self._object_marker)
        if kind == "structure":
            assert self._object_marker is not None
            dense = _structure_dense(values, self._object_marker)
            if self._element.kind == "numeric":
                return dense
            return StoredArray(dense, self._element.with_marker(None), copy=False)
        if kind in ("identity", "stored"):
            if self._element.kind == "numeric":
                return values
            return StoredArray(values, _element_from_target(target), copy=False)
        _interpret.check_output_capacity(int(values.size), target)
        flat = _interpret.convert_values(values.reshape(-1), self._element.target, target)
        converted = flat.reshape(shape)
        if target.kind == "numeric":
            return converted
        return StoredArray(converted, _element_from_target(target), copy=False)

    def _conversion_snapshot(self) -> np.ndarray[Any, Any]:
        """Own one carrier snapshot used for both validation and conversion.

        The shape and expected byte count are captured before the bytes are
        taken and compared against them afterwards, so a caller that resizes
        the array mid-call is refused rather than silently producing a result
        of a shape nothing validated.
        """
        _check_array_carrier(self._values, self._element)
        shape = self.shape
        expected = int(self._values.size) * self._element.itemsize
        payload = self._values.tobytes(order="C")
        if len(payload) != expected:
            raise ValueError("The array values changed size during the conversion snapshot.")
        return np.frombuffer(payload, dtype=self._element.dtype).reshape(shape).copy()


def _empty_interpretation(
    element: StoredElement, marker: str, target: Target
) -> tuple[str, Target]:
    """Apply the dated series' empty-payload rules to an empty array.

    Julia builds an empty typed array for any target, so a foreign marker is
    preserved on the kind default, where the stored width is unambiguous, and
    refused on any other width.
    """
    if element.kind == "numeric":
        if element.julia_name != JULIA_KIND_DEFAULTS[element.native_kind]:
            raise TypeError(
                "An empty numeric payload has no stored width; a foreign marker is "
                f"kept on the kind default {JULIA_KIND_DEFAULTS[element.native_kind]}, "
                f"not on {element.julia_name}."
            )
        return "element", target
    if element.is_wide:
        raise ValueError(
            f"An empty {element.julia_name} carrier with the foreign marker {marker!r} "
            "has no storable width; use an empty numeric array or an unmarked wide carrier."
        )
    raise TypeError(
        f"Julia cannot load an empty {element.julia_name} array; the foreign marker "
        f"{marker!r} on one is not supported."
    )


def resolve_array_interpretation(
    values: np.ndarray[Any, Any], element: StoredElement, object_marker: str | None
) -> tuple[str, Target]:
    """Resolve marker precedence for a plain array and check the declared conversion.

    Returns ``("identity"|"structure"|"element"|"stored", target)``. A present
    whole-object marker wins and leaves the element marker inactive, exactly as
    in Julia. Unlike a dated series, a parameterised container token converts
    every element there, so it is accepted here as an element conversion.
    """
    size = int(values.size)
    ndim = int(values.ndim)
    base = element.target
    if object_marker is not None:
        if element.kind in ("date", "duration") and not size:
            raise TypeError(
                "Julia cannot load an empty date or duration array; a whole-object "
                "marker on one is not supported."
            )
        kind, target = _interpret.array_object_interpretation(object_marker, base, ndim)
        if kind == "element":
            _interpret.check_route(base, target)
            _interpret.check_values(values.reshape(-1), base, target)
        elif kind == "structure" and ndim != 2:
            raise TypeError("A structural reconstruction marker applies to matrices only.")
        return kind, target
    marker = element.marker
    if marker is None:
        if element.kind == "numeric":
            raise TypeError(
                "Ordinary numeric values without a reconstruction marker belong in a plain "
                "NumPy array, not a StoredArray."
            )
        return "stored", base
    resolved = _interpret.resolve_token(marker)
    if resolved is None:
        raise TypeError(
            f"Unsupported reconstruction marker {marker!r}; marker text is compared with "
            "a finite table and never evaluated."
        )
    target = resolved
    if target == base:
        return "stored", base
    if element.kind == "numeric" and target.is_bool:
        raise TypeError(
            'A "Bool" marker on an ordinary numeric payload reads as a Boolean array; '
            "build one with a bool dtype instead of a StoredArray."
        )
    if not size:
        return _empty_interpretation(element, marker, target)
    _interpret.check_route(base, target)
    _interpret.check_values(values.reshape(-1), base, target)
    return "element", target


@dataclass(frozen=True, slots=True)
class StoredText:
    """A DataEcon text vector kept as exact stored bytes plus its marker.

    ``values`` holds each element's stored bytes without its NUL terminator,
    and ``marker`` is the preserved ``jeltype`` text (``"Symbol"`` for a Julia
    ``Symbol`` vector, for instance). Ordinary text that is valid UTF-8 and
    carries no foreign marker is returned as an ordinary ``list[str]`` instead;
    this container exists for the values that would otherwise lose their marker
    or their exact bytes.
    """

    values: tuple[bytes, ...]
    marker: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple) or any(
            not isinstance(item, bytes) for item in self.values
        ):
            raise TypeError("StoredText values must be a tuple of bytes objects.")
        if any(b"\0" in item for item in self.values):
            raise ValueError(
                "DataEcon packs text elements as NUL-separated bytes, so an element "
                "cannot contain NUL; the boundary itself would be lost."
            )
        _interpret.check_marker_text(self.marker, "element")

    @classmethod
    def from_list(cls, strings: Any, marker: str | None = None) -> StoredText:
        """Build a container from Python strings, encoding each as UTF-8."""
        items = list(strings)
        if any(type(item) is not str for item in items):
            raise TypeError("StoredText.from_list takes plain Python strings.")
        return cls(tuple(item.encode("utf-8") for item in items), marker)

    def __len__(self) -> int:
        return len(self.values)

    @property
    def is_text(self) -> bool:
        """Whether every element decodes as UTF-8."""
        try:
            self.tolist()
        except ValueError:
            return False
        return True

    def tolist(self) -> list[str]:
        """Decode every element as strict UTF-8, raising if any element is not text."""
        out: list[str] = []
        for index, item in enumerate(self.values):
            try:
                out.append(item.decode("utf-8"))
            except UnicodeDecodeError as error:
                raise ValueError(
                    f"Stored text element {index} is not valid UTF-8; its exact bytes are "
                    "preserved and can be read from .values."
                ) from error
        return out
