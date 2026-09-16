# SPDX-License-Identifier: MIT
"""Stored representations for plain DataEcon arrays and text vectors.

Ordinary numeric and Boolean arrays travel as plain NumPy arrays, ordinary
text vectors as a list of Python strings and ordinary text matrices and tensors
as NumPy ``str_`` arrays. Two small adapter containers carry what those
familiar types cannot hold losslessly:

* :class:`StoredArray` pairs a contiguous carrier with a
  :class:`~tsecon.dataecon._represented.StoredElement`, so ``MIT``/``Duration``
  codes, 128-bit integers, ComplexF16 pairs and preserved reconstruction
  markers keep their stored kind, width and bytes. It holds values of one to
  five dimensions (DataEcon's axis limit); :meth:`StoredArray.to_interpreted`
  is the explicit conversion to the value Julia's loader would build, and
  :meth:`StoredArray.diagonal`, :meth:`StoredArray.symmetric` and
  :meth:`StoredArray.hermitian` build the dense, marker-bearing form Julia's
  own writer stores for its LinearAlgebra wrappers.
* :class:`StoredText` keeps each element's exact stored bytes together with the
  preserved ``jeltype`` text and the stored shape, which is how a ``Symbol``
  array and text that is not valid UTF-8 survive a read-modify-write at any
  rank from one to five.

Neither container evaluates marker text, and neither is part of the core
package: they exist only at the DataEcon adapter boundary.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _interpret, _printed
from ._interpret import MAX_AXES, Target
from ._metadata import JULIA_KIND_DEFAULTS, JULIA_NUMERIC_TYPES, MAX_BYTES, MAX_INT64
from ._represented import (
    BOOL_MARKER,
    StoredElement,
    _element_from_target,
    _opaque_error,
    element_interpretation,
    element_tolist,
    printed_elements,
)

__all__ = ["MAX_UNICODE_BYTES", "StoredArray", "StoredText"]

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
    if not 1 <= values.ndim <= _interpret.MAX_AXES:
        raise ValueError(
            f"A StoredArray carrier must have one to {_interpret.MAX_AXES} dimensions."
        )
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


def structure_dense(
    values: np.ndarray[Any, Any], token: str, uplo: str = "U"
) -> np.ndarray[Any, Any]:
    """Materialise a LinearAlgebra wrapper into the dense square matrix Julia stores.

    ``Diagonal`` keeps only the diagonal and zeroes the rest;
    ``Symmetric``/``Hermitian`` copy the authoritative triangle (``uplo`` is
    ``"U"`` or ``"L"``, Julia's constructor argument) onto the other one.
    ``Hermitian`` also conjugates the mirrored entries and clears the
    imaginary part of the diagonal, keeping the real part's bits. Julia's
    writer runs exactly this (``Matrix{T}(value)``) before storing, and its
    loader rebuilds the wrapper from the stored matrix with the upper
    triangle, so the stored bytes are the same whichever triangle was
    authoritative.

    The mirroring copies elements rather than adding a zeroed triangle: adding
    a structural ``+0.0`` to a stored ``-0.0`` would silently clear a sign the
    reference preserves, and a NaN payload is carried bit for bit. Copying also
    keeps the two-word 128-bit and ComplexF16 carriers in range, so every
    supported element works here.
    """
    if values.ndim != 2:
        raise ValueError(f"A {token} matrix needs a two-dimensional carrier.")
    rows, columns = values.shape
    if rows != columns:
        raise ValueError(f"The {token} reconstruction marker applies to square matrices only.")
    if uplo not in ("U", "L"):
        raise ValueError('uplo must be "U" (upper triangle) or "L" (lower triangle).')
    if token == "Diagonal":
        # An explicit C-ordered allocation: zeros_like would copy a
        # Fortran-ordered input's layout into the carrier.
        out = np.zeros(values.shape, dtype=values.dtype)
        index = np.arange(rows)
        out[index, index] = values[index, index]
        return out
    out = values.copy(order="C")
    upper_rows, upper_columns = np.triu_indices(rows, 1)
    if uplo == "U":
        source_rows, source_columns = upper_rows, upper_columns
    else:
        source_rows, source_columns = upper_columns, upper_rows
    mirrored = values[source_rows, source_columns]
    if token == "Hermitian":
        mirrored = _conjugated(mirrored)
    out[source_columns, source_rows] = mirrored
    if token == "Hermitian":
        _clear_diagonal_imaginary(out, rows)
    return out


def _structure_dense(values: np.ndarray[Any, Any], token: str) -> np.ndarray[Any, Any]:
    """Julia's loader-side reconstruction: the upper triangle of the stored matrix."""
    return structure_dense(values, token, "U")


def _structure_input(values: Any, token: str) -> tuple[np.ndarray[Any, Any], StoredElement]:
    """Return a structure constructor's input, unconverted, with its element descriptor.

    A NumPy array of an ordinary numeric dtype gives a numeric descriptor;
    a Boolean array gets the Int8 descriptor with the inactive ``"Bool"``
    element marker, which is exactly what Julia's writer stores for a Boolean
    wrapper (its loader then rebuilds an Int8 wrapper, because ``jtype``
    wins); the array itself is returned as given and converted only once the
    shape and capacity checks have passed. A :class:`StoredArray` supplies
    the represented families; it must not already carry a whole-object
    marker.
    """
    if isinstance(values, StoredArray):
        if values.object_marker is not None:
            raise ValueError(
                f"Build the {token} matrix from an unmarked carrier; this StoredArray already "
                f"carries the whole-object marker {values.object_marker!r}."
            )
        values.validate()
        return values.values, values.element
    if not isinstance(values, np.ndarray):
        raise TypeError(f"A {token} matrix is built from a NumPy array or a StoredArray.")
    if values.dtype.kind == "b":
        return values, StoredElement.numeric(np.dtype("<i1"), BOOL_MARKER)
    if (
        values.dtype.char in ("g", "G")
        or not values.dtype.isnative
        or values.dtype not in {dtype for _, dtype in JULIA_NUMERIC_TYPES.values()}
    ):
        raise TypeError(
            f"A {token} matrix takes an ordinary native-endian numeric or Boolean NumPy "
            "array, or a StoredArray for represented elements."
        )
    return values, StoredElement.numeric(values.dtype)


def _structure_side(values: np.ndarray[Any, Any], token: str, uplo: str) -> int:
    """Validate ``uplo`` and the input's rank and shape; return the square's side.

    Only ``Diagonal`` accepts a vector (the diagonal itself); every other
    input must already be square. Nothing is allocated here, so an invalid
    request is refused before any conversion or expansion.
    """
    if uplo not in ("U", "L"):
        raise ValueError('uplo must be "U" (upper triangle) or "L" (lower triangle).')
    if values.ndim == 1 and token == "Diagonal":
        return int(values.shape[0])
    if values.ndim != 2:
        raise ValueError(f"A {token} matrix needs a two-dimensional carrier.")
    rows, columns = (int(extent) for extent in values.shape)
    if rows != columns:
        raise ValueError(f"The {token} reconstruction marker applies to square matrices only.")
    return rows


def _check_structure_capacity(side: int, element: StoredElement, token: str) -> None:
    """Refuse a square whose payload would exceed the limit, in Python integers."""
    _interpret.check_payload_bytes(
        side * side * element.dtype.itemsize,
        f"A {side}-by-{side} {token} matrix of {element.julia_name}",
    )


def _structure_matrix(values: Any, token: str, uplo: str) -> StoredArray:
    # Order matters: the original input's type, rank, shape and prospective
    # output size are checked before anything is converted or expanded, so a
    # long diagonal vector or a broadcast Boolean view is refused without the
    # square (or the Int8 copy) ever being requested.
    carrier, element = _structure_input(values, token)
    side = _structure_side(carrier, token, uplo)
    _check_structure_capacity(side, element, token)
    if carrier.dtype.kind == "b":
        carrier = carrier.astype(np.int8, order="C")
    if carrier.ndim == 1:
        square = np.zeros((side, side), dtype=carrier.dtype)
        index = np.arange(side)
        square[index, index] = carrier
        carrier = square
    _check_array_values(carrier, element)
    if carrier.size == 0 and (
        element.kind != "numeric"
        or element.marker is not None
        or element.julia_name != JULIA_KIND_DEFAULTS[element.native_kind]
    ):
        # Under a whole-object marker the element token is inactive, so an
        # empty payload can only be read back at its kind's default width
        # (Julia's loader loses the width the same way).
        raise ValueError(
            f"An empty {token} matrix can only hold the kind default element "
            "(Float64, Int64, UInt64 or ComplexF64); the stored width of "
            f"{element.julia_name} would be lost under the wrapper marker."
        )
    dense = structure_dense(carrier, token, uplo)
    return StoredArray(dense, element, copy=False, object_marker=token)


class StoredArray:
    """A plain DataEcon array kept in its stored element representation.

    ``values`` is a C-contiguous NumPy carrier of one to five dimensions
    holding the exact stored bytes for ``element``; a carrier of two or more
    dimensions is in the natural NumPy row-major order and the adapter writes
    DataEcon's column-major payload from it. ``object_marker`` is a preserved
    ``jtype`` text. The container is copied by default so the file's bytes
    cannot be mutated through an alias.
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

    # ---- structure-marked matrices ---------------------------------------

    @classmethod
    def diagonal(cls, values: Any) -> StoredArray:
        """Build the ``Diagonal``-marked matrix Julia's writer stores for ``Diagonal(v)``.

        ``values`` is the diagonal itself (one-dimensional) or a square matrix
        whose diagonal is taken, as Julia's ``Diagonal(A)`` does: an ordinary
        numeric or Boolean NumPy array, or a :class:`StoredArray` of a
        represented element. The result holds the full dense matrix with
        zeros off the diagonal (the diagonal's bits, signed zeros and NaN
        payloads included, are copied) and the ``"Diagonal"`` marker.
        """
        return _structure_matrix(values, "Diagonal", "U")

    @classmethod
    def symmetric(cls, values: Any, uplo: str = "U") -> StoredArray:
        """Build the ``Symmetric``-marked matrix Julia's writer stores for ``Symmetric(A, uplo)``.

        Only the authoritative triangle of the square input matters: with
        ``uplo="U"`` (Julia's default) the strict upper triangle is copied onto
        the lower one, with ``"L"`` the reverse; the diagonal is kept bit for
        bit. The materialised dense matrix is what Julia stores, and the
        loader rebuilds ``Symmetric(M)`` from it whichever triangle was given.
        """
        return _structure_matrix(values, "Symmetric", uplo)

    @classmethod
    def hermitian(cls, values: Any, uplo: str = "U") -> StoredArray:
        """Build the ``Hermitian``-marked matrix Julia's writer stores for ``Hermitian(A, uplo)``.

        The authoritative triangle (``uplo`` ``"U"`` by default, or ``"L"``) is
        conjugated onto the other one and the diagonal's imaginary part is
        cleared to ``+0.0`` while its real part keeps its bits, exactly as
        ``Matrix(Hermitian(A, uplo))`` does; real element families are
        mirrored without conjugation.
        """
        return _structure_matrix(values, "Hermitian", uplo)

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
        """One for a plain vector, two for a matrix, three to five for a tensor."""
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
        """Convert the stored values to Python objects, nesting one list per dimension.

        Dates and durations become core ``MIT``/``Duration`` objects with the
        element frequency, 128-bit values become Python ``int`` and ComplexF16
        values become Python ``complex``. A preserved marker does not change
        this; use :meth:`to_interpreted` for Julia's converted values.
        """
        _check_array_carrier(self._values, self._element)
        return self._nested_list(self._values)

    def _nested_list(self, values: np.ndarray[Any, Any]) -> list[Any]:
        if values.ndim == 1:
            return element_tolist(np.ascontiguousarray(values), self._element)
        return [self._nested_list(row) for row in values]

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

    def to_interpreted(  # noqa: PLR0911 - one return per result kind
        self,
    ) -> np.ndarray[Any, Any] | StoredArray | StoredText | str:
        """Return the value Julia's loader would build, without changing what is stored.

        Ordinary numeric and Boolean targets give an owning NumPy array
        (object dtype for ``BigInt``/``BigFloat`` and for an empty abstract
        type, ``datetime64[D]``/``[ms]`` for ``Date``/``DateTime`` on a numeric
        payload, ``str_`` of one character for ``Char``); date, duration, wide
        and exact rational/complex targets give an unmarked
        :class:`StoredArray`; a ``Symbol`` element marker on a numeric payload
        gives the :class:`StoredText` of Julia's printed elements, and a
        ``Symbol`` whole-object marker the ``str`` Julia prints for the whole
        array (``[1, 2]``, ``Int8[1 3; 2 4]``, ``Matrix{Float64}(undef, 0,
        2)``; the bytes of a UInt8 vector name the symbol directly). A
        ``Diagonal``, ``Symmetric`` or ``Hermitian`` marker gives the dense
        matrix Julia reconstructs, which discards the entries that wrapper
        ignores. Conversions reproduce the reference's rounding and raise
        ``ValueError`` on values the target cannot hold; an opaque marker
        raises ``TypeError``.
        """
        values = self._conversion_snapshot()
        shape = values.shape
        kind, target = resolve_array_interpretation(values, self._element, self._object_marker)
        if kind == "printed":
            element = self._element
            return _printed.numeric_symbol(
                values, element.kind, element.dtype, element.julia_name, element.frequency
            )
        if kind == "bytes":
            return _printed.byte_symbol(values.tobytes())
        if kind == "structure":
            assert self._object_marker is not None
            dense = _structure_dense(values, self._object_marker)
            if self._element.kind == "numeric":
                return dense
            return StoredArray(dense, self._element.with_marker(None), copy=False)
        if kind == "opaque":
            raise _opaque_error(target)
        if kind == "empty":
            return np.empty(shape, dtype=target.dtype)
        if kind == "symbol":
            flat = values.reshape(-1, order="F")
            return StoredText(printed_elements(flat, self._element), "Symbol", shape)
        if kind in ("identity", "stored"):
            if self._element.kind == "numeric":
                return values
            return StoredArray(values, _element_from_target(target), copy=False)
        _interpret.check_output_capacity(int(values.size), target)
        flat = _interpret.convert_values(values.reshape(-1), self._element.target, target)
        converted = flat.reshape(shape)
        if target.kind == "numeric" or kind in ("object", "datetime", "char"):
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


def resolve_array_interpretation(
    values: np.ndarray[Any, Any], element: StoredElement, object_marker: str | None
) -> tuple[str, Target]:
    """Resolve marker precedence for a plain array and check the declared conversion.

    Returns ``("identity"|"structure"|"element"|"stored"|"printed"|"bytes",
    target)``. A present whole-object marker wins and leaves the element
    marker inactive, exactly as in Julia (``Symbol`` prints the stored array;
    the ``printed`` and ``bytes`` kinds). Unlike a dated series, a
    parameterised container token converts every element there, so it is
    accepted here as an element conversion.
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
            # Unreachable through the vocabulary (structure tokens resolve for
            # rank two only); kept as the contract's own guard.
            raise TypeError("A structural reconstruction marker applies to matrices only.")
        return kind, target
    if element.marker is None and element.kind == "numeric":
        raise TypeError(
            "Ordinary numeric values without a reconstruction marker belong in a plain "
            "NumPy array, not a StoredArray."
        )
    # The element table, routes, value rules, empty-only tokens, opaque markers
    # and the plain-array Bool width rule are shared with the dated containers;
    # the flat column-major values are what Julia's map visits.
    return element_interpretation(values.reshape(-1, order="F"), element, "array")


# ---- text arrays: bounded packing and materialization ------------------------

# A fixed-width NumPy Unicode array costs ``count * longest element * 4``
# bytes, not the packed payload size: one long element pads every other one.
# Automatic materialization on read is therefore capped; above the cap the
# lossless shaped StoredText is returned instead, and explicit ``to_numpy``
# refuses. Four times the native payload cap is the size of a uniform
# 128 MiB ASCII payload, so no padding-free payload is ever refused.
MAX_UNICODE_BYTES = 4 * MAX_BYTES


def check_text_capacity(total: int) -> None:
    """Refuse a packed text payload that would exceed the native limit."""
    if total > MAX_BYTES:
        raise ValueError(
            f"The packed text payload would need {total} bytes, above the "
            f"{MAX_BYTES}-byte DataEcon limit."
        )


def text_array_shape(array: Any) -> tuple[int, ...]:
    """Validate a ``str_`` array's kind, rank, dimensions and minimum packed size.

    Runs before any ravel, copy or ``tolist`` of the array: a broadcast or
    strided view can be tiny in memory while its logical element count is
    enormous, and every element costs at least its NUL terminator.
    """
    if not isinstance(array, np.ndarray) or array.dtype.kind != "U":
        raise TypeError(
            "DataEcon text arrays take NumPy str_ (unicode) arrays; bytes and object "
            "arrays are not converted implicitly."
        )
    if not 1 <= array.ndim <= MAX_AXES:
        raise ValueError(
            f"write_array supports NumPy arrays of one to {MAX_AXES} dimensions; DataEcon "
            f"stores at most {MAX_AXES} axes."
        )
    shape = tuple(int(n) for n in array.shape)
    check_text_capacity(text_count(shape))
    return shape


def text_count(shape: tuple[int, ...]) -> int:
    """Return the element count of a shape, accumulated in Python integers and bounded."""
    count = 1
    for n in shape:
        if n < 0:
            raise ValueError("Invalid negative DataEcon text array dimension.")
        count *= n
        if count > MAX_INT64:
            raise ValueError(
                "The DataEcon text array element count exceeds the signed 64-bit range."
            )
    return count


def nested_text(
    items: Any, depth: int = 1, counted: list[int] | None = None
) -> tuple[tuple[int, ...], list[str]]:
    """Flatten a rectangular nested list/tuple of ``str`` into row-major order plus its shape.

    Every level must be a ``list`` or ``tuple`` and every leaf exactly ``str``
    (no subclasses, no bytes); other iterables such as dicts, sets or
    generators are refused rather than coerced through iteration. Every
    sibling must have the same shape, so a ragged input is refused rather than
    padded. An empty sequence ends the recursion with a zero-length dimension.
    Leaves are counted as they are found and the minimum packed size (one
    terminator each) is checked against the native limit as the walk proceeds,
    so an oversized input is refused before it is flattened in full.
    """
    if counted is None:
        counted = [0]
    if depth > MAX_AXES:
        raise ValueError(f"DataEcon text arrays have at most {MAX_AXES} dimensions.")
    if not isinstance(items, (list, tuple)):
        raise TypeError(
            "A DataEcon text array takes plain Python strings in nested lists or tuples."
        )
    entries = list(items)
    if any(type(item) is str for item in entries):
        if any(type(item) is not str for item in entries):
            raise TypeError("A DataEcon text array takes plain Python strings.")
        counted[0] += len(entries)
        check_text_capacity(counted[0])
        return (len(entries),), entries
    if not entries:
        return (0,), []
    shape: tuple[int, ...] | None = None
    flat: list[str] = []
    for item in entries:
        item_shape, item_flat = nested_text(item, depth + 1, counted)
        if shape is None:
            shape = item_shape
        elif item_shape != shape:
            raise ValueError("A DataEcon text array must be rectangular; the input is ragged.")
        flat.extend(item_flat)
    assert shape is not None
    return (len(entries), *shape), flat


def pack_strings(items: Iterable[str], total: int = 0) -> tuple[bytes, ...]:
    """Encode ``str`` elements as UTF-8, sizing the packed payload before encoding.

    The packed size is accumulated in Python integers, in UTF-8 bytes (not
    characters, which is the reference writer's mistake), and refused as soon
    as it exceeds the limit, so an oversized input never has every element
    encoded. ``total`` lets a chunked caller carry the running size across
    chunks. Element order is irrelevant to the total.
    """
    encoded = []
    for item in items:
        data = item.encode("utf-8")
        total += len(data) + 1
        check_text_capacity(total)
        if b"\0" in data:
            raise ValueError(
                "DataEcon packs text elements as NUL-separated bytes, so an element cannot "
                "contain NUL; the element boundary itself would be lost. Julia's own writer "
                "refuses these values too."
            )
        encoded.append(data)
    return tuple(encoded)


# Column-major traversal of a str_ array in bounded chunks. A fixed-width
# array's logical values can dwarf its memory (a broadcast of one 80 KB
# string is 1.6 GB when flattened), so the array is never raveled or listed
# as a whole: each chunk gathers at most this many bytes of fixed-width
# storage, converts them to Python strings and feeds the cumulative UTF-8
# check, which refuses an oversized payload long before the whole input has
# been touched.
TEXT_CHUNK_BYTES = 16 * 1024 * 1024


def text_array_bytes(array: np.ndarray[Any, Any], shape: tuple[int, ...]) -> tuple[bytes, ...]:
    """Encode a validated ``str_`` array's elements in column-major order, chunk by chunk."""
    count = text_count(shape)
    if count == 0:
        return ()
    step = max(1, TEXT_CHUNK_BYTES // max(1, array.dtype.itemsize))
    if array.ndim == 1:
        # A one-dimensional slice is always a view, whatever the stride.
        def chunk(start: int, stop: int) -> Any:
            return array[start:stop]
    elif array.flags.f_contiguous:
        # A Fortran-contiguous array flattens to a view in column-major order.
        flat = array.reshape(-1, order="F")

        def chunk(start: int, stop: int) -> Any:
            return flat[start:stop]
    else:

        def chunk(start: int, stop: int) -> Any:
            # Gather only this chunk's elements; the indices are the
            # column-major positions, so a broadcast or strided input never
            # materializes more than one chunk at a time.
            return array[np.unravel_index(np.arange(start, stop), shape, order="F")]

    encoded: list[bytes] = []
    total = 0
    for start in range(0, count, step):
        piece = pack_strings(chunk(start, min(start + step, count)).tolist(), total)
        total += sum(len(item) + 1 for item in piece)
        encoded.extend(piece)
    return tuple(encoded)


def column_major(flat: Any, shape: tuple[int, ...]) -> tuple[Any, ...]:
    """Reorder row-major elements into DataEcon's column-major storage order."""
    if len(shape) == 1:
        return tuple(flat)
    order = np.arange(len(flat)).reshape(shape).ravel(order="F")
    return tuple(flat[index] for index in order)


def nest_rows(flat: list[Any], shape: tuple[int, ...]) -> list[Any]:
    """Nest column-major elements as row-major lists without any padded intermediate."""
    strides = []
    stride = 1
    for n in shape:
        strides.append(stride)
        stride *= n

    def build(dim: int, base: int) -> list[Any]:
        if dim == len(shape) - 1:
            return [flat[base + i * strides[dim]] for i in range(shape[dim])]
        return [build(dim + 1, base + i * strides[dim]) for i in range(shape[dim])]

    return build(0, 0)


def unicode_nbytes(decoded: list[str]) -> int:
    """Return the bytes a fixed-width NumPy Unicode array of these elements would allocate."""
    width = max((len(item) for item in decoded), default=1)
    return len(decoded) * max(width, 1) * 4


def unicode_array(decoded: list[str], shape: tuple[int, ...]) -> np.ndarray[Any, Any]:
    """Allocate the owning, C-contiguous ``str_`` array of column-major ``decoded``.

    This is the only place a fixed-width Unicode array is allocated for text,
    after the caller has checked :func:`unicode_nbytes` against the cap.
    """
    width = max((len(item) for item in decoded), default=1)
    out = np.empty(shape, dtype=f"<U{max(width, 1)}")
    if decoded:
        order = np.arange(len(decoded)).reshape(shape, order="F").ravel(order="C")
        out.reshape(-1)[:] = [decoded[index] for index in order]
    return out


@dataclass(frozen=True, slots=True)
class StoredText:
    """A DataEcon text array kept as exact stored bytes plus its marker and shape.

    ``values`` holds each element's stored bytes without its NUL terminator, in
    DataEcon's own column-major (Fortran) storage order, and ``marker`` is the
    preserved ``jeltype`` text (``"Symbol"`` for a Julia ``Symbol`` array, for
    instance). ``shape`` is the stored shape, one to five dimensions; it
    defaults to the flat vector shape, so a one-dimensional container is built
    exactly as before. Ordinary text that is valid UTF-8 and carries no foreign
    marker is returned as an ordinary ``list[str]`` (vectors) or NumPy ``str_``
    array (matrices and tensors) instead, unless that array would exceed
    :data:`MAX_UNICODE_BYTES`; this container exists for the values that would
    otherwise lose their marker, their exact bytes or their shape, or cost a
    padded allocation far beyond the packed payload.

    ``object_marker`` is a preserved whole-object ``jtype`` text. When present
    it wins over the element marker in Julia (a ``Vector{String}`` whole-object
    marker over a ``Symbol`` element marker loads plain strings); both are kept
    verbatim and written back. :meth:`to_interpreted` applies Julia's
    reconstruction; :meth:`tolist` and :meth:`to_numpy` stay the strict decodes
    of the stored text regardless of any marker.

    The container is immutable. :meth:`tolist` and :meth:`to_numpy` are the
    explicit strict decodes in logical (row-major) indexing; :meth:`from_list`
    and :meth:`from_numpy` build a container from Python strings.
    """

    values: tuple[bytes, ...]
    marker: str | None = None
    shape: tuple[int, ...] | None = None
    object_marker: str | None = None

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
        _interpret.check_marker_text(self.object_marker, "whole-object")
        shape = (len(self.values),) if self.shape is None else self.shape
        if (
            not isinstance(shape, tuple)
            or any(type(n) is not int or n < 0 for n in shape)
            or not 1 <= len(shape) <= MAX_AXES
        ):
            raise ValueError(
                f"StoredText shape must be a tuple of one to {MAX_AXES} non-negative integers."
            )
        if text_count(shape) != len(self.values):
            raise ValueError(
                f"StoredText shape {shape} describes {text_count(shape)} elements but "
                f"{len(self.values)} values were given."
            )
        object.__setattr__(self, "shape", shape)

    @classmethod
    def from_list(cls, strings: Any, marker: str | None = None) -> StoredText:
        """Build a container from Python strings, encoding each as UTF-8.

        A flat list or tuple gives a vector; a rectangular nested list or tuple
        gives a matrix or tensor of the nested shape, indexed as
        ``strings[i][j]``. Other iterables are refused rather than coerced.
        """
        shape, flat = nested_text(strings)
        return cls(column_major(pack_strings(flat), shape), marker, shape)

    @classmethod
    def from_numpy(cls, array: Any, marker: str | None = None) -> StoredText:
        """Build a container from a NumPy ``str_`` (``<U``) array of any layout."""
        shape = text_array_shape(array)
        encoded = text_array_bytes(array, shape)
        if len(encoded) != text_count(shape):
            raise ValueError("The text array changed size during the snapshot.")
        return cls(encoded, marker, shape)

    def __len__(self) -> int:
        """Return the number of stored elements (the product of the shape)."""
        return len(self.values)

    @property
    def ndim(self) -> int:
        """The number of stored dimensions."""
        assert self.shape is not None
        return len(self.shape)

    @property
    def size(self) -> int:
        """The number of stored elements."""
        return len(self.values)

    @property
    def is_text(self) -> bool:
        """Whether every element decodes as UTF-8."""
        try:
            self._decoded()
        except ValueError:
            return False
        return True

    def _decoded(self) -> list[str]:
        out: list[str] = []
        for index, item in enumerate(self.values):
            try:
                out.append(item.decode("utf-8"))
            except UnicodeDecodeError as error:
                where = f"element {index}"
                if self.ndim > 1:
                    assert self.shape is not None
                    position = np.unravel_index(index, self.shape, order="F")
                    where += f" (index {tuple(int(i) for i in position)})"
                raise ValueError(
                    f"Stored text {where} is not valid UTF-8; its exact bytes are "
                    "preserved and can be read from .values."
                ) from error
        return out

    @property
    def unicode_nbytes(self) -> int:
        """The bytes :meth:`to_numpy` would allocate: count times the longest element times four."""
        return unicode_nbytes(self._decoded())

    def tolist(self) -> list[Any]:
        """Decode every element as strict UTF-8, raising if any element is not text.

        A vector returns ``list[str]``; a matrix or tensor returns nested lists
        in row-major nesting, so ``tolist()[i][j]`` is the element Julia stores
        at ``[i+1, j+1]``. An empty dimension nests as an empty list. No
        fixed-width array is built: the lists hold each string at its own size.
        """
        decoded = self._decoded()
        if self.ndim == 1:
            return decoded
        assert self.shape is not None
        return nest_rows(decoded, self.shape)

    @property
    def active_marker(self) -> str | None:
        """The marker Julia acts on: the whole-object text when present, else the element text."""
        return self.object_marker if self.object_marker is not None else self.marker

    def to_interpreted(self) -> list[str] | np.ndarray[Any, Any] | StoredText | str:
        """Return the value Julia's loader builds from the stored text and its markers.

        A whole-object ``Symbol`` gives the ``str`` Julia prints for the whole
        String array (``["a", "b"]``, ``["a" "c"; "b" "d"]``,
        ``Matrix{String}(undef, 0, 2)``, with Julia's own escaping of each
        element; a character this Python's Unicode tables leave unassigned
        raises ``ValueError``, since Julia's tables decide whether it prints
        raw). The whole-object identities (``Vector{String}``, ``Vector``,
        ``AbstractVector``, ``AbstractArray``, ``Array{String,1}``,
        ``Array{String, 1}``, ``Array{String}``, ``Array``, ``Any``; the
        ``Matrix``/``Array{String,N}`` forms at higher ranks) give the plain
        text (``list[str]`` for a vector, a ``str_`` array otherwise) whatever
        the element marker says, as in Julia; ``Vector{Any}``,
        ``Vector{AbstractString}``, ``Vector{SubString{String}}``,
        ``Vector{Union{String,Symbol}}``, ``Matrix{Any}``,
        ``Matrix{AbstractString}`` and ``Array{Any,N}`` give an object array of
        ``str``; ``Vector{Symbol}``/``Matrix{Symbol}``/``Array{Symbol,N}``
        load only an empty text (as an empty ``StoredText`` with the
        ``Symbol`` marker). Without a whole-object marker the element marker
        rules apply: ``String``/``AbstractString``/``Any`` are the plain text,
        ``Symbol`` and ``SubString{String}`` the text values (Julia's
        ``Symbol``/``SubString`` elements, kept here as ``str``), ``Char``
        and other empty-only tokens load an empty text only, and an unknown
        marker raises ``TypeError``. Text that is not valid UTF-8 raises
        ``ValueError`` wherever strings are produced.
        """
        assert self.shape is not None
        ndim = len(self.shape)
        token = self.object_marker
        if token is not None:
            kind = text_object_interpretation(token, ndim, self.size)
        else:
            kind = text_element_interpretation(self.marker, self.size)
        if kind == "identity":
            return self._plain()
        if kind == "object":
            plain = np.array(self._decoded(), dtype=object)
            return unicode_array_like(plain, self.shape)
        if kind == "symbol":
            return StoredText(self.values, "Symbol", self.shape)
        if kind == "printed":
            return _printed.text_symbol(self.values, self.shape)
        assert kind == "empty"
        # Vector{Char}: an object array of one-character str (str_ drops U+0000).
        return np.empty(
            self.shape, dtype=object if self.active_marker in ("Char", "Vector{Char}") else "<U1"
        )

    def _plain(self) -> list[str] | np.ndarray[Any, Any]:
        assert self.shape is not None
        decoded = self._decoded()
        if self.ndim == 1:
            return decoded
        if unicode_nbytes(decoded) > MAX_UNICODE_BYTES:
            raise ValueError(
                "A fixed-width str_ array of this text would exceed the allocation limit; "
                "use tolist() or keep the StoredText."
            )
        return unicode_array(decoded, self.shape)

    def to_numpy(self) -> np.ndarray[Any, Any]:
        """Decode into an owning, C-contiguous NumPy ``str_`` array of the stored shape.

        Refuses before allocating when the fixed-width array would exceed
        :data:`MAX_UNICODE_BYTES`; :meth:`tolist` and the container itself
        remain available for such values.
        """
        assert self.shape is not None
        decoded = self._decoded()
        needed = unicode_nbytes(decoded)
        if needed > MAX_UNICODE_BYTES:
            raise ValueError(
                f"A fixed-width str_ array of this text would allocate {needed} bytes, above "
                f"the {MAX_UNICODE_BYTES}-byte limit; use tolist() or keep the StoredText."
            )
        return unicode_array(decoded, self.shape)


# ---- text whole-object and element markers ----------------------------------

# Every spelling below was loaded individually by the pinned Julia over
# nonempty and empty String vectors, matrices and tensors, with and without a
# Symbol element marker (the whole-object token wins). Julia's rewrite of the
# identities drops the markers; this adapter preserves them.
_TEXT_IDENTITY_TOKENS: dict[int, tuple[str, ...]] = {
    1: (
        "Vector{String}",
        "Vector",
        "AbstractVector",
        "AbstractArray",
        "Array{String,1}",
        "Array{String, 1}",
        "Array{String}",
        "Array",
        "Any",
    ),
    2: (
        "Matrix{String}",
        "Array{String,2}",
        "Array{String, 2}",
        "Matrix",
        "Array",
        "AbstractArray",
        "AbstractMatrix",
        "Any",
    ),
    **{
        n: (f"Array{{String,{n}}}", f"Array{{String, {n}}}", "Array", "AbstractArray", "Any")
        for n in range(3, MAX_AXES + 1)
    },
}
_TEXT_OBJECT_TOKENS: dict[int, tuple[str, ...]] = {
    1: (
        "Vector{Any}",
        "Vector{AbstractString}",
        "Vector{SubString{String}}",
        "Vector{Union{String,Symbol}}",
    ),
    2: ("Matrix{Any}", "Matrix{AbstractString}"),
    **{n: (f"Array{{Any,{n}}}",) for n in range(3, MAX_AXES + 1)},
}
_TEXT_SYMBOL_TOKENS: dict[int, tuple[str, ...]] = {
    1: ("Vector{Symbol}", "Vector{ Symbol }"),
    2: ("Matrix{Symbol}",),
    **{n: (f"Array{{Symbol,{n}}}",) for n in range(3, MAX_AXES + 1)},
}
_TEXT_EMPTY_TOKENS: dict[int, tuple[str, ...]] = {1: ("Vector{Char}",)}
# Element tokens on text: identities, the two preserved text families, and the
# empty-only tokens (``Char`` loads an empty ``Vector{Char}`` only).
_TEXT_IDENTITY_ELEMENTS = ("String", "AbstractString", "Any")
_TEXT_VALUE_ELEMENTS = ("Symbol", "SubString{String}")
_TEXT_EMPTY_ELEMENTS = ("Char", "Int64", "Vector{String}")


def text_object_interpretation(token: str, ndim: int, size: int) -> str:
    """Classify a whole-object token on text.

    ``identity``, ``object``, ``symbol`` (a typed empty), ``empty`` or
    ``printed`` (``Symbol``: the scalar Symbol of the printed array).
    ``TypeError`` names a spelling Julia fails on (``String``, a wrong rank)
    or an unknown one, which stays preserved and unevaluated.
    """
    if token in _TEXT_IDENTITY_TOKENS.get(ndim, ()):
        return "identity"
    if token == "Symbol":
        return "printed"
    if token in _TEXT_OBJECT_TOKENS.get(ndim, ()):
        return "object"
    if token in _TEXT_SYMBOL_TOKENS.get(ndim, ()) or token in _TEXT_EMPTY_TOKENS.get(ndim, ()):
        if size:
            raise TypeError(
                f"Julia has no {token} conversion for nonempty text (MethodError); only an "
                "empty text loads as an empty array of that type."
            )
        return "symbol" if token in _TEXT_SYMBOL_TOKENS.get(ndim, ()) else "empty"
    raise TypeError(
        f"Unsupported whole-object reconstruction marker on a {ndim}-dimensional text array; "
        "the marker text is preserved and never evaluated."
    )


def text_element_interpretation(marker: str | None, size: int) -> str:
    """Classify an element token on text: ``identity``, ``symbol`` or ``empty``."""
    if marker is None or marker in _TEXT_IDENTITY_ELEMENTS:
        return "identity"
    if marker in _TEXT_VALUE_ELEMENTS:
        return "identity" if marker != "Symbol" else "symbol"
    if marker in _TEXT_EMPTY_ELEMENTS:
        if size:
            raise TypeError(
                f"Julia loads the element marker {marker!r} on text from an empty payload "
                "only (MethodError otherwise)."
            )
        return "empty"
    raise TypeError(
        "Unsupported reconstruction marker on a text array; the marker text is preserved "
        "and never evaluated."
    )


def unicode_array_like(flat: np.ndarray[Any, Any], shape: tuple[int, ...]) -> np.ndarray[Any, Any]:
    """Reshape column-major object elements into a row-major object array of ``shape``."""
    out = np.empty(shape, dtype=object)
    if flat.size:
        order = np.arange(flat.size).reshape(shape, order="F").ravel(order="C")
        out.reshape(-1)[:] = flat[order]
    return out
