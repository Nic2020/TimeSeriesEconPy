# SPDX-License-Identifier: MIT
"""Read and write DataEcon scalars and numeric or Boolean series.

Scalars cover every Julia numeric width with a NumPy scalar type (Float16/32/64,
Int8/16/32/64, UInt8/16/32/64, Complex64/128), strings, and MIT dates or
Durations over every core frequency (Unit, Daily, BDaily, Weekly with any end
day, Monthly, Quarterly, HalfYearly and Yearly); series carry supported numeric
NumPy dtypes, Boolean values, or StoredSeries carriers for MIT/Duration,
Int128/UInt128 and ComplexF16 elements over the monthly, quarterly, half-yearly,
annual, daily, business-daily, weekly and Unit frequencies. Plain numeric/Boolean
NumPy arrays of one to five dimensions, text vectors, represented ``StoredArray``
carriers and lossless integer/MIT ranges use ``write_array``/``read_array``;
``MVTSeries`` travel through the series methods. Files open read-only by
default, append without overwriting with ``"a"``, or truncate with ``"w"``; explicit
``overwrite=True`` writes, ``delete``, ``truncate``, ``is_empty`` and
in-memory databases (``open_dataecon_memory``) mirror the Julia file
operations. Objects live in nested catalogs addressed by ``/``-separated
paths (``new_catalog``, ``list_objects``, ``catalog_size``, ``exists``,
``object_info``); string attributes are read and written per object
(``get_attribute``, ``get_attributes``, ``set_attribute``). Use
``open_dataecon`` as a context manager. Whole ``Workspace`` trees travel
through ``write_workspace``/``read_workspace`` (nested Workspaces are
catalogs; members that cannot be stored or loaded are reported, or raised
with ``strict=True``) and the one-call ``save_workspace``/``load_workspace``
file forms. The native extension loads on first use; importing the core
package does not require it. Marker-mapped scalars (``Symbol``, ``Rational``,
``Date``, ...), Int128/UInt128/ComplexF16 scalars and represented
``MVTSeries`` elements are not supported yet.
"""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import TYPE_CHECKING, Literal

from tsecon.mvtseries import MVTSeries
from tsecon.workspace import Workspace

from ._arrays import MAX_UNICODE_BYTES, StoredArray, StoredText
from ._codec import (
    ArrayValue,
    MatrixPayload,
    ScalarResult,
    ScalarValue,
    SeriesValue,
    TensorPayload,
    decode_array,
    decode_matrix,
    decode_scalar,
    decode_series,
    encode_array,
    encode_scalar,
    encode_series,
)
from ._errors import DataEconError
from ._represented import COMPLEXF16, INT128, UINT128, StoredElement, StoredSeries
from ._workspace import (
    LoadedWorkspace,
    SkippedMember,
    WorkspaceReport,
    classify,
    read_tree,
    reader_for,
    write_tree,
)

if TYPE_CHECKING:
    from . import _native

__all__ = [
    "COMPLEXF16",
    "INT128",
    "MAX_UNICODE_BYTES",
    "UINT128",
    "ArrayValue",
    "DataEconError",
    "DataEconFile",
    "LoadedWorkspace",
    "ObjectInfo",
    "ScalarResult",
    "ScalarValue",
    "SeriesValue",
    "SkippedMember",
    "StoredArray",
    "StoredElement",
    "StoredSeries",
    "StoredText",
    "WorkspaceReport",
    "load_workspace",
    "open_dataecon",
    "open_dataecon_memory",
    "save_workspace",
]

_loader_lock = RLock()
_dll_directories: list[object] = []
MEMORY_PATH = ":memory:"


def _open_native(path: str, readonly: bool, memory: bool = False) -> _native.FileHandle:
    with _loader_lock:
        if sys.platform == "win32" and not _dll_directories:
            binary = Path(__file__).parent / "_binary"
            if binary.is_dir():
                _dll_directories.append(os.add_dll_directory(str(binary)))
        try:
            backend = importlib.import_module("tsecon.dataecon._native")
        except (ImportError, OSError) as exc:
            raise ImportError(
                "DataEcon native support is unavailable. Install a wheel with DataEcon support "
                "or build with TSECON_DATAECON_ROOT configured; see the DataEcon build guide."
            ) from exc
        return backend.FileHandle(path, readonly, memory)  # type: ignore[no-any-return]


ROOT_PATH = "/"
RESERVED_ATTRIBUTES = frozenset({"jtype", "jeltype", "DE_VERSION"})
_KINDS = {12: "series", 21: "series", 10: "array", 11: "array", 20: "array", 30: "array"}


def _validate_component(component: str, path: str) -> None:
    # The native name rule: nonempty, not blank, no '/', and (a C string) no NUL.
    if not component:
        raise ValueError(
            f"Invalid DataEcon path {path!r}: empty component (repeated, leading or trailing '/')."
        )
    if "\0" in component:
        raise ValueError(f"Invalid DataEcon path {path!r}: NUL is not allowed.")
    if not component.strip():
        raise ValueError(f"Invalid DataEcon path {path!r}: blank component.")


def _split_path(path: str) -> list[str]:
    """Split a path into validated components; the root is the empty list.

    A leading ``/`` is optional (paths are root-relative); ``/`` and ``""`` are
    the root catalog. Components are separated by ``/`` and each must be a
    valid native object name. Dots, backslashes and blanks inside a name are
    ordinary characters on every platform.
    """
    if not isinstance(path, str):
        raise TypeError("DataEcon paths must be strings.")
    if path in ("", "/"):
        return []
    text = path[1:] if path.startswith("/") else path
    components = text.split("/")
    for component in components:
        _validate_component(component, path)
    return components


def _normalize(path: str) -> str:
    """Return the canonical full path (``/a/b``, or ``/`` for the root)."""
    return "/" + "/".join(_split_path(path))


def _object_path(path: str) -> str:
    """Canonical full path of a non-root object (reads, writes and deletes)."""
    components = _split_path(path)
    if not components:
        raise ValueError("The root catalog '/' is not an object; give an object path.")
    return "/" + "/".join(components)


def _parent_and_leaf(path: str) -> tuple[str, str]:
    """Split an object path into its parent catalog path and leaf name."""
    components = _split_path(path)
    if not components:
        raise ValueError("The root catalog '/' is not an object; give an object path.")
    return "/" + "/".join(components[:-1]), components[-1]


def _validate_name(name: str) -> None:
    # Retained for the handle-level name checks and unit tests: a single
    # component, never a path.
    if not isinstance(name, str):
        raise TypeError("DataEcon object names must be strings.")
    if not name or "/" in name or "\0" in name or not name.strip():
        raise ValueError("Expected a nonempty, non-blank object name without '/' or NUL.")


def _validate_attribute_name(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("DataEcon attribute names must be strings.")
    if "\0" in name:
        raise ValueError("Attribute names cannot contain NUL (the native library truncates).")


@dataclass(frozen=True, slots=True)
class ObjectInfo:
    """One catalog entry or object record.

    ``kind`` is ``"catalog"``, ``"scalar"``, ``"series"`` (a dated ``TSeries``
    or ``MVTSeries``, read with ``read_series``), ``"array"`` (a plain vector,
    range, matrix or tensor, read with ``read_array``) or ``"unknown"`` for a
    native-only object type. ``depth`` is the absolute depth (the root is 0).
    ``id`` and ``parent_id`` are the native object ids: they are not promised
    to survive replacement, deletion or truncation.
    """

    id: int
    parent_id: int
    path: str
    name: str
    kind: str
    object_class: int
    object_type: int
    depth: int


def _info(record: tuple[int, int, int, int, str, str, int, int]) -> ObjectInfo:
    oid, pid, obj_class, obj_type, name, fullpath, depth, _created = record
    if obj_class == 0:
        kind = "catalog"
    elif obj_class == 1:
        kind = "scalar"
    else:
        kind = _KINDS.get(obj_type, "unknown")
    return ObjectInfo(oid, pid, fullpath, name, kind, obj_class, obj_type, depth)


class DataEconFile:
    """Own a DataEcon file; prefer construction through ``open_dataecon``.

    Closing commits native work. A failed close quarantines the owner: it cannot
    be used or retried, and native resources may remain until process exit.
    Context exit preserves a body exception and adds any close failure as a note.
    """

    def __init__(self, path: str | os.PathLike[str], mode: Literal["r", "a", "w"] = "r") -> None:
        if mode not in ("r", "a", "w"):
            raise ValueError(
                "DataEcon mode must be 'r' (read-only), 'a' (create/append) or "
                "'w' (create/truncate)."
            )
        self.path = os.fspath(path)
        if not isinstance(self.path, str):
            raise TypeError("DataEcon paths must be strings or string-valued PathLike objects.")
        if not self.path or "\0" in self.path:
            raise ValueError("DataEcon paths must be nonempty and contain no NUL.")
        if self.path == MEMORY_PATH:
            # SQLite's in-memory name must never be normalized into a disk path.
            raise ValueError(
                "Use open_dataecon_memory() for an in-memory DataEcon database; "
                "':memory:' is not a file path."
            )
        self.path = os.path.abspath(self.path)
        if sys.platform == "win32" and not self.path.isascii():
            raise ValueError(
                "DataEcon 0.4.0 on Windows requires an ASCII file path; "
                "Unicode object names are supported."
            )
        self._lock = RLock()
        self._state = "closed"
        self._mode = mode
        self._handle = _open_native(self.path, mode == "r")
        self._state = "open"
        if mode == "w":
            self.truncate()

    @classmethod
    def in_memory(cls) -> DataEconFile:
        """Open a private, writable, empty in-memory database (see ``open_dataecon_memory``)."""
        self = cls.__new__(cls)
        self.path = MEMORY_PATH
        self._lock = RLock()
        self._state = "closed"
        self._mode = "a"
        self._handle = _open_native(MEMORY_PATH, False, memory=True)
        self._state = "open"
        return self

    @property
    def closed(self) -> bool:
        """Whether this owner is no longer usable, including a failed close."""
        return self._state != "open"

    def _require_open(self) -> None:
        if self.closed:
            raise ValueError(f"DataEcon file is {self._state}.")

    def _require_writable(self) -> None:
        # Enforced in Python before any native call: a native write, delete or
        # truncate on a read-only connection would leave a failed statement
        # behind, and the native close/truncate finalization path is unsafe
        # after that (see the design record for the observed access violation).
        if self._mode == "r":
            raise ValueError("Cannot write through a read-only DataEcon file.")

    def read_series(self, name: str) -> SeriesValue:
        """Read one series or MVTSeries by path into owning storage.

        ``name`` is a root name or a nested catalog path such as
        ``"/inputs/gdp"`` (the leading ``/`` is optional); a missing object or
        catalog raises ``DataEconError`` (code -989).

        A multivariate object returns an ``MVTSeries`` with its dated row axis
        and named columns, whose values own writable storage. Column names are
        stored newline-joined, so a name can contain neither a newline nor NUL;
        duplicate names and a zero-column object are refused rather than read
        back as a narrower value. A plain matrix is read with
        :meth:`read_array` instead.

        The result survives file closure.

        Ordinary numeric and Boolean elements return ``TSeries``. Dates,
        durations, Int128/UInt128 and ComplexF16 elements return ``StoredSeries``
        with their exact carrier bytes and separate element frequency. A Bool
        marker on a nonempty wide carrier is preserved; explicit ``to_bool()``
        converts it. On ordinary carriers it produces Boolean values directly.
        Any other supported reconstruction marker (a numeric, ``MIT{F}`` or
        ``Duration{F}`` element token, or a whole-object ``TSeries``/``Vector``
        token) is preserved: the result is a ``StoredSeries`` holding the stored
        kind, bytes and exact marker text, and ``to_interpreted()`` returns the
        value Julia's loader would build. A marker whose conversion the stored
        values cannot satisfy raises ``ValueError``; unknown tokens raise
        ``TypeError``. No marker text is evaluated.

        The stored axis must carry a supported frequency code and a first date
        inside the reliable native range of that frequency; quarterly,
        half-yearly and annual series also decode their last date, while
        calendar series follow Julia and
        check only the stored first date (later observations are consecutive
        codes). Calendar codes are never converted through ``datetime``, so
        years outside 1..9999 are returned as codes. The native object is
        loaded first; frequency, axis, length, payload size and date bounds
        are validated before the borrowed payload is copied. Reconstruction
        attributes and native date decoding are checked afterward, before
        constructing the result. Unsupported or malformed data raises
        ``TypeError`` or ``ValueError``.
        """
        with self._lock:
            self._require_open()
            payload, metadata, _, marker, object_marker, names = self._handle.read(
                _object_path(name)
            )
            if len(metadata) == 13:
                matrix = decode_matrix(metadata, payload, marker, object_marker, names)
                if not isinstance(matrix, MVTSeries):
                    raise TypeError(
                        "This object is a plain DataEcon matrix; read it with read_array."
                    )
                return matrix
            return decode_series(
                metadata[6],
                metadata[7],
                payload,
                metadata[2],
                metadata[3],
                metadata[5],
                marker,
                object_marker,
            )

    def write_series(self, name: str, series: SeriesValue, *, overwrite: bool = False) -> None:
        """Write a series or MVTSeries by path; an existing name fails unless overwritten.

        ``name`` may be a nested path; every catalog on it must already exist
        (``DataEconError`` -989 otherwise, nothing is created implicitly) and
        the parent must be a catalog (``ValueError``).

        An ``MVTSeries`` stores its dated row axis, its newline-joined column
        names and a column-major payload. Its element rules are the ordinary
        numeric and Boolean ones; represented MVTSeries elements are not
        supported yet.

        ``StoredSeries`` additionally carries full-Int64 date/duration element
        codes (without scalar date packing), Int128/UInt128 or ComplexF16 bytes.
        Its element frequency is independent of the dated axis. Live carrier
        and marker validation precede snapshotting and any overwrite deletion.
        Empty wide carriers need a type marker; empty date/duration series
        retain Julia's marker but remain unloadable by the pinned Julia loader.
        Nonempty Bool-marked wide carriers preserve bytes, type and marker;
        marker failure leaves unmarked wide values. A wide Bool-marked empty
        cannot preserve its width and is refused. Every other preserved marker
        is written back verbatim (``jeltype`` first, then ``jtype``), after the
        interpretation has been validated against the snapshot.

        Ordinary series carry native-endian signed/unsigned integers through
        64 bits, float16/32/64, complex64/128 or Boolean values over the Monthly,
        Quarterly, HalfYearly, Yearly, Daily, BDaily, Weekly or Unit frequency
        (any fiscal anchor or week end day). The first date must lie inside the
        reliable native range of the frequency and reproduce through the
        native codec before anything is stored; year/period series also check
        their last date, and calendar series may run past the window like
        Julia's; Unit codes use the full signed-64-bit range without a native
        date codec (``ValueError`` otherwise; ``TypeError`` for other dtypes or
        non-series input).

        With ``overwrite=True`` an existing root scalar or series of that name
        is deleted after the new series has been fully validated and just
        before it is stored (Julia's delete-then-store); an existing catalog is
        refused with ``ValueError``. Overwrite is not atomic and has no
        rollback: once the delete has run the original is gone, and a native
        store failure after it leaves the name either absent or holding a
        partial, unreadable replacement (the native store creates the object
        before its payload). A native write failure may also leave an unused
        axis. No transaction/rollback or automatic file deletion is promised.
        A failed reconstruction-marker write can leave a readable object with
        a different dtype: Bool becomes Int8 and narrow empties use the native
        wide default; a preserved foreign marker that fails to write leaves the
        plain stored values, and a failed whole-object marker write after a
        successful element marker leaves that element marker active, so the
        object then reads as a different value or is refused. The error is
        reported; no implicit cleanup delete is attempted.
        After a native store or marker failure, close may also fail and quarantine
        the owner. Do not retry that close; reopen the file to inspect the residue.

        Boolean series carry Julia's Bool marker. Empty narrow/Boolean series
        need a marker to preserve dtype; Julia reloads these as plain vectors,
        while Python preserves their stored date in an owning TSeries.
        """
        with self._lock:
            self._require_open()
            parent, leaf = _parent_and_leaf(name)
            encoded = encode_series(series)
            self._require_writable()
            if isinstance(encoded, MatrixPayload):
                self._write_matrix(leaf, encoded, overwrite, parent)
                return
            self._handle.write(
                leaf,
                encoded.frequency,
                encoded.first,
                encoded.payload,
                bool(overwrite),
                encoded.element,
                encoded.element_frequency,
                encoded.length,
                encoded.marker,
                encoded.object_marker,
                parent=parent,
            )

    def read_array(self, name: str) -> ArrayValue:
        """Read a plain array, text vector or unit-step range by path.

        Ordinary numeric and Boolean vectors, matrices and tensors of three to
        five dimensions return owning, writable, C-contiguous NumPy arrays of
        the stored shape, including singleton and zero-length dimensions.
        Integer ranges return Python ``range`` values and MIT ranges return
        ``MITRange``.

        Text vectors return a ``list[str]`` when every element is valid UTF-8
        and no foreign marker is stored; text matrices and tensors (two to
        five dimensions) return an owning NumPy ``str_`` array of the stored
        shape under the same conditions, provided that fixed-width array
        would not exceed ``MAX_UNICODE_BYTES`` (count times the longest
        element times four). Anything else returns a :class:`StoredText`,
        which keeps each element's exact bytes, the preserved marker and the
        shape, with ``tolist()``/``to_numpy()`` as explicit decodes.
        Elements carrying a date, duration, 128-bit or ComplexF16 encoding, or
        a preserved reconstruction marker, return a :class:`StoredArray`: a
        contiguous carrier plus its stored element descriptor, with
        ``to_interpreted()`` as the explicit conversion. A ``StoredArray`` has
        no first date; a dated matrix is an ``MVTSeries`` read with
        :meth:`read_series` instead.

        Reconstruction text is matched against a finite table and never
        evaluated. A marker whose conversion the stored values cannot satisfy
        raises ``ValueError``; an unsupported token raises ``TypeError``.
        """
        with self._lock:
            self._require_open()
            payload, metadata, _, marker, object_marker, names = self._handle.read_array(
                _object_path(name)
            )
            return decode_array(metadata, payload, marker, object_marker, names)

    def write_array(self, name: str, value: ArrayValue, *, overwrite: bool = False) -> None:
        """Write a plain array, text vector or lossless unit-step range by path.

        NumPy arrays of one to five dimensions (DataEcon stores at most five
        axes) retain their supported native numeric/Boolean dtype. Any input
        layout is accepted: C-contiguous, Fortran-contiguous, sliced and
        transposed inputs are snapshotted in logical order before any native
        mutation, without being retained or modified. Matrices and tensors are
        stored column-major, as DataEcon expects.

        Text is written from a flat or rectangular nested sequence of ``str``,
        a NumPy ``str_`` array of one to five dimensions (any layout), or a
        :class:`StoredText`, which also carries a preserved marker, exact bytes
        and a shape. Matrices and tensors are packed column-major like numeric
        ones. Element bytes are sized in UTF-8, not characters. An element
        cannot contain NUL, because that byte separates the packed elements.
        NumPy ``bytes_`` and object arrays are refused rather than converted.

        :class:`StoredArray` writes its carrier, element descriptor and any
        preserved markers unchanged. Python integer ranges are accepted only as
        ``range(1, stop)`` because DataEcon stores their length but not their
        starting value. ``MITRange`` retains its frequency and first code,
        including Unit codes, and must have step one. Empty ranges keep their
        Python range form even though Julia's reconstruction marker makes its
        loader return a typed empty vector.

        Overwrite validates first, then deletes and stores without rollback,
        with the same residue and close-failure rules as other writes.
        """
        with self._lock:
            self._require_open()
            parent, leaf = _parent_and_leaf(name)
            encoded = encode_array(value)
            self._require_writable()
            if isinstance(encoded, MatrixPayload):
                self._write_matrix(leaf, encoded, overwrite, parent)
                return
            if isinstance(encoded, TensorPayload):
                self._handle.write_tensor(
                    leaf,
                    encoded.object_type,
                    encoded.element,
                    encoded.element_frequency,
                    encoded.shape,
                    encoded.payload,
                    bool(overwrite),
                    encoded.marker,
                    encoded.object_marker,
                    parent=parent,
                )
                return
            self._handle.write_array(
                leaf,
                encoded.object_type,
                encoded.axis_type,
                encoded.frequency,
                encoded.first,
                encoded.payload,
                bool(overwrite),
                encoded.element,
                encoded.element_frequency,
                encoded.length,
                encoded.marker,
                encoded.object_marker,
                parent=parent,
            )

    def _write_matrix(
        self, name: str, encoded: MatrixPayload, overwrite: bool, parent: str = ROOT_PATH
    ) -> None:
        """Store an encoded two-dimensional payload; the caller holds the file lock.

        Column-major bytes, both axes and any markers were fully validated by
        the codec. The store itself is not atomic: an overwrite deletes first,
        and a later axis, store or attribute failure leaves the residue the
        one-dimensional writes document.
        """
        self._handle.write_matrix(
            name,
            encoded.object_type,
            encoded.element,
            encoded.element_frequency,
            encoded.axis1_type,
            encoded.rows,
            encoded.frequency,
            encoded.first,
            encoded.columns,
            encoded.names,
            encoded.payload,
            bool(overwrite),
            encoded.marker,
            encoded.object_marker,
            parent=parent,
        )

    def read_scalar(self, name: str) -> ScalarResult:
        """Read a scalar by path as an exact-width Python or NumPy value.

        Float64, Int64 and ComplexF64 objects return Python ``float``, ``int``
        and ``complex``; every other supported width returns the sized NumPy
        scalar of the same width (``np.float16``/``np.float32``,
        ``np.int8``/``np.int16``/``np.int32``, ``np.uint8``..``np.uint64``,
        ``np.complex64``); strings, dates and durations return ``str``, ``MIT``
        and ``Duration``. The result is independent of the file and survives
        closure. Integers, dates and durations are decoded exactly and never
        pass through floating point; narrow floats keep their stored bits;
        strings are decoded strictly as UTF-8. Dates are validated against the
        reliable native range of their frequency; Unit dates are plain 64-bit
        codes and are not range-checked. Int128, UInt128 and ComplexF16
        objects, which Julia can write, raise ``ValueError`` until a Python
        representation is chosen; a stored Julia ``Bool`` reads as ``np.int8``,
        exactly as Julia itself reloads it.
        """
        with self._lock:
            self._require_open()
            payload, metadata, _ = self._handle.read_scalar(_object_path(name))
            return decode_scalar(metadata[1], metadata[2], payload)

    def write_scalar(self, name: str, value: ScalarValue, *, overwrite: bool = False) -> None:
        """Write a scalar by path; by default an existing name fails and is never replaced.

        ``name`` may be a nested path whose catalogs all exist; the parent
        must be a catalog. Missing catalogs are never created implicitly.

        With ``overwrite=True`` an existing root scalar or series of that name
        is deleted after the new value has been fully validated and just before
        it is stored; an existing catalog is refused with ``ValueError``.
        Overwrite is not atomic and has no rollback: once the delete has run
        the original is gone, and a native store failure after it leaves the
        name either absent or holding a partial, unreadable replacement (the
        native store creates the object before its payload).

        Accepted exactly by type: Python bool / NumPy bool_ (stored as Int8),
        Python float / NumPy float64 (Float64),
        NumPy float32 / float16, Python int / NumPy int64 (Int64), NumPy
        int8 / int16 / int32, NumPy uint8 / uint16 / uint32 / uint64, Python
        complex / NumPy complex128 (ComplexF64), NumPy complex64, str (UTF-8
        string), MIT (date) and Duration over the Unit, Daily, BDaily, Weekly,
        Monthly, Quarterly, HalfYearly or Yearly frequencies. Each NumPy
        scalar is stored at its own width from its own bytes, so precision is
        whatever the caller already chose and nothing is widened. Booleans are
        normalized to Int8 zero/one without a marker, like Julia, and read
        back as np.int8. Arrays, bytes, Decimal, Fraction, subclasses and other types are
        rejected without implicit conversion. Out-of-range integers or
        durations, dates outside the reliable native range of their frequency,
        and strings containing NUL or lone surrogates raise ValueError. Native
        failures may leave a partial object; no rollback is promised.
        """
        with self._lock:
            self._require_open()
            parent, leaf = _parent_and_leaf(name)
            kind, frequency, payload = encode_scalar(value)
            self._require_writable()
            self._handle.write_scalar(
                leaf, kind, frequency, payload, bool(overwrite), parent=parent
            )

    def delete(self, name: str, *, recursive: bool = False) -> None:
        """Delete one object by path and everything stored with it.

        A missing path raises ``DataEconError`` (code -989) and the owner stays
        usable. A catalog is refused with ``ValueError`` unless
        ``recursive=True``, in which case every nested catalog and object goes
        with it (the native format cascades). The root catalog is always
        refused (use ``truncate``). Deleting a series leaves its shared axis
        row in place; values returned by earlier reads remain valid. Read-only
        owners reject deletion before any native call.
        """
        with self._lock:
            self._require_open()
            path = _object_path(name)
            self._require_writable()
            self._handle.delete(path, bool(recursive))

    # -- catalogs and addressing ------------------------------------------------

    def new_catalog(self, path: str, *, parents: bool = False, exist_ok: bool = False) -> None:
        """Create a catalog at ``path`` (Julia's ``new_catalog``).

        Every catalog above it must exist unless ``parents=True``, which creates
        the missing intermediate catalogs one at a time (not atomic: a failure
        leaves the catalogs created so far). An existing object of any class at
        ``path`` raises ``DataEconError`` (code -985), except an existing
        catalog when ``exist_ok=True``. Replacing an object by a catalog is an
        explicit ``delete`` first; there is no overwrite form. Read-only owners
        refuse before any native call.
        """
        with self._lock:
            self._require_open()
            components = _split_path(path)
            if not components:
                raise ValueError("The root catalog always exists.")
            self._require_writable()
            if exist_ok or parents:
                existing = self._describe_or_none("/" + "/".join(components))
                if existing is not None:
                    if existing.kind == "catalog" and exist_ok:
                        return
                    raise DataEconError(
                        -985, "new_catalog", self.path, "Object already exists.", path
                    )
            if parents:
                for depth in range(1, len(components)):
                    prefix = "/" + "/".join(components[:depth])
                    found = self._describe_or_none(prefix)
                    if found is None:
                        self._handle.new_catalog(
                            components[depth - 1], parent="/" + "/".join(components[: depth - 1])
                        )
                    elif found.kind != "catalog":
                        raise ValueError(f"Cannot create {path!r}: {prefix!r} is not a catalog.")
            self._handle.new_catalog(components[-1], parent="/" + "/".join(components[:-1]))

    def _describe_or_none(self, path: str) -> ObjectInfo | None:
        # Caller holds the file lock and passes a normalized path.
        try:
            return _info(self._handle.describe_path(path))
        except DataEconError as exc:
            if exc.code == -989:
                return None
            raise

    def exists(self, path: str) -> bool:
        """Whether an object or catalog exists at ``path`` (the root always does)."""
        with self._lock:
            self._require_open()
            return bool(self._handle.exists(_normalize(path)))

    def object_info(self, path: str) -> ObjectInfo:
        """Describe the object or catalog at ``path`` (``DataEconError`` -989 if missing)."""
        with self._lock:
            self._require_open()
            return _info(self._handle.describe_path(_normalize(path)))

    def object_info_by_id(self, object_id: int) -> ObjectInfo:
        """Describe an object by its native id (the explicit advanced interface).

        Ids come from :attr:`ObjectInfo.id` or :meth:`object_id`; a stale id
        raises ``DataEconError`` (code -989) and a negative one ``ValueError``.
        """
        with self._lock:
            self._require_open()
            return _info(self._handle.describe_id(object_id))

    def object_id(self, path: str) -> int:
        """Native object id of the object at ``path`` (0 for the root).

        Ids are not promised to survive replacement (``overwrite=True`` assigns
        a new id), deletion or truncation (ids restart).
        """
        return self.object_info(path).id

    def object_path(self, object_id: int) -> str:
        """Full path of the object with the given native id (``/`` for 0)."""
        return self.object_info_by_id(object_id).path

    def catalog_size(self, path: str = ROOT_PATH) -> int:
        """Count the objects directly inside a catalog (Julia's ``catalog_size``).

        A non-catalog path raises ``ValueError``; a missing one ``DataEconError``.
        """
        with self._lock:
            self._require_open()
            return int(self._handle.catalog_size(_normalize(path)))

    def list_objects(
        self, path: str = ROOT_PATH, *, recursive: bool = False, max_depth: int | None = None
    ) -> list[ObjectInfo]:
        """List the members of a catalog as owned :class:`ObjectInfo` records.

        Members are sorted by the UTF-8 bytes of their names (the order the
        native library returns); catalogs are included as entries. With
        ``recursive=True`` the listing descends into catalogs (never into the
        children of other objects, as Julia) in depth-first pre-order, at most
        ``max_depth`` levels below ``path`` (``1`` is the immediate members;
        ``None`` is unlimited). Each level is one native search that is fully
        consumed and finalized before anything is returned; no handle outlives
        the call. A non-catalog path raises ``ValueError``.
        """
        if max_depth is not None and (type(max_depth) is not int or max_depth < 0):
            raise ValueError("max_depth must be a nonnegative integer or None.")
        limit = max_depth if recursive else min(1, max_depth if max_depth is not None else 1)
        with self._lock:
            self._require_open()
            start = self._handle.describe_path(_normalize(path))
            if start[2] != 0:
                raise ValueError(f"{path!r} is not a catalog.")
            if limit == 0:
                return []
            base = "" if start[5] == "/" else start[5]
            results: list[ObjectInfo] = []
            # Depth-first pre-order with an explicit stack of entries still to
            # emit: catalog trees can be arbitrarily deep (the writer imposes
            # no limit), so no Python recursion is involved. An entry is
            # appended when popped; if it is a catalog within the depth limit
            # its members are fetched and pushed in reverse byte order, so
            # they are emitted next, before the catalog's later siblings. Each
            # native search is fully consumed and finalized inside
            # list_children before the next one starts.
            stack = self._members(start[0], base, start[6], 1)
            while stack:
                info, level = stack.pop()
                results.append(info)
                if info.object_class == 0 and (limit is None or level < limit):
                    stack.extend(self._members(info.id, info.path, info.depth, level + 1))
            return results

    def _children(self, parent_id: int) -> list[tuple[int, int, int, int, str]]:
        # Owned (id, parent id, class, type, name) rows of one catalog, sorted
        # by the UTF-8 bytes of the name (the native order made explicit); one
        # native search, fully consumed and finalized inside list_children.
        # Caller holds the file lock.
        return sorted(self._handle.list_children(parent_id), key=lambda r: r[4].encode("utf-8"))

    def _members(
        self, parent_id: int, parent_path: str, parent_depth: int, level: int
    ) -> list[tuple[ObjectInfo, int]]:
        # The members of one catalog as (record, level) pairs in reverse UTF-8
        # byte order, ready to be pushed on the traversal stack.
        rows = reversed(self._children(parent_id))
        return [
            (
                _info(
                    (
                        oid,
                        pid,
                        obj_class,
                        obj_type,
                        name,
                        f"{parent_path}/{name}",
                        parent_depth + 1,
                        0,
                    )
                ),
                level,
            )
            for oid, pid, obj_class, obj_type, name in rows
        ]

    # -- workspaces and generic objects --------------------------------------------

    def write_workspace(
        self,
        workspace: Workspace,
        path: str = ROOT_PATH,
        *,
        overwrite: bool = False,
        strict: bool = False,
    ) -> WorkspaceReport:
        """Store every member of a ``Workspace`` below a catalog (Julia's ``writedb``).

        Each member is stored under its key through the typed writer its
        Python type selects: a nested ``Workspace`` becomes a catalog holding
        its own members; ``TSeries``, ``MVTSeries`` and ``StoredSeries`` go
        through :meth:`write_series`; NumPy arrays, ``StoredArray``,
        ``StoredText``, ``list``/``tuple`` text, ``range`` and ``MITRange``
        through :meth:`write_array`; ``bool``, ``int``, ``float``, ``complex``,
        ``str``, ``MIT``, ``Duration`` and NumPy scalars through
        :meth:`write_scalar`, each with that method's own validation, markers
        and residue rules and encoded once. Members are visited depth-first in
        insertion order from an explicit stack, so any nesting depth is fine.
        A key must be a valid object name (nonempty, not blank, no ``/`` or
        NUL); it is a name, never a path.

        ``path`` is the destination catalog (``"/"`` is the root); it must
        exist (``DataEconError`` -989) and be a catalog (``ValueError``), and
        is never created here. To store a Workspace as a new catalog ``name``
        under ``parent``, write ``Workspace(name=ws)`` into ``parent``.

        By default a member that cannot be stored is skipped and recorded in
        the returned :class:`WorkspaceReport` with its full path, category and
        reason, and the traversal continues (Julia logs and continues): a
        value of an unsupported type, a value the typed writer refuses, an
        invalid key, an existing name without ``overwrite``, a native store
        failure, or a nested Workspace that is its own ancestor (a cycle; the
        same Workspace under two names is stored twice). A nested Workspace
        that cannot be created is one entry with ``subtree=True`` and none of
        its members is attempted. With ``strict=True`` the first such member
        raises its original ``TypeError``, ``ValueError`` or
        ``DataEconError`` with a note naming the member; members already
        stored stay stored. Neither mode is a transaction: nothing is rolled
        back, and a native failure leaves the residue :meth:`write_series`
        documents (a partial object, a possible later close failure).

        With ``overwrite=True`` a member replaces an existing object of its
        name: a value replaces a scalar/series/array through the typed
        writer's delete-then-store; a nested Workspace replaces an existing
        catalog **entirely** (the old contents are deleted first; there is no
        merge) or an existing value; a value never replaces a catalog (that
        member is reported as ``exists``; delete it explicitly). Replaced
        objects lose their attributes. Only the codec's own markers are
        written; user attributes are not part of a Workspace.
        """
        if not isinstance(workspace, Workspace):
            raise TypeError("write_workspace requires a Workspace.")
        with self._lock:
            self._require_open()
            normalized = _normalize(path)
            self._require_writable()
            start = self._handle.describe_path(normalized)
            if start[2] != 0:
                raise ValueError(f"{path!r} is not a catalog.")
            base = "" if normalized == ROOT_PATH else normalized
            return write_tree(self, workspace, base, normalized, bool(overwrite), bool(strict))

    def read_workspace(self, path: str = ROOT_PATH, *, strict: bool = False) -> LoadedWorkspace:
        """Load a catalog and everything below it into a ``Workspace`` (Julia's ``readdb``).

        Returns a :class:`LoadedWorkspace` named tuple ``(workspace, report)``.
        Every member is read with the typed reader for its stored class and
        type - :meth:`read_scalar`, :meth:`read_series` or :meth:`read_array`
        - and inserted unchanged, so representations are exactly what those
        readers return (``StoredSeries``/``StoredArray``/``StoredText``
        carriers with their markers, empty series with their axis, ``range``
        and ``MITRange``); nested catalogs become nested Workspaces. Each
        catalog is listed once and its members inserted in the UTF-8 byte
        order of their names (the native order, also Julia's), not in the
        order they were written; the traversal is depth-first pre-order from
        an explicit stack (a nested catalog is filled before its later
        siblings, as Julia recurses), so any depth is fine. Before a member is
        read, the path rebuilt from its listed name must resolve to the listed
        object itself and the name must be a valid, unused key: a foreign
        object whose name holds NUL (cut short by the C string), ``/`` or
        repeats a sibling is reported as ``invalid`` rather than read as
        another object or silently overwriting a key. Children of non-catalog
        objects are never visited. The result owns its storage and survives
        closing the file.

        By default a member that cannot be loaded is skipped and recorded in
        the report with its path, category and reason (Julia logs and
        continues): a native-only object type (no native call is made), a
        scalar carrying a reconstruction marker, an unsupported or malformed
        encoding, or a native load failure. A catalog whose listing fails is
        one entry with ``subtree=True`` and is left out of its parent; a
        failing member inside a catalog is reported alone and its siblings
        survive. With ``strict=True`` the first such member raises its
        original exception with a note naming it. ``path`` must be a catalog
        (``ValueError`` otherwise; use :meth:`read_object` for one object).
        Attributes other than the codec's markers are not read.
        """
        with self._lock:
            self._require_open()
            normalized = _normalize(path)
            start = self._handle.describe_path(normalized)
            if start[2] != 0:
                raise ValueError(f"{path!r} is not a catalog; use read_object for one object.")
            base = "" if normalized == ROOT_PATH else normalized
            return read_tree(self, start[0], base, normalized, bool(strict))

    def write_object(self, path: str, value: object, *, overwrite: bool = False) -> None:
        """Store one value through the typed writer for its type (Julia's ``write_data``).

        The dispatch is the one :meth:`write_workspace` uses; the typed
        method's rules apply unchanged. A ``Workspace`` is refused with
        ``TypeError`` (store it with :meth:`write_workspace`), as is any type
        without a storage class.
        """
        family = classify(value)
        if family == "workspace":
            raise TypeError(
                "A Workspace is a catalog tree; store it with "
                "write_workspace(Workspace(name=ws), parent)."
            )
        if family == "series":
            self.write_series(path, value, overwrite=overwrite)  # type: ignore[arg-type]
        elif family == "array":
            self.write_array(path, value, overwrite=overwrite)  # type: ignore[arg-type]
        elif family == "scalar":
            self.write_scalar(path, value, overwrite=overwrite)  # type: ignore[arg-type]
        else:
            raise TypeError(
                f"No DataEcon storage class for a value of type {type(value).__name__}."
            )

    def read_object(self, path: str) -> object:
        """Read one object with the typed reader its stored class selects (Julia's ``read_data``).

        Scalars, dated series/MVTSeries and plain arrays return exactly what
        :meth:`read_scalar`, :meth:`read_series` and :meth:`read_array`
        return. A catalog raises ``ValueError`` (use :meth:`read_workspace`);
        a native-only object type raises ``TypeError``.
        """
        with self._lock:
            self._require_open()
            info = _info(self._handle.describe_path(_object_path(path)))
            if info.object_class == 0:
                raise ValueError(f"{path!r} is a catalog; read it with read_workspace.")
            reader = reader_for(info.object_class, info.object_type)
            if reader is None:
                raise TypeError(
                    f"Object class {info.object_class} type {info.object_type} is a native-only "
                    "encoding with no Julia or Python loader."
                )
            return getattr(self, reader)(info.path)

    # -- attributes ----------------------------------------------------------------

    def get_attribute(self, path: str, name: str) -> str | None:
        """Read one string attribute of the object or catalog at ``path``.

        Returns ``None`` when the attribute is absent. The reconstruction
        markers ``jtype``/``jeltype`` and the root's ``DE_VERSION`` are readable
        like any other attribute. A SQL NULL value (which the C API can store
        but Julia cannot read) raises ``ValueError``.

        The value is exact for valid UTF-8 text without NUL. The native library returns
        a C string with no byte length, so a value stored by a foreign SQL writer with
        an embedded NUL is cut at that NUL here, exactly as Julia reads it; the ABI
        offers no way to detect the cut.
        """
        _validate_attribute_name(name)
        with self._lock:
            self._require_open()
            oid = self._handle.describe_path(_normalize(path))[0]
            value = self._handle.get_attribute(oid, name)
            return None if value is None else str(value)

    def get_attributes(self, path: str) -> dict[str, str]:
        """Read every attribute of the object or catalog at ``path``.

        The library only enumerates names as one delimiter-joined string plus
        their count, so the names are requested twice, joined by two delimiters
        of equal length that differ at every byte. The first text is split and
        the split is accepted only when the piece count equals the native
        count, no name repeats, and re-joining the pieces with the second
        delimiter reproduces the second text exactly; two different name lists
        cannot satisfy both joins, so acceptance is a proof of the split, not a
        guess. On a collision the delimiters grow (a run a name cannot contain
        once it is longer than any name), so the loop always ends; a damaged
        name (one holding NUL) can never be proved and raises ``ValueError``,
        as does a SQL NULL value. Each value is then read individually with
        :meth:`get_attribute`, so the enumeration is exact for valid UTF-8
        attribute names and values without NUL;
        a foreign value containing NUL is cut at that NUL, as in Julia.
        """
        with self._lock:
            self._require_open()
            oid = self._handle.describe_path(_normalize(path))[0]
            result: dict[str, str] = {}
            for attribute in self._handle.attribute_names(oid):
                value = self._handle.get_attribute(oid, attribute)
                if value is None:
                    raise ValueError(f"Attribute {attribute!r} disappeared while reading {path!r}.")
                result[str(attribute)] = str(value)
            return result

    def set_attribute(self, path: str, name: str, value: str) -> None:
        """Write one string attribute on the object or catalog at ``path``.

        An existing attribute of that name is replaced. Names and values are
        ``str`` without NUL (the native library would truncate at NUL); empty
        strings and any Unicode are stored exactly. The reserved keys
        ``jtype``, ``jeltype`` and ``DE_VERSION`` are refused with
        ``ValueError``: the pinned Julia loader evaluates the first two as
        code, and the third is library-owned file metadata. Read-only owners
        refuse before any native call; the write is one native statement
        without rollback.
        """
        _validate_attribute_name(name)
        if not isinstance(value, str):
            raise TypeError("DataEcon attribute values must be strings.")
        if "\0" in value:
            raise ValueError("Attribute values cannot contain NUL (the native library truncates).")
        if name in RESERVED_ATTRIBUTES:
            raise ValueError(
                f"Attribute {name!r} is reserved: jtype/jeltype are reconstruction markers "
                "evaluated by Julia and DE_VERSION is library metadata."
            )
        with self._lock:
            self._require_open()
            normalized = _normalize(path)
            self._require_writable()
            oid = self._handle.describe_path(normalized)[0]
            self._handle.set_attribute(oid, name, value)

    def truncate(self) -> None:
        """Empty the file so it is as if just created (Julia's ``empty!``).

        Pending writes are committed first, object ids restart and the owner
        stays usable for new writes. Read-only owners reject truncation before
        any native call. A native truncate failure quarantines the owner like a
        failed close: it cannot be used or closed again and native resources
        may remain until process exit.
        """
        with self._lock:
            self._require_open()
            self._require_writable()
            self._state = "truncate-failed"
            self._handle.truncate()
            self._state = "open"

    def is_empty(self) -> bool:
        """Whether the root catalog holds no objects (Julia's ``isempty``)."""
        with self._lock:
            self._require_open()
            return self._handle.catalog_size(ROOT_PATH) == 0

    def close(self) -> None:
        """Close once; quarantine after failure instead of retrying native cleanup."""
        with self._lock:
            if self._state == "closed":
                return
            self._require_open()
            self._state = "close-failed"
            self._handle.close()
            self._state = "closed"

    def __enter__(self) -> DataEconFile:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            if exc is None:
                raise
            exc.add_note(f"DataEcon cleanup also failed: {cleanup}")


def open_dataecon(path: str | os.PathLike[str], mode: Literal["r", "a", "w"] = "r") -> DataEconFile:
    """Open a file: read-only by default, ``a`` creates/appends, ``w`` creates/truncates.

    ``"w"`` empties an existing file immediately after opening it (Julia's
    ``truncate=true``). The literal path ``":memory:"`` is rejected; use
    ``open_dataecon_memory`` for an in-memory database.
    """
    return DataEconFile(path, mode)


def open_dataecon_memory() -> DataEconFile:
    """Open a private, writable, empty in-memory DataEcon database (Julia's ``opendaecmem``).

    The database lives only in this process and is discarded on close; it cannot
    be reopened or shared with Julia. Its ``path`` attribute is the label
    ``":memory:"``; no file of that name is created.
    """
    return DataEconFile.in_memory()


def save_workspace(
    file: str | os.PathLike[str],
    workspace: Workspace,
    path: str = ROOT_PATH,
    *,
    mode: Literal["a", "w"] = "a",
    overwrite: bool = False,
    strict: bool = False,
) -> WorkspaceReport:
    """Open a file, store a ``Workspace`` below a catalog and close (Julia's ``writedb(file, …)``).

    ``mode="a"`` creates or appends (Julia's ``append=true``); ``mode="w"``
    empties an existing file first. The file is closed on every exit path;
    the function owns the handle only for this call. Everything else follows
    :meth:`DataEconFile.write_workspace`, including its report and residue.
    """
    # Argument validation precedes the open: mode "w" truncates an existing
    # file on open, so a wrong Workspace or path argument must be refused
    # before anything is touched (and before a new file is created).
    if mode not in ("a", "w"):
        raise ValueError(
            "save_workspace mode must be 'a' (create/append) or 'w' (create/truncate)."
        )
    if not isinstance(workspace, Workspace):
        raise TypeError("save_workspace requires a Workspace.")
    _normalize(path)
    with open_dataecon(file, mode) as db:
        return db.write_workspace(workspace, path, overwrite=overwrite, strict=strict)


def load_workspace(
    file: str | os.PathLike[str], path: str = ROOT_PATH, *, strict: bool = False
) -> LoadedWorkspace:
    """Open a file read-only, load a catalog into a ``Workspace`` and close (Julia's ``readdb``).

    The returned Workspace and its values own their storage and stay usable
    after the file is closed; the function owns the handle only for this
    call. Everything else follows :meth:`DataEconFile.read_workspace`.
    """
    _normalize(path)
    with open_dataecon(file, "r") as db:
        return db.read_workspace(path, strict=strict)
