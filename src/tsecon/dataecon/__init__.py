# SPDX-License-Identifier: MIT
"""Read and write DataEcon scalars and numeric or Boolean series.

Scalars cover every Julia numeric width with a NumPy scalar type (Float16/32/64,
Int8/16/32/64, UInt8/16/32/64, Complex64/128), strings, and MIT dates or
Durations over every core frequency (Unit, Daily, BDaily, Weekly with any end
day, Monthly, Quarterly, HalfYearly and Yearly); series carry supported numeric
NumPy dtypes, Boolean values, or StoredSeries carriers for MIT/Duration,
Int128/UInt128 and ComplexF16 elements over the monthly, quarterly, half-yearly,
annual, daily, business-daily, weekly and Unit frequencies. Plain one-dimensional
numeric/Boolean NumPy arrays and lossless integer/MIT ranges use
``write_array``/``read_array``. Files open read-only by
default, append without overwriting with ``"a"``, or truncate with ``"w"``; explicit
``overwrite=True`` writes, ``delete``, ``truncate``, ``is_empty`` and
in-memory databases (``open_dataecon_memory``) mirror the Julia file
operations. Use ``open_dataecon`` as a context manager. The native extension
loads on first use; importing the core package does not require it. Other
data types, catalogs and general attributes are not supported yet.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import TYPE_CHECKING, Literal

from tsecon.mvtseries import MVTSeries

from ._arrays import StoredArray, StoredText
from ._codec import (
    ArrayValue,
    MatrixPayload,
    ScalarResult,
    ScalarValue,
    SeriesValue,
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

if TYPE_CHECKING:
    from . import _native

__all__ = [
    "COMPLEXF16",
    "INT128",
    "UINT128",
    "ArrayValue",
    "DataEconError",
    "DataEconFile",
    "ScalarResult",
    "ScalarValue",
    "SeriesValue",
    "StoredArray",
    "StoredElement",
    "StoredSeries",
    "StoredText",
    "open_dataecon",
    "open_dataecon_memory",
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


def _validate_name(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("DataEcon object names must be strings.")
    if not name or "/" in name or "\0" in name:
        raise ValueError("Expected a nonempty root object name without '/' or NUL.")


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
        """Read one root series or MVTSeries into owning storage.

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
            _validate_name(name)
            payload, metadata, _, marker, object_marker, names = self._handle.read(name)
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
        """Write a root series or MVTSeries; an existing name fails unless overwritten.

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
            _validate_name(name)
            encoded = encode_series(series)
            self._require_writable()
            if isinstance(encoded, MatrixPayload):
                self._write_matrix(name, encoded, overwrite)
                return
            self._handle.write(
                name,
                encoded.frequency,
                encoded.first,
                encoded.payload,
                bool(overwrite),
                encoded.element,
                encoded.element_frequency,
                encoded.length,
                encoded.marker,
                encoded.object_marker,
            )

    def read_array(self, name: str) -> ArrayValue:
        """Read a plain array, text vector or unit-step range.

        Ordinary numeric and Boolean vectors and matrices return owning,
        writable, C-contiguous NumPy arrays of the stored shape, including
        singleton and zero-length dimensions. Integer ranges return Python
        ``range`` values and MIT ranges return ``MITRange``.

        Text vectors return a ``list[str]`` when every element is valid UTF-8
        and no foreign marker is stored, and a :class:`StoredText` otherwise,
        which keeps each element's exact bytes and the preserved marker.
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
            _validate_name(name)
            payload, metadata, _, marker, object_marker, names = self._handle.read_array(name)
            return decode_array(metadata, payload, marker, object_marker, names)

    def write_array(self, name: str, value: ArrayValue, *, overwrite: bool = False) -> None:
        """Write a plain array, text vector or lossless unit-step range.

        One- and two-dimensional NumPy arrays retain their supported native
        numeric/Boolean dtype. Any input layout is accepted: C-contiguous,
        Fortran-contiguous, sliced and transposed inputs are snapshotted in
        logical order before any native mutation, without being retained or
        modified. A matrix is stored column-major, as DataEcon expects.

        Text is written from a sequence of ``str`` or from a
        :class:`StoredText`, which also carries a preserved marker and exact
        bytes. Element bytes are sized in UTF-8, not characters. An element
        cannot contain NUL, because that byte separates the packed elements.

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
            _validate_name(name)
            encoded = encode_array(value)
            self._require_writable()
            if isinstance(encoded, MatrixPayload):
                self._write_matrix(name, encoded, overwrite)
                return
            self._handle.write_array(
                name,
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
            )

    def _write_matrix(self, name: str, encoded: MatrixPayload, overwrite: bool) -> None:
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
        )

    def read_scalar(self, name: str) -> ScalarResult:
        """Read a root scalar as an exact-width Python or NumPy value.

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
            _validate_name(name)
            payload, metadata, _ = self._handle.read_scalar(name)
            return decode_scalar(metadata[1], metadata[2], payload)

    def write_scalar(self, name: str, value: ScalarValue, *, overwrite: bool = False) -> None:
        """Write a root scalar; by default an existing name fails and is never replaced.

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
            _validate_name(name)
            kind, frequency, payload = encode_scalar(value)
            self._require_writable()
            self._handle.write_scalar(name, kind, frequency, payload, bool(overwrite))

    def delete(self, name: str, *, recursive: bool = False) -> None:
        """Delete one root object and everything stored with it.

        A missing name raises ``DataEconError`` (code -989) and the owner stays
        usable. A root catalog is refused with ``ValueError`` unless
        ``recursive=True``, in which case every nested catalog and object goes
        with it (the native format cascades). Deleting a series leaves its
        shared axis row in place; values returned by earlier reads remain
        valid. Read-only owners reject deletion before any native call.
        """
        with self._lock:
            self._require_open()
            _validate_name(name)
            self._require_writable()
            self._handle.delete(name, bool(recursive))

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
            return self._handle.catalog_size() == 0

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
