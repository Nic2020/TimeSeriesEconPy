# SPDX-License-Identifier: MIT
"""Read and write DataEcon scalars and numeric or Boolean series.

Scalars cover every Julia numeric width with a NumPy scalar type (Float16/32/64,
Int8/16/32/64, UInt8/16/32/64, Complex64/128), strings, and MIT dates or
Durations over every core frequency (Unit, Daily, BDaily, Weekly with any end
day, Monthly, Quarterly, HalfYearly and Yearly); series carry supported numeric
NumPy dtypes, Boolean values, or StoredSeries carriers for MIT/Duration,
Int128/UInt128 and ComplexF16 elements over the monthly, quarterly, half-yearly,
annual, daily, business-daily and weekly frequencies. Files open read-only by
default, append without overwriting with ``"a"``, or truncate with ``"w"``; explicit
``overwrite=True`` writes, ``delete``, ``truncate``, ``is_empty`` and
in-memory databases (``open_dataecon_memory``) mirror the Julia file
operations. Use ``open_dataecon`` as a context manager. The native extension
loads on first use; importing the core package does not require it. Other
data types, Unit-frequency series, catalogs and general attributes are not
supported yet.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import TYPE_CHECKING, Literal

from ._codec import (
    ScalarResult,
    ScalarValue,
    SeriesValue,
    decode_scalar,
    decode_series,
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
    "DataEconError",
    "DataEconFile",
    "ScalarResult",
    "ScalarValue",
    "SeriesValue",
    "StoredElement",
    "StoredSeries",
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
        """Read one root series into owning storage that survives file closure.

        Ordinary numeric and Boolean elements return ``TSeries``. Dates,
        durations, Int128/UInt128 and ComplexF16 elements return ``StoredSeries``
        with their exact carrier bytes and separate element frequency. A Bool
        marker on a nonempty wide carrier is preserved; explicit ``to_bool()``
        converts it. On ordinary carriers it produces Boolean values directly.

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
            payload, metadata, _, marker = self._handle.read(name)
            return decode_series(
                metadata[6], metadata[7], payload, metadata[2], metadata[3], metadata[5], marker
            )

    def write_series(self, name: str, series: SeriesValue, *, overwrite: bool = False) -> None:
        """Write a root series; by default an existing name fails and is never replaced.

        ``StoredSeries`` additionally carries full-Int64 date/duration element
        codes (without scalar date packing), Int128/UInt128 or ComplexF16 bytes.
        Its element frequency is independent of the dated axis. Live carrier
        and marker validation precede snapshotting and any overwrite deletion.
        Empty wide carriers need a type marker; empty date/duration series
        retain Julia's marker but remain unloadable by the pinned Julia loader.
        Nonempty Bool-marked wide carriers preserve bytes, type and marker;
        marker failure leaves unmarked wide values. A wide Bool-marked empty
        cannot preserve its width and is refused. Other foreign markers remain
        unsupported.

        Ordinary series carry native-endian signed/unsigned integers through
        64 bits, float16/32/64, complex64/128 or Boolean values over the Monthly,
        Quarterly, HalfYearly, Yearly, Daily, BDaily or Weekly frequency (any
        fiscal anchor or week end day). The first date must lie inside the
        reliable native range of the frequency and reproduce through the
        native codec before anything is stored; year/period series also check
        their last date, and calendar series may run past the window like
        Julia's (``ValueError`` otherwise; ``TypeError`` for other dtypes,
        Unit series or non-series input).

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
        wide default. The error is reported; no implicit cleanup delete is attempted.
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
