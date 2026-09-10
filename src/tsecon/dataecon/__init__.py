# SPDX-License-Identifier: MIT
"""Read and write monthly Float64 series in DataEcon files, including empty series.

Use ``open_dataecon`` as a context manager. The native extension loads on first
use; importing the core package does not require it. Other data types,
catalogs and general attributes are not supported yet.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import TYPE_CHECKING, Literal

from tsecon.tseries import TSeries

from ._codec import decode_series, encode_series
from ._errors import DataEconError

if TYPE_CHECKING:
    from . import _native

__all__ = ["DataEconError", "DataEconFile", "open_dataecon"]

_loader_lock = RLock()
_dll_directories: list[object] = []


def _open_native(path: str, readonly: bool) -> _native.FileHandle:
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
        return backend.FileHandle(path, readonly)  # type: ignore[no-any-return]


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

    def __init__(self, path: str | os.PathLike[str], mode: Literal["r", "a"] = "r") -> None:
        if mode not in ("r", "a"):
            raise ValueError("DataEcon mode must be 'r' (read-only) or 'a' (create/append).")
        self.path = os.fspath(path)
        if not isinstance(self.path, str):
            raise TypeError("DataEcon paths must be strings or string-valued PathLike objects.")
        if not self.path or "\0" in self.path:
            raise ValueError("DataEcon paths must be nonempty and contain no NUL.")
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

    @property
    def closed(self) -> bool:
        """Whether this owner is no longer usable, including a failed close."""
        return self._state != "open"

    def _require_open(self) -> None:
        if self.closed:
            raise ValueError(f"DataEcon file is {self._state}.")

    def read_series(self, name: str) -> TSeries:
        """Read one root series into owning storage that survives file closure."""
        with self._lock:
            self._require_open()
            _validate_name(name)
            year, month, payload, _, _ = self._handle.read(name)
            return decode_series(year, month, payload)

    def write_series(self, name: str, series: TSeries) -> None:
        """Append a root series; existing names fail and are never overwritten.

        A native write failure may leave a partial object or unused axis. No
        transaction/rollback or automatic file deletion is promised.
        """
        with self._lock:
            self._require_open()
            _validate_name(name)
            year, month, payload = encode_series(series)
            if self._mode == "r":
                raise ValueError("Cannot write through a read-only DataEcon file.")
            self._handle.write(name, year, month, payload)

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


def open_dataecon(path: str | os.PathLike[str], mode: Literal["r", "a"] = "r") -> DataEconFile:
    """Open a file, defaulting to read-only; ``a`` creates or appends without truncation."""
    return DataEconFile(path, mode)
