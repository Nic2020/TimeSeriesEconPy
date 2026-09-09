# SPDX-License-Identifier: MIT
"""Exceptions from the DataEcon native library."""

from __future__ import annotations


class DataEconError(RuntimeError):
    """A native DataEcon failure, preserving its code and operation context.

    Positive codes originate in SQLite; negative codes originate in DataEcon.
    A failed write may leave a partially created object or an unused axis.
    """

    def __init__(
        self, code: int, operation: str, path: str, message: str, name: str | None = None
    ) -> None:
        self.code = code
        self.operation = operation
        self.path = path
        self.name = name
        self.native_message = message
        location = f"{path!r}" if name is None else f"{path!r}, object {name!r}"
        super().__init__(f"DataEcon {operation} failed for {location} (code {code}): {message}")
