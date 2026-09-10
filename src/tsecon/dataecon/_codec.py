# SPDX-License-Identifier: MIT
"""The deliberately narrow monthly Float64 DataEcon conversion contract."""

from __future__ import annotations

import sys
from typing import TypeAlias

import numpy as np

from tsecon.frequencies import Monthly
from tsecon.mit import mm
from tsecon.tseries import TSeries

# Native sqlite3_bind_blob takes a C int despite daec.h accepting int64_t.
# Bound both allocations and integer conversions well below that limit.
MAX_BYTES = 128 * 1024 * 1024
MIN_DATE = -(2**31)
MAX_DATE = 2**31 - 1
Metadata: TypeAlias = tuple[int, int, int, int, int, int, int, int, int]


def validate_metadata(metadata: Metadata) -> None:
    """Validate native metadata before the wrapper dereferences a value pointer."""
    cls, kind, element, element_freq, axis, length, frequency, first, nbytes = metadata
    if (cls, kind, element, element_freq, axis, frequency) != (2, 12, 4, 0, 1, 32):
        raise TypeError("DataEcon supports only monthly Float64 TSeries.")
    if length < 0:
        raise ValueError("Invalid negative DataEcon series length.")
    if nbytes != length * 8 or not 0 <= nbytes <= MAX_BYTES:
        raise ValueError("Invalid or oversized DataEcon series payload.")
    if not MIN_DATE <= first <= MAX_DATE or (length and first + length - 1 > MAX_DATE):
        raise ValueError("DataEcon dates must fit the native signed 32-bit date range.")


def encode_series(series: TSeries) -> tuple[int, int, bytes]:
    """Return year, month and an independent contiguous Float64 byte snapshot."""
    if not isinstance(series, TSeries):
        raise TypeError("write_series requires a TSeries.")
    if sys.byteorder != "little":
        raise RuntimeError(
            "DataEcon interchange is currently supported on little-endian hosts only."
        )
    if series.frequency != Monthly() or series.values.dtype != np.dtype(np.float64):
        raise TypeError("DataEcon supports only monthly native-endian float64 TSeries.")
    length = len(series.values)
    validate_metadata((2, 12, 4, 0, 1, length, 32, series.firstdate.value, length * 8))
    year, month_index = divmod(series.firstdate.value, 12)
    return year, month_index + 1, series.values.tobytes(order="C")


def decode_series(year: int, month: int, payload: bytes) -> TSeries:
    """Construct an owning core series; no native storage escapes the adapter."""
    if sys.byteorder != "little":
        raise RuntimeError(
            "DataEcon interchange is currently supported on little-endian hosts only."
        )
    values = np.frombuffer(payload, dtype=np.float64).copy()
    return TSeries(mm(year, month), values)
