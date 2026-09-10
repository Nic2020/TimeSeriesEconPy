# SPDX-License-Identifier: MIT
"""Float64 scalar and monthly/quarterly series DataEcon conversions."""

from __future__ import annotations

import struct
import sys
from typing import TypeAlias

import numpy as np

from tsecon.frequencies import Monthly, Quarterly
from tsecon.mit import MIT
from tsecon.tseries import TSeries

# Native sqlite3_bind_blob takes a C int despite daec.h accepting int64_t.
# Bound both allocations and integer conversions well below that limit.
MAX_BYTES = 128 * 1024 * 1024
MIN_DATE = -(2**31)
MAX_DATE = 2**31 - 1
MIN_QUARTERLY_DATE = -131200
_FREQUENCIES: dict[int, Monthly | Quarterly] = {
    32: Monthly(),
    65: Quarterly(1),
    66: Quarterly(2),
    67: Quarterly(3),
}
Metadata: TypeAlias = tuple[int, int, int, int, int, int, int, int, int]
ScalarMetadata: TypeAlias = tuple[int, int, int, int]


def series_frequency(code: int) -> Monthly | Quarterly:
    """Resolve only canonical supported native frequency codes."""
    try:
        return _FREQUENCIES[code]
    except KeyError:
        raise TypeError("DataEcon supports only monthly or quarterly Float64 TSeries.") from None


def validate_scalar_metadata(metadata: ScalarMetadata) -> None:
    """Validate scalar type and length before dereferencing native memory."""
    cls, kind, frequency, nbytes = metadata
    if (cls, kind, frequency) != (1, 4, 0):
        raise TypeError("DataEcon scalar support requires Float64 with no frequency.")
    if nbytes != 8:
        raise ValueError("DataEcon Float64 scalars require exactly eight payload bytes.")


def encode_scalar(value: float | np.float64) -> bytes:
    """Snapshot an exact Python float or NumPy float64 without implicit coercion."""
    if type(value) is not float and type(value) is not np.float64:
        raise TypeError("write_scalar requires a Python float or NumPy float64.")
    if sys.byteorder != "little":
        raise RuntimeError("DataEcon interchange requires a little-endian host.")
    return struct.pack("<d", value)


def decode_scalar(payload: bytes) -> float:
    """Return an independent Python float from a validated byte snapshot."""
    validate_scalar_metadata((1, 4, 0, len(payload)))
    if sys.byteorder != "little":
        raise RuntimeError("DataEcon interchange requires a little-endian host.")
    return float(struct.unpack("<d", payload)[0])


def validate_metadata(metadata: Metadata) -> None:
    """Validate native metadata before the wrapper dereferences a value pointer."""
    cls, kind, element, element_freq, axis, length, frequency, first, nbytes = metadata
    series_frequency(frequency)
    if (cls, kind, element, element_freq, axis) != (2, 12, 4, 0, 1):
        raise TypeError("DataEcon supports only monthly or quarterly Float64 TSeries.")
    if length < 0:
        raise ValueError("Invalid negative DataEcon series length.")
    if nbytes != length * 8 or not 0 <= nbytes <= MAX_BYTES:
        raise ValueError("Invalid or oversized DataEcon series payload.")
    if not MIN_DATE <= first <= MAX_DATE or (length and first + length - 1 > MAX_DATE):
        raise ValueError("DataEcon dates must fit the native signed 32-bit date range.")
    if frequency != 32 and first < MIN_QUARTERLY_DATE:
        raise ValueError("Date is outside the reliable native quarterly date range.")


def encode_series(series: TSeries) -> tuple[int, int, int, bytes]:
    """Return frequency, year, period and an independent Float64 byte snapshot."""
    if not isinstance(series, TSeries):
        raise TypeError("write_series requires a TSeries.")
    if sys.byteorder != "little":
        raise RuntimeError(
            "DataEcon interchange is currently supported on little-endian hosts only."
        )
    code = next((code for code, freq in _FREQUENCIES.items() if freq == series.frequency), None)
    if code is None or series.values.dtype != np.dtype(np.float64):
        raise TypeError(
            "DataEcon supports only monthly or quarterly native-endian float64 TSeries."
        )
    length = len(series.values)
    validate_metadata((2, 12, 4, 0, 1, length, code, series.firstdate.value, length * 8))
    year, period_index = divmod(series.firstdate.value, series_frequency(code).periods_per_year)
    return code, year, period_index + 1, series.values.tobytes(order="C")


def decode_series(code: int, year: int, period: int, payload: bytes) -> TSeries:
    """Construct an owning core series; no native storage escapes the adapter."""
    if sys.byteorder != "little":
        raise RuntimeError(
            "DataEcon interchange is currently supported on little-endian hosts only."
        )
    values = np.frombuffer(payload, dtype=np.float64).copy()
    return TSeries(MIT.from_yp(series_frequency(code), year, period), values)
