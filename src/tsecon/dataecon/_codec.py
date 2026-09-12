# SPDX-License-Identifier: MIT
"""Scalar and series conversions between core objects and native DataEcon payloads.

Scalars: Float64, Int64, UTF-8 strings, and MIT dates or Durations over the
unit, daily, business-daily, weekly (every end day), monthly, quarterly,
half-yearly and annual frequencies. Series: Float64 values over the monthly,
quarterly, half-yearly and annual frequencies.
"""

from __future__ import annotations

import struct
import sys
from typing import TypeAlias

import numpy as np

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

# Native sqlite3_bind_blob takes a C int despite daec.h accepting int64_t.
# Bound both allocations and integer conversions well below that limit.
MAX_BYTES = 128 * 1024 * 1024
MIN_DATE = -(2**31)
MAX_DATE = 2**31 - 1
MIN_MONTHLY_DATE = -393600
MIN_QUARTERLY_DATE = -131200
MIN_HALFYEARLY_DATE = -65600
_FREQUENCIES: dict[int, Monthly | Quarterly | HalfYearly | Yearly] = {
    32: Monthly(),
    65: Quarterly(1),
    66: Quarterly(2),
    67: Quarterly(3),
    **{128 + month: HalfYearly(month) for month in range(1, 7)},
    **{256 + month: Yearly(month) for month in range(1, 13)},
}
# The native decoder adds EPOCH_L * periods_per_year in uint32 arithmetic and
# then divides; below these codes the wrapped sum decodes to a different year.
# Annual (division by one) preserves the whole signed 32-bit range. Series
# writes already reject monthly codes below -393600 through the native
# round-trip check; scalar dates apply the explicit table before any C call.
_MIN_DATES: dict[int, tuple[int, str]] = {
    **dict.fromkeys((65, 66, 67), (MIN_QUARTERLY_DATE, "quarterly")),
    **dict.fromkeys(range(129, 135), (MIN_HALFYEARLY_DATE, "half-yearly")),
}
_SCALAR_MIN_DATES: dict[int, int] = {
    32: MIN_MONTHLY_DATE,
    **{code: minimum for code, (minimum, _) in _MIN_DATES.items()},
    **dict.fromkeys(range(257, 269), MIN_DATE),
}
MIN_INT64 = -(2**63)
MAX_INT64 = 2**63 - 1
# Scalar-only frequencies. Unit (11) is Julia's generic pass-through: the code
# is a plain signed 64-bit integer and no native codec applies. Daily (12),
# business daily (13) and weekly (16 + ISO end day, 17..23) use the native
# calendar codec, whose encoder accepts years in [-32800, 32800] and whose
# decoders shift by fixed uint32 constants. The exact round-trip windows below
# were verified natively on every code: daily from 1 March -32800, business
# daily and weekly from the last week of December -32800, all through 31
# December 32800. Code 16 (a second Sunday alias) and 24..31 are never written
# by Julia and decode to invalid anchors, so they are rejected like bare codes.
UNIT_FREQUENCY = 11
_CALENDAR_FREQUENCIES: dict[int, Daily | BDaily | Weekly] = {
    12: Daily(),
    13: BDaily(),
    **{16 + day: Weekly(day) for day in range(1, 8)},
}
_CALENDAR_RANGES: dict[int, tuple[int, int]] = {
    12: (-11980259, 11979954),
    13: (-8557114, 8557110),
    **dict.fromkeys(range(17, 24), (-1711422, 1711422)),
}
_SCALAR_FREQUENCIES: dict[int, Frequency] = {
    **_FREQUENCIES,
    UNIT_FREQUENCY: Unit(),
    **_CALENDAR_FREQUENCIES,
}
# Native scalar type codes: type_integer (which the header also names
# type_signed) is 1, type_date 3, type_float 4 and type_string 6. Type 1 with a
# supported frequency is a Duration; type 3 always carries a frequency.
KIND_INTEGER = 1
KIND_DATE = 3
KIND_FLOAT = 4
KIND_STRING = 6
Metadata: TypeAlias = tuple[int, int, int, int, int, int, int, int, int]
ScalarMetadata: TypeAlias = tuple[int, int, int, int]
ScalarValue: TypeAlias = float | np.float64 | int | np.int64 | str | MIT | Duration
ScalarResult: TypeAlias = float | int | str | MIT | Duration
_SCALAR_SUPPORT = (
    "DataEcon scalar support covers Float64, Int64, strings, and MIT dates or Durations "
    "over the Unit, Daily, BDaily, Weekly, Monthly, Quarterly, HalfYearly and Yearly "
    "frequencies."
)


def series_frequency(code: int) -> Monthly | Quarterly | HalfYearly | Yearly:
    """Resolve only canonical supported native frequency codes."""
    try:
        return _FREQUENCIES[code]
    except KeyError:
        raise TypeError(
            "DataEcon supports only monthly, quarterly, half-yearly or annual Float64 TSeries."
        ) from None


def scalar_frequency(code: int) -> Frequency:
    """Resolve a supported native frequency code for a date or duration scalar."""
    try:
        return _SCALAR_FREQUENCIES[code]
    except KeyError:
        raise TypeError(_SCALAR_SUPPORT) from None


def scalar_frequency_code(frequency: object) -> int:
    """Return the canonical native code for a supported core frequency object."""
    code = next((code for code, freq in _SCALAR_FREQUENCIES.items() if freq == frequency), None)
    if code is None:
        raise TypeError(_SCALAR_SUPPORT)
    return code


def validate_date_code(frequency: int, code: int) -> None:
    """Require a date code the native codec decodes exactly for this frequency.

    Unit codes are plain signed 64-bit integers with no date bound. Calendar and
    year/period codes must lie inside their verified native windows; the backend
    then confirms the native round trip for those frequencies.
    """
    scalar_frequency(frequency)
    if frequency == UNIT_FREQUENCY:
        if not MIN_INT64 <= code <= MAX_INT64:
            raise ValueError("Unit date codes must fit the signed 64-bit range.")
        return
    if frequency in _CALENDAR_RANGES:
        minimum, maximum = _CALENDAR_RANGES[frequency]
    else:
        minimum, maximum = _SCALAR_MIN_DATES[frequency], MAX_DATE
    if not minimum <= code <= maximum:
        raise ValueError("Date is outside the reliable native date range for its frequency.")


def validate_scalar_metadata(metadata: ScalarMetadata) -> None:
    """Validate scalar class, type, frequency and length before dereferencing native memory."""
    cls, kind, frequency, nbytes = metadata
    if cls != 1:
        raise TypeError(_SCALAR_SUPPORT)
    if kind == KIND_STRING:
        if frequency != 0:
            raise TypeError("DataEcon string scalars carry no frequency.")
        if not 1 <= nbytes <= MAX_BYTES:
            raise ValueError(
                "DataEcon string scalars require a NUL-terminated payload within the size limit."
            )
        return
    if kind == KIND_FLOAT:
        if frequency != 0:
            raise TypeError("DataEcon Float64 scalars carry no frequency.")
    elif kind == KIND_INTEGER:
        if frequency != 0:
            scalar_frequency(frequency)
    elif kind == KIND_DATE:
        scalar_frequency(frequency)
    else:
        raise TypeError(_SCALAR_SUPPORT)
    if nbytes != 8:
        raise ValueError("DataEcon numeric, date and duration scalars require exactly eight bytes.")


def _encode_text(value: str) -> bytes:
    if "\0" in value:
        raise ValueError(
            "DataEcon strings cannot contain NUL; the native format is NUL-terminated."
        )
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(
            "DataEcon strings must be encodable as UTF-8 (no lone surrogates)."
        ) from None
    if len(encoded) >= MAX_BYTES:
        raise ValueError("DataEcon string exceeds the payload size limit.")
    return encoded + b"\0"


def encode_scalar(value: ScalarValue) -> tuple[int, int, bytes]:
    """Return the native type code, frequency code and an owned payload snapshot.

    Only exact Python float/NumPy float64, Python int/NumPy int64, Python str,
    core MIT and core Duration values are accepted. Integers and durations never
    pass through floating point; strings are UTF-8 plus a NUL terminator; dates
    must lie inside the reliable native range of their frequency (Unit dates are
    unbounded 64-bit codes). Subclasses, other widths, bytes, bool and every
    other type are rejected without conversion.
    """
    if type(value) is float or type(value) is np.float64:
        kind, frequency, packed = KIND_FLOAT, 0, struct.pack("<d", value)
    elif type(value) is int or type(value) is np.int64:
        number = int(value)
        if not MIN_INT64 <= number <= MAX_INT64:
            raise ValueError("write_scalar integers must fit the signed 64-bit range.")
        kind, frequency, packed = KIND_INTEGER, 0, struct.pack("<q", number)
    elif type(value) is str:
        kind, frequency, packed = KIND_STRING, 0, _encode_text(value)
    elif type(value) is MIT:
        frequency = scalar_frequency_code(value.frequency)
        validate_date_code(frequency, value.value)
        kind, packed = KIND_DATE, struct.pack("<q", value.value)
    elif type(value) is Duration:
        frequency = scalar_frequency_code(value.frequency)
        if not MIN_INT64 <= value.value <= MAX_INT64:
            raise ValueError("write_scalar durations must fit the signed 64-bit range.")
        kind, packed = KIND_INTEGER, struct.pack("<q", value.value)
    else:
        raise TypeError(
            "write_scalar requires a Python float, NumPy float64, Python int, NumPy int64, "
            "str, MIT or Duration."
        )
    if sys.byteorder != "little":
        raise RuntimeError("DataEcon interchange requires a little-endian host.")
    return kind, frequency, packed


def decode_scalar(kind: int, frequency: int, payload: bytes) -> ScalarResult:
    """Return an independent core value from validated metadata and a byte snapshot."""
    validate_scalar_metadata((1, kind, frequency, len(payload)))
    if sys.byteorder != "little":
        raise RuntimeError("DataEcon interchange requires a little-endian host.")
    if kind == KIND_STRING:
        if payload[-1] != 0 or b"\0" in payload[:-1]:
            raise ValueError("DataEcon string payload is not a single NUL-terminated string.")
        try:
            return payload[:-1].decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("DataEcon string payload is not valid UTF-8.") from None
    if kind == KIND_FLOAT:
        return float(struct.unpack("<d", payload)[0])
    number = int(struct.unpack("<q", payload)[0])
    if frequency == 0:
        return number
    if kind == KIND_DATE:
        validate_date_code(frequency, number)
        return MIT(scalar_frequency(frequency), number)
    return Duration(scalar_frequency(frequency), number)


def validate_metadata(metadata: Metadata) -> None:
    """Validate native metadata before the wrapper dereferences a value pointer."""
    cls, kind, element, element_freq, axis, length, frequency, first, nbytes = metadata
    series_frequency(frequency)
    if (cls, kind, element, element_freq, axis) != (2, 12, 4, 0, 1):
        raise TypeError(
            "DataEcon supports only monthly, quarterly, half-yearly or annual Float64 TSeries."
        )
    if length < 0:
        raise ValueError("Invalid negative DataEcon series length.")
    if nbytes != length * 8 or not 0 <= nbytes <= MAX_BYTES:
        raise ValueError("Invalid or oversized DataEcon series payload.")
    if not MIN_DATE <= first <= MAX_DATE or (length and first + length - 1 > MAX_DATE):
        raise ValueError("DataEcon dates must fit the native signed 32-bit date range.")
    if frequency in _MIN_DATES and first < _MIN_DATES[frequency][0]:
        raise ValueError(
            f"Date is outside the reliable native {_MIN_DATES[frequency][1]} date range."
        )


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
            "DataEcon supports only monthly, quarterly, half-yearly or annual "
            "native-endian float64 TSeries."
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
