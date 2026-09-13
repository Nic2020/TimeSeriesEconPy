# SPDX-License-Identifier: MIT
"""Shared native metadata constants; no codecs or native imports."""

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
# Series axes: the year/period families plus the calendar families. Unit series
# (code 11) are not yet supported; Julia writes them as a plain Int64 axis.
_SERIES_FREQUENCIES: dict[int, Frequency] = {**_FREQUENCIES, **_CALENDAR_FREQUENCIES}
_SCALAR_FREQUENCIES: dict[int, Frequency] = {
    **_FREQUENCIES,
    UNIT_FREQUENCY: Unit(),
    **_CALENDAR_FREQUENCIES,
}
# Native scalar type codes: type_integer (which the header also names
# type_signed) is 1, type_unsigned 2, type_date 3, type_float 4, type_complex 5
# and type_string 6. Type 1 with a supported frequency is a Duration; type 3
# always carries a frequency. Julia reloads types 1, 2, 4 and 5 by byte width
# alone (frequency zero) and stores every width without a marker.
KIND_INTEGER = 1
KIND_UNSIGNED = 2
KIND_DATE = 3
KIND_FLOAT = 4
KIND_COMPLEX = 5
KIND_STRING = 6
