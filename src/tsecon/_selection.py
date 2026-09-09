# SPDX-License-Identifier: MIT
"""Shared normalization of inclusive date ranges into positional slices."""

from tsecon.mit import MIT
from tsecon.mitrange import MITRange


def date_slice(firstdate: MIT, length: int, dates: MITRange) -> slice:
    """Return a bounded slice selecting exactly the dates in iteration order.

    Callers validate frequency. Empty ranges select no storage, even when their
    anchor lies outside it. Bounds concern actual selected dates, not the nominal
    stop, which need not be reached by a stride.
    """
    count = len(dates)
    if count == 0:
        return slice(0, 0)
    first = dates.start.value - firstdate.value
    last = first + (count - 1) * dates.step
    if min(first, last) < 0 or max(first, last) >= length:
        msg = f"MITRange {dates!s} is not contained in stored range starting at {firstdate!s}."
        raise IndexError(msg)
    # A negative stop of -1 would be interpreted relative to the array end.
    stop = last + 1 if dates.step > 0 else (last - 1 if last else None)
    return slice(first, stop, dates.step)
