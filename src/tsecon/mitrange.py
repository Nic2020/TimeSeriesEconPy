# SPDX-License-Identifier: MIT
"""MITRange — an inclusive integer range of frequency-tagged moments in time.

Mirrors Julia's ``UnitRange{MIT{F}}`` and ``StepRange{MIT{F}}``. A range is
defined by ``(start, stop, step)`` where ``start`` and ``stop`` are
:class:`~tsecon.mit.MIT` values of the same frequency and ``step`` is a
positive integer number of periods (or a :class:`~tsecon.mit.Duration` of the
same frequency). Ranges with ``start > stop`` are empty.

``MITRange`` supports ``len``, iteration, indexing (by ``int`` or ``MIT``),
slicing, and membership testing.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import overload

from tsecon.frequencies import Frequency, Unit, prettyprint_frequency
from tsecon.mit import MIT, Duration

__all__ = [
    "MITRange",
    "mitrange",
    "rangeof_span",
]


@dataclass(frozen=True, slots=True)
class MITRange:
    """An inclusive range of MITs of a single frequency.

    Construct with ``MITRange(start, stop)`` for unit steps, or
    ``MITRange(start, stop, step)`` for a step range. ``step`` may be a
    positive ``int`` (interpreted in the range's frequency) or a
    :class:`Duration` of the same frequency.
    """

    start: MIT
    stop: MIT
    step: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.start, MIT) or not isinstance(self.stop, MIT):
            raise TypeError("MITRange endpoints must be MIT instances.")
        if self.start.frequency != self.stop.frequency:
            raise TypeError(
                f"Cannot construct MITRange across frequencies: "
                f"{prettyprint_frequency(self.start.frequency)} and "
                f"{prettyprint_frequency(self.stop.frequency)}."
            )
        if not isinstance(self.step, int) or isinstance(self.step, bool):
            raise TypeError(f"step must be int, got {type(self.step).__name__}")
        if self.step <= 0:
            raise ValueError(f"step must be a positive integer, got {self.step}")

    # -- introspection -----------------------------------------------------

    @property
    def frequency(self) -> Frequency:
        """The frequency shared by ``start`` and ``stop``."""
        return self.start.frequency

    def __len__(self) -> int:
        span = self.stop.value - self.start.value
        if span < 0:
            return 0
        return span // self.step + 1

    def __bool__(self) -> bool:
        return len(self) > 0

    def is_empty(self) -> bool:
        """Return True if the range contains no MITs."""
        return len(self) == 0

    def first(self) -> MIT:
        """Return the first MIT in the range. Raises if empty."""
        if not self:
            raise IndexError("MITRange is empty.")
        return self.start

    def last(self) -> MIT:
        """Return the last MIT in the range (largest value <= stop). Raises if empty."""
        if not self:
            raise IndexError("MITRange is empty.")
        n = len(self) - 1
        return MIT(self.frequency, self.start.value + n * self.step)

    # -- iteration ---------------------------------------------------------

    def __iter__(self) -> Iterator[MIT]:
        for i in range(len(self)):
            yield MIT(self.frequency, self.start.value + i * self.step)

    def __contains__(self, item: object) -> bool:
        if not isinstance(item, MIT):
            return False
        if item.frequency != self.frequency:
            return False
        offset = item.value - self.start.value
        if offset < 0 or item.value > self.last().value:
            return False
        return offset % self.step == 0

    # -- indexing ----------------------------------------------------------

    @overload
    def __getitem__(self, index: int) -> MIT: ...
    @overload
    def __getitem__(self, index: slice) -> MITRange: ...
    def __getitem__(self, index: int | slice) -> MIT | MITRange:
        if isinstance(index, slice):
            length = len(self)
            start_i, stop_i, step_i = index.indices(length)
            if start_i >= stop_i:
                empty_start = self.start
                empty_stop = MIT(self.frequency, self.start.value - 1)
                return MITRange(empty_start, empty_stop, self.step * max(1, step_i))
            new_start = MIT(self.frequency, self.start.value + start_i * self.step)
            new_last = MIT(
                self.frequency,
                self.start.value + (stop_i - 1) * self.step,
            )
            return MITRange(new_start, new_last, self.step * step_i)
        if isinstance(index, bool):
            raise TypeError("MITRange indices must be int or slice, not bool.")
        if not isinstance(index, int):
            raise TypeError(f"MITRange indices must be int or slice, got {type(index).__name__}")
        n = len(self)
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError(f"MITRange index {index} out of range (len={n}).")
        return MIT(self.frequency, self.start.value + index * self.step)

    # -- equality / hash ---------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MITRange):
            return NotImplemented
        if self.is_empty() and other.is_empty():
            return self.frequency == other.frequency
        return self.start == other.start and self.last() == other.last() and self.step == other.step

    def __hash__(self) -> int:
        if self.is_empty():
            return hash(("MITRange-empty", self.frequency))
        return hash(("MITRange", self.start, self.last(), self.step))

    # -- repr / str --------------------------------------------------------

    def __repr__(self) -> str:
        if self.step == 1:
            return f"{self.start}:{self.stop}"
        return f"{self.start}:{self.step}:{self.stop}"

    def __str__(self) -> str:
        return repr(self)


def mitrange(start: MIT, stop: MIT, step: int | Duration = 1) -> MITRange:
    """Functional constructor for :class:`MITRange`.

    Mirrors Julia's ``start:stop`` and ``start:step:stop`` colon syntax.
    ``step`` may be an ``int`` or a :class:`Duration` of the same frequency.
    """
    if isinstance(step, Duration):
        if step.frequency != start.frequency:
            raise TypeError(
                f"step Duration frequency {prettyprint_frequency(step.frequency)} "
                f"does not match range frequency {prettyprint_frequency(start.frequency)}."
            )
        step_int = step.value
    else:
        step_int = int(step)
    return MITRange(start, stop, step_int)


def _to_unitrange(x: object) -> MITRange | None:
    """Return an MITRange view of ``x`` or None if not range-compatible."""
    if isinstance(x, MITRange):
        return x
    if isinstance(x, MIT):
        return MITRange(x, x)
    if hasattr(x, "frequency") and hasattr(x, "start") and hasattr(x, "stop"):
        # Duck-type fallback for TSeries-like objects added in later milestones.
        return MITRange(x.start, x.stop)
    return None


def rangeof_span(*args: object) -> MITRange:
    """Return the smallest ``MITRange`` covering all argument ranges.

    All arguments must share a single frequency. Raises ``TypeError`` if
    frequencies are mixed.
    """
    chosen: MITRange | None = None
    for arg in args:
        sub = _to_unitrange(arg)
        if sub is None or sub.is_empty():
            continue
        if chosen is None:
            chosen = sub
            continue
        if sub.frequency != chosen.frequency:
            raise TypeError(
                f"Mixing frequencies not allowed: {prettyprint_frequency(chosen.frequency)} "
                f"and {prettyprint_frequency(sub.frequency)}."
            )
        lo = MIT(chosen.frequency, min(chosen.start.value, sub.start.value))
        hi = MIT(chosen.frequency, max(chosen.stop.value, sub.stop.value))
        chosen = MITRange(lo, hi)
    if chosen is None:
        # Match Julia's behavior of returning an empty `1U:0U` for no args.
        return MITRange(MIT(Unit(), 1), MIT(Unit(), 0))
    return chosen
