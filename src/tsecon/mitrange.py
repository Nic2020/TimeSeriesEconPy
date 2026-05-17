# SPDX-License-Identifier: MIT
"""MITRange — an inclusive integer range of frequency-tagged moments in time.

Mirrors Julia's ``UnitRange{MIT{F}}`` and ``StepRange{MIT{F}}``. A range is
defined by ``(start, stop, step)`` where ``start`` and ``stop`` are
:class:`~tsecon.mit.MIT` values of the same frequency and ``step`` is a
nonzero integer number of periods (or a :class:`~tsecon.mit.Duration` of the
same frequency). ``step`` may be negative — ``MITRange(10U, 1U, -1)`` walks
backward, mirroring Julia's ``10U:-1:1U``. Ranges whose endpoints do not
match the sign of ``step`` (e.g. ``MITRange(1U, 10U, -1)``) are empty.

``MITRange`` supports ``len``, bidirectional iteration, indexing (by ``int``
or ``MIT``), slicing, and membership testing.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, overload

from tsecon.frequencies import Frequency, Unit, prettyprint_frequency
from tsecon.mit import MIT, Duration

__all__ = [
    "MITRange",
    "mitrange",
    "rangeof",
    "rangeof_span",
]


@dataclass(frozen=True, slots=True)
class MITRange:
    """An inclusive range of MITs of a single frequency.

    Construct with ``MITRange(start, stop)`` for unit steps, or
    ``MITRange(start, stop, step)`` for a step range. ``step`` may be a
    nonzero ``int`` (interpreted in the range's frequency) or a
    :class:`Duration` of the same frequency. Negative ``step`` walks
    backward (``MITRange(10U, 1U, -1)`` mirrors Julia's ``10U:-1:1U``);
    ranges whose endpoints have the opposite sense to ``step`` are empty.
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
        if self.step == 0:
            raise ValueError("step must be nonzero.")

    # -- introspection -----------------------------------------------------

    @property
    def frequency(self) -> Frequency:
        """The frequency shared by ``start`` and ``stop``."""
        return self.start.frequency

    def __len__(self) -> int:
        span = self.stop.value - self.start.value
        # Sign mismatch between span and step means the range is empty
        # (e.g. start=1, stop=10, step=-1, or start=10, stop=1, step=+1).
        if (span > 0 and self.step < 0) or (span < 0 and self.step > 0):
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
        """Return the last MIT in the range. Raises if empty.

        For a forward range (``step > 0``) this is the largest MIT not past
        ``stop``; for a reversed range (``step < 0``) it is the smallest
        MIT not past ``stop``. In both cases, ``last() == start + (len-1) * step``.
        """
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
        if self.is_empty():
            return False
        start_val = self.start.value
        last_val = self.last().value
        lo = min(start_val, last_val)
        hi = max(start_val, last_val)
        if item.value < lo or item.value > hi:
            return False
        # Same modulo check works for both signs because Python's `%` returns
        # a value with the divisor's sign (e.g. -4 % -2 == 0).
        return (item.value - start_val) % self.step == 0

    # -- indexing ----------------------------------------------------------

    @overload
    def __getitem__(self, index: int) -> MIT: ...
    @overload
    def __getitem__(self, index: slice) -> MITRange: ...
    def __getitem__(self, index: int | slice) -> MIT | MITRange:
        if isinstance(index, slice):
            length = len(self)
            # Delegate index arithmetic to Python's built-in range; works for
            # both ascending and descending slice steps.
            sub = range(length)[index]
            if len(sub) == 0:
                # Pick a canonical empty representation (step=1, start..start-1)
                # so the sign of self.step doesn't accidentally produce a non-
                # empty result via step composition.
                empty_start = self.start
                empty_stop = MIT(self.frequency, self.start.value - 1)
                return MITRange(empty_start, empty_stop, 1)
            new_start = MIT(self.frequency, self.start.value + sub[0] * self.step)
            new_stop = MIT(self.frequency, self.start.value + sub[-1] * self.step)
            return MITRange(new_start, new_stop, self.step * sub.step)
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


def _range_bounds(rng: MITRange) -> tuple[int, int]:
    """Return ``(low_value, high_value)`` for ``rng``, sign-aware.

    For a forward range ``(low, high) == (start.value, last().value)``;
    for a reversed range ``(low, high) == (last().value, start.value)``.
    Caller must guard against empty ranges.
    """
    s, e = rng.start.value, rng.last().value
    return (s, e) if s <= e else (e, s)


def _apply_drop(rng: MITRange, drop: int) -> MITRange:
    """Return a copy of ``rng`` with ``drop`` elements skipped at one end.

    ``drop > 0`` skips at the start; ``drop < 0`` skips at the end; ``drop == 0``
    is a no-op. Each unit of ``drop`` corresponds to one element of the range
    (i.e., one ``step`` of the underlying frequency). If ``|drop| >= len(rng)``
    the result is empty.
    """
    if drop == 0:
        return rng
    n = len(rng)
    if abs(drop) >= n:
        empty_stop = MIT(rng.frequency, rng.start.value - 1)
        return MITRange(rng.start, empty_stop, 1)
    if drop > 0:
        new_start = MIT(rng.frequency, rng.start.value + drop * rng.step)
        return MITRange(new_start, rng.stop, rng.step)
    # drop < 0: contract from the far end. The end of the range is `last()`;
    # shifting it back by |drop| elements means subtracting |drop| * step.
    last_val = rng.last().value
    new_last = MIT(rng.frequency, last_val + drop * rng.step)
    return MITRange(rng.start, new_last, rng.step)


def rangeof(
    obj: object,
    *,
    drop: int = 0,
    method: Literal["intersect", "union"] = "intersect",
) -> MITRange:
    """Return the range of ``obj``, optionally dropping leading/trailing periods.

    Mirrors Julia's ``rangeof(t; drop=n)`` and ``rangeof(w; method=intersect)``
    (see [`TimeSeriesEcon.jl/src/workspaces.jl`](https://github.com/bankofcanada/TimeSeriesEcon.jl)
    and ``src/tseries.jl``). The Python equivalents — :attr:`TSeries.range` /
    :attr:`MVTSeries.range` properties, :meth:`Workspace.rangeof`,
    :func:`rangeof_span` — remain available; this function unifies them under
    a single kwarg-bearing call site that mirrors the Julia surface 1-to-1 and
    is the recommended target for line-by-line tutorial / model code ports.

    Parameters
    ----------
    obj
        A :class:`MITRange`, :class:`~tsecon.mit.MIT`,
        :class:`~tsecon.tseries.TSeries`, :class:`~tsecon.mvtseries.MVTSeries`,
        or :class:`~tsecon.workspace.Workspace`. Anything else raises
        :class:`TypeError` naming the offending type.
    drop
        Skip ``drop`` elements at the start (``drop > 0``) or at the end
        (``drop < 0``). ``drop == 0`` returns the full range. ``|drop| >=
        len(range)`` returns an empty range. Mirrors Julia's ``drop=`` kwarg
        exactly.
    method
        ``"intersect"`` (default) → intersection of all member ranges;
        ``"union"`` → ``rangeof_span`` semantics. Only meaningful for
        :class:`~tsecon.workspace.Workspace`; passing ``method="union"`` for
        any other type raises :class:`TypeError` to catch the likely
        wrong-object-type mistake.

    Returns
    -------
    MITRange
        The (possibly shrunk) range. ``step`` is preserved for
        :class:`MITRange` inputs; :class:`TSeries` / :class:`MVTSeries` /
        :class:`~tsecon.workspace.Workspace` inputs always yield ``step=1``.

    Raises
    ------
    TypeError
        Unsupported ``obj`` type, or ``method="union"`` passed for a
        non-Workspace input.
    ValueError
        ``method`` is not one of ``"intersect"`` / ``"union"``.

    Examples
    --------
    >>> import tsecon as ts
    >>> a = ts.TSeries(ts.qq(2020, 1), [1.0, 2.0, 3.0, 4.0])
    >>> ts.rangeof(a)
    2020Q1:2020Q4
    >>> ts.rangeof(a, drop=1)
    2020Q2:2020Q4
    >>> ts.rangeof(a, drop=-1)
    2020Q1:2020Q3

    The tutorial-1 ``@rec`` idiom ports line-by-line::

        # Julia:  @rec rangeof(a, drop=1) a[t] = (1-ρ)*a_ss + ρ*a[t-1]
        ts.rec(ts.rangeof(a, drop=1), a, lambda t: (1 - rho) * a_ss + rho * a[t - 1])

    Workspace intersection vs union:

    >>> w = ts.Workspace(a=a, b=ts.TSeries(ts.qq(2020, 3), [10.0, 20.0]))
    >>> ts.rangeof(w)
    2020Q3:2020Q4
    >>> ts.rangeof(w, method="union")
    2020Q1:2020Q4
    """
    if method not in ("intersect", "union"):
        msg = f"method must be 'intersect' or 'union', got {method!r}."
        raise ValueError(msg)

    # Deferred imports break the import cycle: mitrange is a foundational
    # module imported by tseries / mvtseries / workspace, so we cannot
    # import those at module load time.
    from tsecon.mvtseries import MVTSeries
    from tsecon.tseries import TSeries
    from tsecon.workspace import Workspace

    if isinstance(obj, Workspace):
        base = obj.rangeof(method=method)
    elif isinstance(obj, (MITRange, MIT, TSeries, MVTSeries)):
        if method != "intersect":
            msg = (
                f"method={method!r} is only meaningful for Workspace; "
                f"got {type(obj).__name__}."
            )
            raise TypeError(msg)
        if isinstance(obj, MITRange):
            base = obj
        elif isinstance(obj, MIT):
            base = MITRange(obj, obj)
        else:
            base = obj.range
    else:
        msg = (
            f"rangeof() does not support {type(obj).__name__}; expected "
            f"MITRange, MIT, TSeries, MVTSeries, or Workspace."
        )
        raise TypeError(msg)

    return _apply_drop(base, drop)


def rangeof_span(*args: object) -> MITRange:
    """Return the smallest forward ``MITRange`` covering all argument ranges.

    All arguments must share a single frequency. Raises ``TypeError`` if
    frequencies are mixed. The returned span is always forward-stepped
    (``step=1``) regardless of the direction of the inputs.
    """
    chosen_lo: int | None = None
    chosen_hi: int | None = None
    chosen_freq: Frequency | None = None
    for arg in args:
        sub = _to_unitrange(arg)
        if sub is None or sub.is_empty():
            continue
        if chosen_freq is None:
            chosen_freq = sub.frequency
        elif sub.frequency != chosen_freq:
            raise TypeError(
                f"Mixing frequencies not allowed: {prettyprint_frequency(chosen_freq)} "
                f"and {prettyprint_frequency(sub.frequency)}."
            )
        sub_lo, sub_hi = _range_bounds(sub)
        chosen_lo = sub_lo if chosen_lo is None else min(chosen_lo, sub_lo)
        chosen_hi = sub_hi if chosen_hi is None else max(chosen_hi, sub_hi)
    if chosen_freq is None or chosen_lo is None or chosen_hi is None:
        # Match Julia's behavior of returning an empty `1U:0U` for no args.
        return MITRange(MIT(Unit(), 1), MIT(Unit(), 0))
    return MITRange(MIT(chosen_freq, chosen_lo), MIT(chosen_freq, chosen_hi))
