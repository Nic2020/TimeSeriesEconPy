# SPDX-License-Identifier: MIT
"""Recursive (sequential) computation over a time series range.

Mirrors ``TimeSeriesEcon.jl/src/recursive.jl``, which exports the ``@rec``
macro. Python has no parse-time macros, so the Julia surface ::

    @rec t=3U:10U s[t] = s[t-1] + s[t-2]

ports to a higher-order function call ::

    rec(MITRange(MIT(Unit(), 3), MIT(Unit(), 10)), s,
        lambda t: s[t - 1] + s[t - 2])

The semantics match: each step writes ``target[t] = fn(t)`` *before* the
next step runs, so a closure that reads ``target[t - k]`` always sees the
freshly-committed value.

For multi-target recurrences (two series updated together each step),
write the explicit ``for t in rng`` loop — the wrapper buys nothing once
there is more than one target.

Performance notes (see also [decision 01](decisions/01_acceleration_strategy.md)
and the M1.5 milestone in ``MASTER_PLAN.md``)
-----------------------------------------------------------------------
Per-iteration cost is dominated by the lambda call and the
:class:`~tsecon.tseries.TSeries` ``__getitem__`` / ``__setitem__``
dispatch (each indexing step allocates a fresh :class:`~tsecon.mit.MIT`
for ``t - k`` and walks ``_ts_values_inds``). For a single 80-120 period
recurrence (≈ 20-30 years of quarterly data) this is a few hundred
microseconds — fine. For a data pipeline that runs ``rec`` thousands of
times the cumulative cost can dominate, which is exactly the workload
M1.5 will benchmark against Julia's ``@rec`` and a specialized Cython
kernel for the linear-recurrence common case (AR(p), Fibonacci, lag
polynomials).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tsecon.mit import MIT
from tsecon.mitrange import MITRange
from tsecon.tseries import TSeries

__all__ = ["rec"]


def rec(
    rng: MITRange,
    target: TSeries,
    fn: Callable[[MIT], Any],
) -> None:
    """Compute ``target[t] = fn(t)`` for each ``t`` in ``rng``, in order.

    The Python equivalent of TimeSeriesEcon.jl's ``@rec`` macro. Each
    iteration's write is committed before the next iteration runs, so
    ``fn`` may reference ``target[t - k]`` (or any other series) freely:
    by the time step ``t`` reads ``target[t - k]``, that period has
    already been written by a prior step (assuming ``k >= 1`` and the
    recurrence is well-defined).

    The ``target`` is mutated in place. Assignment to an MIT outside
    ``target.range`` extends the storage via the TSeries auto-resize
    rule, matching the Julia ``@rec`` behaviour.

    Parameters
    ----------
    rng : MITRange
        The range to iterate over. Its frequency must match
        ``target``'s frequency.
    target : TSeries
        The series being computed. Mutated in place.
    fn : Callable[[MIT], scalar]
        Returns the value to assign at each step. Typically a lambda
        closing over ``target`` (and any other series the recurrence
        references).

    Raises
    ------
    TypeError
        If ``rng.frequency`` does not match ``target.frequency``.

    Examples
    --------
    Fibonacci over a Unit range::

        >>> from tsecon import MIT, MITRange, TSeries, rec
        >>> from tsecon.frequencies import Unit
        >>> import numpy as np
        >>> s = TSeries(MIT(Unit(), 1), np.array([1.0, 1.0]))
        >>> rec(MITRange(MIT(Unit(), 3), MIT(Unit(), 10)), s,
        ...     lambda t: s[t - 1] + s[t - 2])
        >>> [float(s[MIT(Unit(), i)]) for i in (1, 5, 10)]
        [1.0, 5.0, 55.0]

    AR(1) over a quarterly range, reading from a second series::

        rec(rng, consumption,
            lambda t: beta * consumption[t - 1] + (1 - beta) * income[t])
    """
    if rng.frequency != target.frequency:
        msg = (
            f"rec: range frequency {type(rng.frequency).__name__} does not match "
            f"target frequency {type(target.frequency).__name__}."
        )
        raise TypeError(msg)
    for t in rng:
        target[t] = fn(t)
