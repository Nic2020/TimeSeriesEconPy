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
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from tsecon._rec_kernels import rec_linear_numpy
from tsecon.mit import MIT
from tsecon.mitrange import MITRange
from tsecon.tseries import TSeries

# Try to load the optional Cython-compiled accelerator. When the wheel was
# built without the C toolchain (or for editable installs that skipped the
# build hook), the import fails silently and rec_linear falls back to the
# NumPy reference. The public surface is otherwise unchanged.
try:
    from tsecon._rec_kernels_cy import rec_linear_cython  # type: ignore[import-not-found]

    _CYTHON_AVAILABLE: Final[bool] = True
except ImportError:
    _CYTHON_AVAILABLE = False  # type: ignore[misc]

__all__ = ["rec", "rec_linear", "rec_linear_is_cython"]


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


def rec_linear_is_cython() -> bool:
    """Return True iff the Cython-compiled rec_linear kernel was importable.

    Useful for tests, benchmarks, and diagnostic prints — the public
    :func:`rec_linear` itself is implementation-agnostic. When this
    returns ``False`` the same calls go through the pure-NumPy kernel
    in ``_rec_kernels.py``; behaviour is identical, only speed differs.
    """
    return _CYTHON_AVAILABLE


def rec_linear(
    target: TSeries,
    coeffs: npt.ArrayLike,
    lags: npt.ArrayLike,
    rng: MITRange,
) -> None:
    """Compute a linear-combination recurrence over ``rng`` in place.

    Specialised closed-form sibling of :func:`rec` for the common case
    where each step's value is a fixed linear combination of *earlier*
    values of the *same* series. Covers Fibonacci, AR(p), arbitrary lag
    polynomials — the recurrences that account for ~80% of pipeline
    workloads per the M1 benchmark (see
    ``claude_files/MASTER_PLAN.md`` § M1.5).

    The body of the loop is::

        for i, t in enumerate(rng):
            target[t] = sum(coeffs[k] * target[t - lags[k]] for k in range(len(coeffs)))

    so each ``lags[k]`` must be ``>= 1`` and ``target`` must already
    contain valid values for every ``rng.first - lags[k]`` position
    (the *initial conditions*). The first iteration writes
    ``target[rng.first]``; ``target`` is resized in place to cover
    ``rng`` if needed, filling new positions with NaN before the
    recurrence runs.

    Dispatch
    --------
    When the Cython extension ``tsecon._rec_kernels_cy`` is importable
    (the typical wheel install), this function delegates to the
    compiled :func:`rec_linear_cython` kernel — same contract,
    much faster. Without the extension, the call goes through
    :func:`rec_linear_numpy`, the pure-Python reference. Use
    :func:`rec_linear_is_cython` to check which path is active.

    Parameters
    ----------
    target : TSeries
        The series being computed. Must have ``float64`` dtype. Mutated
        in place. Initial conditions for every lagged read must already
        be present; positions inside ``rng`` are (re)written.
    coeffs : array-like of float
        Recurrence weights. Will be coerced to a 1-D ``float64`` array.
    lags : array-like of int
        Positive lag offsets, one per coefficient. Must satisfy
        ``min(lags) >= 1``. Coerced to a 1-D ``int64`` array.
    rng : MITRange
        The range to compute, frequency-matched to ``target``.

    Raises
    ------
    TypeError
        If ``rng.frequency`` does not match ``target.frequency``, or if
        ``target.dtype`` is not ``float64``.
    ValueError
        If ``len(coeffs) != len(lags)``, if any lag is ``< 1``, or if
        ``target`` does not contain the initial-condition positions
        ``rng.first - lags[k]`` for every ``k``.

    Examples
    --------
    Fibonacci over a Unit range::

        >>> import numpy as np
        >>> from tsecon import MIT, MITRange, TSeries, rec_linear
        >>> from tsecon.frequencies import Unit
        >>> s = TSeries(MIT(Unit(), 1), np.array([1.0, 1.0, 0.0, 0.0, 0.0]))
        >>> rec_linear(s, [1.0, 1.0], [1, 2],
        ...            MITRange(MIT(Unit(), 3), MIT(Unit(), 5)))
        >>> s.values.tolist()
        [1.0, 1.0, 2.0, 3.0, 5.0]

    AR(2) recurrence over quarterly data::

        rec_linear(consumption, [0.5, 0.3], [1, 2], rng)

    See Also
    --------
    rec : General higher-order form ``target[t] = fn(t)``. Use it when
        the recurrence is nonlinear or reads from series other than
        ``target``.
    """
    if rng.frequency != target.frequency:
        msg = (
            f"rec_linear: range frequency {type(rng.frequency).__name__} does not "
            f"match target frequency {type(target.frequency).__name__}."
        )
        raise TypeError(msg)
    if rng.step != 1:
        msg = f"rec_linear: range must have step=1, got step={rng.step}."
        raise ValueError(msg)
    coeffs_arr = np.asarray(coeffs, dtype=np.float64)
    lags_arr = np.asarray(lags, dtype=np.int64)
    if coeffs_arr.ndim != 1 or lags_arr.ndim != 1:
        msg = "rec_linear: coeffs and lags must be 1-D."
        raise ValueError(msg)
    if coeffs_arr.shape[0] != lags_arr.shape[0]:
        msg = (
            f"rec_linear: coeffs (n={coeffs_arr.shape[0]}) and lags "
            f"(n={lags_arr.shape[0]}) must have the same length."
        )
        raise ValueError(msg)
    if coeffs_arr.shape[0] == 0:
        msg = "rec_linear: coeffs/lags must be non-empty."
        raise ValueError(msg)
    if int(lags_arr.min()) < 1:
        msg = f"rec_linear: all lags must be >= 1, got min lag {int(lags_arr.min())}."
        raise ValueError(msg)
    if target.dtype != np.float64:
        msg = (
            f"rec_linear: target.dtype must be float64 to feed the kernel, "
            f"got {target.dtype}. Construct with TSeries(..., dtype=np.float64) or "
            f"call .astype(np.float64) first."
        )
        raise TypeError(msg)
    if len(rng) == 0:
        return

    rng_first = rng.start
    max_lag = int(lags_arr.max())
    earliest_read = rng_first.value - max_lag
    if earliest_read < target.firstdate.value:
        earliest_mit = MIT(target.frequency, earliest_read)
        msg = (
            f"rec_linear: initial conditions missing — recurrence reads "
            f"target[{earliest_mit!s}] but target starts at {target.firstdate!s}."
        )
        raise ValueError(msg)

    # Ensure target covers rng (auto-extend mirrors `rec`'s setitem behaviour).
    target._ensure_covers(rng)
    values = target._values
    offset = rng_first.value - target.firstdate.value
    count = len(rng)

    if _CYTHON_AVAILABLE:
        rec_linear_cython(values, offset, count, coeffs_arr, lags_arr)
    else:
        rec_linear_numpy(values, offset, count, coeffs_arr, lags_arr)
