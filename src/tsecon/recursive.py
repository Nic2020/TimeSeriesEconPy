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

Performance notes (see also ``claude_files/decisions/01_acceleration_strategy.md``
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

from tsecon._kernel_dispatch import _dispatch_kernel
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
    # Stub matches the cython name so `_dispatch_kernel` resolves cleanly
    # when the extension isn't built (see `_kernel_dispatch.py` docstring).
    rec_linear_cython = rec_linear_numpy

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

    Backcasting over a reversed range (Julia's
    ``@rec t=10U:-1:1U s[t] = s[t+1] - g``)::

        >>> s = TSeries(MIT(Unit(), 1), np.zeros(10))
        >>> s[MIT(Unit(), 10)] = 20.0
        >>> rec(MITRange(MIT(Unit(), 9), MIT(Unit(), 1), step=-1), s,
        ...     lambda t: s[t + 1] - 2.0)
        >>> s.values.tolist()
        [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
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
    where each step's value is a fixed linear combination of *earlier-
    written* values of the *same* series. Covers Fibonacci, AR(p),
    arbitrary lag polynomials — the recurrences that account for ~80%
    of pipeline workloads per the M1 benchmark (see
    ``claude_files/MASTER_PLAN.md`` § M1.5).

    The body of the loop is::

        for t in rng:
            target[t] = sum(coeffs[k] * target[t - lags[k]] for k in range(len(coeffs)))

    Both forward (``rng.step == +1``) and backward (``rng.step == -1``)
    iteration are supported. The contract on ``lags`` matches the
    direction so that every read references an already-written or
    initial-condition position:

    * **Forward** (``rng.step == +1``, walking from ``rng.first`` up to
      ``rng.last``): all ``lags[k] >= 1``; ``target`` must contain
      valid values for every ``rng.first - lags[k]`` position before
      the call (the *initial conditions* at the start).
    * **Backward** (``rng.step == -1``, walking from ``rng.first`` down
      to ``rng.last``): all ``lags[k] <= -1``; ``target`` must contain
      valid values for every ``rng.first - lags[k]`` position
      (= ``rng.first + |lags[k]|``, the initial conditions at the end —
      this is *backcasting*).

    ``target`` is resized in place to cover ``rng`` if needed (NaN
    padding for new positions inside ``rng`` is overwritten by the
    recurrence; positions outside ``rng`` keep whatever they had,
    which is where the initial conditions live).

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
        Nonzero lag offsets, one per coefficient. Sign must match
        ``rng.step``: all ``>= 1`` for a forward range, all ``<= -1``
        for a backward range. Coerced to a 1-D ``int64`` array.
    rng : MITRange
        The range to compute, frequency-matched to ``target``. ``step``
        must be ``+1`` (forward) or ``-1`` (backward).

    Raises
    ------
    TypeError
        If ``rng.frequency`` does not match ``target.frequency``, or if
        ``target.dtype`` is not ``float64``.
    ValueError
        If ``rng.step`` is not ``±1``, if ``len(coeffs) != len(lags)``,
        if any ``lags[k]`` has the wrong sign for ``rng.step``, or if
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

    Backcasting via a reversed range (Julia's
    ``@rec t=10U:-1:1U s[t] = s[t+1] - g`` with ``g = 0`` constant
    drift becomes ``s[t] = s[t+1]``)::

        rec_linear(s, [1.0], [-1],
                   MITRange(MIT(Unit(), 10), MIT(Unit(), 1), step=-1))

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
    if rng.step not in (1, -1):
        msg = (
            f"rec_linear: rng.step must be +1 (forward) or -1 (backward), "
            f"got step={rng.step}."
        )
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
    step = int(rng.step)
    if step > 0:
        # Forward iteration reads earlier positions; lags must be >= 1.
        min_lag = int(lags_arr.min())
        if min_lag < 1:
            msg = (
                f"rec_linear: with a forward range (step=+1), all lags must be "
                f">= 1; got min lag {min_lag}. For backward recurrences use a "
                f"reversed range (step=-1) and negative lags."
            )
            raise ValueError(msg)
    else:
        # Backward iteration reads later positions; lags must be <= -1.
        max_lag = int(lags_arr.max())
        if max_lag > -1:
            msg = (
                f"rec_linear: with a backward range (step=-1), all lags must be "
                f"<= -1; got max lag {max_lag}. For forward recurrences use a "
                f"forward range (step=+1) and positive lags."
            )
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
    # The kernel reads target[out_idx - lags[k]] at each step. The extremes:
    #   forward (step=+1, lags>0): earliest read is rng.first - max(lags).
    #   backward (step=-1, lags<0): latest read is rng.first - min(lags)
    #     (= rng.first + |min(lags)|, a position past rng.first).
    if step > 0:
        earliest_read = rng_first.value - int(lags_arr.max())
        if earliest_read < target.firstdate.value:
            earliest_mit = MIT(target.frequency, earliest_read)
            msg = (
                f"rec_linear: initial conditions missing — recurrence reads "
                f"target[{earliest_mit!s}] but target starts at {target.firstdate!s}."
            )
            raise ValueError(msg)
    else:
        latest_read = rng_first.value - int(lags_arr.min())
        if latest_read > target.lastdate.value:
            latest_mit = MIT(target.frequency, latest_read)
            msg = (
                f"rec_linear: initial conditions missing — backward recurrence "
                f"reads target[{latest_mit!s}] but target ends at {target.lastdate!s}."
            )
            raise ValueError(msg)

    # Ensure target covers rng (auto-extend mirrors `rec`'s setitem behaviour).
    target._ensure_covers(rng)
    values = target._values
    offset = rng_first.value - target.firstdate.value
    count = len(rng)

    # `target.dtype == np.float64` was validated above; `target._ensure_covers`
    # has just produced a 1-D contiguous buffer; coeffs/lags were created via
    # `np.asarray(..., dtype=...)` with explicit dtypes. The fast-path
    # contract from `_is_kernel_eligible` is satisfied unconditionally here,
    # so the dispatcher's only decision is Cython vs NumPy reference.
    _dispatch_kernel(
        _CYTHON_AVAILABLE,
        rec_linear_cython,
        rec_linear_numpy,
        values,
        offset,
        count,
        step,
        coeffs_arr,
        lags_arr,
    )
