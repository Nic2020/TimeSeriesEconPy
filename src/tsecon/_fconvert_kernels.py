# SPDX-License-Identifier: MIT
"""NumPy reference kernel for the lower-frequency aggregation step.

The pure-Python / NumPy sibling of ``_fconvert_kernels_cy.pyx``. Both
modules expose an ``aggregate_groups_*(values, group_starts,
group_lengths, method_code)`` kernel with the same signature and return
value; the Cython version (when compiled) is faster, while this module
is the canonical fallback for installs without a C toolchain.

The split exists for the three-flavor benchmark thread (pure-NumPy
reference vs. compiled Cython vs. Julia upstream). Each kernel is
timed independently so the comparison table can show both "NumPy
reference" and "compiled Cython" without public-API dispatch bias.
See [Cython strategy](../../docs/design/cython_strategy.md).

Layout note
-----------
``_fconvert_kernels.py`` (NumPy reference) and
``_fconvert_kernels_cy.pyx`` (Cython compiled) live adjacent under
``src/tsecon/`` — the ``_cy`` suffix on the ``.pyx`` is forced by
Python's import semantics (a compiled ``_fconvert_kernels.pyd`` next to
a ``_fconvert_kernels.py`` would shadow the ``.py`` module entirely,
leaving callers unable to address the NumPy reference for the
benchmark). Same precedent as the ``_rec_kernels`` / ``_indexing_kernels``
/ ``_stats_kernels`` pairs.

Aggregation vs gather vs recurrence: where Cython buys you something
--------------------------------------------------------------------
The four M1.5 ports now establish a structural classification of
where Cython actually helps a Python time-series library:

* **Recursion** (``_rec_kernels``) — non-vectorisable inner loop.
  Cython buys ~65x over NumPy.
* **Gather** (``_indexing_kernels``) — vectorisable (``np.take``).
  Cython buys ~1.1x over NumPy.
* **Scalar reduction** (``_stats_kernels``) — vectorisable but pays
  per-call dispatch + 0-D scalar boxing. Cython buys ~5-40x over
  NumPy.
* **Group aggregation** (this module) — *partially* vectorisable.
  The inner reduction over each group is C, but the outer per-group
  loop must materialise one Python call (or one NumPy reduction) per
  group. For a Quarterly→Yearly conversion with 25 output groups, the
  outer loop pays 25x the per-call dispatch tax. The Cython kernel
  fuses the outer loop into C, eliminating per-group dispatch.

That asymmetry — the "outer Python loop over inner C reductions" — is
itself a JSS-paper finding: **the per-group dispatch cost dominates
when groups are small (4-12 elements), exactly the regime
frequency-conversion lives in.**

Kernel contract
---------------
``aggregate_groups_numpy(values, group_starts, group_lengths, method_code) -> out``

* ``values`` — 1-D contiguous ``float64`` array, treated as read-only.
* ``group_starts`` — 1-D contiguous ``int64`` array of length
  ``n_groups``. Each entry is the start index into ``values`` of one
  group. Groups must be non-overlapping but need not be contiguous in
  the input (i.e. start[g] + length[g] <= start[g+1] is permitted but
  not required).
* ``group_lengths`` — 1-D contiguous ``int64`` array of length
  ``n_groups``. Each ``length[g] >= 1`` (the kernel does not divide
  by zero or index empty groups; callers ensure this).
* ``method_code`` — integer selector for the reduction:
  * ``0`` = mean (sum / length)
  * ``1`` = sum
  * ``2`` = min
  * ``3`` = max
  * ``4`` = first  (values[start[g]])
  * ``5`` = last   (values[start[g] + length[g] - 1])
* Returns a freshly-allocated 1-D ``float64`` array of length
  ``n_groups``.

The kernel performs no input validation — the public ``fconvert``
wrappers in :mod:`tsecon.fconvert._tseries` handle method translation,
range arithmetic, and dtype checks before invoking the kernel. The
kernel assumes:

* ``values.ndim == 1`` and ``values.dtype == float64``
* ``group_starts.ndim == 1`` and ``group_starts.dtype == int64``
* ``group_lengths.ndim == 1`` and ``group_lengths.dtype == int64``
* ``group_starts.shape == group_lengths.shape``
* ``0 <= start[g]`` and ``start[g] + length[g] <= len(values)`` for every ``g``
* ``length[g] >= 1`` for every ``g``
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "METHOD_FIRST",
    "METHOD_LAST",
    "METHOD_MAX",
    "METHOD_MEAN",
    "METHOD_MIN",
    "METHOD_SUM",
    "aggregate_groups_numpy",
]

METHOD_MEAN: int = 0
METHOD_SUM: int = 1
METHOD_MIN: int = 2
METHOD_MAX: int = 3
METHOD_FIRST: int = 4
METHOD_LAST: int = 5


def aggregate_groups_numpy(
    values: npt.NDArray[np.float64],
    group_starts: npt.NDArray[np.int64],
    group_lengths: npt.NDArray[np.int64],
    method_code: int,
) -> npt.NDArray[np.float64]:
    """Aggregate ``values`` over ``n_groups`` contiguous groups.

    Pure-NumPy reference. See module docstring for the kernel contract;
    the matching Cython kernel in ``_fconvert_kernels_cy.pyx`` (when
    compiled) carries the same signature and behaviour but fuses the
    outer per-group loop into C.

    The implementation loops over groups in Python and invokes the
    NumPy reduction per group. This is the column we're trying to beat
    — the per-group dispatch cost is exactly what the Cython kernel
    removes.
    """
    n_groups = group_starts.shape[0]
    out = np.empty(n_groups, dtype=np.float64)
    if method_code == METHOD_MEAN:
        for g in range(n_groups):
            s = int(group_starts[g])
            length = int(group_lengths[g])
            out[g] = values[s : s + length].mean()
    elif method_code == METHOD_SUM:
        for g in range(n_groups):
            s = int(group_starts[g])
            length = int(group_lengths[g])
            out[g] = values[s : s + length].sum()
    elif method_code == METHOD_MIN:
        for g in range(n_groups):
            s = int(group_starts[g])
            length = int(group_lengths[g])
            out[g] = values[s : s + length].min()
    elif method_code == METHOD_MAX:
        for g in range(n_groups):
            s = int(group_starts[g])
            length = int(group_lengths[g])
            out[g] = values[s : s + length].max()
    elif method_code == METHOD_FIRST:
        for g in range(n_groups):
            out[g] = values[int(group_starts[g])]
    elif method_code == METHOD_LAST:
        for g in range(n_groups):
            s = int(group_starts[g])
            length = int(group_lengths[g])
            out[g] = values[s + length - 1]
    else:
        msg = f"aggregate_groups_numpy: unknown method_code {method_code!r}."
        raise ValueError(msg)
    return out
