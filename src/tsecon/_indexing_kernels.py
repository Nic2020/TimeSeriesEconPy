# SPDX-License-Identifier: MIT
"""NumPy reference kernel for the vectorised time-series gather.

The pure-Python / NumPy sibling of ``_indexing_kernels_cy.pyx``. Both
modules expose a ``gather_*(values, indices)`` kernel with the same
signature and return value; the Cython version (when compiled) is
slightly faster, while this module is the canonical fallback for
installs without a C toolchain.

The split exists for the three-flavor benchmark thread (pure-NumPy
reference vs. compiled Cython vs. Julia upstream). Each kernel is
timed independently so the comparison table can show both "vectorised
NumPy" and "compiled Cython" without public-API dispatch bias.
See [Cython strategy](../../docs/design/cython_strategy.md).

Layout note
-----------
``_indexing_kernels.py`` (NumPy reference) and
``_indexing_kernels_cy.pyx`` (Cython compiled) live adjacent under
``src/tsecon/`` — the ``_cy`` suffix on the ``.pyx`` is forced by
Python's import semantics (a compiled ``_indexing_kernels.pyd`` next to
a ``_indexing_kernels.py`` would shadow the ``.py`` module entirely,
leaving callers unable to address the NumPy reference for the
benchmark). Same precedent as the ``_rec_kernels`` pair.

Indexing vs recursion: why the NumPy reference is fast
------------------------------------------------------
Unlike the recurrence kernel (where each step depends on a
just-written predecessor and the inner loop is structurally
non-vectorisable), the gather operation *is* vectorisable: ``values[
indices]`` runs entirely in NumPy's C layer with a single Python call.
So this NumPy reference is already close to its theoretical ceiling —
the Cython kernel buys a *marginal* further win by removing NumPy's
per-call dispatch overhead, not a 50x improvement of the kind
``rec_linear_cython`` shows over ``rec_linear_numpy``.

That asymmetry is itself a JSS-paper finding: **for vectorisable ops,
exposing the vectorised API is the win; Cython is only the cherry on
top.** For non-vectorisable ops (like ``rec_linear``), Cython is the
load-bearing fix.

Kernel contract
---------------
``gather_numpy(values, indices) -> out``

* ``values`` — 1-D contiguous ``float64`` array, treated as read-only.
* ``indices`` — 1-D contiguous ``int64`` array of non-negative offsets
  into ``values``.
* Returns a freshly-allocated 1-D ``float64`` array of length
  ``len(indices)`` containing ``values[indices[i]]`` for each ``i``.

The kernel performs no input validation — the public
:func:`tsecon.indexing.lookup` wrapper handles type / shape / bounds
checks before invoking the kernel. The kernel assumes:

* ``values.ndim == 1`` and ``indices.ndim == 1``
* ``values.dtype == float64`` and ``indices.dtype == int64``
* ``0 <= indices[i] < len(values)`` for every ``i``
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["gather_numpy"]


def gather_numpy(
    values: npt.NDArray[np.float64],
    indices: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """Return ``values[indices]`` as a freshly-allocated 1-D ``float64`` array.

    Pure-NumPy reference. See module docstring for the kernel
    contract; the matching Cython kernel in ``_indexing_kernels_cy.pyx``
    (when compiled) carries the same signature and behaviour but
    eliminates NumPy's per-call dispatch overhead in exchange for a
    marginal further speedup.

    The implementation uses :func:`numpy.take` rather than fancy
    indexing for two reasons: ``take`` always allocates a fresh array
    (matching the kernel contract — callers must be able to mutate the
    result without aliasing ``values``), and it skips the ``__array_*``
    protocol dispatch that fancy indexing would otherwise pay through.
    """
    return np.take(values, indices)
