# SPDX-License-Identifier: MIT
"""NumPy reference kernel for the linear-recurrence form of :func:`rec`.

The pure-Python / NumPy sibling of ``_rec_kernels_cy.pyx``. Both
modules expose kernels with the same signature; the Cython version
(when compiled) is much faster, while this module is the canonical
fallback for installs without a C toolchain.

The split exists for the three-flavor benchmark thread (pure-NumPy
reference vs. compiled Cython vs. Julia upstream). Each kernel is
timed independently so the comparison table can show *both* "pure
Python's ceiling" and "what Cython buys us" without dispatch overhead
biasing either measurement. See
[Cython strategy](../../docs/design/cython_strategy.md).

Layout note
-----------
The two source files are kept adjacent under ``src/tsecon/`` to make
the comparison physically obvious in PR review and JSS paper
discussion: ``_rec_kernels.py`` (NumPy reference) and
``_rec_kernels_cy.pyx`` (Cython compiled). The ``_cy`` suffix is forced
by Python's import semantics — a compiled ``_rec_kernels.pyd`` next to
a ``_rec_kernels.py`` would shadow the ``.py`` module, leaving callers
unable to address the NumPy reference for the three-flavor benchmark.

Kernel contract
---------------
Both kernels share a low-level signature that excludes any TSeries
dispatch or MIT arithmetic from the timed window::

    rec_linear_numpy(values, offset, count, step, coeffs, lags) -> None
    rec_linear_cython(values, offset, count, step, coeffs, lags) -> None

* ``values`` — 1-D ``float64`` array, mutated in place. Both source
  reads and destination writes go through this buffer.
* ``offset`` — integer position of the *first* destination index. The
  kernel writes positions ``offset, offset + step, ..., offset + (count-1) * step``.
* ``count`` — number of destination steps to execute.
* ``step`` — direction of iteration, ``+1`` for forward or ``-1`` for
  backward. Other strides are rejected by the wrapper; the kernel is
  written as if any signed nonzero step worked, but bounds-validation
  upstream only certifies ``±1``.
* ``coeffs`` — 1-D ``float64`` array of recurrence weights.
* ``lags`` — 1-D ``int64`` array of nonzero lag offsets. ``coeffs``
  and ``lags`` are zip-aligned: step ``i`` computes
  ``values[offset + i*step] = Σ_k coeffs[k] * values[offset + i*step - lags[k]]``.
  For correctness, every ``lags[k]`` must have the same sign as
  ``step`` (forward step + positive lag reads an earlier-written
  position; backward step + negative lag reads a later-written one).

The kernels perform no input validation — the public
:func:`tsecon.recursive.rec_linear` wrapper handles shape and bounds
checks before invoking the kernel. The kernel assumes:

* ``coeffs.shape == lags.shape``
* ``sign(lags[k]) == sign(step)`` for every ``k`` (so each read
  references an already-written or initial-condition position).
* For forward iteration: ``offset - max(lags) >= 0`` and
  ``offset + (count - 1) >= 0`` (every read and write is in-range).
* For backward iteration: ``offset - min(lags) < len(values)`` (the
  furthest read forward in array index space is in-range) and
  ``offset - (count - 1) >= 0`` (the furthest write back is in-range).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["rec_linear_numpy"]


def rec_linear_numpy(
    values: npt.NDArray[np.float64],
    offset: int,
    count: int,
    step: int,
    coeffs: npt.NDArray[np.float64],
    lags: npt.NDArray[np.int64],
) -> None:
    """Compute ``values[offset + i*step] = Σ_k coeffs[k] * values[offset + i*step - lags[k]]``.

    Pure-Python / NumPy reference. See module docstring for the kernel
    contract; the matching Cython kernel in ``_rec_kernels_cy.pyx``
    (when compiled) carries the same signature and behaviour.
    """
    n_terms = coeffs.shape[0]
    for i in range(count):
        out_idx = offset + i * step
        acc = 0.0
        for k in range(n_terms):
            acc += coeffs[k] * values[out_idx - lags[k]]
        values[out_idx] = acc
