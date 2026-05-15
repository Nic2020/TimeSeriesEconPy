# SPDX-License-Identifier: MIT
"""NumPy reference kernel for the linear-recurrence form of :func:`rec`.

The pure-Python / NumPy sibling of ``_rec_kernels_cy.pyx``. Both
modules expose kernels with the same signature; the Cython version
(when compiled) is much faster, while this module is the canonical
fallback for installs without a C toolchain.

The split exists for the three-flavor benchmark thread described in
``claude_files/decisions/17_cython_dispatch_strategy.md`` and
``claude_files/paper/NOTES.md`` § "Three-flavor benchmark". Each kernel
is timed independently (option β in decision 17) so the comparison
table can show *both* "pure Python's ceiling" and "what Cython buys us"
without dispatch overhead biasing either measurement.

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

    rec_linear_numpy(values, offset, count, coeffs, lags) -> None
    rec_linear_cython(values, offset, count, coeffs, lags) -> None

* ``values`` — 1-D ``float64`` array, mutated in place. Both source
  reads and destination writes go through this buffer.
* ``offset`` — integer position of the *first* destination index. The
  kernel writes positions ``offset, offset + 1, ..., offset + count - 1``.
* ``count`` — number of destination steps to execute.
* ``coeffs`` — 1-D ``float64`` array of recurrence weights.
* ``lags`` — 1-D ``int64`` array of positive lag offsets (each ``>= 1``
  so every read references an earlier-written position). ``coeffs``
  and ``lags`` are zip-aligned: step ``i`` computes
  ``values[offset + i] = Σ_k coeffs[k] * values[offset + i - lags[k]]``.

The kernels perform no input validation — the public
:func:`tsecon.recursive.rec_linear` wrapper handles shape and bounds
checks before invoking the kernel. The kernel assumes:

* ``coeffs.shape == lags.shape``
* ``min(lags) >= 1``
* ``offset - max(lags) >= 0`` (every read is in-range)
* ``offset + count <= len(values)`` (every write is in-range)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["rec_linear_numpy"]


def rec_linear_numpy(
    values: npt.NDArray[np.float64],
    offset: int,
    count: int,
    coeffs: npt.NDArray[np.float64],
    lags: npt.NDArray[np.int64],
) -> None:
    """Compute ``values[offset + i] = Σ_k coeffs[k] * values[offset + i - lags[k]]``.

    Pure-Python / NumPy reference. See module docstring for the kernel
    contract; the matching Cython kernel in ``_rec_kernels_cy.pyx``
    (when compiled) carries the same signature and behaviour.
    """
    n_terms = coeffs.shape[0]
    for i in range(count):
        out_idx = offset + i
        acc = 0.0
        for k in range(n_terms):
            acc += coeffs[k] * values[out_idx - lags[k]]
        values[out_idx] = acc
