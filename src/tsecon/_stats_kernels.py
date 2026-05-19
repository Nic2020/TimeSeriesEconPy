# SPDX-License-Identifier: MIT
"""NumPy reference kernels for the scalar-reduction statistics.

The pure-Python / NumPy sibling of ``_stats_kernels_cy.pyx``. Both modules
expose ``mean / var / std / cor`` kernels with the same signature; the
Cython version (when compiled) is faster, while this module is the canonical
fallback for installs without a C toolchain.

The split exists for the three-flavor benchmark thread (pure-NumPy
reference vs. compiled Cython vs. Julia upstream). Each kernel is timed
independently so the comparison table can show "vectorised NumPy" and
"compiled Cython" without public-API dispatch bias. See
[Cython strategy](../../docs/design/cython_strategy.md).

Layout note
-----------
``_stats_kernels.py`` (NumPy reference) and ``_stats_kernels_cy.pyx``
(Cython compiled) live adjacent under ``src/tsecon/``. The ``_cy`` suffix
on the ``.pyx`` is forced by Python's import semantics (a compiled
``_stats_kernels.pyd`` next to a ``_stats_kernels.py`` would shadow the
``.py`` module entirely, leaving callers unable to address the NumPy
reference for the benchmark). Same precedent as the ``_rec_kernels`` and
``_indexing_kernels`` pairs.

Stats vs gather vs recurrence: where Cython buys you something
--------------------------------------------------------------
The three M1.5 ports establish a structural classification of where
Cython actually helps a Python time-series library:

* **Recursion** (``_rec_kernels``) — non-vectorisable inner loop. The NumPy
  reference is itself a Python ``for`` loop because each step depends on
  a just-written predecessor. Cython buys ~65x over NumPy here because
  the entire inner loop is interpreter overhead.
* **Gather** (``_indexing_kernels``) — vectorisable. The NumPy reference
  is :func:`numpy.take`, already a tight C loop. Cython buys only ~1.1x
  over NumPy because there is nothing left to remove except per-call
  dispatch overhead.
* **Scalar reduction** (this module) — vectorisable, but the NumPy call
  *boxes its result as a 0-D numpy scalar* and pays per-call dispatch
  every time. The reduction itself is a tight C loop, so the inner work
  is fast; but a length-100 ``np.mean(arr)`` call still costs 5-10 us of
  Python C API + dtype dispatch + numpy scalar allocation. The Cython
  kernel is a tight C loop that returns a plain Python ``float``,
  shaving the per-call tax for tiny inputs (where the inner loop is
  cheap relative to the dispatch around it). Cython buys ~5-15x over
  NumPy here, sitting between recursion (big win) and gather (marginal).

That asymmetry is itself a JSS-paper finding: **for tiny scalar
reductions, Cython removes per-call constants that NumPy can't avoid
without breaking its general API contract.**

Kernel contract
---------------
``mean_numpy(values) -> float``
``var_numpy(values, ddof) -> float``
``std_numpy(values, ddof) -> float``
``cor_numpy(x, y) -> float``

* ``values`` / ``x`` / ``y`` — 1-D contiguous ``float64`` arrays, treated
  as read-only.
* ``ddof`` — integer Bessel correction (``1`` for sample, ``0`` for
  population). The kernel does not validate ``ddof``; callers ensure
  ``len(values) > ddof``.
* Returns a plain Python ``float`` — explicitly *not* a 0-D numpy scalar,
  so the kernel-direct row of the comparison table measures the same
  return shape as the Cython kernel.

The kernels perform no input validation — the public ``tsecon._stats``
wrappers handle TSeries / MVTSeries dispatch, dtype checks, and the
``len(values) <= ddof`` (return NaN) edge case before invoking the
kernel. The kernel assumes:

* ``values.ndim == 1`` and ``values.dtype == float64``
* ``cor`` inputs satisfy ``x.shape[0] == y.shape[0] >= 2``
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt

__all__ = ["cor_numpy", "mean_numpy", "std_numpy", "var_numpy"]


def mean_numpy(values: npt.NDArray[np.float64]) -> float:
    """Return ``float(np.mean(values))``.

    Pure-NumPy reference. The matching Cython kernel
    :func:`tsecon._stats_kernels_cy.mean_cython` carries the same
    signature and behaviour but returns the result without going through
    NumPy's per-call dispatch + 0-D scalar allocation.
    """
    return float(np.mean(values))


def var_numpy(values: npt.NDArray[np.float64], ddof: int) -> float:
    """Return ``float(np.var(values, ddof=ddof))``.

    Pure-NumPy reference. See module docstring for the kernel contract.
    """
    return float(np.var(values, ddof=ddof))


def std_numpy(values: npt.NDArray[np.float64], ddof: int) -> float:
    """Return ``float(np.std(values, ddof=ddof))``.

    Pure-NumPy reference. See module docstring for the kernel contract.
    """
    return float(np.std(values, ddof=ddof))


def cor_numpy(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    """Return the scalar Pearson correlation ``float(np.corrcoef(x, y)[0, 1])``.

    Pure-NumPy reference. ``np.corrcoef`` allocates an internal 2-by-N
    stacked matrix, computes the 2-by-2 correlation matrix, and returns
    that; this wrapper extracts the off-diagonal scalar so the return
    shape matches the Cython kernel and the public ``cor(x, y)`` form.

    Constant-input guard
    --------------------
    Both inputs are scanned for bit-exact constancy up front and ``nan``
    is returned (plus a ``RuntimeWarning``) when either is constant.
    ``np.corrcoef`` itself only returns ``nan`` when the centred values
    are *FP-exactly* zero; on FP-noisy constants (e.g. ``np.full(100, 1e-60)``
    where pairwise summation produces a slightly-off mean) it silently
    returns ``1.0``. The explicit guard collapses both regimes to ``nan``
    so this kernel agrees with :func:`tsecon._stats_kernels_cy.cor_cython`
    on any constant input, not just the FP-exact subset. ``min == max`` is
    used as the detector because ``np.var`` would itself drift below
    machine precision on FP-noisy constants and fail to fire.
    """
    if x.min() == x.max() or y.min() == y.max():
        warnings.warn(
            "invalid value encountered in cor (constant input)",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])
