# SPDX-License-Identifier: MIT
"""NumPy reference kernel for the anchored-cumsum step of :func:`undiff`.

The pure-Python / NumPy sibling of ``_math_kernels_cy.pyx``. Both
modules expose a ``cumsum_anchored_*`` kernel with the same signature
and behaviour; the Cython version (when compiled) fuses the cumsum and
the constant-shift correction into a single C pass, while this module
is the canonical fallback for installs without a C toolchain.

The split exists for the multi-flavor benchmark thread described in
``claude_files/decisions/17_cython_dispatch_strategy.md`` and
``claude_files/paper/NOTES.md`` § "Three-flavor benchmark". Each kernel
is timed independently (option β in decision 17) so the comparison
table can show both "vectorised NumPy" and "compiled Cython" without
public-API dispatch bias.

Layout note
-----------
``_math_kernels.py`` (NumPy reference) and ``_math_kernels_cy.pyx``
(Cython compiled) live adjacent under ``src/tsecon/`` — the ``_cy``
suffix on the ``.pyx`` is forced by Python's import semantics (a
compiled ``_math_kernels.pyd`` next to a ``_math_kernels.py`` would
shadow the ``.py`` module entirely, leaving callers unable to address
the NumPy reference for the benchmark). Same precedent as the four
M1.5 kernel pairs (``_rec_kernels`` / ``_indexing_kernels`` /
``_stats_kernels`` / ``_fconvert_kernels``).

Anchored-cumsum vs the N=4 multi-port classification
----------------------------------------------------
The five M1.6.2 ports now establish a structural classification of
where Cython actually helps a Python time-series library (extends the
session-21 N=4 table in ``decisions/18_cython_port_plan.md``):

* **Recursion** (``_rec_kernels``) — non-vectorisable inner loop.
  Cython buys ~65x over NumPy.
* **Gather** (``_indexing_kernels``) — vectorisable (``np.take``).
  Cython buys ~1.1x over NumPy.
* **Scalar reduction** (``_stats_kernels``) — vectorisable but pays
  per-call dispatch + 0-D scalar boxing. Cython buys ~5-40x over
  NumPy.
* **Group aggregation** (``_fconvert_kernels``) — partially vectorisable
  with an outer per-group Python loop over inner C reductions. Cython
  buys ~25-80x over NumPy.
* **Anchored cumsum** (this module) — fully vectorisable
  (``np.cumsum``) plus a constant-shift correction. The NumPy
  reference is already close to its ceiling (one ``np.cumsum`` C call,
  one in-place ``+=``); Cython buys whatever is left of NumPy's
  per-call dispatch + the cost of writing through an intermediate
  cumsum buffer. Expected to land closer to the gather regime
  (low single-digit multiplier) than to the recursion regime,
  refining the N=4 classification's "outer-loop tax" band downward
  for fully-vectorisable ops.

See ``paper/NOTES.md`` § "undiff kernel — the N=5 row" for the
empirical confirmation once benchmarks land.

Kernel contract
---------------
``cumsum_anchored_numpy(values, offset, count, anchor_value, anchor_relative_idx) -> None``

* ``values`` — 1-D contiguous ``float64`` array, mutated in place. On
  entry, ``values[offset : offset + count]`` holds the differenced
  series ``dvar``; on exit, the same slice holds the integrated
  result ``cumsum(dvar) + (anchor_value − cumsum(dvar)[anchor_relative_idx])``
  (with the convention below for negative ``anchor_relative_idx``).
  Positions outside ``[offset, offset + count)`` are untouched.
* ``offset`` — non-negative integer position of the first integrated
  element in ``values``.
* ``count`` — number of elements to integrate (``>= 0``). ``count == 0``
  is a no-op.
* ``anchor_value`` — the integrated series equals this value at the
  anchor position. Plain Python ``float`` (or coercible).
* ``anchor_relative_idx`` — position of the anchor relative to
  ``offset``. Two regimes:

  * ``anchor_relative_idx >= 0`` — anchor falls inside the integrated
    chunk. ``correction = anchor_value − cumsum[anchor_relative_idx]``.
    Must satisfy ``anchor_relative_idx < count``.
  * ``anchor_relative_idx < 0`` — anchor falls strictly before
    ``offset``. The cumsum reference is treated as ``0`` (no
    integration has accumulated at the anchor position), so
    ``correction = anchor_value``. This is the regime
    :func:`tsecon._math.undiff_inplace` exercises: the anchor is
    ``var[fromdate]`` and the integrated chunk starts at
    ``fromdate + 1``.

The kernel performs no input validation — the public
:func:`tsecon._math.undiff` / :func:`tsecon._math.undiff_inplace`
wrappers handle anchor resolution, dvar extension for out-of-range
anchors, dtype promotion, and frequency checks before invoking the
kernel. The kernel assumes:

* ``values.ndim == 1`` and ``values.dtype == float64``
* ``values`` is C-contiguous
* ``0 <= offset`` and ``offset + count <= len(values)``
* ``count >= 0``
* either ``anchor_relative_idx < 0`` or ``0 <= anchor_relative_idx < count``
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["cumsum_anchored_numpy"]


def cumsum_anchored_numpy(
    values: npt.NDArray[np.float64],
    offset: int,
    count: int,
    anchor_value: float,
    anchor_relative_idx: int,
) -> None:
    """Integrate the slice so the result equals ``anchor_value`` at the anchor position.

    Pure-NumPy reference. The slice is ``values[offset : offset + count]``.
    See module docstring for the kernel contract;
    the matching Cython kernel in ``_math_kernels_cy.pyx`` (when
    compiled) carries the same signature and behaviour but fuses the
    cumsum + constant-shift into a single C pass.

    The implementation runs :func:`numpy.cumsum` over the slice
    in-place, reads the reference value at the anchor position
    (or ``0.0`` when ``anchor_relative_idx < 0``), then adds the
    correction back into the slice with an in-place ``+=``. The
    intermediate cumsum lives in ``values`` itself, so total
    allocation is a single intermediate scratch buffer that NumPy uses
    for the ``cumsum`` call.
    """
    if count == 0:
        return
    chunk = values[offset : offset + count]
    np.cumsum(chunk, out=chunk)
    reference = chunk[anchor_relative_idx] if anchor_relative_idx >= 0 else 0.0
    correction = anchor_value - reference
    chunk += correction
