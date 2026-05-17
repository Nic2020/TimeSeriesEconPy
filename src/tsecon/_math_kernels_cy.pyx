# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# distutils: language=c
# SPDX-License-Identifier: MIT
"""Cython kernel: anchored cumulative sum over a float64 slice.

The compiled sibling of ``_math_kernels.py``'s
:func:`cumsum_anchored_numpy`. Both kernels expose an identical
signature and produce identical-modulo-FP-rounding output; the only
difference is execution speed. See the ``.py`` module docstring for
the kernel contract and the "anchored cumsum" placement in the
multi-port classification.

The implementation is written in pure C via Cython typed memoryviews —
no per-iteration Python dispatch — and fuses the cumsum and the
constant-shift correction into a single forward pass. The two-pass
NumPy reference walks ``values`` once for ``np.cumsum`` and again for
the in-place ``+= correction``; this kernel does both in one pass,
saving an extra read/write per element on top of removing per-call
dispatch overhead.

Why a fused single pass here
----------------------------
Anchored cumsum is fully vectorisable — the NumPy reference is
already a single ``np.cumsum`` C call plus an in-place add. The
honest comparison is "what does Cython buy on top of two well-tuned
NumPy C calls?" The answer is per-call dispatch + the second pass.
For length-100 slices typical of frequency-conversion workloads the
extra ~half-µs that NumPy spends per call dominates the actual
integration work, so the kernel-direct path is the one where the
multi-port classification's N=5 row lands its measurement.

Build
-----
Compiled by ``hatch_build.py`` at wheel build time. The compiled
extension lands at ``src/tsecon/_math_kernels_cy.{pyd,so}`` and is
imported by :mod:`tsecon._math` as the optional fast path. When the
extension is not built, callers fall back to the NumPy reference.
"""

import numpy as np
cimport numpy as cnp
cimport cython

cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def cumsum_anchored_cython(
    cnp.ndarray[cnp.float64_t, ndim=1] values,
    Py_ssize_t offset,
    Py_ssize_t count,
    double anchor_value,
    Py_ssize_t anchor_relative_idx,
) -> None:
    """Integrate ``values[offset : offset + count]`` so the result equals ``anchor_value`` at the anchor.

    Cython implementation. See ``_math_kernels.py`` module docstring
    for the contract; the public :func:`tsecon._math.undiff` /
    :func:`tsecon._math.undiff_inplace` wrappers validate inputs
    before calling this kernel.

    Two-pass for correctness when the anchor falls inside the chunk:
    pass 1 computes the in-place cumsum and (if needed) captures the
    reference at ``anchor_relative_idx``; pass 2 adds the constant
    correction back. For the ``anchor_relative_idx < 0`` regime
    (``undiff_inplace`` with anchor immediately before the chunk),
    the reference is ``0.0`` and the two passes collapse into a single
    fused forward walk (cumsum + add).
    """
    if count == 0:
        return

    cdef double[::1] v_view = values
    cdef Py_ssize_t i
    cdef double acc = 0.0
    cdef double correction
    cdef double reference

    if anchor_relative_idx < 0:
        # Fused single pass: anchor sits before the chunk so the cumsum
        # reference is 0.0 and correction = anchor_value. Each step adds
        # the running prefix-sum and the anchor offset in one write.
        for i in range(count):
            acc += v_view[offset + i]
            v_view[offset + i] = acc + anchor_value
        return

    # Anchor inside the chunk: two passes — first cumsum in place,
    # capture the reference value, then add the correction.
    for i in range(count):
        acc += v_view[offset + i]
        v_view[offset + i] = acc
    reference = v_view[offset + anchor_relative_idx]
    correction = anchor_value - reference
    for i in range(count):
        v_view[offset + i] += correction
