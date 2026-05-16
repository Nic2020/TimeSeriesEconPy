# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# distutils: language=c
# SPDX-License-Identifier: MIT
"""Cython kernel: gather values at integer offsets into a fresh float64 array.

The compiled sibling of ``_indexing_kernels.py``'s :func:`gather_numpy`.
Both kernels expose an identical signature and produce identical
output; the only difference is execution speed. See the ``.py`` module
docstring for the kernel contract and the "indexing vs recursion"
asymmetry note.

The implementation is written in pure C via Cython typed memoryviews —
no per-iteration Python dispatch — so each step compiles to an
``int64`` load, a ``double *`` load, and a store. The gather is
structurally vectorisable (`np.take` already runs the same C loop
under the hood), so this kernel's win over the NumPy reference is
modest — usually 1.5–3× rather than the 50× ``rec_linear_cython``
shows over ``rec_linear_numpy``. The marginal gain is NumPy's
per-call dispatch overhead (``__array_function__`` machinery, dtype
inspection, ufunc registration lookup), not the inner loop.

Build
-----
Compiled by ``hatch_build.py`` at wheel build time. The compiled
extension lands at ``src/tsecon/_indexing_kernels_cy.{pyd,so}`` and is
imported by :mod:`tsecon.indexing` as the optional fast path. When the
extension is not built, callers fall back to the NumPy reference.
"""

import numpy as np
cimport numpy as cnp
cimport cython

cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def gather_cython(
    cnp.ndarray[cnp.float64_t, ndim=1] values,
    cnp.ndarray[cnp.int64_t, ndim=1] indices,
) -> cnp.ndarray:
    """Return ``values[indices]`` as a freshly-allocated 1-D ``float64`` array.

    Cython implementation. See ``_indexing_kernels.py`` module docstring
    for the contract; the wrapper :func:`tsecon.indexing.lookup`
    validates inputs before calling this kernel.
    """
    cdef Py_ssize_t n = indices.shape[0]
    cdef Py_ssize_t i
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(n, dtype=np.float64)

    cdef double[::1] vals_view = values
    cdef long long[::1] idx_view = indices
    cdef double[::1] out_view = out

    for i in range(n):
        out_view[i] = vals_view[<Py_ssize_t>idx_view[i]]

    return out
