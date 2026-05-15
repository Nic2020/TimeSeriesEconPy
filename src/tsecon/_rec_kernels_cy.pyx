# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# distutils: language=c
# SPDX-License-Identifier: MIT
"""Cython kernel: linear recurrence over a float64 array.

The compiled sibling of ``_rec_kernels.py``'s :func:`rec_linear_numpy`.
Both kernels expose an identical signature and produce identical
output; the only difference is execution speed. See the ``.py`` module
docstring for the kernel contract.

The implementation is written in pure C via Cython typed memoryviews —
no per-iteration Python dispatch — so each step inside the inner loops
compiles to a `double *` load, a multiply-add, and a store. The
recurrence is structurally non-vectorizable (each step depends on a
just-written predecessor), which is exactly why the Python equivalent
pays a heavy interpreter tax that the C path eliminates.

Build
-----
Compiled by ``hatch_build.py`` at wheel build time. The compiled
extension lands at ``src/tsecon/_rec_kernels_cy.{pyd,so}`` and is
imported by :mod:`tsecon.recursive` as the optional fast path. When the
extension is not built, callers fall back to the NumPy reference.
"""

import numpy as np
cimport numpy as cnp
cimport cython

cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def rec_linear_cython(
    cnp.ndarray[cnp.float64_t, ndim=1] values,
    Py_ssize_t offset,
    Py_ssize_t count,
    cnp.ndarray[cnp.float64_t, ndim=1] coeffs,
    cnp.ndarray[cnp.int64_t, ndim=1] lags,
) -> None:
    """Compute ``values[offset + i] = Σ_k coeffs[k] * values[offset + i - lags[k]]``.

    Cython implementation. See ``_rec_kernels.py`` module docstring for
    the contract; the wrapper :func:`tsecon.recursive.rec_linear`
    validates inputs before calling this kernel.
    """
    cdef Py_ssize_t n_terms = coeffs.shape[0]
    cdef Py_ssize_t i, k, out_idx
    cdef double acc
    cdef double[::1] vals_view = values
    cdef double[::1] c_view = coeffs
    cdef long long[::1] l_view = lags

    for i in range(count):
        out_idx = offset + i
        acc = 0.0
        for k in range(n_terms):
            acc += c_view[k] * vals_view[out_idx - <Py_ssize_t>l_view[k]]
        vals_view[out_idx] = acc
