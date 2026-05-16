# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# distutils: language=c
# SPDX-License-Identifier: MIT
"""Cython kernels: scalar-reduction stats over float64 arrays.

The compiled sibling of ``_stats_kernels.py``'s reference kernels.
Both modules expose ``mean / var / std / cor`` with identical
signatures and produce identical-modulo-FP-rounding output; the only
difference is execution speed. See the ``.py`` module docstring for
the kernel contract and the "stats vs gather vs recurrence" framing.

The implementation is written in pure C via Cython typed memoryviews —
no per-iteration Python dispatch — and returns a plain Python ``float``
rather than a 0-D numpy scalar. NumPy's ``np.mean(arr)`` etc. fast for
big inputs but pay a 5–10 µs per-call dispatch + scalar-boxing tax that
dominates length-100 timings; the C path eliminates both.

Numerical note
--------------
The variance kernel uses a two-pass algorithm (compute the mean, then
sum the centered squared diffs) — same shape as ``np.var`` and
``Statistics.var`` in Julia. For length-100 well-conditioned inputs the
results agree with NumPy to within a few ulps; the bit-for-bit-equal
contract that the rec_linear kernels carry does not apply here because
NumPy uses pairwise summation internally and the kernel uses naive
left-to-right accumulation. The wrapping ``_stats.var/std`` uses
``np.testing.assert_allclose(rtol=1e-12)``-grade equivalence, not
``assert_array_equal``.

Build
-----
Compiled by ``hatch_build.py`` at wheel build time. The compiled
extension lands at ``src/tsecon/_stats_kernels_cy.{pyd,so}`` and is
imported by :mod:`tsecon._stats` as the optional fast path. When the
extension is not built, callers fall back to the NumPy reference.
"""

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport sqrt

cnp.import_array()


@cython.boundscheck(False)
@cython.wraparound(False)
def mean_cython(cnp.ndarray[cnp.float64_t, ndim=1] values) -> float:
    """Return the arithmetic mean of ``values`` as a Python ``float``."""
    cdef Py_ssize_t i, n = values.shape[0]
    cdef double s = 0.0
    cdef double[::1] v_view = values

    for i in range(n):
        s += v_view[i]
    return s / n


@cython.boundscheck(False)
@cython.wraparound(False)
def var_cython(
    cnp.ndarray[cnp.float64_t, ndim=1] values,
    int ddof,
) -> float:
    """Return the sample variance (with ``ddof`` Bessel correction)."""
    cdef Py_ssize_t i, n = values.shape[0]
    cdef double s = 0.0, m, d, ssd = 0.0
    cdef double[::1] v_view = values

    for i in range(n):
        s += v_view[i]
    m = s / n
    for i in range(n):
        d = v_view[i] - m
        ssd += d * d
    return ssd / (n - ddof)


@cython.boundscheck(False)
@cython.wraparound(False)
def std_cython(
    cnp.ndarray[cnp.float64_t, ndim=1] values,
    int ddof,
) -> float:
    """Return the sample standard deviation (with ``ddof`` Bessel correction)."""
    cdef Py_ssize_t i, n = values.shape[0]
    cdef double s = 0.0, m, d, ssd = 0.0
    cdef double[::1] v_view = values

    for i in range(n):
        s += v_view[i]
    m = s / n
    for i in range(n):
        d = v_view[i] - m
        ssd += d * d
    return sqrt(ssd / (n - ddof))


@cython.boundscheck(False)
@cython.wraparound(False)
def cor_cython(
    cnp.ndarray[cnp.float64_t, ndim=1] x,
    cnp.ndarray[cnp.float64_t, ndim=1] y,
) -> float:
    """Return the scalar Pearson correlation between ``x`` and ``y``."""
    cdef Py_ssize_t i, n = x.shape[0]
    cdef double sx = 0.0, sy = 0.0
    cdef double mx, my, dx, dy
    cdef double sxx = 0.0, syy = 0.0, sxy = 0.0
    cdef double[::1] x_view = x
    cdef double[::1] y_view = y

    for i in range(n):
        sx += x_view[i]
        sy += y_view[i]
    mx = sx / n
    my = sy / n
    for i in range(n):
        dx = x_view[i] - mx
        dy = y_view[i] - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    return sxy / sqrt(sxx * syy)
