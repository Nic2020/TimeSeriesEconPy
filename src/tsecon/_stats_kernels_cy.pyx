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

import warnings

import numpy as np
cimport numpy as cnp
cimport cython
from libc.math cimport fabs, frexp, isfinite, ldexp, sqrt

cnp.import_array()


cdef inline int _scale_exponent(double largest) noexcept nogil:
    """The binary exponent of ``largest`` (``ldexp(largest, -e)`` in [0.5, 1)); 0 if zero/nonfinite.

    The same exponent as ``_stats_kernels._scale_exponent`` (see
    "Correlation scaling" in that module's docstring). It is applied per
    value with ``ldexp`` because ``2**-e`` itself need not be representable
    (a subnormal maximum needs ``e = -1073``).
    """
    cdef int exponent = 0
    if largest == 0.0 or not isfinite(largest):
        return 0
    frexp(largest, &exponent)
    return exponent


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
    cdef double sx, sy
    cdef double mx, my, dx, dy
    cdef double sxx = 0.0, syy = 0.0, sxy = 0.0
    cdef double x0, y0, xmax, ymax, ax, ay
    cdef int x_exponent, y_exponent
    cdef bint x_const = True, y_const = True
    cdef double[::1] x_view = x
    cdef double[::1] y_view = y

    # First pass: track the largest magnitudes (a NaN sticks, as NumPy's
    # ``max`` would make it, and disables the scaling for that input) and
    # detect bit-exact constant inputs.
    # Constancy detection lives here so it agrees with cor_numpy across
    # lengths where the sequential mean would be FP-exact (centred sums
    # zero) vs FP-noisy (centred sums tiny but non-zero); a non-constant
    # FP-noisy input must produce a defined correlation, and a constant
    # input must produce nan regardless of length.
    x0 = x_view[0]
    y0 = y_view[0]
    xmax = fabs(x0)
    ymax = fabs(y0)
    for i in range(1, n):
        ax = fabs(x_view[i])
        ay = fabs(y_view[i])
        if ax > xmax or ax != ax:
            xmax = ax
        if ay > ymax or ay != ay:
            ymax = ay
        if x_view[i] != x0:
            x_const = False
        if y_view[i] != y0:
            y_const = False
    if x_const or y_const:
        # Constant input on at least one side: correlation is mathematically
        # undefined. Match np.corrcoef's FP-exact-constant behaviour
        # (nan + RuntimeWarning). See the docstring of tsecon._stats.cor.
        warnings.warn(
            "invalid value encountered in cor (constant input)",
            RuntimeWarning,
            stacklevel=2,
        )
        return float("nan")
    # Scale every value by 2**-exponent (ldexp, per value) before anything
    # is summed, so the sums, the mean, the centred values and their squares
    # stay in the normal range for any finite input (see "Correlation
    # scaling" in _stats_kernels.py): sums of values near 1e308 no longer
    # overflow, squares of centred values near 1e-158 are no longer
    # subnormal (where the two kernels' summation orders visibly disagreed),
    # and where the unscaled arithmetic stayed in range the result is
    # unchanged. Same order of operations as cor_numpy's scaled np.corrcoef.
    x_exponent = _scale_exponent(xmax)
    y_exponent = _scale_exponent(ymax)
    sx = 0.0
    sy = 0.0
    for i in range(n):
        sx += ldexp(x_view[i], -x_exponent)
        sy += ldexp(y_view[i], -y_exponent)
    mx = sx / n
    my = sy / n
    for i in range(n):
        dx = ldexp(x_view[i], -x_exponent) - mx
        dy = ldexp(y_view[i], -y_exponent) - my
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    # Split the denominator into ``sqrt(sxx) * sqrt(syy)`` so each factor
    # stays in normal range; ``sqrt(sxx * syy)`` underflows to 0 when both
    # sxx and syy fall below ~1e-154 (closes BUGS B4 from session 28).
    return sxy / (sqrt(sxx) * sqrt(syy))
