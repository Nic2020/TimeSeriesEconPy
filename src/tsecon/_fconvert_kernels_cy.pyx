# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
# distutils: language=c
# SPDX-License-Identifier: MIT
"""Cython kernel: aggregate float64 values over contiguous groups.

The compiled sibling of ``_fconvert_kernels.py``'s
:func:`aggregate_groups_numpy`. Both kernels expose an identical
signature and produce identical-modulo-FP-rounding output; the only
difference is execution speed. See the ``.py`` module docstring for
the kernel contract and the "aggregation vs gather vs recurrence"
framing.

The implementation is written in pure C via Cython typed memoryviews —
no per-iteration Python dispatch — so each step compiles to an inner
sequential reduction over the contiguous group window. Frequency-
conversion groups are small (4-12 elements for typical Quarterly→Yearly
or Monthly→Quarterly conversions), so the per-group dispatch overhead
in the NumPy reference dominates the actual numerical work.

Numerical note
--------------
The mean / sum kernels use naive left-to-right accumulation, matching
the Julia upstream. For length-100 input ranges (≤25 output groups)
the results agree with NumPy to within a few ulps; the wrappers use
``np.testing.assert_allclose(rtol=1e-12)``-grade equivalence, not
``assert_array_equal``.

Build
-----
Compiled by ``hatch_build.py`` at wheel build time. The compiled
extension lands at ``src/tsecon/_fconvert_kernels_cy.{pyd,so}`` and is
imported by :mod:`tsecon.fconvert._tseries` as the optional fast path.
When the extension is not built, callers fall back to the NumPy
reference.
"""

import numpy as np
cimport numpy as cnp
cimport cython

cnp.import_array()


# Method-code constants must mirror _fconvert_kernels.py exactly. The
# wrappers pass plain Python ints, so we don't redeclare these as cdef
# enums — the cdef int comparisons in the kernel body are what matters.


@cython.boundscheck(False)
@cython.wraparound(False)
def aggregate_groups_cython(
    cnp.ndarray[cnp.float64_t, ndim=1] values,
    cnp.ndarray[cnp.int64_t, ndim=1] group_starts,
    cnp.ndarray[cnp.int64_t, ndim=1] group_lengths,
    int method_code,
) -> cnp.ndarray:
    """Aggregate ``values`` over ``n_groups`` contiguous groups.

    Cython implementation. See ``_fconvert_kernels.py`` module
    docstring for the contract; the public ``fconvert`` wrappers
    validate inputs before calling this kernel.
    """
    cdef Py_ssize_t n_groups = group_starts.shape[0]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] out = np.empty(n_groups, dtype=np.float64)

    cdef double[::1] vals_view = values
    cdef long long[::1] starts_view = group_starts
    cdef long long[::1] lengths_view = group_lengths
    cdef double[::1] out_view = out

    cdef Py_ssize_t g, i, s, length, end_idx
    cdef double acc, m, x

    if method_code == 0:  # mean
        for g in range(n_groups):
            s = <Py_ssize_t>starts_view[g]
            length = <Py_ssize_t>lengths_view[g]
            acc = 0.0
            for i in range(s, s + length):
                acc += vals_view[i]
            out_view[g] = acc / length
    elif method_code == 1:  # sum
        for g in range(n_groups):
            s = <Py_ssize_t>starts_view[g]
            length = <Py_ssize_t>lengths_view[g]
            acc = 0.0
            for i in range(s, s + length):
                acc += vals_view[i]
            out_view[g] = acc
    elif method_code == 2:  # min
        for g in range(n_groups):
            s = <Py_ssize_t>starts_view[g]
            length = <Py_ssize_t>lengths_view[g]
            m = vals_view[s]
            for i in range(s + 1, s + length):
                x = vals_view[i]
                if x < m:
                    m = x
            out_view[g] = m
    elif method_code == 3:  # max
        for g in range(n_groups):
            s = <Py_ssize_t>starts_view[g]
            length = <Py_ssize_t>lengths_view[g]
            m = vals_view[s]
            for i in range(s + 1, s + length):
                x = vals_view[i]
                if x > m:
                    m = x
            out_view[g] = m
    elif method_code == 4:  # first
        for g in range(n_groups):
            s = <Py_ssize_t>starts_view[g]
            out_view[g] = vals_view[s]
    elif method_code == 5:  # last
        for g in range(n_groups):
            s = <Py_ssize_t>starts_view[g]
            length = <Py_ssize_t>lengths_view[g]
            end_idx = s + length - 1
            out_view[g] = vals_view[end_idx]
    else:
        raise ValueError(f"aggregate_groups_cython: unknown method_code {method_code!r}.")

    return out
