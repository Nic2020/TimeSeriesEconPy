# SPDX-License-Identifier: MIT
"""Shared Cython-vs-NumPy dispatch helpers for the four M1.5 kernels.

Codifies the canonical pattern for the four ports landed in sessions
17-21 (``rec_linear`` / ``lookup`` / ``stats`` / ``fconvert``) and gives
future M2/M3 kernel ports a single import to depend on. See
``claude_files/decisions/20_kernel_dispatch_template.md`` for the
rationale and the call-site recipe.

The pattern has two pieces:

1. :func:`_is_kernel_eligible` — returns ``True`` iff every supplied
   array is ``float64`` AND ``C_CONTIGUOUS``. This is the kernel
   fast-path contract; the four M1.5 kernels share it because all four
   ``.pyx`` extensions were compiled with typed memoryviews specialised
   for that shape (a non-``float64`` or non-contiguous array would
   either raise ``BufferError`` or silently misinterpret the buffer).

2. :func:`_dispatch_kernel` — picks the Cython kernel when its
   module-local availability flag is ``True``, else the NumPy
   reference. Each kernel module keeps its own ``_CYTHON_AVAILABLE``
   constant (the four extensions can in principle be built
   independently, even if in practice they ship together via the
   single ``hatch_build.py`` hook) and passes it in; the helper itself
   stays pure.

Call-site recipe (mirrors :mod:`tsecon._stats`):

.. code-block:: python

    flat = _ravel_for_kernel(values)
    if flat is not None and flat.shape[0] > 0:
        return _dispatch_kernel(_CYTHON_AVAILABLE, mean_cython, mean_numpy, flat)
    return np.mean(values)

For the multi-array kernels (e.g. ``cor``, ``fconvert.aggregate_groups``)
the eligibility check is the explicit guard:

.. code-block:: python

    if _is_kernel_eligible(xv, yv):
        return _dispatch_kernel(_CYTHON_AVAILABLE, cor_cython, cor_numpy, xv, yv)
    return float(np.corrcoef(xv, yv)[0, 1])

The two helpers are kept separate (rather than fused into a single
``_dispatch_kernel(eligibility_arrays=..., args=...)`` shape) because
the kernels have heterogeneous signatures (in-place mutation, scalar
return, ndarray return; 2 to 5 args) and the call-site readability
suffers when both checks are wedged into one function call.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["_dispatch_kernel", "_is_kernel_eligible"]


def _is_kernel_eligible(*arrays: np.ndarray) -> bool:
    """Return ``True`` iff every array is ``float64`` AND ``C_CONTIGUOUS``.

    The canonical fast-path contract for the four M1.5 Cython kernels.
    Combine with each kernel module's ``_CYTHON_AVAILABLE`` flag at the
    call site (typically by chaining this check with
    :func:`_dispatch_kernel`).

    Parameters
    ----------
    *arrays : np.ndarray
        One or more arrays the candidate Cython call would read. Every
        array must satisfy both conditions for the fast path to be
        eligible; mismatches return ``False`` and the caller takes the
        NumPy fallback.

    Returns
    -------
    bool
        ``True`` iff every ``a`` in ``arrays`` has ``a.dtype ==
        np.float64`` and ``a.flags["C_CONTIGUOUS"]``.
    """
    return all(a.dtype == np.float64 and a.flags["C_CONTIGUOUS"] for a in arrays)


def _dispatch_kernel(
    cython_available: bool,
    cython_fn: Callable[..., Any],
    numpy_fn: Callable[..., Any],
    /,
    *args: Any,
) -> Any:
    """Call ``cython_fn(*args)`` when ``cython_available`` is ``True``, else ``numpy_fn(*args)``.

    Caller is responsible for pre-validating eligibility (typically via
    :func:`_is_kernel_eligible` or a kernel-specific shim like
    ``_ravel_for_kernel``); this helper only picks between the two
    pre-validated kernel implementations.

    Parameters
    ----------
    cython_available : bool
        The kernel module's ``_CYTHON_AVAILABLE`` flag. Passed in (rather
        than imported here) so the helper stays agnostic of which kernel
        module the call sits in.
    cython_fn : callable
        The Cython-compiled kernel.
    numpy_fn : callable
        The NumPy reference sibling. Must have an identical signature to
        ``cython_fn`` and return a bit-for-bit-equivalent result.
    *args : Any
        Forwarded verbatim to the chosen kernel.

    Returns
    -------
    Any
        Whatever the chosen kernel returns. The helper is signature-
        agnostic; the kernel may return a scalar, an ``np.ndarray``, or
        nothing (in-place mutation).
    """
    return (cython_fn if cython_available else numpy_fn)(*args)
