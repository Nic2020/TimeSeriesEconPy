# SPDX-License-Identifier: MIT
"""Vectorised lookup over a TSeries — the array-of-keys companion to ``t[k]``.

TimeSeriesEcon.jl users get fast per-element access for free: Julia's
``for k in keys; s += t[k]; end`` compiles to inlined ``getindex`` calls
and a tight numerical loop, so the per-element pattern is already
competitive. The Python equivalent — ``for k in keys: s += t[k]`` —
pays per-iteration ``__getitem__`` dispatch, MIT object allocation,
and numpy-scalar boxing; the session-16 benchmark clocked this at
**935x slower than Julia** for 100 MIT lookups.

This module provides a public vectorised lookup ::

    >>> from tsecon import TSeries, qq, lookup
    >>> import numpy as np
    >>> t = TSeries(qq(2020, 1), np.arange(100.0))
    >>> keys = [qq(2020, 1) + i for i in range(100)]
    >>> values = lookup(t, keys)
    >>> values.shape
    (100,)

so the user-facing pattern becomes ``lookup(t, keys)`` (returns an
ndarray of values at the requested positions) instead of a per-element
Python loop. The vectorised path runs entirely in NumPy / Cython C
code; the 935x gap is an **API-shape gap**, not a Python-speed gap.

Dispatch
--------
:func:`lookup` validates the keys (frequency-checks MITs, bounds-checks
integer offsets), translates everything to int64 offsets into the
TSeries' underlying ``float64`` buffer, then dispatches to the
:mod:`~tsecon._indexing_kernels_cy` Cython kernel when importable or
:mod:`~tsecon._indexing_kernels`'s NumPy reference otherwise. Both
kernels share the contract ``gather(values, indices) -> out`` and
return identical bit-for-bit output (see the kernels' module
docstrings).

The two kernels' relative speed differs from the ``rec_linear`` story:
the gather is *vectorisable*, so the NumPy reference is already a tight
C loop via :func:`numpy.take`, and the Cython kernel only buys a
marginal further win (~1.5-3x, mostly from removing NumPy's per-call
dispatch overhead). The big win here is exposing the vectorised API at
all — that's where the 935x → ~1x recovery happens. See
``claude_files/decisions/18_cython_port_plan.md`` for the empirical
classification and ``claude_files/paper/NOTES.md`` § "Indexing
kernel" for the JSS-paper framing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from tsecon._indexing_kernels import gather_numpy
from tsecon._kernel_dispatch import _dispatch_kernel, _is_kernel_eligible
from tsecon.mit import MIT
from tsecon.tseries import TSeries

# Try to load the optional Cython-compiled accelerator. When the wheel was
# built without the C toolchain (or for editable installs that skipped the
# build hook), the import fails silently and lookup falls back to the NumPy
# reference. The public surface is otherwise unchanged.
try:
    from tsecon._indexing_kernels_cy import gather_cython  # type: ignore[import-not-found]

    _CYTHON_AVAILABLE: Final[bool] = True
except ImportError:
    _CYTHON_AVAILABLE = False  # type: ignore[misc]
    # Stub matches the cython name so `_dispatch_kernel` resolves cleanly
    # when the extension isn't built (see `_kernel_dispatch.py` docstring).
    gather_cython = gather_numpy

__all__ = ["lookup", "lookup_is_cython"]


def lookup_is_cython() -> bool:
    """Return True iff the Cython-compiled gather kernel was importable.

    Useful for tests, benchmarks, and diagnostic prints — the public
    :func:`lookup` itself is implementation-agnostic. When this returns
    ``False`` the same calls go through the pure-NumPy kernel in
    ``_indexing_kernels.py``; behaviour is identical, only speed
    differs.
    """
    return _CYTHON_AVAILABLE


def lookup(
    t: TSeries,
    keys: Sequence[MIT] | Sequence[int] | npt.ArrayLike,
) -> npt.NDArray[Any]:
    """Return ``t``'s values at every position in ``keys``, vectorised.

    The array-of-keys companion to ``t[k]``: while ``t[k]`` reads a
    single MIT- or int-keyed position, :func:`lookup` reads many at
    once with a single NumPy/Cython gather, avoiding the per-element
    ``__getitem__`` dispatch tax that makes the Python loop pattern
    **935x slower than the Julia equivalent** on the session-16
    benchmark (see ``MASTER_PLAN.md`` § M1.5 and decision 18).

    Parameters
    ----------
    t : TSeries
        The series to gather values from. Not mutated.
    keys : sequence of MIT, sequence of int, or 1-D array-like
        The positions to read. MIT keys are frequency-checked against
        ``t`` and translated to integer offsets into ``t``'s underlying
        values buffer; integer keys are interpreted positionally
        (matching ``t[i]`` semantics — i.e. ``i = 0`` is
        ``t.firstdate``). All keys must use the same form; mixing MITs
        and ints in one call is rejected.

    Returns
    -------
    ndarray
        Freshly allocated 1-D array of the same length as ``keys`` and
        the same dtype as ``t.values``. Independent of the source
        buffer — mutations to the result do not affect ``t``.

    Raises
    ------
    TypeError
        If any MIT key has a frequency different from ``t``'s, or if
        keys are mixed MIT/int, or if a key is neither MIT nor int.
    IndexError
        If any translated offset falls outside ``[0, len(t))``.

    Examples
    --------
    Pick four MIT positions::

        >>> from tsecon import TSeries, qq, lookup
        >>> import numpy as np
        >>> t = TSeries(qq(2020, 1), np.arange(100.0))
        >>> lookup(t, [qq(2020, 1), qq(2020, 3), qq(2022, 1), qq(2024, 4)])
        array([ 0.,  2.,  8., 19.])

    Or pick four integer-offset positions (positional indexing,
    matching ``t[i]``)::

        >>> lookup(t, [0, 5, 10, 99])
        array([ 0.,  5., 10., 99.])

    Notes
    -----
    For a single key, prefer the direct indexer ``t[k]`` — it returns
    a scalar without an array allocation and dispatches through the
    same code path. :func:`lookup` is for the *bulk* case (e.g. 10+
    keys) where vectorisation amortises the per-call setup cost.

    See Also
    --------
    lookup_is_cython : Check whether the compiled kernel is active.
    """
    keys_seq: Sequence[Any]
    if isinstance(keys, np.ndarray):
        if keys.ndim != 1:
            msg = f"lookup: keys array must be 1-D, got ndim={keys.ndim}."
            raise ValueError(msg)
        keys_seq = keys.tolist() if keys.dtype == object else keys  # type: ignore[assignment]
    else:
        keys_seq = list(keys)  # type: ignore[arg-type]

    n = len(keys_seq)
    if n == 0:
        return np.empty(0, dtype=t.values.dtype)

    # Classify by first element; the loop below verifies the rest match.
    first = keys_seq[0]
    if isinstance(first, MIT):
        if first.frequency != t.frequency:
            msg = (
                f"lookup: MIT key frequency {type(first.frequency).__name__} does not "
                f"match TSeries frequency {type(t.frequency).__name__}."
            )
            raise TypeError(msg)
        firstdate_value = t.firstdate.value
        offsets = np.empty(n, dtype=np.int64)
        for i, k in enumerate(keys_seq):
            if not isinstance(k, MIT):
                msg = (
                    f"lookup: keys must be a homogeneous sequence — got MIT at "
                    f"index 0 but {type(k).__name__} at index {i}."
                )
                raise TypeError(msg)
            if k.frequency != t.frequency:
                msg = (
                    f"lookup: MIT key at index {i} has frequency "
                    f"{type(k.frequency).__name__}, expected "
                    f"{type(t.frequency).__name__}."
                )
                raise TypeError(msg)
            offsets[i] = k.value - firstdate_value
    elif isinstance(first, (int, np.integer)) and not isinstance(first, bool):
        # Integer keys are positional offsets into t.values, matching t[i] semantics.
        try:
            offsets = np.asarray(keys_seq, dtype=np.int64)
        except (TypeError, ValueError) as exc:
            msg = (
                "lookup: integer keys must be a homogeneous sequence of ints "
                "(no MITs, no floats, no None)."
            )
            raise TypeError(msg) from exc
        if offsets.ndim != 1:
            msg = "lookup: integer keys must form a 1-D sequence."
            raise ValueError(msg)
    else:
        msg = f"lookup: keys must be MIT or int objects, got {type(first).__name__} at index 0."
        raise TypeError(msg)

    # Bounds check the offsets against t's storage.
    length = len(t.values)
    if offsets.size > 0:
        lo = int(offsets.min())
        hi = int(offsets.max())
        if lo < 0 or hi >= length:
            msg = (
                f"lookup: offsets out of range [0, {length}) — got min={lo}, max={hi}. "
                f"TSeries range is {t.range!s}."
            )
            raise IndexError(msg)

    values = t.values
    if values.dtype != np.float64:
        # Non-float64 dtypes go through NumPy's native fancy indexing — the
        # kernels are float64-specialised for the hot path; other dtypes are
        # uncommon enough that the per-call overhead doesn't justify a
        # parallel kernel family.
        return values.take(offsets)
    if _is_kernel_eligible(values):
        return _dispatch_kernel(  # type: ignore[no-any-return]
            _CYTHON_AVAILABLE, gather_cython, gather_numpy, values, offsets
        )
    # float64 but non-contiguous (e.g. a sliced ``arr[::2]`` view): the
    # Cython kernel would raise BufferError on its typed-memoryview cast,
    # so route to the NumPy reference (which uses ``np.take`` and handles
    # any stride pattern). See F11 review file.
    return gather_numpy(values, offsets)
