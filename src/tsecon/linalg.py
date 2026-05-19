# SPDX-License-Identifier: MIT
r"""Matrix multiplication on TSeries / MVTSeries.

Mirrors ``TimeSeriesEcon.jl/src/linalg.jl`` exactly. The Julia module
overloads ``*``, ``\`` and ``/`` for every combination of
``TSeries`` / ``MVTSeries`` / ``AbstractMatrix``; every method body is
of the shape ``op(_vals(A), _vals(B))`` — i.e. the operation strips
frequency / range / column-name labels and returns the underlying
``Vector`` or ``Matrix`` (the upstream test
``x * x3 == _vals(x) * _vals(x3)`` confirms). The Python port keeps
that semantics: ``@`` returns a plain :class:`numpy.ndarray`; users
who want the result back as a labelled object wrap it explicitly.

Spelling — ``@`` (PEP 465) over Julia's ``*``. Element-wise ``*`` is
how NumPy / pandas / xarray spell broadcasting, and the existing
``__mul__`` / ``__array_ufunc__`` machinery already covers it (range
intersection, frequency check, etc.). Overloading ``*`` for the
matrix product would break that contract, so the dunders use ``@``
exclusively; the Julia ``A * t`` idiom ports as ``A @ t`` (see
``docs/design/migration_from_julia.md``).

Not ported:

* ``transpose`` / ``adjoint``. Julia's overloads also strip labels
  (``transpose(t)`` returns a 1×N row vector; ``adjoint(mvts)`` a
  ``k × n`` matrix). A ``.T`` property would have no clean semantics
  on TSeries / MVTSeries — the row axis is *time*, not data, so a
  transposed object has no natural type to wrap into. Users wanting
  the bare transpose write ``np.asarray(t).T``. Recorded as an
  intentional non-port.
* ``\`` / ``/`` (linear-solve overloads). ``@`` covers the
  coefficient-matrix-times-series case that motivates the module;
  callers needing a solve use ``np.linalg.solve(A, np.asarray(t))``
  directly. Promote on demand.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _matmul_strip(left: Any, right: Any) -> Any:
    """Matrix multiply with all TSeries / MVTSeries labels stripped.

    Implements ``Base.:(*)(A, B) = *(_vals(A), _vals(B))`` from Julia's
    ``linalg.jl``. :func:`numpy.asarray` invokes the
    :meth:`~tsecon.tseries.TSeries.__array__` /
    :meth:`~tsecon.mvtseries.MVTSeries.__array__` overrides so the
    frequency / range / column-name labels are dropped uniformly.

    NumPy raises :class:`ValueError` on shape mismatch and
    :class:`TypeError` on unsupported operand types; we let those
    propagate.
    """
    return np.matmul(np.asarray(left), np.asarray(right))
