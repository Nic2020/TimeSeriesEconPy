# SPDX-License-Identifier: MIT
"""DataFrame interop: pandas and polars adapters.

Both libraries are optional extras (install with ``pip install
"TimeSeriesEconPy[pandas]"`` or ``[polars]``). The submodules here
lazy-import the underlying library only when a conversion function is called,
so ``import tsecon.interop`` is cheap and does not require pandas or polars to
be installed.

Public API mirrors the symmetry between the two backends:

* ``to_pandas(obj, *, index=...)`` / ``from_pandas(obj, *, freq=..., wide=...)``
* ``to_polars(obj, *, time_col=...)`` / ``from_polars(obj, *, freq=..., wide=...,
  time_col=...)``

Instance method delegates are attached to :class:`~tsecon.TSeries`,
:class:`~tsecon.MVTSeries`, and :class:`~tsecon.Workspace` at import time.
"""

from __future__ import annotations

from typing import Any

from tsecon.interop.pandas import from_pandas, to_pandas
from tsecon.interop.polars import from_polars, to_polars
from tsecon.mvtseries import MVTSeries
from tsecon.tseries import TSeries
from tsecon.workspace import Workspace

__all__ = ["from_pandas", "from_polars", "to_pandas", "to_polars"]


# ---------------------------------------------------------------------------
# Method bindings
# ---------------------------------------------------------------------------


def _self_to_pandas(self: Any, **kwargs: Any) -> Any:
    """Convert this object to a pandas DataFrame / Series.

    See :func:`tsecon.interop.to_pandas` for the supported kwargs.
    """
    return to_pandas(self, **kwargs)


def _self_to_polars(self: Any, **kwargs: Any) -> Any:
    """Convert this object to a polars DataFrame.

    See :func:`tsecon.interop.to_polars` for the supported kwargs.
    """
    return to_polars(self, **kwargs)


TSeries.to_pandas = _self_to_pandas  # type: ignore[attr-defined]
TSeries.to_polars = _self_to_polars  # type: ignore[attr-defined]
MVTSeries.to_pandas = _self_to_pandas  # type: ignore[attr-defined]
MVTSeries.to_polars = _self_to_polars  # type: ignore[attr-defined]
Workspace.to_pandas = _self_to_pandas  # type: ignore[attr-defined]
Workspace.to_polars = _self_to_polars  # type: ignore[attr-defined]
