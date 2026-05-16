# TSeries

One-dimensional time-indexed array. Wraps a NumPy `ndarray` via composition (no
ndarray subclassing — see [design/tseries_protocols.md](../design/tseries_protocols.md))
and implements the `__array_ufunc__` / `__array_function__` protocols so element-wise
operations alignment-merge by `MIT` intersection.

::: tsecon.tseries
