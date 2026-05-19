# Linear algebra (`@` matrix multiply)

`TSeries` and `MVTSeries` overload Python's `@` operator (PEP 465) so the
upstream `TimeSeriesEcon.jl` idiom

```julia
A * t        # Julia: A is a Matrix, t is a TSeries
```

ports as

```python
A @ t        # Python: A is a numpy.ndarray, t is a tsecon.TSeries
```

Both directions are supported (`A @ t`, `t @ A`, `mvts @ A`, `A @ mvts`,
`t @ t`, `mvts @ mvts`).

## What `@` returns

A plain [`numpy.ndarray`][]. **Frequency / range / column-name labels are
stripped.** This matches the Julia upstream exactly: every method body in
`linalg.jl` is of the shape `op(_vals(A), _vals(B))`, returning a bare
`Vector` or `Matrix`. The upstream test
[`x * x3 == _vals(x) * _vals(x3)`](https://github.com/bankofcanada/TimeSeriesEcon.jl/blob/master/test/test_various.jl)
documents the same contract.

If you want a labelled result, wrap explicitly:

```python
out_vals = A @ t                                  # numpy.ndarray
out_t = TSeries(t.range, out_vals)                # back to a labelled TSeries
```

Element-wise multiplication keeps its existing semantics — `*` between two
TSeries still does the range-intersection, frequency-checked broadcast
(see [TSeries reference](tseries.md)).

## What's intentionally not ported

* **`transpose` / `adjoint`.** Julia's overloads return a 1×N row vector
  (from a `TSeries`) or a `k × n` matrix (from an `MVTSeries`) with all
  labels stripped. A `.T` property on `TSeries` / `MVTSeries` would have no
  clean semantics in Python — the row axis of an `MVTSeries` is *time*,
  not data, so a transposed object has no natural type to wrap. Users
  wanting the bare transpose write `np.asarray(x).T`.
* **`\` / `/` (linear-solve).** `@` covers the common
  `A @ t` coefficient-matrix case. Callers needing a solve use
  [`numpy.linalg.solve`][] directly:
  `numpy.linalg.solve(A, np.asarray(t))`.

## Module reference

::: tsecon.linalg
