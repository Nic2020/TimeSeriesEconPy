# MVTSeries

Two-dimensional time-indexed table — rows are `MIT`-keyed, columns are string-named.
Storage is a 2-D `ndarray` plus a cached `dict[str, TSeries]` of column views, so
`mvts.a[date] = v` writes through to the underlying matrix.

::: tsecon.mvtseries
