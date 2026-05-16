# Statistics

`mean` / `var` / `std` / `median` / `quantile` / `stdm` / `varm` / `cor` / `cov`
work on both `TSeries` and `MVTSeries` (the matrix variants iterate the values flat,
matching Julia). `mean` / `var` / `std` / `cor` route through a Cython kernel when
the compiled extension is importable — see
[design/cython_strategy.md](../design/cython_strategy.md) and `stats_is_cython()`.

::: tsecon._stats
