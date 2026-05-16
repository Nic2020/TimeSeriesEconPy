# DataFrame interop (pandas, polars)

Lazy-imported converters between `TSeries` / `MVTSeries` / `Workspace` and
`pandas.DataFrame` / `polars.DataFrame`. Neither library is a hard dependency;
install with the optional extra:

```bash
pip install 'TimeSeriesEconPy[pandas]'   # or [polars], or [all]
```

The pandas surface accepts `index="auto" | "mit" | "date"`; the polars surface
keeps the time axis as a column (polars has no row index) and requires explicit
`freq=` on `from_polars(date_column=...)` since polars carries no period semantics.

## tsecon.interop.pandas

::: tsecon.interop.pandas

## tsecon.interop.polars

::: tsecon.interop.polars
