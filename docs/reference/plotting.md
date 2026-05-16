# Plotting

`tsecon.plot(*series, backend="auto" | "matplotlib" | "plotly", ...)` plus
`TSeries.plot(...)` / `MVTSeries.plot(...)` method delegates. Both backends are
optional install extras; the dispatcher lazy-imports via `importlib.util.find_spec`
and raises `BackendNotAvailableError` with an install hint when neither is present.

```bash
pip install 'TimeSeriesEconPy[matplotlib]'   # or [plotly], or [all]
```

::: tsecon.plotting
