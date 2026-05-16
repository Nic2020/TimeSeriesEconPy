# FAQ

## Where does `TimeSeriesEconPy` come from?

It's a Python port of [`TimeSeriesEcon.jl`](https://github.com/bankofcanada/TimeSeriesEcon.jl),
the Bank of Canada's Julia time-series language. The primary motivation is that
the BoC pipeline now spans environments where Julia isn't available
(MS Fabric, Databricks); the port keeps the same vocabulary so models written
against either side translate idiom-for-idiom.

The pinned upstream commit lives in [`tsecon._mirror`](reference/mirror.md).

## How does it compare to pandas?

pandas is a general-purpose DataFrame library. `tsecon` is a *time-series
primitive* library — it stops at `TSeries` / `MVTSeries` / `Workspace` and
leaves higher-level analytics to downstream packages. Concretely:

- `tsecon` has first-class fiscal-year frequencies (`Yearly(end_month=3)`,
  `Quarterly(end_month=2)`); pandas can express these via `Q-FEB` aliases but
  the result is less ergonomic.
- `tsecon` has lossless frequency conversion (`fconvert`) with the same
  method-code semantics as the Julia upstream.
- `tsecon` integrates with pandas (and polars) via lazy-imported converters —
  see [`tsecon.interop`](reference/interop.md). You can move data back and
  forth without `tsecon` becoming a hard dependency of your pandas workflow.

## How does it compare to `TimeSeriesEcon.jl`?

Semantics: identical wherever Python allows. The
[Migration from Julia](design/migration_from_julia.md) page is the one-page
idiom map. The [Cython strategy](design/cython_strategy.md) page shows the
performance picture: for non-vectorisable hot paths we Cythonise and close the
gap to within microbenchmark noise; for already-vectorised paths NumPy is
already at Julia's speed.

## Why no `2020Q1` literal sugar?

See [Frequency model](design/frequency_model.md).

## Where do I file a bug?

[GitHub Issues](https://github.com/Nic2020/TimeSeriesEconPy/issues).

## What's the license?

MIT. See [`LICENSE`](https://github.com/Nic2020/TimeSeriesEconPy/blob/main/LICENSE).
