# TimeSeriesEconPy

A time-series language for macroeconomics, ported from
[TimeSeriesEcon.jl](https://github.com/bankofcanada/TimeSeriesEcon.jl)
(Bank of Canada).

> **Status:** Development release. Core time-series operations, plotting and
> table adapters, X-13 integration, and DataEcon file interchange are implemented.
> See the [DataEcon compatibility guide](https://Nic2020.github.io/TimeSeriesEconPy/dataecon/)
> for supported representations and conversion limits.

## Install

Install the latest published development release:

```bash
python -m pip install --upgrade --pre TimeSeriesEconPy
```

The Python import name is `tsecon`. Prebuilt wheels target CPython 3.11, 3.12
and 3.13 on Windows x86-64, Linux x86-64 (glibc 2.28 or newer), and macOS
Apple Silicon (macOS 11 or newer). These wheels bundle the numerical kernels,
X-13 executable and DataEcon runtime; no separate Julia installation is needed.
Source builds require a compiler and additional setup for the native runtimes;
see the [build instructions](https://Nic2020.github.io/TimeSeriesEconPy/dataecon/).

Optional extras:

```bash
pip install "TimeSeriesEconPy[matplotlib]"   # matplotlib plotting backend
pip install "TimeSeriesEconPy[plotly]"       # plotly plotting backend
pip install "TimeSeriesEconPy[pandas]"       # pandas interop
pip install "TimeSeriesEconPy[polars]"       # polars interop
pip install "TimeSeriesEconPy[holidays]"     # country/subdivision BDaily holiday calendars
pip install "TimeSeriesEconPy[all]"          # everything
```

## Goals

1. Mirror TimeSeriesEcon.jl's user-facing concepts (Frequencies, MIT, TSeries,
   MVTSeries, Workspace) with idiomatic Python ergonomics.
2. Run on MS Fabric and Databricks where Julia isn't available.
3. Stay lean enough to ship through enterprise package mirrors (Sonatype
   Nexus) without binary headaches.

## Documentation

Full docs at <https://Nic2020.github.io/TimeSeriesEconPy/>.

[Release notes](https://github.com/Nic2020/TimeSeriesEconPy/blob/main/CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

The original Julia package, [TimeSeriesEcon.jl](https://github.com/bankofcanada/TimeSeriesEcon.jl),
is maintained by the Bank of Canada. TimeSeriesEconPy is an independent port
and is not affiliated with the Bank of Canada.

This Python port was developed in collaboration with Claude Code (Anthropic's
Claude Opus 4.7), used as an interactive programming assistant. Per-commit
AI-assistance attribution is recorded via `Co-Authored-By` trailers in the
git history.
