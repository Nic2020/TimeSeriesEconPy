# TimeSeriesEconPy

A time-series language for macroeconomics, ported from
[TimeSeriesEcon.jl](https://github.com/bankofcanada/TimeSeriesEcon.jl)
(Bank of Canada).

> **Status:** Pre-alpha. No public API yet. M0 (repo skeleton) is in progress.
> Not ready for use.

## Install (when released)

```bash
pip install TimeSeriesEconPy
```

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
