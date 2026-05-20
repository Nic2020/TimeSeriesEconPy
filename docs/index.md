# TimeSeriesEconPy

A time-series language for macroeconomics, ported from
[TimeSeriesEcon.jl](https://github.com/bankofcanada/TimeSeriesEcon.jl) (Bank of
Canada).

The primary motivation is environments where Julia isn't available
(MS Fabric, Databricks). The port keeps the same vocabulary — `Frequency`,
`MIT`, `TSeries`, `MVTSeries`, `Workspace` — so models translate idiom-for-idiom.

## Install

```bash
pip install TimeSeriesEconPy            # core
pip install 'TimeSeriesEconPy[all]'     # core + matplotlib + plotly + pandas + polars
```

Python 3.11 or newer. The base install ships compiled Cython kernels via
per-platform wheels (Linux x86_64 / Windows AMD64 / macOS arm64); the pure-Python
fallback works in any environment but the four kernel-backed paths
(`rec_linear` / `lookup` / `mean·std·cor` / `fconvert_*_aggregate`) run faster
with the wheel.

## First TSeries

```python exec="true" source="material-block" session="index" result="text"
import numpy as np

import tsecon
from tsecon import qq, TSeries

t = TSeries(qq(2020, 1), np.array([100.0, 101.2, 102.3, 103.5]))
print(t)
print("mean:", tsecon.mean(t))
print("first date:", t.firstdate)
```

## What's inside

- **[Tutorials](tutorials/1_timeseriesecon.md)** — narrative ports of the upstream
  Julia tutorial corpus.
- **[Reference](reference/frequencies.md)** — auto-generated API documentation.
- **[Design notes](design/decisions.md)** — the deviations from the Julia
  upstream that aren't obvious from reading the code, including the
  [Cython strategy](design/cython_strategy.md) and the N=4 operation-shape
  classification that's the central empirical contribution of the JSS paper.
- **[API index](api_index.md)** — flat listing of every public symbol.

## Quick links

- [GitHub repository](https://github.com/Nic2020/TimeSeriesEconPy)
- [Issues](https://github.com/Nic2020/TimeSeriesEconPy/issues)
- [TimeSeriesEcon.jl (Julia upstream)](https://github.com/bankofcanada/TimeSeriesEcon.jl)
- [Migration from Julia](design/migration_from_julia.md)
