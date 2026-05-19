# 4-column comparison harness — tsecon / Julia / pandas / polars

This is the cross-language benchmark suite that satisfies the M1 exit
criteria for empirical performance evidence. It runs the same set of scenarios in
`TimeSeriesEconPy` (Python) and `TimeSeriesEcon.jl` (Julia), and — for the
paper-strategic subset — also in pandas and polars, then writes a
side-by-side comparison table. The pandas / polars columns underwrite the
JSS Section-1 framing ("a time-series language is needed because
mixed-frequency operations are awkward on a DataFrame, and even where they
work the cost is high"); the Julia column underwrites the Section-5
performance claims.

## How it works

Each scenario is defined once per backend, in adjacent files kept
deliberately parallel so a reviewer can diff them by eye:

```
benchmarks/compare/
├── scenarios.py        # tsecon definitions     (SETUP[name], RUN[name])
├── scenarios_pandas.py # pandas definitions     (paper-strategic subset)
├── scenarios_polars.py # polars definitions     (paper-strategic subset)
├── run.py              # Python driver — times all in-process Python
│                       #   backends via `timeit`, invokes the Julia
│                       #   runner via `subprocess`
├── README.md           # this file
├── results/            # date-stamped JSON history
└── julia/
    ├── Project.toml    # pins TimeSeriesEcon + BenchmarkTools + JSON
    ├── scenarios.jl    # Julia definitions (SCENARIOS dict)
    └── runner.jl       # Julia CLI — times one scenario via `BenchmarkTools`
```

`scenarios.py` (tsecon) covers the full 40+ scenario surface; pandas and
polars cover the ~21 paper-strategic scenarios for which a natural
DataFrame idiom exists. Scenarios absent from a backend's registry
appear as `n/a` cells in the comparison table — those `n/a`s are
themselves paper findings, not gaps.

The Python driver invokes Julia per-scenario via `subprocess`. We chose
subprocess over `juliacall` (the in-process bridge) because the harness runs
infrequently and the goal is paper-grade, side-by-side timing — process
isolation is easier to reason about, and we avoid pulling a heavy Julia
runtime into the Python install footprint. The trade-off is per-scenario JIT
warmup, which BenchmarkTools accounts for; the wall-clock cost of starting
Julia (~5 s per scenario at the time of writing) is paid back many times
over by the cleaner CI story.

## Scenarios

The full scenario list (40+ rows) lives in `scenarios.py`; the table
below names the load-bearing rows that drive the paper narrative. The
two **mixed-frequency** rows at the bottom were added 2026-05-16
specifically to expose the friction DataFrame pipelines hit when data
spans multiple frequencies — the Section-1 framing question.

| Name | Description |
|---|---|
| `construct_tseries_qq_100` | `TSeries(qq, arr)` from a length-100 ndarray. |
| `indexing_mit_lookup_100` | Sum `t[mit]` over 100 MIT keys. |
| `arith_add_misaligned` | Add two 100-period TSeries with a 50-period overlap. |
| `shift_quarterly_lag1` | `shift(t, -1)` over a 100-period quarterly TSeries. |
| `moving_average_quarterly_4` | 4-period moving average over the same input. |
| `fconvert_qq_to_yy_mean` | Quarterly → Yearly with `method="mean"`. |
| `rec_ar2_100` | 100-step AR(2) recurrence (the M1.5 Cython candidate). |
| `workspace_merge_5_series` | Merge two Workspaces each holding five TSeries. |
| `mixed_freq_qq_minus_mm_mean` | `qq_gdp − fconvert(Q, mm_cpi, mean)` — single-conversion mixed-freq op. |
| `mixed_freq_pipeline_three_freq` | Y + Q + M → quarterly via two `fconvert` calls — three-frequency pipeline. |

## Running

Run all scenarios (typical use):

```bash
uv run python benchmarks/compare/run.py
```

Run a subset (debugging / focused runs):

```bash
uv run python benchmarks/compare/run.py --only rec_ar2_100,shift_quarterly_lag1
```

Run only the tsecon column (fast smoke; works without Julia / pandas /
polars installed):

```bash
uv run python benchmarks/compare/run.py --python-only
```

Drop pandas / polars columns even when they're installed (e.g. for the
tsecon-vs-Julia-only summary the paper has historically used):

```bash
uv run python benchmarks/compare/run.py --no-pandas --no-polars
```

Increase the per-scenario time budget for stabler numbers (default 2 s):

```bash
uv run python benchmarks/compare/run.py --seconds 5
```

If `julia` is not on `PATH`, or `pandas` / `polars` aren't importable, the
script prints a one-line warning per missing backend and continues with
the remaining columns. Each absent backend's cells read `n/a`. No error.

## First-time setup

Once per machine, instantiate the Julia environment:

```bash
cd benchmarks/compare/julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This downloads BenchmarkTools.jl, JSON.jl, and TimeSeriesEcon.jl and writes
a `Manifest.toml`. The first run after instantiation pays a one-time
precompilation cost (~30 s); subsequent runs are fast.

Python deps come from `pyproject.toml`. `uv sync --all-extras` covers the
4-column harness fully (tsecon + pandas + polars in one shot). If you
only want a subset, `uv sync --extra pandas` / `--extra polars` install
each backend individually.

## Outputs

Each successful run writes two artefacts:

* **Markdown table** to stdout. Use `--markdown PATH` to also write it to a
  file.
* **JSON snapshot** to `results/<UTC-timestamp>.json` with the
  full per-scenario record (median, min, sample count for each language).
  Each scenario has `python` / `julia` / `pandas` / `polars` blocks;
  absent backends serialise as `null`. Use `--no-json` to suppress.

Results are committed to the repo so the history of measurements is
auditable. Treat the JSON files as a time series — the paper draws
comparisons against the date-stamped baseline.

## Reading the numbers

`run.py` reports the **median** as the headline. Both `timeit` and
`BenchmarkTools.@benchmark` also report the minimum, which is what the
microbenchmark literature recommends for noise-resistant estimates (it
filters out the system-scheduling jitter that affects the median). We
include both so a reader can pick whichever statistic they trust.

A Python/Julia ratio of 1.0 means "same speed". Anything > 1.0 means Python
is slower. Ratios > ~50× on a single primitive call are the canonical hot-
path candidates for the Cython port (see the
[Cython strategy](../../docs/design/cython_strategy.md) design note).

## Caveats

* **First-run JIT warmup** — Julia's first scenario in a process pays a
  one-shot JIT cost. `BenchmarkTools` runs a warmup pass internally so this
  doesn't enter the reported time, but it does add to wall-clock duration.
* **GC pauses** — Both timers report the *median* of many samples to filter
  GC noise; for adversarial workloads we'd want `min` instead.
* **Python interpreter loop overhead is the floor** — A pure-Python lambda
  call costs ~50-100 ns regardless of what's inside; for scenarios where
  the underlying op completes in tens of nanoseconds (e.g. integer add)
  Python is fundamentally rate-limited by its own dispatch loop. The
  `rec_ar2_100` scenario is the cleanest illustration of this.
* **Single platform per run.** The harness writes the host platform / Python
  version / Julia version into the JSON, but does not orchestrate
  cross-platform runs. CI matrix work is a future job.
