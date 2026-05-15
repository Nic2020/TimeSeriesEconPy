# Julia ↔ Python comparison harness

This is the cross-language benchmark suite called out in
[decision 08](../../../claude_files/decisions/08_test_and_benchmark_stack.md)
and the M1 exit criteria. It runs the same set of scenarios in both
`TimeSeriesEconPy` (Python) and `TimeSeriesEcon.jl` (Julia) and writes a
side-by-side comparison table.

## How it works

Each scenario is defined twice — once in `scenarios.py` and once in
`julia/scenarios.jl`. The two files are intentionally kept side-by-side so
that diffing them by eye verifies semantic parity:

```
benchmarks/compare/
├── scenarios.py        # Python definitions (SETUP[name], RUN[name])
├── run.py              # Python driver — times Python via `timeit`,
│                       #   invokes the Julia runner via `subprocess`
├── README.md           # this file
├── results/            # date-stamped JSON history
└── julia/
    ├── Project.toml    # pins TimeSeriesEcon + BenchmarkTools + JSON
    ├── scenarios.jl    # Julia definitions (SCENARIOS dict)
    └── runner.jl       # Julia CLI — times one scenario via `BenchmarkTools`
```

The Python driver invokes Julia per-scenario via `subprocess`. We chose
subprocess over `juliacall` (the in-process bridge) because the harness runs
infrequently and the goal is paper-grade, side-by-side timing — process
isolation is easier to reason about, and we avoid pulling a heavy Julia
runtime into the Python install footprint. The trade-off is per-scenario JIT
warmup, which BenchmarkTools accounts for; the wall-clock cost of starting
Julia (~5 s per scenario at the time of writing) is paid back many times
over by the cleaner CI story.

## Scenarios

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

## Running

Run all scenarios (typical use):

```bash
uv run python benchmarks/compare/run.py
```

Run a subset (debugging / focused runs):

```bash
uv run python benchmarks/compare/run.py --only rec_ar2_100,shift_quarterly_lag1
```

Run only the Python column (fast smoke; works without Julia installed):

```bash
uv run python benchmarks/compare/run.py --python-only
```

Increase the per-scenario time budget for stabler numbers (default 2 s):

```bash
uv run python benchmarks/compare/run.py --seconds 5
```

If `julia` is not on `PATH`, the script prints a one-line warning and runs
only the Python column. The Julia column then reads `n/a`. No error.

## First-time setup

Once per machine, instantiate the Julia environment:

```bash
cd benchmarks/compare/julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

This downloads BenchmarkTools.jl, JSON.jl, and TimeSeriesEcon.jl and writes
a `Manifest.toml`. The first run after instantiation pays a one-time
precompilation cost (~30 s); subsequent runs are fast.

Python deps come from `pyproject.toml` (`uv sync --all-extras` covers it).

## Outputs

Each successful run writes two artefacts:

* **Markdown table** to stdout. Use `--markdown PATH` to also write it to a
  file.
* **JSON snapshot** to `results/<UTC-timestamp>_<short-sha>.json` with the
  full per-scenario record (median, min, sample count for each language).
  Use `--no-json` to suppress.

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
path candidates for the M1.5 Cython port (see
[decision 01](../../../claude_files/decisions/01_acceleration_strategy.md)).

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
