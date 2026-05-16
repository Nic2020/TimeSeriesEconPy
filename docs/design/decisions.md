# Design notes

`TimeSeriesEconPy` is a port — not a fresh design — of
[`TimeSeriesEcon.jl`](https://github.com/bankofcanada/TimeSeriesEcon.jl). Most
shape decisions therefore start from "what does the Julia upstream do?" and end at
"what is the smallest deviation that's idiomatic in Python?". The notes in this
section document the deviations that *aren't* obvious from reading the code.

The full decision history lives outside this site in
[`claude_files/decisions/`](https://github.com/Nic2020/claude-files-TimeSeriesEconPy)
(a separate private repo). The pages here are a curated, paper-voice cross-section
of those records.

## In this section

- [Frequency model](frequency_model.md) — cached singletons, why no `2020Q1` literal,
  why `MIT != int`.
- [TSeries protocols](tseries_protocols.md) — composition over subclassing,
  `__array_ufunc__` / `__array_function__`, alignment by MIT intersection.
- [Cython strategy](cython_strategy.md) — when we reach for a Cython kernel, the
  three-flavour benchmark, the N=4 operation-shape classification.
- [JSON serialization](serialization.md) — the `_type` discriminator and the
  composite-type round-trip.
- [Migration from Julia](migration_from_julia.md) — the one-page idiom map for
  readers coming from the Julia upstream.

## Locked decisions

| #  | Topic                             | Choice                                                                                          |
|----|-----------------------------------|-------------------------------------------------------------------------------------------------|
| 01 | Acceleration                      | Pure NumPy in M1; Cython for empirically-justified hot paths in M1.5+. Numba ruled out.         |
| 02 | TSeries internals                 | Composition + `__array_ufunc__` / `__array_function__` (xarray-style). No `ndarray` subclassing.|
| 03 | Naming                            | PyPI: `TimeSeriesEconPy`; import: `tsecon`.                                                     |
| 04 | Python floor + license            | Python 3.11+; MIT.                                                                              |
| 05 | DataFrame interop                 | Lazy / optional. Neither pandas nor polars is a hard dep.                                       |
| 06 | Plotting                          | Thin adapter, matplotlib + plotly as optional deps.                                             |
| 07 | v1.0 milestones                   | M1 (core) → M1.5 (Cython) → M2 (X13) → M3 (DataEcon) → M4 (recursive, holidays, helpers).       |
| 08 | Test stack                        | pytest + hypothesis + pytest-benchmark + a side-by-side Julia comparison harness.               |
| 09 | Version mirroring                 | Pinned `MIRRORS_JULIA_SHA` constant + weekly CI diff.                                           |
| 10 | Build system                      | `hatchling` backend + `uv` env/dep manager.                                                     |
| 11 | Docs                              | MkDocs + Material + mkdocstrings + `markdown-exec` (Documenter `@repl` analogue).               |
| 12 | Code quality                      | `ruff` (lint+format) + `mypy --strict` on `src` + pre-commit hooks.                             |
| 15 | Frequency model                   | Cached-singleton frozen dataclasses; constructor functions (`qq` / `mm` / `yy` / …).            |
| 16 | Constructor wrap-vs-copy          | Public constructors **wrap** user-provided containers; `copy=True` opt-in (xarray pattern).     |
| 17 | Cython dispatch                   | Always-fast public API + direct-kernel scenarios in the cross-language benchmark harness.       |
| 18 | Cython port plan (M1.5)           | Four ports landed: `rec_linear`, `indexing`, `stats_scalar`, `fconvert_lower_aggregate`.        |
