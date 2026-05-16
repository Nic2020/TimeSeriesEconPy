# Tutorial 1 — TimeSeriesEcon

!!! warning "Tutorial in progress"
    This page is the scaffold for the section-by-section port of
    [TutorialsEcon.jl/1.TimeSeriesEcon](https://bankofcanada.github.io/DocsEcon.jl/dev/Tutorials/1.TimeSeriesEcon/main/).
    The narrative lands in a follow-up session — see the
    [section status table](#section-status) below for what's queued. Until then,
    use the [reference pages](../reference/frequencies.md) for the API surface
    and the [migration guide](../design/migration_from_julia.md) for the
    Julia ↔ Python idiom map.

This tutorial ports the upstream Julia tutorial section-by-section. Each section
shows the Python idiom only — the Julia source is linked once per section in a
side-by-side admonition so a reader migrating from Julia can find the original.
Code blocks are executed at `mkdocs build` time via `markdown-exec` (the same
trick `Documenter.jl` plays with `@repl tse`), so a broken example fails CI.

## Section status

| #  | Section                          | Status | Notes                                                                |
|----|----------------------------------|:------:|----------------------------------------------------------------------|
|  1 | Frequency and Time               | ⬜     | Queued. Surface ready.                                               |
|  2 | Ranges (`MITRange`)              | ⬜     | Queued.                                                              |
|  3 | TSeries — Creation               | ⬜     | Queued. Needs the wrap-vs-copy call-out per [decision 16](../design/decisions.md#locked-decisions). |
|  4 | TSeries — Access (read / write)  | ⬜     | Queued.                                                              |
|  5 | Arithmetic with TSeries          | ⬜     | Queued.                                                              |
|  6 | Shifts (lag / lead)              | ⬜     | Queued.                                                              |
|  7 | Diff and undiff                  | ⬜     | Queued — both `diff` and `undiff` are ported.                        |
|  8 | Moving average                   | ⬜     | Queued — `moving` / `moving_average` / `moving_sum` are ported.      |
|  9 | Recursive assignments            | ⬜     | Queued — `tsecon.rec(rng, target, fn)` plus `tsecon.rec_linear(...)` for the AR(p) common case. |
| 10 | Multi-variate Time Series        | ⬜     | Queued — `MVTSeries` is fully ported.                                |
| 11 | Plotting                         | ⬜     | Queued — matplotlib will be the default backend on this page.        |
| 12 | Workspaces                       | ⬜     | Queued.                                                              |
| 13 | MVTSeries vs Workspace           | ⬜     | Queued.                                                              |
| 14 | `overlay`                        | 🔵    | Blocked on `various.jl` (M4).                                        |
| 15 | `compare` / `@compare`           | 🔵    | Blocked on `various.jl` (M4).                                        |
| 16 | BDaily holidays                  | 🟡    | Options are ported; `holidays` integration is M4.                    |
| 17 | Options                          | ⬜     | Queued — `getoption` / `setoption` / `option_scope` are ported.      |

Legend: ⬜ ready to port (no library blocker) · 🟡 partial · 🔵 blocked on later milestone.

When a section ships, its row moves to a heading on this page with executable
code blocks (sharing the `session="tse"` state) and a Julia ↔ Python admonition
at the bottom.
