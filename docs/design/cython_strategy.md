# Cython strategy

!!! info "Work in progress"
    This page is a stub. The full thesis arrives in the JSS paper.
    Decision records: `claude_files/decisions/01_acceleration_strategy.md`,
    `17_cython_dispatch_strategy.md`, `18_cython_port_plan.md`.

## The rule

We reach for a Cython kernel **only when an empirical benchmark says NumPy alone
isn't enough**. The cross-language harness in `benchmarks/compare/` is the
gatekeeper: every hot path runs in three flavours — pure Python (or NumPy + a
Python loop), Cython-compiled, and Julia — and we Cythonise the path only if the
NumPy column lags Julia by more than the harness's measurement noise floor.

Reasons:

- The Python ecosystem has no shortage of "I assumed I needed Cython, then
  measured and didn't" stories. We don't want to be one.
- Each Cython kernel adds a compiled extension, which adds a per-platform wheel
  to build and ship. Reading the existing
  [`wheels.yml`](https://github.com/Nic2020/TimeSeriesEconPy/blob/main/.github/workflows/wheels.yml)
  is one good way to internalise how much that costs.
- The paper's central empirical contribution is the *classification* below, and
  the classification is only meaningful if every cell in it was measured.

## The N=4 operation-shape classification

Four kernels are shipped in `M1.5`. They line up with four structurally distinct
operation shapes:

| Operation shape                         | Example                  | Cython-over-NumPy speedup |
|-----------------------------------------|--------------------------|---------------------------|
| Non-vectorisable recurrence             | `rec_linear` (AR(p))     | ~65×                      |
| Outer loop of per-group C reductions    | `fconvert_*_aggregate`   | ~25–80×                   |
| Scalar reduction over a 1-D array       | `mean` / `var` / `cor`   | ~8–40×                    |
| Already-vectorised gather               | `lookup` (`np.take`)     | ~1.1×                     |

The big takeaway: **the Python ↔ Julia gap is operation-shaped, not language-shaped.**
For "already-vectorised" operations, Python is *already* at Julia's speed because
both end up calling the same BLAS/LAPACK or hand-tuned C routine. For tight
non-vectorisable recurrences, Python needs Cython to close the gap, and once it
does, the residual gap is within the microbenchmark noise floor on Julia itself.

## The introspection hooks

Every Cython port ships an `<op>_is_cython() -> bool` helper, re-exported at top
level. Public code that wants to assert "we're running the fast path" calls
those helpers. They are also the per-cell smoke-test that the wheel CI matrix
runs against each built artifact, so a `.pyx` that silently fails to compile in
a wheel cell is caught before publish.

## The three-flavour benchmark

`benchmarks/compare/` runs each scenario in three flavours and emits a JSON
record under `benchmarks/compare/results/<UTC-timestamp>_<commit>.json`. The
JSS paper draws on these records as the empirical backbone of Section 5.
