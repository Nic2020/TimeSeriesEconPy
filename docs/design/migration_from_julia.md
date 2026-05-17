# Migration from Julia

!!! info "Work in progress"
    This page is a stub. The exhaustive idiom table lives in
    `claude_files/docs/TUTORIAL_1_PORT_MAP.md` and will be folded in here once
    the tutorial port has settled.

A reader coming from
[`TimeSeriesEcon.jl`](https://github.com/bankofcanada/TimeSeriesEcon.jl) will
recognise every concept in `tsecon`. The list below catalogues the *visible*
differences — what the spellings, not the semantics, look like.

| Concept                            | Julia                              | Python (`tsecon`)                       |
|------------------------------------|------------------------------------|-----------------------------------------|
| Quarterly literal                  | `2020Q1`                           | `qq(2020, 1)`                           |
| Monthly literal                    | `2020M3`                           | `mm(2020, 3)`                           |
| Yearly literal                     | `2020Y`                            | `yy(2020)`                              |
| Inclusive range                    | `2020Q1:2021Q4`                    | `MITRange(qq(2020, 1), qq(2021, 4))`    |
| Frequency-of                       | `frequencyof(t)`                   | `frequency_of(t)`                       |
| In-place op                        | `lag!(x)` / `lead!(x)`             | `lag_inplace(x)` / `lead_inplace(x)`    |
| Whole-object arithmetic            | `x + y` vs `x .+ y`                | `x + y` (one spelling — always element-wise) |
| Dot-broadcast                      | `log.(x)`                          | `np.log(x)`                             |
| Macro recurrence                   | `@rec` / `@rec(rng, expr)`         | `tsecon.rec(rng, target, fn)`           |
| Linear recurrence                  | `@rec`-via-AR(p) idiom             | `tsecon.rec_linear(target, coeffs, lags, rng)` |
| Workspace deletion                 | `delete!(w, :start)`               | `del w.start`                           |
| Range step                         | `2000M1:2:2000M8`                  | `MITRange(mm(2000, 1), mm(2000, 8), step=2)` |
| Reversed range (backcast)          | `10U:-1:1U`                        | `MITRange(MIT(Unit(),10), MIT(Unit(),1), step=-1)` |
| Last element                       | `x[end]`                           | `x[x.lastdate]`                         |
| `overlay` (TSeries first-non-NaN wins) | `overlay(x1, x2, x3)`           | `overlay(x1, x2, x3)`                   |
| `overlay` (forced range)           | `overlay(2020Q1:2020Q4, x1, x2)`   | `overlay(x1, x2, rng=MITRange(qq(2020,1), qq(2020,4)))` |
| `compare` / `@compare`             | `@compare(v1, v2, atol=1e-5)`      | `compare(v1, v2, atol=1e-5)` (returns `CompareResult`) |
| `reindex` (label shift)            | `reindex(t, 2021Q1 => 1U)`         | `reindex(t, (qq(2021, 1), MIT(Unit(), 1)))` |
| `rangeof` with `drop=`             | `rangeof(t, drop=1)`               | `rangeof(t, drop=1)`                    |
| `rangeof(workspace; method=)`      | `rangeof(w, method=union)`         | `rangeof(w, method="union")`            |

## Semantics that are identical

- MIT intersection on element-wise ops (`x + y`).
- Resize-on-assign when an out-of-range MIT key is written to.
- Frequency-mismatch raises on arithmetic between mismatched-frequency series.
- `fconvert(t, target_freq, method=…)` round-trip behaviour (mean / sum / first /
  last / min / max).

## Semantics that subtly differ

- **`MIT == int` is `False`.** Use `int(mit)` to extract the underlying offset.
- **Default container fill is NaN, not uninitialised.** `TSeries(rng)` is filled
  with NaN; Julia's `TSeries(rng)` leaves the buffer at whatever was on the heap.
- **`copy=False` is the default constructor mode**, matching xarray and matching
  the Julia upstream's pass-the-array-by-reference behaviour. Reach for
  `copy=True` (or `.copy()`) for an alias break.
- **Recurrences split into two entry points.** Julia's `@rec` is a single
  parse-time macro that accepts any RHS expression. Python lacks parse-time
  macros, so `tsecon` ships two entry points: `rec(rng, target, fn)` for the
  general higher-order form (any nonlinear / multi-series body, expressed as
  a `lambda`), and `rec_linear(target, coeffs, lags, rng)` for the closed-form
  pure-linear-AR(p) narrowing (`target[t] = Σ_k coeffs[k] * target[t - lags[k]]`,
  no constant term, no exogenous reads). The narrowing is what enables the
  Cython-backed kernel path; for `target[t] = target[t-1] + c` reach for
  `undiff` instead, for `target[t] = β·target[t-1] + γ·y[t]` reach for the
  general `rec`.
- **`compare` returns a `CompareResult`, not a `Bool`.** Julia's
  `compare(v1, v2)` prints to stdout and returns `Bool`; the Python form
  returns a structured `CompareResult(equal, differences)` that is truthy
  on equality, whose `__str__` reproduces the printed diff, and whose
  `.differences` exposes one entry per leaf for callers that want to
  introspect rather than re-parse stdout. The classic `if compare(v1, v2):
  ...` one-liner still works; `quiet=True` suppresses the stdout side
  effect when only the structured result is needed (e.g. inside a test
  runner). Julia's `@compare` macro folds into the same `compare`
  function: it existed only to capture variable names for the printed
  diff, which Python users can supply directly via `left=` / `right=`.
- **`reindex` takes a 2-tuple, not a `Pair`.** Julia's `Pair{<:MIT,<:MIT}`
  becomes a Python `(old_mit, new_mit)` 2-tuple. Dispatch over MIT /
  MITRange / TSeries / MVTSeries / Workspace is identical; the `copy=`
  kwarg is the same (default `False`, wrap-by-default).
