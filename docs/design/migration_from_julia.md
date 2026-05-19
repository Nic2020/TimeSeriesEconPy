# Migration from Julia

!!! info "Work in progress"
    This page is a stub. An exhaustive idiom table will be folded in here
    once the tutorial port has settled.

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
| `TSeries(rng, ini::Function)`      | `TSeries(rng, zeros)` / `TSeries(rng, rand)` | `TSeries(rng, np.zeros)` / `TSeries(rng, rng_np.random)` |
| `mean(mvts; dims=1)` (per-column)  | `mean(mvts; dims=1)`               | `mean(mvts, axis=0)` (NumPy convention) |
| `mean(mvts; dims=2)` (per-row)     | `mean(mvts; dims=2)`               | `mean(mvts, axis=1)`                    |
| Matrix product (coefficient-matrix × series) | `A * t`                  | `A @ t` (PEP 465; returns `ndarray`)    |
| TSeries-of-vectors dot             | `_vals(t1) * _vals(t2)` (raises in Julia) | `t1 @ t2` (inner product scalar) |
| Transpose / adjoint                | `transpose(t)` / `adjoint(mvts)`   | *not ported* — use `np.asarray(x).T`    |
| In-place copy Workspace → MVTSeries | `copyto!(mvts, w; verbose=, trange=)` | `tsecon.copyto(mvts, w, *, verbose=, trange=)` |
| `@weval` macro (Workspace-scoped eval) | `@weval w b + a`                  | `w.b + w.a` *(preferred)* or `with w.as_namespace() as ns: ns.b + ns.a` |

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
- **`Statistics.*` axis kwarg is `axis=`, not `dims=`.** Julia's
  `mean(mvts; dims=1)` reduces along rows (per-column); Python uses NumPy's
  `axis=0` for the same operation. `dims=2` (per-row, Julia) maps to
  `axis=1` (per-row, Python). `axis=None` (default) reduces flat to a
  scalar — the existing behaviour. Returns: `axis=0` → single-row
  MVTSeries (column names preserved); `axis=1` → 1-D TSeries indexed by
  the input range. The convention switch is the same "Python idiom wins
  for kwargs that are language-conventional" rule used elsewhere in this
  table; the underlying behaviour matches Julia.
- **Matrix multiply uses `@`, not `*`.** Julia's `linalg.jl` overloads `*`
  for matrix-times-TSeries / -MVTSeries; the Python port uses PEP 465's
  `@` instead, leaving element-wise `*` to mean element-wise (which is
  how NumPy / pandas / xarray spell it). Both behave like Julia
  *otherwise*: the result is a plain `numpy.ndarray` with frequency /
  range / column-name labels stripped (Julia's overloads forward to
  `*(_vals(A), _vals(B))` and return a bare `Vector` / `Matrix`; the
  test `x * x3 == _vals(x) * _vals(x3)` upstream documents the same).
  `transpose` / `adjoint` are not ported — neither has clean semantics
  on a TSeries / MVTSeries whose row axis is time. See
  [`reference/linalg.md`](../reference/linalg.md). One related divergence
  worth flagging: in Julia, `Vector * Vector` raises a `MethodError`,
  so `_vals(t1) * _vals(t2)` doesn't actually work even though the
  overload is defined; in Python, `t1 @ t2` returns the inner product
  (NumPy 1-D matmul semantics) — the Python form is *more* useful, not
  less.
- **`copyto` is a free function, not a method or a `!`-suffixed mutator.**
  Julia spells the in-place materialiser `Base.copyto!(mvts, w; verbose=,
  trange=)` — the `!` suffix is the Julia convention for an in-place
  mutator. Python has no syntactic mutation marker; `tsecon` exports
  `copyto(dst: MVTSeries, src: Workspace, *, verbose=False, trange=None)`
  as a free function, mirroring NumPy's `np.copyto(dst, src)` signature
  and convention. The function is still in-place — `id(dst._values)` is
  preserved across the call — and returns `dst` for chaining. The
  per-column write goes through the destination's existing `__setitem__`
  with frequency / contained-range checks; a non-`TSeries` value at a
  matching key raises `TypeError` (matching Julia's behaviour, which
  would error inside the inner `copyto!`). `verbose=True` emits a single
  `UserWarning` at the end of the loop with all missing-from-workspace
  column names joined, matching Julia's end-of-loop `@warn`.
- **`set_holidays_map(country, subdivision)` delegates to `python-holidays`.**
  The Julia upstream ships bundled CSVs (`TimeSeriesEcon.jl/src/holidays/`)
  and reads them on first call. Python delegates to the
  [`python-holidays`](https://pypi.org/project/holidays/) PyPI package
  as the single source of truth — install the optional extra with
  `pip install "TimeSeriesEconPy[holidays]"`. Country / subdivision
  codes follow ISO 3166 conventions rather than the Julia CSVs'
  space-separated forms; a small number of names diverge (e.g. Julia's
  `"Yukon Territory"` is `"YT"` in Python). Use `get_holidays_options()`
  and `get_holidays_options("CA")` to discover the supported codes. The
  map's default range (`bdaily("1970-01-01")` to `bdaily("2049-12-31")`)
  is identical.
- **`@weval` becomes plain attribute access (or a snapshot context manager).**
  Julia's `@weval(w, EXPR)` macro rewrites `EXPR` so bare identifiers
  resolve against `w`'s namespace; Python has no AST-rewriting macros.
  The Pythonic spellings are direct attribute access (`w.b + w.a`, the
  default — almost always sufficient) or, when several unprefixed reads
  cluster, the snapshot context manager `Workspace.as_namespace()`:

  ```python
  with w.as_namespace() as ns:
      w.b1 = ns.b + ns.a       # ports Julia's `w.b1 = @weval w b + a`
      w.alpha = math.sin(math.pi) + ns.a - ns.a
  ```

  The namespace is a `types.SimpleNamespace` snapshot taken at
  `__enter__`: mutations to `w` after entry are invisible, and mutating
  the namespace does not write back to `w`. Use `w.x = ...` directly
  for writes. Values are held by reference, so mutating a contained
  TSeries / MVTSeries through the namespace still mutates the same
  object inside `w`. Non-identifier keys (e.g. `"1x"`) are silently
  omitted from the snapshot — keep using `w["..."]` for those. No
  `eval`-based form is provided: deferring identifier resolution to
  runtime `eval` of arbitrary expression strings would break static
  tooling (mypy, IDE refactors, "find references") and is a security
  footgun on workspaces loaded from untrusted sources.
