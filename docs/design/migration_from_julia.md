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
