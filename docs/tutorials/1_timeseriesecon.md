# Tutorial 1 — TimeSeriesEcon

A narrative port of the upstream Julia tutorial
[TutorialsEcon.jl/1.TimeSeriesEcon](https://bankofcanada.github.io/DocsEcon.jl/dev/Tutorials/1.TimeSeriesEcon/main/),
showing only Python idioms in code blocks and dropping a *Julia ↔ Python*
note per section so anyone migrating from the Julia codebase can find the
original passage. Every code block on this page runs at `mkdocs build`
time via the [`markdown-exec`](https://pawamoy.github.io/markdown-exec/)
plugin — the same trick `Documenter.jl` plays with `@repl tse` blocks
upstream — so a broken example fails CI rather than silently rotting on
the site.

## Section status

| #  | Section                          | Status |
|----|----------------------------------|:------:|
|  1 | [Frequency and Time](#1-frequency-and-time)               | 🟢    |
|  2 | [Ranges (`MITRange`)](#2-ranges)              | 🟢    |
|  3 | [TSeries — Creation](#3-tseries-creation)               | 🟢    |
|  4 | [TSeries — Access (read / write)](#4-tseries-access)  | 🟢    |
|  5 | [Arithmetic with TSeries](#5-arithmetic-with-tseries)          | 🟢    |
|  6 | [Shifts (lag / lead)](#6-shifts)              | 🟢    |
|  7 | [Diff and undiff](#7-diff-and-undiff)                  | 🟢    |
|  8 | [Moving average](#8-moving-average)                   | 🟢    |
|  9 | [Recursive assignments](#9-recursive-assignments)            | 🟢    |
| 10 | [Multi-variate Time Series](#10-multi-variate-time-series-mvtseries)        | 🟢    |
| 11 | [Plotting](#11-plotting)                         | 🟢    |
| 12 | [Workspaces](#12-workspaces)                       | 🟢    |
| 13 | [MVTSeries vs Workspace](#13-mvtseries-vs-workspace)           | 🟢    |
| 14 | [`overlay`](#14-overlay)                        | 🔵    |
| 15 | [`compare` / `@compare`](#15-compare)           | 🔵    |
| 16 | [BDaily holidays](#16-bdaily-holidays)                  | 🟡    |
| 17 | [Options](#17-options)                          | 🟢    |

Legend: 🟢 ported · 🟡 partial · 🔵 stubbed (depends on a later milestone).

<!-- Hidden setup block: imports, seeded RNG, matplotlib non-interactive
backend, and a tiny helper that emits a captured matplotlib figure as a
base64-PNG `<img>` tag. Shared via session="tut1" with every executable
block on this page. -->

```python exec="true" source="material-block" session="tut1"
import base64
import io
import math

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np

import tsecon
from tsecon import (
    BDaily,
    Daily,
    Duration,
    HalfYearly,
    MIT,
    MITRange,
    MVTSeries,
    Monthly,
    Quarterly,
    TSeries,
    Unit,
    Weekly,
    Workspace,
    Yearly,
    bdaily,
    daily,
    diff,
    frequency_of,
    lag,
    lead,
    mean,
    mit2yp,
    mm,
    moving,
    pct,
    period,
    plot,
    ppy,
    qq,
    rec,
    rec_linear,
    shift,
    undiff,
    weekly,
    year,
    yy,
)

# Reproducible RNG for any `rand`-style examples below.
rng_np = np.random.default_rng(20260517)

def _show(fig, alt=""):
    """Encode a matplotlib Figure as a base64 PNG and print the <img> tag.

    The surrounding fenced block needs `html="true"` so markdown-exec
    forwards the captured stdout as raw HTML instead of wrapping it
    in a code-formatted output block.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    print(f'<img src="data:image/png;base64,{data}" alt="{alt}" />')
```

## 1. Frequency and Time { #1-frequency-and-time }

In a time series the values are evenly spaced in time and each value is
labelled with the moment in which it occurred. TimeSeriesEconPy provides
the same two data types upstream's `TimeSeriesEcon.jl` does for these
concepts: a `Frequency` describes the *spacing*, an `MIT`
(moment-in-time) labels one *point*, and a `Duration` measures the
distance between two `MIT`s of the same frequency.

### Frequencies

The abstract type `Frequency` represents the idea of a sampling
cadence. All concrete frequencies are special cases. Four are
*period-of-year* (calendar) frequencies — `Yearly`, `HalfYearly`,
`Quarterly`, `Monthly` — and are defined by a number of periods per
year. Three are *calendar-date* frequencies that depend on the
Gregorian calendar: `Weekly`, `BDaily` (business-daily, Mon–Fri), and
`Daily`. The last, `Unit`, is not based on the calendar and simply
counts observations.

Typically you do not work with the `Frequency` types directly — you
get them via the constructor functions like `qq()` introduced below.
Every `Frequency` is a cached singleton: `Yearly() is Yearly()` is
`True`, so identity checks are valid.

#### End-months and end-days

`Yearly`, `HalfYearly`, and `Quarterly` have implicit default
end-months — 12 (December), 6 (June), and 3 (March) respectively.
Most uses of `Yearly()` implicitly mean `Yearly(end_month=12)`. You
can pick a variant explicitly:

```python exec="true" source="material-block" session="tut1"
print(Yearly())                 # default end_month = 12
print(Yearly(end_month=3))      # fiscal year ending in March
print(Quarterly())              # default end_month = 3
print(Quarterly(end_month=2))   # broadcaster's calendar
```

`Weekly` similarly has an implicit default end-day of Sunday:
`Weekly()` is `Weekly(end_day=7)`.

### Moments and durations

`MIT` values label particular moments in time, `Duration` values
measure the distance between two `MIT`s of the same frequency.

```python exec="true" source="material-block" session="tut1"
print(type(qq(2020, 1)))
print(type(mm(2021, 5) - mm(2020, 3)))
```

#### Creating MIT instances

Pythonic equivalents of Julia's `2020Q1` / `2020M3` / `2020Y` literal
suffixes are the constructor functions `qq` / `mm` / `yy`. The arguments
are `(year, period)` (or just `(year,)` for `yy`).

```python exec="true" source="material-block" session="tut1"
print(qq(2022, 1))
print(qq(2020, 3))
print(yy(2020))
print(mm(2022, 5))
```

For half-yearly the same pattern works through the lower-level
`MIT.from_yp` factory because there's no dedicated `hh()` shorthand:

```python exec="true" source="material-block" session="tut1"
print(MIT.from_yp(HalfYearly(), 2022, 1))
print(MIT.from_yp(HalfYearly(), 2022, 2))
```

For variant end-months, instantiate the frequency first and use
`MIT.from_yp`:

```python exec="true" source="material-block" session="tut1"
print(MIT.from_yp(Quarterly(end_month=2), 2022, 1))
print(MIT.from_yp(Yearly(end_month=11), 2020, 1))
```

For the calendar-date frequencies (`Daily`, `BDaily`, `Weekly`), a
date object or ISO string is required:

```python exec="true" source="material-block" session="tut1"
print(weekly("2022-01-03"))
print(bdaily("2022-01-03"))
print(daily("2022-01-03"))
```

`bdaily(...)` from a date that lands on a weekend raises by default
(matching Julia's `:strict` bias). Pass `bias=` to opt in to a
specific rounding rule:

```python exec="true" source="material-block" session="tut1"
print(bdaily("2022-01-01", bias="previous"))   # 2021-12-31
print(bdaily("2022-01-01", bias="next"))       # 2022-01-03
print(bdaily("2022-01-01", bias="nearest"))    # 2021-12-31
```

The bias for *all* implicit weekend-to-business-day rounding can be
flipped globally via [§17 Options](#17-options).

### Arithmetic with time

`MIT - MIT → Duration` of the shared frequency. We can add or subtract
a `Duration` to an `MIT` to get another `MIT`, add and subtract two
`Duration`s freely, but adding two `MIT`s is a type error — the
operation has no economic meaning.

```python exec="true" source="material-block" session="tut1"
a = qq(2001, 2) - qq(2000, 1)   # Duration of Quarterly
print(a)
```

A plain `int` on either side of `+` / `-` is automatically treated as a
`Duration` in the `MIT`'s own frequency. This is the same Julia
shorthand `2000Q1 + 6`:

```python exec="true" source="material-block" session="tut1"
print(qq(2000, 1) + Duration(Quarterly(), 6))   # explicit
print(qq(2000, 1) + 6)                          # same thing, idiomatic
```

Mixing frequencies raises:

```python exec="true" source="material-block" session="tut1"
try:
    qq(2000, 1) + Duration(Monthly(), 6)
except TypeError as exc:
    print("TypeError:", exc)
```

Multiplication of an `MIT` by an integer is not allowed:

```python exec="true" source="material-block" session="tut1"
try:
    qq(2000, 1) * 5
except TypeError as exc:
    print("TypeError:", exc)
```

### Other operations

`frequency_of(...)` returns the frequency of its argument; `ppy(...)`
gives the number of periods per year; `year(...)` / `period(...)` /
`mit2yp(...)` extract calendar coordinates.

```python exec="true" source="material-block" session="tut1"
print(frequency_of(yy(2000)))
print(frequency_of(qq(2020, 1) - qq(2019, 3)))

t = qq(2020, 3)
print("ppy:", ppy(t.frequency))   # ppy takes a Frequency, not an MIT
print("year:", year(t))
print("period:", period(t))
print("mit2yp:", mit2yp(t))
```

Note that `ppy` accepts a `Frequency` (not an `MIT`) — pass
`t.frequency` rather than `t`. As in Julia, the returned value is the
hardcoded sentinel (52 / 365 / 260) for `Weekly` / `Daily` / `BDaily`
regardless of the actual year, and `year` / `period` / `mit2yp` are
not defined for `Weekly` because a week can straddle a year boundary.

!!! info "Julia ↔ Python"
    Corresponds to *Frequency and Time* upstream. The most visible
    idiom difference is the absence of `2020Q1` literal sugar — we
    chose constructor functions (`qq(2020, 1)`) over operator-overloading
    Python's `int` because operator-overload sugar has no precedent in
    the scientific Python stack and would be a discoverability footgun.
    Equality between an `MIT` and a plain integer is *not* supported in
    Python (would violate the `__eq__`/`__hash__` contract); use
    `int(mit)` to extract the underlying value.

## 2. Ranges { #2-ranges }

`MITRange(start, stop)` is the equivalent of Julia's `2000M1:2001M9`
unit-step range, with a `step=` keyword for non-unit strides. Both
bounds are inclusive (same as Julia). All standard collection
operations work: `len`, iteration, indexing, slicing, `reversed`.

```python exec="true" source="material-block" session="tut1"
rng = MITRange(mm(2000, 1), mm(2001, 9))
print(rng)
print("len:", len(rng))
print("first:", rng.first())
print("last:", rng.last())
print("rng[3:5]:", rng[3:5])
```

Julia's broadcast addition `rng .+ 6` (shift the whole range by 6
periods) does not have a `+` operator overload here — express the
shift directly on the endpoints:

```python exec="true" source="material-block" session="tut1"
print(MITRange(rng.start + 6, rng.stop + 6))
```

For step ranges, pass `step=`. The step is a nonzero integer in the
range's own frequency.

```python exec="true" source="material-block" session="tut1"
rng2 = MITRange(mm(2000, 1), mm(2000, 8), step=2)
print(list(rng2))
```

A **negative `step`** walks the range backward, mirroring Julia's
`10U:-1:1U` (`StepRange{MIT}` form). This is the natural way to write a
backcasting recurrence — see § *Recursive assignments* below — and is
also how reversed iteration order survives into downstream consumers
like [`rec`](#recursive-assignments) and indexing without needing a
separate "is this iteration reversed?" flag.

```python exec="true" source="material-block" session="tut1"
back_rng = MITRange(qq(2021, 4), qq(2020, 1), step=-1)
print(list(back_rng)[:4], "...")
```

For one-shot iteration in reverse without changing the range's identity,
wrap with `reversed`:

```python exec="true" source="material-block" session="tut1"
print(list(reversed(MITRange(mm(2020, 1), mm(2020, 4)))))
```

Calendar-date ranges work the same way; for `BDaily`, opt-in `bias=`
on each endpoint to round into the range when a weekend is supplied:

```python exec="true" source="material-block" session="tut1"
print(MITRange(daily("2022-01-01"), daily("2022-01-31")))
print(MITRange(bdaily("2022-01-01", bias="next"),
               bdaily("2022-01-31", bias="previous")))
```

!!! info "Julia ↔ Python"
    Corresponds to *Ranges* upstream. Julia's `bd"2022-01-01:2022-01-31"`
    string-macro syntax has no Python analogue, but the same semantics
    survive: explicit `bias=` per endpoint replaces the macro's implicit
    "round into the range" behaviour. Julia's `rng .+ 6` is not an
    operator overload here — write `MITRange(rng.start + 6, rng.stop + 6)`
    explicitly.

## 3. TSeries — Creation { #3-tseries-creation }

`TSeries` is the workhorse 1-D time-series type — a NumPy `ndarray`
wearing an `MIT`-indexed labelled axis. It implements
`__array_ufunc__` and `__array_function__`, so every NumPy operation
flows through transparently while keeping the time-axis alignment
honest (see the [TSeries protocols design note](../design/tseries_protocols.md)).

The basic constructor takes a starting `MIT` and a 1-D array of values:

```python exec="true" source="material-block" session="tut1"
vals = rng_np.random(5)
ts = TSeries(qq(2020, 1), vals)
print(ts)
```

!!! warning "Important caveat — buffer aliasing on construction"
    `TSeries(qq(2020, 1), vals)` **does not copy** the underlying
    array — `vals` and the new `TSeries` share storage, and every
    modification to one is immediately reflected in the other. This
    matches both the Julia upstream *and* xarray's `DataArray`
    (which the protocol design follows; see
    [decision 16](../design/decisions.md#locked-decisions)).

    To break the alias, pass `copy=True` at construction time, or
    call `.copy()` after the fact:

    ```python
    ts = TSeries(qq(2020, 1), vals, copy=True)   # explicit at construct time
    ts = TSeries(qq(2020, 1), vals).copy()        # equivalent
    ts = TSeries(qq(2020, 1), vals.copy())        # equivalent (extra allocation)
    ```

    For container types — `Workspace`, `MVTSeries` — a shallow `.copy()`
    still shares value references; reach for `copy.deepcopy(...)` (or
    the chainable `.copy(deep=True)`) when you want a full break.

You can also construct a `TSeries` from a range alone. Without a value
argument the storage is NaN-filled (Julia's is uninitialised); pass a
scalar or an initialiser function to fill it:

```python exec="true" source="material-block" session="tut1"
rng = MITRange(qq(2020, 1), qq(2021, 4))
print(TSeries(rng))                # NaN-filled
print(TSeries(rng, math.pi))       # scalar fill
print(TSeries(rng, rng_np.random(len(rng))))   # explicit allocation
print(TSeries.zeros(rng))          # classmethod shortcut for `np.zeros`
print(TSeries.ones(rng))           # likewise for `np.ones`
```

!!! info "Element-type caveat"
    `TSeries(rng, 0)` infers `int64`. If you'll go on to do arithmetic
    that produces `NaN` (any frequency-conversion or aligned operator),
    pick `TSeries.zeros(rng)` (which forces `float64`) or pass `dtype=`
    explicitly. Same caveat applies upstream: `TSeries(rng, 0)` is
    `Int`, `TSeries(rng, zeros)` is `Float64`.

`TSeries.similar(...)` and `.copy()` make new instances. `similar`
returns a NaN-filled `TSeries` with the same range / dtype (or a
specified range), `copy` returns an exact value-for-value duplicate:

```python exec="true" source="material-block" session="tut1"
t = TSeries(rng, 2.7)
s = t.similar()
c = t.copy()
print("similar:", s)
print("copy:", c)
```

!!! info "Julia ↔ Python"
    Corresponds to *Creation of TSeries* upstream. The Julia
    function-as-initialiser idiom `TSeries(rng, rand)` becomes the
    explicit `TSeries(rng, rng_np.random(len(rng)))` here — we
    chose not to mirror callable initialisers because the NumPy idiom
    is already explicit and the callable form has fewer guardrails. The
    wrap-vs-copy semantics are deliberately the same as Julia *and*
    xarray; see [decision 16](../design/decisions.md#locked-decisions) for the rationale.

## 4. TSeries — Access { #4-tseries-access }

### Reading

Indexing by `MIT` returns a scalar; indexing by `MITRange` returns a
new `TSeries`; indexing by integer or slice falls through to NumPy
semantics on the underlying buffer.

```python exec="true" source="material-block" session="tut1"
rng = MITRange(qq(2000, 1), qq(2001, 1))
t = TSeries(rng, rng_np.random(len(rng)))
print("t:", t)
print("t[qq(2000,1)]:", t[qq(2000, 1)])
print("t[qq(2000,2):qq(2000,4)]:", t[MITRange(qq(2000, 2), qq(2000, 4))])
print("t[1]:", t[1])
print("t[2:4]:", t[2:4])   # ndarray slice (loses date labels)
```

Out-of-range reads raise:

```python exec="true" source="material-block" session="tut1"
try:
    t[qq(1999, 1)]
except IndexError as exc:
    print("IndexError:", exc)

try:
    t[MITRange(qq(2001, 1), qq(2001, 3))]
except IndexError as exc:
    print("IndexError:", exc)
```

Python has no `begin` / `end` keyword inside `[]`, so the "last *n* by
date" idiom uses the `lastdate` attribute explicitly:

```python exec="true" source="material-block" session="tut1"
print(t[MITRange(t.lastdate - 2, t.lastdate)])     # last 3
print(t[MITRange(t.firstdate + 1, t.lastdate - 1)])  # drop first and last
```

### Writing

Indexed assignment mutates a single position. Range assignment writes
multiple positions; the right-hand side must size-match, *or* be a
scalar broadcast.

```python exec="true" source="material-block" session="tut1"
t[qq(2000, 2)] = 5
print(t)

t[MITRange(t.firstdate, t.firstdate + 2)] = [1, 2, 3]
print(t)

t[MITRange(t.lastdate - 2, t.lastdate)] = 42        # scalar broadcast
print(t)
```

Python has no Julia-style `.=` vs `=` distinction. A scalar on the
right-hand side is broadcast to the slice. To reset the entire
`TSeries` to a constant, slice with `[:]`:

```python exec="true" source="material-block" session="tut1"
t[:] = math.pi
print(t)
```

A bare `t = math.pi` would *rebind* the variable to the float
`math.pi`, leaving the original `TSeries` untouched — same as Julia.

Unlike NumPy `ndarray`s, `TSeries` resize on assignment outside the
stored range. Any gap that's neither in the old range nor the
assignment range is NaN-filled.

```python exec="true" source="material-block" session="tut1"
t[MITRange(qq(1999, 1), qq(1999, 2))] = -3.7
print(t)
```

Resize works only for `MIT`-keyed assignment. An out-of-bounds
**integer** index still raises, matching the underlying `ndarray`:

```python exec="true" source="material-block" session="tut1"
try:
    t[15] = 3.5
except IndexError as exc:
    print("IndexError:", exc)
```

Assigning a `TSeries` to a range copies the right-hand side's values
*restricted* to the assignment range:

```python exec="true" source="material-block" session="tut1"
q = TSeries(t.range, 100.0)
t[MITRange(qq(1999, 3), qq(2000, 2))] = q[MITRange(qq(1999, 3), qq(2000, 2))]
print(t)
```

!!! info "Julia ↔ Python"
    Corresponds to *Access to Elements of TSeries*. Julia's
    `begin` / `end` keywords inside `[]` have no Python analogue — the
    semantically equivalent spellings are `t.firstdate` and `t.lastdate`.
    Resize-on-MIT-assign and bounds-error-on-int-assign carry across
    unchanged. The Julia `t .= 42` / `t[:end-2:end] .= q` dot-equals
    distinction collapses to plain `=` here because every Python
    operator is already element-wise.

## 5. Arithmetic with TSeries { #5-arithmetic-with-tseries }

Two kinds of operations: *whole-object* (treat the series as a vector)
and *element-wise* (treat it as a NumPy array). In Python every
arithmetic operator and every NumPy ufunc is element-wise — there's
no `+` vs `.+` distinction — but the alignment-by-MIT rule is the same
across both.

When two `TSeries` are added or subtracted, the result spans the
*intersection* of their ranges (anything outside is treated as
missing). Multiplying or dividing by a scalar preserves the range.

```python exec="true" source="material-block" session="tut1"
x = TSeries(MITRange(qq(2020, 1), qq(2020, 4)), rng_np.random(4))
y = TSeries(MITRange(qq(2020, 3), qq(2021, 2)), rng_np.random(4))
print("x:", x)
print("y:", y)
print("x + y:", x + y)
print("x - y:", x - y)
print("2 * y:", 2 * y)
print("y / 2:", y / 2)
```

NumPy ufuncs flow through transparently. `1 + x` broadcasts the scalar
across `x`'s range; `2 / y` broadcasts the reciprocal; `y ** 3`
cubes elementwise.

```python exec="true" source="material-block" session="tut1"
print("np.log(x):", np.log(x))
print("1 + x:", 1 + x)
print("x + y (vectorised):", x + y)
print("2 / y:", 2 / y)
print("y ** 3:", y ** 3)
```

In-place assignment over the intersection follows the same rule.
Slicing the left-hand side with the wider range resizes the target:

```python exec="true" source="material-block" session="tut1"
z = x.copy()
common = MITRange(max(z.firstdate, y.firstdate),
                  min(z.lastdate, y.lastdate))
z[common] = (1 + y)[common]         # write only within z ∩ y
print("z:", z)

z[y.range] = 3 + y                   # resize z to cover y's range
print("z (resized):", z)
```

Julia's `z .= 1 .+ y` implicitly aligns the assignment to the
intersection of `z` and `y`; in Python the alignment is explicit (the
`common` range above). On the resize-on-assign side both languages
behave the same — the left-hand side is grown to match the assignment
range, with any pre-existing gap NaN-filled.

Mixing a `TSeries` and a same-length NumPy array works in either
order; the result preserves the `TSeries`'s range:

```python exec="true" source="material-block" session="tut1"
v = 3 * np.ones(x.shape)
print(x + v)
```

!!! info "Julia ↔ Python"
    Corresponds to *Arithmetic with TSeries*. Julia separates whole-object
    arithmetic from element-wise via the `.` prefix; Python doesn't, so
    `2y` becomes `2 * y` and `log.(x)` becomes `np.log(x)`. Range
    intersection on `+` / `-` and broadcasting-on-scalar are identical.
    Mechanically the implementation rides on `__array_ufunc__` /
    `__array_function__` (xarray-style composition) rather than the
    `AbstractArray` inheritance Julia uses; see
    [TSeries protocols](../design/tseries_protocols.md).

## 6. Shifts { #6-shifts }

`lag(x)` and `lead(x)` shift the **labels** of the data (not the
data). `shift(x, k)` is the underlying primitive: positive `k` is a
lead, negative `k` is a lag (matches Julia). Each has an `_inplace`
variant that mutates in place rather than allocating.

```python exec="true" source="material-block" session="tut1"
print("x:        ", x)
print("lag(x):   ", lag(x))
print("lead(x):  ", lead(x))
print("lag(x, 3):", lag(x, 3))
print("shift(x, -1) == lag(x):", shift(x, -1).equals(lag(x)))
```

!!! info "Julia ↔ Python"
    Corresponds to *Shifts*. Julia's mutation-suffix `lag!` / `lead!`
    becomes the `_inplace` suffix — the leading-bang naming convention
    has no Python tradition, and conventional Python mutators (`list.sort`,
    `dict.update`) don't use one either. The sign convention on `shift`
    is unchanged.

## 7. Diff and undiff { #7-diff-and-undiff }

`diff(x)` defaults to `k=-1` (first-difference at lag 1). Positive `k`
takes a lead-difference; either way the result has the same range as
`x` with the first `|k|` entries NaN-filled.

```python exec="true" source="material-block" session="tut1"
dx = diff(x)
print(dx)
```

`undiff(dx)` is the inverse. In its plain form it's
`np.cumsum`-on-the-values lifted onto the same range — which means the
**first value of x is lost** because `diff` couldn't observe it.

```python exec="true" source="material-block" session="tut1"
print(undiff(dx))
```

To recover the original series exactly, anchor the inverse to a known
value at a known date via `undiff(dx, anchor=(date, value))`:

```python exec="true" source="material-block" session="tut1"
x2 = undiff(dx, anchor=(x.firstdate, float(x[x.firstdate])))
print("x ≈ x2:", x.allclose(x2))
```

The same idiom works for any `(MIT, value)` pair, not just
`x.firstdate`. Pass a `TSeries` as the anchor instead of a tuple to
align `dx` over a partial overlap (see
[`tsecon.undiff` reference](../reference/math.md)).

!!! info "Julia ↔ Python"
    Corresponds to *Diff and Undiff*. Julia's anchor syntax
    `firstdate(x) => first(x)` becomes the keyword `anchor=(...)`
    here — we collapsed Julia's `Pair` to a 2-tuple because Python has
    no first-class pair literal. The `k=-1` default and the
    `cumsum`-equivalent baseline behaviour are unchanged.

## 8. Moving average { #8-moving-average }

`moving(t, n)` computes the moving average over a window of length `|n|`.
A positive `n` is backwards-looking (includes lags up to `n - 1`), a
negative `n` is forwards-looking (includes leads up to `|n| - 1`); the
window always includes the current value. `moving_sum` and
`moving_average` are the two specialisations; `moving` is an alias for
`moving_average`. The accumulator is always `float64` regardless of
the input dtype (matches Julia's `zeros(out_len)`).

```python exec="true" source="material-block" session="tut1"
tt = TSeries(qq(2020, 1), np.arange(1.0, 11.0))
print(tt)
print("moving(tt, -4):", moving(tt, -4))   # forwards-looking 4-window
print("moving(tt, 6):", moving(tt, 6))     # backwards-looking 6-window
```

!!! info "Julia ↔ Python"
    Corresponds to *Moving Average*. The window-sign convention is the
    same as Julia's. `moving` here is `moving_average`; for sums the
    explicit `moving_sum(t, n)` is one extra import. Both overload onto
    `MVTSeries` (per-column reduction); see
    [the math reference](../reference/math.md).

## 9. Recursive assignments { #9-recursive-assignments }

Time-series recurrences look like

$$ a_t = (1 - \rho) \, a_{ss} + \rho \, a_{t-1} + \varepsilon_t. $$

Julia's `@rec` macro rewrites the body so each iteration sees the
previously-written value at `t-1`. Python has no parse-time macros,
so the equivalent is a higher-order function: `rec(rng, target, fn)`
calls `target[t] = fn(t)` once per step, in order, committing each
write before the next step's `fn` runs.

```python exec="true" source="material-block" session="tut1"
a_ss = 1.0
rho = 0.6
a = TSeries(MITRange(qq(2020, 1), qq(2022, 1)), a_ss)
a[a.firstdate] += 0.1   # impulse

rec(MITRange(a.firstdate + 1, a.lastdate), a,
    lambda t: (1 - rho) * a_ss + rho * a[t - 1])

print(a)
```

For the *linear-recurrence* common case — AR(p), Fibonacci, arbitrary
lag polynomials — there's a closed-form sibling `rec_linear(target,
coeffs, lags, rng)` that bypasses the per-step Python lambda call.
When the compiled Cython kernel is available (typical `pip install`),
`rec_linear` is roughly two orders of magnitude faster than `rec` for
the same recurrence (see the
[Cython strategy design note](../design/cython_strategy.md)). Same
AR(1) recurrence rewritten in closed form:

```python exec="true" source="material-block" session="tut1"
b = TSeries(MITRange(qq(2020, 1), qq(2022, 1)), a_ss)
b[b.firstdate] += 0.1

# b[t] = (1 - rho) * a_ss + rho * b[t - 1]
#      = c0  +  rho * b[t - 1]    where  c0 = (1 - rho) * a_ss
# rec_linear handles the rho * b[t - 1] part; the additive constant
# gets folded in as a coefficient with lag 0 — but rec_linear requires
# lags >= 1, so for AR(1)-with-constant we typically subtract the
# steady state, run the homogeneous recurrence, and add back.
deviation = b - a_ss
rec_linear(deviation, [rho], [1],
           MITRange(deviation.firstdate + 1, deviation.lastdate))
b = deviation + a_ss
print(b)
print("rec vs rec_linear match:", a.allclose(b))
print("rec_linear is using Cython:", tsecon.rec_linear_is_cython())
```

If you need to drop the first *n* periods of a range (Julia's
`rangeof(a, drop=n)`), build the sub-range explicitly:

```python exec="true" source="material-block" session="tut1"
print("a.range:           ", a.range)
print("drop=1 (forward):  ", MITRange(a.firstdate + 1, a.lastdate))
print("drop=-1 (backward):", MITRange(a.firstdate, a.lastdate - 1))
```

### Backcasting (reversed range) { #9b-backcasting }

A *backcast* runs the same recurrence machinery backward in time — the
classic case is "we know the terminal value, what was the path that led
to it." In Julia this reads `@rec t=10U:-1:1U s[t] = s[t+1] - g`. The
Python port works by feeding `rec` a reversed `MITRange` (`step=-1`); no
new entry point is needed because `rec` iterates `for t in rng` and
`MITRange` carries the direction:

```python exec="true" source="material-block" session="tut1"
# Backcast: anchor `a_ss` at the end, walk backward applying
# s[t] = s[t+1] - g for a constant drift g.
g = 0.05
back = TSeries(MITRange(qq(2020, 1), qq(2022, 4)), 0.0)
back[back.lastdate] = a_ss
rec(MITRange(back.lastdate - 1, back.firstdate, step=-1), back,
    lambda t: back[t + 1] - g)
print(back)
```

The same idiom works for `rec_linear` with negative lags: a reversed
range plus same-sign lags reads "already-written" future positions and
backfills the series. See the
[`rec_linear` docstring](../reference/recursive.md#rec_linear) for the
sign-of-lag-matches-sign-of-step contract.

For multi-target recurrences (two series updated together each
step), write the explicit `for t in rng:` loop — `rec` only handles
the single-target case; the wrapper buys nothing once more than one
series is being written. See the
[recursive reference](../reference/recursive.md).

!!! info "Julia ↔ Python"
    Corresponds to *Recursive assignments*. The Julia `@rec` macro
    rewrites the body at parse time; the Python `rec(rng, target, fn)`
    higher-order form is more verbose but matches the *semantics*
    exactly — each iteration commits before the next runs, so
    `target[t - k]` always reads the freshly-written value. For
    performance-critical recurrences the Cython-backed `rec_linear`
    closes the gap to Julia's `@rec` (see
    [Cython strategy](../design/cython_strategy.md) for the empirical
    classification).

## 10. Multi-variate Time Series (`MVTSeries`) { #10-multi-variate-time-series-mvtseries }

`MVTSeries` is a 2-D `TSeries`-of-`TSeries` — every column shares the
same frequency, the same range, and the same element type. Rows are
labelled by `MIT`, columns by `str` (Julia's symbols become plain
strings in Python).

### Construction

Positional form: starting `MIT`, column names, 2-D matrix of values.

```python exec="true" source="material-block" session="tut1"
mv = MVTSeries(qq(2020, 1), ("a", "b"), rng_np.random((6, 2)))
print(mv)
```

Range form: pass an `MITRange` and either a value initialiser (scalar,
`numpy` callable like `np.zeros`, or a 2-D ndarray) plus a name tuple:

```python exec="true" source="material-block" session="tut1"
print(MVTSeries(MITRange(qq(2020, 1), qq(2021, 3)),
                ("one", "too", "tree"), np.zeros))
```

Keyword form: each kwarg supplies one column. Values can be
`TSeries`, 1-D arrays, or scalars (broadcast). When `firstdate_or_range`
is omitted, the range is taken as `rangeof_span` of the kwarg series.

```python exec="true" source="material-block" session="tut1"
data = MVTSeries(
    MITRange(qq(2020, 1), qq(2021, 1)),
    hex=TSeries(qq(2019, 1), np.arange(1.0, 21.0)),
    why=np.zeros(5),
    zed=3,
)
print(data)
```

`copy=True` forces a fresh allocation in the array form; the kwarg
form always allocates fresh storage (per-column data is *copied* into
the wide 2-D buffer). `similar(...)` and `.copy()` work the same way
they do on `TSeries`.

### Access

`MVTSeries` indexing has four shapes, all matching Julia's:

* **Row + column** with `mv[mit, name]` (scalar) or
  `mv[mit_range, names]` (sub-MVTSeries).
* **Single MIT** returns the whole row as a 1-D `ndarray`. To get
  a single-row `MVTSeries` back, pass a length-1 range.
* **Single column** with `mv[name]` (returns a `TSeries`) or with the
  `mv.name` attribute shortcut.
* **Tuple of column names** returns an `MVTSeries` with those columns.

```python exec="true" source="material-block" session="tut1"
print("data[qq(2020, 2), 'hex']:", data[qq(2020, 2), "hex"])
print("data[qq(2020, 2)]:       ", data[qq(2020, 2)])           # ndarray row
print("data[qq(2020, 2):qq(2020, 2)]:")
print(data[MITRange(qq(2020, 2), qq(2020, 2))])                  # row as MVTSeries

print("data['zed']:", data["zed"])         # column as TSeries
print("data[('zed',)]:")
print(data[("zed",)])                       # tuple → 1-column MVTSeries
print("data.zed (attribute access):", data.zed)
```

### Iterating columns

To loop over `(name, TSeries)` pairs, use the `.columns` accessor —
it returns a `dict[str, TSeries]` whose values are *views* on the
underlying 2-D buffer (no copy):

```python exec="true" source="material-block" session="tut1"
for name, series in data.columns.items():
    print(f"Average of {name!r} is {mean(series):.4f}.")
```

The `for name in data` form iterates *rows* (NumPy semantics on a 2-D
array). For column iteration always go through `.columns`.

!!! info "Julia ↔ Python"
    Corresponds to *Multi-variate Time Series*. Julia's column names
    are `Symbol`s (`:hex`); Python's are plain `str` (`"hex"`). The
    Julia `columns(data)` helper becomes the `data.columns` accessor,
    which returns a `dict[str, TSeries]` rather than a Julia
    `Generator` — same use case, more Pythonic ergonomics for the
    common `for name, series in data.columns.items()` loop. The
    storage layout (single 2-D buffer + cached column views) is the
    same.

## 11. Plotting { #11-plotting }

`tsecon.plot(...)` is a thin adapter over matplotlib (default) and
plotly. The TSeries arm overlays all inputs on a single axes; the
MVTSeries arm builds a panel grid (one subplot per variable, sharing a
legend across datasets). Backend selection is lazy: neither
matplotlib nor plotly is a hard dependency — install with
`pip install 'TimeSeriesEconPy[matplotlib]'` (or `[plotly]`). On this
page we use matplotlib because the build needs static PNG output.

```python exec="true" html="true" source="material-block" session="tut1"
rand_q = TSeries(a.range, 1 + 0.1 * rng_np.random(len(a)))
fig = plot(a, rand_q, label=["a", "rand"])
_show(fig, alt="TSeries overlay")
```

!!! info "Mixed-frequency overlay"
    Our `plot(...)` adapter requires all overlaid series to share a
    frequency — the underlying matplotlib / plotly axis is one *or* the
    other (numeric YP coordinates *or* native datetime), not both. The
    Julia upstream's `Plots.plot(...)` converts every MIT to a float
    via `float(2020Q1) == 2020.0`, which gives mixed-frequency overlays
    for free but loses unit information on the axis. We'll keep an eye
    on whether this is worth the trade-off after the first round of
    user feedback.

For an `MVTSeries`, each variable goes in its own subplot. Layout
defaults follow the Julia recipe: 1→(1,1), 2→(1,2), …, 10→(5,2).

```python exec="true" html="true" source="material-block" session="tut1"
db_a = MVTSeries(
    MITRange(qq(2020, 1), qq(2023, 4)),
    x=0.5,
    y=np.arange(0, 16) / 15.0,
    z=rng_np.random(16),
)
fig = plot(db_a, label="db_a")
_show(fig, alt="MVTSeries panel grid")
```

You can plot several `MVTSeries` together. The union of column names
determines the panel set; a dataset missing a variable is simply
skipped in that subplot.

```python exec="true" html="true" source="material-block" session="tut1"
db_b = MVTSeries(
    MITRange(qq(2020, 1), qq(2023, 4)),
    y=0.5,
    z=np.linspace(1.0, 0.0, 16),
    w=rng_np.random(16),
)
fig = plot(db_a, db_b, label=["db_a", "db_b"])
_show(fig, alt="Multi-MVTSeries overlay panel grid")
```

`vars=` selects and orders the panels (cap of 10 variables). `trange=`
restricts the rendered window. Every other matplotlib keyword (e.g.
`figsize=`) passes through.

```python exec="true" html="true" source="material-block" session="tut1"
fig = plot(
    db_a, db_b,
    label=["db_a", "db_b"],
    vars=["y", "z"],
    trange=MITRange(qq(2020, 3), qq(2022, 3)),
    figsize=(7, 4),
)
_show(fig, alt="Plot with vars= and trange=")
```

!!! info "Julia ↔ Python"
    Corresponds to *Plotting*. The Julia upstream is a Plots.jl recipe
    that picks a default panel layout; here it's a matplotlib /
    plotly dispatcher with the same panel-shape table. `trange=`
    survives unchanged because it operates on the time axis directly;
    Julia's `xlim=(MIT, MIT)` (which converts MITs to floats) does not
    have a Python equivalent — when frequencies differ across panels
    we currently fall back to matplotlib's native `xlim=` on the
    underlying datetime axis. See [decision 06](../design/decisions.md#locked-decisions)
    for the broader plotting strategy.

## 12. Workspaces { #12-workspaces }

`Workspace` is the heterogeneous-bag-of-things container — ranges,
scalars, time series of any frequency, nested workspaces. Internally
it's an order-preserving `dict[str, Any]` with attribute access; most
`dict` operations work directly.

```python exec="true" source="material-block" session="tut1"
w = Workspace()
w.rng = a.range                       # the AR(1) impulse-response from §9
w.start = w.rng.start
w.a = a.copy()                        # own the values, don't alias
print(w)
```

Use `del` to drop a member:

```python exec="true" source="material-block" session="tut1"
del w.start
print(w)
```

The kwargs constructor mirrors Julia's name-value-pair list — useful
when the workspace is being built up from a known schema:

```python exec="true" source="material-block" session="tut1"
print(Workspace(
    rng=MITRange(qq(2020, 1), qq(2021, 4)),
    alpha=0.1,
    v=TSeries(qq(2020, 1), rng_np.random(6)),
))
```

A `dict` (or any mapping) can be passed positionally too:

```python exec="true" source="material-block" session="tut1"
datalist = {
    "rng": MITRange(qq(2020, 1), qq(2021, 4)),
    "alpha": 0.1,
    "v": TSeries(qq(2020, 1), rng_np.random(6)),
}
print(Workspace(datalist))
```

To turn an `MVTSeries` into a `Workspace`, pass its columns dict
through the same constructor. The TSeries values are aliased into the
new workspace by default — pass `copy.deepcopy` if you want them
detached.

```python exec="true" source="material-block" session="tut1"
import copy
w_a = Workspace(db_a.columns)                # aliased columns (fast)
w_a_owned = Workspace(copy.deepcopy(dict(db_a.columns)))  # owned copies
print(w_a)
```

To go the other direction (Workspace → MVTSeries), construct an
`MVTSeries` from `w_a.rangeof()` and the dict of TSeries:

```python exec="true" source="material-block" session="tut1"
print(MVTSeries(w_a.rangeof(), **{k: v for k, v in w_a.items() if isinstance(v, TSeries)}))
```

`Workspace.rangeof()` returns the *intersection* of the ranges of all
TSeries members; `Workspace.rangeof_span()` returns the *union*.
Members that aren't TSeries are ignored for both.

!!! info "Julia ↔ Python"
    Corresponds to *Workspaces*. Julia's `:symbol` member names are
    `str` here. `delete!(w, :start)` becomes `del w.start`. Workspace's
    `.copy(deep=False)` follows xarray's wrap-vs-copy contract: a
    shallow copy shares value references, a deep copy doesn't (see
    [decision 16](../design/decisions.md#locked-decisions)). The pandas / polars
    interop layer attaches `Workspace.to_pandas(...)` etc. as method
    delegates — see the [interop reference](../reference/interop.md).

## 13. MVTSeries vs Workspace { #13-mvtseries-vs-workspace }

The two types overlap deliberately. Both store named time-series
data, both support attribute access (`db.x`) and item access
(`db["x"]`).

* **`MVTSeries` is a matrix.** All variables share a frequency, a
  range, and an element type. The 2-D `.values` buffer is contiguous
  in memory, which makes linear-algebra and statistics calls
  cheap — `mean(mvts)` reduces over the flat buffer in a single C call,
  `mvts @ A` works because `__array__` is wired to the 2-D ndarray.
  Adding or removing a column requires a fresh allocation.
* **`Workspace` is a dictionary.** It can hold heterogeneous types,
  time series of *different* frequencies, nested workspaces, plain
  scalars, ranges — anything. You can grow or shrink it in place.
  Linear algebra on the *workspace* is not defined (you'd have to
  pick out the relevant TSeries first).

The right call is usually clear from the data: if every value is a
TSeries of the same frequency and you'll be doing column-wise
reductions on it, reach for `MVTSeries`; otherwise `Workspace`.

The two convert into each other freely. See [§12](#12-workspaces) for
the round-trip pattern.

!!! info "Julia ↔ Python"
    Corresponds to *MVTSeries vs Workspace*. The trade-off is the
    same: `MVTSeries` for stats / linear-algebra, `Workspace` for
    heterogeneous bags. One small Python-specific note: Python's
    `dict` preserves insertion order natively (since 3.7), so
    `Workspace` doesn't need an `OrderedDict` — see
    [the JSON serialization design note](../design/serialization.md)
    for how that order is preserved on round-trip.

## 14. `overlay` { #14-overlay }

!!! warning "Not yet available"
    `overlay` is part of `various.jl` upstream and is queued for
    [M4](../design/decisions.md#locked-decisions). The two modes — `TSeries`-first
    (range union, first non-NaN wins) and `Workspace`/`MVTSeries`-first
    (recursive merge) — will land together once the rest of the
    `various.jl` surface is ported.

    For now, the *first-non-NaN-wins* TSeries pattern can be expressed
    by hand using `np.where(np.isnan(a), b, a)` on aligned values; the
    workspace recursive merge is more involved and warrants the
    dedicated implementation.

## 15. `compare` / `@compare` { #15-compare }

!!! warning "Not yet available"
    Like `overlay`, `compare` and `@compare` live in `various.jl`
    upstream and are queued for [M4](../design/decisions.md#locked-decisions). The
    Python port will collapse the macro and function forms into a
    single `compare(v1, v2, *, atol=, rtol=, nans=False,
    ignoremissing=False, showequal=False)` — Python has no parse-time
    macros for the auto-labelling Julia uses, so the names must be
    supplied explicitly.

## 16. BDaily holidays { #16-bdaily-holidays }

The holidays *kwargs* (`skip_all_nans=`, `skip_holidays=`,
`holidays_map=`) are ported and wired into `shift` / `lag` / `lead` /
`diff` / `pct`, into `fconvert` lower-frequency aggregation paths,
and into the `_stats.py` reductions (`mean` / `std` / `var` / etc.).
The bundled-holidays-by-country *loader* (`set_holidays_map("CA",
"ON")` in Julia) is M4 work — it'll wrap the
[`holidays`](https://pypi.org/project/holidays/) PyPI package rather
than re-shipping the Julia data files. For now, build the map yourself
and pass it in via `set_holidays_map(ts_bool)` or as a per-call
`holidays_map=`.

A taste of the `skip_all_nans` knob in action — non-NaN handling is
unchanged from upstream:

```python exec="true" source="material-block" session="tut1"
ts = TSeries(bdaily("2022-01-03"), np.arange(1.0, 11.0))
ts[bdaily("2022-01-07")] = np.nan
print("pct(ts):")
print(pct(ts))
print()
print("pct(ts, skip_all_nans=True):")
print(pct(ts, skip_all_nans=True))
```

When `skip_all_nans=True`, `pct` (and `diff` / `shift`) replace NaN
inputs with the nearest non-NaN value *in the shift direction* before
computing — so the `pct` for `2022-01-10` is taken relative to
`2022-01-06` (the most recent non-NaN before the gap) rather than the
NaN at `2022-01-07`. See [`tsecon.pct`](../reference/math.md) for the
full kwargs surface.

!!! info "Julia ↔ Python"
    Corresponds to *BDaily Holidays*. The functional surface for the
    three kwargs is identical; the country / region map loader
    (`set_holidays_map("CA", "ON")`) lands later because the Python
    ecosystem already has a maintained holidays package and rebuilding
    Julia's bundled binary would be wasteful.

## 17. Options { #17-options }

A small process-global dictionary holds the package's settings.
Three options exist today: `bdaily_creation_bias`, `bdaily_holidays_map`,
and `x13path` (the last is M2 plumbing; ignore for now).

```python exec="true" source="material-block" session="tut1"
print("default bias:", tsecon.getoption("bdaily_creation_bias"))
```

Setting `bdaily_creation_bias` changes the default `bias=` for every
`bdaily(...)` constructor call that omits it. The default is
`"strict"`, which raises when the input date lands on a weekend; the
other valid values are `"previous"`, `"next"`, `"nearest"`.

```python exec="true" source="material-block" session="tut1"
print(bdaily("2022-01-01", bias="next"))   # explicit per-call → 2022-01-03

tsecon.setoption("bdaily_creation_bias", "next")
print(bdaily("2022-01-01"))                # now the default does the same
tsecon.setoption("bdaily_creation_bias", "strict")  # restore
```

For test-friendly scoped overrides, use the `option_scope(**kwargs)`
context manager — it restores the prior values on exit:

```python exec="true" source="material-block" session="tut1"
from tsecon import option_scope

with option_scope(bdaily_creation_bias="previous"):
    print("inside the scope:", bdaily("2022-01-01"))   # → 2021-12-31
print("after the scope:", tsecon.getoption("bdaily_creation_bias"))
```

The `bdaily_holidays_map` option holds a `TSeries[BDaily, bool]` (or
`None`); `True` means business day, `False` means holiday. The
bundled-by-country loader (Julia's
`set_holidays_map("CA", "ON")`) is M4 work — see [§16](#16-bdaily-holidays).
For now, build the map yourself and pass it to `set_holidays_map(...)`:

```python exec="true" source="material-block" session="tut1"
from tsecon import clear_holidays_map, get_holidays_map, set_holidays_map

cal = TSeries.trues(MITRange(bdaily("2022-01-03"), bdaily("2022-12-30")))
cal[bdaily("2022-01-03")] = False   # mark New Year's observed
set_holidays_map(cal)
print("map length:", len(get_holidays_map()))
clear_holidays_map()
print("after clear:", get_holidays_map())
```

!!! info "Julia ↔ Python"
    Corresponds to *Options*. Julia's `setoption(:foo, value)` symbol-
    arg sugar becomes `setoption("foo", value)` with plain strings —
    Python has no `Symbol` type and faking one would be pure friction.
    The `option_scope` context manager is the Pythonic equivalent of
    `with` `withTimeSeriesEcon.setoption` callable patterns in Julia
    test suites. The set-by-country-code loader is M4; everything else
    is the same.

---

## What's next

Sections marked 🔵 (`overlay`, `compare`) land alongside the rest of
`various.jl` in M4. The 🟡-marked half of BDaily holidays (the
country-code map loader) also lands in M4, on top of the existing
[holidays PyPI package](https://pypi.org/project/holidays/). Until
then, everything in this tutorial is shipped and tested.

If you came from `TimeSeriesEcon.jl`, the
[migration guide](../design/migration_from_julia.md) collects the
recurring idiom differences in one place. If you're starting fresh,
the [reference pages](../reference/frequencies.md) cover the public
API exhaustively.
