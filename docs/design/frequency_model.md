# Frequency model

!!! info "Work in progress"
    This page is a stub. Paper-voice content arrives in a follow-up writing pass.

Three design choices that diverge from the Julia upstream:

1. **Cached-singleton frozen dataclasses.** `Yearly()` returns the same object as
   `Yearly()` — equality is identity. Frequencies are slotted dataclasses with a
   tiny per-class instance cache so the runtime cost of "compare two frequencies"
   is one `is` check.

2. **Constructor functions, not literal sugar.** Julia writes `2020Q1`; the
   Python analogue is `qq(2020, 1)`. We don't overload `int * Q1` because there
   is no precedent in the scientific Python stack for `*` returning anything but
   a number, and discoverability matters more than parity with the Julia spelling.

3. **`MIT` is not equal to a bare `int`.** Julia's `MIT{Quarterly}` interconverts
   with `Int` freely. Python can't do that without breaking the `__eq__` /
   `__hash__` invariant required by `dict` and `set`, so `qq(2020, 1) == 8080` is
   `False`. Use `int(mit)` to extract the underlying offset explicitly.

## Why cached singletons?

`Yearly()` is conceptually a *type*, not a value. The Julia upstream encodes that
with `Yearly{end_month}` as a parametric type. Python doesn't have first-class
parametric types at runtime, so we approximate them with frozen dataclasses, and
we cache the instances because the alternative ("construct a fresh `Yearly()`
each call site") would invalidate every `is`-based hot path.

## Why no `2020Q1` sugar?

A reader's first encounter with `2020Q1` in Julia produces a `MIT{Quarterly}`
literal. The equivalent in Python — `__rmul__` on a `Q1` sentinel object,
returning `MIT(Quarterly(), …)` — would be cute but would also be the *only*
place in the scientific Python ecosystem where `*` returns a non-number. We
chose the readable spelling (`qq(2020, 1)`) over the cute one.
