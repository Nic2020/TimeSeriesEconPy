# Recursive

`tsecon.rec(rng, target, fn)` is the higher-order Python analogue of Julia's `@rec`
macro: walks `rng`, calling `fn(target, t)` at each step. `tsecon.rec_linear(target,
coeffs, lags, rng)` is the specialised closed-form for linear recurrences (AR(p),
Fibonacci, lag polynomials) and routes through a Cython kernel when available;
introspect with `rec_linear_is_cython()`. See
[design/cython_strategy.md](../design/cython_strategy.md) for the three-flavour
benchmark.

::: tsecon.recursive
