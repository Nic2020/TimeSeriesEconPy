# TSeries protocols

!!! info "Work in progress"
    This page is a stub. Paper-voice content arrives in a follow-up writing session.
    The full rationale lives in `claude_files/decisions/02_tseries_internal_design.md`.

`TSeries` and `MVTSeries` *wrap* NumPy arrays — they don't subclass `ndarray`.
This is the same pattern `xarray.DataArray` uses, for the same reason:
subclassing `ndarray` is a well-known source of surprises (every operation has
to decide whether to return your subclass or a plain `ndarray`, and getting that
wrong silently demotes types deep inside other libraries' code).

The integration with NumPy is via three protocols:

- **`__array_ufunc__`** — intercepts `np.log(t)`, `t1 + t2`, `t * 2`, etc.
  Element-wise ufuncs alignment-merge by `MIT` intersection before delegating to
  the underlying ndarrays.

- **`__array_function__`** — intercepts higher-level NumPy functions
  (`np.concatenate`, `np.cumsum`, …). Some of these we rebroadcast as time-aware
  operations; others we delegate to the values and return a plain ndarray.

- **`__array__`** — the escape hatch: `np.asarray(t)` returns the underlying
  values (no copy when contiguous). Used internally and by interop code that
  doesn't want to know what `t` is.

## The MIT-intersection alignment rule

`t1 + t2` returns a `TSeries` whose range is `t1.range ∩ t2.range`. If the
ranges don't overlap, the result is an empty `TSeries`. This matches the Julia
upstream and is the source of the most-cited "Python and Julia agree on the
semantics, here's the unit test that proves it" property test in the suite.
