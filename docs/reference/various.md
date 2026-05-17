# Misc helpers (overlay, compare, reindex)

Three sibling functions from upstream `TimeSeriesEcon.jl/src/various.jl`,
sharing the *LikeWorkspace* dispatch (Workspace / MVTSeries / Mapping):

* [`overlay`](#tsecon._various.overlay) — first-valid-wins composition
  over TSeries / Workspace / MVTSeries values. The dtype-appropriate
  *typenan* (NaN for floats, `iinfo.max` for integers, `False` for
  booleans) marks "missing"; the leftmost non-missing value wins
  position-by-position.
* [`compare`](#tsecon._various.compare) — recursive structural / numeric
  comparison. Returns a [`CompareResult`](#tsecon._various.CompareResult)
  (truthy on equality; `.differences` carries the per-leaf diff).
  Tolerance kwargs (`atol` / `rtol` / `nans`) match the Julia
  `compare` / `@compare` surface name-for-name.
* [`reindex`](#tsecon._various.reindex) — shift every MIT-keyed position
  inside a container so that `from` maps to `to`; values are preserved.
  Dispatches over MIT / MITRange / TSeries / MVTSeries / Workspace.

::: tsecon._various
