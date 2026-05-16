# Upstream mirror metadata

The `MIRRORS_JULIA_*` constants pin the upstream `TimeSeriesEcon.jl` commit that
this Python port mirrors. A weekly CI workflow diffs the pin against
`bankofcanada/TimeSeriesEcon.jl@origin/main` and updates a rolling tracking issue.

::: tsecon._mirror
