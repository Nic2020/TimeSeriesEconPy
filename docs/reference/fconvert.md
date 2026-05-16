# Frequency conversion (fconvert)

Round-trip frequency conversion across YP frequencies (Yearly / HalfYearly / Quarterly /
Monthly) and calendar frequencies (Daily / BDaily / Weekly). Public surface: `fconvert`,
`fconvert_tseries`, `fconvert_mit`, `fconvert_range`, `fconvert_parts`, `extend_series`,
`strip_tseries` / `strip_tseries_inplace`, `trim_series`.

YP→YP aggregate (lower-frequency) paths route through a Cython kernel that fuses the
two nested loops in `aggregate_groups_numpy`; introspect with `fconvert_is_cython()`.

::: tsecon.fconvert
