# Math (shift, diff, moving, undiff)

Time-series transforms re-exported at top level: `shift` / `lag` / `lead` / `diff`
/ `pct` / `apct` / `ytypct` / `moving` / `moving_average` / `moving_sum` / `undiff`
and their `_inplace` variants where applicable. `BDaily` kwargs
(`skip_all_nans` / `skip_holidays` / `holidays_map`) are wired into `shift` / `lag`
/ `lead` / `diff` / `pct`.

::: tsecon._math
