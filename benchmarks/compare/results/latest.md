| Scenario | Description | Python (median) | Julia (median) | Ratio (Py / Jl) |
|---|---|---:|---:|---:|
| `construct_tseries_qq_100` | TSeries(qq, arr) from length-100 ndarray | 1.23 µs | 10.1 ns | 121.34x |
| `indexing_mit_lookup_100` | Sum t[mit] over 100 keys | 87.86 µs | 96.5 ns | 910.06x |
| `arith_add_misaligned` | 100Q + 100Q with 50Q overlap | 16.68 µs | 2.78 µs | 6.01x |
| `shift_quarterly_lag1` | shift(t, -1) | 3.37 µs | 109.2 ns | 30.87x |
| `moving_average_quarterly_4` | moving_average(t, 4) | 11.23 µs | 9.80 µs | 1.15x |
| `fconvert_qq_to_yy_mean` | fconvert(Yearly, t, method='mean') | 194.95 µs | 1.01 µs | 192.36x |
| `rec_ar2_100` | AR(2) recurrence over 100 quarters | 705.61 µs | 506.1 ns | 1394.24x |
| `rec_linear_ar2_100_numpy` | AR(2) over 100 quarters — rec_linear NumPy kernel | 137.08 µs | 489.8 ns | 279.84x |
| `workspace_merge_5_series` | Workspace merge: 5 + 5 series | 2.19 µs | 4.81 µs | 0.46x |
| `rec_linear_ar2_100_cython` | AR(2) over 100 quarters — rec_linear Cython kernel | 2.13 µs | 492.4 ns | 4.33x |
