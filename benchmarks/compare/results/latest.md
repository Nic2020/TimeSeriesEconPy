| Scenario | Description | Python (median) | Julia (median) | Ratio (Py / Jl) |
|---|---|---:|---:|---:|
| `construct_tseries_qq_100` | TSeries(qq, arr) from length-100 ndarray | 1.19 µs | 10.6 ns | 112.64x |
| `indexing_mit_lookup_100` | Sum t[mit] over 100 keys | 89.86 µs | 102.4 ns | 877.25x |
| `arith_add_misaligned` | 100Q + 100Q with 50Q overlap | 16.57 µs | 2.84 µs | 5.82x |
| `shift_quarterly_lag1` | shift(t, -1) | 3.56 µs | 108.8 ns | 32.75x |
| `moving_average_quarterly_4` | moving_average(t, 4) | 10.95 µs | 10.57 µs | 1.04x |
| `fconvert_qq_to_yy_mean` | fconvert(Yearly, t, method='mean') | 196.61 µs | 887.8 ns | 221.46x |
| `rec_ar2_100` | AR(2) recurrence over 100 quarters | 663.46 µs | 475.8 ns | 1394.54x |
| `rec_linear_ar2_100_numpy` | AR(2) over 100 quarters — rec_linear NumPy kernel | 134.70 µs | 494.4 ns | 272.45x |
| `workspace_merge_5_series` | Workspace merge: 5 + 5 series | 2.14 µs | 4.73 µs | 0.45x |
