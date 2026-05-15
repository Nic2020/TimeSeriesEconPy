| Scenario | Description | Python (median) | Julia (median) | Ratio (Py / Jl) |
|---|---|---:|---:|---:|
| `construct_tseries_qq_100` | TSeries(qq, arr) from length-100 ndarray | 1.23 µs | 10.7 ns | 114.94x |
| `indexing_mit_lookup_100` | Sum t[mit] over 100 keys | 100.12 µs | 105.7 ns | 947.30x |
| `arith_add_misaligned` | 100Q + 100Q with 50Q overlap | 17.16 µs | 2.97 µs | 5.78x |
| `shift_quarterly_lag1` | shift(t, -1) | 3.72 µs | 135.7 ns | 27.39x |
| `moving_average_quarterly_4` | moving_average(t, 4) | 11.43 µs | 9.70 µs | 1.18x |
| `fconvert_qq_to_yy_mean` | fconvert(Yearly, t, method='mean') | 220.63 µs | 1.04 µs | 212.05x |
| `rec_ar2_100` | AR(2) recurrence over 100 quarters | 673.50 µs | 539.3 ns | 1248.88x |
| `workspace_merge_5_series` | Workspace merge: 5 + 5 series | 2.31 µs | 4.64 µs | 0.50x |
