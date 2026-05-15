| Scenario | Description | Python (median) | Julia (median) | Ratio (Py / Jl) |
|---|---|---:|---:|---:|
| `construct_tseries_qq_100` | TSeries(qq, arr) from length-100 ndarray | 1.28 µs | 10.8 ns | 118.14x |
| `construct_mvts_qq_100x5` | MVTSeries(qq, 5 cols, 100x5 ndarray) | 10.30 µs | 729.3 ns | 14.12x |
| `indexing_mit_lookup_100` | Sum t[mit] over 100 keys | 90.19 µs | 96.5 ns | 934.90x |
| `indexing_int_lookup_100` | Sum t[int] over 100 keys | 63.34 µs | 86.9 ns | 729.11x |
| `indexing_mitrange_slice` | t[MITRange] — single 60-period slice | 4.79 µs | 68.4 ns | 70.09x |
| `indexing_mvts_column` | mvts['c'] — column access | 494.8 ns | 7.5 ns | 65.91x |
| `arith_add_misaligned` | 100Q + 100Q with 50Q overlap | 16.71 µs | 2.90 µs | 5.76x |
| `arith_add_aligned` | 100Q + 100Q same range | 17.31 µs | 2.52 µs | 6.86x |
| `arith_mul_scalar` | t * 2.5 | 15.13 µs | 2.25 µs | 6.73x |
| `shift_quarterly_lag1` | shift(t, -1) | 3.71 µs | 112.4 ns | 32.98x |
| `diff_quarterly` | diff(t) | 23.01 µs | 2.81 µs | 8.18x |
| `pct_quarterly` | pct(t) | 57.62 µs | 4.36 µs | 13.21x |
| `mean_quarterly_100` | mean(t) | 7.44 µs | 32.9 ns | 226.26x |
| `std_quarterly_100` | std(t) | 22.89 µs | 80.6 ns | 283.94x |
| `cor_two_tseries` | cor(a, b) on two TSeries | 98.71 µs | 146.5 ns | 673.73x |
| `cor_mvts_5_columns` | cor(mvts) — 5x5 corr matrix | 93.01 µs | 19.50 µs | 4.77x |
| `cov_mvts_5_columns` | cov(mvts) — 5x5 cov matrix | 65.65 µs | 19.90 µs | 3.30x |
| `moving_average_quarterly_4` | moving_average(t, 4) | 10.94 µs | 9.53 µs | 1.15x |
| `moving_sum_quarterly_4` | moving_sum(t, 4) | 8.91 µs | 7.26 µs | 1.23x |
| `undiff_quarterly` | undiff(t) | 70.91 µs | 4.08 µs | 17.40x |
| `fconvert_qq_to_yy_mean` | fconvert(Yearly, t, method='mean') | 202.35 µs | 970.0 ns | 208.61x |
| `fconvert_qq_to_yy_sum` | fconvert(Yearly, t, method='sum') | 83.87 µs | 845.9 ns | 99.14x |
| `fconvert_yy_to_qq_const` | fconvert(Quarterly, t, method='const') (higher) | 12.17 µs | 257.7 ns | 47.24x |
| `fconvert_mm_to_qq_mean` | fconvert(Quarterly, monthly_t, method='mean') | 281.05 µs | 995.8 ns | 282.23x |
| `rec_ar2_100` | AR(2) over 100 quarters — general rec + lambda | 660.21 µs | 527.2 ns | 1252.34x |
| `rec_linear_ar2_100_pylist` | AR(2) over 100 — rec_linear, pure-Python list | 13.47 µs | 576.6 ns | 23.36x |
| `rec_linear_ar2_100_numpy` | AR(2) over 100 — rec_linear NumPy kernel | 141.96 µs | 537.4 ns | 264.15x |
| `workspace_merge_5_series` | Workspace merge: 5 + 5 series | 2.13 µs | 5.14 µs | 0.41x |
| `workspace_filter_5_series` | Workspace filter: 10 down to 5 series | 4.27 µs | 7.36 µs | 0.58x |
| `rec_linear_ar2_100_cython` | AR(2) over 100 quarters — rec_linear Cython kernel | 2.10 µs | 552.8 ns | 3.81x |
