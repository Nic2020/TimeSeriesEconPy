| Scenario | Description | Python (median) | Julia (median) | Ratio (Py / Jl) |
|---|---|---:|---:|---:|
| `construct_tseries_qq_100` | TSeries(qq, arr) from length-100 ndarray | 1.32 µs | 11.1 ns | 119.10x |
| `construct_mvts_qq_100x5` | MVTSeries(qq, 5 cols, 100x5 ndarray) | 11.28 µs | 723.4 ns | 15.60x |
| `indexing_mit_lookup_100` | Sum t[mit] over 100 keys | 88.05 µs | 149.9 ns | 587.23x |
| `indexing_int_lookup_100` | Sum t[int] over 100 keys | 62.78 µs | 124.2 ns | 505.30x |
| `indexing_mitrange_slice` | t[MITRange] — single 60-period slice | 5.27 µs | 86.7 ns | 60.84x |
| `indexing_mvts_column` | mvts['c'] — column access | 487.4 ns | 6.9 ns | 70.63x |
| `indexing_lookup_100_api` | lookup(t, mit_keys) — public vectorised API | 65.40 µs | 275.6 ns | 237.32x |
| `indexing_lookup_100_numpy` | gather_numpy(values, ix) — NumPy kernel | 2.40 µs | 148.6 ns | 16.14x |
| `arith_add_misaligned` | 100Q + 100Q with 50Q overlap | 12.81 µs | 2.47 µs | 5.19x |
| `arith_add_aligned` | 100Q + 100Q same range | 12.99 µs | 2.30 µs | 5.65x |
| `arith_mul_scalar` | t * 2.5 | 12.60 µs | 2.11 µs | 5.97x |
| `shift_quarterly_lag1` | shift(t, -1) | 3.80 µs | 167.2 ns | 22.74x |
| `diff_quarterly` | diff(t) | 19.66 µs | 2.82 µs | 6.97x |
| `pct_quarterly` | pct(t) | 46.73 µs | 3.98 µs | 11.76x |
| `mean_quarterly_100` | mean(t) | 1.88 µs | 39.4 ns | 47.82x |
| `std_quarterly_100` | std(t) | 2.10 µs | 80.3 ns | 26.16x |
| `cor_two_tseries` | cor(a, b) on two TSeries | 5.49 µs | 110.7 ns | 49.61x |
| `cor_mvts_5_columns` | cor(mvts) — 5x5 corr matrix | 65.29 µs | 15.50 µs | 4.21x |
| `cov_mvts_5_columns` | cov(mvts) — 5x5 cov matrix | 46.81 µs | 15.10 µs | 3.10x |
| `mean_quarterly_100_numpy` | mean_numpy(values) — NumPy kernel | 6.66 µs | 31.5 ns | 211.44x |
| `std_quarterly_100_numpy` | std_numpy(values, 1) — NumPy kernel | 17.90 µs | 73.8 ns | 242.67x |
| `cor_two_tseries_numpy` | cor_numpy(x, y) — NumPy kernel | 60.64 µs | 102.3 ns | 592.79x |
| `moving_average_quarterly_4` | moving_average(t, 4) | 9.29 µs | 11.60 µs | 0.80x |
| `moving_sum_quarterly_4` | moving_sum(t, 4) | 7.84 µs | 9.10 µs | 0.86x |
| `undiff_quarterly` | undiff(t) | 56.75 µs | 5.50 µs | 10.32x |
| `fconvert_qq_to_yy_mean` | fconvert(Yearly, t, method='mean') | 207.71 µs | 1.19 µs | 174.30x |
| `fconvert_qq_to_yy_sum` | fconvert(Yearly, t, method='sum') | 86.17 µs | 1.08 µs | 79.80x |
| `fconvert_yy_to_qq_const` | fconvert(Quarterly, t, method='const') (higher) | 12.12 µs | 361.5 ns | 33.54x |
| `fconvert_mm_to_qq_mean` | fconvert(Quarterly, monthly_t, method='mean') | 310.01 µs | 1.35 µs | 229.63x |
| `rec_ar2_100` | AR(2) over 100 quarters — general rec + lambda | 700.61 µs | 805.5 ns | 869.81x |
| `rec_linear_ar2_100_pylist` | AR(2) over 100 — rec_linear, pure-Python list | 15.12 µs | 864.2 ns | 17.49x |
| `rec_linear_ar2_100_numpy` | AR(2) over 100 — rec_linear NumPy kernel | 151.91 µs | 864.2 ns | 175.78x |
| `workspace_merge_5_series` | Workspace merge: 5 + 5 series | 2.45 µs | 6.15 µs | 0.40x |
| `workspace_filter_5_series` | Workspace filter: 10 down to 5 series | 4.47 µs | 9.30 µs | 0.48x |
| `rec_linear_ar2_100_cython` | AR(2) over 100 quarters — rec_linear Cython kernel | 2.49 µs | 864.9 ns | 2.87x |
| `indexing_lookup_100_cython` | gather_cython(values, ix) — Cython kernel | 2.31 µs | 167.2 ns | 13.83x |
| `mean_quarterly_100_cython` | mean_cython(values) — Cython kernel | 848.1 ns | 29.3 ns | 28.93x |
| `std_quarterly_100_cython` | std_cython(values, 1) — Cython kernel | 1.01 µs | 70.8 ns | 14.29x |
| `cor_two_tseries_cython` | cor_cython(x, y) — Cython kernel | 1.53 µs | 118.4 ns | 12.96x |
