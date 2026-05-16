| Scenario | Description | tsecon (median) | Julia (median) | Pandas (median) | Polars (median) | Ratio (Py / Jl) |
|---|---|---:|---:|---:|---:|---:|
| `construct_tseries_qq_100` | TSeries(qq, arr) from length-100 ndarray | 2.13 µs | 22.8 ns | 35.37 µs | 43.01 µs | 93.07x |
| `construct_mvts_qq_100x5` | MVTSeries(qq, 5 cols, 100x5 ndarray) | 21.70 µs | 1.36 µs | 150.24 µs | 108.67 µs | 15.95x |
| `indexing_mit_lookup_100` | Sum t[mit] over 100 keys | 183.43 µs | 286.9 ns | 1.91 ms | 29.88 ms | 639.42x |
| `indexing_int_lookup_100` | Sum t[int] over 100 keys | 129.87 µs | 239.8 ns | n/a | n/a | 541.54x |
| `indexing_mitrange_slice` | t[MITRange] — single 60-period slice | 9.84 µs | 146.9 ns | n/a | n/a | 66.99x |
| `indexing_mvts_column` | mvts['c'] — column access | 955.6 ns | 15.4 ns | n/a | n/a | 61.93x |
| `indexing_lookup_100_api` | lookup(t, mit_keys) — public vectorised API | 136.20 µs | 543.4 ns | n/a | n/a | 250.67x |
| `indexing_lookup_100_numpy` | gather_numpy(values, ix) — NumPy kernel | 3.11 µs | 271.3 ns | n/a | n/a | 11.45x |
| `arith_add_misaligned` | 100Q + 100Q with 50Q overlap | 25.08 µs | 4.71 µs | 435.48 µs | 3.02 ms | 5.32x |
| `arith_add_aligned` | 100Q + 100Q same range | 23.32 µs | 4.37 µs | 88.58 µs | 5.44 µs | 5.34x |
| `arith_mul_scalar` | t * 2.5 | 18.03 µs | 3.92 µs | 82.06 µs | 16.81 µs | 4.60x |
| `shift_quarterly_lag1` | shift(t, -1) | 7.28 µs | 285.5 ns | 85.42 µs | 308.70 µs | 25.48x |
| `diff_quarterly` | diff(t) | 35.61 µs | 5.00 µs | 91.89 µs | 274.80 µs | 7.12x |
| `pct_quarterly` | pct(t) | 88.28 µs | 7.17 µs | 307.49 µs | 281.21 µs | 12.31x |
| `mean_quarterly_100` | mean(t) | 3.41 µs | 62.4 ns | 37.06 µs | 890.3 ns | 54.68x |
| `std_quarterly_100` | std(t) | 3.78 µs | 126.2 ns | 77.68 µs | 1.51 µs | 29.98x |
| `cor_two_tseries` | cor(a, b) on two TSeries | 10.06 µs | 195.5 ns | 327.24 µs | 294.90 µs | 51.45x |
| `cor_mvts_5_columns` | cor(mvts) — 5x5 corr matrix | 115.57 µs | 28.50 µs | n/a | n/a | 4.06x |
| `cov_mvts_5_columns` | cov(mvts) — 5x5 cov matrix | 83.01 µs | 28.90 µs | n/a | n/a | 2.87x |
| `mean_quarterly_100_numpy` | mean_numpy(values) — NumPy kernel | 10.74 µs | 56.3 ns | n/a | n/a | 190.82x |
| `std_quarterly_100_numpy` | std_numpy(values, 1) — NumPy kernel | 34.32 µs | 70.2 ns | n/a | n/a | 488.53x |
| `cor_two_tseries_numpy` | cor_numpy(x, y) — NumPy kernel | 109.16 µs | 197.0 ns | n/a | n/a | 554.20x |
| `moving_average_quarterly_4` | moving_average(t, 4) | 16.87 µs | 20.50 µs | 172.39 µs | 72.41 µs | 0.82x |
| `moving_sum_quarterly_4` | moving_sum(t, 4) | 14.51 µs | 16.30 µs | 171.57 µs | 80.27 µs | 0.89x |
| `undiff_quarterly` | undiff(t) | 108.85 µs | 6.00 µs | n/a | n/a | 18.14x |
| `fconvert_qq_to_yy_mean` | fconvert(Yearly, t, method='mean') | 63.66 µs | 2.11 µs | 1.20 ms | 1.11 ms | 30.15x |
| `fconvert_qq_to_yy_sum` | fconvert(Yearly, t, method='sum') | 66.19 µs | 1.91 µs | 1.27 ms | 1.18 ms | 34.66x |
| `fconvert_yy_to_qq_const` | fconvert(Quarterly, t, method='const') (higher) | 23.06 µs | 580.5 ns | 961.90 µs | 1.54 ms | 39.73x |
| `fconvert_mm_to_qq_mean` | fconvert(Quarterly, monthly_t, method='mean') | 64.01 µs | 2.49 µs | 1.16 ms | 1.23 ms | 25.71x |
| `fconvert_qq_to_yy_mean_numpy` | aggregate_groups_numpy 25x4 mean - NumPy kernel | 281.84 µs | 2.04 µs | n/a | n/a | 138.16x |
| `fconvert_qq_to_yy_sum_numpy` | aggregate_groups_numpy 25x4 sum - NumPy kernel | 137.80 µs | 1.90 µs | n/a | n/a | 72.52x |
| `fconvert_mm_to_qq_mean_numpy` | aggregate_groups_numpy 40x3 mean - NumPy kernel | 461.29 µs | 2.48 µs | n/a | n/a | 186.00x |
| `rec_ar2_100` | AR(2) over 100 quarters — general rec + lambda | 1.32 ms | 1.56 µs | 7.70 ms | 116.73 µs | 847.85x |
| `rec_linear_ar2_100_pylist` | AR(2) over 100 — rec_linear, pure-Python list | 28.91 µs | 1.59 µs | n/a | n/a | 18.18x |
| `rec_linear_ar2_100_numpy` | AR(2) over 100 — rec_linear NumPy kernel | 283.73 µs | 1.54 µs | n/a | n/a | 183.68x |
| `workspace_merge_5_series` | Workspace merge: 5 + 5 series | 4.41 µs | 11.10 µs | n/a | n/a | 0.40x |
| `workspace_filter_5_series` | Workspace filter: 10 down to 5 series | 8.10 µs | 17.20 µs | n/a | n/a | 0.47x |
| `mixed_freq_qq_minus_mm_mean` | qq_gdp - fconvert(Q, mm_cpi, mean) — mixed freq | 104.75 µs | 8.90 µs | 1.38 ms | 3.54 ms | 11.77x |
| `mixed_freq_pipeline_three_freq` | Y+Q+M → quarterly via fconvert — mixed freq | 161.90 µs | 7.76 µs | 2.53 ms | 6.01 ms | 20.87x |
| `rec_linear_ar2_100_cython` | AR(2) over 100 quarters — rec_linear Cython kernel | 4.61 µs | 1.54 µs | n/a | n/a | 2.98x |
| `indexing_lookup_100_cython` | gather_cython(values, ix) — Cython kernel | 4.32 µs | 258.1 ns | n/a | n/a | 16.73x |
| `mean_quarterly_100_cython` | mean_cython(values) — Cython kernel | 1.57 µs | 63.7 ns | n/a | n/a | 24.57x |
| `std_quarterly_100_cython` | std_cython(values, 1) — Cython kernel | 1.81 µs | 129.6 ns | n/a | n/a | 13.96x |
| `cor_two_tseries_cython` | cor_cython(x, y) — Cython kernel | 2.81 µs | 198.1 ns | n/a | n/a | 14.21x |
| `fconvert_qq_to_yy_mean_cython` | aggregate_groups_cython 25x4 mean - Cython kernel | 5.29 µs | 2.03 µs | n/a | n/a | 2.61x |
| `fconvert_qq_to_yy_sum_cython` | aggregate_groups_cython 25x4 sum - Cython kernel | 5.31 µs | 1.90 µs | n/a | n/a | 2.79x |
| `fconvert_mm_to_qq_mean_cython` | aggregate_groups_cython 40x3 mean - Cython kernel | 5.35 µs | 2.41 µs | n/a | n/a | 2.22x |
