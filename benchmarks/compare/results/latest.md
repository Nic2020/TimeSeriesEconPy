| Scenario | Description | tsecon (median) | Julia (median) | Pandas (median) | Polars (median) | Ratio (Py / Jl) |
|---|---|---:|---:|---:|---:|---:|
| `construct_tseries_qq_100` | TSeries(qq, arr) from length-100 ndarray | 1.49 µs | 12.3 ns | 23.08 µs | 28.27 µs | 120.75x |
| `construct_mvts_qq_100x5` | MVTSeries(qq, 5 cols, 100x5 ndarray) | 12.50 µs | 896.4 ns | 125.25 µs | 71.40 µs | 13.94x |
| `indexing_mit_lookup_100` | Sum t[mit] over 100 keys | 108.07 µs | 126.2 ns | 1.21 ms | 23.79 ms | 856.21x |
| `indexing_int_lookup_100` | Sum t[int] over 100 keys | 79.70 µs | 110.4 ns | n/a | n/a | 721.90x |
| `indexing_mitrange_slice` | t[MITRange] — single 60-period slice | 5.81 µs | 108.4 ns | n/a | n/a | 53.65x |
| `indexing_mvts_column` | mvts['c'] — column access | 604.6 ns | 10.5 ns | n/a | n/a | 57.58x |
| `indexing_lookup_100_api` | lookup(t, mit_keys) — public vectorised API | 82.25 µs | 374.0 ns | n/a | n/a | 219.92x |
| `indexing_lookup_100_numpy` | gather_numpy(values, ix) — NumPy kernel | 2.76 µs | 187.4 ns | n/a | n/a | 14.71x |
| `arith_add_misaligned` | 100Q + 100Q with 50Q overlap | 17.62 µs | 3.02 µs | 367.28 µs | 2.03 ms | 5.83x |
| `arith_add_aligned` | 100Q + 100Q same range | 17.43 µs | 2.67 µs | 67.88 µs | 3.61 µs | 6.54x |
| `arith_mul_scalar` | t * 2.5 | 15.98 µs | 2.42 µs | 64.64 µs | 10.84 µs | 6.60x |
| `shift_quarterly_lag1` | shift(t, -1) | 4.20 µs | 144.7 ns | 61.80 µs | 228.06 µs | 29.03x |
| `lead_quarterly_lag1` | lead(t, 1) | 4.52 µs | 201.4 ns | 65.07 µs | 228.60 µs | 22.46x |
| `diff_quarterly` | diff(t) | 24.26 µs | 3.23 µs | 69.54 µs | 251.23 µs | 7.52x |
| `pct_quarterly` | pct(t) | 59.39 µs | 4.73 µs | 247.72 µs | 246.14 µs | 12.56x |
| `ytypct_quarterly_100` | ytypct(t) — year-on-year % | 58.59 µs | 3.35 µs | 322.86 µs | 273.10 µs | 17.49x |
| `mean_quarterly_100` | mean(t) | 2.08 µs | 34.3 ns | 25.75 µs | 480.3 ns | 60.61x |
| `std_quarterly_100` | std(t) | 2.29 µs | 83.9 ns | 60.76 µs | 861.1 ns | 27.28x |
| `quantile_quarterly_100` | quantile(t, 0.5) — median | 88.41 µs | 511.6 ns | 452.59 µs | 1.34 µs | 172.82x |
| `cor_two_tseries` | cor(a, b) on two TSeries | 6.39 µs | 125.1 ns | 274.19 µs | 256.00 µs | 51.07x |
| `cov_two_tseries` | cov(a, b) on two TSeries | 71.85 µs | 378.8 ns | 233.80 µs | 256.13 µs | 189.70x |
| `cor_mvts_5_columns` | cor(mvts) — 5x5 corr matrix | 96.70 µs | 20.50 µs | n/a | n/a | 4.72x |
| `cov_mvts_5_columns` | cov(mvts) — 5x5 cov matrix | 69.98 µs | 20.30 µs | n/a | n/a | 3.45x |
| `mean_quarterly_100_numpy` | mean_numpy(values) — NumPy kernel | 7.54 µs | 35.6 ns | n/a | n/a | 211.47x |
| `std_quarterly_100_numpy` | std_numpy(values, 1) — NumPy kernel | 23.13 µs | 82.9 ns | n/a | n/a | 279.22x |
| `cor_two_tseries_numpy` | cor_numpy(x, y) — NumPy kernel | 90.16 µs | 127.9 ns | n/a | n/a | 704.88x |
| `moving_average_quarterly_4` | moving_average(t, 4) | 11.74 µs | 12.10 µs | 138.71 µs | 73.80 µs | 0.97x |
| `moving_sum_quarterly_4` | moving_sum(t, 4) | 9.79 µs | 9.17 µs | 137.12 µs | 76.13 µs | 1.07x |
| `undiff_quarterly` | undiff(t) | 74.84 µs | 5.72 µs | n/a | n/a | 13.09x |
| `fconvert_qq_to_yy_mean` | fconvert(Yearly, t, method='mean') | 221.71 µs | 1.15 µs | 891.17 µs | 629.51 µs | 192.79x |
| `fconvert_qq_to_yy_sum` | fconvert(Yearly, t, method='sum') | 127.52 µs | 1.08 µs | 940.61 µs | 764.04 µs | 118.26x |
| `fconvert_yy_to_qq_const` | fconvert(Quarterly, t, method='const') (higher) | 14.02 µs | 330.9 ns | 725.30 µs | 1.32 ms | 42.37x |
| `fconvert_yy_to_qq_linear` | fconvert(Quarterly, t, method='linear') (higher) | 330.79 µs | 2.57 µs | 1.26 ms | 1.29 ms | 128.88x |
| `fconvert_yy_to_qq_even` | fconvert(Quarterly, t, method='even') (higher) | 17.18 µs | 415.4 ns | 849.03 µs | 1.45 ms | 41.37x |
| `fconvert_mm_to_qq_mean` | fconvert(Quarterly, monthly_t, method='mean') | 327.02 µs | 1.42 µs | 862.90 µs | 671.28 µs | 230.30x |
| `fconvert_qq_to_yy_mean_numpy` | aggregate_groups_numpy 25x4 mean - NumPy kernel | 166.90 µs | 1.16 µs | n/a | n/a | 143.88x |
| `fconvert_qq_to_yy_sum_numpy` | aggregate_groups_numpy 25x4 sum - NumPy kernel | 76.04 µs | 1.06 µs | n/a | n/a | 71.57x |
| `fconvert_mm_to_qq_mean_numpy` | aggregate_groups_numpy 40x3 mean - NumPy kernel | 262.11 µs | 1.39 µs | n/a | n/a | 188.57x |
| `rec_ar2_100` | AR(2) over 100 quarters — general rec + lambda | 780.05 µs | 640.7 ns | 5.56 ms | 68.89 µs | 1217.42x |
| `rec_linear_ar2_100_pylist` | AR(2) over 100 — rec_linear, pure-Python list | 15.91 µs | 640.9 ns | n/a | n/a | 24.82x |
| `rec_linear_ar2_100_numpy` | AR(2) over 100 — rec_linear NumPy kernel | 166.92 µs | 638.3 ns | n/a | n/a | 261.49x |
| `workspace_merge_5_series` | Workspace merge: 5 + 5 series | 2.60 µs | 6.03 µs | n/a | n/a | 0.43x |
| `workspace_filter_5_series` | Workspace filter: 10 down to 5 series | 4.84 µs | 9.27 µs | n/a | n/a | 0.52x |
| `mixed_freq_qq_minus_mm_mean` | qq_gdp - fconvert(Q, mm_cpi, mean) — mixed freq | 812.76 µs | 6.05 µs | 1.10 ms | 2.09 ms | 134.34x |
| `mixed_freq_pipeline_three_freq` | Y+Q+M → quarterly via fconvert — mixed freq | 859.29 µs | 5.21 µs | 2.07 ms | 4.60 ms | 164.80x |
| `rec_linear_ar2_100_cython` | AR(2) over 100 quarters — rec_linear Cython kernel | 2.64 µs | 640.2 ns | n/a | n/a | 4.12x |
| `indexing_lookup_100_cython` | gather_cython(values, ix) — Cython kernel | 2.49 µs | 187.3 ns | n/a | n/a | 13.30x |
| `mean_quarterly_100_cython` | mean_cython(values) — Cython kernel | 889.6 ns | 34.6 ns | n/a | n/a | 25.73x |
| `std_quarterly_100_cython` | std_cython(values, 1) — Cython kernel | 1.05 µs | 84.1 ns | n/a | n/a | 12.43x |
| `cor_two_tseries_cython` | cor_cython(x, y) — Cython kernel | 1.68 µs | 130.4 ns | n/a | n/a | 12.85x |
| `fconvert_qq_to_yy_mean_cython` | aggregate_groups_cython 25x4 mean - Cython kernel | 3.17 µs | 1.17 µs | n/a | n/a | 2.71x |
| `fconvert_qq_to_yy_sum_cython` | aggregate_groups_cython 25x4 sum - Cython kernel | 3.12 µs | 1.09 µs | n/a | n/a | 2.85x |
| `fconvert_mm_to_qq_mean_cython` | aggregate_groups_cython 40x3 mean - Cython kernel | 3.19 µs | 1.44 µs | n/a | n/a | 2.21x |
