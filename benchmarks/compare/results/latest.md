| Scenario | Description | tsecon (median) | Julia (median) | Pandas (median) | Polars (median) | Ratio (Py / Jl) |
|---|---|---:|---:|---:|---:|---:|
| `construct_tseries_qq_100` | TSeries(qq, arr) from length-100 ndarray | 2.47 µs | 21.8 ns | 34.28 µs | 44.61 µs | 113.07x |
| `construct_mvts_qq_100x5` | MVTSeries(qq, 5 cols, 100x5 ndarray) | 20.95 µs | 1.33 µs | 150.77 µs | 111.43 µs | 15.79x |
| `indexing_mit_lookup_100` | Sum t[mit] over 100 keys | 181.35 µs | 287.1 ns | 2.00 ms | 28.27 ms | 631.77x |
| `indexing_int_lookup_100` | Sum t[int] over 100 keys | 131.30 µs | 239.7 ns | n/a | n/a | 547.75x |
| `indexing_mitrange_slice` | t[MITRange] — single 60-period slice | 9.76 µs | 145.2 ns | n/a | n/a | 67.22x |
| `indexing_mvts_column` | mvts['c'] — column access | 994.5 ns | 15.4 ns | n/a | n/a | 64.45x |
| `indexing_lookup_100_api` | lookup(t, mit_keys) — public vectorised API | 133.33 µs | 522.7 ns | n/a | n/a | 255.08x |
| `indexing_lookup_100_numpy` | gather_numpy(values, ix) — NumPy kernel | 4.85 µs | 264.1 ns | n/a | n/a | 18.36x |
| `arith_add_misaligned` | 100Q + 100Q with 50Q overlap | 25.57 µs | 4.71 µs | 454.54 µs | 3.08 ms | 5.43x |
| `arith_add_aligned` | 100Q + 100Q same range | 25.16 µs | 4.37 µs | 89.70 µs | 5.27 µs | 5.76x |
| `arith_mul_scalar` | t * 2.5 | 22.82 µs | 3.94 µs | 80.04 µs | 17.07 µs | 5.79x |
| `shift_quarterly_lag1` | shift(t, -1) | 7.18 µs | 247.1 ns | 72.04 µs | 279.20 µs | 29.05x |
| `diff_quarterly` | diff(t) | 33.61 µs | 4.97 µs | 82.43 µs | 285.73 µs | 6.77x |
| `pct_quarterly` | pct(t) | 81.20 µs | 7.11 µs | 299.00 µs | 286.56 µs | 11.41x |
| `mean_quarterly_100` | mean(t) | 3.39 µs | 56.1 ns | 35.29 µs | 858.4 ns | 60.48x |
| `std_quarterly_100` | std(t) | 3.83 µs | 126.2 ns | 74.73 µs | 1.45 µs | 30.37x |
| `cor_two_tseries` | cor(a, b) on two TSeries | 9.88 µs | 197.5 ns | 319.65 µs | 302.91 µs | 50.02x |
| `cor_mvts_5_columns` | cor(mvts) — 5x5 corr matrix | 112.34 µs | 28.90 µs | n/a | n/a | 3.89x |
| `cov_mvts_5_columns` | cov(mvts) — 5x5 cov matrix | 78.65 µs | 28.70 µs | n/a | n/a | 2.74x |
| `mean_quarterly_100_numpy` | mean_numpy(values) — NumPy kernel | 11.81 µs | 68.0 ns | n/a | n/a | 173.55x |
| `std_quarterly_100_numpy` | std_numpy(values, 1) — NumPy kernel | 32.68 µs | 127.4 ns | n/a | n/a | 256.46x |
| `cor_two_tseries_numpy` | cor_numpy(x, y) — NumPy kernel | 104.56 µs | 234.1 ns | n/a | n/a | 446.65x |
| `moving_average_quarterly_4` | moving_average(t, 4) | 17.76 µs | 11.80 µs | 171.28 µs | 81.53 µs | 1.51x |
| `moving_sum_quarterly_4` | moving_sum(t, 4) | 14.78 µs | 16.30 µs | 169.16 µs | 80.46 µs | 0.91x |
| `undiff_quarterly` | undiff(t) | 103.89 µs | 10.25 µs | n/a | n/a | 10.14x |
| `fconvert_qq_to_yy_mean` | fconvert(Yearly, t, method='mean') | 64.12 µs | 1.19 µs | 1.21 ms | 1.10 ms | 53.88x |
| `fconvert_qq_to_yy_sum` | fconvert(Yearly, t, method='sum') | 65.10 µs | 1.93 µs | 1.27 ms | 1.29 ms | 33.73x |
| `fconvert_yy_to_qq_const` | fconvert(Quarterly, t, method='const') (higher) | 22.67 µs | 592.9 ns | 933.64 µs | 1.74 ms | 38.24x |
| `fconvert_mm_to_qq_mean` | fconvert(Quarterly, monthly_t, method='mean') | 65.55 µs | 2.38 µs | 1.10 ms | 1.08 ms | 27.57x |
| `fconvert_qq_to_yy_mean_numpy` | aggregate_groups_numpy 25x4 mean - NumPy kernel | 266.40 µs | 2.10 µs | n/a | n/a | 126.86x |
| `fconvert_qq_to_yy_sum_numpy` | aggregate_groups_numpy 25x4 sum - NumPy kernel | 123.24 µs | 1.96 µs | n/a | n/a | 62.88x |
| `fconvert_mm_to_qq_mean_numpy` | aggregate_groups_numpy 40x3 mean - NumPy kernel | 424.13 µs | 2.46 µs | n/a | n/a | 172.72x |
| `rec_ar2_100` | AR(2) over 100 quarters — general rec + lambda | 1.30 ms | 1.55 µs | 7.62 ms | 114.53 µs | 837.33x |
| `rec_linear_ar2_100_pylist` | AR(2) over 100 — rec_linear, pure-Python list | 27.95 µs | 1.57 µs | n/a | n/a | 17.80x |
| `rec_linear_ar2_100_numpy` | AR(2) over 100 — rec_linear NumPy kernel | 286.13 µs | 1.58 µs | n/a | n/a | 181.10x |
| `workspace_merge_5_series` | Workspace merge: 5 + 5 series | 4.34 µs | 11.20 µs | n/a | n/a | 0.39x |
| `workspace_filter_5_series` | Workspace filter: 10 down to 5 series | 8.07 µs | 17.10 µs | n/a | n/a | 0.47x |
| `mixed_freq_qq_minus_mm_mean` | qq_gdp - fconvert(Q, mm_cpi, mean) — mixed freq | 102.63 µs | 9.13 µs | 1.36 ms | 3.48 ms | 11.24x |
| `mixed_freq_pipeline_three_freq` | Y+Q+M → quarterly via fconvert — mixed freq | 160.81 µs | 7.90 µs | 2.51 ms | 7.04 ms | 20.36x |
| `rec_linear_ar2_100_cython` | AR(2) over 100 quarters — rec_linear Cython kernel | 4.52 µs | 1.57 µs | n/a | n/a | 2.88x |
| `indexing_lookup_100_cython` | gather_cython(values, ix) — Cython kernel | 4.30 µs | 276.1 ns | n/a | n/a | 15.58x |
| `mean_quarterly_100_cython` | mean_cython(values) — Cython kernel | 1.55 µs | 66.1 ns | n/a | n/a | 23.48x |
| `std_quarterly_100_cython` | std_cython(values, 1) — Cython kernel | 1.89 µs | 125.8 ns | n/a | n/a | 15.02x |
| `cor_two_tseries_cython` | cor_cython(x, y) — Cython kernel | 2.77 µs | 217.0 ns | n/a | n/a | 12.78x |
| `fconvert_qq_to_yy_mean_cython` | aggregate_groups_cython 25x4 mean - Cython kernel | 5.29 µs | 2.03 µs | n/a | n/a | 2.61x |
| `fconvert_qq_to_yy_sum_cython` | aggregate_groups_cython 25x4 sum - Cython kernel | 5.29 µs | 1.97 µs | n/a | n/a | 2.68x |
| `fconvert_mm_to_qq_mean_cython` | aggregate_groups_cython 40x3 mean - Cython kernel | 5.45 µs | 2.34 µs | n/a | n/a | 2.33x |
