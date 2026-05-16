| Scenario | Description | Python (median) | Julia (median) | Ratio (Py / Jl) |
|---|---|---:|---:|---:|
| `construct_tseries_qq_100` | TSeries(qq, arr) from length-100 ndarray | 1.18 µs | 12.0 ns | 97.97x |
| `construct_mvts_qq_100x5` | MVTSeries(qq, 5 cols, 100x5 ndarray) | 10.13 µs | 724.0 ns | 13.99x |
| `indexing_mit_lookup_100` | Sum t[mit] over 100 keys | 89.96 µs | 150.3 ns | 598.68x |
| `indexing_int_lookup_100` | Sum t[int] over 100 keys | 64.38 µs | 124.0 ns | 519.09x |
| `indexing_mitrange_slice` | t[MITRange] — single 60-period slice | 4.82 µs | 86.7 ns | 55.53x |
| `indexing_mvts_column` | mvts['c'] — column access | 487.1 ns | 8.0 ns | 60.89x |
| `indexing_lookup_100_api` | lookup(t, mit_keys) — public vectorised API | 63.98 µs | 282.3 ns | 226.63x |
| `indexing_lookup_100_numpy` | gather_numpy(values, ix) — NumPy kernel | 2.26 µs | 143.5 ns | 15.74x |
| `arith_add_misaligned` | 100Q + 100Q with 50Q overlap | 11.98 µs | 2.15 µs | 5.57x |
| `arith_add_aligned` | 100Q + 100Q same range | 11.93 µs | 1.99 µs | 5.99x |
| `arith_mul_scalar` | t * 2.5 | 10.80 µs | 1.87 µs | 5.78x |
| `shift_quarterly_lag1` | shift(t, -1) | 3.50 µs | 164.9 ns | 21.20x |
| `diff_quarterly` | diff(t) | 17.33 µs | 2.62 µs | 6.61x |
| `pct_quarterly` | pct(t) | 42.52 µs | 3.37 µs | 12.63x |
| `mean_quarterly_100` | mean(t) | 5.76 µs | 30.5 ns | 189.02x |
| `std_quarterly_100` | std(t) | 15.43 µs | 65.3 ns | 236.11x |
| `cor_two_tseries` | cor(a, b) on two TSeries | 55.45 µs | 102.7 ns | 539.99x |
| `cor_mvts_5_columns` | cor(mvts) — 5x5 corr matrix | 54.33 µs | 14.80 µs | 3.67x |
| `cov_mvts_5_columns` | cov(mvts) — 5x5 cov matrix | 36.74 µs | 15.30 µs | 2.40x |
| `moving_average_quarterly_4` | moving_average(t, 4) | 8.13 µs | 11.10 µs | 0.73x |
| `moving_sum_quarterly_4` | moving_sum(t, 4) | 7.18 µs | 8.68 µs | 0.83x |
| `undiff_quarterly` | undiff(t) | 50.03 µs | 5.35 µs | 9.35x |
| `fconvert_qq_to_yy_mean` | fconvert(Yearly, t, method='mean') | 169.66 µs | 1.00 µs | 168.95x |
| `fconvert_qq_to_yy_sum` | fconvert(Yearly, t, method='sum') | 74.23 µs | 1.04 µs | 71.52x |
| `fconvert_yy_to_qq_const` | fconvert(Quarterly, t, method='const') (higher) | 12.14 µs | 350.7 ns | 34.62x |
| `fconvert_mm_to_qq_mean` | fconvert(Quarterly, monthly_t, method='mean') | 278.54 µs | 1.25 µs | 222.61x |
| `rec_ar2_100` | AR(2) over 100 quarters — general rec + lambda | 613.17 µs | 800.6 ns | 765.93x |
| `rec_linear_ar2_100_pylist` | AR(2) over 100 — rec_linear, pure-Python list | 13.76 µs | 800.6 ns | 17.19x |
| `rec_linear_ar2_100_numpy` | AR(2) over 100 — rec_linear NumPy kernel | 143.86 µs | 800.6 ns | 179.70x |
| `workspace_merge_5_series` | Workspace merge: 5 + 5 series | 2.15 µs | 5.06 µs | 0.43x |
| `workspace_filter_5_series` | Workspace filter: 10 down to 5 series | 4.07 µs | 8.78 µs | 0.46x |
| `rec_linear_ar2_100_cython` | AR(2) over 100 quarters — rec_linear Cython kernel | 2.23 µs | 763.4 ns | 2.93x |
| `indexing_lookup_100_cython` | gather_cython(values, ix) — Cython kernel | 2.05 µs | 169.7 ns | 12.06x |
