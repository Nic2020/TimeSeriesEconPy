# SPDX-License-Identifier: MIT
#
# Benchmark scenarios for the Julia ↔ Python comparison harness.
#
# Mirrors `../scenarios.py`. Each scenario provides a `*_setup()` returning a
# state object and a `*_run(state)` performing the operation being timed. The
# pair is registered in the `SCENARIOS` / `DESCRIPTION` dictionaries at the
# bottom of the file.
#
# `runner.jl` invokes the chosen scenario with `BenchmarkTools.@belapsed` to
# match Python's `timeit.repeat(..., number=N, repeat=R)` semantics — both
# report the *minimum* execution time over many repetitions, which is the
# noise-resistant statistic recommended for microbenchmarks.

module Scenarios

using TimeSeriesEcon
using Statistics
using Random
import TimeSeriesEcon: qq, mm, yy

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

construct_tseries_qq_100_setup() = (start = qq(2020, 1), values = collect(0.0:99.0))

construct_tseries_qq_100_run(state) = TSeries(state.start, state.values)

# ---------------------------------------------------------------------------
# Indexing (sum 100 MIT positions)
# ---------------------------------------------------------------------------

function indexing_mit_lookup_100_setup()
    start = qq(2020, 1)
    t = TSeries(start, collect(0.0:99.0))
    keys = [start + i for i in 0:99]
    return (t = t, keys = keys)
end

function indexing_mit_lookup_100_run(state)
    s = 0.0
    for k in state.keys
        s += state.t[k]
    end
    return s
end

# ---------------------------------------------------------------------------
# Vectorised lookup — Julia counterpart for the M1.5 second Cython port.
# Python provides three scenarios (full API / NumPy kernel / Cython kernel)
# that all map to the same Julia operation: a vectorised fancy-indexing
# gather. The Julia idiom is just ``state.t.values[state.indices]`` (or
# equivalently ``state.t[state.keys]`` for MIT keys — both compile to a
# similar gather). All three Python flavors share the single Julia run
# here, matching the rec_linear precedent.
# ---------------------------------------------------------------------------

function indexing_lookup_100_api_setup()
    start = qq(2020, 1)
    t = TSeries(start, collect(0.0:99.0))
    keys = [start + i for i in 0:99]
    return (t = t, keys = keys)
end

indexing_lookup_100_api_run(state) = state.t[state.keys]

function indexing_lookup_100_kernel_setup()
    values = collect(0.0:99.0)
    indices = collect(1:100)  # Julia is 1-indexed
    return (values = values, indices = indices)
end

indexing_lookup_100_kernel_run(state) = state.values[state.indices]

indexing_lookup_100_numpy_setup() = indexing_lookup_100_kernel_setup()
indexing_lookup_100_numpy_run(state) = indexing_lookup_100_kernel_run(state)
indexing_lookup_100_cython_setup() = indexing_lookup_100_kernel_setup()
indexing_lookup_100_cython_run(state) = indexing_lookup_100_kernel_run(state)

# ---------------------------------------------------------------------------
# Arithmetic with misalignment (50-period overlap)
# ---------------------------------------------------------------------------

function arith_add_misaligned_setup()
    a = TSeries(qq(2020, 1), collect(0.0:99.0))
    b = TSeries(qq(2032, 1), 0.5 .* collect(0.0:99.0))
    return (a = a, b = b)
end

arith_add_misaligned_run(state) = state.a .+ state.b

# ---------------------------------------------------------------------------
# Shift
# ---------------------------------------------------------------------------

shift_quarterly_lag1_setup() = (t = TSeries(qq(2020, 1), collect(0.0:99.0)),)

shift_quarterly_lag1_run(state) = shift(state.t, -1)

# ---------------------------------------------------------------------------
# Moving average (n = 4)
# ---------------------------------------------------------------------------

moving_average_quarterly_4_setup() = (t = TSeries(qq(2020, 1), collect(0.0:99.0)),)

moving_average_quarterly_4_run(state) = moving_average(state.t, 4)

# ---------------------------------------------------------------------------
# Frequency conversion (qq → yy, method=:mean)
# ---------------------------------------------------------------------------

fconvert_qq_to_yy_mean_setup() = (t = TSeries(qq(2020, 1), collect(0.0:99.0)),)

fconvert_qq_to_yy_mean_run(state) = fconvert(Yearly, state.t; method = :mean)

# ---------------------------------------------------------------------------
# rec — AR(2) over a 100-period quarterly range (M1.5 Cython candidate)
# ---------------------------------------------------------------------------

function rec_ar2_100_setup()
    start = qq(2020, 1)
    target = TSeries(start, zeros(Float64, 102))
    target[start] = 1.0
    target[start + 1] = 1.0
    return (target = target, start = start)
end

function rec_ar2_100_run(state)
    target = state.target
    start = state.start
    @rec t = (start + 2):(start + 101) target[t] = 0.5 * target[t - 1] + 0.3 * target[t - 2]
    return target
end

# ---------------------------------------------------------------------------
# rec — backcasting via reversed range (M1.6.1). Julia's `@rec` macro
# handles the negative-step `start:-1:stop` range natively, so the Julia
# row is one line. See ../scenarios.py for the Python counterpart.
# ---------------------------------------------------------------------------

function rec_backcasting_via_lambda_setup()
    start = qq(2020, 1)
    n = 100
    target = TSeries(start, zeros(Float64, n))
    target[start + (n - 1)] = 100.0
    return (target = target, start = start, n = n)
end

function rec_backcasting_via_lambda_run(state)
    target = state.target
    start = state.start
    n = state.n
    @rec t = (start + (n - 2)):-1:start target[t] = target[t + 1] - 0.5
    return target
end

# ---------------------------------------------------------------------------
# rec_linear three-flavor scenarios — see ../scenarios.py for the framing.
# Julia has no kernel split (the @rec macro inlines into native code at
# compile time), so all Python rec_linear variants share the same Julia
# counterpart. The aliases below let the harness invoke "Julia for the
# rec_linear_ar2_100_numpy scenario" without a special-case in run.py.
# ---------------------------------------------------------------------------

rec_linear_ar2_100_numpy_setup() = rec_ar2_100_setup()
rec_linear_ar2_100_numpy_run(state) = rec_ar2_100_run(state)
rec_linear_ar2_100_cython_setup() = rec_ar2_100_setup()
rec_linear_ar2_100_cython_run(state) = rec_ar2_100_run(state)
rec_linear_ar2_100_pylist_setup() = rec_ar2_100_setup()
rec_linear_ar2_100_pylist_run(state) = rec_ar2_100_run(state)

# ---------------------------------------------------------------------------
# Inventory expansion (session 18) — matches scenarios in ../scenarios.py.
# ---------------------------------------------------------------------------

function construct_mvts_qq_100x5_setup()
    return (
        start = qq(2020, 1),
        cols = (:a, :b, :c, :d, :e),
        values = reshape(collect(0.0:499.0), (100, 5)),
    )
end

construct_mvts_qq_100x5_run(state) =
    MVTSeries(state.start, state.cols, state.values)

function indexing_int_lookup_100_setup()
    t = TSeries(qq(2020, 1), collect(0.0:99.0))
    return (t = t, keys = 1:100)
end

function indexing_int_lookup_100_run(state)
    s = 0.0
    for i in state.keys
        s += state.t[i]
    end
    return s
end

function indexing_mitrange_slice_setup()
    start = qq(2020, 1)
    t = TSeries(start, collect(0.0:99.0))
    return (t = t, rng = (start + 20):(start + 79))
end

indexing_mitrange_slice_run(state) = state.t[state.rng]

function indexing_mvts_column_setup()
    mvts = MVTSeries(
        qq(2020, 1),
        (:a, :b, :c, :d, :e),
        reshape(collect(0.0:499.0), (100, 5)),
    )
    return (mvts = mvts,)
end

indexing_mvts_column_run(state) = state.mvts.c

function arith_add_aligned_setup()
    start = qq(2020, 1)
    a = TSeries(start, collect(0.0:99.0))
    b = TSeries(start, 0.5 .* collect(0.0:99.0))
    return (a = a, b = b)
end

arith_add_aligned_run(state) = state.a .+ state.b

arith_mul_scalar_setup() = (t = TSeries(qq(2020, 1), collect(0.0:99.0)),)

arith_mul_scalar_run(state) = state.t .* 2.5

diff_quarterly_setup() = (t = TSeries(qq(2020, 1), collect(0.0:99.0)),)

diff_quarterly_run(state) = diff(state.t)

# Avoid zero-division by starting at 1.0 (mirrors the Python counterpart).
pct_quarterly_setup() = (t = TSeries(qq(2020, 1), collect(1.0:100.0)),)

pct_quarterly_run(state) = pct(state.t)

mean_quarterly_100_setup() = (t = TSeries(qq(2020, 1), collect(0.0:99.0)),)

mean_quarterly_100_run(state) = mean(state.t)

std_quarterly_100_setup() = (t = TSeries(qq(2020, 1), collect(0.0:99.0)),)

std_quarterly_100_run(state) = std(state.t)

# Use deterministic seeded RNG so the per-call work is constant across runs.
# The integer seed (20260515) is shared with the Python scenarios for
# traceability, but the resulting byte-level sequences differ — MersenneTwister
# (MT19937) and NumPy's `default_rng` (PCG64) are different algorithms, so the
# same seed yields different floats. This is benign here: every scenario seeded
# this way (cor / cov / quantile / std on length-100 vectors and 100x5 matrices)
# is timed on an operation whose cost depends only on array shape and dtype,
# not on the values. Cross-backend *timing* comparisons are still apples-to-apples;
# cross-backend *numerical* parity is not asserted from this harness.
function _seeded_normal(n::Int)
    rng = MersenneTwister(20260515)
    return randn(rng, n)
end

function _seeded_normal_mat(rows::Int, cols::Int)
    rng = MersenneTwister(20260515)
    return randn(rng, rows, cols)
end

function cor_two_tseries_setup()
    start = qq(2020, 1)
    rng = MersenneTwister(20260515)
    a = TSeries(start, randn(rng, 100))
    b = TSeries(start, randn(rng, 100))
    return (a = a, b = b)
end

cor_two_tseries_run(state) = cor(state.a, state.b)

# Stats kernel-direct aliases — Julia has no kernel split (mean / std / cor on
# a TSeries / Vector all inline directly), so the three Python flavors
# (public API + NumPy kernel + Cython kernel) all share the same Julia
# counterpart. Same precedent as the rec_linear_*_{numpy,cython} aliases
# above.

mean_quarterly_100_numpy_setup() = mean_quarterly_100_setup()
mean_quarterly_100_numpy_run(state) = mean_quarterly_100_run(state)
mean_quarterly_100_cython_setup() = mean_quarterly_100_setup()
mean_quarterly_100_cython_run(state) = mean_quarterly_100_run(state)

std_quarterly_100_numpy_setup() = std_quarterly_100_setup()
std_quarterly_100_numpy_run(state) = std_quarterly_100_run(state)
std_quarterly_100_cython_setup() = std_quarterly_100_setup()
std_quarterly_100_cython_run(state) = std_quarterly_100_run(state)

cor_two_tseries_numpy_setup() = cor_two_tseries_setup()
cor_two_tseries_numpy_run(state) = cor_two_tseries_run(state)
cor_two_tseries_cython_setup() = cor_two_tseries_setup()
cor_two_tseries_cython_run(state) = cor_two_tseries_run(state)

function cor_mvts_5_columns_setup()
    rng = MersenneTwister(20260515)
    mvts = MVTSeries(qq(2020, 1), (:a, :b, :c, :d, :e), randn(rng, 100, 5))
    return (mvts = mvts,)
end

cor_mvts_5_columns_run(state) = cor(state.mvts)

function cov_mvts_5_columns_setup()
    rng = MersenneTwister(20260515)
    mvts = MVTSeries(qq(2020, 1), (:a, :b, :c, :d, :e), randn(rng, 100, 5))
    return (mvts = mvts,)
end

cov_mvts_5_columns_run(state) = cov(state.mvts)

# MVTSeries axis= reductions (M1.6.3f — closes G11). Julia's
# Statistics.mean(::MVTSeries; dims=1) reduces along rows (per-column →
# 1-row MVTSeries); dims=2 reduces along columns (per-row → row vector).
# Our Python port uses NumPy's axis= convention (axis=0 ≡ dims=1, axis=1 ≡
# dims=2), so the harness compares axis=0 Python ↔ dims=1 Julia and
# axis=1 Python ↔ dims=2 Julia.
function mean_mvts_axis0_5cols_setup()
    rng = MersenneTwister(20260518)
    mvts = MVTSeries(qq(2020, 1), (:a, :b, :c, :d, :e), randn(rng, 100, 5))
    return (mvts = mvts,)
end

mean_mvts_axis0_5cols_run(state) = mean(state.mvts; dims=1)

function mean_mvts_axis1_100rows_setup()
    rng = MersenneTwister(20260518)
    mvts = MVTSeries(qq(2020, 1), (:a, :b, :c, :d, :e), randn(rng, 100, 5))
    return (mvts = mvts,)
end

mean_mvts_axis1_100rows_run(state) = mean(state.mvts; dims=2)

# ---------------------------------------------------------------------------
# F14 expansion (session 30) — quantile / cov(x,y) / ytypct / lead, plus the
# two missing higher-freq fconvert methods (linear, even). See
# claude_files/reviews/2026-05-16_holistic/F14_benchmark_coverage_gaps.md.
# ---------------------------------------------------------------------------

function quantile_quarterly_100_setup()
    rng = MersenneTwister(20260515)
    return (t = TSeries(qq(2020, 1), randn(rng, 100)),)
end

# Statistics.quantile rejects offset arrays — TSeries is offset-indexed via
# its MIT axis, so we pass the raw values vector. Python tsecon's quantile
# does the same internally (np.quantile on the underlying ndarray).
quantile_quarterly_100_run(state) = quantile(state.t.values, 0.5)

function cov_two_tseries_setup()
    rng = MersenneTwister(20260515)
    start = qq(2020, 1)
    a = TSeries(start, randn(rng, 100))
    b = TSeries(start, randn(rng, 100))
    return (a = a, b = b)
end

cov_two_tseries_run(state) = cov(state.a, state.b)

ytypct_quarterly_100_setup() = (t = TSeries(qq(2020, 1), collect(1.0:100.0)),)

ytypct_quarterly_100_run(state) = ytypct(state.t)

lead_quarterly_lag1_setup() = (t = TSeries(qq(2020, 1), collect(0.0:99.0)),)

lead_quarterly_lag1_run(state) = lead(state.t, 1)

fconvert_yy_to_qq_linear_setup() = (t = TSeries(yy(2020), collect(0.0:24.0)),)

fconvert_yy_to_qq_linear_run(state) = fconvert(Quarterly, state.t; method = :linear)

fconvert_yy_to_qq_even_setup() = (t = TSeries(yy(2020), collect(0.0:24.0)),)

fconvert_yy_to_qq_even_run(state) = fconvert(Quarterly, state.t; method = :even)

moving_sum_quarterly_4_setup() = (t = TSeries(qq(2020, 1), collect(0.0:99.0)),)

moving_sum_quarterly_4_run(state) = moving_sum(state.t, 4)

undiff_quarterly_setup() = (t = TSeries(qq(2020, 1), collect(0.0:99.0)),)

undiff_quarterly_run(state) = undiff(state.t)

fconvert_qq_to_yy_sum_setup() = (t = TSeries(qq(2020, 1), collect(0.0:99.0)),)

fconvert_qq_to_yy_sum_run(state) = fconvert(Yearly, state.t; method = :sum)

fconvert_yy_to_qq_const_setup() = (t = TSeries(yy(2020), collect(0.0:24.0)),)

fconvert_yy_to_qq_const_run(state) = fconvert(Quarterly, state.t; method = :const)

fconvert_mm_to_qq_mean_setup() = (t = TSeries(mm(2020, 1), collect(0.0:119.0)),)

fconvert_mm_to_qq_mean_run(state) = fconvert(Quarterly, state.t; method = :mean)

# fconvert kernel-direct aliases — Julia has no kernel split (frequency
# conversion inlines directly), so the three Python flavors (public API
# + NumPy kernel + Cython kernel) all share the same Julia counterpart.
# Same precedent as the rec_linear / stats *_{numpy,cython} aliases.

fconvert_qq_to_yy_mean_numpy_setup() = fconvert_qq_to_yy_mean_setup()
fconvert_qq_to_yy_mean_numpy_run(state) = fconvert_qq_to_yy_mean_run(state)
fconvert_qq_to_yy_mean_cython_setup() = fconvert_qq_to_yy_mean_setup()
fconvert_qq_to_yy_mean_cython_run(state) = fconvert_qq_to_yy_mean_run(state)

fconvert_qq_to_yy_sum_numpy_setup() = fconvert_qq_to_yy_sum_setup()
fconvert_qq_to_yy_sum_numpy_run(state) = fconvert_qq_to_yy_sum_run(state)
fconvert_qq_to_yy_sum_cython_setup() = fconvert_qq_to_yy_sum_setup()
fconvert_qq_to_yy_sum_cython_run(state) = fconvert_qq_to_yy_sum_run(state)

fconvert_mm_to_qq_mean_numpy_setup() = fconvert_mm_to_qq_mean_setup()
fconvert_mm_to_qq_mean_numpy_run(state) = fconvert_mm_to_qq_mean_run(state)
fconvert_mm_to_qq_mean_cython_setup() = fconvert_mm_to_qq_mean_setup()
fconvert_mm_to_qq_mean_cython_run(state) = fconvert_mm_to_qq_mean_run(state)

# undiff kernel-direct aliases — Julia has no kernel split (`undiff` inlines
# directly), so the three Python flavors (public API + NumPy kernel +
# Cython kernel) all share the same Julia counterpart. Same precedent as
# the rec_linear / stats / fconvert *_{numpy,cython} aliases.

undiff_quarterly_numpy_setup() = undiff_quarterly_setup()
undiff_quarterly_numpy_run(state) = undiff_quarterly_run(state)
undiff_quarterly_cython_setup() = undiff_quarterly_setup()
undiff_quarterly_cython_run(state) = undiff_quarterly_run(state)

function workspace_filter_5_series_setup()
    start = qq(2020, 1)
    arr = collect(0.0:39.0)
    w = Workspace()
    for name in (:a, :b, :c, :d, :e, :f, :g, :h, :i, :j)
        w[name] = TSeries(start, copy(arr))
    end
    return (w = w, keep = Set((:a, :b, :c, :d, :e)))
end

workspace_filter_5_series_run(state) =
    filter(((k, v),) -> k in state.keep, state.w)

# ---------------------------------------------------------------------------
# Workspace merge (5 series each)
# ---------------------------------------------------------------------------

function workspace_merge_5_series_setup()
    start = qq(2020, 1)
    arr = collect(0.0:39.0)
    w1 = Workspace()
    w2 = Workspace()
    for name in (:a, :b, :c, :d, :e)
        w1[name] = TSeries(start, copy(arr))
    end
    for name in (:f, :g, :h, :i, :j)
        w2[name] = TSeries(start, copy(arr))
    end
    return (w1 = w1, w2 = w2)
end

workspace_merge_5_series_run(state) = merge(state.w1, state.w2)

# ---------------------------------------------------------------------------
# Mixed-frequency scenarios — added 2026-05-17 for the pandas/polars
# 4-column comparison. Julia counterparts to the Python scenarios in
# ../scenarios.py § "Mixed-frequency". Julia's TSeries carries its own
# frequency type-parameter, so the mixed-freq pipeline reads exactly the
# same as the Python tsecon form: fconvert + ordinary arithmetic.
# ---------------------------------------------------------------------------

function mixed_freq_qq_minus_mm_mean_setup()
    return (
        gdp = TSeries(qq(2020, 1), collect(0.0:99.0)),
        cpi = TSeries(mm(2020, 1), collect(0.0:299.0)),
    )
end

mixed_freq_qq_minus_mm_mean_run(state) =
    state.gdp .- fconvert(Quarterly, state.cpi; method = :mean)

function mixed_freq_pipeline_three_freq_setup()
    return (
        unemp = TSeries(yy(2020), collect(0.0:24.0)),
        gdp = TSeries(qq(2020, 1), collect(0.0:99.0)),
        cpi = TSeries(mm(2020, 1), collect(0.0:299.0)),
    )
end

mixed_freq_pipeline_three_freq_run(state) =
    fconvert(Quarterly, state.unemp; method = :const) .+
    state.gdp .+
    fconvert(Quarterly, state.cpi; method = :mean)

# ---------------------------------------------------------------------------
# overlay / compare / reindex — M1.6.3b (`various.jl` pull-forward)
# ---------------------------------------------------------------------------

function overlay_three_tseries_setup()
    a = TSeries(qq(2020, 1), collect(0.0:99.0))
    a.values[1:7:end] .= NaN
    b = TSeries(qq(2019, 1), fill(100.0, 100))
    b.values[1:5:end] .= NaN
    c = TSeries(qq(2021, 1), fill(200.0, 100))
    return (a = a, b = b, c = c)
end

overlay_three_tseries_run(state) = overlay(state.a, state.b, state.c)

function compare_workspaces_equal_5_keys_setup()
    start = qq(2020, 1)
    arr = collect(0.0:99.0)
    w1 = Workspace()
    w2 = Workspace()
    for name in (:a, :b, :c, :d, :e)
        w1[name] = TSeries(start, copy(arr))
        w2[name] = TSeries(start, copy(arr))
    end
    return (w1 = w1, w2 = w2)
end

# `quiet=true` keeps the printed-diff cost out of the timed body (the
# Python side does the same — see `_run_compare_workspaces_equal_5_keys`).
compare_workspaces_equal_5_keys_run(state) =
    compare(state.w1, state.w2; quiet = true)

function compare_workspaces_differ_5_keys_setup()
    start = qq(2020, 1)
    arr = collect(0.0:99.0)
    w1 = Workspace()
    w2 = Workspace()
    for name in (:a, :b, :c, :d, :e)
        w1[name] = TSeries(start, copy(arr))
        w2[name] = TSeries(start, copy(arr))
    end
    w2[:c].values[51] = -999.0
    return (w1 = w1, w2 = w2)
end

compare_workspaces_differ_5_keys_run(state) =
    compare(state.w1, state.w2; quiet = true)

function reindex_tseries_100_setup()
    return (
        t = TSeries(qq(2020, 1), collect(0.0:99.0)),
        pair = qq(2020, 1) => 1U,
    )
end

reindex_tseries_100_run(state) = reindex(state.t, state.pair)

# rangeof(t; drop=1) — the tutorial-1 @rec idiom (mirrors the Python
# `rangeof_tseries_drop1` scenario added with M1.6.3c / G5 closure). On
# the Julia side `rangeof(t; drop=1)` is a closure call on a
# UnitRange{MIT{F}}; expected to be very fast.
function rangeof_tseries_drop1_setup()
    return (t = TSeries(qq(2020, 1), collect(0.0:99.0)),)
end

rangeof_tseries_drop1_run(state) = rangeof(state.t, drop = 1)

# 100x100 matrix * length-100 TSeries — VAR-style coefficient-matrix
# multiply (mirrors the Python `linalg_matrix_tseries_100` scenario added
# with M1.6.3g / G12 closure). Julia's `linalg.jl` overload returns a
# plain `Vector` (strips labels via `_vals`); the Python `@` overload
# matches that.
function linalg_matrix_tseries_100_setup()
    rng = MersenneTwister(20260518)
    matrix = randn(rng, 100, 100)
    t = TSeries(qq(2020, 1), randn(rng, 100))
    return (a = matrix, t = t)
end

linalg_matrix_tseries_100_run(state) = state.a * state.t

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# (setup, run) pairs keyed by scenario name. The Python side keeps the same
# names in `scenarios.py`.
const SCENARIOS = Dict{String, Tuple{Function, Function}}(
    # Construction
    "construct_tseries_qq_100"     => (construct_tseries_qq_100_setup,     construct_tseries_qq_100_run),
    "construct_mvts_qq_100x5"      => (construct_mvts_qq_100x5_setup,      construct_mvts_qq_100x5_run),
    # Indexing
    "indexing_mit_lookup_100"      => (indexing_mit_lookup_100_setup,      indexing_mit_lookup_100_run),
    "indexing_int_lookup_100"      => (indexing_int_lookup_100_setup,      indexing_int_lookup_100_run),
    "indexing_mitrange_slice"      => (indexing_mitrange_slice_setup,      indexing_mitrange_slice_run),
    "indexing_mvts_column"         => (indexing_mvts_column_setup,         indexing_mvts_column_run),
    # Indexing — M1.5 second Cython port (vectorised lookup)
    "indexing_lookup_100_api"      => (indexing_lookup_100_api_setup,      indexing_lookup_100_api_run),
    "indexing_lookup_100_numpy"    => (indexing_lookup_100_numpy_setup,    indexing_lookup_100_numpy_run),
    "indexing_lookup_100_cython"   => (indexing_lookup_100_cython_setup,   indexing_lookup_100_cython_run),
    # Arithmetic
    "arith_add_misaligned"         => (arith_add_misaligned_setup,         arith_add_misaligned_run),
    "arith_add_aligned"            => (arith_add_aligned_setup,            arith_add_aligned_run),
    "arith_mul_scalar"             => (arith_mul_scalar_setup,             arith_mul_scalar_run),
    # Shift family
    "shift_quarterly_lag1"         => (shift_quarterly_lag1_setup,         shift_quarterly_lag1_run),
    "lead_quarterly_lag1"          => (lead_quarterly_lag1_setup,          lead_quarterly_lag1_run),
    "diff_quarterly"               => (diff_quarterly_setup,               diff_quarterly_run),
    "pct_quarterly"                => (pct_quarterly_setup,                pct_quarterly_run),
    "ytypct_quarterly_100"         => (ytypct_quarterly_100_setup,         ytypct_quarterly_100_run),
    # Stats
    "mean_quarterly_100"           => (mean_quarterly_100_setup,           mean_quarterly_100_run),
    "std_quarterly_100"            => (std_quarterly_100_setup,            std_quarterly_100_run),
    "quantile_quarterly_100"       => (quantile_quarterly_100_setup,       quantile_quarterly_100_run),
    "cor_two_tseries"              => (cor_two_tseries_setup,              cor_two_tseries_run),
    "cov_two_tseries"              => (cov_two_tseries_setup,              cov_two_tseries_run),
    "cor_mvts_5_columns"           => (cor_mvts_5_columns_setup,           cor_mvts_5_columns_run),
    "cov_mvts_5_columns"           => (cov_mvts_5_columns_setup,           cov_mvts_5_columns_run),
    "mean_mvts_axis0_5cols"        => (mean_mvts_axis0_5cols_setup,        mean_mvts_axis0_5cols_run),
    "mean_mvts_axis1_100rows"      => (mean_mvts_axis1_100rows_setup,      mean_mvts_axis1_100rows_run),
    # Stats — M1.5 third Cython port (kernel-direct + public API)
    "mean_quarterly_100_numpy"     => (mean_quarterly_100_numpy_setup,     mean_quarterly_100_numpy_run),
    "mean_quarterly_100_cython"    => (mean_quarterly_100_cython_setup,    mean_quarterly_100_cython_run),
    "std_quarterly_100_numpy"      => (std_quarterly_100_numpy_setup,      std_quarterly_100_numpy_run),
    "std_quarterly_100_cython"     => (std_quarterly_100_cython_setup,     std_quarterly_100_cython_run),
    "cor_two_tseries_numpy"        => (cor_two_tseries_numpy_setup,        cor_two_tseries_numpy_run),
    "cor_two_tseries_cython"       => (cor_two_tseries_cython_setup,       cor_two_tseries_cython_run),
    # Moving / undiff
    "moving_average_quarterly_4"   => (moving_average_quarterly_4_setup,   moving_average_quarterly_4_run),
    "moving_sum_quarterly_4"       => (moving_sum_quarterly_4_setup,       moving_sum_quarterly_4_run),
    "undiff_quarterly"             => (undiff_quarterly_setup,             undiff_quarterly_run),
    # undiff — M1.6.2 fifth Cython port (kernel-direct + public API)
    "undiff_quarterly_numpy"       => (undiff_quarterly_numpy_setup,       undiff_quarterly_numpy_run),
    "undiff_quarterly_cython"      => (undiff_quarterly_cython_setup,      undiff_quarterly_cython_run),
    # fconvert
    "fconvert_qq_to_yy_mean"       => (fconvert_qq_to_yy_mean_setup,       fconvert_qq_to_yy_mean_run),
    "fconvert_qq_to_yy_sum"        => (fconvert_qq_to_yy_sum_setup,        fconvert_qq_to_yy_sum_run),
    "fconvert_yy_to_qq_const"      => (fconvert_yy_to_qq_const_setup,      fconvert_yy_to_qq_const_run),
    "fconvert_yy_to_qq_linear"     => (fconvert_yy_to_qq_linear_setup,     fconvert_yy_to_qq_linear_run),
    "fconvert_yy_to_qq_even"       => (fconvert_yy_to_qq_even_setup,       fconvert_yy_to_qq_even_run),
    "fconvert_mm_to_qq_mean"       => (fconvert_mm_to_qq_mean_setup,       fconvert_mm_to_qq_mean_run),
    # fconvert — M1.5 fourth Cython port (kernel-direct + public API)
    "fconvert_qq_to_yy_mean_numpy"  => (fconvert_qq_to_yy_mean_numpy_setup,  fconvert_qq_to_yy_mean_numpy_run),
    "fconvert_qq_to_yy_mean_cython" => (fconvert_qq_to_yy_mean_cython_setup, fconvert_qq_to_yy_mean_cython_run),
    "fconvert_qq_to_yy_sum_numpy"   => (fconvert_qq_to_yy_sum_numpy_setup,   fconvert_qq_to_yy_sum_numpy_run),
    "fconvert_qq_to_yy_sum_cython"  => (fconvert_qq_to_yy_sum_cython_setup,  fconvert_qq_to_yy_sum_cython_run),
    "fconvert_mm_to_qq_mean_numpy"  => (fconvert_mm_to_qq_mean_numpy_setup,  fconvert_mm_to_qq_mean_numpy_run),
    "fconvert_mm_to_qq_mean_cython" => (fconvert_mm_to_qq_mean_cython_setup, fconvert_mm_to_qq_mean_cython_run),
    # Recursion (general)
    "rec_ar2_100"                  => (rec_ar2_100_setup,                  rec_ar2_100_run),
    "rec_backcasting_via_lambda"   => (rec_backcasting_via_lambda_setup,   rec_backcasting_via_lambda_run),
    # Recursion (kernel-direct, four-flavor)
    "rec_linear_ar2_100_pylist"    => (rec_linear_ar2_100_pylist_setup,    rec_linear_ar2_100_pylist_run),
    "rec_linear_ar2_100_numpy"     => (rec_linear_ar2_100_numpy_setup,     rec_linear_ar2_100_numpy_run),
    "rec_linear_ar2_100_cython"    => (rec_linear_ar2_100_cython_setup,    rec_linear_ar2_100_cython_run),
    # Workspace
    "workspace_merge_5_series"     => (workspace_merge_5_series_setup,     workspace_merge_5_series_run),
    "workspace_filter_5_series"    => (workspace_filter_5_series_setup,    workspace_filter_5_series_run),
    # Mixed-frequency (pandas/polars friction demonstrators)
    "mixed_freq_qq_minus_mm_mean"     => (mixed_freq_qq_minus_mm_mean_setup,     mixed_freq_qq_minus_mm_mean_run),
    "mixed_freq_pipeline_three_freq"  => (mixed_freq_pipeline_three_freq_setup,  mixed_freq_pipeline_three_freq_run),
    # various.jl helpers (M1.6.3b)
    "overlay_three_tseries"           => (overlay_three_tseries_setup,           overlay_three_tseries_run),
    "compare_workspaces_equal_5_keys" => (compare_workspaces_equal_5_keys_setup, compare_workspaces_equal_5_keys_run),
    "compare_workspaces_differ_5_keys"=> (compare_workspaces_differ_5_keys_setup, compare_workspaces_differ_5_keys_run),
    "reindex_tseries_100"             => (reindex_tseries_100_setup,             reindex_tseries_100_run),
    # rangeof (M1.6.3c — mirrors Python G5 closure)
    "rangeof_tseries_drop1"           => (rangeof_tseries_drop1_setup,           rangeof_tseries_drop1_run),
    # linalg (M1.6.3g — mirrors Python G12 closure)
    "linalg_matrix_tseries_100"       => (linalg_matrix_tseries_100_setup,       linalg_matrix_tseries_100_run),
)

end  # module Scenarios
