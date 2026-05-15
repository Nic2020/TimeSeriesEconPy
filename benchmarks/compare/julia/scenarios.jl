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

# Use deterministic seeded RNG so the per-call work is constant across runs
# (matches Python's `np.random.default_rng(seed=20260515)`).
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
    # Arithmetic
    "arith_add_misaligned"         => (arith_add_misaligned_setup,         arith_add_misaligned_run),
    "arith_add_aligned"            => (arith_add_aligned_setup,            arith_add_aligned_run),
    "arith_mul_scalar"             => (arith_mul_scalar_setup,             arith_mul_scalar_run),
    # Shift family
    "shift_quarterly_lag1"         => (shift_quarterly_lag1_setup,         shift_quarterly_lag1_run),
    "diff_quarterly"               => (diff_quarterly_setup,               diff_quarterly_run),
    "pct_quarterly"                => (pct_quarterly_setup,                pct_quarterly_run),
    # Stats
    "mean_quarterly_100"           => (mean_quarterly_100_setup,           mean_quarterly_100_run),
    "std_quarterly_100"            => (std_quarterly_100_setup,            std_quarterly_100_run),
    "cor_two_tseries"              => (cor_two_tseries_setup,              cor_two_tseries_run),
    "cor_mvts_5_columns"           => (cor_mvts_5_columns_setup,           cor_mvts_5_columns_run),
    "cov_mvts_5_columns"           => (cov_mvts_5_columns_setup,           cov_mvts_5_columns_run),
    # Moving / undiff
    "moving_average_quarterly_4"   => (moving_average_quarterly_4_setup,   moving_average_quarterly_4_run),
    "moving_sum_quarterly_4"       => (moving_sum_quarterly_4_setup,       moving_sum_quarterly_4_run),
    "undiff_quarterly"             => (undiff_quarterly_setup,             undiff_quarterly_run),
    # fconvert
    "fconvert_qq_to_yy_mean"       => (fconvert_qq_to_yy_mean_setup,       fconvert_qq_to_yy_mean_run),
    "fconvert_qq_to_yy_sum"        => (fconvert_qq_to_yy_sum_setup,        fconvert_qq_to_yy_sum_run),
    "fconvert_yy_to_qq_const"      => (fconvert_yy_to_qq_const_setup,      fconvert_yy_to_qq_const_run),
    "fconvert_mm_to_qq_mean"       => (fconvert_mm_to_qq_mean_setup,       fconvert_mm_to_qq_mean_run),
    # Recursion (general)
    "rec_ar2_100"                  => (rec_ar2_100_setup,                  rec_ar2_100_run),
    # Recursion (kernel-direct, four-flavor)
    "rec_linear_ar2_100_pylist"    => (rec_linear_ar2_100_pylist_setup,    rec_linear_ar2_100_pylist_run),
    "rec_linear_ar2_100_numpy"     => (rec_linear_ar2_100_numpy_setup,     rec_linear_ar2_100_numpy_run),
    "rec_linear_ar2_100_cython"    => (rec_linear_ar2_100_cython_setup,    rec_linear_ar2_100_cython_run),
    # Workspace
    "workspace_merge_5_series"     => (workspace_merge_5_series_setup,     workspace_merge_5_series_run),
    "workspace_filter_5_series"    => (workspace_filter_5_series_setup,    workspace_filter_5_series_run),
)

end  # module Scenarios
