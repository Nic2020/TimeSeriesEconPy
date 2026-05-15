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
import TimeSeriesEcon: qq

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
    "construct_tseries_qq_100"   => (construct_tseries_qq_100_setup,   construct_tseries_qq_100_run),
    "indexing_mit_lookup_100"    => (indexing_mit_lookup_100_setup,    indexing_mit_lookup_100_run),
    "arith_add_misaligned"       => (arith_add_misaligned_setup,       arith_add_misaligned_run),
    "shift_quarterly_lag1"       => (shift_quarterly_lag1_setup,       shift_quarterly_lag1_run),
    "moving_average_quarterly_4" => (moving_average_quarterly_4_setup, moving_average_quarterly_4_run),
    "fconvert_qq_to_yy_mean"     => (fconvert_qq_to_yy_mean_setup,     fconvert_qq_to_yy_mean_run),
    "rec_ar2_100"                    => (rec_ar2_100_setup,                    rec_ar2_100_run),
    "rec_linear_ar2_100_numpy"       => (rec_linear_ar2_100_numpy_setup,       rec_linear_ar2_100_numpy_run),
    "rec_linear_ar2_100_cython"      => (rec_linear_ar2_100_cython_setup,      rec_linear_ar2_100_cython_run),
    "workspace_merge_5_series"       => (workspace_merge_5_series_setup,       workspace_merge_5_series_run),
)

end  # module Scenarios
