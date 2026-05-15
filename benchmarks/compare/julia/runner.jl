# SPDX-License-Identifier: MIT
#
# Julia CLI entry for the comparison harness.
#
# Usage (typically invoked by `../run.py`)::
#
#     julia --project=. runner.jl --scenario NAME --output PATH [--samples N] [--seconds S]
#
# Writes a JSON document::
#
#     {
#       "scenario": "<NAME>",
#       "language": "julia",
#       "median_seconds": <Float64>,
#       "min_seconds":    <Float64>,
#       "samples":        <Int>,
#       "julia_version":  "<VERSION>",
#       "tseriesecon_version": "<VERSION>"
#     }
#
# `BenchmarkTools.@belapsed` returns the *minimum* observed time over many
# samples — the recommended noise-resistant statistic. `@benchmark` is used
# when we also want the median (it carries the full Trial object).

using BenchmarkTools
using JSON
using Pkg

include("scenarios.jl")

function parse_args(argv::Vector{String})
    parsed = Dict{String, Any}(
        "scenario" => nothing,
        "output"   => nothing,
        "seconds"  => 5.0,
        "samples"  => 10_000,
    )
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg == "--scenario"
            parsed["scenario"] = argv[i + 1]; i += 2
        elseif arg == "--output"
            parsed["output"] = argv[i + 1]; i += 2
        elseif arg == "--seconds"
            parsed["seconds"] = parse(Float64, argv[i + 1]); i += 2
        elseif arg == "--samples"
            parsed["samples"] = parse(Int, argv[i + 1]); i += 2
        elseif arg == "--list"
            for k in sort(collect(keys(Scenarios.SCENARIOS)))
                println(k)
            end
            exit(0)
        else
            error("Unknown CLI flag: $arg")
        end
    end
    parsed["scenario"] === nothing && error("--scenario is required")
    parsed["output"]   === nothing && error("--output is required")
    return parsed
end

function tseriesecon_version()::String
    for (_, dep) in Pkg.dependencies()
        if dep.name == "TimeSeriesEcon"
            return string(dep.version)
        end
    end
    return "unknown"
end

function run_scenario(name::String, seconds::Float64, samples::Int)
    haskey(Scenarios.SCENARIOS, name) ||
        error("unknown scenario: $name (use --list to see available)")
    setup_fn, run_fn = Scenarios.SCENARIOS[name]
    state = setup_fn()
    # Warmup so JIT / type-inference cost isn't charged to the first sample.
    run_fn(state)
    # BenchmarkTools auto-tunes evaluations-per-sample; we cap wall time and
    # sample count. `$run_fn($state)` interpolates so the variable lookup
    # cost isn't measured.
    trial = @benchmark $run_fn($state) seconds = seconds samples = samples
    return (
        median_seconds = BenchmarkTools.median(trial).time * 1e-9,
        min_seconds    = BenchmarkTools.minimum(trial).time * 1e-9,
        samples        = length(trial),
    )
end

function main(argv::Vector{String})
    parsed = parse_args(argv)
    res = run_scenario(parsed["scenario"], parsed["seconds"], parsed["samples"])
    payload = Dict(
        "scenario" => parsed["scenario"],
        "language" => "julia",
        "median_seconds" => res.median_seconds,
        "min_seconds"    => res.min_seconds,
        "samples"        => res.samples,
        "julia_version"  => string(VERSION),
        "tseriesecon_version" => tseriesecon_version(),
    )
    open(parsed["output"], "w") do io
        JSON.print(io, payload)
    end
    println("ok $(parsed["scenario"]): median=$(round(res.median_seconds * 1e6; digits=2)) µs")
    return 0
end

# Allow `include("runner.jl")` without auto-running:
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
