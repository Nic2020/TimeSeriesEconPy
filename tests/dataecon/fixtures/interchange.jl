# SPDX-License-Identifier: MIT
# Run in an isolated Julia project with the pinned local TimeSeriesEcon checkout.
# julia --project=<environment> interchange.jl <action> <file> <checkout>
# Actions: generate/verify, generate-empty/verify-empty,
# generate-scalars/verify-scalars, verify-wheel.
using TimeSeriesEcon
using Test, SHA, TOML, Pkg

const DE = TimeSeriesEcon.DataEcon
const C = DE.C
const SOURCE_SHA = "fc0a0d01aed4903ea6c64b12d78d0bdf68468df6"
const NATIVE_SHA = "1a108688a044380f808bebf64079e32dbb9cd1a4"

action, filename, checkout = ARGS
@assert realpath(pkgdir(TimeSeriesEcon)) == realpath(checkout)
actual_sha = strip(read(`git -c safe.directory=$checkout -C $checkout rev-parse HEAD`, String))
@assert actual_sha == SOURCE_SHA
@assert unsafe_string(C.de_version()) == "0.4.0"
@assert string(Pkg.dependencies()[Base.PkgId(C.DataEcon_jll).uuid].version) == "0.4.0+0"
expected = TSeries(2024M1, [1.25, -2.5, 0.0, 4.75])
scalar_cases = ["scalar_finite" => 1.25, "scalar_negative" => -2.5,
    "scalar_zero" => 0.0, "scalar_negative_zero" => -0.0,
    "scalar_nan" => NaN, "scalar_positive_inf" => Inf, "scalar_negative_inf" => -Inf]


if action == "generate"
    ispath(filename) && error("Output already exists; use a fresh fixture path.")
    DE.opendaec(filename; write=true) do db
        DE.store_tseries(db, DE.root_id, "sample", expected)
    end
elseif action == "generate-empty"
    ispath(filename) && error("Output already exists; use a fresh fixture path.")
    DE.opendaec(filename; write=true) do db
        DE.store_tseries(db, DE.root_id, "empty", TSeries(2024M1, Float64[]))
        DE.store_tseries(db, DE.root_id, "empty_later", TSeries(2025M7, Float64[]))
        DE.store_tseries(db, DE.root_id, "empty_float32", TSeries(2024M1, Float32[]))
    end
elseif action == "generate-scalars"
    ispath(filename) && error("Output already exists; use a fresh fixture path.")
    DE.opendaec(filename; write=true) do db
        for (name, value) in scalar_cases
            DE.store_scalar(db, DE.root_id, name, value)
        end
        DE.store_scalar(db, DE.root_id, "scalar_float32", Float32(1.25))
        DE.store_scalar(db, DE.root_id, "scalar_integer", Int64(7))
    end
elseif !(action in ("verify", "verify-empty", "verify-scalars", "verify-wheel"))
    error("Unknown action; use generate/verify, generate-empty/verify-empty, generate-scalars/verify-scalars or verify-wheel.")
end

if action in ("generate", "verify", "verify-wheel")
    @testset "DataEcon monthly interchange" begin
        y = DE.opendaec(filename) do db
            id = DE.find_object(db, DE.root_id, "sample")
            @test isempty(DE.get_all_attributes(db, id))
            value = DE.load_tseries(db, id)
            ts = Ref{C.tseries_t}()
            @test C.de_load_tseries(db, id, ts) == 0
            @test Int(ts[].object.obj_class) == 2
            @test Int(ts[].object.obj_type) == 12
            @test Int(ts[].eltype) == 4
            @test Int(ts[].elfreq) == 0
            @test Int(ts[].axis.ax_type) == 1
            @test ts[].axis.length == 4
            @test Int(ts[].axis.frequency) == 32
            @test ts[].axis.first == 24288
            @test ts[].nbytes == 32
            value
        end
        @test firstdate(y) == 2024M1
        @test lastdate(y) == 2024M4
        @test frequencyof(y) == Monthly
        @test eltype(y) == Float64
        @test y.values == expected.values
    end
end

if action in ("generate-empty", "verify-empty", "verify-wheel")
    reference_fixture = action != "verify-wheel"
    @testset "DataEcon empty monthly interchange" begin
        cases = [("empty", 2024M1, 24288, Float64), ("empty_later", 2025M7, 24306, Float64)]
        reference_fixture && push!(cases, ("empty_float32", 2024M1, 24288, Float32))
        DE.opendaec(filename) do db
            for (name, anchor, code, ET) in cases
                id = DE.find_object(db, DE.root_id, name)
                arr = Ref{C.tseries_t}()
                @test C.de_load_tseries(db, id, arr) == 0
                ts = arr[]
                @test Int.((ts.object.obj_class, ts.object.obj_type, ts.eltype, ts.elfreq,
                    ts.axis.ax_type, ts.axis.length, ts.axis.frequency, ts.axis.first,
                    ts.nbytes)) == (2, 12, 4, 0, 1, 0, 32, code, 0)
                @test ts.value == C_NULL
                attrs = Dict(string(k) => string(v) for (k,v) in DE.get_all_attributes(db, id))
                value = DE.load_tseries(db, id)
                @test isempty(value)
                @test eltype(value) == ET
                if reference_fixture
                    # Pinned Julia's jeltype reconstruction discards an empty
                    # TSeries wrapper. The axis remains intact in the file.
                    @test attrs == Dict("jeltype" => string(ET))
                    @test value isa Vector{ET}
                else
                    # Python omits the redundant Float64 marker to preserve the
                    # dated wrapper through the unmodified Julia public loader.
                    @test isempty(attrs)
                    @test value isa TSeries
                    @test firstdate(value) == anchor
                    @test lastdate(value) == anchor - 1
                    @test frequencyof(value) == Monthly
                end
            end
        end
    end
end

if action in ("generate-scalars", "verify-scalars", "verify-wheel")
    @testset "DataEcon Float64 scalar interchange" begin
        DE.opendaec(filename) do db
            for (name, expected_scalar) in scalar_cases
                id = DE.find_object(db, DE.root_id, name)
                scal = Ref{C.scalar_t}()
                @test C.de_load_scalar(db, id, scal) == 0
                v = scal[]
                @test Int.((v.object.obj_class, v.object.obj_type, v.frequency, v.nbytes)) == (1,4,0,8)
                @test v.value != C_NULL
                @test isempty(DE.get_all_attributes(db, id))
                value = DE.load_scalar(db, id)
                @test value isa Float64
                @test isequal(value, expected_scalar)
            end
            if action != "verify-wheel"
                @test DE.load_scalar(db, DE.find_object(db, DE.root_id, "scalar_float32")) isa Float32
                @test DE.load_scalar(db, DE.find_object(db, DE.root_id, "scalar_integer")) isa Int64
            end
        end
    end
end

if action in ("generate", "generate-empty", "generate-scalars")
    layout = Dict{String,Any}(
        "enums" => sizeof.([C.class_t, C.type_t, C.frequency_t, C.axis_type_t]),
    )
    for T in (C.object_t, C.axis_t, C.tseries_t, C.scalar_t)
        layout[string(nameof(T))] = Dict(
            "size" => sizeof(T),
            "offsets" => [Int(fieldoffset(T, i)) for i in 1:fieldcount(T)],
        )
    end
    provenance = Dict(
        "julia_version" => string(VERSION),
        "timeseriesecon_version" => string(pkgversion(TimeSeriesEcon)),
        "timeseriesecon_sha" => SOURCE_SHA,
        "native_source_sha" => NATIVE_SHA,
        "dataecon_jll_version" => string(Pkg.dependencies()[Base.PkgId(C.DataEcon_jll).uuid].version),
        "native_version" => unsafe_string(C.de_version()),
        "native_sha256" => bytes2hex(sha256(read(C.DataEcon_jll.libdaec_path))),
        "header_sha256" => bytes2hex(sha256(read(C.DataEcon_jll.daec_header))),
        "fixture_sha256" => bytes2hex(sha256(read(filename))),
        "platform" => Sys.MACHINE,
        "layout" => layout,
    )
    open(replace(filename, r"\.daec$" => ".toml"), "w") do io
        TOML.print(io, provenance; sorted=true)
    end
end
