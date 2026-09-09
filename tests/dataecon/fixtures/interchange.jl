# SPDX-License-Identifier: MIT
# Run in an isolated Julia project with the pinned local TimeSeriesEcon checkout.
# julia --project=<environment> interchange.jl generate|verify <file> <checkout>
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
expected = TSeries(2024M1, [1.25, -2.5, 0.0, 4.75])

if action == "generate"
    ispath(filename) && error("Output already exists; use a fresh fixture path.")
    DE.opendaec(filename; write=true) do db
        DE.store_tseries(db, DE.root_id, "sample", expected)
    end
elseif action != "verify"
    error("Expected generate or verify.")
end

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

if action == "generate"
    layout = Dict{String,Any}(
        "enums" => sizeof.([C.class_t, C.type_t, C.frequency_t, C.axis_type_t]),
    )
    for T in (C.object_t, C.axis_t, C.tseries_t)
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
