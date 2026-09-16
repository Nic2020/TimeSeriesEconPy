# Remaining reconstruction routes of the pinned Julia loader: a bounded supplement
# to scalar_routes_probe.jl and the julia_scalar_markers fixture. Every row stores
# a raw payload through the C entry point with an injected jtype/jeltype and
# records what the pinned loader builds (type and canonical text) or raises. The
# Python tables accept only routes this file shows loading; the outcomes are in
# julia_marker_routes.toml. Sections: scalar (payload x token grid over date,
# duration, wide and narrow payloads), symbol (Julia's printed Symbol of numeric
# and date payloads), rational (Rational{T} of Float16/Float32 bit patterns),
# calendar (Date/DateTime of the remaining payload families), series (element
# tokens on dated series and plain vectors, with matrix/MVTSeries/tensor spot
# checks), empty (empty-only tokens per container), text (whole-object and
# element tokens on text vectors/matrices/tensors) and bool (wider Bool markers
# on plain arrays, with Julia's own rewrite), then object (whole-object Symbol
# on numeric containers, with the display-size dependence of dated ones), the
# text escaping and shape cases, char (the Char token on scalars and nonempty
# numeric containers), the explicit Complex{MIT{F}}/Complex{Duration{F}} tokens
# and the printed calendar MITs at extreme codes.
# Usage:
#   julia --startup-file=no --project=<isolated-project> tests/dataecon/fixtures/marker_routes_probe.jl <new.daec> <absolute-source-checkout> <new.toml>
using TimeSeriesEcon, Test, TOML, Dates, Random
Core.eval(Main, :(using Dates))
const DE = TimeSeriesEcon.DataEcon
const C = DE.C
const PIN = "fc0a0d01aed4903ea6c64b12d78d0bdf68468df6"
filename, checkout, report = ARGS
@assert realpath(pkgdir(TimeSeriesEcon)) == realpath(checkout)
@assert strip(read(`git -c safe.directory=$checkout -C $checkout rev-parse HEAD`, String)) == PIN
@assert unsafe_string(C.de_version()) == "0.4.0"
@assert !ispath(filename) && !ispath(report)

le(x) = Vector{UInt8}(reinterpret(UInt8, [x]))
le_bytes(v::AbstractVector) = Vector{UInt8}(reinterpret(UInt8, collect(v)))
cstr(s) = (b = Vector{UInt8}(codeunits(String(s))); push!(b, 0x00); b)
packed(strings) = (b = UInt8[]; for s in strings; append!(b, codeunits(s)); push!(b, 0x00); end; b)
text(x::Bool) = string(x)
text(x::Integer) = string(BigInt(x))
text(x::Rational) = string(BigInt(numerator(x)), "//", BigInt(denominator(x)))
text(x::Complex{<:Union{Integer,Rational}}) = string(text(real(x)), ",", text(imag(x)))
text(x::Union{Float16,Float32,Float64,ComplexF16,ComplexF32,ComplexF64}) = bytes2hex(le(x))
text(x::BigFloat) = string(x)
text(x::Date) = string(Dates.year(x), "-", Dates.month(x), "-", Dates.day(x))
text(x::DateTime) = string(text(Date(x)), "T", Dates.hour(x), ":", Dates.minute(x), ":", Dates.second(x), ".", Dates.millisecond(x))
text(x::Symbol) = text(String(x))
text(x::AbstractString) = isvalid(x) ? String(x) : string("bytes:", bytes2hex(codeunits(String(x))))
text(x::Char) = Base.ismalformed(x) ? string("bits:", string(reinterpret(UInt32, x), base=16)) : string("U+", string(UInt32(x), base=16))
text(x::Union{MIT,Duration}) = string(x)
text(x::Complex{<:Union{MIT,Duration}}) = string(text(real(x)), ",", text(imag(x)))
text(x::AbstractArray) = join((text(v) for v in x), ";")
text(x::TSeries) = text(x.values)
text(x::MVTSeries) = text(x.values)
text(x) = repr(x)

const rows = Any[]
const counter = Ref(0)
nextname(prefix) = string(prefix, "_", counter[] += 1)
outcome!(row, loaded) = (if loaded isa Exception
    row["error"] = string(nameof(typeof(loaded)))
else
    row["type"] = string(typeof(loaded))
    row["value"] = text(loaded)
end; push!(rows, row); loaded)

function store_scalar!(db, name, kind, freq, payload, token)
    id = Ref{C.obj_id_t}()
    rc = GC.@preserve payload C.de_store_scalar(db, DE.root_id, name, kind, C.frequency_t(freq), length(payload), pointer(payload), id)
    @assert rc == 0
    token === nothing || DE.set_attribute(db, id[], "jtype", token)
    return id[]
end
function scalar_row!(db, section, pname, kind, freq, payload, token)
    name = nextname(section)
    id = store_scalar!(db, name, kind, freq, payload, token)
    loaded = try DE.load_scalar(db, id) catch e e end
    outcome!(Dict{String,Any}("section" => section, "name" => name, "payload" => pname, "kind" => Int(kind),
        "frequency" => Int(freq), "nbytes" => length(payload), "hex" => bytes2hex(payload), "token" => token), loaded)
end

# Dated monthly series (2024M1 anchor) or plain vector through the C ABI.
function store_vec!(db, name, eltype, elfreq, payload, n; dated=true, marker=nothing, objmarker=nothing)
    ax = Ref{C.axis_id_t}()
    if dated
        @assert C.de_axis_range(db, n, C.frequency_t(32), Int64(24288), ax) == 0
    else
        @assert C.de_axis_plain(db, n, ax) == 0
    end
    id = Ref{C.obj_id_t}()
    GC.@preserve payload begin
        ptr = isempty(payload) ? C_NULL : pointer(payload)
        @assert C.de_store_tseries(db, DE.root_id, name, dated ? C.type_tseries : C.type_vector, eltype,
            C.frequency_t(elfreq), ax[], length(payload), ptr, id) == 0
    end
    marker === nothing || DE.set_attribute(db, id[], "jeltype", marker)
    objmarker === nothing || DE.set_attribute(db, id[], "jtype", objmarker)
    return id[]
end
function store_mat!(db, name, eltype, elfreq, payload, rows_, cols; mvt=false, marker=nothing, objmarker=nothing)
    a1 = Ref{C.axis_id_t}(); a2 = Ref{C.axis_id_t}()
    if mvt
        @assert C.de_axis_range(db, rows_, C.frequency_t(32), Int64(24288), a1) == 0
        buf = cstr(join(("c" * string(i) for i in 1:cols), "\n"))
        GC.@preserve buf @assert C.de_axis_names(db, cols, Ptr{Cchar}(pointer(buf)), a2) == 0
    else
        @assert C.de_axis_plain(db, rows_, a1) == 0
        @assert C.de_axis_plain(db, cols, a2) == 0
    end
    id = Ref{C.obj_id_t}()
    GC.@preserve payload begin
        ptr = isempty(payload) ? C_NULL : pointer(payload)
        @assert C.de_store_mvtseries(db, DE.root_id, name, mvt ? C.type_mvtseries : C.type_matrix, eltype,
            C.frequency_t(elfreq), a1[], a2[], length(payload), ptr, id) == 0
    end
    marker === nothing || DE.set_attribute(db, id[], "jeltype", marker)
    objmarker === nothing || DE.set_attribute(db, id[], "jtype", objmarker)
    return id[]
end
function store_tensor!(db, name, eltype, elfreq, payload, dims; marker=nothing, objmarker=nothing)
    axes = C.axis_id_t[]
    for n in dims
        ax = Ref{C.axis_id_t}(); @assert C.de_axis_plain(db, n, ax) == 0; push!(axes, ax[])
    end
    id = Ref{C.obj_id_t}()
    GC.@preserve payload axes begin
        ptr = isempty(payload) ? C_NULL : pointer(payload)
        @assert C.de_store_ndtseries(db, DE.root_id, name, C.type_tensor, eltype, C.frequency_t(elfreq), length(dims), pointer(axes), length(payload), ptr, id) == 0
    end
    marker === nothing || DE.set_attribute(db, id[], "jeltype", marker)
    objmarker === nothing || DE.set_attribute(db, id[], "jtype", objmarker)
    return id[]
end
load_any(db, id, container) = try
    container == "tensor" ? DE.load_ndtseries(db, id) : container in ("matrix", "mvtseries") ? DE.load_mvtseries(db, id) : DE.load_tseries(db, id)
catch e
    e
end

f64(x) = le(Float64(x)); i64(x) = le(Int64(x))
rng = Xoshiro(20260915)

DE.opendaec(filename; write=true) do db
    # ---- scalar grid: date, duration, wide, narrow and unsigned payloads x tokens
    scalar_payloads = Any[
        ("mit_m_2024M1", C.type_date, 32, i64(24288)), ("mit_m_neg", C.type_date, 32, i64(-24288)),
        ("mit_d_2024", C.type_date, 12, i64(739252)), ("mit_u_7", C.type_date, 11, i64(7)),
        ("mit_q3_8097", C.type_date, 67, i64(8097)), ("mit_y12_2024", C.type_date, 268, i64(2024)),
        ("mit_h6_4049", C.type_date, 134, i64(4049)), ("mit_w7_105604", C.type_date, 23, i64(105604)),
        ("mit_bd_528037", C.type_date, 13, i64(528037)),
        ("dur_m_5", C.type_integer, 32, i64(5)), ("dur_m_neg5", C.type_integer, 32, i64(-5)),
        ("dur_u_3", C.type_integer, 11, i64(3)), ("dur_d_10", C.type_integer, 12, i64(10)), ("dur_q3_7", C.type_integer, 67, i64(7)),
        ("i8_5", C.type_integer, 0, le(Int8(5))), ("i16_neg300", C.type_integer, 0, le(Int16(-300))),
        ("i32_7", C.type_integer, 0, le(Int32(7))), ("i64_3", C.type_integer, 0, i64(3)), ("i64_neg3", C.type_integer, 0, i64(-3)),
        ("i64_2p62", C.type_integer, 0, i64(Int64(2)^62)), ("i64_neg2p40", C.type_integer, 0, i64(-Int64(2)^40)),
        ("i128_2p70", C.type_integer, 0, le(Int128(2)^70)), ("i128_neg5", C.type_integer, 0, le(Int128(-5))),
        ("u8_200", C.type_unsigned, 0, le(UInt8(200))), ("u32_4e9", C.type_unsigned, 0, le(UInt32(4000000000))),
        ("u64_5", C.type_unsigned, 0, le(UInt64(5))), ("u64_max", C.type_unsigned, 0, le(typemax(UInt64))),
        ("u64_2p63", C.type_unsigned, 0, le(UInt64(2)^63)), ("u128_2p64", C.type_unsigned, 0, le(UInt128(2)^64)),
        ("u128_5", C.type_unsigned, 0, le(UInt128(5))),
        ("f16_1p5", C.type_float, 0, le(Float16(1.5))), ("f16_third", C.type_float, 0, le(Float16(1/3))), ("f16_neg2", C.type_float, 0, le(Float16(-2))),
        ("f32_1p5", C.type_float, 0, le(Float32(1.5))), ("f32_tenth", C.type_float, 0, le(Float32(0.1))), ("f32_1e10", C.type_float, 0, le(Float32(1e10))),
        ("f64_2", C.type_float, 0, f64(2.0)), ("f64_1p5", C.type_float, 0, f64(1.5)), ("f64_inf", C.type_float, 0, f64(Inf)),
        ("f64_neginf", C.type_float, 0, f64(-Inf)), ("f64_nan", C.type_float, 0, f64(NaN)), ("f64_third", C.type_float, 0, f64(1/3)),
        ("c64_2_0", C.type_complex, 0, le(ComplexF64(2.0, 0.0))), ("c64_1p5_neg2", C.type_complex, 0, le(ComplexF64(1.5, -2.0))),
        ("c64_half_third", C.type_complex, 0, le(ComplexF64(0.5, 1/3))), ("c64_1_2", C.type_complex, 0, le(ComplexF64(1.0, 2.0))),
        ("c32_2_0", C.type_complex, 0, le(ComplexF32(2.0, 0.0))), ("c32_tenth_0", C.type_complex, 0, le(ComplexF32(0.1, 0.0))),
        ("c32_1_2", C.type_complex, 0, le(ComplexF32(1.0, 2.0))), ("c16_2_0", C.type_complex, 0, le(ComplexF16(2.0, 0.0))),
        ("c16_third_0", C.type_complex, 0, le(ComplexF16(Float16(1/3), Float16(0)))), ("c16_1_2", C.type_complex, 0, le(ComplexF16(1.0, 2.0))),
        ("str_abc", C.type_string, 0, cstr("abc")),
    ]
    scalar_tokens = [
        "Any", "Number", "Real", "Integer", "Signed", "Unsigned", "AbstractFloat", "Complex", "Rational",
        "Rational{Int64}", "Rational{Int32}", "Rational{Int8}", "Rational{Int128}", "Rational{UInt8}", "Rational{UInt64}",
        "Complex{Int64}", "Complex{Int}", "Complex{Int32}", "Complex{Int8}", "Complex{Int128}", "Complex{UInt64}", "Complex{Bool}",
        "Complex{Rational{Int64}}", "Complex{Rational{Int8}}", "Complex{Rational{Int128}}", "Complex{Rational{UInt8}}",
        "BigInt", "BigFloat", "Date", "DateTime", "Symbol", "String",
        "Union{Int64,Float64}", "Union{Int64, Float64}", "MIT{Monthly}", "MIT{Daily}", "MIT{Unit}", "MIT{Quarterly{3}}",
        "Duration{Monthly}", "Duration{Unit}", "Int64", "Int8", "UInt64", "Float64", "Float32", "ComplexF64", "Int128", "Bool",
    ]
    for (pname, kind, freq, payload) in scalar_payloads, token in scalar_tokens
        scalar_row!(db, "scalar", pname, kind, freq, payload, token)
    end

    # ---- symbol: Julia's printed forms of numeric and date payloads
    edges64 = Float64[0.0, -0.0, 1.0, -1.0, 0.1, 0.5, 1.5, 100.0, 1e5, 123456.0, 1234567.0, 12345678.0, 1e15, 1e16, 1e17, 1e21, 1e22, 1e23,
        1e-4, 1e-5, 0.001, 0.0001, 0.00001, 123.456, 1/3, 2/3, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308,
        Inf, -Inf, NaN, 9007199254740993.0, 4503599627370496.5, 0.30000000000000004, 1e300, 123456789012345678.0, 99999999999999.9,
        999999999999999.9, 9999999999999998.0, 1.0e-7, 1.5e-7, 12.0, 120.0, 1200.0, 0.012]
    for x in edges64
        scalar_row!(db, "symbol", "f64", C.type_float, 0, le(x), "Symbol")
    end
    for _ in 1:150
        scalar_row!(db, "symbol", "f64", C.type_float, 0, le(reinterpret(Float64, rand(rng, UInt64))), "Symbol")
    end
    for _ in 1:60
        scalar_row!(db, "symbol", "f64", C.type_float, 0, le(rand(rng) * 10.0^rand(rng, -8:20)), "Symbol")
    end
    edges32 = Float32[0.0, -0.0, 1.0, 0.1, 1.5, 1e5, 1e6, 1e7, 1e8, 1e10, 1e-4, 1e-5, 123456.0, 1234567.0, 12345678.0, Inf, -Inf, NaN,
        1.0f-45, 3.4028235f38, 1.1754944f-38, 0.3f0, 16777217.0, 99999.99]
    for x in edges32
        scalar_row!(db, "symbol", "f32", C.type_float, 0, le(x), "Symbol")
    end
    for _ in 1:120
        scalar_row!(db, "symbol", "f32", C.type_float, 0, le(reinterpret(Float32, rand(rng, UInt32))), "Symbol")
    end
    for _ in 1:40
        scalar_row!(db, "symbol", "f32", C.type_float, 0, le(Float32(rand(rng) * 10.0^rand(rng, -6:12))), "Symbol")
    end
    edges16 = Float16[0.0, -0.0, 1.0, 0.1, 1.5, 1000.0, 10000.0, 65504.0, 1e-4, 1e-5, 6.0e-8, 6.104e-5, Inf, -Inf, NaN, 0.3, 2048.0, 4096.0, 100.0, 12345.0]
    for x in edges16
        scalar_row!(db, "symbol", "f16", C.type_float, 0, le(x), "Symbol")
    end
    for _ in 1:120
        scalar_row!(db, "symbol", "f16", C.type_float, 0, le(reinterpret(Float16, rand(rng, UInt16))), "Symbol")
    end
    for (re, im) in ((1.0, 2.0), (1.5, -2.0), (0.0, 0.0), (-0.0, -0.0), (0.0, -0.0), (Inf, NaN), (NaN, Inf), (-Inf, -Inf), (1e20, 1e-20), (1/3, 2/3), (5.0, 0.0), (0.0, 5.0))
        scalar_row!(db, "symbol", "c64", C.type_complex, 0, le(ComplexF64(re, im)), "Symbol")
        scalar_row!(db, "symbol", "c32", C.type_complex, 0, le(ComplexF32(re, im)), "Symbol")
        scalar_row!(db, "symbol", "c16", C.type_complex, 0, le(ComplexF16(Float16(re), Float16(im))), "Symbol")
    end
    for _ in 1:30
        scalar_row!(db, "symbol", "c64", C.type_complex, 0, le(ComplexF64(reinterpret(Float64, rand(rng, UInt64)), reinterpret(Float64, rand(rng, UInt64)))), "Symbol")
        scalar_row!(db, "symbol", "c32", C.type_complex, 0, le(ComplexF32(reinterpret(Float32, rand(rng, UInt32)), reinterpret(Float32, rand(rng, UInt32)))), "Symbol")
        scalar_row!(db, "symbol", "c16", C.type_complex, 0, le(ComplexF16(reinterpret(Float16, rand(rng, UInt16)), reinterpret(Float16, rand(rng, UInt16)))), "Symbol")
    end
    for freq in (11, 12, 13, 17, 18, 19, 20, 21, 22, 23, 32, 65, 66, 67, 129, 130, 131, 132, 133, 134, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268)
        for code in (0, 1, 7, -7, 24288, 8097, 4049, 2024, 739252, 105604, 528037, -1)
            scalar_row!(db, "symbol", string("mit_", freq), C.type_date, freq, i64(code), "Symbol")
            scalar_row!(db, "symbol", string("dur_", freq), C.type_integer, freq, i64(code), "Symbol")
        end
    end
    for x in (Int8(-5), Int16(300), UInt8(200), UInt64(2)^63, Int128(2)^100, -Int128(2)^100, UInt128(2)^127, typemin(Int64), true, false)
        kind = x isa Unsigned ? C.type_unsigned : C.type_integer
        scalar_row!(db, "symbol", string(typeof(x)), kind, 0, le(x), "Symbol")
    end

    # ---- rational: Rational{T} of Float16/Float32 bit patterns and narrow complexes
    params = ("Int8", "Int16", "Int32", "Int64", "Int128", "UInt8", "UInt32", "UInt64", "UInt128")
    f32vals = Float32[0.1, 1/3, 2/3, 0.5, -0.5, 1.5, 65536.0, 16777216.0, 16777217.0, 3.4028235f38, 1.0f-45, 1.1754944f-38, 0.0, -0.0, Inf, -Inf, NaN,
        123.456, 1e-3, 3.0e-1, 5.0, 2.5, 127.0, 128.0, -128.0, -129.0, 255.0, 256.0, 1.0e6, 7.7f-2]
    for _ in 1:100
        push!(f32vals, reinterpret(Float32, rand(rng, UInt32)))
    end
    for _ in 1:40
        push!(f32vals, Float32(rand(rng) * 10.0^rand(rng, -6:8)))
    end
    for x in f32vals, p in params
        scalar_row!(db, "rational", "f32", C.type_float, 0, le(x), string("Rational{", p, "}"))
    end
    f16vals = Float16[0.1, 1/3, 2/3, 0.5, 1.5, 65504.0, 2048.0, 2049.0, 6.0e-8, 6.104e-5, 0.0, -0.0, Inf, -Inf, NaN, 123.4, 1e-3, 0.3, 127.0, 128.0, -128.0, -129.0, 255.0, 256.0, 1000.0]
    for _ in 1:100
        push!(f16vals, reinterpret(Float16, rand(rng, UInt16)))
    end
    for x in f16vals, p in params
        scalar_row!(db, "rational", "f16", C.type_float, 0, le(x), string("Rational{", p, "}"))
    end
    for x in (0.1f0, 1/3, 1.5f0, 2.5f0, 128.0f0), p in ("Int8", "Int64", "UInt8")
        scalar_row!(db, "rational", "c32", C.type_complex, 0, le(ComplexF32(x, 0)), string("Rational{", p, "}"))
        scalar_row!(db, "rational", "c16", C.type_complex, 0, le(ComplexF16(Float16(x), Float16(0))), string("Rational{", p, "}"))
        scalar_row!(db, "rational", "c32", C.type_complex, 0, le(ComplexF32(x, 0)), "Rational")
        scalar_row!(db, "rational", "c16", C.type_complex, 0, le(ComplexF16(Float16(x), Float16(0))), "Rational")
    end
    for x in (0.1f0, 1.5f0), p in ("Int8", "Int64")
        scalar_row!(db, "rational", "f32", C.type_float, 0, le(x), string("Complex{Rational{", p, "}}"))
        scalar_row!(db, "rational", "f16", C.type_float, 0, le(Float16(x)), string("Complex{Rational{", p, "}}"))
        scalar_row!(db, "rational", "c32", C.type_complex, 0, le(ComplexF32(x, x)), string("Complex{Rational{", p, "}}"))
        scalar_row!(db, "rational", "c16", C.type_complex, 0, le(ComplexF16(Float16(x), Float16(x))), string("Complex{Rational{", p, "}}"))
    end
    for x in (0.1f0, 1/3, 1.5f0), p in ("Int8", "Int64")
        scalar_row!(db, "rational", "f32", C.type_float, 0, le(x), "Rational")
        scalar_row!(db, "rational", "f16", C.type_float, 0, le(Float16(x)), "Rational")
    end

    # ---- calendar: Date/DateTime of the remaining payload families
    for tok in ("Date", "DateTime")
        for x in Float32[1.5, 1234567.9, 1e10, -1e-3, 16777216.0, 16777217.0, 2.0f6, -1.0f9, 9.2233715f15, 9.2233725f15, 1.0f20, Inf, NaN, 0.0, -0.0, 86399.999, 1.0e-3, -1.5]
            scalar_row!(db, "calendar", "f32", C.type_float, 0, le(x), tok)
        end
        for _ in 1:40
            scalar_row!(db, "calendar", "f32", C.type_float, 0, le(Float32(rand(rng) * 10.0^rand(rng, -3:16))), tok)
        end
        for x in Float16[1.5, 1000.5, 65504.0, -1.0e-3, 0.0, Inf, NaN, 2048.0, 2049.0, 100.25, -1.5]
            scalar_row!(db, "calendar", "f16", C.type_float, 0, le(x), tok)
        end
        for _ in 1:25
            scalar_row!(db, "calendar", "f16", C.type_float, 0, le(reinterpret(Float16, rand(rng, UInt16))), tok)
        end
        for x in UInt64[0, 1, 1710460800, 9223372036854775, 9223372036854776, UInt64(2)^63, typemax(UInt64), 18446744073709551, 18446744073709552]
            scalar_row!(db, "calendar", "u64", C.type_unsigned, 0, le(x), tok)
        end
        for x in Int128[0, 1, -1, 1710460800, 9223372036854775, 9223372036854776, -9223372036854775, -9223372036854776, Int128(2)^63, Int128(2)^70, -Int128(2)^70]
            scalar_row!(db, "calendar", "i128", C.type_integer, 0, le(x), tok)
        end
        for x in UInt128[0, 1, 1710460800, 9223372036854775, 9223372036854776, UInt128(2)^63, UInt128(2)^70]
            scalar_row!(db, "calendar", "u128", C.type_unsigned, 0, le(x), tok)
        end
        for (re, im) in ((1.5, 0.0), (1234567.9, 0.0), (1.5, 1.0), (1e10, -0.0))
            scalar_row!(db, "calendar", "c32", C.type_complex, 0, le(ComplexF32(re, im)), tok)
            scalar_row!(db, "calendar", "c16", C.type_complex, 0, le(ComplexF16(Float16(re), Float16(im))), tok)
        end
        for x in Int64[0, -1, 1710460800, -62135596800, 9223372036854775, 9223372036854776, -9223372036854775, -9223372036854776, typemax(Int64)]
            scalar_row!(db, "calendar", "i64", C.type_integer, 0, le(x), tok)
        end
        for x in (Int8(-5), Int16(300), Int32(-7), UInt8(200), UInt16(60000), UInt32(4000000000))
            scalar_row!(db, "calendar", string(typeof(x)), x isa Unsigned ? C.type_unsigned : C.type_integer, 0, le(x), tok)
        end
    end

    # ---- series: element tokens on dated series and plain vectors, spot checks on other containers
    bases = Any[
        ("Int8", C.type_integer, 0, Int8[-1, 0, 2]), ("Int64", C.type_integer, 0, Int64[-1, 0, 2]),
        ("Int64_big", C.type_integer, 0, Int64[Int64(2)^62, -3, 9007199254740993]),
        ("UInt8", C.type_unsigned, 0, UInt8[0, 1, 200]), ("UInt64", C.type_unsigned, 0, UInt64[0, 1, typemax(UInt64)]),
        ("Float64_int", C.type_float, 0, Float64[-1.0, 0.0, 2.0]), ("Float64", C.type_float, 0, Float64[0.5, -0.0, 2.5]),
        ("Float64_inf", C.type_float, 0, Float64[Inf, 1.0]), ("Float64_nan", C.type_float, 0, Float64[NaN, 1.0]),
        ("Float32", C.type_float, 0, Float32[0.5, 2.5, 0.1]), ("Float16", C.type_float, 0, Float16[0.5, 2.5, 0.1]),
        ("ComplexF64", C.type_complex, 0, ComplexF64[1 + 0im, 2 + 0im]), ("ComplexF64_imag", C.type_complex, 0, ComplexF64[1 + 2im, 0.5 - 0.25im]),
        ("ComplexF64_frac", C.type_complex, 0, ComplexF64[0.5 + 0im, 1.5 + 0im]), ("ComplexF32", C.type_complex, 0, ComplexF32[1 + 0im, 2 + 0im]),
        ("ComplexF16", C.type_complex, 0, ComplexF16[complex(Float16(1), Float16(0)), complex(Float16(2), Float16(0))]),
        ("Int128", C.type_integer, 0, Int128[-1, 0, Int128(2)^70]), ("UInt128", C.type_unsigned, 0, UInt128[0, 1, UInt128(2)^70]),
        ("MIT{Monthly}", C.type_date, 32, Int64[24288, 24289, 0]), ("MIT{Daily}", C.type_date, 12, Int64[739252, 0, -1]),
        ("Duration{Monthly}", C.type_integer, 32, Int64[-1, 0, 2]), ("Duration{Unit}", C.type_integer, 11, Int64[-1, 0, 2]),
    ]
    series_tokens = ["Rational", "Rational{Int64}", "Rational{Int8}", "Rational{Int128}", "Rational{UInt8}", "Rational{Int32}",
        "Complex", "Complex{Int64}", "Complex{Int}", "Complex{Int8}", "Complex{Int128}", "Complex{Bool}", "Complex{UInt8}",
        "Complex{Rational{Int64}}", "Complex{Rational{Int8}}",
        "Any", "Number", "Real", "Integer", "Signed", "Unsigned", "AbstractFloat", "Union{Int64,Float64}", "Union{Int64, Float64}",
        "BigInt", "BigFloat", "Date", "DateTime", "Symbol", "String", "NoSuchType", "MIT{Monthly}", "Duration{Monthly}"]
    for (label, code, elfreq, vals) in bases, token in series_tokens, dated in (true, false)
        payload = le_bytes(vals)
        name = nextname(dated ? "ser" : "vec")
        id = store_vec!(db, name, code, elfreq, payload, length(vals); dated=dated, marker=token)
        loaded = load_any(db, id, "series")
        outcome!(Dict{String,Any}("section" => "series", "name" => name, "container" => dated ? "tseries" : "vector",
            "base" => label, "kind" => Int(code), "frequency" => elfreq, "hex" => bytes2hex(payload), "token" => token), loaded)
    end
    for (label, code, elfreq, vals) in (("Int64", C.type_integer, 0, Int64[-1, 0, 2, 5]), ("Float64", C.type_float, 0, Float64[0.5, -1.0, 2.5, 4.0]),
                                        ("Int8", C.type_integer, 0, Int8[-1, 0, 2, 5]), ("MIT{Monthly}", C.type_date, 32, Int64[24288, 24289, 24290, 24291]))
        payload = le_bytes(vals)
        for token in ("Rational{Int64}", "Rational", "Complex{Int64}", "Complex", "Complex{Rational{Int64}}", "Real", "AbstractFloat", "Signed", "Unsigned", "Any", "NoSuchType", "Symbol")
            for container in ("matrix", "mvtseries", "tensor")
                name = nextname(container)
                id = container == "tensor" ? store_tensor!(db, name, code, elfreq, payload, (1, 2, 2); marker=token) :
                     store_mat!(db, name, code, elfreq, payload, 2, 2; mvt=(container == "mvtseries"), marker=token)
                loaded = load_any(db, id, container)
                outcome!(Dict{String,Any}("section" => "series", "name" => name, "container" => container, "base" => label,
                    "kind" => Int(code), "frequency" => elfreq, "hex" => bytes2hex(payload), "token" => token), loaded)
            end
        end
    end

    # ---- empty: empty-only tokens per container and kind
    empty_tokens = ["Date", "DateTime", "Dates.Date", "Dates.DateTime", "Symbol", "String", "Char", "MIT", "Duration",
        "MIT{Quarterly}", "MIT{Frequency}", "MIT{Monthly}", "Duration{Monthly}", "Any", "Real", "Number", "Integer", "Signed", "Unsigned",
        "AbstractFloat", "Complex", "Union{Int64,Float64}", "Union{Int64, Float64}", "Union{}", "Nothing", "Missing", "Vector{Int64}",
        "BigInt", "BigFloat", "Bool", "Int128", "ComplexF16", "Rational", "Rational{Int64}", "Complex{Int64}", "Complex{Rational{Int64}}",
        "NoSuchType", "Int8", "Float32"]
    for (label, code) in (("Int64", C.type_integer), ("UInt64", C.type_unsigned), ("Float64", C.type_float), ("ComplexF64", C.type_complex))
        for token in empty_tokens
            for container in ("tseries", "vector", "matrix", "mvtseries", "tensor")
                name = nextname("empty")
                id = container in ("tseries", "vector") ? store_vec!(db, name, code, 0, UInt8[], 0; dated=(container == "tseries"), marker=token) :
                     container == "tensor" ? store_tensor!(db, name, code, 0, UInt8[], (0, 2, 2); marker=token) :
                     store_mat!(db, name, code, 0, UInt8[], 0, 2; mvt=(container == "mvtseries"), marker=token)
                loaded = load_any(db, id, container)
                outcome!(Dict{String,Any}("section" => "empty", "name" => name, "container" => container, "base" => label,
                    "kind" => Int(code), "frequency" => 0, "hex" => "", "token" => token), loaded)
            end
        end
    end
    # An empty date/duration series with a foreign token (Julia's loader fails before any marker).
    for token in ("Any", "Date", "Int64")
        name = nextname("empty")
        id = store_vec!(db, name, C.type_date, 32, UInt8[], 0; dated=true, marker=token)
        outcome!(Dict{String,Any}("section" => "empty", "name" => name, "container" => "tseries", "base" => "MIT{Monthly}",
            "kind" => Int(C.type_date), "frequency" => 32, "hex" => "", "token" => token), load_any(db, id, "series"))
    end

    # ---- text: whole-object and element tokens on text vectors, matrices and tensors
    text_object_tokens = ["Vector{String}", "Vector", "AbstractVector", "AbstractArray", "Array{String,1}", "Array{String, 1}", "Array", "Any",
        "Vector{Any}", "Vector{AbstractString}", "Vector{SubString{String}}", "Vector{Symbol}", "Symbol", "String", "Matrix{String}",
        "Vector{Char}", "NoSuchType", "Array{String}", "Vector{ Symbol }", "Vector{Union{String,Symbol}}"]
    for (label, strings) in (("ab", ["a", "b"]), ("empty", String[]), ("unicode", ["é", "🙂"]))
        payload = packed(strings)
        for objtoken in text_object_tokens, marker in (nothing, "Symbol")
            name = nextname("text")
            id = store_vec!(db, name, C.type_string, 0, payload, length(strings); dated=false, marker=marker, objmarker=objtoken)
            outcome!(Dict{String,Any}("section" => "text", "name" => name, "container" => "vector", "base" => label, "kind" => 6,
                "frequency" => 0, "hex" => bytes2hex(payload), "token" => objtoken, "eltoken" => something(marker, "")), load_any(db, id, "vector"))
        end
        for marker in ("Symbol", "String", "AbstractString", "SubString{String}", "Char", "Any", "NoSuchType", "Int64", "Vector{String}")
            name = nextname("text")
            id = store_vec!(db, name, C.type_string, 0, payload, length(strings); dated=false, marker=marker)
            outcome!(Dict{String,Any}("section" => "text", "name" => name, "container" => "vector", "base" => label, "kind" => 6,
                "frequency" => 0, "hex" => bytes2hex(payload), "token" => "", "eltoken" => marker), load_any(db, id, "vector"))
        end
    end
    for (label, strings, dims) in (("m22", ["a", "b", "c", "d"], (2, 2)), ("m02", String[], (0, 2)))
        payload = packed(strings)
        for objtoken in ("Matrix{String}", "Array{String,2}", "Array{String, 2}", "Matrix", "Array", "AbstractArray", "AbstractMatrix", "Any",
                         "Matrix{Any}", "Matrix{AbstractString}", "Matrix{Symbol}", "Vector{String}", "Symbol", "NoSuchType"), marker in (nothing, "Symbol")
            name = nextname("textm")
            id = store_mat!(db, name, C.type_string, 0, payload, dims...; marker=marker, objmarker=objtoken)
            outcome!(Dict{String,Any}("section" => "text", "name" => name, "container" => "matrix", "base" => label, "kind" => 6,
                "frequency" => 0, "hex" => bytes2hex(payload), "token" => objtoken, "eltoken" => something(marker, "")), load_any(db, id, "matrix"))
        end
    end
    for (label, strings, dims) in (("t221", ["a", "b", "c", "d"], (2, 2, 1)), ("t022", String[], (0, 2, 2)))
        payload = packed(strings)
        for objtoken in ("Array{String,3}", "Array{String, 3}", "Array", "AbstractArray", "Any", "Array{Any,3}", "Array{Symbol,3}", "Array{String,2}", "NoSuchType"), marker in (nothing, "Symbol")
            name = nextname("textt")
            id = store_tensor!(db, name, C.type_string, 0, payload, dims; marker=marker, objmarker=objtoken)
            outcome!(Dict{String,Any}("section" => "text", "name" => name, "container" => "tensor", "base" => label, "kind" => 6,
                "frequency" => 0, "hex" => bytes2hex(payload), "token" => objtoken, "eltoken" => something(marker, "")), load_any(db, id, "tensor"))
        end
    end

    # ---- bool: wider Bool element markers on plain arrays and Julia's own rewrite
    for (label, code, vals) in (("Int64", C.type_integer, Int64[0, 1, 1, 0]), ("Int16", C.type_integer, Int16[1, 0, 1, 1]), ("UInt8", C.type_unsigned, UInt8[0, 1, 0, 1]),
                                ("Float64", C.type_float, Float64[0.0, 1.0, -0.0, 1.0]), ("ComplexF64", C.type_complex, ComplexF64[0, 1, 1, 0]),
                                ("Int64_two", C.type_integer, Int64[0, 2, 1, 0]), ("Float64_half", C.type_float, Float64[0.5, 1.0, 0.0, 1.0]),
                                ("Int128", C.type_integer, Int128[0, 1, 1, 0]), ("ComplexF64_imag", C.type_complex, ComplexF64[0, 1 + 1im, 1, 0]))
        payload = le_bytes(vals)
        for container in ("vector", "matrix", "tensor")
            name = nextname("bool")
            id = container == "vector" ? store_vec!(db, name, code, 0, payload, 4; dated=false, marker="Bool") :
                 container == "matrix" ? store_mat!(db, name, code, 0, payload, 2, 2; marker="Bool") :
                 store_tensor!(db, name, code, 0, payload, (1, 2, 2); marker="Bool")
            loaded = load_any(db, id, container)
            row = Dict{String,Any}("section" => "bool", "name" => name, "container" => container, "base" => label, "kind" => Int(code),
                "frequency" => 0, "hex" => bytes2hex(payload), "token" => "Bool")
            if !(loaded isa Exception)
                rid = container == "vector" ? DE.store_tseries(db, name * "_rw", loaded) :
                      container == "matrix" ? DE.store_mvtseries(db, name * "_rw", loaded) : DE.store_ndtseries(db, name * "_rw", loaded)
                info = DE.get_all_attributes(db, rid)
                row["rewrite_attributes"] = Dict(string(k) => string(v) for (k, v) in info)
                if container == "vector"
                    ref = Ref{C.tseries_t}(); @assert C.de_load_tseries(db, rid, ref) == 0
                    a = ref[]; row["rewrite_metadata"] = Int[a.eltype, a.nbytes]
                elseif container == "matrix"
                    ref = Ref{C.mvtseries_t}(); @assert C.de_load_mvtseries(db, rid, ref) == 0
                    a = ref[]; row["rewrite_metadata"] = Int[a.eltype, a.nbytes]
                else
                    ref = Ref{C.ndtseries_t}(); @assert C.de_load_ndtseries(db, rid, ref) == 0
                    a = ref[]; row["rewrite_metadata"] = Int[a.eltype, a.nbytes]
                end
            end
            outcome!(row, loaded)
        end
    end

    # ---- object: whole-object Symbol (Julia's printed array) and String on numeric containers
    object_bases = Any[
        ("Int64", C.type_integer, 0, Int64[1, -2, 3, 4]), ("Int8", C.type_integer, 0, Int8[1, -2, 3, 4]),
        ("Int16", C.type_integer, 0, Int16[1, -2, 3, 4]), ("Int32", C.type_integer, 0, Int32[1, -2, 3, 4]),
        ("UInt8", C.type_unsigned, 0, UInt8[0, 1, 200, 255]), ("UInt16", C.type_unsigned, 0, UInt16[0, 1, 200, 65535]),
        ("UInt32", C.type_unsigned, 0, UInt32[0, 1, 200, 4000000000]), ("UInt64", C.type_unsigned, 0, UInt64[0, 1, typemax(UInt64), 7]),
        ("Float64", C.type_float, 0, Float64[0.5, -0.0, 2.5, 1e10]), ("Float64_special", C.type_float, 0, Float64[NaN, Inf, -Inf, 1e-7]),
        ("Float64_int", C.type_float, 0, Float64[1.0, 2.0, 123456.0, 1234567.0]),
        ("Float32", C.type_float, 0, Float32[0.5, -0.0, 2.5, 1e10]), ("Float16", C.type_float, 0, Float16[0.5, -0.0, 2.5, 1e4]),
        ("ComplexF64", C.type_complex, 0, ComplexF64[1 + 2im, 0.5 - 0.25im, complex(NaN, Inf), complex(-0.0, -0.0)]),
        ("ComplexF32", C.type_complex, 0, ComplexF32[1 + 2im, 0.5 - 0.25im, complex(NaN32, Inf32), complex(-0.0f0, -0.0f0)]),
        ("ComplexF16", C.type_complex, 0, ComplexF16[complex(Float16(1), Float16(2)), complex(Float16(0.5), Float16(-0.25)), complex(NaN16, Inf16), complex(Float16(-0.0), Float16(-0.0))]),
        ("Int128", C.type_integer, 0, Int128[-1, 0, Int128(2)^70, 5]), ("UInt128", C.type_unsigned, 0, UInt128[0, 1, UInt128(2)^70, 5]),
        ("MIT{Monthly}", C.type_date, 32, Int64[24288, 24289, 0, -1]), ("MIT{Daily}", C.type_date, 12, Int64[739252, 0, -1, 3652060]),
        ("MIT{Quarterly{3}}", C.type_date, 67, Int64[8097, 0, -1, 5]), ("MIT{Weekly{7}}", C.type_date, 23, Int64[105604, 0, -1, 5]),
        ("MIT{Unit}", C.type_date, 11, Int64[7, 0, -1, 5]),
        ("Duration{Monthly}", C.type_integer, 32, Int64[-1, 0, 2, 5]), ("Duration{Unit}", C.type_integer, 11, Int64[-1, 0, 2, 5]),
    ]
    object_shapes = (("vector", (4,)), ("matrix", (2, 2)), ("matrix", (1, 4)), ("matrix", (4, 1)), ("tensor", (2, 2, 1)),
        ("tensor", (1, 2, 2)), ("tensor", (2, 1, 2)), ("tensor", (1, 1, 1, 4)), ("tensor", (1, 1, 2, 2, 1)))
    function object_row!(db, label, code, elfreq, payload, container, dims, token; marker=nothing)
        name = nextname("obj")
        id = container == "vector" ? store_vec!(db, name, code, elfreq, payload, dims[1]; dated=false, marker=marker, objmarker=token) :
             container == "matrix" ? store_mat!(db, name, code, elfreq, payload, dims...; marker=marker, objmarker=token) :
             store_tensor!(db, name, code, elfreq, payload, dims; marker=marker, objmarker=token)
        outcome!(Dict{String,Any}("section" => "object", "name" => name, "container" => container, "base" => label, "kind" => Int(code),
            "frequency" => elfreq, "hex" => bytes2hex(payload), "token" => token, "eltoken" => something(marker, ""), "dims" => Int[dims...]),
            load_any(db, id, container))
    end
    for (label, code, elfreq, vals) in object_bases
        payload = le_bytes(vals)
        for (container, dims) in object_shapes, token in ("Symbol", "String")
            object_row!(db, label, code, elfreq, payload, container, dims, token)
        end
        # A jeltype alongside the whole-object token: the loader applies jtype only.
        object_row!(db, label, code, elfreq, payload, "vector", (4,), "Symbol"; marker="Bool")
    end
    for (label, code) in (("Int64", C.type_integer), ("UInt64", C.type_unsigned), ("Float64", C.type_float), ("ComplexF64", C.type_complex))
        for (container, dims) in (("vector", (0,)), ("matrix", (0, 2)), ("matrix", (2, 0)), ("matrix", (0, 0)), ("tensor", (0, 2, 2)), ("tensor", (2, 0, 2)))
            object_row!(db, label, code, 0, UInt8[], container, dims, "Symbol")
        end
    end
    # A UInt8 vector names the Symbol with its bytes (Symbol(::Vector{UInt8})): NUL is refused.
    for (label, bytes_) in (("UInt8_ab", UInt8[0x61, 0x62]), ("UInt8_invalid", UInt8[0x61, 0xff]), ("UInt8_nul", UInt8[0x61, 0x00, 0x62]))
        object_row!(db, label, C.type_unsigned, 0, bytes_, "vector", (length(bytes_),), "Symbol")
    end
    # Dated containers: the printed TSeries/MVTSeries depends on the loading session's display size.
    for n in (3, 30)
        vals = Float64[i * 0.5 for i in 1:n]
        payload = le_bytes(vals)
        name = nextname("obj")
        id = store_vec!(db, name, C.type_float, 0, payload, n; dated=true, objmarker="Symbol")
        for (env, lines, columns) in (("default", nothing, nothing), ("LINES=8,COLUMNS=30", "8", "30"))
            loaded = withenv("LINES" => lines, "COLUMNS" => columns) do
                load_any(db, id, "series")
            end
            outcome!(Dict{String,Any}("section" => "object", "name" => name, "container" => "tseries", "base" => "Float64", "kind" => Int(C.type_float),
                "frequency" => 0, "hex" => bytes2hex(payload), "token" => "Symbol", "eltoken" => "", "dims" => Int[n], "env" => env), loaded)
        end
        payload2 = le_bytes(Float64[vals; vals])
        name = nextname("obj")
        id = store_mat!(db, name, C.type_float, 0, payload2, n, 2; mvt=true, objmarker="Symbol")
        for (env, lines, columns) in (("default", nothing, nothing), ("LINES=8,COLUMNS=30", "8", "30"))
            loaded = withenv("LINES" => lines, "COLUMNS" => columns) do
                load_any(db, id, "mvtseries")
            end
            outcome!(Dict{String,Any}("section" => "object", "name" => name, "container" => "mvtseries", "base" => "Float64", "kind" => Int(C.type_float),
                "frequency" => 0, "hex" => bytes2hex(payload2), "token" => "Symbol", "eltoken" => "", "dims" => Int[n, 2], "env" => env), loaded)
        end
    end
    # An empty dated series prints without any display-size dependence.
    name = nextname("obj")
    id = store_vec!(db, name, C.type_float, 0, UInt8[], 0; dated=true, objmarker="Symbol")
    outcome!(Dict{String,Any}("section" => "object", "name" => name, "container" => "tseries", "base" => "Float64", "kind" => Int(C.type_float),
        "frequency" => 0, "hex" => "", "token" => "Symbol", "eltoken" => "", "dims" => Int[0], "env" => "default"), load_any(db, id, "series"))

    # ---- text (continued): escaping cases of Julia's printed String arrays and more shapes
    escapes = ["a\"b", "c\\d", "\$x", "tab\t", "nl\n", "cr\r", "\a\b\v\f", "\e", "\x7f", "\x01", "\x1f", "é", "🙂", " ", "​", " ",
        "", "" * "1", "" * "g", "\U000e0001", "\U000e0001" * "a", "\U000f0000",
        String(UInt8[0xed, 0xa0, 0x80]), String(UInt8[0xed, 0xa0, 0x80, 0x31]), String(UInt8[0xff]), String(UInt8[0xc0, 0x80]),
        String(UInt8[0xe2, 0x82]), String(UInt8[0xe2, 0x82, 0x41]), String(UInt8[0x80, 0x41]), String(UInt8[0xf8, 0x88, 0x80, 0x80, 0x80]),
        String(UInt8[0xf4, 0x90, 0x80, 0x80]), String(UInt8[0xf4, 0x90, 0x80, 0x80, 0x62]), String(UInt8[0xe0, 0x80, 0x80]), String(UInt8[0xf0, 0x80, 0x80, 0x80]),
        String(UInt8[0xc2]), String(UInt8[0xc2, 0x41]), " ", "", "日本", "a b", "𝔘"]
    # Characters this Julia's Unicode tables assign but older tables may not (or that
    # no table assigns): kept apart so the stable escapes are pinned exactly.
    unassigned = ["\U0001fae9", "\u2ffc", "\u0378"]
    for (label, strings) in (("escapes", escapes), ("unassigned", unassigned))
        payload = packed(strings)
        for objtoken in ("Symbol", "Vector{String}", "Any"), marker in (nothing, "Symbol")
            name = nextname("text")
            id = store_vec!(db, name, C.type_string, 0, payload, length(strings); dated=false, marker=marker, objmarker=objtoken)
            outcome!(Dict{String,Any}("section" => "text", "name" => name, "container" => "vector", "base" => label, "kind" => 6,
                "frequency" => 0, "hex" => bytes2hex(payload), "token" => objtoken, "eltoken" => something(marker, "")), load_any(db, id, "vector"))
        end
        for marker in ("Symbol", "String", "Char", "Any")
            name = nextname("text")
            id = store_vec!(db, name, C.type_string, 0, payload, length(strings); dated=false, marker=marker)
            outcome!(Dict{String,Any}("section" => "text", "name" => name, "container" => "vector", "base" => label, "kind" => 6,
                "frequency" => 0, "hex" => bytes2hex(payload), "token" => "", "eltoken" => marker), load_any(db, id, "vector"))
        end
    end
    for (label, strings, dims) in (("m12", ["a", "b"], (1, 2)), ("m21", ["a", "b"], (2, 1)), ("m20", String[], (2, 0)), ("m00", String[], (0, 0)),
                                   ("m22e", ["a\"b", "c\\d", "\$x", "é"], (2, 2)))
        payload = packed(strings)
        for objtoken in ("Symbol", "Matrix{String}", "Any"), marker in (nothing, "Symbol")
            name = nextname("textm")
            id = store_mat!(db, name, C.type_string, 0, payload, dims...; marker=marker, objmarker=objtoken)
            outcome!(Dict{String,Any}("section" => "text", "name" => name, "container" => "matrix", "base" => label, "kind" => 6,
                "frequency" => 0, "hex" => bytes2hex(payload), "token" => objtoken, "eltoken" => something(marker, "")), load_any(db, id, "matrix"))
        end
    end
    for (label, strings, dims) in (("t222", ["a", "b", "c", "d", "e", "f", "g", "h"], (2, 2, 2)), ("t212", ["a", "b", "c", "d"], (2, 1, 2)),
                                   ("t1114", ["a", "b", "c", "d"], (1, 1, 1, 4)), ("t11221", ["a", "b", "c", "d"], (1, 1, 2, 2, 1)),
                                   ("t202", String[], (2, 0, 2)), ("t0000", String[], (0, 0, 0, 0)))
        payload = packed(strings)
        for objtoken in ("Symbol", "Any"), marker in (nothing, "Symbol")
            name = nextname("textt")
            id = store_tensor!(db, name, C.type_string, 0, payload, dims; marker=marker, objmarker=objtoken)
            outcome!(Dict{String,Any}("section" => "text", "name" => name, "container" => "tensor", "base" => label, "kind" => 6,
                "frequency" => 0, "hex" => bytes2hex(payload), "token" => objtoken, "eltoken" => something(marker, "")), load_any(db, id, "tensor"))
        end
    end

    # ---- char: the Char token on scalars and on nonempty numeric containers
    for (pname, kind, freq, payload) in scalar_payloads
        scalar_row!(db, "char", pname, kind, freq, payload, "Char")
    end
    for x in (Int64(97), Int64(0x1f642), Int64(0xd800), Int64(0x10ffff), Int64(0x110000), Int64(0x200000), Int64(-1), Int64(2)^32,
              Int8(97), Int8(-1), UInt8(255), UInt32(0x1f642), UInt64(97), Int128(97), Int128(2)^70, UInt128(97), Int16(0x7f), UInt16(0xfffe))
        scalar_row!(db, "char", string(typeof(x), "_", x isa Unsigned ? string(UInt64(x % UInt64)) : string(x)), x isa Unsigned ? C.type_unsigned : C.type_integer, 0, le(x), "Char")
    end
    for x in (97.0, 97.5, -1.0, 1.0f0, Float16(97))
        scalar_row!(db, "char", string(typeof(x), "_", x), C.type_float, 0, le(x), "Char")
    end
    scalar_row!(db, "char", "c64_97_0", C.type_complex, 0, le(ComplexF64(97, 0)), "Char")
    scalar_row!(db, "char", "c64_97_1", C.type_complex, 0, le(ComplexF64(97, 1)), "Char")
    char_bases = Any[
        ("Int64", C.type_integer, 0, Int64[97, 98, 0x1f642, 0]), ("Int64_surrogate", C.type_integer, 0, Int64[97, 0xd800]),
        ("Int64_invalid", C.type_integer, 0, Int64[97, 0x110000]), ("Int64_huge", C.type_integer, 0, Int64[97, 0x200000]),
        ("Int64_neg", C.type_integer, 0, Int64[97, -1]), ("Int8", C.type_integer, 0, Int8[97, 98, 0]), ("Int8_neg", C.type_integer, 0, Int8[97, -1]),
        ("UInt8", C.type_unsigned, 0, UInt8[97, 255]), ("UInt64", C.type_unsigned, 0, UInt64[97, 98]), ("Int128", C.type_integer, 0, Int128[97, 98]),
        ("UInt128", C.type_unsigned, 0, UInt128[97, 98]), ("Float64", C.type_float, 0, Float64[97.0, 98.0]), ("Float64_frac", C.type_float, 0, Float64[97.5, 98.0]),
        ("Float32", C.type_float, 0, Float32[97.0, 98.0]), ("ComplexF64", C.type_complex, 0, ComplexF64[97, 98]),
        ("MIT{Monthly}", C.type_date, 32, Int64[97, 98]), ("Duration{Monthly}", C.type_integer, 32, Int64[97, 98]),
    ]
    for (label, code, elfreq, vals) in char_bases, dated in (true, false)
        payload = le_bytes(vals)
        name = nextname(dated ? "ser" : "vec")
        id = store_vec!(db, name, code, elfreq, payload, length(vals); dated=dated, marker="Char")
        outcome!(Dict{String,Any}("section" => "series", "name" => name, "container" => dated ? "tseries" : "vector",
            "base" => label, "kind" => Int(code), "frequency" => elfreq, "hex" => bytes2hex(payload), "token" => "Char"), load_any(db, id, "series"))
    end
    for (label, code, elfreq, vals) in (("Int64", C.type_integer, 0, Int64[97, 98, 99, 100]), ("Int64_invalid", C.type_integer, 0, Int64[97, 98, 99, 0x110000]))
        payload = le_bytes(vals)
        for container in ("matrix", "mvtseries", "tensor")
            name = nextname(container)
            id = container == "tensor" ? store_tensor!(db, name, code, elfreq, payload, (1, 2, 2); marker="Char") :
                 store_mat!(db, name, code, elfreq, payload, 2, 2; mvt=(container == "mvtseries"), marker="Char")
            outcome!(Dict{String,Any}("section" => "series", "name" => name, "container" => container, "base" => label,
                "kind" => Int(code), "frequency" => elfreq, "hex" => bytes2hex(payload), "token" => "Char"), load_any(db, id, container))
        end
    end

    # ---- dated complex: explicit Complex{MIT{F}} / Complex{Duration{F}} tokens
    dated_complex_tokens = ("Complex{MIT{Monthly}}", "Complex{Duration{Monthly}}", "Complex{MIT{Daily}}", "Complex{MIT{Unit}}")
    for (pname, kind, freq, payload) in scalar_payloads, token in dated_complex_tokens
        scalar_row!(db, "scalar", pname, kind, freq, payload, token)
    end
    for (label, code, elfreq, vals) in bases, token in dated_complex_tokens, dated in (true, false)
        payload = le_bytes(vals)
        name = nextname(dated ? "ser" : "vec")
        id = store_vec!(db, name, code, elfreq, payload, length(vals); dated=dated, marker=token)
        outcome!(Dict{String,Any}("section" => "series", "name" => name, "container" => dated ? "tseries" : "vector",
            "base" => label, "kind" => Int(code), "frequency" => elfreq, "hex" => bytes2hex(payload), "token" => token), load_any(db, id, "series"))
    end
    for token in ("Complex{MIT{Monthly}}", "Complex{Duration{Monthly}}")
        for container in ("tseries", "vector", "matrix", "mvtseries", "tensor"), (label, code) in (("Int64", C.type_integer), ("Float64", C.type_float))
            name = nextname("empty")
            id = container in ("tseries", "vector") ? store_vec!(db, name, code, 0, UInt8[], 0; dated=(container == "tseries"), marker=token) :
                 container == "tensor" ? store_tensor!(db, name, code, 0, UInt8[], (0, 2, 2); marker=token) :
                 store_mat!(db, name, code, 0, UInt8[], 0, 2; mvt=(container == "mvtseries"), marker=token)
            outcome!(Dict{String,Any}("section" => "empty", "name" => name, "container" => container, "base" => label,
                "kind" => Int(code), "frequency" => 0, "hex" => "", "token" => token), load_any(db, id, container))
        end
    end

    # ---- symbol (continued): printed calendar MITs at extreme codes (Julia's Int64 date arithmetic)
    extreme_codes = Int64[0, -1, -366, -1000000, 3652059, 3652060, Int64(2)^40, Int64(2)^51 + 3, Int64(2)^52 + 1, Int64(2)^53 + 1, Int64(2)^60,
        typemax(Int64), typemin(Int64), typemin(Int64) + 1, -Int64(2)^53 - 1, 9223372036854775, 92233720368547758, 922337203685477580, -922337203685477580]
    for freq in (12, 13, 17, 18, 19, 20, 21, 22, 23), code in extreme_codes
        scalar_row!(db, "symbol", string("mitx_", freq), C.type_date, freq, i64(code), "Symbol")
    end
    for freq in (11, 32, 67, 268), code in (typemax(Int64), typemin(Int64), Int64(2)^60, -Int64(2)^60)
        scalar_row!(db, "symbol", string("mitx_", freq), C.type_date, freq, i64(code), "Symbol")
        scalar_row!(db, "symbol", string("durx_", freq), C.type_integer, freq, i64(code), "Symbol")
    end
    # Symbol element markers on calendar-date vectors at the same extremes.
    for freq in (12, 13, 23)
        vals = Int64[0, -1, 3652060, typemax(Int64), typemin(Int64)]
        payload = le_bytes(vals)
        name = nextname("vec")
        id = store_vec!(db, name, C.type_date, freq, payload, length(vals); dated=false, marker="Symbol")
        outcome!(Dict{String,Any}("section" => "series", "name" => name, "container" => "vector", "base" => string("MITx{", freq, "}"),
            "kind" => Int(C.type_date), "frequency" => freq, "hex" => bytes2hex(payload), "token" => "Symbol"), load_any(db, id, "series"))
    end
end
open(report, "w") do io
    TOML.print(io, Dict("pin" => PIN, "julia_version" => string(VERSION), "rows" => rows); sorted=true)
end
println("rows: ", length(rows), " loaded: ", count(r -> haskey(r, "type"), rows), " errors: ", count(r -> haskey(r, "error"), rows))
