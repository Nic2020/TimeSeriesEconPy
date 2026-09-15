# Scalar reconstruction routes of the pinned Julia loader: a bounded supplement to
# the julia_scalar_markers fixture. Every row stores an ordinary payload through
# the C entry point with an injected jtype and records what the pinned loader
# builds (type and canonical text) or raises. The Python scalar table accepts
# only routes this file shows loading; its outcomes are in julia_scalar_routes.toml.
# Usage:
#   julia --startup-file=no --project=<isolated-project> tests/dataecon/fixtures/scalar_routes_probe.jl <new.daec> <absolute-source-checkout> <new.toml>
using TimeSeriesEcon, Test, TOML, Dates
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
cstr(s) = (b = Vector{UInt8}(codeunits(String(s))); push!(b, 0x00); b)
text(x::Integer) = string(BigInt(x))
text(x::Rational) = string(BigInt(numerator(x)), "//", BigInt(denominator(x)))
text(x::Complex{<:Union{Integer,Rational}}) = string(text(real(x)), ",", text(imag(x)))
text(x::Union{Float16,Float32,Float64,ComplexF16,ComplexF32,ComplexF64}) = bytes2hex(le(x))
text(x::BigFloat) = string(x)
text(x::BigInt) = string(x)
text(x::Date) = string(Dates.year(x), "-", Dates.month(x), "-", Dates.day(x))
text(x::DateTime) = string(text(Date(x)), "T", Dates.hour(x), ":", Dates.minute(x), ":", Dates.second(x), ".", Dates.millisecond(x))
text(x::Symbol) = String(x)
text(x::AbstractString) = String(x)
text(x::Union{MIT,Duration}) = string(x)
text(x) = repr(x)

f64(x) = le(Float64(x)); i64(x) = le(Int64(x))
payloads = Any[
    ("i8_3", C.type_integer, UInt8[0x03]), ("i8_neg3", C.type_integer, UInt8[0xfd]),
    ("i16_300", C.type_integer, le(Int16(300))), ("i32_7", C.type_integer, le(Int32(7))),
    ("i64_3", C.type_integer, i64(3)), ("i64_neg3", C.type_integer, i64(-3)),
    ("i64_2p53p1", C.type_integer, i64(9007199254740993)),
    ("i128_2p70", C.type_integer, le(Int128(2)^70)), ("i128_neg5", C.type_integer, le(Int128(-5))),
    ("u8_3", C.type_unsigned, UInt8[0x03]), ("u64_max", C.type_unsigned, le(typemax(UInt64))),
    ("u128_2p64", C.type_unsigned, le(UInt128(2)^64)),
    ("f16_1p5", C.type_float, le(Float16(1.5))), ("f32_1p5", C.type_float, le(Float32(1.5))),
    ("f64_2", C.type_float, f64(2.0)), ("f64_1p5", C.type_float, f64(1.5)), ("f64_neg0", C.type_float, f64(-0.0)),
    ("f64_nan", C.type_float, f64(NaN)), ("f64_inf", C.type_float, f64(Inf)), ("f64_2p53p1", C.type_float, f64(9007199254740993.0)),
    ("c64_2_0", C.type_complex, le(ComplexF64(2.0, 0.0))), ("c64_1_2", C.type_complex, le(ComplexF64(1.0, 2.0))),
    ("c32_2_0", C.type_complex, le(ComplexF32(2.0, 0.0))), ("c16_2_0", C.type_complex, le(ComplexF16(2.0, 0.0))),
    ("str_abc", C.type_string, cstr("abc")),
]
tokens = [
    "Rational", "Rational{Int}", "Complex{Int}", "Complex", "Any", "Real", "Number", "Integer", "Signed", "Unsigned",
    "AbstractFloat", "Symbol", "String", "Date", "DateTime", "Dates.DateTime", "Base.Float64", "Core.Int64",
    "TimeSeriesEcon.Int64", "Float64 ", "BigInt", "BigFloat", "Int", "UInt", "Complex{Float64}", "Complex{Float32}",
    "Complex{Float16}", "MIT{Monthly}", "Duration{Monthly}", "Bool",
]
rows = Any[]
DE.opendaec(filename; write=true) do db
    for (pname, kind, payload) in payloads, token in tokens
        name = string(pname, "__", replace(token, r"[^A-Za-z0-9]" => "_"))
        id = Ref{C.obj_id_t}()
        rc = GC.@preserve payload C.de_store_scalar(db, DE.root_id, name, kind, C.frequency_t(0),
            length(payload), pointer(payload), id)
        @assert rc == 0
        DE.set_attribute(db, id[], "jtype", token)
        loaded = try
            DE.load_scalar(db, id[])
        catch e
            e
        end
        row = Dict{String,Any}("name" => name, "payload" => pname, "kind" => Int(kind), "nbytes" => length(payload),
            "hex" => bytes2hex(payload), "token" => token)
        if loaded isa Exception
            row["error"] = string(nameof(typeof(loaded)))
        else
            row["type"] = string(typeof(loaded))
            row["value"] = text(loaded)
        end
        push!(rows, row)
    end
    # Date/DateTime on a monthly date kind and Symbol on wide integers are in the fixture; add the
    # unix2datetime overflow edge: 1000 * x wraps for an Int64 payload beyond 2^63 / 1000.
    for (pname, payload, token) in (("i64_wrap", i64(9223372036854775), "DateTime"), ("i64_wrap1", i64(9223372036854776), "DateTime"),
            ("f64_big", f64(1e300), "DateTime"), ("f64_2p53ms", f64(9007199254740.993), "DateTime"))
        name = string(pname, "__", token)
        id = Ref{C.obj_id_t}()
        rc = GC.@preserve payload C.de_store_scalar(db, DE.root_id, name, C.type_integer, C.frequency_t(0),
            length(payload), pointer(payload), id)
        if pname == "f64_big" || pname == "f64_2p53ms"
            # re-store as float kind
            DE.delete_object(db, id[])
            rc = GC.@preserve payload C.de_store_scalar(db, DE.root_id, name, C.type_float, C.frequency_t(0),
                length(payload), pointer(payload), id)
        end
        @assert rc == 0
        DE.set_attribute(db, id[], "jtype", token)
        loaded = try
            DE.load_scalar(db, id[])
        catch e
            e
        end
        row = Dict{String,Any}("name" => name, "payload" => pname, "kind" => (startswith(pname, "f64") ? 4 : 1),
            "nbytes" => length(payload), "hex" => bytes2hex(payload), "token" => token)
        if loaded isa Exception
            row["error"] = string(nameof(typeof(loaded)))
        else
            row["type"] = string(typeof(loaded))
            row["value"] = text(loaded)
        end
        push!(rows, row)
    end
end
open(report, "w") do io
    TOML.print(io, Dict("pin" => PIN, "julia_version" => string(VERSION), "rows" => rows); sorted=true)
end
println("rows: ", length(rows), " loaded: ", count(r -> haskey(r, "type"), rows), " errors: ", count(r -> haskey(r, "error"), rows))
