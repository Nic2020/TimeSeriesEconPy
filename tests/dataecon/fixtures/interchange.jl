# SPDX-License-Identifier: MIT
# Run in an isolated Julia project with the pinned local TimeSeriesEcon checkout.
# julia --project=<environment> interchange.jl <action> <file> <checkout>
# Actions: generate/verify, generate-empty/verify-empty,
# generate-scalars/verify-scalars, generate-quarterly/verify-quarterly, verify-wheel.
# Annual actions: generate-annual/verify-annual (also checked by verify-wheel).
# Half-yearly actions: generate-halfyearly/verify-halfyearly (also checked by verify-wheel).
# Int64 scalar actions: generate-int64/verify-int64 (also checked by verify-wheel).
# String scalar actions: generate-strings/verify-strings (also checked by verify-wheel).
# Date/duration scalar actions: generate-dates/verify-dates (also checked by verify-wheel).
# Unit/calendar scalar actions: generate-calendar/verify-calendar (also checked by verify-wheel).
# Numeric width scalar actions: generate-widths/verify-widths (also checked by verify-wheel).
# File operation actions: generate-fileops/verify-fileops (verify-wheel also checks the
# Python-written overwrite/delete outcomes and the auxiliary "fileops/<name>-fileops.daec"
# output, kept out of the workflow's single-file "*.daec" discovery directory).
# Calendar series actions: generate-calendar-series/verify-calendar-series (also checked
# by verify-wheel): Daily, BDaily and Weekly{1..7} Float64 TSeries.
# Numeric/Boolean series actions: generate-series-elements/verify-series-elements (also
# checked by verify-wheel).
# Represented series actions: generate-represented-elements/verify-represented-elements
# (also checked by verify-wheel): MIT/Duration elements over all 32 element frequencies,
# Int128/UInt128, ComplexF16, marked empties, wide Bool markers and reference controls.
# Foreign marker actions: generate-foreign-markers/verify-foreign-markers (also checked
# by verify-wheel): jeltype/jtype reconstruction markers on numeric, wide and
# date/duration sources, with Julia's own loaded values materialized as siblings.
# Plain-array/Unit-series actions: generate-arrays-unit/verify-arrays-unit (also
# checked by verify-wheel): ordinary one-dimensional vectors, lossless ranges,
# and Unit-frequency numeric/Boolean TSeries.
# Matrix/MVTSeries/text actions: generate-matrices-text/verify-matrices-text (also
# checked by verify-wheel): column-major plain matrices over every ordinary and
# represented element, LinearAlgebra structure markers, MVTSeries over every axis
# family with their names axis, and packed text vectors including multibyte text
# that Julia's own writer cannot size correctly.
# N-dimensional actions: generate-tensors/verify-tensors (also checked by
# verify-wheel): column-major plain arrays with three to five plain axes over
# every ordinary and represented element, singleton and zero-length
# dimensions, Julia's BitArray tokens and preserved element/object markers.
# Catalog actions: generate-catalogs/verify-catalogs (also checked by
# verify-wheel under the /catalogs prefix): nested catalogs, every object
# family at nested paths, Unicode names, name ordering, a child stored under a
# scalar, and attributes on objects, catalogs and the root.
# Workspace actions: generate-workspace/verify-workspace (also checked by
# verify-wheel under the /workspace prefix): one writedb of a mixed Workspace
# (every writer dispatch family, nested and empty Workspaces, byte-ordered
# keys, Julia-only marked scalars) read back with readdb.
# Text array actions: generate-text-arrays/verify-text-arrays (also checked by
# verify-wheel): String, Symbol and SubString matrices and tensors of ranks two
# to five, multibyte rows stored through the C entry points, marked and
# unmarked empties, foreign String/AbstractString tokens and a Workspace of
# text arrays written with writedb.
# Represented MVTSeries and structure actions: generate-represented-mvtseries/
# verify-represented-mvtseries (also checked by verify-wheel): MVTSeries with
# MIT/Duration elements of an independent frequency, Int128/UInt128 and
# ComplexF16 elements over every axis family, marked empties, foreign element
# and identity object markers stored through the C entry points, and
# Diagonal/Symmetric/Hermitian matrices materialised from either triangle
# over ordinary, Boolean and represented elements.
# Scalar marker actions: generate-scalar-markers/verify-scalar-markers (reference
# only; not checked by verify-wheel): Int128/UInt128/ComplexF16, Symbol and other
# string forms, Date/DateTime, Rational{T}, integer Complex{T}, Irrational and
# foreign jtype markers injected on ordinary payloads, with Julia's loaded
# type, value, rewrite and error outcomes materialised as siblings.
using TimeSeriesEcon
using Test, SHA, TOML, Pkg, Dates, LinearAlgebra
# Julia rebuilds a Diagonal/Symmetric/Hermitian jtype marker with
# Core.eval(Main, ...), so those names must be in Main itself. The
# wheel verifier runs this script inside an anonymous module, where a
# plain `using` above would not reach Main and the pinned loader would
# raise UndefVarError on its own markers.
Core.eval(Main, :(using LinearAlgebra))
# The same holds for Julia's own Date/DateTime scalar markers.
Core.eval(Main, :(using Dates))

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
quarterly_cases = [("cross_year", 8099, [1.25, -2.5, 0.0, 4.75]),
    ("negative", -1, [1.25, -2.5]), ("zero", 0, [1.25]),
    ("minimum", -131200, [1.25]), ("maximum", 2147483647, [1.25]),
    ("empty", 8096, Float64[]), ("empty_minimum", -131200, Float64[]),
    ("empty_maximum", 2147483647, Float64[])]

annual_cases = [("cross_year", 2024, [1.25, -2.5, 0.0, 4.75]),
    ("negative", -1, [1.25, -2.5]), ("zero", 0, [1.25]),
    ("minimum", -2147483648, [1.25]), ("maximum", 2147483647, [1.25]),
    ("empty", 2024, Float64[]), ("empty_minimum", -2147483648, Float64[]),
    ("empty_maximum", 2147483647, Float64[])]
annual_native_empties = [("native_empty", 2024, Float64[]),
    ("native_empty_minimum", -2147483648, Float64[]),
    ("native_empty_maximum", 2147483647, Float64[])]
# Half-yearly codes are 2 * year + period - 1; -65600 is the lowest code the
# native decoder reproduces (EPOCH_L * ppy = 32800 * 2 in uint32 arithmetic).
halfyearly_cases = [("cross_year", 4048, [1.25, -2.5, 0.0, 4.75]),
    ("negative", -1, [1.25, -2.5]), ("zero", 0, [1.25]),
    ("minimum", -65600, [1.25]), ("maximum", 2147483647, [1.25]),
    ("empty", 4048, Float64[]), ("empty_minimum", -65600, Float64[]),
    ("empty_maximum", 2147483647, Float64[])]
halfyearly_below_minimum = ("below_minimum", -65601, [1.25])
halfyearly_native_empties = [("native_empty", 4048, Float64[]),
    ("native_empty_minimum", -65600, Float64[]),
    ("native_empty_maximum", 2147483647, Float64[])]
# Int64 scalars: metadata (1,1,0,8), little-endian two's complement, no attributes.
# Values around 2^53 expose any floating-point detour; both endpoints are included.
int64_cases = [("zero", 0), ("one", 1), ("negative_one", -1), ("seven", 7),
    ("negative_seven", -7), ("pow53", 2^53), ("pow53_plus_one", 2^53 + 1),
    ("pow53_minus_one", 2^53 - 1), ("negative_pow53_minus_one", -(2^53 + 1)),
    ("pow62_plus_one", 2^62 + 1), ("max", typemax(Int64)), ("min", typemin(Int64)),
    ("min_plus_one", typemin(Int64) + 1), ("max_minus_one", typemax(Int64) - 1)]
# The same encoding written directly through the C ABI, as Python writes it.
int64_native_cases = [("native_pow53_plus_one", 2^53 + 1), ("native_min", typemin(Int64)),
    ("native_max", typemax(Int64)), ("native_negative_one", -1)]
# Unsupported-representation controls sharing the scalar class.
int64_controls = [("bool_true", true), ("int32", Int32(7)), ("int128", Int128(7)),
    ("uint64", UInt64(7)), ("duration_monthly", 2024M4 - 2024M1),
    ("mit_monthly", 2024M1), ("rational", 1 // 2), ("string", "7")]
# Numeric widths: Julia stores Float16/32, Int8/16/32, UInt8..UInt64 and ComplexF32/F64
# at their own width with metadata (1,type,0,sizeof) and no attribute, and reloads them by
# (type, nbytes). Names and values mirror tests/dataecon/test_widths.py and the checker.
width_signed(T) = [("zero", T(0)), ("one", T(1)), ("negative_one", T(-1)), ("seven", T(7)),
    ("min", typemin(T)), ("max", typemax(T)), ("min_plus_one", typemin(T) + T(1)),
    ("max_minus_one", typemax(T) - T(1))]
width_unsigned(T) = [("zero", T(0)), ("one", T(1)), ("seven", T(7)), ("max", typemax(T)),
    ("max_minus_one", typemax(T) - T(1)), ("high_bit", typemax(T) ÷ T(2) + T(1))]
width_floats(T) = [("zero", T(0.0)), ("negative_zero", T(-0.0)),
    (T == Float16 ? "one_and_half" : "one_quarter", T == Float16 ? T(1.5) : T(1.25)),
    ("tenth", T(0.1)), ("max", floatmax(T)), ("negative_max", -floatmax(T)),
    ("min_normal", floatmin(T)), ("min_subnormal", nextfloat(T(0.0))), ("nan", T(NaN)),
    ("inf", T(Inf)), ("negative_inf", T(-Inf)),
    (T == Float16 ? "two_pow_11_plus_one" : "two_pow_24_plus_one", T == Float16 ? T(2049) : T(16777217)),
    ("pi", T(pi))]
width_complex(T) = [("plain", Complex{T}(T(1.5), T(-2.25))),
    ("negative_zero_real", Complex{T}(T(-0.0), T(0.0))),
    ("negative_zero_imag", Complex{T}(T(0.0), T(-0.0))),
    ("nan_real", Complex{T}(T(NaN), T(1.0))), ("inf_imag", Complex{T}(T(1.0), T(-Inf))),
    ("max_min", Complex{T}(floatmax(T), floatmin(T))),
    ("subnormal", Complex{T}(nextfloat(T(0.0)), T(0.0))),
    ("tenths", Complex{T}(T(0.1), T(0.2)))]
width_groups = [("f16", width_floats(Float16)), ("f32", width_floats(Float32)),
    ("i8", width_signed(Int8)), ("i16", width_signed(Int16)), ("i32", width_signed(Int32)),
    ("u8", width_unsigned(UInt8)), ("u16", width_unsigned(UInt16)),
    ("u32", width_unsigned(UInt32)), ("u64", width_unsigned(UInt64)),
    ("c32", width_complex(Float32)), ("c64", width_complex(Float64))]
width_cases = [("w_$(prefix)_$(suffix)", value) for (prefix, cases) in width_groups for (suffix, value) in cases]
# The same encodings written directly through the C ABI, as Python writes them.
width_native_cases = [("f16_tenth", Float16(0.1)), ("f16_nan", Float16(NaN)),
    ("f32_tenth", 0.1f0), ("f32_negative_zero", -0.0f0), ("i8_min", typemin(Int8)),
    ("i16_max", typemax(Int16)), ("i32_min", typemin(Int32)), ("u8_max", typemax(UInt8)),
    ("u16_max", typemax(UInt16)), ("u32_max", typemax(UInt32)), ("u64_max", typemax(UInt64)),
    ("u64_high_bit", UInt64(2)^63), ("c32_plain", ComplexF32(1.5, -2.25)),
    ("c64_plain", ComplexF64(8.0, 3.0)), ("c64_negative_zero", ComplexF64(-0.0, -0.0))]
# Controls: Julia-supported widths without a NumPy scalar type, Bool (byte-identical
# to Int8 and reloaded as Int8 by Julia), Int8 twins, and a Complex{Int} marker object.
width_controls = [("int128", Int128(7)), ("int128_max", typemax(Int128)), ("uint128", UInt128(7)),
    ("uint128_max", typemax(UInt128)), ("c16", ComplexF16(1.5, -2.25)), ("bool_true", true),
    ("bool_false", false), ("int8_one", Int8(1)), ("int8_zero", Int8(0)), ("complex_int", 8 + 3im)]
width_expected_type(value) = value isa Bool ? Int8 : typeof(value)
width_bits(value) = bytes2hex(reinterpret(UInt8, [value]))
# File operations: the series written, deleted, overwritten and truncated around.
fileops_series = TSeries(2024M1, [1.0, 2.0, 3.0])
# String scalars: metadata (1,6,0,n), UTF-8 bytes plus one NUL terminator, no attributes.
string_cases = [("ascii", "hello"), ("empty", ""), ("digits", "7"), ("space", " "),
    ("latin", "héllo wörld"), ("cjk", "日本語"), ("emoji", "🙂 ok"), ("combining", "é"),
    ("whitespace", "a\nb\tc\r\n"), ("delimiter", "a‖b"), ("slash", "a/b"),
    ("quote", "say \"hi\""), ("expression", "error(123)"), ("long", repeat("x", 1000)),
    ("long_utf8", repeat("é", 500))]
# Controls: Julia truncates an embedded NUL on load and does not validate UTF-8;
# Python rejects both payloads. A Symbol carries a jtype marker that Python rejects.
string_controls = [("embedded_nul", "a\0b"), ("invalid_utf8", String(UInt8[0x66, 0xff, 0x6f]))]
# Payloads Julia never writes: no terminator, NULL payload, and a frequency code.
string_natives = [("no_terminator", UInt8[0x68, 0x69], C.freq_none),
    ("empty_payload", UInt8[], C.freq_none), ("only_nul", UInt8[0x00], C.freq_none),
    ("with_frequency", UInt8[0x68, 0x69, 0x00], C.freq_monthly)]
# Year/period frequency families for MIT date and Duration scalars:
# (label, type, native code, periods per year, reliable minimum code).
date_families = Any[("m", Monthly, 32, 12, -393600)]
for a in 1:3
    push!(date_families, ("q$(a)", Quarterly{a}, 64 + a, 4, -131200))
end
for a in 1:6
    push!(date_families, ("h$(a)", HalfYearly{a}, 128 + a, 2, -65600))
end
for a in 1:12
    push!(date_families, ("y$(a)", Yearly{a}, 256 + a, 1, -2147483648))
end
date_codes(ppy, minimum) = [("typical", 2024 * ppy), ("cross_year", 2024 * ppy + ppy - 1),
    ("negative_one", -1), ("zero", 0), ("minimum", minimum), ("maximum", 2147483647)]
duration_values = [("zero", 0), ("one", 1), ("negative_one", -1),
    ("pow53_plus_one", 2^53 + 1), ("max", typemax(Int64)), ("min", typemin(Int64))]
# Outstanding frequencies (unit/calendar) and non-canonical anchors that Julia
# collapses to canonical native codes; Python rejects the former for now.
date_controls = [("unit_mit", MIT{Unit}(5)), ("unit_duration", Duration{Unit}(3)),
    ("daily_mit", d"2024-01-15"), ("daily_duration", d"2024-01-15" - d"2024-01-01"),
    ("bdaily_mit", bd"2024-01-15"), ("weekly7_mit", w"2024-01-14"),
    ("quarterly4_mit", MIT{Quarterly{4}}(2024, 1))]
# Native metadata Julia never writes for dates/durations.
date_natives = [("date_no_freq", C.type_date, C.freq_none, 8),
    ("date_four_bytes", C.type_date, C.freq_monthly, 4),
    ("date_mixed_bits", C.type_date, C.frequency_t(192), 8),
    ("date_bare_quarterly", C.type_date, C.frequency_t(64), 8),
    ("duration_four_bytes", C.type_signed, C.freq_monthly, 4)]
# Calendar scalar families: (label, type, native code, end day or 0). Unit is code 11.
calendar_families = Any[("d", Daily, 12, 0), ("b", BDaily, 13, 0)]
for ed in 1:7
    push!(calendar_families, ("w$(ed)", Weekly{ed}, 16 + ed, ed))
end
# Exact native calendar windows: the decoder shifts bound the minimum (1 March
# -32800 for daily, the last week of December -32800 for business daily and
# weekly); the encoder's year check bounds the maximum in December 32800.
calendar_windows = Dict(12 => (-11980259, 11979954), 13 => (-8557114, 8557110),
    (16 + ed => (-1711422, 1711422) for ed in 1:7)...)
_weekday_on_or_after(d) = d + Day(dayofweek(d) > 5 ? 8 - dayofweek(d) : 0)
_calendar_mit(::Type{Daily}, d, ed) = daily(d)
_calendar_mit(::Type{BDaily}, d, ed) = bdaily(_weekday_on_or_after(d))
_calendar_mit(::Type{<:Weekly}, d, ed) = weekly(d, ed)
function calendar_codes(F, code, ed)
    at(d) = Int(_calendar_mit(F, d, ed))
    lo, hi = calendar_windows[code]
    return [("typical", at(Date(2024, 1, 15))), ("year_end", at(Date(2024, 12, 31))),
        ("year_start", at(Date(2025, 1, 1))), ("leap_day", at(Date(2024, 2, 29))),
        ("first_day", at(Date(1, 1, 1))), ("year_zero", at(Date(0, 6, 15))),
        ("negative_year", at(Date(-5, 3, 3))), ("negative_one", -1), ("zero", 0),
        ("minimum", lo), ("maximum", hi), ("py_max_year", at(Date(9999, 12, 31))),
        ("beyond_py_year", at(Date(10000, 1, 3)))]
end
# Unit codes are Julia's generic Int64 pass-through: both endpoints and values beyond Int32.
unit_values = [("min", typemin(Int64)), ("neg_pow40", -2^40), ("below_int32", -2^31 - 1),
    ("int32_min", -2^31), ("negative_one", -1), ("zero", 0), ("five", 5),
    ("int32_max", 2^31 - 1), ("beyond_int32", 2^31), ("pow40", 2^40), ("max", typemax(Int64))]
# Julia stores these below-window codes with a "codes differ" warning; daily and
# business daily reload as another date, weekly happens to reload correctly.
calendar_below_window = [("d", Daily, -11980260), ("b", BDaily, -8557115), ("w7", Weekly{7}, -1711423)]
# Native encodings Julia never writes: the Sunday alias 16, the unused code 14,
# an out-of-range weekly anchor, a four-byte unit payload, the first daily code
# above the window and an Int32-wrapped daily code.
calendar_natives = [("date_weekly16", C.type_date, C.frequency_t(16), 8, 105557),
    ("date_freq14", C.type_date, C.frequency_t(14), 8, 738900),
    ("date_weekly24", C.type_date, C.frequency_t(24), 8, 105557),
    ("date_unit_four_bytes", C.type_date, C.freq_unit, 4, 5),
    ("date_daily_above_maximum", C.type_date, C.freq_daily, 8, 11979955),
    ("date_daily_int32_wrap", C.type_date, C.freq_daily, 8, 2^32 + 738900),
    ("duration_weekly16", C.type_signed, C.frequency_t(16), 8, 2)]
# Calendar-frequency Float64 series: the axis stores the packed calendar code of
# the first observation; the verified scalar windows bound every observation.
# Nonempty cases cross a year end, leap day 2024 and a weekend (Friday start),
# and reach both window endpoints and the years beyond Python's datetime.
_series_mit(::Type{Daily}, d, ed) = daily(d)
_series_mit(::Type{BDaily}, d, ed) = bdaily(d)
_series_mit(::Type{<:Weekly}, d, ed) = weekly(d, ed)
function calendar_series_cases(F, code, ed)
    at(d) = Int(_series_mit(F, d, ed))
    lo, hi = calendar_windows[code]
    return [("cross_year", at(Date(2024, 12, 30)), [1.25, -2.5, 0.0, 4.75]),
        ("leap", at(Date(2024, 2, 28)), [1.25, -2.5, 0.0]),
        ("weekend", at(Date(2024, 1, 12)), [1.25, -2.5]),
        ("negative", -1, [1.25, -2.5]), ("zero", 0, [1.25]),
        ("minimum", lo, [1.25]), ("maximum", hi, [1.25]),
        ("beyond_py_year", at(Date(9999, 12, 31)), [1.25, -2.5, 0.0, 4.75, 8.5]),
        # Trailing observations may run past the window: only the first date is
        # packed, as in Julia. Two values from the maximum, and 1,000 values
        # straddling it (500 past the window).
        ("last_beyond_maximum", hi, [1.25, -2.5]),
        ("long_span", hi - 499, [0.25k for k in 1:1000]),
        ("empty", at(Date(2024, 1, 15)), Float64[]),
        ("empty_minimum", lo, Float64[]), ("empty_maximum", hi, Float64[])]
end
# Reference-only control per family: Julia stores a below-window first date
# with a "codes differ" warning; native objects Julia never writes: a
# marker-free empty series, a first date just above the window and an
# Int32-wrapped first date. Python rejects all but the marker-free empty.
function calendar_series_controls(F, code, ed)
    lo, hi = calendar_windows[code]
    return [("below_window", lo - 1, [1.25])]
end
function calendar_series_natives(F, code, ed)
    lo, hi = calendar_windows[code]
    return [("native_empty", Int(_series_mit(F, Date(2024, 1, 15), ed)), Float64[]),
        ("native_first_above_maximum", hi + 1, [1.25]),
        ("native_int32_wrap", 2^32 + Int(_series_mit(F, Date(2024, 1, 15), ed)), [1.25])]
end
function store_native_series!(db, name, freq, code, values)
    axis = Ref{C.axis_id_t}()
    id = Ref{C.obj_id_t}()
    @test C.de_axis_range(db, length(values), C.frequency_t(freq), code, axis) == 0
    GC.@preserve values begin
        ptr = isempty(values) ? C_NULL : pointer(values)
        @test C.de_store_tseries(db, DE.root_id, name, C.type_tseries, C.type_float,
            C.freq_none, axis[], 8length(values), ptr, id) == 0
    end
end
function load_series_raw(db, id)
    arr = Ref{C.tseries_t}()
    @test C.de_load_tseries(db, id, arr) == 0
    ts = arr[]
    metadata = Int.((ts.object.obj_class, ts.object.obj_type, ts.eltype, ts.elfreq,
        ts.axis.ax_type, ts.axis.length, ts.axis.frequency, ts.axis.first, ts.nbytes))
    attrs = Dict(string(k) => string(v) for (k, v) in DE.get_all_attributes(db, id))
    return metadata, ts.value == C_NULL, attrs
end
# Function barrier: F is chosen at runtime.
function write_calendar_series!(db, ::Type{F}, label, code, ed) where {F}
    for (suffix, first, values) in calendar_series_cases(F, code, ed)
        DE.store_tseries(db, DE.root_id, "cs_$(label)_$(suffix)", TSeries(MIT{F}(first), copy(values)))
    end
    for (suffix, first, values) in calendar_series_controls(F, code, ed)
        ts = TSeries(MIT{F}(first), copy(values))
        @test_logs (:warn, r"MIT codes differ") DE.store_tseries(db, DE.root_id, "ctl_$(label)_$(suffix)", ts)
    end
    for (suffix, first, values) in calendar_series_natives(F, code, ed)
        store_native_series!(db, "cs_$(label)_$(suffix)", code, first, values)
    end
end
function verify_calendar_series(db, ::Type{F}, label, code, ed, reference_fixture) where {F}
    for (suffix, first, values) in calendar_series_cases(F, code, ed)
        id = DE.find_object(db, DE.root_id, "cs_$(label)_$(suffix)")
        metadata, null_value, attrs = load_series_raw(db, id)
        @test metadata == (2, 12, 4, 0, 1, length(values), code, first, 8length(values))
        @test null_value == isempty(values)
        value = @test_logs DE.load_tseries(db, id)
        if reference_fixture && isempty(values)
            @test attrs == Dict("jeltype" => "Float64")
            @test value isa Vector{Float64} && isempty(value)
        else
            @test isempty(attrs)
            @test value isa TSeries{F,Float64}
            @test Int(firstdate(value)) == first
            @test Int(lastdate(value)) == first + length(values) - 1
            @test value.values == values
        end
    end
    reference_fixture || return
    lo, hi = calendar_windows[code]
    for (suffix, first, values) in calendar_series_controls(F, code, ed)
        id = DE.find_object(db, DE.root_id, "ctl_$(label)_$(suffix)")
        metadata, _, attrs = load_series_raw(db, id)
        @test metadata == (2, 12, 4, 0, 1, length(values), code, first, 8length(values))
        @test isempty(attrs)
        if F <: Weekly
            # The weekly decoder is exact modulo 2^32 for this code.
            @test Int(firstdate(@test_logs DE.load_tseries(db, id))) == first
        else
            value = @test_logs (:warn, r"MIT codes differ") DE.load_tseries(db, id)
            @test value isa TSeries{F,Float64} && Int(firstdate(value)) != first
        end
    end
    for (suffix, first, values) in calendar_series_natives(F, code, ed)
        id = DE.find_object(db, DE.root_id, "cs_$(label)_$(suffix)")
        metadata, _, attrs = load_series_raw(db, id)
        @test metadata == (2, 12, 4, 0, 1, length(values), code, first, 8length(values))
        @test isempty(attrs)
        if suffix == "native_int32_wrap"
            value = @test_logs (:warn, r"MIT codes differ") DE.load_tseries(db, id)
            @test Int(firstdate(value)) == first - 2^32
        else
            value = @test_logs DE.load_tseries(db, id)
            @test value isa TSeries{F,Float64} && Int(firstdate(value)) == first
            @test Int(lastdate(value)) == first + length(values) - 1
        end
    end
end



# Numeric/Boolean series actions share cases with the installed verifier.
function series_element_values(T)
    T <: Signed && return T[typemin(T), -1, 0, typemax(T)]
    T <: Unsigned && return T[0, 1, typemax(T)]
    T == Bool && return Bool[false, true, false]
    T == Float16 && return collect(reinterpret(T, UInt16[0x8000,1,0x3d00,0x7e55,0x7c00]))
    T == Float32 && return collect(reinterpret(T, UInt32[0x80000000,1,0x3fa00000,0x7fc00055,0x7f800000]))
    T == Float64 && return collect(reinterpret(T, UInt64[0x8000000000000000,1,0x3ff4000000000000,0x7ff8000000000055,0x7ff0000000000000]))
    return T[complex(-0.0,1.25), complex(2.5,-3.0)]
end
const series_element_types = (Int8,Int16,Int32,Int64,UInt8,UInt16,UInt32,UInt64,
    Float16,Float32,Float64,ComplexF32,ComplexF64,Bool)
function series_element_cases()
    cases = Any[]
    for T in series_element_types
        push!(cases, ("es_$(T)",2024M11,series_element_values(T)))
        push!(cases, ("es_$(T)_empty",2024M11,T[]))
    end
    families = Any[(Monthly,32),(Daily,12),(BDaily,13)]
    append!(families,[(Weekly{d},16+d) for d in 1:7])
    append!(families,[(Quarterly{m},64+m) for m in 1:3])
    append!(families,[(HalfYearly{m},128+m) for m in 1:6])
    append!(families,[(Yearly{m},256+m) for m in 1:12])
    for (F,code) in families
        push!(cases,("es_axis_$(code)",convert(MIT{F},Int64(100)),Int16[-7,0,23]))
    end
    for (F,code,maximum) in Any[(Daily,12,11979954),(BDaily,13,8557110),
            [(Weekly{d},16+d,1711422) for d in 1:7]...]
        push!(cases,("es_trailing_$(code)",convert(MIT{F},Int64(maximum)),Int8[-1,1]))
    end
    return cases
end

# Represented series elements (also checked by verify-wheel): MIT/Duration elements
# with their own element frequency over all 32 canonical codes, Int128/UInt128 and
# ComplexF16 carriers, marked empties and Julia rewrites. The fixture additionally
# holds reference-only controls written through the C ABI: malformed widths/kinds/
# frequencies, noncanonical element codes, contradictory markers, the foreign Bool
# rows, the wide-Bool rows and other foreign markers, each with its recorded Julia
# load outcome. Actions: generate-represented-elements/verify-represented-elements.
const represented_anchor = 2024M11   # code 24298
const represented_codes = Int64[typemin(Int64), -1, 0, typemax(Int64)]
represented_families = Any[(Unit, 11), (Daily, 12), (BDaily, 13), (Monthly, 32)]
append!(represented_families, [(Weekly{d}, 16 + d) for d in 1:7])
append!(represented_families, [(Quarterly{m}, 64 + m) for m in 1:3])
append!(represented_families, [(HalfYearly{m}, 128 + m) for m in 1:6])
append!(represented_families, [(Yearly{m}, 256 + m) for m in 1:12])
# Function barrier: F is chosen at runtime.
function represented_date_cases!(cases, ::Type{F}, code) where {F}
    push!(cases, ("rs_mit_$(code)", represented_anchor, collect(reinterpret(MIT{F}, represented_codes))))
    push!(cases, ("rs_mit_$(code)_empty", represented_anchor, MIT{F}[]))
    push!(cases, ("rs_dur_$(code)", represented_anchor, collect(reinterpret(Duration{F}, represented_codes))))
    push!(cases, ("rs_dur_$(code)_empty", represented_anchor, Duration{F}[]))
end
represented_int128_words = Int128[Int128(2)^64, -(Int128(2)^64) - 1,
    Int128(0x0123456789abcdef) << 64 | 0x0fedcba987654321, typemin(Int128) + 1, typemax(Int128) - 1]
represented_uint128_words = UInt128[UInt128(2)^64, UInt128(2)^127, typemax(UInt128) - 1]
represented_c16_real = UInt16[0x8000, 0x0001, 0x7e55, 0x7c00, 0xfc00, 0x7bff]
represented_c16_imag = UInt16[0x3d00, 0x8001, 0x7e00, 0x0000, 0x7bff, 0xfbff]
represented_complexf16 = [ComplexF16(reinterpret(Float16, r), reinterpret(Float16, i))
    for (r, i) in zip(represented_c16_real, represented_c16_imag)]
# Shared with the installed verifier: (name, first date, values). Python writes the
# same objects from StoredSeries containers into the primary interchange output.
function represented_cases()
    cases = Any[]
    for (F, code) in represented_families
        represented_date_cases!(cases, F, code)
    end
    push!(cases, ("rs_mit_268_on_daily", MIT{Daily}(Date(2024, 11, 1)),
        MIT{Yearly{12}}[MIT{Yearly{12}}(2024), MIT{Yearly{12}}(2025)]))
    push!(cases, ("rs_dur_12_on_yearly", MIT{Yearly{12}}(2024),
        Duration{Daily}[Duration{Daily}(-5), Duration{Daily}(0), Duration{Daily}(7)]))
    push!(cases, ("rs_int128_endpoints", represented_anchor, Int128[typemin(Int128), -1, 0, typemax(Int128)]))
    push!(cases, ("rs_int128_words", represented_anchor, represented_int128_words))
    push!(cases, ("rs_uint128_endpoints", represented_anchor, UInt128[0, UInt128(2)^64, typemax(UInt128)]))
    push!(cases, ("rs_uint128_words", represented_anchor, represented_uint128_words))
    push!(cases, ("rs_complexf16_bits", represented_anchor, represented_complexf16))
    for T in (Int128, UInt128, ComplexF16)
        push!(cases, ("rs_$(T)_empty", represented_anchor, T[]))
    end
    return cases
end
# Julia rewrites of loaded represented values must preserve metadata, bytes and markers.
represented_rewrites = ("rs_mit_32", "rs_int128_words", "rs_complexf16_bits")
# Wide carriers with a Bool marker: Python writes preserved StoredSeries containers
# (kinds 1, 2 and 5 at their own width) and their explicit to_bool() conversions;
# the fixture stores the same bytes through the C ABI. Julia loads both as Bool.
represented_bool_cases = [("rs_bool_int128", 1, collect(reinterpret(UInt8, Int128[0, 1]))),
    ("rs_bool_uint128", 2, collect(reinterpret(UInt8, UInt128[0, 1]))),
    ("rs_bool_complexf16", 5, collect(reinterpret(UInt8, ComplexF16[0, 1])))]
# Julia's rewrite of a Bool-converted foreign or wide carrier stores the canonical
# one-byte encoding with the marker; it never preserves the foreign width.
represented_bool_rewrites = (("fb_int16_01", UInt8[0, 1, 0]), ("fb_float64_01", UInt8[0, 1]),
    ("wb_uint128_01", UInt8[0, 1]), ("wb_int128_010", UInt8[0, 1, 0]),
    ("wb_complexf16_01_negzero_imag", UInt8[0, 1]))
# Reference-only controls: (name, kind, element frequency, payload, length, marker,
# expected load). The expectation is an exception type for a failed load, otherwise
# the loaded element type (a Vector of it for an empty payload).
function represented_controls()
    i64(v...) = collect(reinterpret(UInt8, Int64[v...]))
    c16(r, i) = ComplexF16(reinterpret(Float16, UInt16(r)), reinterpret(Float16, UInt16(i)))
    bytesof(v) = collect(reinterpret(UInt8, v))
    controls = Any[
        # Empty date/duration elements: the pinned loader has no zero-width method,
        # with or without the marker Julia's own writer stores (rs_*_empty rows).
        ("ctl_mit_32_empty_unmarked", 3, 32, UInt8[], 0, nothing, MethodError),
        ("ctl_dur_32_empty_unmarked", 1, 32, UInt8[], 0, nothing, MethodError),
        # Element widths and kinds Julia never writes with an element frequency.
        ("ctl_dur_32_width4", 1, 32, bytesof(Int32[1, 2]), 2, nothing, MethodError),
        ("ctl_mit_32_width16", 3, 32, bytesof(Int128[1, 2]), 2, nothing, MethodError),
        ("ctl_unsigned_elfreq_32", 2, 32, bytesof(UInt64[1, 2]), 2, nothing, MethodError),
        ("ctl_float_elfreq_32", 4, 32, bytesof(Float64[1, 2]), 2, nothing, MethodError),
        ("ctl_complex_elfreq_32", 5, 32, bytesof(ComplexF64[1, 2]), 2, nothing, MethodError),
        # A matching marker on nonempty dates is an identity conversion; contradictory
        # markers convert or fail case by case through Julia's generic loader.
        ("ctl_mit_32_marked_nonempty", 3, 32, i64(24298, 24299), 2, "MIT{Monthly}", MIT{Monthly}),
        ("ctl_mit_32_marker_quarterly", 3, 32, i64(0, 1), 2, "MIT{Quarterly{3}}", MethodError),
        ("ctl_mit_32_marker_int64", 3, 32, i64(0, 1), 2, "Int64", Int64),
        ("ctl_mit_32_marker_float64", 3, 32, i64(0, 1), 2, "Float64", Float64),
        ("ctl_mit_32_marker_bool", 3, 32, i64(0, 1), 2, "Bool", Bool),
        ("ctl_dur_32_marker_bool", 1, 32, i64(0, 1), 2, "Bool", Bool),
        ("ctl_dur_32_marker_int64", 1, 32, i64(0, 1), 2, "Int64", Int64),
        ("ctl_int128_marker_uint128", 1, 0, bytesof(Int128[1]), 1, "UInt128", UInt128),
        ("ctl_int128_marker_float64", 1, 0, bytesof(Int128[1]), 1, "Float64", Float64),
        ("ctl_uint128_marker_int128", 2, 0, bytesof(UInt128[1]), 1, "Int128", Int128),
        ("ctl_complexf16_marker_float16", 5, 0, bytesof(ComplexF16[1]), 1, "Float16", Float16),
        # Empty payloads whose wide marker contradicts the native kind.
        ("ctl_kind1_empty_marker_uint128", 1, 0, UInt8[], 0, "UInt128", UInt128),
        ("ctl_kind1_empty_marker_complexf16", 1, 0, UInt8[], 0, "ComplexF16", ComplexF16),
        ("ctl_kind5_empty_marker_int128", 5, 0, UInt8[], 0, "Int128", Int128),
    ]
    # Noncanonical element frequency codes with valid Int64 codes: 14, 15 and the
    # monthly alias 33 fail in Julia (the element loader dispatches on the enum value,
    # unlike the scalar path); the weekly aliases 16 and 24..31 and the bare family
    # codes 64/128/256 load as frequency types Julia's writer never emits.
    for code in (14, 15, 33)
        push!(controls, ("ctl_mit_elfreq_$(code)", 3, code, i64(0, 1), 2, nothing, ErrorException))
    end
    push!(controls, ("ctl_dur_elfreq_14", 1, 14, i64(0, 1), 2, nothing, ErrorException))
    noncanonical = Any[(16, Weekly{0}), (64, Quarterly{0}), (128, HalfYearly{0}), (256, Yearly{0})]
    append!(noncanonical, [(16 + d, Weekly{d}) for d in 8:15])
    for (code, F) in noncanonical
        push!(controls, ("ctl_mit_elfreq_$(code)", 3, code, i64(0, 1), 2, nothing, MIT{F}))
    end
    for (code, F) in ((16, Weekly{0}), (24, Weekly{8}), (64, Quarterly{0}))
        push!(controls, ("ctl_dur_elfreq_$(code)", 1, code, i64(0, 1), 2, nothing, Duration{F}))
    end
    # Foreign Bool markers on ordinary numeric carriers: exact 0/1 values load as
    # Bool (signed zero included), anything else is an InexactError, empties are Bool[].
    foreign_bool = Any[
        ("fb_int16_01", 1, bytesof(Int16[0, 1, 0]), 3, Bool),
        ("fb_int16_2", 1, bytesof(Int16[2]), 1, InexactError),
        ("fb_int16_neg1", 1, bytesof(Int16[-1]), 1, InexactError),
        ("fb_int64_01", 1, bytesof(Int64[0, 1]), 2, Bool),
        ("fb_int128_01", 1, bytesof(Int128[1, 0]), 2, Bool),
        ("fb_uint8_01", 2, UInt8[0, 1, 1], 3, Bool),
        ("fb_uint8_2", 2, UInt8[2], 1, InexactError),
        ("fb_uint64_1", 2, bytesof(UInt64[1]), 1, Bool),
        ("fb_float16_01", 4, bytesof(Float16[0, 1]), 2, Bool),
        ("fb_float32_half", 4, bytesof(Float32[0.5]), 1, InexactError),
        ("fb_float64_01", 4, bytesof(Float64[0.0, 1.0]), 2, Bool),
        ("fb_float64_negzero", 4, bytesof(Float64[-0.0]), 1, Bool),
        ("fb_float64_nan", 4, bytesof(Float64[NaN]), 1, InexactError),
        ("fb_float64_two", 4, bytesof(Float64[2.0]), 1, InexactError),
        ("fb_complexf64_10", 5, bytesof(ComplexF64[1 + 0im, 0 + 0im]), 2, Bool),
        ("fb_complexf64_0i", 5, bytesof(ComplexF64[0 + 1im]), 1, InexactError),
        ("fb_complexf16_1", 5, bytesof(ComplexF16[1 + 0im]), 1, Bool),
        ("fb_int16_empty", 1, UInt8[], 0, Bool),
        ("fb_uint8_empty", 2, UInt8[], 0, Bool),
        ("fb_float64_empty", 4, UInt8[], 0, Bool),
        ("fb_complex_empty", 5, UInt8[], 0, Bool),
        # Wide carriers with a Bool marker: the same 0/1 rule at 16 or 4 bytes.
        ("wb_uint128_01", 2, bytesof(UInt128[0, 1]), 2, Bool),
        ("wb_uint128_2", 2, bytesof(UInt128[2]), 1, InexactError),
        ("wb_uint128_hiword", 2, bytesof(UInt128[UInt128(2)^64]), 1, InexactError),
        ("wb_uint128_max", 2, bytesof(UInt128[typemax(UInt128)]), 1, InexactError),
        ("wb_int128_hiword", 1, bytesof(Int128[Int128(2)^64]), 1, InexactError),
        ("wb_int128_neg1", 1, bytesof(Int128[-1]), 1, InexactError),
        ("wb_int128_010", 1, bytesof(Int128[0, 1, 0]), 3, Bool),
        ("wb_complexf16_negzero", 5, bytesof([c16(0x8000, 0x0000)]), 1, Bool),
        ("wb_complexf16_01_negzero_imag", 5, bytesof([c16(0x0000, 0x0000), c16(0x3c00, 0x8000)]), 2, Bool),
        ("wb_complexf16_imag_one", 5, bytesof([c16(0x0000, 0x3c00)]), 1, InexactError),
        ("wb_complexf16_nan", 5, bytesof([c16(0x7e00, 0x0000)]), 1, InexactError),
        ("wb_complexf16_nan_payload_imag", 5, bytesof([c16(0x3c00, 0x7e55)]), 1, InexactError),
        ("wb_complexf16_inf", 5, bytesof([c16(0x7c00, 0x0000)]), 1, InexactError),
        ("wb_complexf16_two", 5, bytesof([c16(0x4000, 0x0000)]), 1, InexactError),
        ("wb_complexf16_subnormal", 5, bytesof([c16(0x0001, 0x0000)]), 1, InexactError),
        # Empty Bool-marked payloads of kinds 1, 2 and 5 carry no width: all give Bool[].
        ("wb_kind1_empty", 1, UInt8[], 0, Bool),
        ("wb_kind2_empty", 2, UInt8[], 0, Bool),
        ("wb_kind5_empty", 5, UInt8[], 0, Bool),
    ]
    for (name, kind, payload, len, expect) in foreign_bool
        push!(controls, (name, kind, 0, payload, len, "Bool", expect))
    end
    # Other foreign markers Julia's generic loader honours, fails on, or cannot resolve.
    foreign_markers = Any[
        ("fm_int16_as_int64", 1, bytesof(Int16[-1, 7]), 2, "Int64", Int64),
        ("fm_int64_as_float64", 1, bytesof(Int64[3]), 1, "Float64", Float64),
        ("fm_float64_as_int8", 4, bytesof(Float64[3.0, 2.5]), 2, "Int8", InexactError),
        ("fm_int64_as_int128", 1, bytesof(Int64[3]), 1, "Int128", Int128),
        ("fm_int64_as_complexf16", 1, bytesof(Int64[3]), 1, "ComplexF16", ComplexF16),
        ("fm_int64_as_mit", 1, bytesof(Int64[24298]), 1, "MIT{Monthly}", MIT{Monthly}),
        ("fm_int64_unknown", 1, bytesof(Int64[3]), 1, "NotAType", UndefVarError),
    ]
    for (name, kind, payload, len, marker, expect) in foreign_markers
        push!(controls, (name, kind, 0, payload, len, marker, expect))
    end
    return controls
end
function store_native_element_series!(db, name, kind, elfreq, payload::Vector{UInt8}, len, marker)
    axis = Ref{C.axis_id_t}()
    id = Ref{C.obj_id_t}()
    @test C.de_axis_range(db, len, C.freq_monthly, Int(represented_anchor), axis) == 0
    GC.@preserve payload begin
        ptr = isempty(payload) ? C_NULL : pointer(payload)
        @test C.de_store_tseries(db, DE.root_id, name, C.type_tseries, C.type_t(kind),
            C.frequency_t(elfreq), axis[], length(payload), ptr, id) == 0
    end
    marker === nothing || DE.set_attribute(db, id[], "jeltype", marker)
    return id[]
end
function raw_series(db, id)
    arr = Ref{C.tseries_t}()
    @test C.de_load_tseries(db, id, arr) == 0
    ts = arr[]
    metadata = Int.((ts.object.obj_class, ts.object.obj_type, ts.eltype, ts.elfreq,
        ts.axis.ax_type, ts.axis.length, ts.axis.frequency, ts.axis.first, ts.nbytes))
    bytes = ts.nbytes == 0 ? UInt8[] : copy(unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(ts.value), ts.nbytes))
    attrs = Dict(string(k) => string(v) for (k, v) in DE.get_all_attributes(db, id))
    return metadata, bytes, attrs
end
function verify_represented_case(db, name, first, values)
    T = eltype(values)
    id = DE.find_object(db, DE.root_id, name)
    metadata, bytes, attrs = raw_series(db, id)
    element = DE.I._eltypefreq(T)
    @test metadata == (2, 12, Int(element.eltype), Int(element.elfreq), 1, length(values),
        Int(DE.I._to_de_scalar_freq(frequencyof(first))), Int(first), sizeof(values))
    @test bytes == collect(reinterpret(UInt8, values))
    # Every writer marks an empty series with the exact type token; nonempty
    # represented series carry no marker.
    @test attrs == (isempty(values) ? Dict("jeltype" => string(T)) : Dict{String,String}())
    if isempty(values) && element.elfreq != C.freq_none
        # The pinned loader has no zero-width method for date/duration elements.
        @test_throws MethodError DE.load_tseries(db, id)
    elseif isempty(values)
        loaded = DE.load_tseries(db, id)
        @test loaded isa Vector{T} && isempty(loaded)
    else
        loaded = DE.load_tseries(db, id)
        @test loaded isa TSeries{frequencyof(first),T}
        @test firstdate(loaded) == first
        @test collect(reinterpret(UInt8, loaded.values)) == bytes
    end
end
function verify_represented_control(db, name, kind, elfreq, payload, len, marker, expect)
    id = DE.find_object(db, DE.root_id, name)
    metadata, bytes, attrs = raw_series(db, id)
    @test metadata == (2, 12, kind, elfreq, 1, len, 32, Int(represented_anchor), length(payload))
    @test bytes == payload
    @test attrs == (marker === nothing ? Dict{String,String}() : Dict("jeltype" => marker))
    if expect <: Exception
        @test_throws expect DE.load_tseries(db, id)
    elseif len == 0
        loaded = DE.load_tseries(db, id)
        @test loaded isa Vector{expect} && isempty(loaded)
    else
        loaded = DE.load_tseries(db, id)
        @test loaded isa TSeries{Monthly,expect}
        @test firstdate(loaded) == represented_anchor
    end
end
function verify_canonical_bool(db, name, expected::Vector{UInt8})
    id = DE.find_object(db, DE.root_id, name)
    metadata, bytes, attrs = raw_series(db, id)
    @test metadata == (2, 12, 1, 0, 1, length(expected), 32, Int(represented_anchor), length(expected))
    @test bytes == expected
    @test attrs == Dict("jeltype" => "Bool")
    loaded = DE.load_tseries(db, id)
    @test loaded isa TSeries{Monthly,Bool} && loaded.values == Bool.(expected)
end

# Foreign reconstruction markers (also checked by verify-wheel): numeric, wide and
# date/duration sources stored through the C ABI with jeltype/jtype attributes that
# Julia's own writer never emits for them. The generator loads every object with the
# pinned loader and materializes the outcome inside the fixture: a "<name>_julia"
# sibling written by Julia's own writer from the loaded value, or a "<name>_error"
# string scalar naming the exception type. Python compares its preserved container
# and explicit interpretation with those siblings; nothing is evaluated from text.
# Actions: generate-foreign-markers/verify-foreign-markers.
const foreign_anchor = 2024M11   # code 24298
const foreign_numeric_types = (Int8, Int16, Int32, Int64, Int128, UInt8, UInt16, UInt32,
    UInt64, UInt128, Float16, Float32, Float64, ComplexF16, ComplexF32, ComplexF64)
const foreign_matrix_sources = (Int8, Int64, Int128, UInt8, UInt64, UInt128, Float16,
    Float64, ComplexF16, ComplexF64)
const foreign_tokens = ("Bool", string.(foreign_numeric_types)..., "MIT{Monthly}", "Duration{Monthly}")
foreign_token_name(token) = replace(token, "{" => "_", "}" => "", ", " => "_", "," => "_")
foreign_c16(r, i) = ComplexF16(reinterpret(Float16, UInt16(r)), reinterpret(Float16, UInt16(i)))
# Curated cases shared with the installed verifier: (name, values, jeltype, jtype).
# Python writes the same objects from StoredSeries containers plus its explicit
# interpretation as "<name>_interpreted"; Julia loads both and compares them.
function foreign_shared_cases()
    Any[
        ("fx_int16_as_int64", Int16[1, 2], "Int64", nothing),
        ("fx_int64_precision_as_float64", Int64[2^53 + 1], "Float64", nothing),
        ("fx_int128_as_float64", Int128[Int128(2)^100 + 1], "Float64", nothing),
        ("fx_mit_monthly_as_bool", MIT{Monthly}[MIT{Monthly}(0), MIT{Monthly}(1)], "Bool", nothing),
        ("fx_duration_monthly_as_bool", Duration{Monthly}[Duration{Monthly}(0), Duration{Monthly}(1)], "Bool", nothing),
        ("fx_int64_as_mit_monthly", Int64[1, 2], "MIT{Monthly}", nothing),
        ("fx_int64_as_duration_daily", Int64[-5, 7], "Duration{Daily}", nothing),
        ("fx_float64_as_complexf16", Float64[1.1, 65520.0], "ComplexF16", nothing),
        ("fx_float64_as_float16", Float64[-0.0, 5e-8, 65504.0, 65520.0, 2.0^-25, 3.0 * 2.0^-25], "Float16", nothing),
        ("fx_int64_midpoints_as_float32", Int64[2^54 + 2^30 + 1, 2^54 + 2^30 - 1], "Float32", nothing),
        ("fx_uint64_midpoints_as_float64", UInt64[UInt64(2)^63 + 2^10 + 1, UInt64(2)^63 + 2^10 - 1, typemax(UInt64)], "Float64", nothing),
        ("fx_int128_midpoints_as_float32", Int128[Int128(2)^100 + Int128(2)^76 + 1, Int128(2)^100 + Int128(2)^76 - 1], "Float32", nothing),
        ("fx_uint128_max_as_float32", UInt128[typemax(UInt128), UInt128(2)^127], "Float32", nothing),
        ("fx_mit_monthly_as_float64", MIT{Monthly}[MIT{Monthly}(-1), MIT{Monthly}(-13), MIT{Monthly}(typemin(Int64))], "Float64", nothing),
        ("fx_duration_monthly_as_float64", Duration{Monthly}[Duration{Monthly}(-1), Duration{Monthly}(-13)], "Float64", nothing),
        ("fx_mit_daily_as_float32", MIT{Daily}[MIT{Daily}(2^54 + 2^30 + 1), MIT{Daily}(-1)], "Float32", nothing),
        ("fx_mit_quarterly_as_complexf32", MIT{Quarterly{3}}[MIT{Quarterly{3}}(-1), MIT{Quarterly{3}}(7)], "ComplexF32", nothing),
        ("fx_duration_yearly_as_int64", Duration{Yearly{12}}[Duration{Yearly{12}}(typemin(Int64)), Duration{Yearly{12}}(3)], "Int64", nothing),
        ("fx_complexf64_negzero_as_int8", ComplexF64[complex(1.0, -0.0), complex(0.0, -0.0)], "Int8", nothing),
        ("fx_complexf16_as_int64", ComplexF16[ComplexF16(1, 0), ComplexF16(2, 0)], "Int64", nothing),
        ("fx_uint128_as_int128", UInt128[UInt128(2)^127 - 1, 0], "Int128", nothing),
        ("fx_int64_as_uint128", Int64[typemax(Int64), 0], "UInt128", nothing),
        ("fx_int16_alias_as_int", Int16[3, 4], "Int", nothing),
        ("fx_float64_empty_as_int16", Float64[], "Int16", nothing),
        ("fx_int64_empty_as_mit_monthly", Int64[], "MIT{Monthly}", nothing),
        ("fx_complexf64_empty_as_uint128", ComplexF64[], "UInt128", nothing),
        ("fx_outer_tseries_inactive_eltype", Int16[1, 2], "NoSuchElement", "TSeries"),
        ("fx_outer_tseries_bypasses_bool", Int8[0, 1], "Bool", "TSeries{Monthly, Int8}"),
        ("fx_outer_identity_int128", Int128[Int128(2)^100], nothing, "TSeries{Monthly, Int128, Vector{Int128}}"),
        ("fx_outer_identity_mit", MIT{Monthly}[MIT{Monthly}(1)], "Bool", "TSeries{Monthly, MIT{Monthly}}"),
        ("fx_outer_vector_empty", Float64[], "Int16", "Vector"),
        ("fx_outer_vector_float64_empty", Int64[], nothing, "Vector{Float64}"),
    ]
end
# Fixture-only cases: the source/target matrix on 0/1 values, empties per kind,
# date/duration sources over several families, value boundaries and precedence.
function foreign_fixture_cases()
    cases = foreign_shared_cases()
    push!(cases, ("fx_int64_alias_as_int", Int64[3, 4], "Int", nothing))
    for S in foreign_matrix_sources, t in foreign_tokens
        push!(cases, ("fx_m_$(S)_$(foreign_token_name(t))", S[0, 1], t, nothing))
    end
    for S in (Int64, UInt64, Float64, ComplexF64), t in foreign_tokens
        push!(cases, ("fx_e_$(S)_$(foreign_token_name(t))", S[], t, nothing))
    end
    for (F, tag) in ((Monthly, "monthly"), (Yearly{12}, "yearly12"), (Daily, "daily"), (Unit, "unit")),
        (K, kind) in ((MIT, "mit"), (Duration, "dur"))
        S = K{F}
        for t in unique(("Bool", "Int8", "Int64", "Int128", "UInt64", "Float16", "Float32", "Float64",
                  "ComplexF16", "ComplexF32", "ComplexF64", "MIT{Monthly}", "Duration{Monthly}", string(S)))
            push!(cases, ("fx_d_$(kind)_$(tag)_$(foreign_token_name(t))", S[S(0), S(1)], t, nothing))
        end
        push!(cases, ("fx_d_$(kind)_$(tag)_negative_float64", S[S(-1), S(-13), S(2^54 + 2^30 + 1)], "Float64", nothing))
        push!(cases, ("fx_d_$(kind)_$(tag)_negative_float32", S[S(-1), S(-13), S(2^54 + 2^30 + 1)], "Float32", nothing))
        push!(cases, ("fx_d_$(kind)_$(tag)_two_as_bool", S[S(2)], "Bool", nothing))
        push!(cases, ("fx_d_$(kind)_$(tag)_empty_as_bool", S[], "Bool", nothing))
    end
    boundaries = Any[
        ("fx_v_int64_bounds", Int64[typemin(Int64), typemax(Int64)]),
        ("fx_v_int64_precision", Int64[2^53 + 1, 2^62 + 1]),
        ("fx_v_int128_bounds", Int128[typemin(Int128), typemax(Int128)]),
        ("fx_v_uint128_bounds", UInt128[0, typemax(UInt128)]),
        ("fx_v_uint64_high", UInt64[UInt64(2)^63, typemax(UInt64)]),
        ("fx_v_float64_fraction", Float64[3, 2.5]),
        ("fx_v_float64_zero", Float64[-0.0, 0.0]),
        ("fx_v_float64_special", Float64[NaN, Inf, -Inf]),
        ("fx_v_float64_overflow", Float64[65520, 1e300]),
        ("fx_v_float64_int_edge", Float64[-2.0^63, 2.0^63]),
        ("fx_v_float32_subnormal", Float32[1.4f-45, -1.4f-45]),
        ("fx_v_float16_max", Float16[65504]),
        ("fx_v_complexf64_imag", ComplexF64[1 + im, 0 - im]),
        ("fx_v_complexf64_zeroimag", ComplexF64[complex(-0.0, -0.0), complex(1.0, -0.0)]),
        ("fx_v_complexf16_inexact", ComplexF16[ComplexF16(1.5, 0)]),
    ]
    for (name, v) in boundaries
        for t in ("Bool", "Int8", "Int16", "Int32", "Int64", "Int128", "UInt8", "UInt64", "UInt128",
                  "Float16", "Float32", "Float64", "ComplexF16", "ComplexF64", "MIT{Monthly}", "Duration{Monthly}")
            push!(cases, ("$(name)_$(foreign_token_name(t))", v, t, nothing))
        end
    end
    outer = Any[
        ("fx_o_tseries", Int16[1, 2], nothing, "TSeries"),
        ("fx_o_tseries_empty", Int16[], nothing, "TSeries"),
        ("fx_o_tseries_full", Int16[1, 2], nothing, "TSeries{Monthly, Int16, Vector{Int16}}"),
        ("fx_o_tseries_short", Int16[1, 2], nothing, "TSeries{Monthly, Int16}"),
        ("fx_o_tseries_mismatch_type", Int8[0, 1], nothing, "TSeries{Monthly, Int16}"),
        ("fx_o_tseries_mismatch_axis", Int8[0, 1], nothing, "TSeries{Quarterly{3}, Int8}"),
        ("fx_o_tseries_no_space", Int16[1, 2], nothing, "TSeries{Monthly,Int16}"),
        ("fx_o_tseries_leading_space", Int16[1, 2], nothing, " TSeries"),
        ("fx_o_vector_nonempty", Float64[1.0], nothing, "Vector"),
        ("fx_o_vector_complex_empty", ComplexF64[], nothing, "Vector"),
        ("fx_o_vector_mit_empty", MIT{Monthly}[], nothing, "Vector"),
        ("fx_o_float64", Int16[1, 2], nothing, "Float64"),
        ("fx_o_unknown", Int16[1, 2], nothing, "NoSuchMarker"),
        ("fx_o_empty_string", Int16[1, 2], "Int64", ""),
        ("fx_o_eltype_empty_string", Int16[1, 2], "", nothing),
        ("fx_o_identity_dated_daily_axis", MIT{Yearly{12}}[MIT{Yearly{12}}(2024)], nothing, "TSeries"),
    ]
    append!(cases, outer)
    return cases
end
foreign_axis(values) = eltype(values) <: MIT{Yearly{12}} && length(values) == 1 && Int(values[1]) == 2024 ? (C.freq_daily, 739191, MIT{Daily}(Date(2024, 11, 1))) : (C.freq_monthly, Int(foreign_anchor), foreign_anchor)
function store_foreign_case!(db, name, values, marker, outer)
    element = DE.I._eltypefreq(eltype(values))
    payload = collect(reinterpret(UInt8, values))
    freq, first, _ = foreign_axis(values)
    axis = Ref{C.axis_id_t}()
    id = Ref{C.obj_id_t}()
    @test C.de_axis_range(db, length(values), freq, first, axis) == 0
    GC.@preserve payload begin
        ptr = isempty(payload) ? C_NULL : pointer(payload)
        @test C.de_store_tseries(db, DE.root_id, name, C.type_tseries, element.eltype,
            element.elfreq, axis[], length(payload), ptr, id) == 0
    end
    marker === nothing || DE.set_attribute(db, id[], "jeltype", marker)
    outer === nothing || DE.set_attribute(db, id[], "jtype", outer)
    return id[]
end
foreign_loaded_values(x) = x isa TSeries ? x.values : x
foreign_empty_dated(x) = (v = foreign_loaded_values(x); v isa AbstractVector && isempty(v) && eltype(v) <: Union{MIT,Duration})
# Compare a loaded value with a stored object: by loading it, or by raw metadata
# when the value is an empty MIT/Duration vector the pinned loader cannot rebuild.
function foreign_same_stored(db, id, loaded)
    if foreign_empty_dated(loaded)
        metadata, bytes, attrs = raw_series(db, id)
        element = DE.I._eltypefreq(eltype(foreign_loaded_values(loaded)))
        return metadata[3] == Int(element.eltype) && metadata[4] == Int(element.elfreq) &&
            metadata[6] == 0 && isempty(bytes) && attrs == Dict("jeltype" => string(eltype(foreign_loaded_values(loaded))))
    end
    return foreign_same_load(loaded, DE.load_tseries(db, id))
end
function foreign_same_load(a, b)
    va, vb = foreign_loaded_values(a), foreign_loaded_values(b)
    va isa AbstractVector && vb isa AbstractVector || return false
    eltype(va) == eltype(vb) && length(va) == length(vb) || return false
    isempty(va) && return true
    return collect(reinterpret(UInt8, va)) == collect(reinterpret(UInt8, vb))
end
# sibling: "_julia" in the reference fixture (Julia's own loaded value), "_interpreted"
# in the installed-wheel output (Python's explicit interpretation of the preserved object).
function verify_foreign_case(db, name, values, marker, outer; generate::Bool, sibling::String="_julia")
    id = DE.find_object(db, DE.root_id, name)
    metadata, bytes, attrs = raw_series(db, id)
    element = DE.I._eltypefreq(eltype(values))
    freq, first, anchor = foreign_axis(values)
    @test metadata == (2, 12, Int(element.eltype), Int(element.elfreq), 1, length(values), Int(freq), first, length(bytes))
    @test bytes == collect(reinterpret(UInt8, values))
    expected_attrs = Dict{String,String}()
    marker === nothing || (expected_attrs["jeltype"] = marker)
    outer === nothing || (expected_attrs["jtype"] = outer)
    @test attrs == expected_attrs
    loaded = nothing
    failure = nothing
    try
        loaded = DE.load_tseries(db, id)
    catch e
        failure = string(nameof(typeof(e)))
    end
    if generate
        if failure === nothing
            stored = loaded isa TSeries ? loaded : TSeries(anchor, loaded)
            DE.store_tseries(db, DE.root_id, "$(name)_julia", stored)
        else
            DE.store_scalar(db, DE.root_id, "$(name)_error", failure)
        end
    end
    sibling_id = DE.find_object(db, DE.root_id, "$(name)$(sibling)", false)
    error_scalar = DE.find_object(db, DE.root_id, "$(name)_error", false)
    if failure === nothing
        @test sibling_id !== missing && error_scalar === missing
        sibling_id === missing && return
        @test foreign_same_stored(db, sibling_id, loaded)
        if loaded isa TSeries && !foreign_empty_dated(loaded)
            @test firstdate(loaded) == firstdate(DE.load_tseries(db, sibling_id))
        end
    else
        @test sibling_id === missing && error_scalar !== missing
        error_scalar === missing && return
        @test DE.load_scalar(db, error_scalar) == failure
    end
end

if action == "generate-series-elements"
    ispath(filename) && error("Use a fresh fixture path")
    DE.opendaec(filename; readonly=false) do db
        for (name,first,values) in series_element_cases()
            DE.store_tseries(db,name,TSeries(first,values))
        end
        for T in series_element_types
            axis = Ref{C.axis_id_t}()
            DE.I._check(C.de_axis_range(db,0,C.freq_monthly,Int(2024M11),axis))
            id = Ref{C.obj_id_t}()
            DE.I._check(C.de_store_tseries(db,0,"es_$(T)_unmarked",C.type_tseries,
                DE.I._to_de_scalar_type(T),C.freq_none,axis[],0,C_NULL,id))
        end
        for T in (Int128,UInt128,ComplexF16)
            DE.store_tseries(db,"unsupported_$(T)",TSeries(2024M11,T[1,2]))
            DE.store_tseries(db,"unsupported_$(T)_empty",TSeries(2024M11,T[]))
        end
        for (name,values,token) in (("bad_bool",Int8[2],"Bool"),
                ("foreign_bool",Int64[1],"Bool"),("unknown_marker",Float64[],"Unknown"),
                ("wrong_marker",Float64[],"Int8"))
            id = DE.store_tseries(db,name,TSeries(2024M11,values))
            DE.set_attribute(db,id,"jeltype",token)
        end
    end
elseif action == "generate"
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
elseif action == "generate-quarterly"
    ispath(filename) && error("Output already exists; use a fresh fixture path.")
    DE.opendaec(filename; write=true) do db
        for anchor in 1:3
            F = Quarterly{anchor}
            for (suffix, code, values) in quarterly_cases
                DE.store_tseries(db, DE.root_id, "q$(anchor)_$(suffix)", TSeries(MIT{F}(code), copy(values)))
            end
            # No reconstruction attribute: the encoding used for Python writes.
            axis = Ref{C.axis_id_t}()
            id = Ref{C.obj_id_t}()
            @assert C.de_axis_range(db, 0, C.frequency_t(64+anchor), 8096, axis) == 0
            @assert C.de_store_tseries(db, DE.root_id, "q$(anchor)_native_empty",
                C.type_tseries, C.type_float, C.freq_none, axis[], 0, C_NULL, id) == 0
        end
    end
elseif action == "generate-annual"
    ispath(filename) && error("Output already exists; use a fresh fixture path.")
    DE.opendaec(filename; write=true) do db
        for anchor in 1:12
            F = Yearly{anchor}
            for (suffix, code, values) in annual_cases
                DE.store_tseries(db, DE.root_id, "y$(anchor)_$(suffix)", TSeries(MIT{F}(code), copy(values)))
            end
            for (suffix, code, _) in annual_native_empties
                axis = Ref{C.axis_id_t}()
                id = Ref{C.obj_id_t}()
                @assert C.de_axis_range(db, 0, C.frequency_t(256+anchor), code, axis) == 0
                @assert C.de_store_tseries(db, DE.root_id, "y$(anchor)_$(suffix)",
                    C.type_tseries, C.type_float, C.freq_none, axis[], 0, C_NULL, id) == 0
            end
        end
    end
elseif action == "generate-halfyearly"
    ispath(filename) && error("Output already exists; use a fresh fixture path.")
    DE.opendaec(filename; write=true) do db
        for anchor in 1:6
            F = HalfYearly{anchor}
            for (suffix, code, values) in [halfyearly_cases; halfyearly_below_minimum]
                DE.store_tseries(db, DE.root_id, "h$(anchor)_$(suffix)", TSeries(MIT{F}(code), copy(values)))
            end
            for (suffix, code, _) in halfyearly_native_empties
                axis = Ref{C.axis_id_t}()
                id = Ref{C.obj_id_t}()
                @assert C.de_axis_range(db, 0, C.frequency_t(128+anchor), code, axis) == 0
                @assert C.de_store_tseries(db, DE.root_id, "h$(anchor)_$(suffix)",
                    C.type_tseries, C.type_float, C.freq_none, axis[], 0, C_NULL, id) == 0
            end
        end
    end
elseif action == "generate-int64"
    ispath(filename) && error("Output already exists; use a fresh fixture path.")
    DE.opendaec(filename; write=true) do db
        for (suffix, value) in int64_cases
            DE.store_scalar(db, DE.root_id, "int_$(suffix)", value)
        end
        for (suffix, value) in int64_native_cases
            id = Ref{C.obj_id_t}()
            box = Ref{Int64}(value)
            @assert C.de_store_scalar(db, DE.root_id, "int_$(suffix)", C.type_signed,
                C.freq_none, 8, box, id) == 0
        end
        for (suffix, value) in int64_controls
            DE.store_scalar(db, DE.root_id, "ctl_$(suffix)", value)
        end
    end
elseif action == "generate-strings"
    ispath(filename) && error("Output already exists; use a fresh fixture path.")
    DE.opendaec(filename; write=true) do db
        for (suffix, value) in string_cases
            DE.store_scalar(db, DE.root_id, "str_$(suffix)", value)
        end
        for (suffix, value) in string_controls
            DE.store_scalar(db, DE.root_id, "ctl_str_$(suffix)", value)
        end
        DE.store_scalar(db, DE.root_id, "ctl_symbol", :hello)
        for (suffix, bytes, freq) in string_natives
            id = Ref{C.obj_id_t}()
            ptr = isempty(bytes) ? C_NULL : pointer(bytes)
            GC.@preserve bytes begin
                @assert C.de_store_scalar(db, DE.root_id, "native_str_$(suffix)", C.type_string,
                    freq, length(bytes), ptr, id) == 0
            end
        end
    end
elseif action == "generate-dates"
    ispath(filename) && error("Output already exists; use a fresh fixture path.")
    DE.opendaec(filename; write=true) do db
        for (label, F, code, ppy, minimum) in date_families
            for (suffix, value) in date_codes(ppy, minimum)
                DE.store_scalar(db, DE.root_id, "mit_$(label)_$(suffix)", MIT{F}(value))
            end
            if ppy > 1
                # Stored intact by Julia; its loader decodes another date and warns.
                DE.store_scalar(db, DE.root_id, "ctl_mit_$(label)_below_minimum", MIT{F}(minimum - 1))
            end
            for (suffix, value) in duration_values
                DE.store_scalar(db, DE.root_id, "dur_$(label)_$(suffix)", Duration{F}(value))
            end
        end
        for (suffix, value) in date_controls
            DE.store_scalar(db, DE.root_id, "ctl_$(suffix)", value)
        end
        for (suffix, kind, freq, nbytes) in date_natives
            id = Ref{C.obj_id_t}()
            bytes = UInt8[reinterpret(UInt8, [Int64(24288)]); zeros(UInt8, 8)][1:nbytes]
            GC.@preserve bytes begin
                @assert C.de_store_scalar(db, DE.root_id, "native_$(suffix)", kind, freq,
                    nbytes, pointer(bytes), id) == 0
            end
        end
    end
elseif action == "generate-calendar"
    ispath(filename) && error("Output already exists; use a fresh fixture path.")
    DE.opendaec(filename; write=true) do db
        for (label, F, code, ed) in calendar_families
            for (suffix, value) in calendar_codes(F, code, ed)
                DE.store_scalar(db, DE.root_id, "mit_$(label)_$(suffix)", MIT{F}(value))
            end
            for (suffix, value) in duration_values
                DE.store_scalar(db, DE.root_id, "dur_$(label)_$(suffix)", Duration{F}(value))
            end
        end
        for (suffix, value) in unit_values
            DE.store_scalar(db, DE.root_id, "mit_u_$(suffix)", MIT{Unit}(value))
            DE.store_scalar(db, DE.root_id, "dur_u_$(suffix)", Duration{Unit}(value))
        end
        for (label, F, value) in calendar_below_window
            @test_logs (:warn, r"MIT codes differ") DE.store_scalar(db, DE.root_id, "ctl_mit_$(label)_below_window", MIT{F}(value))
        end
        # Non-canonical anchor: stored as code 17 and reloaded as Weekly{1}.
        @test_logs (:warn, r"MIT codes differ") DE.store_scalar(db, DE.root_id, "ctl_weekly8_mit", MIT{Weekly{8}}(105557))
        for (suffix, kind, freq, nbytes, value) in calendar_natives
            id = Ref{C.obj_id_t}()
            bytes = UInt8[reinterpret(UInt8, [Int64(value)]); zeros(UInt8, 8)][1:nbytes]
            GC.@preserve bytes begin
                @assert C.de_store_scalar(db, DE.root_id, "native_$(suffix)", kind, freq,
                    nbytes, pointer(bytes), id) == 0
            end
        end
    end
elseif action == "generate-widths"
    ispath(filename) && error("Output already exists; use a fresh fixture path.")
    DE.opendaec(filename; write=true) do db
        for (name, value) in width_cases
            DE.store_scalar(db, DE.root_id, name, value)
        end
        for (suffix, value) in width_native_cases
            id = Ref{C.obj_id_t}()
            box = Ref(value)
            @assert C.de_store_scalar(db, DE.root_id, "wn_$(suffix)",
                DE.I._to_de_scalar_type(value), C.freq_none, sizeof(value), box, id) == 0
        end
        for (suffix, value) in width_controls
            DE.store_scalar(db, DE.root_id, "wctl_$(suffix)", value)
        end
        # Marker experiment (not Julia's own behavior): an Int8 byte with jtype="Bool"
        # reloads as `true` in Julia; Python rejects every scalar reconstruction attribute.
        id = Ref{C.obj_id_t}()
        box = Ref(Int8(1))
        @assert C.de_store_scalar(db, DE.root_id, "wctl_marker_true", C.type_signed,
            C.freq_none, 1, box, id) == 0
        DE.set_attribute(db, id[], "jtype", "Bool")
    end
elseif action == "generate-fileops"
    ispath(filename) && error("Output already exists; use a fresh fixture path.")
    # Populate, truncate (ids restart at 1), repopulate, delete a scalar, a series
    # sharing its axis with `s2`, an attribute-carrying Rational and a nested
    # catalog; then reopen with overwrite=true and replace a scalar by a string
    # and a scalar by a series. `keep` stays for Python's recursive-delete tests.
    let db = DE.opendaec(filename; write=true)
        DE.store_scalar(db, "gone_before_truncate", 1)
        DE.new_catalog(db, "gonecat")
        DE.store_scalar(db, "/gonecat/x", 2)
        DE.truncatedaec(db)
        @assert isempty(db)
        DE.new_catalog(db, "keep")
        DE.store_scalar(db, "/keep/z", 4)
        DE.new_catalog(db, "/keep/inner")
        DE.store_scalar(db, "/keep/inner/w", 5)
        DE.store_scalar(db, "a", 1)
        DE.store_scalar(db, "b", "text")
        DE.store_tseries(db, "s", fileops_series)
        DE.store_scalar(db, "r", 1 // 2)
        DE.new_catalog(db, "deleted_catalog")
        DE.new_catalog(db, "/deleted_catalog/sub")
        DE.store_scalar(db, "/deleted_catalog/sub/y", 3.0)
        DE.store_scalar(db, "/deleted_catalog/x", 2.0)
        DE.store_scalar(db, "overwritten_series", 1)
        DE.store_tseries(db, "s2", fileops_series)
        for name in ("b", "s", "r", "deleted_catalog")
            DE.delete_object(db, name)
        end
        DE.closedaec!(db)
    end
    DE.opendaec(filename; write=true, overwrite=true) do db
        DE.store_scalar(db, "a", "replaced")
        DE.store_tseries(db, "overwritten_series", fileops_series)
    end
elseif action == "generate-calendar-series"
    ispath(filename) && error("Output already exists; use a fresh fixture path.")
    DE.opendaec(filename; write=true) do db
        for (label, F, code, ed) in calendar_families
            write_calendar_series!(db, F, label, code, ed)
        end
        # Noncanonical Julia anchors collapse to canonical codes on write; native
        # axis codes Julia never writes; an empty Float32 series marker control.
        @test_logs (:warn, r"MIT codes differ") DE.store_tseries(db, DE.root_id, "ctl_weekly8_series", TSeries(MIT{Weekly{8}}(105557), [1.25]))
        for (suffix, freq) in [("weekly16", 16), ("weekly24", 24), ("freq14", 14)]
            store_native_series!(db, "native_axis_$(suffix)", freq, 105557, [1.25])
        end
        DE.store_tseries(db, DE.root_id, "ctl_empty_float32_daily", TSeries(daily("2024-01-15"), Float32[]))
    end
elseif action == "generate-represented-elements"
    ispath(filename) && error("Output already exists; use a fresh fixture path.")
    DE.opendaec(filename; write=true) do db
        for (name, first, values) in represented_cases()
            DE.store_tseries(db, DE.root_id, name, TSeries(first, values))
        end
        for name in represented_rewrites
            DE.store_tseries(db, DE.root_id, "$(name)_rewrite", DE.load_tseries(db, DE.find_object(db, DE.root_id, name)))
        end
        for (name, kind, elfreq, payload, len, marker, _) in represented_controls()
            store_native_element_series!(db, name, kind, elfreq, payload, len, marker)
        end
        for (name, kind, payload) in represented_bool_cases
            store_native_element_series!(db, name, kind, 0, payload, 2, "Bool")
        end
        for (name, _) in represented_bool_rewrites
            DE.store_tseries(db, DE.root_id, "$(name)_rewrite", DE.load_tseries(db, DE.find_object(db, DE.root_id, name)))
        end
        for (name, _, _) in represented_bool_cases
            DE.store_tseries(db, DE.root_id, "$(name)_converted", DE.load_tseries(db, DE.find_object(db, DE.root_id, name)))
        end
        # A date element type without an element frequency is refused by the native
        # library before any object is created (DE_BAD_ELTYPE_DATE).
        axis = Ref{C.axis_id_t}()
        id = Ref{C.obj_id_t}()
        @test C.de_axis_range(db, 2, C.freq_monthly, Int(represented_anchor), axis) == 0
        payload = collect(reinterpret(UInt8, Int64[1, 2]))
        rc = GC.@preserve payload C.de_store_tseries(db, DE.root_id, "ctl_date_elfreq_none",
            C.type_tseries, C.type_date, C.freq_none, axis[], length(payload), pointer(payload), id)
        @test rc == Int(C.DE_BAD_ELTYPE_DATE)
        C.de_clear_error()
    end
elseif action == "generate-foreign-markers"
    ispath(filename) && error("Output already exists; use a fresh fixture path.")
    DE.opendaec(filename; write=true) do db
        for (name, values, marker, outer) in foreign_fixture_cases()
            store_foreign_case!(db, name, values, marker, outer)
        end
    end
elseif !(action in ("verify", "verify-empty", "verify-scalars", "verify-quarterly", "verify-annual", "verify-halfyearly", "verify-int64", "verify-strings", "verify-dates", "verify-calendar", "verify-widths", "verify-fileops", "verify-calendar-series", "verify-series-elements", "generate-series-elements", "verify-represented-elements", "verify-foreign-markers", "generate-arrays-unit", "verify-arrays-unit", "generate-matrices-text", "verify-matrices-text", "generate-tensors", "verify-tensors", "generate-catalogs", "verify-catalogs", "generate-workspace", "verify-workspace", "generate-text-arrays", "verify-text-arrays", "generate-represented-mvtseries", "verify-represented-mvtseries", "generate-scalar-markers", "verify-scalar-markers", "verify-wheel"))
    error("Unknown fixture action.")
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

if action in ("generate-quarterly", "verify-quarterly", "verify-wheel")
    @testset "DataEcon quarterly interchange" begin
        reference_fixture = action != "verify-wheel"
        cases = copy(quarterly_cases)
        reference_fixture && push!(cases, ("native_empty", 8096, Float64[]))
        DE.opendaec(filename) do db
            for anchor in 1:3, (suffix, code, values) in cases
                F = Quarterly{anchor}
                id = DE.find_object(db, DE.root_id, "q$(anchor)_$(suffix)")
                arr = Ref{C.tseries_t}()
                @test C.de_load_tseries(db, id, arr) == 0
                ts = arr[]
                @test Int.((ts.object.obj_class, ts.object.obj_type, ts.eltype, ts.elfreq,
                    ts.axis.ax_type, ts.axis.length, ts.axis.frequency, ts.axis.first,
                    ts.nbytes)) == (2,12,4,0,1,length(values),64+anchor,code,8length(values))
                @test (ts.value == C_NULL) == isempty(values)
                attrs = Dict(string(k)=>string(v) for (k,v) in DE.get_all_attributes(db, id))
                value = DE.load_tseries(db, id)
                if reference_fixture && isempty(values) && suffix != "native_empty"
                    @test attrs == Dict("jeltype"=>"Float64")
                    @test value isa Vector{Float64}
                    @test isempty(value)
                else
                    @test isempty(attrs)
                    @test value isa TSeries
                    @test frequencyof(value) == F
                    @test eltype(value) == Float64
                    @test Int(firstdate(value)) == code
                    @test Int(lastdate(value)) == code + length(values) - 1
                    @test value.values == values
                end
            end
        end
    end
end

if action in ("generate-annual", "verify-annual", "verify-wheel")
    @testset "DataEcon annual interchange" begin
        reference_fixture = action != "verify-wheel"
        cases = copy(annual_cases)
        reference_fixture && append!(cases, annual_native_empties)
        DE.opendaec(filename) do db
            for anchor in 1:12, (suffix, code, values) in cases
                F = Yearly{anchor}
                id = DE.find_object(db, DE.root_id, "y$(anchor)_$(suffix)")
                arr = Ref{C.tseries_t}()
                @test C.de_load_tseries(db, id, arr) == 0
                ts = arr[]
                @test Int.((ts.object.obj_class, ts.object.obj_type, ts.eltype, ts.elfreq,
                    ts.axis.ax_type, ts.axis.length, ts.axis.frequency, ts.axis.first,
                    ts.nbytes)) == (2,12,4,0,1,length(values),256+anchor,code,8length(values))
                @test (ts.value == C_NULL) == isempty(values)
                attrs = Dict(string(k)=>string(v) for (k,v) in DE.get_all_attributes(db, id))
                value = DE.load_tseries(db, id)
                if reference_fixture && isempty(values) && !startswith(suffix, "native_")
                    @test attrs == Dict("jeltype"=>"Float64")
                    @test value isa Vector{Float64}
                    @test isempty(value)
                else
                    @test isempty(attrs)
                    @test value isa TSeries
                    @test frequencyof(value) == F
                    @test eltype(value) == Float64
                    @test Int(firstdate(value)) == code
                    @test Int(lastdate(value)) == code + length(values) - 1
                    @test value.values == values
                end
            end
        end
    end
end

if action in ("generate-halfyearly", "verify-halfyearly", "verify-wheel")
    @testset "DataEcon half-yearly interchange" begin
        reference_fixture = action != "verify-wheel"
        cases = copy(halfyearly_cases)
        reference_fixture && append!(cases, [halfyearly_below_minimum; halfyearly_native_empties])
        DE.opendaec(filename) do db
            for anchor in 1:6, (suffix, code, values) in cases
                F = HalfYearly{anchor}
                id = DE.find_object(db, DE.root_id, "h$(anchor)_$(suffix)")
                arr = Ref{C.tseries_t}()
                @test C.de_load_tseries(db, id, arr) == 0
                ts = arr[]
                @test Int.((ts.object.obj_class, ts.object.obj_type, ts.eltype, ts.elfreq,
                    ts.axis.ax_type, ts.axis.length, ts.axis.frequency, ts.axis.first,
                    ts.nbytes)) == (2,12,4,0,1,length(values),128+anchor,code,8length(values))
                @test (ts.value == C_NULL) == isempty(values)
                attrs = Dict(string(k)=>string(v) for (k,v) in DE.get_all_attributes(db, id))
                if suffix == "below_minimum"
                    # Stored intact, but the pinned loader decodes another date and
                    # warns. Python rejects this stored code instead of misdating it.
                    value = @test_logs (:warn, r"MIT codes differ") DE.load_tseries(db, id)
                    @test isempty(attrs)
                    @test value isa TSeries
                    @test frequencyof(value) == F
                    @test Int(firstdate(value)) != code
                    @test value.values == values
                elseif reference_fixture && isempty(values) && !startswith(suffix, "native_")
                    value = DE.load_tseries(db, id)
                    @test attrs == Dict("jeltype"=>"Float64")
                    @test value isa Vector{Float64}
                    @test isempty(value)
                else
                    value = DE.load_tseries(db, id)
                    @test isempty(attrs)
                    @test value isa TSeries
                    @test frequencyof(value) == F
                    @test eltype(value) == Float64
                    @test Int(firstdate(value)) == code
                    @test Int(lastdate(value)) == code + length(values) - 1
                    @test value.values == values
                end
            end
        end
    end
end

if action in ("generate-int64", "verify-int64", "verify-wheel")
    @testset "DataEcon Int64 scalar interchange" begin
        reference_fixture = action != "verify-wheel"
        cases = copy(int64_cases)
        reference_fixture && append!(cases, int64_native_cases)
        DE.opendaec(filename) do db
            for (suffix, expected_int) in cases
                id = DE.find_object(db, DE.root_id, "int_$(suffix)")
                scal = Ref{C.scalar_t}()
                @test C.de_load_scalar(db, id, scal) == 0
                v = scal[]
                @test Int.((v.object.obj_class, v.object.obj_type, v.frequency, v.nbytes)) == (1,1,0,8)
                @test v.value != C_NULL
                @test isempty(DE.get_all_attributes(db, id))
                value = DE.load_scalar(db, id)
                @test value isa Int64
                @test value === expected_int
            end
            if reference_fixture
                expected_controls = [("bool_true", Int8, (1,1,0,1)), ("int32", Int32, (1,1,0,4)),
                    ("int128", Int128, (1,1,0,16)), ("uint64", UInt64, (1,2,0,8)),
                    ("duration_monthly", Duration{Monthly}, (1,1,32,8)),
                    ("mit_monthly", MIT{Monthly}, (1,3,32,8)),
                    ("rational", Rational{Int64}, (1,4,0,8)), ("string", String, (1,6,0,2))]
                for (suffix, T, metadata) in expected_controls
                    id = DE.find_object(db, DE.root_id, "ctl_$(suffix)")
                    scal = Ref{C.scalar_t}()
                    @test C.de_load_scalar(db, id, scal) == 0
                    v = scal[]
                    @test Int.((v.object.obj_class, v.object.obj_type, v.frequency, v.nbytes)) == metadata
                    # Julia reloads a stored Bool as Int8; Python rejects these controls.
                    @test DE.load_scalar(db, id) isa T
                end
            end
        end
    end
end

if action in ("generate-strings", "verify-strings", "verify-wheel")
    @testset "DataEcon string scalar interchange" begin
        reference_fixture = action != "verify-wheel"
        DE.opendaec(filename) do db
            for (suffix, expected_string) in string_cases
                id = DE.find_object(db, DE.root_id, "str_$(suffix)")
                scal = Ref{C.scalar_t}()
                @test C.de_load_scalar(db, id, scal) == 0
                v = scal[]
                @test Int.((v.object.obj_class, v.object.obj_type, v.frequency, v.nbytes)) == (1,6,0,sizeof(expected_string)+1)
                @test v.value != C_NULL
                @test unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(v.value), Int(v.nbytes); own=false) == [codeunits(expected_string); 0x00]
                @test isempty(DE.get_all_attributes(db, id))
                value = DE.load_scalar(db, id)
                @test value isa String
                @test value == expected_string
            end
            if reference_fixture
                # Julia's loader stops at the first NUL and does not validate UTF-8.
                @test DE.load_scalar(db, DE.find_object(db, DE.root_id, "ctl_str_embedded_nul")) == "a"
                @test codeunits(DE.load_scalar(db, DE.find_object(db, DE.root_id, "ctl_str_invalid_utf8"))) == UInt8[0x66, 0xff, 0x6f]
                symbol_id = DE.find_object(db, DE.root_id, "ctl_symbol")
                @test DE.get_all_attributes(db, symbol_id) == Dict("jtype" => "Symbol")
                @test DE.load_scalar(db, symbol_id) === :hello
                @test DE.load_scalar(db, DE.find_object(db, DE.root_id, "native_str_with_frequency")) == "hi"
                @test DE.load_scalar(db, DE.find_object(db, DE.root_id, "native_str_only_nul")) == ""
                @test_throws ArgumentError DE.load_scalar(db, DE.find_object(db, DE.root_id, "native_str_empty_payload"))
            end
        end
    end
end

if action in ("generate-dates", "verify-dates", "verify-wheel")
    @testset "DataEcon date and duration scalar interchange" begin
        reference_fixture = action != "verify-wheel"
        DE.opendaec(filename) do db
            for (label, F, code, ppy, minimum) in date_families
                for (suffix, expected_code) in date_codes(ppy, minimum)
                    id = DE.find_object(db, DE.root_id, "mit_$(label)_$(suffix)")
                    scal = Ref{C.scalar_t}()
                    @test C.de_load_scalar(db, id, scal) == 0
                    v = scal[]
                    @test Int.((v.object.obj_class, v.object.obj_type, v.frequency, v.nbytes)) == (1,3,code,8)
                    @test unsafe_load(Ptr{Int64}(v.value)) == expected_code
                    @test isempty(DE.get_all_attributes(db, id))
                    value = DE.load_scalar(db, id)
                    @test value isa MIT{F}
                    @test Int(value) == expected_code
                end
                for (suffix, expected_value) in duration_values
                    id = DE.find_object(db, DE.root_id, "dur_$(label)_$(suffix)")
                    scal = Ref{C.scalar_t}()
                    @test C.de_load_scalar(db, id, scal) == 0
                    v = scal[]
                    @test Int.((v.object.obj_class, v.object.obj_type, v.frequency, v.nbytes)) == (1,1,code,8)
                    @test unsafe_load(Ptr{Int64}(v.value)) == expected_value
                    @test isempty(DE.get_all_attributes(db, id))
                    value = DE.load_scalar(db, id)
                    @test value isa Duration{F}
                    @test Int(value) == expected_value
                end
                if reference_fixture && ppy > 1
                    id = DE.find_object(db, DE.root_id, "ctl_mit_$(label)_below_minimum")
                    value = @test_logs (:warn, r"MIT codes differ") DE.load_scalar(db, id)
                    @test value isa MIT{F}
                    @test Int(value) != minimum - 1
                end
            end
            if reference_fixture
                expected_controls = [("unit_mit", MIT{Unit}, (1,3,11,8)), ("unit_duration", Duration{Unit}, (1,1,11,8)),
                    ("daily_mit", MIT{Daily}, (1,3,12,8)), ("daily_duration", Duration{Daily}, (1,1,12,8)),
                    ("bdaily_mit", MIT{BDaily}, (1,3,13,8)), ("weekly7_mit", MIT{Weekly{7}}, (1,3,23,8)),
                    ("quarterly4_mit", MIT{Quarterly{1}}, (1,3,65,8))]
                for (suffix, T, metadata) in expected_controls
                    id = DE.find_object(db, DE.root_id, "ctl_$(suffix)")
                    scal = Ref{C.scalar_t}()
                    @test C.de_load_scalar(db, id, scal) == 0
                    v = scal[]
                    @test Int.((v.object.obj_class, v.object.obj_type, v.frequency, v.nbytes)) == metadata
                    @test DE.load_scalar(db, id) isa T
                end
                for (suffix, kind, freq, nbytes) in date_natives
                    id = DE.find_object(db, DE.root_id, "native_$(suffix)")
                    scal = Ref{C.scalar_t}()
                    @test C.de_load_scalar(db, id, scal) == 0
                    v = scal[]
                    @test Int.((v.object.obj_class, v.object.obj_type, v.frequency, v.nbytes)) == (1, Int(kind), Int(freq), nbytes)
                end
                # Julia cannot load these encodings either (no frequency, other widths, mixed bits).
                for suffix in ("date_no_freq", "date_four_bytes", "date_mixed_bits", "duration_four_bytes")
                    @test_throws Exception DE.load_scalar(db, DE.find_object(db, DE.root_id, "native_$(suffix)"))
                end
                # A bare quarterly code decodes in Julia as an invalid anchor; Python rejects it.
                @test DE.load_scalar(db, DE.find_object(db, DE.root_id, "native_date_bare_quarterly")) isa MIT{Quarterly{0}}
            end
        end
    end
end

if action in ("generate-calendar", "verify-calendar", "verify-wheel")
    @testset "DataEcon unit and calendar scalar interchange" begin
        reference_fixture = action != "verify-wheel"
        DE.opendaec(filename) do db
            function load_exact(name, T, metadata, expected)
                id = DE.find_object(db, DE.root_id, name)
                scal = Ref{C.scalar_t}()
                @test C.de_load_scalar(db, id, scal) == 0
                v = scal[]
                @test Int.((v.object.obj_class, v.object.obj_type, v.frequency, v.nbytes)) == metadata
                @test unsafe_load(Ptr{Int64}(v.value)) == expected
                @test isempty(DE.get_all_attributes(db, id))
                value = @test_logs DE.load_scalar(db, id)
                @test value isa T
                @test Int(value) == expected
            end
            for (label, F, code, ed) in calendar_families
                for (suffix, expected_code) in calendar_codes(F, code, ed)
                    load_exact("mit_$(label)_$(suffix)", MIT{F}, (1, 3, code, 8), expected_code)
                end
                for (suffix, expected_value) in duration_values
                    load_exact("dur_$(label)_$(suffix)", Duration{F}, (1, 1, code, 8), expected_value)
                end
            end
            for (suffix, expected_value) in unit_values
                load_exact("mit_u_$(suffix)", MIT{Unit}, (1, 3, 11, 8), expected_value)
                load_exact("dur_u_$(suffix)", Duration{Unit}, (1, 1, 11, 8), expected_value)
            end
            if reference_fixture
                for (label, F, value) in calendar_below_window
                    id = DE.find_object(db, DE.root_id, "ctl_mit_$(label)_below_window")
                    if F <: Weekly
                        # The weekly decoder is exact modulo 2^32 for this code.
                        @test Int(DE.load_scalar(db, id)) == value
                    else
                        loaded = @test_logs (:warn, r"MIT codes differ") DE.load_scalar(db, id)
                        @test loaded isa MIT{F} && Int(loaded) != value
                    end
                end
                @test DE.load_scalar(db, DE.find_object(db, DE.root_id, "ctl_weekly8_mit")) isa MIT{Weekly{1}}
                for (suffix, kind, freq, nbytes, value) in calendar_natives
                    id = DE.find_object(db, DE.root_id, "native_$(suffix)")
                    scal = Ref{C.scalar_t}()
                    @test C.de_load_scalar(db, id, scal) == 0
                    v = scal[]
                    @test Int.((v.object.obj_class, v.object.obj_type, v.frequency, v.nbytes)) == (1, Int(kind), Int(freq), nbytes)
                end
                # Julia's loader maps code 16 to an invalid Weekly{0} anchor and reads a
                # code above the window unchanged; Python rejects both.
                @test (@test_logs (:warn, r"MIT codes differ") DE.load_scalar(db, DE.find_object(db, DE.root_id, "native_date_weekly16"))) isa MIT{Weekly{0}}
                @test DE.load_scalar(db, DE.find_object(db, DE.root_id, "native_date_weekly24")) isa MIT{Weekly{8}}
                @test Int(DE.load_scalar(db, DE.find_object(db, DE.root_id, "native_date_daily_above_maximum"))) == 11979955
                @test Int(@test_logs (:warn, r"MIT codes differ") DE.load_scalar(db, DE.find_object(db, DE.root_id, "native_date_daily_int32_wrap"))) == 738900
                for suffix in ("date_freq14", "date_unit_four_bytes")
                    @test_throws Exception DE.load_scalar(db, DE.find_object(db, DE.root_id, "native_$(suffix)"))
                end
            end
        end
    end
end

if action in ("generate-widths", "verify-widths", "verify-wheel")
    @testset "DataEcon numeric width scalar interchange" begin
        reference_fixture = action != "verify-wheel"
        cases = copy(width_cases)
        reference_fixture && append!(cases, [("wn_$(s)", v) for (s, v) in width_native_cases])
        # Python also writes a built-in complex, stored as ComplexF64 like Julia's 8.0 + 3.0im.
        reference_fixture || push!(cases, ("w_pyc_plain", ComplexF64(8.0, 3.0)))
        DE.opendaec(filename) do db
            for (name, expected) in cases
                id = DE.find_object(db, DE.root_id, name)
                scal = Ref{C.scalar_t}()
                @test C.de_load_scalar(db, id, scal) == 0
                v = scal[]
                @test Int.((v.object.obj_class, v.object.obj_type, v.frequency, v.nbytes)) ==
                    (1, Int(DE.I._to_de_scalar_type(expected)), 0, sizeof(expected))
                @test v.value != C_NULL
                payload = copy(unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(v.value), Int(v.nbytes); own=false))
                @test bytes2hex(payload) == width_bits(expected)
                @test isempty(DE.get_all_attributes(db, id))
                value = DE.load_scalar(db, id)
                @test typeof(value) == typeof(expected)
                @test width_bits(value) == width_bits(expected)
            end
            if reference_fixture
                for (suffix, written) in width_controls
                    id = DE.find_object(db, DE.root_id, "wctl_$(suffix)")
                    scal = Ref{C.scalar_t}()
                    @test C.de_load_scalar(db, id, scal) == 0
                    v = scal[]
                    @test Int.((v.object.obj_class, v.object.obj_type, v.frequency, v.nbytes)) ==
                        (1, Int(DE.I._to_de_scalar_type(DE.I._to_de_scalar_val(written))), 0,
                            sizeof(DE.I._to_de_scalar_val(written)))
                    loaded = DE.load_scalar(db, id)
                    # Julia reloads a stored Bool as Int8; Complex{Int} comes back through its marker.
                    @test loaded isa width_expected_type(written)
                    @test loaded == written
                end
                @test DE.load_scalar(db, "wctl_marker_true") === true
                @test DE.get_attribute(db, "wctl_marker_true", "jtype") == "Bool"
            end
        end
    end
end

if action in ("generate-fileops", "verify-fileops", "verify-wheel")
    @testset "DataEcon file operations" begin
        if action != "verify-wheel"
            DE.opendaec(filename) do db
                @test !isempty(db)
                @test DE.catalog_size(db, DE.root_id) == 4
                @test sort(DE.list_catalog(db; quiet=true)) ==
                    ["/a", "/keep/inner/w", "/keep/z", "/overwritten_series", "/s2"]
                @test DE.find_object(db, DE.root_id, "keep") == 1   # ids restarted after truncation
                @test DE.load_scalar(db, "a") == "replaced"
                @test DE.load_tseries(db, "s2") == fileops_series
                @test DE.load_tseries(db, "overwritten_series") == fileops_series
                @test DE.load_scalar(db, "/keep/z") === 4
                @test DE.load_scalar(db, "/keep/inner/w") === 5
                for name in ("b", "s", "r", "deleted_catalog", "gone_before_truncate", "gonecat")
                    @test DE.find_object(db, DE.root_id, name, false) === missing
                end
                @test DE.find_fullpath(db, "/deleted_catalog/sub/y", false) === missing
                @test isempty(DE.get_all_attributes(db, "a"))
            end
        else
            DE.opendaec(filename) do db
                @test DE.find_object(db, DE.root_id, "fo_deleted", false) === missing
                @test DE.load_scalar(db, "fo_overwritten") == "two"
                @test DE.load_tseries(db, "fo_series_overwritten") == fileops_series
            end
            # The checker writes auxiliary outputs below the discovery directory so the
            # workflow's "*.daec" glob still selects exactly one primary file.
            sibling = joinpath(dirname(filename), "fileops",
                               replace(basename(filename), r"\.daec$" => "-fileops.daec"))
            @test isfile(sibling)
            @test basename(filename) == only(filter(endswith(".daec"), readdir(dirname(filename))))
            DE.opendaec(sibling) do db
                @test !isempty(db)
                @test sort(DE.list_catalog(db; quiet=true)) == ["/after_truncate", "/after_truncate_series"]
                @test DE.find_object(db, DE.root_id, "after_truncate") == 1
                @test DE.find_object(db, DE.root_id, "junk", false) === missing
                @test DE.load_scalar(db, "after_truncate") === 42
                @test DE.load_tseries(db, "after_truncate_series") == fileops_series
            end
        end
    end
end

if action in ("generate-calendar-series", "verify-calendar-series", "verify-wheel")
    @testset "DataEcon calendar series interchange" begin
        reference_fixture = action != "verify-wheel"
        DE.opendaec(filename) do db
            for (label, F, code, ed) in calendar_families
                verify_calendar_series(db, F, label, code, ed, reference_fixture)
            end
            if reference_fixture
                id = DE.find_object(db, DE.root_id, "ctl_weekly8_series")
                @test load_series_raw(db, id)[1][7] == 17
                @test DE.load_tseries(db, id) isa TSeries{Weekly{1},Float64}
                id = DE.find_object(db, DE.root_id, "native_axis_weekly16")
                @test (@test_logs (:warn, r"MIT codes differ") DE.load_tseries(db, id)) isa TSeries{Weekly{0},Float64}
                @test DE.load_tseries(db, DE.find_object(db, DE.root_id, "native_axis_weekly24")) isa TSeries{Weekly{8},Float64}
                @test_throws Exception DE.load_tseries(db, DE.find_object(db, DE.root_id, "native_axis_freq14"))
                id = DE.find_object(db, DE.root_id, "ctl_empty_float32_daily")
                @test load_series_raw(db, id)[3] == Dict("jeltype" => "Float32")
                @test DE.load_tseries(db, id) isa Vector{Float32}
            end
        end
    end
end

if action in ("generate-series-elements", "verify-series-elements", "verify-wheel")
    @testset "DataEcon numeric and Boolean series interchange" begin
        reference_fixture = action != "verify-wheel"
        DE.opendaec(filename) do db
            for (name,first,values) in series_element_cases()
                T = eltype(values)
                id = DE.find_object(db,0,name)
                ref = Ref{C.tseries_t}()
                @test C.de_load_tseries(db,id,ref) == 0
                a = ref[]
                metadata = Int.((a.object.obj_class,a.object.obj_type,a.eltype,a.elfreq,
                    a.axis.ax_type,a.axis.length,a.axis.frequency,a.axis.first,a.nbytes))
                bytes = a.nbytes == 0 ? UInt8[] : copy(unsafe_wrap(Vector{UInt8},Ptr{UInt8}(a.value),a.nbytes))
                @test metadata == (2,12,Int(DE.I._to_de_scalar_type(T)),0,1,length(values),
                    Int(DE.I._to_de_scalar_freq(frequencyof(first))),Int(first),sizeof(values))
                @test bytes == collect(reinterpret(UInt8,values))
                needs_marker = T == Bool || (isempty(values) && (reference_fixture || !(T in (Int64,UInt64,Float64,ComplexF64))))
                attrs = DE.get_all_attributes(db,id)
                @test attrs == (needs_marker ? Dict("jeltype"=>string(T)) : Dict{String,String}())
                loaded = DE.load_tseries(db,id)
                @test eltype(loaded) == T
                @test (loaded isa TSeries) == (!isempty(values) || !needs_marker)
                if loaded isa TSeries
                    @test firstdate(loaded) == first
                    @test loaded.values == values || isequal(loaded.values,values)
                    @test collect(reinterpret(UInt8,loaded.values)) == bytes
                else
                    @test isempty(loaded)
                end
            end
        end
    end
end

if action in ("generate-represented-elements", "verify-represented-elements", "verify-wheel")
    @testset "DataEcon represented series interchange" begin
        reference_fixture = action != "verify-wheel"
        DE.opendaec(filename) do db
            for (name, first, values) in represented_cases()
                verify_represented_case(db, name, first, values)
            end
            for name in represented_rewrites
                original = raw_series(db, DE.find_object(db, DE.root_id, name))
                @test raw_series(db, DE.find_object(db, DE.root_id, "$(name)_rewrite")) == original
            end
            for (name, kind, payload) in represented_bool_cases
                verify_represented_control(db, name, kind, 0, payload, 2, "Bool", Bool)
                verify_canonical_bool(db, "$(name)_converted", UInt8[0, 1])
            end
            if reference_fixture
                for (name, kind, elfreq, payload, len, marker, expect) in represented_controls()
                    verify_represented_control(db, name, kind, elfreq, payload, len, marker, expect)
                end
                for (name, expected) in represented_bool_rewrites
                    verify_canonical_bool(db, "$(name)_rewrite", expected)
                end
                @test DE.find_object(db, DE.root_id, "ctl_date_elfreq_none", false) === missing
            end
        end
    end
end

if action in ("generate-foreign-markers", "verify-foreign-markers", "verify-wheel")
    @testset "DataEcon foreign marker interchange" begin
        reference_fixture = action != "verify-wheel"
        DE.opendaec(filename; write=(action == "generate-foreign-markers")) do db
            # The reference fixture compares Julia's load with its own writer's sibling;
            # the installed-wheel output compares it with Python's explicit interpretation
            # of the preserved object, which carries no foreign marker.
            cases = reference_fixture ? foreign_fixture_cases() : foreign_shared_cases()
            for (name, values, marker, outer) in cases
                verify_foreign_case(db, name, values, marker, outer;
                    generate=(action == "generate-foreign-markers"),
                    sibling=(reference_fixture ? "_julia" : "_interpreted"))
            end
        end
    end
end

function array_unit_values(T)
    T <: Signed && return T[typemin(T), -1, 0, typemax(T)]
    T <: Unsigned && return T[0, 1, typemax(T)]
    T === Bool && return Bool[false, true, false]
    T === Float16 && return reinterpret(Float16, UInt16[0x8000, 0x0001, 0x3d00, 0x7e55, 0x7c00])
    T === Float32 && return reinterpret(Float32, UInt32[0x80000000, 1, 0x3fa00000, 0x7fc00055, 0x7f800000])
    T === Float64 && return reinterpret(Float64, UInt64[0x8000000000000000, 1, 0x3ff4000000000000, 0x7ff8000000000055, 0x7ff0000000000000])
    T === ComplexF32 && return ComplexF32[complex(-0.0f0, 1.25f0), complex(2.5f0, -3.0f0)]
    T === ComplexF64 && return ComplexF64[complex(-0.0, 1.25), complex(2.5, -3.0)]
    error("unsupported array/Unit fixture type")
end
const array_unit_types = [Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64,
    Float16, Float32, Float64, ComplexF32, ComplexF64, Bool]

if action == "generate-arrays-unit"
    DE.opendaec(filename; readonly=false) do db
        for T in array_unit_types
            DE.store_tseries(db, "array_$(T)", collect(array_unit_values(T)))
            DE.store_tseries(db, "array_$(T)_empty", T[])
            DE.store_tseries(db, "unit_$(T)", TSeries(MIT{Unit}(-2), collect(array_unit_values(T))))
            DE.store_tseries(db, "unit_$(T)_empty", TSeries(MIT{Unit}(-2), T[]))
        end
        DE.store_tseries(db, "range_int", 1:5)
        DE.store_tseries(db, "range_int_empty", 1:0)
        DE.store_tseries(db, "range_monthly", MIT{Monthly}(-3):MIT{Monthly}(1))
        DE.store_tseries(db, "range_unit", MIT{Unit}(-3):MIT{Unit}(1))
        DE.store_tseries(db, "range_unit_empty", MIT{Unit}(5):MIT{Unit}(4))
        for (label, F, code, _, minimum) in date_families
            DE.store_tseries(db, "range_all_$(label)", MIT{F}(0):MIT{F}(1))
            DE.store_tseries(db, "range_min_$(label)", MIT{F}(minimum):MIT{F}(minimum))
            DE.store_tseries(db, "range_max_$(label)", MIT{F}(typemax(Int32)):MIT{F}(typemax(Int32)))
        end
        for (label, F, code, _) in calendar_families
            minimum, maximum = calendar_windows[code]
            DE.store_tseries(db, "range_all_$(label)", MIT{F}(0):MIT{F}(1))
            DE.store_tseries(db, "range_min_$(label)", MIT{F}(minimum):MIT{F}(minimum))
            DE.store_tseries(db, "range_max_$(label)", MIT{F}(maximum):MIT{F}(maximum))
        end
        DE.store_tseries(db, "range_all_u", MIT{Unit}(0):MIT{Unit}(1))
        DE.store_tseries(db, "range_min_u", MIT{Unit}(typemin(Int64)):MIT{Unit}(typemin(Int64)))
        DE.store_tseries(db, "range_max_u", MIT{Unit}(typemax(Int64)):MIT{Unit}(typemax(Int64)))
        DE.store_tseries(db, "unit_min", TSeries(MIT{Unit}(typemin(Int64)), Float64[1]))
        DE.store_tseries(db, "unit_max", TSeries(MIT{Unit}(typemax(Int64)), Float64[1]))
        DE.store_tseries(db, "unit_max_span", TSeries(MIT{Unit}(typemax(Int64)-1), Float64[1, 2]))
        DE.store_tseries(db, "unit_mit_monthly",
            TSeries(MIT{Unit}(-2), MIT{Monthly}[MIT{Monthly}(-1), MIT{Monthly}(0)]))
        DE.store_tseries(db, "unit_int128",
            TSeries(MIT{Unit}(-2), Int128[typemin(Int128), typemax(Int128)]))
        DE.store_tseries(db, "unit_complexf16",
            TSeries(MIT{Unit}(-2), ComplexF16[complex(Float16(-0.0), Float16(1.25))]))
    end
end

if action in ("generate-arrays-unit", "verify-arrays-unit", "verify-wheel")
    @testset "DataEcon arrays and Unit series interchange" begin
        reference_fixture = action != "verify-wheel"
        DE.opendaec(filename) do db
            for T in array_unit_types, empty in (false, true)
                expected = empty ? T[] : collect(array_unit_values(T))
                for (prefix, objtype, axis, freq, first) in
                    (("array", 10, 0, 0, 0), ("unit", 12, 1, 11, -2))
                    name = "$(prefix)_$(T)$(empty ? "_empty" : "")"
                    id = DE.find_object(db, DE.root_id, name)
                    raw = Ref{C.tseries_t}()
                    @test C.de_load_tseries(db, id, raw) == 0
                    a = raw[]
                    @test (Int(a.object.obj_type), Int(a.axis.ax_type), Int(a.axis.length),
                        Int(a.axis.frequency), Int(a.axis.first), Int(a.nbytes)) ==
                        (objtype, axis, length(expected), freq, first, sizeof(expected))
                    attrs = Dict(string(k)=>string(v) for (k,v) in DE.get_all_attributes(db, id))
                    value = DE.load_tseries(db, id)
                    actual = value isa TSeries ? value.values : value
                    @test eltype(actual) === T
                    @test isequal(actual, expected)
                    if !empty
                        @test attrs == (T === Bool ? Dict("jeltype"=>"Bool") : Dict())
                    elseif reference_fixture
                        @test attrs == Dict("jeltype"=>string(T))
                    end
                end
            end
            for (name, expected, meta) in (
                ("range_int", 1:5, (11, 0, 5, 0, 0)),
                ("range_monthly", MIT{Monthly}(-3):MIT{Monthly}(1), (11, 1, 5, 32, -3)),
                ("range_unit", MIT{Unit}(-3):MIT{Unit}(1), (11, 1, 5, 11, -3)),
            )
                id = DE.find_object(db, DE.root_id, name)
                raw = Ref{C.tseries_t}()
                @test C.de_load_tseries(db, id, raw) == 0
                a = raw[]
                @test (Int(a.object.obj_type), Int(a.axis.ax_type), Int(a.axis.length),
                    Int(a.axis.frequency), Int(a.axis.first)) == meta
                @test DE.load_tseries(db, id) == expected
            end
            all_ranges = Any[("u", Unit, 11, typemin(Int64), typemax(Int64))]
            append!(all_ranges, [(label, F, code, minimum, typemax(Int32))
                                 for (label, F, code, _, minimum) in date_families])
            append!(all_ranges, [(label, F, code, calendar_windows[code]...)
                                 for (label, F, code, _) in calendar_families])
            for (label, F, code, minimum, maximum) in all_ranges,
                (part, first, len) in (("all", 0, 2), ("min", minimum, 1), ("max", maximum, 1))
                name = "range_$(part)_$(label)"
                id = DE.find_object(db, DE.root_id, name)
                raw = Ref{C.tseries_t}()
                @test C.de_load_tseries(db, id, raw) == 0
                a = raw[]
                @test (Int(a.object.obj_type), Int(a.axis.ax_type), Int(a.axis.length),
                    Int(a.axis.frequency), Int(a.axis.first), Int(a.nbytes)) ==
                    (11, 1, len, code, first, 0)
                value = DE.load_tseries(db, id)
                @test value == MIT{F}(first):MIT{F}(first + len - 1)
            end
            empty_id = DE.find_object(db, DE.root_id, "range_unit_empty")
            empty_raw = Ref{C.tseries_t}()
            @test C.de_load_tseries(db, empty_id, empty_raw) == 0
            @test (Int(empty_raw[].object.obj_type), Int(empty_raw[].axis.ax_type),
                Int(empty_raw[].axis.length), Int(empty_raw[].axis.frequency),
                Int(empty_raw[].axis.first), Int(empty_raw[].nbytes)) == (11, 1, 0, 11, 5, 0)
            @test DE.get_attribute(db, empty_id, "jeltype") == "MIT{Unit}"
            @test isempty(DE.load_tseries(db, empty_id))
            int_empty_id = DE.find_object(db, DE.root_id, "range_int_empty")
            int_empty_raw = Ref{C.tseries_t}()
            @test C.de_load_tseries(db, int_empty_id, int_empty_raw) == 0
            @test (Int(int_empty_raw[].object.obj_type), Int(int_empty_raw[].axis.ax_type),
                Int(int_empty_raw[].axis.length), Int(int_empty_raw[].axis.frequency),
                Int(int_empty_raw[].axis.first), Int(int_empty_raw[].nbytes)) == (11, 0, 0, 0, 0, 0)
            @test DE.get_attribute(db, int_empty_id, "jeltype") == "Int64"
            @test isempty(DE.load_tseries(db, int_empty_id))
            for (name, first, values) in (("unit_min", typemin(Int64), [1.0]),
                                         ("unit_max", typemax(Int64), [1.0]),
                                         ("unit_max_span", typemax(Int64)-1, [1.0, 2.0]))
                value = DE.load_tseries(db, name)
                @test value isa TSeries{Unit}
                @test Int(firstdate(value)) == first
                @test value.values == values
            end
            for (name, T, expected) in (
                ("unit_mit_monthly", MIT{Monthly}, MIT{Monthly}[MIT{Monthly}(-1), MIT{Monthly}(0)]),
                ("unit_int128", Int128, Int128[typemin(Int128), typemax(Int128)]),
                ("unit_complexf16", ComplexF16, ComplexF16[complex(Float16(-0.0), Float16(1.25))]),
            )
                value = DE.load_tseries(db, name)
                @test value isa TSeries{Unit}
                @test eltype(value) === T
                @test isequal(value.values, expected)
            end
        end
    end
end

#############################################################################
# Matrices, MVTSeries and text vectors

function matrix_text_values(T)
    T <: Signed && return T[typemin(T), -1, 0, typemax(T), 7, 9]
    T <: Unsigned && return T[0, 1, typemax(T), 7, 9, 11]
    T === Bool && return Bool[false, true, false, true, true, false]
    T === Float16 && return reinterpret(Float16, UInt16[0x8000, 0x0001, 0x3d00, 0x7e55, 0x7c00, 0x0002])
    T === Float32 && return reinterpret(Float32, UInt32[0x80000000, 1, 0x3fa00000, 0x7fc00055, 0x7f800000, 2])
    T === Float64 && return reinterpret(Float64, UInt64[0x8000000000000000, 1, 0x3ff4000000000000, 0x7ff8000000000055, 0x7ff0000000000000, 2])
    T === ComplexF32 && return ComplexF32[complex(-0.0f0, 1.25f0), complex(2.5f0, -3.0f0), complex(0.0f0, -0.0f0), complex(1.0f0, 2.0f0), complex(3.0f0, 4.0f0), complex(5.0f0, 6.0f0)]
    T === ComplexF64 && return ComplexF64[complex(-0.0, 1.25), complex(2.5, -3.0), complex(0.0, -0.0), complex(1.0, 2.0), complex(3.0, 4.0), complex(5.0, 6.0)]
    error("unsupported matrix fixture type")
end
const matrix_text_types = [Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64,
    Float16, Float32, Float64, ComplexF32, ComplexF64, Bool]
const matrix_text_strings = ["alpha", "", "z"]
const matrix_text_multibyte = ["\u00e9", "\U0001f642", "a\u00e9"]
const matrix_text_symbols = [:alpha, Symbol(""), :z]

# Julia's own writer sizes a packed string buffer by character count while the
# native packer sizes it in UTF-8 bytes, so multibyte text cannot be written
# through `store_tseries`. The fixture stores the correctly sized payload
# through the same C entry points the library itself uses.
function store_packed_text!(db, name, strings; marker=nothing)
    buffer = UInt8[]
    for item in strings
        append!(buffer, Vector{UInt8}(codeunits(String(item))))
        push!(buffer, 0x00)
    end
    ax = Ref{C.axis_id_t}()
    @test C.de_axis_plain(db, length(strings), ax) == 0
    id = Ref{C.obj_id_t}()
    GC.@preserve buffer begin
        ptr = isempty(buffer) ? C_NULL : pointer(buffer)
        @test C.de_store_tseries(db, DE.root_id, name, C.type_vector, C.type_string,
                                 C.freq_none, ax[], length(buffer), ptr, id) == 0
    end
    marker === nothing || DE.set_attribute(db, id[], "jeltype", marker)
    return id[]
end

const matrix_axis_families = [("u", Unit, 11, typemin(Int64), typemax(Int64)),
    ("d", Daily, 12, -11980259, 11979954), ("b", BDaily, 13, -8557114, 8557110),
    ("w7", Weekly{7}, 23, -1711422, 1711422), ("m", Monthly, 32, -393600, 2147483647),
    ("q1", Quarterly{1}, 65, -131200, 2147483647),
    ("h1", HalfYearly{1}, 129, -65600, 2147483647),
    ("y1", Yearly{1}, 257, typemin(Int32), typemax(Int32))]

if action == "generate-matrices-text"
    DE.opendaec(filename; readonly=false) do db
        for T in matrix_text_types
            DE.store_mvtseries(db, "matrix_$(T)", reshape(collect(matrix_text_values(T)), 2, 3))
            DE.store_mvtseries(db, "matrix_$(T)_empty", Matrix{T}(undef, 0, 0))
        end
        DE.store_mvtseries(db, "matrix_zero_rows", Matrix{Float64}(undef, 0, 3))
        DE.store_mvtseries(db, "matrix_zero_cols", Matrix{Float64}(undef, 3, 0))
        DE.store_mvtseries(db, "matrix_1x1", reshape(Float64[2.5], 1, 1))
        DE.store_mvtseries(db, "matrix_mit",
            reshape(MIT{Monthly}[MIT{Monthly}(-1), MIT{Monthly}(0), MIT{Monthly}(1),
                                 MIT{Monthly}(2)], 2, 2))
        DE.store_mvtseries(db, "matrix_duration",
            reshape(Duration{Monthly}[Duration{Monthly}(1), Duration{Monthly}(2),
                                      Duration{Monthly}(3), Duration{Monthly}(4)], 2, 2))
        DE.store_mvtseries(db, "matrix_int128",
            reshape(Int128[typemin(Int128), -1, 0, typemax(Int128)], 2, 2))
        DE.store_mvtseries(db, "matrix_uint128",
            reshape(UInt128[0, 1, 2, typemax(UInt128)], 2, 2))
        DE.store_mvtseries(db, "matrix_complexf16",
            reshape(ComplexF16[complex(Float16(-0.0), Float16(1.25)),
                               complex(Float16(2.0), Float16(3.0))], 1, 2))
        DE.store_mvtseries(db, "matrix_diagonal", Diagonal(Float64[3.0, 5.0]))
        DE.store_mvtseries(db, "matrix_symmetric", Symmetric(reshape(Float64[1, 2, 3, 4], 2, 2)))
        DE.store_mvtseries(db, "matrix_hermitian",
            Hermitian(reshape(ComplexF64[complex(1.0, 0.0), complex(2.0, -1.0),
                                         complex(2.0, 1.0), complex(4.0, 0.0)], 2, 2)))
        DE.store_mvtseries(db, "mvts_float64",
            MVTSeries(MIT{Monthly}(2024, 1), (:a, :b), reshape(Float64[1, 2, 3, 4, 5, 6], 3, 2)))
        DE.store_mvtseries(db, "mvts_int64",
            MVTSeries(MIT{Quarterly{3}}(2020, 1), (:x, :y), reshape(Int64[1, 2, 3, 4], 2, 2)))
        DE.store_mvtseries(db, "mvts_bool",
            MVTSeries(MIT{Monthly}(2024, 1), (:a, :b), reshape(Bool[true, false, false, true], 2, 2)))
        DE.store_mvtseries(db, "mvts_one_col",
            MVTSeries(MIT{Monthly}(2024, 1), (:only,), reshape(Float64[1, 2], 2, 1)))
        DE.store_mvtseries(db, "mvts_unicode",
            MVTSeries(MIT{Monthly}(2024, 1), (Symbol("a\u00e9"), Symbol("\U0001f642")),
                      reshape(Float64[1, 2, 3, 4], 2, 2)))
        DE.store_mvtseries(db, "mvts_empty_name",
            MVTSeries(MIT{Monthly}(2024, 1), (Symbol(""), :b), reshape(Float64[1, 2, 3, 4], 2, 2)))
        DE.store_mvtseries(db, "mvts_zero_rows",
            MVTSeries(MIT{Monthly}(2024, 1):MIT{Monthly}(2023, 12), (:a, :b),
                      Matrix{Float64}(undef, 0, 2)))
        for (label, F, code, lo, hi) in matrix_axis_families
            DE.store_mvtseries(db, "mvts_min_$(label)",
                MVTSeries(MIT{F}(lo), (:a,), reshape(Float64[1], 1, 1)))
            DE.store_mvtseries(db, "mvts_max_$(label)",
                MVTSeries(MIT{F}(hi), (:a,), reshape(Float64[1], 1, 1)))
        end
        DE.store_tseries(db, "text_ascii", matrix_text_strings)
        DE.store_tseries(db, "text_symbol", matrix_text_symbols)
        DE.store_tseries(db, "text_empty", String[])
        store_packed_text!(db, "text_multibyte", matrix_text_multibyte)
        store_packed_text!(db, "text_multibyte_symbol", matrix_text_multibyte; marker="Symbol")
        store_packed_text!(db, "text_one_empty_string", [""])
    end
end

if action in ("generate-matrices-text", "verify-matrices-text", "verify-wheel")
    @testset "DataEcon matrices, MVTSeries and text interchange" begin
        reference_fixture = action != "verify-wheel"
        DE.opendaec(filename) do db
            # The struct's value and names pointers are borrowed and valid
            # only until the next library call, so the names axis is copied
            # before the attribute lookup and the returned struct's pointers
            # are never dereferenced afterwards.
            function raw_matrix(name)
                id = DE.find_object(db, DE.root_id, name)
                ref = Ref{C.mvtseries_t}()
                @test C.de_load_mvtseries(db, id, ref) == 0
                a = ref[]
                names = a.axis2.ax_type == C.axis_names && a.axis2.names != C_NULL ?
                    Base.unsafe_string(a.axis2.names) : nothing
                attrs = Dict(string(k) => string(v) for (k, v) in DE.get_all_attributes(db, id))
                return id, a, attrs, names
            end
            for T in matrix_text_types
                expected = reshape(collect(matrix_text_values(T)), 2, 3)
                id, a, attrs, _ = raw_matrix("matrix_$(T)")
                @test (Int(a.object.obj_class), Int(a.object.obj_type), Int(a.axis1.ax_type),
                       Int(a.axis1.length), Int(a.axis2.ax_type), Int(a.axis2.length),
                       Int(a.nbytes)) == (3, 20, 0, 2, 0, 3, sizeof(expected))
                value = DE.load_mvtseries(db, id)
                @test value isa Matrix{T}
                @test isequal(value, expected)
                @test attrs == (T === Bool ? Dict("jeltype" => "Bool") : Dict())
                # Every empty two-dimensional object keeps its stored shape but
                # Julia's own loader returns a flat typed vector for it.
                id, a, attrs, _ = raw_matrix("matrix_$(T)_empty")
                @test (Int(a.axis1.length), Int(a.axis2.length), Int(a.nbytes)) == (0, 0, 0)
                loaded = DE.load_mvtseries(db, id)
                if haskey(attrs, "jeltype")
                    # Julia writes the element token on every empty object, and
                    # its marker path then returns a flat typed vector.
                    @test attrs["jeltype"] == string(T)
                    @test loaded == T[]
                else
                    # Python omits a token that only repeats the kind default,
                    # so the loader keeps the stored zero-by-zero shape.
                    @test T in (Int64, UInt64, Float64, ComplexF64)
                    @test loaded == Matrix{T}(undef, 0, 0)
                end
            end
            for (name, rows, columns) in (("matrix_zero_rows", 0, 3), ("matrix_zero_cols", 3, 0),
                                          ("matrix_1x1", 1, 1))
                id, a, _, _ = raw_matrix(name)
                @test (Int(a.axis1.length), Int(a.axis2.length)) == (rows, columns)
            end
            @test DE.load_mvtseries(db, DE.find_object(db, DE.root_id, "matrix_1x1")) ==
                reshape(Float64[2.5], 1, 1)
            for (name, expected, eltype_code, elfreq) in (
                ("matrix_mit", reshape(MIT{Monthly}[MIT{Monthly}(-1), MIT{Monthly}(0),
                                                    MIT{Monthly}(1), MIT{Monthly}(2)], 2, 2), 3, 32),
                ("matrix_duration", reshape(Duration{Monthly}[Duration{Monthly}(1),
                    Duration{Monthly}(2), Duration{Monthly}(3), Duration{Monthly}(4)], 2, 2), 1, 32),
                ("matrix_int128", reshape(Int128[typemin(Int128), -1, 0, typemax(Int128)], 2, 2), 1, 0),
                ("matrix_uint128", reshape(UInt128[0, 1, 2, typemax(UInt128)], 2, 2), 2, 0),
                ("matrix_complexf16", reshape(ComplexF16[complex(Float16(-0.0), Float16(1.25)),
                    complex(Float16(2.0), Float16(3.0))], 1, 2), 5, 0))
                id, a, _, _ = raw_matrix(name)
                @test (Int(a.eltype), Int(a.elfreq)) == (eltype_code, elfreq)
                @test isequal(DE.load_mvtseries(db, id), expected)
            end
            for (name, token, expected) in (
                ("matrix_diagonal", "Diagonal", Diagonal(Float64[3.0, 5.0])),
                ("matrix_symmetric", "Symmetric", Symmetric(reshape(Float64[1, 2, 3, 4], 2, 2))),
                ("matrix_hermitian", "Hermitian",
                 Hermitian(reshape(ComplexF64[complex(1.0, 0.0), complex(2.0, -1.0),
                                              complex(2.0, 1.0), complex(4.0, 0.0)], 2, 2))))
                id, a, attrs, _ = raw_matrix(name)
                @test attrs["jtype"] == token
                # The writer materialises the authoritative triangle, so the
                # dense stored bytes plus the marker rebuild the value exactly.
                @test isequal(DE.load_mvtseries(db, id), expected)
            end
            for (name, first, freq, names, expected) in (
                ("mvts_float64", MIT{Monthly}(2024, 1), 32, ["a", "b"],
                 reshape(Float64[1, 2, 3, 4, 5, 6], 3, 2)),
                ("mvts_int64", MIT{Quarterly{3}}(2020, 1), 67, ["x", "y"],
                 reshape(Int64[1, 2, 3, 4], 2, 2)),
                ("mvts_bool", MIT{Monthly}(2024, 1), 32, ["a", "b"],
                 reshape(Bool[true, false, false, true], 2, 2)),
                ("mvts_one_col", MIT{Monthly}(2024, 1), 32, ["only"],
                 reshape(Float64[1, 2], 2, 1)),
                ("mvts_unicode", MIT{Monthly}(2024, 1), 32, ["a\u00e9", "\U0001f642"],
                 reshape(Float64[1, 2, 3, 4], 2, 2)),
                ("mvts_empty_name", MIT{Monthly}(2024, 1), 32, ["", "b"],
                 reshape(Float64[1, 2, 3, 4], 2, 2)))
                id, a, _, stored_names = raw_matrix(name)
                @test (Int(a.object.obj_type), Int(a.axis1.ax_type), Int(a.axis2.ax_type)) ==
                    (21, 1, 2)
                @test (Int(a.axis1.frequency), Int(a.axis1.first)) == (freq, Int(first))
                @test (Int(a.axis1.length), Int(a.axis2.length)) == size(expected)
                @test split(stored_names, '\n') == names
                value = DE.load_mvtseries(db, id)
                @test value isa MVTSeries
                @test isequal(value.values, expected)
                @test string.(collect(keys(value.columns))) == names
                @test first == firstdate(value)
            end
            id, a, attrs, _ = raw_matrix("mvts_zero_rows")
            @test (Int(a.axis1.length), Int(a.axis2.length), Int(a.nbytes)) == (0, 2, 0)
            if haskey(attrs, "jeltype")
                @test DE.load_mvtseries(db, id) == Float64[]
            else
                @test DE.load_mvtseries(db, id) isa MVTSeries
            end
            for (label, F, code, lo, hi) in matrix_axis_families
                for (suffix, value) in (("min", lo), ("max", hi))
                    id, a, _, _ = raw_matrix("mvts_$(suffix)_$(label)")
                    @test (Int(a.axis1.frequency), Int(a.axis1.first)) == (code, value)
                    loaded = DE.load_mvtseries(db, id)
                    @test firstdate(loaded) == MIT{F}(value)
                    @test isequal(loaded.values, reshape(Float64[1], 1, 1))
                end
            end
            for (name, expected, marker) in (
                ("text_ascii", matrix_text_strings, nothing),
                ("text_multibyte", matrix_text_multibyte, nothing),
                ("text_one_empty_string", [""], nothing),
                ("text_symbol", matrix_text_symbols, "Symbol"),
                ("text_multibyte_symbol", Symbol.(matrix_text_multibyte), "Symbol"))
                id = DE.find_object(db, DE.root_id, name)
                raw = Ref{C.tseries_t}()
                @test C.de_load_tseries(db, id, raw) == 0
                a = raw[]
                @test (Int(a.object.obj_type), Int(a.eltype), Int(a.axis.length)) ==
                    (10, 6, length(expected))
                # The payload is the UTF-8 bytes of every element plus one
                # terminator each, which is what the native packer sizes.
                @test Int(a.nbytes) ==
                    sum(item -> sizeof(String(item)), expected; init=0) + length(expected)
                attrs = Dict(string(k) => string(v) for (k, v) in DE.get_all_attributes(db, id))
                @test isequal(DE.load_tseries(db, id), expected)
                @test get(attrs, "jeltype", nothing) == marker
            end
            id = DE.find_object(db, DE.root_id, "text_empty")
            raw = Ref{C.tseries_t}()
            @test C.de_load_tseries(db, id, raw) == 0
            @test (Int(raw[].axis.length), Int(raw[].nbytes)) == (0, 0)
            @test DE.load_tseries(db, id) == String[]
        end
    end
end

# ---- N-dimensional plain arrays (three to five plain axes) -------------------

# Julia's own writer has no empty-array method above rank two, so empties are
# stored through the same C entry points the library uses, carrying the element
# token Julia writes on every other empty array. Marked objects are stored the
# same way so both markers are exact.
function store_raw_tensor!(db, name, eltype, elfreq, payload::Vector{UInt8}, dims;
                           marker=nothing, objmarker=nothing)
    ids = C.axis_id_t[]
    for n in dims
        ax = Ref{C.axis_id_t}()
        @test C.de_axis_plain(db, n, ax) == 0
        push!(ids, ax[])
    end
    id = Ref{C.obj_id_t}()
    GC.@preserve payload ids begin
        ptr = isempty(payload) ? C_NULL : pointer(payload)
        @test C.de_store_ndtseries(db, DE.root_id, name, C.type_tensor, eltype, elfreq,
                                   length(ids), pointer(ids), length(payload), ptr, id) == 0
    end
    marker === nothing || DE.set_attribute(db, id[], "jeltype", marker)
    objmarker === nothing || DE.set_attribute(db, id[], "jtype", objmarker)
    return id[]
end

tensor_bytes(v::Vector{T}) where {T} = Vector{UInt8}(reinterpret(UInt8, v))
const tensor_run = Int64.(1:24)
const tensor_mit_values = MIT{Monthly}[MIT{Monthly}(-1), MIT{Monthly}(0), MIT{Monthly}(1), MIT{Monthly}(2)]
const tensor_duration_values = Duration{Quarterly{1}}[Duration{Quarterly{1}}(1), Duration{Quarterly{1}}(-2),
    Duration{Quarterly{1}}(typemax(Int64)), Duration{Quarterly{1}}(0)]
const tensor_complexf16_values = ComplexF16[complex(Float16(-0.0), Float16(1.25)),
    complex(Float16(2.0), Float16(3.0))]

if action == "generate-tensors"
    DE.opendaec(filename; readonly=false) do db
        for T in matrix_text_types
            values = collect(matrix_text_values(T))
            DE.store_ndtseries(db, "tensor_$(T)", reshape(values, 1, 2, 3))
            DE.store_ndtseries(db, "tensor5_$(T)", reshape(values, 3, 1, 2, 1, 1))
            store_raw_tensor!(db, "tensor_$(T)_empty", TimeSeriesEcon.DataEcon.I._to_de_scalar_type(T),
                              C.freq_none, UInt8[], (0, 2, 3); marker=string(T))
        end
        for T in (Int64, Float64, Bool, UInt8)
            DE.store_ndtseries(db, "tensor4_$(T)", reshape(collect(matrix_text_values(T)), 1, 2, 3, 1))
        end
        store_raw_tensor!(db, "tensor_empty_2x0x3", C.type_float, C.freq_none, UInt8[], (2, 0, 3); marker="Float64")
        store_raw_tensor!(db, "tensor_empty_2x3x0", C.type_float, C.freq_none, UInt8[], (2, 3, 0); marker="Float64")
        store_raw_tensor!(db, "tensor_empty_rank5", C.type_float, C.freq_none, UInt8[], (0, 0, 0, 0, 0); marker="Float64")
        store_raw_tensor!(db, "tensor_empty_rank4_bool", C.type_integer, C.freq_none, UInt8[], (0, 1, 1, 1); marker="Bool")
        DE.store_ndtseries(db, "tensor_run_2x3x4", reshape(tensor_run, 2, 3, 4))
        DE.store_ndtseries(db, "tensor_run_4x3x2", reshape(tensor_run, 4, 3, 2))
        DE.store_ndtseries(db, "tensor_run_2x2x2x3", reshape(tensor_run, 2, 2, 2, 3))
        DE.store_ndtseries(db, "tensor_run_2x2x2x3x1", reshape(tensor_run, 2, 2, 2, 3, 1))
        DE.store_ndtseries(db, "tensor_1x1x1", reshape(Float64[2.5], 1, 1, 1))
        DE.store_ndtseries(db, "tensor_1x1x1x1x1", reshape(Int64[42], 1, 1, 1, 1, 1))
        DE.store_ndtseries(db, "tensor_mit", reshape(tensor_mit_values, 2, 1, 2))
        DE.store_ndtseries(db, "tensor_mit_unit5",
            reshape(MIT{Unit}[MIT{Unit}(typemin(Int64)), MIT{Unit}(typemax(Int64))], 1, 2, 1, 1, 1))
        DE.store_ndtseries(db, "tensor_duration_q1", reshape(tensor_duration_values, 1, 2, 2))
        DE.store_ndtseries(db, "tensor_int128", reshape(Int128[typemin(Int128), -1, 0, typemax(Int128)], 2, 1, 2))
        DE.store_ndtseries(db, "tensor_uint128", reshape(UInt128[0, 1, 2, typemax(UInt128)], 1, 2, 2))
        DE.store_ndtseries(db, "tensor_complexf16", reshape(tensor_complexf16_values, 1, 1, 2))
        DE.store_ndtseries(db, "tensor_complexf16_4d", reshape(tensor_complexf16_values, 2, 1, 1, 1))
        DE.store_ndtseries(db, "tensor_bitarray", BitArray(reshape(collect(matrix_text_values(Bool)), 1, 2, 3)))
        DE.store_tseries(db, "bit_vector", BitVector([true, false, true]))
        DE.store_mvtseries(db, "bit_matrix", BitMatrix([true false; false true]))
        cube = tensor_bytes(Int64.(1:8))
        store_raw_tensor!(db, "tensor_marked_float64", C.type_integer, C.freq_none, cube, (2, 2, 2); marker="Float64")
        store_raw_tensor!(db, "tensor_object_float64", C.type_integer, C.freq_none, cube, (2, 2, 2); objmarker="Array{Float64,3}")
        store_raw_tensor!(db, "tensor_object_identity", C.type_integer, C.freq_none, cube, (2, 2, 2); objmarker="Array")
        store_raw_tensor!(db, "tensor_marked_bool_wide", C.type_integer, C.freq_none,
                          tensor_bytes(UInt64[0, 0, 1, 0]), (1, 2, 1); marker="Bool")
    end
end

if action in ("generate-tensors", "verify-tensors", "verify-wheel")
    @testset "DataEcon N-dimensional array interchange" begin
        DE.opendaec(filename) do db
            # The value pointer is borrowed and valid only until the next
            # library call: the payload is copied right after the load, before
            # the attribute lookup, and only owned bytes are returned.
            function raw_tensor(name)
                id = DE.find_object(db, DE.root_id, name)
                ref = Ref{C.ndtseries_t}()
                @test C.de_load_ndtseries(db, id, ref) == 0
                a = ref[]
                payload = a.nbytes == 0 ? UInt8[] :
                    copy(unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(a.value), a.nbytes))
                attrs = Dict(string(k) => string(v) for (k, v) in DE.get_all_attributes(db, id))
                return id, a, attrs, payload
            end
            dims_of(a) = Int[ax.length for ax in a.axis if ax.id > 0]
            function check_tensor(name, expected; attrs_expected=nothing, eltype_code=nothing, elfreq=nothing)
                id, a, attrs, _ = raw_tensor(name)
                @test (Int(a.object.obj_class), Int(a.object.obj_type), Int(a.naxes)) ==
                    (4, 30, ndims(expected))
                @test all(ax.ax_type == C.axis_plain for ax in a.axis[1:Int(a.naxes)])
                @test dims_of(a) == collect(size(expected))
                @test Int(a.nbytes) == length(expected) * sizeof(eltype(expected))
                eltype_code === nothing || @test (Int(a.eltype), Int(a.elfreq)) == (eltype_code, elfreq)
                value = DE.load_ndtseries(db, id)
                @test typeof(value) == typeof(expected)
                @test isequal(value, expected)
                attrs_expected === nothing || @test attrs == attrs_expected
                return a
            end
            for T in matrix_text_types
                values = collect(matrix_text_values(T))
                marks = T === Bool ? Dict("jeltype" => "Bool") : Dict()
                check_tensor("tensor_$(T)", reshape(values, 1, 2, 3); attrs_expected=marks)
                check_tensor("tensor5_$(T)", reshape(values, 3, 1, 2, 1, 1); attrs_expected=marks)
                # Empty tensors keep their axes in storage. Julia's marker path
                # returns a flat typed vector; an unmarked kind default (what
                # Python writes) keeps its shape through `reshape`.
                id, a, attrs, _ = raw_tensor("tensor_$(T)_empty")
                @test (Int(a.naxes), dims_of(a), Int(a.nbytes)) == (3, [0, 2, 3], 0)
                loaded = DE.load_ndtseries(db, id)
                if haskey(attrs, "jeltype")
                    @test attrs["jeltype"] == string(T)
                    @test loaded == T[]
                else
                    @test T in (Int64, UInt64, Float64, ComplexF64)
                    @test loaded == Array{T}(undef, 0, 2, 3)
                end
            end
            for T in (Int64, Float64, Bool, UInt8)
                check_tensor("tensor4_$(T)", reshape(collect(matrix_text_values(T)), 1, 2, 3, 1))
            end
            for (name, dims) in (("tensor_empty_2x0x3", [2, 0, 3]), ("tensor_empty_2x3x0", [2, 3, 0]),
                                 ("tensor_empty_rank5", [0, 0, 0, 0, 0]))
                id, a, attrs, _ = raw_tensor(name)
                @test (Int(a.naxes), dims_of(a), Int(a.nbytes)) == (length(dims), dims, 0)
                loaded = DE.load_ndtseries(db, id)
                @test loaded == (haskey(attrs, "jeltype") ? Float64[] : Array{Float64}(undef, dims...))
            end
            id, a, attrs, _ = raw_tensor("tensor_empty_rank4_bool")
            @test (Int(a.naxes), dims_of(a), attrs) == (4, [0, 1, 1, 1], Dict("jeltype" => "Bool"))
            @test DE.load_ndtseries(db, id) == Bool[]
            # One column-major run at four shapes stores identical bytes.
            run_bytes(name) = raw_tensor(name)[4]
            reference = run_bytes("tensor_run_2x3x4")
            @test reference == tensor_bytes(tensor_run)
            @test all(run_bytes(name) == reference for name in
                ("tensor_run_4x3x2", "tensor_run_2x2x2x3", "tensor_run_2x2x2x3x1"))
            check_tensor("tensor_run_2x3x4", reshape(tensor_run, 2, 3, 4))
            check_tensor("tensor_run_4x3x2", reshape(tensor_run, 4, 3, 2))
            check_tensor("tensor_run_2x2x2x3", reshape(tensor_run, 2, 2, 2, 3))
            check_tensor("tensor_run_2x2x2x3x1", reshape(tensor_run, 2, 2, 2, 3, 1))
            check_tensor("tensor_1x1x1", reshape(Float64[2.5], 1, 1, 1))
            check_tensor("tensor_1x1x1x1x1", reshape(Int64[42], 1, 1, 1, 1, 1))
            check_tensor("tensor_mit", reshape(tensor_mit_values, 2, 1, 2); eltype_code=3, elfreq=32, attrs_expected=Dict())
            check_tensor("tensor_mit_unit5",
                reshape(MIT{Unit}[MIT{Unit}(typemin(Int64)), MIT{Unit}(typemax(Int64))], 1, 2, 1, 1, 1);
                eltype_code=3, elfreq=11)
            check_tensor("tensor_duration_q1", reshape(tensor_duration_values, 1, 2, 2); eltype_code=1, elfreq=65)
            check_tensor("tensor_int128", reshape(Int128[typemin(Int128), -1, 0, typemax(Int128)], 2, 1, 2); eltype_code=1, elfreq=0)
            check_tensor("tensor_uint128", reshape(UInt128[0, 1, 2, typemax(UInt128)], 1, 2, 2); eltype_code=2, elfreq=0)
            check_tensor("tensor_complexf16", reshape(tensor_complexf16_values, 1, 1, 2); eltype_code=5, elfreq=0)
            check_tensor("tensor_complexf16_4d", reshape(tensor_complexf16_values, 2, 1, 1, 1); eltype_code=5, elfreq=0)
            # Julia's BitArray writer stores a rank token plus the Bool element token.
            check_tensor("tensor_bitarray", BitArray(reshape(collect(matrix_text_values(Bool)), 1, 2, 3));
                         attrs_expected=Dict("jtype" => "BitArray{3}", "jeltype" => "Bool"))
            let id = DE.find_object(db, DE.root_id, "bit_vector")
                attrs = Dict(string(k) => string(v) for (k, v) in DE.get_all_attributes(db, id))
                @test attrs == Dict("jtype" => "BitVector", "jeltype" => "Bool")
                @test DE.load_tseries(db, id) == BitVector([true, false, true])
            end
            let id = DE.find_object(db, DE.root_id, "bit_matrix")
                attrs = Dict(string(k) => string(v) for (k, v) in DE.get_all_attributes(db, id))
                @test attrs == Dict("jtype" => "BitMatrix", "jeltype" => "Bool")
                @test DE.load_mvtseries(db, id) == BitMatrix([true false; false true])
            end
            # Preserved markers convert exactly as on vectors; the stored bytes stay Int64.
            cube = reshape(Int64.(1:8), 2, 2, 2)
            check_tensor("tensor_marked_float64", Float64.(cube); attrs_expected=Dict("jeltype" => "Float64"), eltype_code=1, elfreq=0)
            check_tensor("tensor_object_float64", Float64.(cube); attrs_expected=Dict("jtype" => "Array{Float64,3}"), eltype_code=1, elfreq=0)
            check_tensor("tensor_object_identity", cube; attrs_expected=Dict("jtype" => "Array"), eltype_code=1, elfreq=0)
            id, a, attrs, _ = raw_tensor("tensor_marked_bool_wide")
            @test (Int(a.eltype), Int(a.nbytes), dims_of(a), attrs) == (1, 32, [1, 2, 1], Dict("jeltype" => "Bool"))
            @test DE.load_ndtseries(db, id) == reshape(Bool[false, true], 1, 2, 1)
        end
    end
end


# ---- Catalogs, nested paths, listing and attributes -------------------------

# The same tree is written by Julia at the root of the fixture and by Python
# under "/catalogs" in the wheel output; `prefix` selects which one to verify.
const catalog_order_names = ["b", "B", "a", "1", " sp", "_u", "\u00e4", "Z"]
const catalog_attributes = [("note", "second"), ("empty", ""), ("", "empty name"),
    ("unicod\u00e9/\u540d ", "v\u00e4lue \U0001f642\nline\ttab"), ("delim", "a\u2016b"),
    ("unit", "a\x1fb"), ("k\u20163", "v3"), ("run\x1e\x1f\x1f", "r"), ("long", repeat("x", 5000))]
const catalog_dates = MIT{Monthly}[MIT{Monthly}(24288), MIT{Monthly}(24289)]

function write_catalog_tree!(db, prefix)
    cat = DE.new_catalog(db, prefix * "/cat")
    sub = DE.new_catalog(db, cat, "sub")
    DE.new_catalog(db, prefix * "/cat/sub/deep")
    DE.new_catalog(db, prefix * "/cat/\u65e5\u672c\u8a9e")
    DE.store_scalar(db, prefix * "/cat/scalar", 1.5)
    DE.store_scalar(db, sub, "text", "vintage \u2016 3")
    DE.store_tseries(db, prefix * "/cat/sub/vector", Int32[1, 2, 3])
    DE.store_mvtseries(db, prefix * "/cat/sub/matrix", [1.0 2.0; 3.0 4.0])
    DE.store_ndtseries(db, prefix * "/cat/sub/deep/tensor", reshape(Int64.(1:24), 2, 3, 4))
    DE.store_tseries(db, prefix * "/cat/sub/deep/series", TSeries(2024M1, [1.0, 2.0, 3.0]))
    DE.store_mvtseries(db, prefix * "/cat/sub/mvtseries", MVTSeries(2024Q1, (:p, :q), [1.0 2.0; 3.0 4.0]))
    DE.store_tseries(db, prefix * "/cat/sub/bools", TSeries(2024M1, [true, false]))
    DE.store_tseries(db, prefix * "/cat/sub/text_vector", ["a", "b"])
    DE.store_tseries(db, prefix * "/cat/\u65e5\u672c\u8a9e/dates", TSeries(2024M1, catalog_dates))
    DE.store_mvtseries(db, prefix * "/cat/\u65e5\u672c\u8a9e/date_array", reshape(catalog_dates, 1, 2))
    for name in catalog_order_names
        DE.store_scalar(db, cat, name, 1)
    end
    # Julia stores a child under a scalar (the native store does not check the
    # parent's class); Python reads it by path but never lists it.
    DE.store_scalar(db, prefix * "/cat/scalar_parent", 2)
    DE.store_scalar(db, prefix * "/cat/scalar_parent/child", 3)
    DE.set_attribute(db, prefix * "/cat/scalar", "note", "first")
    for (name, value) in catalog_attributes
        DE.set_attribute(db, prefix * "/cat/scalar", name, value)
    end
    DE.set_attribute(db, cat, "owner", "julia")
    DE.set_attribute(db, DE.root_id, "root_note", "hello")
end

function verify_catalog_tree(db, prefix; julia_written::Bool)
    cat = DE.find_fullpath(db, prefix * "/cat")
    @test DE.get_fullpath(db, cat) == prefix * "/cat"
    @test DE.find_object(db, cat, "sub") == DE.find_fullpath(db, prefix * "/cat/sub")
    @test DE.catalog_size(db, prefix * "/cat") == 4 + length(catalog_order_names)
    @test DE.catalog_size(db, prefix * "/cat/sub") == 7
    @test DE.catalog_size(db, prefix * "/cat/sub/deep") == 2
    @test DE.catalog_size(db, prefix * "/cat/\u65e5\u672c\u8a9e") == 2
    @test DE.load_scalar(db, prefix * "/cat/scalar") == 1.5
    @test DE.load_scalar(db, prefix * "/cat/sub/text") == "vintage \u2016 3"
    @test DE.load_tseries(db, prefix * "/cat/sub/vector") == Int32[1, 2, 3]
    @test DE.load_mvtseries(db, prefix * "/cat/sub/matrix") == [1.0 2.0; 3.0 4.0]
    @test DE.load_ndtseries(db, prefix * "/cat/sub/deep/tensor") == reshape(Int64.(1:24), 2, 3, 4)
    @test DE.load_tseries(db, prefix * "/cat/sub/deep/series") == TSeries(2024M1, [1.0, 2.0, 3.0])
    @test DE.load_mvtseries(db, prefix * "/cat/sub/mvtseries") == MVTSeries(2024Q1, (:p, :q), [1.0 2.0; 3.0 4.0])
    @test DE.load_tseries(db, prefix * "/cat/sub/bools") == TSeries(2024M1, [true, false])
    @test DE.get_attribute(db, prefix * "/cat/sub/bools", "jeltype") == "Bool"
    @test DE.load_tseries(db, prefix * "/cat/sub/text_vector") == ["a", "b"]
    @test DE.load_tseries(db, prefix * "/cat/\u65e5\u672c\u8a9e/dates") == TSeries(2024M1, catalog_dates)
    @test DE.load_mvtseries(db, prefix * "/cat/\u65e5\u672c\u8a9e/date_array") == reshape(catalog_dates, 1, 2)
    for name in catalog_order_names
        @test DE.load_scalar(db, cat, name) == 1
    end
    @test DE.load_scalar(db, prefix * "/cat/scalar_parent") == 2
    if julia_written
        @test DE.load_scalar(db, prefix * "/cat/scalar_parent/child") == 3
        @test DE.catalog_size(db, prefix * "/cat/scalar_parent") == 1
    else
        # Python refuses to store under a non-catalog parent.
        @test DE.find_fullpath(db, prefix * "/cat/scalar_parent/child", false) === missing
    end
    # Native listing order is the UTF-8 byte order of the names.
    search = Ref{C.de_search}()
    @test C.de_list_catalog(db, cat, search) == 0
    obj = Ref{C.object_t}()
    listed = String[]
    rc = C.de_next_object(search[], obj)
    while rc == C.DE_SUCCESS
        push!(listed, unsafe_string(obj[].name))
        rc = C.de_next_object(search[], obj)
    end
    @test rc == C.DE_NO_OBJ
    @test C.de_finalize_search(search[]) == 0
    @test listed == sort(listed; by=codeunits)
    @test length(listed) == DE.catalog_size(db, cat)
    # Julia's returned list: non-catalog full paths, recursive through catalogs only.
    paths = DE.list_catalog(db, prefix * "/cat"; quiet=true)
    @test prefix * "/cat/sub/deep/tensor" in paths
    @test prefix * "/cat/\u65e5\u672c\u8a9e/dates" in paths
    @test !(prefix * "/cat/sub" in paths)
    @test !(prefix * "/cat/scalar_parent/child" in paths)
    @test length(paths) == 11 + length(catalog_order_names) + 1
    # Attributes: individual reads are exact for every value; the delimited
    # enumeration needs a delimiter absent from the names (Julia's limitation).
    sid = DE.find_fullpath(db, prefix * "/cat/scalar")
    for (name, value) in catalog_attributes
        @test DE.get_attribute(db, sid, name) == value
    end
    @test DE.get_attribute(db, sid, "missing") === missing
    all = DE.get_all_attributes(db, sid; delim="\x1d\x1c")
    @test all == Dict(catalog_attributes)
    @test length(all) == length(catalog_attributes)
    @test DE.get_all_attributes(db, cat) == Dict("owner" => "julia")
    root_attrs = DE.get_all_attributes(db, DE.root_id)
    @test root_attrs["DE_VERSION"] == "0.4.0"
    @test root_attrs["root_note"] == "hello"
    # Every object lives at the depth its path says (objects_info).
    for (path, depth) in ((prefix * "/cat", 1), (prefix * "/cat/sub/deep/tensor", 4),
                          (prefix * "/cat/\u65e5\u672c\u8a9e/dates", 3))
        fp = Ref{Ptr{Cchar}}()
        d = Ref{Int64}(-1)
        @test C.de_get_object_info(db, DE.find_fullpath(db, path), fp, d, C_NULL) == 0
        @test unsafe_string(fp[]) == path
        @test d[] == depth + count("/", prefix)
    end
end

if action == "generate-catalogs"
    DE.opendaec(filename; readonly=false) do db
        write_catalog_tree!(db, "")
    end
end

if action in ("generate-catalogs", "verify-catalogs")
    @testset "DataEcon catalog and attribute interchange" begin
        DE.opendaec(filename) do db
            verify_catalog_tree(db, ""; julia_written=true)
        end
    end
end

if action == "verify-wheel"
    @testset "DataEcon catalog and attribute interchange" begin
        DE.opendaec(filename) do db
            verify_catalog_tree(db, "/catalogs"; julia_written=false)
        end
    end
end

if action == "verify-wheel"
    @testset "DataEcon Boolean scalar interchange" begin
        DE.opendaec(filename) do db
            for (name, expected) in (("bool_false", Int8(0)), ("bool_true", Int8(1)),
                                     ("bool_numpy_false", Int8(0)), ("bool_numpy_true", Int8(1)))
                id = DE.find_object(db, DE.root_id, name)
                scalar = Ref{C.scalar_t}()
                @test C.de_load_scalar(db, id, scalar) == 0
                v = scalar[]
                @test Int.((v.object.obj_class, v.object.obj_type, v.frequency, v.nbytes)) == (1,1,0,1)
                @test unsafe_load(Ptr{Int8}(v.value)) === expected
                @test isempty(DE.get_all_attributes(db, id))
                @test DE.load_scalar(db, id) === expected
            end
        end
    end
end

# ---- Workspace interchange ---------------------------------------------------

# One mixed Workspace written by Julia's writedb at the root of its fixture and
# by Python's write_workspace under "/workspace" in the wheel output; `prefix`
# selects which one to verify. The Julia tree also carries three marked scalars
# (Symbol, Rational, Date) that Python reports as unsupported members on read.
const workspace_dates = MIT{Monthly}[MIT{Monthly}(24288), MIT{Monthly}(24289)]
const workspace_order = [("b", 1), ("a", 2), ("B", 3), ("\u00e4", 4), ("_", 5), ("10", 6), ("9", 7)]

function workspace_tree(; julia_only::Bool)
    ws = Workspace()
    ws.f = 1.5
    ws.i = 7
    ws.i8 = Int8(-3)
    ws.u = UInt64(2)^63
    ws.c = 1.0 + 2.0im
    ws.s = "vintage \u2016 3"
    ws.b = true
    ws.d = 2020Q1
    ws.dur = 2020M3 - 2020M1
    if julia_only
        ws.sym = :symbol
        ws.r = 1 // 2
        ws.date = Date(2020, 1, 15)
    end
    ws.ts = TSeries(2024M1, [1.0, 2.0, 3.0])
    ws.tsi = TSeries(2020Q1, Int64[1, 2])
    ws.tsb = TSeries(2020Y, [true, false])
    ws.tse = TSeries(2024M1, Float64[])
    ws.tsd = TSeries(2024M1, workspace_dates)
    ws.mv = MVTSeries(2024Q1, (:p, :q), [1.0 2.0; 3.0 4.0])
    ws.mat = [1.0 2.0; 3.0 4.0]
    ws.v = Int32[1, 2, 3]
    ws.vs = ["a", "b"]
    ws.ur = 1:5
    ws.mr = 2020Q1:2020Q4
    ws.t3 = reshape(Int64.(1:24), 2, 3, 4)
    ws.nested = Workspace(x = 1, deeper = Workspace(y = 2.0))
    ws.nested.deeper[Symbol("日本語")] = 9
    ws.empty = Workspace()
    ws.order = Workspace()
    for (name, value) in workspace_order
        ws.order[Symbol(name)] = value
    end
    return ws
end

function verify_workspace_tree(db, prefix; julia_written::Bool)
    expected = workspace_tree(julia_only=julia_written)
    ws = DE.readdb(db, prefix == "" ? "/" : prefix)
    @test ws isa Workspace
    @test [string(k) for k in keys(ws)] == sort([string(k) for k in keys(expected)]; by=codeunits)
    @test ws.f === 1.5
    @test ws.i === 7
    @test ws.i8 === Int8(-3)
    @test ws.u === UInt64(2)^63
    @test ws.c === 1.0 + 2.0im
    @test ws.s == "vintage \u2016 3"
    @test ws.b === Int8(1)                       # Bool stores Int8 without a marker
    @test ws.d === 2020Q1
    @test ws.dur === 2020M3 - 2020M1
    if julia_written
        @test ws.sym === :symbol
        @test ws.r === 1 // 2
        @test ws.date == Date(2020, 1, 15)
    else
        @test !haskey(ws, :sym) && !haskey(ws, :r) && !haskey(ws, :date)
    end
    @test ws.ts == expected.ts
    @test ws.tsi == expected.tsi && eltype(ws.tsi) == Int64
    @test ws.tsb == expected.tsb && eltype(ws.tsb) == Bool
    if julia_written
        @test ws.tse == Float64[]                # Julia's own empty marker reloads as a vector
    else
        # Python omits the redundant empty Float64 marker, so the loader keeps the dated series.
        @test ws.tse isa TSeries && isempty(ws.tse) && firstdate(ws.tse) == 2024M1
    end
    @test ws.tsd == expected.tsd && eltype(ws.tsd) == MIT{Monthly}
    @test ws.mv == expected.mv && collect(colnames(ws.mv)) == [:p, :q]
    @test ws.mat == expected.mat && ws.mat isa Matrix{Float64}
    @test ws.v == expected.v && ws.v isa Vector{Int32}
    @test ws.vs == ["a", "b"]
    @test ws.ur == 1:5
    @test ws.mr == 2020Q1:2020Q4
    @test ws.t3 == expected.t3 && ws.t3 isa Array{Int64,3}
    @test ws.nested isa Workspace && ws.nested.x === 1
    @test ws.nested.deeper.y === 2.0 && ws.nested.deeper[Symbol("日本語")] === 9
    @test [string(k) for k in keys(ws.nested)] == ["deeper", "x"]
    @test ws.empty isa Workspace && isempty(ws.empty)
    @test [string(k) for k in keys(ws.order)] == ["10", "9", "B", "_", "a", "b", "\u00e4"]
    @test [ws.order[Symbol(n)] for (n, _) in workspace_order] == [v for (_, v) in workspace_order]
    # Markers are exactly the codec's; user attributes are not part of a Workspace.
    @test DE.get_all_attributes(db, prefix * "/b") == Dict()
    @test DE.get_all_attributes(db, prefix * "/tsb") == Dict("jeltype" => "Bool")
    @test DE.get_all_attributes(db, prefix * "/tse") == (julia_written ? Dict("jeltype" => "Float64") : Dict())
    @test DE.get_all_attributes(db, prefix * "/tsd") == Dict()
    @test DE.get_all_attributes(db, prefix * "/nested") == Dict()
    @test DE.get_attribute(db, prefix * "/f", "note") == "user attribute"
    @test DE.catalog_size(db, prefix * "/empty") == 0
    @test DE.catalog_size(db, prefix * "/nested") == 2
    @test DE.catalog_size(db, prefix == "" ? DE.root_id : prefix) == length(expected)
end

if action == "generate-workspace"
    DE.opendaec(filename; readonly=false) do db
        DE.writedb(db, workspace_tree(julia_only=true))
        DE.set_attribute(db, "/f", "note", "user attribute")
    end
end

if action in ("generate-workspace", "verify-workspace")
    @testset "DataEcon Workspace interchange" begin
        DE.opendaec(filename) do db
            verify_workspace_tree(db, ""; julia_written=true)
        end
    end
end

if action == "verify-wheel"
    @testset "DataEcon Workspace interchange" begin
        DE.opendaec(filename) do db
            verify_workspace_tree(db, "/workspace"; julia_written=false)
        end
    end
end

# ---- Text matrices and tensors (element type 6 at ranks two to five) ---------

# Julia's own writer stores `Array{String,N}` and `Array{Symbol,N}` through the
# same packed-string payload as vectors (column-major, one NUL per element) and
# reads them back with their shape; it cannot size multibyte text (DE_SHORT_BUF)
# and has no empty-array method above rank two, so those rows are stored
# through the C entry points the library itself uses. A marked empty reloads in
# Julia as a flat typed vector; an unmarked one keeps its shape.
const text_m23 = ["r1c1" "r1c2" "r1c3"; "r2c1" "r2c2" "r2c3"]
const text_multi22 = ["é" "aé"; "\U0001f642" "z"]
const text_t223 = reshape([string("e", i) for i in 1:12], 2, 2, 3)
const text_h24 = [string("h", i) for i in 1:24]

function store_packed_text_array!(db, name, value; marker=nothing)
    buffer = UInt8[]
    for item in vec(value)
        append!(buffer, Vector{UInt8}(codeunits(String(item))))
        push!(buffer, 0x00)
    end
    ids = C.axis_id_t[]
    for n in size(value)
        ax = Ref{C.axis_id_t}()
        @test C.de_axis_plain(db, n, ax) == 0
        push!(ids, ax[])
    end
    id = Ref{C.obj_id_t}()
    GC.@preserve buffer ids begin
        ptr = isempty(buffer) ? C_NULL : pointer(buffer)
        if ndims(value) == 2
            @test C.de_store_mvtseries(db, DE.root_id, name, C.type_matrix, C.type_string,
                                       C.freq_none, ids[1], ids[2], length(buffer), ptr, id) == 0
        else
            @test C.de_store_ndtseries(db, DE.root_id, name, C.type_tensor, C.type_string,
                                       C.freq_none, length(ids), pointer(ids), length(buffer), ptr, id) == 0
        end
    end
    marker === nothing || DE.set_attribute(db, id[], "jeltype", marker)
    return id[]
end

# (name, expected loaded value, expected jeltype, written by Julia's own writer)
function text_array_inventory()
    return [
        ("text2_2x3", text_m23, nothing, true),
        ("text2_3x2", permutedims(text_m23), nothing, true),
        ("text2_blank_2x2", ["" ""; "" ""], nothing, true),
        ("text2_symbol_2x2", [:alpha :b; Symbol("") :d], "Symbol", true),
        ("text2_substring_1x2", SubString{String}[SubString("abc", 1, 2) SubString("abc", 2, 3)], "SubString{String}", true),
        ("text2_empty_0x0", Matrix{String}(undef, 0, 0), "String", true),
        ("text2_empty_0x3", Matrix{String}(undef, 0, 3), "String", true),
        ("text2_empty_3x0", Matrix{String}(undef, 3, 0), "String", true),
        ("text2_symbol_empty_0x2", Matrix{Symbol}(undef, 0, 2), "Symbol", true),
        ("text3_2x2x3", text_t223, nothing, true),
        ("text3_3x2x4", reshape(text_h24, 3, 2, 4), nothing, true),
        ("text3_4x3x2", reshape(text_h24, 4, 3, 2), nothing, true),
        ("text4_1x2x3x1", reshape([string("f", i) for i in 1:6], 1, 2, 3, 1), nothing, true),
        ("text5_2x1x2x1x1", reshape([string("g", i) for i in 1:4], 2, 1, 2, 1, 1), nothing, true),
        ("text3_symbol_1x2x3", reshape([:a, :bb, :ccc, Symbol(""), :e, :f], 1, 2, 3), "Symbol", true),
        ("text5_symbol_1x1x2x1x1", reshape([:p, :q], 1, 1, 2, 1, 1), "Symbol", true),
        ("text3_substring_1x1x2", reshape(SubString{String}[SubString("abc", 1, 2), SubString("abc", 2, 3)], 1, 1, 2), "SubString{String}", true),
        ("text2_multibyte_2x2", text_multi22, nothing, false),
        ("text2_multibyte_symbol_2x2", Symbol.(text_multi22), "Symbol", false),
        ("text3_multibyte_1x2x2", reshape(vec(text_multi22), 1, 2, 2), nothing, false),
        ("text5_multibyte_symbol_2x1x1x1x2", reshape(Symbol.(vec(text_multi22)), 2, 1, 1, 1, 2), "Symbol", false),
        ("text3_empty_0x2x3", Array{String,3}(undef, 0, 2, 3), nothing, false),
        ("text3_empty_marked_0x2x3", Array{String,3}(undef, 0, 2, 3), "String", false),
        ("text5_symbol_empty_0x0x0x0x0", Array{Symbol,5}(undef, 0, 0, 0, 0, 0), "Symbol", false),
        ("text4_empty_1x0x1x0", Array{String,4}(undef, 1, 0, 1, 0), nothing, false),
        ("text2_string_marker_2x3", text_m23, "String", false),
        ("text3_string_marker_2x2x3", text_t223, "String", false),
        ("text2_abstract_2x3", text_m23, "AbstractString", false),
    ]
end

text_workspace() = Workspace(:tm => text_m23, :ts => [:alpha :b; Symbol("") :d], :tt => text_t223,
                             :t5 => reshape([:p, :q], 1, 1, 2, 1, 1), :tv => ["a", "b"])

if action == "generate-text-arrays"
    DE.opendaec(filename; readonly=false) do db
        for (name, value, marker, writable) in text_array_inventory()
            if writable
                ndims(value) == 2 ? DE.store_mvtseries(db, name, value) : DE.store_ndtseries(db, name, value)
            else
                store_packed_text_array!(db, name, value; marker=marker)
            end
        end
        DE.writedb(db, DE.new_catalog(db, DE.root_id, "text_ws"), text_workspace())
    end
end

if action in ("generate-text-arrays", "verify-text-arrays", "verify-wheel")
    @testset "DataEcon text array interchange" begin
        reference_fixture = action != "verify-wheel"
        DE.opendaec(filename) do db
            # Borrowed payload pointers are copied before any further native call.
            function raw_text(id)
                obj = Ref{C.object_t}()
                @test C.de_load_object(db, id, obj) == 0
                if Int(obj[].obj_class) == Int(C.class_mvtseries)
                    ref = Ref{C.mvtseries_t}()
                    @test C.de_load_mvtseries(db, id, ref) == 0
                    a = ref[]
                    dims = Int[a.axis1.length, a.axis2.length]
                    types = Int[a.axis1.ax_type, a.axis2.ax_type]
                    header = (Int(a.object.obj_type), Int(a.eltype), Int(a.elfreq), Int(a.nbytes))
                    value = a.value
                else
                    ref = Ref{C.ndtseries_t}()
                    @test C.de_load_ndtseries(db, id, ref) == 0
                    a = ref[]
                    dims = Int[ax.length for ax in a.axis if ax.id > 0]
                    types = Int[ax.ax_type for ax in a.axis[1:Int(a.naxes)]]
                    header = (Int(a.object.obj_type), Int(a.eltype), Int(a.elfreq), Int(a.nbytes))
                    value = a.value
                end
                payload = header[4] == 0 ? UInt8[] :
                    copy(unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(value), header[4]))
                attrs = Dict(string(k) => string(v) for (k, v) in DE.get_all_attributes(db, id))
                return Int(obj[].obj_class), header, dims, types, payload, attrs
            end
            store_bytes(expected) = let buffer = UInt8[]
                for item in vec(expected)
                    append!(buffer, Vector{UInt8}(codeunits(String(item))))
                    push!(buffer, 0x00)
                end
                buffer
            end
            function check_text(path, expected, marker)
                id = DE.find_fullpath(db, path)
                cls, header, dims, types, payload, attrs = raw_text(id)
                @test (cls, header[1]) == (ndims(expected) == 2 ? (3, 20) : (4, 30))
                @test (header[2], header[3]) == (6, 0)
                @test dims == collect(size(expected))
                @test all(==(0), types)
                # Column-major UTF-8 bytes plus one terminator per element.
                @test payload == store_bytes(expected)
                @test header[4] == sum(item -> sizeof(String(item)), vec(expected); init=0) + length(expected)
                loaded = (cls == 3 ? DE.load_mvtseries : DE.load_ndtseries)(db, id)
                if isempty(expected)
                    # Python marks every empty text array; the fixture also holds
                    # unmarked raw rows, which Julia reads with their shape.
                    reference_fixture && @test get(attrs, "jeltype", nothing) == marker
                    if haskey(attrs, "jeltype")
                        @test loaded == eltype(expected)[]
                    else
                        @test typeof(loaded) == typeof(expected) && size(loaded) == size(expected)
                    end
                else
                    @test get(attrs, "jeltype", nothing) == marker
                    @test !haskey(attrs, "jtype")
                    @test typeof(loaded) == typeof(expected) || marker == "AbstractString"
                    @test isequal(loaded, expected) && size(loaded) == size(expected)
                end
            end
            for (name, value, marker, _) in text_array_inventory()
                check_text("/" * name, value, marker)
            end
            ws = text_workspace()
            back = DE.readdb(db, "/text_ws")
            @test Set(keys(back)) == Set(keys(ws))
            @test all(isequal(back[k], ws[k]) && typeof(back[k]) == typeof(ws[k]) for k in keys(ws))
            check_text("/text_ws/tm", ws.tm, nothing)
            check_text("/text_ws/ts", ws.ts, "Symbol")
            check_text("/text_ws/tt", ws.tt, nothing)
            check_text("/text_ws/t5", ws.t5, "Symbol")
        end
    end
end

# ---- represented MVTSeries and structure-marked matrices --------------------

const rmv_anchor = MIT{Monthly}(2024, 1)   # code 24288
rmv_q(i) = MIT{Quarterly}(i)               # canonical Quarterly{3}, element code 67
rmv_y6(i) = Duration{Yearly{6}}(i)         # element code 262
const rmv_cf16 = reshape(ComplexF16[complex(Float16(-0.0), Float16(1.25)),
                                    complex(Float16(2.0), Float16(3.0)),
                                    complex(reinterpret(Float16, 0x7e55), Float16(-Inf)),
                                    complex(Float16(0.0), Float16(-0.0))], 2, 2)
const rmv_signs = reshape(reinterpret(Float64, UInt64[0x8000000000000000, 0x4020000000000000,
                                                      0x7ff8000000000055, 0x8000000000000000]), 2, 2)
const rmv_csigns = ComplexF64[complex(-0.0, -0.0) complex(-0.0, 7.0); complex(8.0, 8.0) complex(-0.0, -0.0)]

# Store a dated matrix through the raw C ABI with explicit metadata and markers.
function store_raw_mvts!(db, name, eltype, elfreq, payload::Vector{UInt8}, rows, freq, first,
                         names::String; marker=nothing, objmarker=nothing)
    ax1 = Ref{C.axis_id_t}()
    @test C.de_axis_range(db, rows, C.frequency_t(freq), Int64(first), ax1) == 0
    columns = length(split(names, '\n'))
    buf = Vector{UInt8}(codeunits(names))
    push!(buf, 0x00)
    ax2 = Ref{C.axis_id_t}()
    GC.@preserve buf begin
        @test C.de_axis_names(db, columns, Ptr{Cchar}(pointer(buf)), ax2) == 0
    end
    id = Ref{C.obj_id_t}()
    GC.@preserve payload begin
        ptr = isempty(payload) ? C_NULL : pointer(payload)
        @test C.de_store_mvtseries(db, DE.root_id, name, C.type_mvtseries, eltype,
                                   C.frequency_t(elfreq), ax1[], ax2[], length(payload), ptr, id) == 0
    end
    marker === nothing || DE.set_attribute(db, id[], "jeltype", marker)
    objmarker === nothing || DE.set_attribute(db, id[], "jtype", objmarker)
    return id[]
end

rmv_bytes(v::AbstractVector) = Vector{UInt8}(reinterpret(UInt8, collect(v)))

# Each entry: (name, value Julia's writer stores or `nothing` for raw rows,
# expected loaded value or exception type, expected attributes, (eltype,
# elfreq), raw store recipe or `nothing`). The raw recipe is (eltype, elfreq,
# payload, rows, axis frequency, first, names, jeltype, jtype).
function rmv_inventory()
    rows = Any[
        ("rmv_mit_q_on_m", MVTSeries(rmv_anchor, (:a, :b, :c),
            reshape(rmv_q.([-1, 0, 1, 2, typemin(Int64), typemax(Int64)]), 2, 3)),
            :same, Dict{String,String}(), (3, 67), nothing),
        ("rmv_dur_y6_on_d", MVTSeries(MIT{Daily}(738000), (:x, :y),
            reshape(rmv_y6.([1, -2, 3, typemin(Int64), typemax(Int64), 0]), 3, 2)),
            :same, Dict{String,String}(), (1, 262), nothing),
        ("rmv_mit_unit_on_unit", MVTSeries(MIT{Unit}(-5), (:a, :b),
            reshape(MIT{Unit}.([typemin(Int64), 0, 1, typemax(Int64)]), 2, 2)),
            :same, Dict{String,String}(), (3, 11), nothing),
        ("rmv_mit_w7_on_b", MVTSeries(MIT{BDaily}(500000), (:a,),
            reshape(MIT{Weekly{7}}.([-1711422, 1711422, 7]), 3, 1)),
            :same, Dict{String,String}(), (3, 23), nothing),
        ("rmv_int128", MVTSeries(rmv_anchor, (:a, :b),
            reshape(Int128[typemin(Int128), -1, 0, typemax(Int128), 7, Int128(2)^70], 3, 2)),
            :same, Dict{String,String}(), (1, 0), nothing),
        ("rmv_uint128", MVTSeries(rmv_anchor, (:a, :b, :c),
            reshape(UInt128[0, 1, 2, typemax(UInt128), UInt128(2)^70, 5], 2, 3)),
            :same, Dict{String,String}(), (2, 0), nothing),
        ("rmv_complexf16", MVTSeries(rmv_anchor, (:a, :b), rmv_cf16),
            :same, Dict{String,String}(), (5, 0), nothing),
        ("rmv_mit_one_row", MVTSeries(rmv_anchor, (:a, :b, :c), reshape(rmv_q.([1, 2, 3]), 1, 3)),
            :same, Dict{String,String}(), (3, 67), nothing),
        ("rmv_mit_unicode_names", MVTSeries(rmv_anchor, (Symbol("a\u00e9"), Symbol("\U0001f642"), Symbol("")),
            reshape(rmv_q.([1, 2, 3, 4, 5, 6]), 2, 3)),
            :same, Dict{String,String}(), (3, 67), nothing),
        ("rmv_int128_zero_rows", MVTSeries(rmv_anchor:rmv_anchor-1, (:a, :b), Matrix{Int128}(undef, 0, 2)),
            Int128[], Dict("jeltype" => "Int128"), (1, 0), nothing),
        ("rmv_complexf16_zero_rows", MVTSeries(rmv_anchor:rmv_anchor-1, (:a, :b), Matrix{ComplexF16}(undef, 0, 2)),
            ComplexF16[], Dict("jeltype" => "ComplexF16"), (5, 0), nothing),
        ("rmv_dur_zero_rows", MVTSeries(rmv_anchor:rmv_anchor-1, (:a, :b), Matrix{Duration{Yearly{6}}}(undef, 0, 2)),
            MethodError, Dict("jeltype" => "Duration{Yearly{6}}"), (1, 262), nothing),
    ]
    for (label, F, code, lo, hi) in matrix_axis_families
        push!(rows, ("rmv_mit_min_$(label)", MVTSeries(MIT{F}(lo), (:a, :b), reshape(rmv_q.([1, 2]), 1, 2)),
            :same, Dict{String,String}(), (3, 67), nothing))
        push!(rows, ("rmv_mit_max_$(label)", MVTSeries(MIT{F}(hi), (:a, :b), reshape(rmv_q.([1, 2]), 1, 2)),
            :same, Dict{String,String}(), (3, 67), nothing))
    end
    int_payload = rmv_bytes(Int64[1, 0, 2, 1])
    append!(rows, Any[
        ("rmv_bool_on_int128", nothing,
            MVTSeries(rmv_anchor, (:a, :b), Bool[true false; false true]),
            Dict("jeltype" => "Bool"), (1, 0),
            (C.type_integer, 0, rmv_bytes(Int128[1, 0, 0, 1]), 2, 32, 24288, "a\nb", "Bool", nothing)),
        ("rmv_bool_on_int64_01", nothing,
            MVTSeries(rmv_anchor, (:a, :b), Bool[true false; false true]),
            Dict("jeltype" => "Bool"), (1, 0),
            (C.type_integer, 0, rmv_bytes(Int64[1, 0, 0, 1]), 2, 32, 24288, "a\nb", "Bool", nothing)),
        ("rmv_float64_on_int64", nothing,
            MVTSeries(rmv_anchor, (:a, :b), Float64[1 2; 0 1]),
            Dict("jeltype" => "Float64"), (1, 0),
            (C.type_integer, 0, int_payload, 2, 32, 24288, "a\nb", "Float64", nothing)),
        ("rmv_mit_m_on_int64", nothing,
            MVTSeries(rmv_anchor, (:a, :b), MIT{Monthly}[MIT{Monthly}(1) MIT{Monthly}(2); MIT{Monthly}(0) MIT{Monthly}(1)]),
            Dict("jeltype" => "MIT{Monthly}"), (1, 0),
            (C.type_integer, 0, int_payload, 2, 32, 24288, "a\nb", "MIT{Monthly}", nothing)),
        ("rmv_identity_object", nothing,
            MVTSeries(rmv_anchor, (:a, :b), rmv_q.([1 2; 0 1])),
            Dict("jtype" => "MVTSeries"), (3, 67),
            (C.type_date, 67, int_payload, 2, 32, 24288, "a\nb", nothing, "MVTSeries")),
        ("rmv_identity_param", nothing,
            MVTSeries(rmv_anchor, (:a, :b), Int128[1 2; 0 1]),
            Dict("jtype" => "MVTSeries{Monthly, Int128}"), (1, 0),
            (C.type_integer, 0, rmv_bytes(Int128[1, 0, 2, 1]), 2, 32, 24288, "a\nb", nothing, "MVTSeries{Monthly, Int128}")),
        ("rmv_empty_mit_marked", nothing, MethodError,
            Dict("jeltype" => "MIT{Monthly}"), (3, 32),
            (C.type_date, 32, UInt8[], 0, 32, 24288, "a\nb", "MIT{Monthly}", nothing)),
    ])
    return rows
end

# Structure rows: (name, wrapper value, expected attributes, (eltype, elfreq), stored element type).
function rmv_structures()
    A = [1.0 3.0; 20.0 4.0]
    H = ComplexF64[complex(1.0, 9.0) complex(2.0, -1.0); complex(20.0, 20.0) complex(4.0, -9.0)]
    i128 = Int128[typemin(Int128) 3; 20 typemax(Int128)]
    c16 = ComplexF16[complex(Float16(1), Float16(9)) complex(Float16(2), Float16(-1));
                     complex(Float16(20), Float16(20)) complex(Float16(4), Float16(0))]
    mitm = MIT{Monthly}[MIT{Monthly}(1) MIT{Monthly}(3); MIT{Monthly}(20) MIT{Monthly}(4)]
    return Any[
        ("str_sym_u_f64", Symmetric(copy(A), :U), Dict("jtype" => "Symmetric"), (4, 0), Float64),
        ("str_sym_l_f64", Symmetric(copy(A), :L), Dict("jtype" => "Symmetric"), (4, 0), Float64),
        ("str_herm_u_c64", Hermitian(copy(H), :U), Dict("jtype" => "Hermitian"), (5, 0), ComplexF64),
        ("str_herm_l_c64", Hermitian(copy(H), :L), Dict("jtype" => "Hermitian"), (5, 0), ComplexF64),
        ("str_diag_vec_f64", Diagonal(Float64[3.0, -0.0, 5.0]), Dict("jtype" => "Diagonal"), (4, 0), Float64),
        ("str_sym_l_signs", Symmetric(copy(rmv_signs), :L), Dict("jtype" => "Symmetric"), (4, 0), Float64),
        ("str_herm_u_signs", Hermitian(copy(rmv_csigns), :U), Dict("jtype" => "Hermitian"), (5, 0), ComplexF64),
        ("str_sym_u_i128", Symmetric(copy(i128), :U), Dict("jtype" => "Symmetric"), (1, 0), Int128),
        ("str_herm_l_c16", Hermitian(copy(c16), :L), Dict("jtype" => "Hermitian"), (5, 0), ComplexF16),
        ("str_diag_mit_m", Diagonal(copy(mitm)), Dict("jtype" => "Diagonal"), (3, 32), MIT{Monthly}),
        ("str_sym_l_bool", Symmetric(Bool[true true; false false], :L), Dict("jtype" => "Symmetric", "jeltype" => "Bool"), (1, 0), Int8),
        ("str_diag_f16", Diagonal(Float16[1 3; 20 4]), Dict("jtype" => "Diagonal"), (4, 0), Float16),
        ("str_sym_empty_f64", Symmetric(Matrix{Float64}(undef, 0, 0)), Dict("jtype" => "Symmetric", "jeltype" => "Float64"), (4, 0), Float64),
        ("str_diag_empty_f64", Diagonal(Float64[]), Dict("jtype" => "Diagonal", "jeltype" => "Float64"), (4, 0), Float64),
        ("str_herm_1x1_c64", Hermitian(reshape([complex(2.0, 5.0)], 1, 1)), Dict("jtype" => "Hermitian"), (5, 0), ComplexF64),
    ]
end

if action == "generate-represented-mvtseries"
    DE.opendaec(filename; readonly=false) do db
        for (name, value, _, _, _, raw) in rmv_inventory()
            if raw === nothing
                DE.store_mvtseries(db, name, value)
            else
                store_raw_mvts!(db, name, raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7];
                                marker=raw[8], objmarker=raw[9])
            end
        end
        for (name, value, _, _, _) in rmv_structures()
            DE.store_mvtseries(db, name, value)
        end
    end
end

if action in ("generate-represented-mvtseries", "verify-represented-mvtseries", "verify-wheel")
    @testset "DataEcon represented MVTSeries and structure interchange" begin
        reference_fixture = action != "verify-wheel"
        DE.opendaec(filename) do db
            function raw_dated(id)
                ref = Ref{C.mvtseries_t}()
                @test C.de_load_mvtseries(db, id, ref) == 0
                a = ref[]
                header = (Int(a.object.obj_class), Int(a.object.obj_type), Int(a.eltype), Int(a.elfreq))
                ax1 = (Int(a.axis1.ax_type), Int(a.axis1.length), Int(a.axis1.frequency), Int(a.axis1.first))
                ax2 = (Int(a.axis2.ax_type), Int(a.axis2.length))
                names = a.axis2.names == C_NULL ? nothing : unsafe_string(a.axis2.names)
                payload = a.nbytes == 0 ? UInt8[] :
                    copy(unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(a.value), Int(a.nbytes)))
                attrs = Dict(string(k) => string(v) for (k, v) in DE.get_all_attributes(db, id))
                return header, ax1, ax2, names, payload, attrs
            end
            dense_bytes(value) = Vector{UInt8}(reinterpret(UInt8, value isa MVTSeries ? vec(value.values) : vec(Matrix(value))))
            for (name, value, expected, attrs, codes, raw) in rmv_inventory()
                id = DE.find_fullpath(db, "/" * name)
                header, ax1, ax2, names, payload, stored_attrs = raw_dated(id)
                @test header == (3, 21, codes[1], codes[2])
                @test ax1[1] == 1 && ax2[1] == 2
                if raw === nothing
                    @test ax1[2] == size(value, 1) && ax2[2] == size(value, 2)
                    @test ax1[3] == Int(DE.I._to_de_scalar_freq(frequencyof(value)))
                    @test ax1[4] == Int(firstdate(value))
                    @test names == join(string.(colnames(value)), "\n")
                    @test payload == dense_bytes(value)
                elseif !reference_fixture && name == "rmv_bool_on_int64_01"
                    # Python reads the foreign Int64 Bool row as a Boolean
                    # MVTSeries and rewrites it in the canonical Int8 encoding.
                    @test ax1[2] == raw[4] && ax1[3] == raw[5] && ax1[4] == raw[6]
                    @test names == raw[7]
                    @test payload == rmv_bytes(Int8[1, 0, 0, 1])
                else
                    @test ax1[2] == raw[4] && ax1[3] == raw[5] && ax1[4] == raw[6]
                    @test names == raw[7]
                    @test payload == raw[3]
                end
                @test stored_attrs == attrs
                loaded = try
                    DE.load_mvtseries(db, id)
                catch e
                    e
                end
                if expected === :same
                    @test isequal(loaded, value) && typeof(loaded) == typeof(value)
                elseif expected isa Type
                    @test loaded isa expected
                else
                    @test isequal(loaded, expected) && typeof(loaded) == typeof(expected)
                end
            end
            for (name, value, attrs, codes, ET) in rmv_structures()
                id = DE.find_fullpath(db, "/" * name)
                header, ax1, ax2, names, payload, stored_attrs = raw_dated(id)
                @test header == (3, 20, codes[1], codes[2])
                @test ax1[1] == 0 && ax2[1] == 0 && names === nothing
                @test (ax1[2], ax2[2]) == size(value)
                expected_bytes = ET === Int8 ? Vector{UInt8}(reinterpret(UInt8, Int8.(vec(Matrix(value))))) : dense_bytes(value)
                @test payload == expected_bytes
                if reference_fixture || !isempty(value)
                    @test stored_attrs == attrs
                else
                    # Python omits the redundant element token on an empty
                    # ordinary structure; the wrapper marker must be present.
                    @test stored_attrs["jtype"] == attrs["jtype"]
                end
                loaded = DE.load_mvtseries(db, id)
                @test isequal(loaded, value)
                @test nameof(typeof(loaded)) == nameof(typeof(value))
                # The stored bytes are the wrapper's dense form: re-materialising
                # the loaded wrapper reproduces them exactly.
                @test Vector{UInt8}(reinterpret(UInt8, vec(Matrix(loaded)))) == payload
            end
        end
    end
end

# ---- Marker-mapped, wide and exceptional scalars ------------------------------
# Reference rows for the scalar surface Python does not yet read: Int128, UInt128
# and ComplexF16 widths; Symbol, SubString and other string forms including
# invalid UTF-8 and embedded NUL; Date/DateTime; Rational{T}; integer Complex{T};
# Irrational; and foreign jtype markers injected on ordinary payloads through the
# C entry points. Julia's outcome for every row is materialised as siblings:
# "<name>_type" (the loaded type's name) and "<name>_value" (a canonical text of
# the loaded value) when the load succeeds, "<name>_julia" (Julia's own writer
# re-storing the loaded value) with "<name>_rewrite" ("<type>:<canonical text>" of
# what that re-stored object loads as, or the exception type's name: Julia's own
# rewrite is lossy for Bool, for integer components beyond 2^53 and for
# rationals whose float exceeds the parameter) when that writer can store it,
# and "<name>_error" (the exception type's name) when the load fails. Actions:
# generate-scalar-markers/verify-scalar-markers. verify-wheel does not check this
# set: Python writes none of these forms yet.
sm_le(x) = Vector{UInt8}(reinterpret(UInt8, [x]))
sm_cstr(s) = (b = Vector{UInt8}(codeunits(String(s))); push!(b, 0x00); b)
# Julia's own writer: (name, value).
function sm_written_cases()
    return Any[
        ("sm_int128_min", typemin(Int128)), ("sm_int128_max", typemax(Int128)), ("sm_int128_neg1", Int128(-1)),
        ("sm_uint128_max", typemax(UInt128)), ("sm_uint128_zero", UInt128(0)), ("sm_uint128_2p64", UInt128(2)^64),
        ("sm_cf16", ComplexF16(1.5, -2.25)), ("sm_cf16_nan", ComplexF16(NaN16, Inf16)), ("sm_cf16_negzero", ComplexF16(-0.0, 0.0)),
        ("sm_symbol", :alpha), ("sm_symbol_empty", Symbol("")), ("sm_symbol_multibyte", Symbol("é\U0001f642")),
        ("sm_symbol_invalid_utf8", Symbol(String([0x66, 0xff, 0x6f]))), ("sm_symbol_space", Symbol("a b")),
        ("sm_substring", SubString("hello world", 1, 5)), ("sm_generic_string", Test.GenericString("gen")),
        ("sm_string_invalid_utf8", String([0x66, 0xff, 0x6f])), ("sm_string_embedded_nul", "a\0b"), ("sm_string_only_nul", "\0"),
        ("sm_date", Date(2024, 3, 15)), ("sm_date_epoch", Date(1970, 1, 1)), ("sm_date_before_epoch", Date(1969, 12, 31)),
        ("sm_date_year0", Date(0, 1, 1)), ("sm_date_negative_year", Date(-1000, 6, 30)), ("sm_date_year1", Date(1, 1, 1)),
        ("sm_date_year9999", Date(9999, 12, 31)), ("sm_date_year10000", Date(10000, 1, 1)), ("sm_date_year_300k", Date(300000, 1, 1)),
        ("sm_datetime", DateTime(2024, 3, 15, 13, 45, 30)), ("sm_datetime_ms", DateTime(2024, 3, 15, 13, 45, 30, 123)),
        ("sm_datetime_ms_999", DateTime(2024, 3, 15, 23, 59, 59, 999)), ("sm_datetime_epoch", DateTime(1970, 1, 1)),
        ("sm_datetime_negative_ms", DateTime(1969, 12, 31, 23, 59, 59, 999)), ("sm_datetime_year0", DateTime(0, 1, 1, 0, 0, 0, 1)),
        ("sm_datetime_year9999", DateTime(9999, 12, 31, 23, 59, 59, 999)), ("sm_datetime_year10000", DateTime(10000, 1, 1, 0, 0, 0, 7)),
        # Beyond 2^53 milliseconds Julia's own Float64 storage loses the millisecond: written .005, reloaded .004.
        ("sm_datetime_year_300k", DateTime(300000, 6, 1, 12, 0, 0, 5)), ("sm_datetime_year_neg300k", DateTime(-300000, 6, 1, 12, 0, 0, 5)),
        ("sm_rational_half", 1 // 2), ("sm_rational_third", 1 // 3), ("sm_rational_neg_third", -1 // 3), ("sm_rational_dyadic", 3 // 8),
        ("sm_rational_int", 7 // 1), ("sm_rational_zero", 0 // 1), ("sm_rational_inf", 1 // 0), ("sm_rational_neginf", -1 // 0),
        ("sm_rational_big_num", (2^62) // 3), ("sm_rational_2p53", (2^53 + 1) // 1), ("sm_rational_int32", Rational{Int32}(1, 3)),
        ("sm_rational_int8", Rational{Int8}(3, 4)), ("sm_rational_uint8", Rational{UInt8}(3, 4)), ("sm_rational_int128", Rational{Int128}(1, 3)),
        ("sm_rational_int128_big", Rational{Int128}(Int128(2)^100, 3)), ("sm_rational_bool", Rational{Bool}(true, true)),
        ("sm_rational_int16_small", Rational{Int16}(1, 1024)),
        # Julia's Float64 partial quotients above 2^53 round: this reloads as 3//4611686018427387649, not 3//2^62.
        ("sm_rational_3_2p62", 3 // Int64(2)^62),
        ("sm_complex_int", Complex(1, 2)), ("sm_complex_int_neg", Complex(-3, 0)), ("sm_complex_int_zero", Complex(0, 0)),
        ("sm_complex_int8", Complex{Int8}(1, -2)), ("sm_complex_uint8", Complex{UInt8}(1, 2)),
        ("sm_complex_int128_big", Complex{Int128}(Int128(2)^70, 1)), ("sm_complex_int_2p53", Complex(2^53 + 1, 0)),
        ("sm_complex_bool", Complex(true, false)), ("sm_complex_int64_max", Complex(typemax(Int64), typemin(Int64))),
        ("sm_complex_rational", Complex(1 // 2, 1 // 3)), ("sm_complex_int_2p54", Complex(Int64(2)^54, 1)),
        ("sm_complex_int_3x2p61", Complex(3 * Int64(2)^61, -Int64(2)^60)), ("sm_complex_int128_2p100", Complex(Int128(2)^100, Int128(2)^70)),
        ("sm_complex_int128_2p126_neg", Complex(-Int128(2)^126, 0)), ("sm_complex_int8_127", Complex{Int8}(127, -128)),
        ("sm_irrational_pi", pi), ("sm_irrational_e", MathConstants.e),
    ]
end
# Raw C entry point stores with an injected marker: (name, kind, frequency, payload, jtype).
function sm_raw_cases()
    f64(x) = sm_le(Float64(x)); i64(x) = sm_le(Int64(x))
    return Any[
        ("sm_inj_date_on_int64", C.type_integer, 0, i64(1710460800), "Date"),
        ("sm_inj_datetime_on_float", C.type_float, 0, f64(1710510330.123), "DateTime"),
        ("sm_inj_dates_date_qualified", C.type_float, 0, f64(1710460800.0), "Dates.Date"),
        ("sm_inj_date_on_string", C.type_string, 0, sm_cstr("x"), "Date"),
        ("sm_inj_date_on_complex", C.type_complex, 0, sm_le(ComplexF64(1.0, 0.0)), "Date"),
        ("sm_inj_date_on_complex_imag", C.type_complex, 0, sm_le(ComplexF64(1.0, 2.0)), "Date"),
        ("sm_inj_datetime_string_payload", C.type_string, 0, sm_cstr("2024-01-01"), "DateTime"),
        ("sm_inj_rational_on_int64", C.type_integer, 0, i64(7), "Rational{Int64}"),
        ("sm_inj_rational_on_float_nan", C.type_float, 0, f64(NaN), "Rational{Int64}"),
        ("sm_inj_rational_on_float_huge", C.type_float, 0, f64(1e300), "Rational{Int64}"),
        ("sm_inj_rational_int8_on_float", C.type_float, 0, f64(0.5), "Rational{Int8}"),
        ("sm_inj_rational_int8_overflow", C.type_float, 0, f64(1000.0), "Rational{Int8}"),
        ("sm_inj_rational_int8_on_int8_min", C.type_integer, 0, UInt8[0x80], "Rational{Int8}"),
        ("sm_inj_rational_int8_on_int64_min", C.type_integer, 0, i64(-128), "Rational{Int8}"),
        ("sm_inj_rational_bare", C.type_float, 0, f64(0.5), "Rational"),
        ("sm_inj_rat64_2p_neg100", C.type_float, 0, f64(2.0^-100), "Rational{Int64}"),
        ("sm_inj_rat64_2p_neg62", C.type_float, 0, f64(2.0^-62), "Rational{Int64}"),
        ("sm_inj_rat64_2p_neg63", C.type_float, 0, f64(2.0^-63), "Rational{Int64}"),
        ("sm_inj_rat64_2p54", C.type_float, 0, f64(2.0^54), "Rational{Int64}"),
        ("sm_inj_rat64_2p63", C.type_float, 0, f64(2.0^63), "Rational{Int64}"),
        ("sm_inj_rat64_neg2p63", C.type_float, 0, f64(-2.0^63), "Rational{Int64}"),
        ("sm_inj_rat64_neg2p63_edge", C.type_float, 0, f64(-(2.0^63 - 1024)), "Rational{Int64}"),
        ("sm_inj_rat64_subnormal", C.type_float, 0, f64(2.0^-1074), "Rational{Int64}"),
        ("sm_inj_rat64_tenth", C.type_float, 0, f64(0.1), "Rational{Int64}"),
        ("sm_inj_rat64_3_2p62", C.type_float, 0, f64(3 * 2.0^-62), "Rational{Int64}"),
        ("sm_inj_rat8_1_128", C.type_float, 0, f64(1 / 128), "Rational{Int8}"),
        ("sm_inj_rat8_1_64", C.type_float, 0, f64(1 / 64), "Rational{Int8}"),
        ("sm_inj_rat8_tenth", C.type_float, 0, f64(0.1), "Rational{Int8}"),
        ("sm_inj_rat8_127", C.type_float, 0, f64(127.0), "Rational{Int8}"),
        ("sm_inj_rat8_128", C.type_float, 0, f64(128.0), "Rational{Int8}"),
        ("sm_inj_rat8_neg128", C.type_float, 0, f64(-128.0), "Rational{Int8}"),
        ("sm_inj_rat8_neg127", C.type_float, 0, f64(-127.0), "Rational{Int8}"),
        ("sm_inj_ratu8_neg_half", C.type_float, 0, f64(-0.5), "Rational{UInt8}"),
        ("sm_inj_ratu8_half", C.type_float, 0, f64(0.5), "Rational{UInt8}"),
        ("sm_inj_rat128_2p_neg100", C.type_float, 0, f64(2.0^-100), "Rational{Int128}"),
        ("sm_inj_rat128_2p_neg127", C.type_float, 0, f64(2.0^-127), "Rational{Int128}"),
        ("sm_inj_rat128_2p126", C.type_float, 0, f64(2.0^126), "Rational{Int128}"),
        ("sm_inj_rat128_tiny", C.type_float, 0, f64(2.7141251072346146e-36), "Rational{Int128}"),
        ("sm_inj_rat64_on_int64_max", C.type_integer, 0, i64(typemax(Int64)), "Rational{Int64}"),
        ("sm_inj_rat64_on_int128_big", C.type_integer, 0, sm_le(Int128(2)^70), "Rational{Int64}"),
        ("sm_inj_complex_int_on_float", C.type_float, 0, f64(2.0), "Complex{Int64}"),
        ("sm_inj_complex_int_on_float_frac", C.type_float, 0, f64(1.5), "Complex{Int64}"),
        ("sm_inj_complex_on_string", C.type_string, 0, sm_cstr("1+2im"), "Complex{Int64}"),
        ("sm_inj_wide_complex_int128", C.type_integer, 0, sm_le(Int128(5)), "Complex{Int64}"),
        ("sm_inj_cint_on_int64_2p53p1", C.type_integer, 0, i64(9007199254740993), "Complex{Int64}"),
        ("sm_inj_cint_on_float_2p53p1", C.type_float, 0, f64(9007199254740993.0), "Complex{Int64}"),
        ("sm_inj_cint_on_int128_2p70", C.type_integer, 0, sm_le(Int128(2)^70), "Complex{Int64}"),
        ("sm_inj_cint128_on_int128_2p70", C.type_integer, 0, sm_le(Int128(2)^70), "Complex{Int128}"),
        ("sm_inj_cint_on_complex_2p53p1", C.type_complex, 0, sm_le(ComplexF64(9007199254740993.0, 1.0)), "Complex{Int64}"),
        ("sm_inj_cbool_on_float_two", C.type_float, 0, f64(2.0), "Complex{Bool}"),
        ("sm_inj_symbol_on_float", C.type_float, 0, f64(1.5), "Symbol"),
        ("sm_inj_symbol_on_int64", C.type_integer, 0, i64(3), "Symbol"),
        ("sm_inj_symbol_on_date_kind", C.type_date, 32, i64(24288), "Symbol"),
        ("sm_inj_string_nul_symbol", C.type_string, 0, UInt8[0x61, 0x00, 0x62, 0x00], "Symbol"),
        ("sm_inj_string_on_float", C.type_float, 0, f64(1.5), "String"),
        ("sm_inj_substring_on_string", C.type_string, 0, sm_cstr("abc"), "SubString{String}"),
        ("sm_inj_abstractstring_on_string", C.type_string, 0, sm_cstr("abc"), "AbstractString"),
        ("sm_inj_any_on_string", C.type_string, 0, sm_cstr("abc"), "Any"),
        ("sm_inj_char_on_string", C.type_string, 0, sm_cstr("a"), "Char"),
        ("sm_inj_int64_on_float_exact", C.type_float, 0, f64(2.0), "Int64"),
        ("sm_inj_int64_on_float_frac", C.type_float, 0, f64(1.5), "Int64"),
        ("sm_inj_float32_on_float", C.type_float, 0, f64(1.5), "Float32"),
        ("sm_inj_float64_on_int64", C.type_integer, 0, i64(3), "Float64"),
        ("sm_inj_int128_on_int64", C.type_integer, 0, i64(-3), "Int128"),
        ("sm_inj_uint8_on_int64_neg", C.type_integer, 0, i64(-3), "UInt8"),
        ("sm_inj_bool_on_int8_one", C.type_integer, 0, UInt8[0x01], "Bool"),
        ("sm_inj_bool_on_int8_two", C.type_integer, 0, UInt8[0x02], "Bool"),
        ("sm_inj_mit_monthly_on_int64", C.type_integer, 0, i64(24288), "MIT{Monthly}"),
        ("sm_inj_duration_on_int64", C.type_integer, 0, i64(5), "Duration{Monthly}"),
        ("sm_inj_mit_on_date_kind", C.type_date, 32, i64(24288), "Int64"),
        ("sm_inj_bigint_on_int64", C.type_integer, 0, i64(typemax(Int64)), "BigInt"),
        ("sm_inj_bigfloat_on_float", C.type_float, 0, f64(0.1), "BigFloat"),
        ("sm_inj_irrational_on_float", C.type_float, 0, f64(Float64(pi)), "Irrational{:π}"),
        ("sm_inj_unknown", C.type_float, 0, f64(1.5), "NoSuchType"),
        ("sm_inj_evaluated", C.type_float, 0, f64(1.5), "error(\"evaluated\")"),
        ("sm_inj_base_int64", C.type_float, 0, f64(2.0), "Base.Int64"),
        ("sm_inj_spaced", C.type_float, 0, f64(2.0), " Int64 "),
        ("sm_inj_vector_int64", C.type_integer, 0, i64(2), "Vector{Int64}"),
    ]
end
# A canonical, exact text of a loaded scalar (integers and rationals in decimal,
# floats and complex floats as little-endian hex, calendar values by component).
sm_hex(x) = bytes2hex(sm_le(x))
sm_text(x::Integer) = string(BigInt(x))
sm_text(x::Rational) = string(BigInt(numerator(x)), "//", BigInt(denominator(x)))
sm_text(x::Complex{<:Union{Integer,Rational}}) = string(sm_text(real(x)), ",", sm_text(imag(x)))
sm_text(x::Union{Float16,Float32,Float64,ComplexF16,ComplexF32,ComplexF64}) = sm_hex(x)
sm_text(x::BigFloat) = string(x)
sm_text(x::Date) = string(Dates.year(x), "-", Dates.month(x), "-", Dates.day(x))
sm_text(x::DateTime) = string(sm_text(Date(x)), "T", Dates.hour(x), ":", Dates.minute(x), ":", Dates.second(x), ".", Dates.millisecond(x))
sm_text(x::Symbol) = String(x)
sm_text(x::AbstractString) = String(x)
sm_text(x::Union{MIT,Duration}) = string(x)
# Julia's writer recurses without bound on BigInt/BigFloat values, so those loaded
# values are recorded as text only.
sm_writable(x) = !(x isa Union{BigInt,BigFloat})

function sm_raw_scalar(db, id)
    ref = Ref{C.scalar_t}()
    @test C.de_load_scalar(db, id, ref) == 0
    s = ref[]
    payload = s.nbytes == 0 ? UInt8[] : copy(unsafe_wrap(Vector{UInt8}, Ptr{UInt8}(s.value), Int(s.nbytes)))
    attrs = Dict(string(k) => string(v) for (k, v) in DE.get_all_attributes(db, id))
    return (Int(s.object.obj_class), Int(s.object.obj_type), Int(s.frequency), Int(s.nbytes)), payload, attrs
end

function sm_materialize!(db, name, id)
    loaded = try
        DE.load_scalar(db, id)
    catch e
        e
    end
    if loaded isa Exception
        DE.store_scalar(db, DE.root_id, "$(name)_error", string(nameof(typeof(loaded))))
    else
        DE.store_scalar(db, DE.root_id, "$(name)_type", string(typeof(loaded)))
        DE.store_scalar(db, DE.root_id, "$(name)_value", sm_text(loaded))
        if sm_writable(loaded)
            julia_id = DE.store_scalar(db, DE.root_id, "$(name)_julia", loaded)
            DE.store_scalar(db, DE.root_id, "$(name)_rewrite", sm_outcome_text(db, julia_id))
        end
    end
    return loaded
end
function sm_outcome_text(db, id)
    loaded = try
        DE.load_scalar(db, id)
    catch e
        e
    end
    return loaded isa Exception ? string(nameof(typeof(loaded))) : string(typeof(loaded), ":", sm_text(loaded))
end

if action == "generate-scalar-markers"
    ispath(filename) && error("Output already exists; use a fresh fixture path.")
    DE.opendaec(filename; write=true) do db
        for (name, value) in sm_written_cases()
            id = DE.store_scalar(db, DE.root_id, name, value)
            sm_materialize!(db, name, id)
        end
        for (name, kind, freq, payload, jtype) in sm_raw_cases()
            id = Ref{C.obj_id_t}()
            rc = GC.@preserve payload begin
                C.de_store_scalar(db, DE.root_id, name, kind, C.frequency_t(Integer(freq)), length(payload),
                                  isempty(payload) ? C_NULL : pointer(payload), id)
            end
            @test rc == 0
            DE.set_attribute(db, id[], "jtype", jtype)
            sm_materialize!(db, name, id[])
        end
    end
end

if action in ("generate-scalar-markers", "verify-scalar-markers")
    @testset "DataEcon marker-mapped, wide and exceptional scalars" begin
        DE.opendaec(filename) do db
            function verify_siblings(name, loaded)
                error_id = DE.find_object(db, DE.root_id, "$(name)_error", false)
                type_id = DE.find_object(db, DE.root_id, "$(name)_type", false)
                value_id = DE.find_object(db, DE.root_id, "$(name)_value", false)
                julia_id = DE.find_object(db, DE.root_id, "$(name)_julia", false)
                rewrite_id = DE.find_object(db, DE.root_id, "$(name)_rewrite", false)
                if loaded isa Exception
                    @test error_id !== missing && type_id === missing && value_id === missing
                    @test julia_id === missing && rewrite_id === missing
                    error_id === missing || @test DE.load_scalar(db, error_id) == string(nameof(typeof(loaded)))
                else
                    @test error_id === missing && type_id !== missing && value_id !== missing
                    type_id === missing || @test DE.load_scalar(db, type_id) == string(typeof(loaded))
                    value_id === missing || @test DE.load_scalar(db, value_id) == sm_text(loaded)
                    if sm_writable(loaded)
                        @test julia_id !== missing && rewrite_id !== missing
                        if julia_id !== missing && rewrite_id !== missing
                            # The recorded rewrite outcome is what Julia's re-stored object loads as now.
                            @test DE.load_scalar(db, rewrite_id) == sm_outcome_text(db, julia_id)
                        end
                    else
                        @test julia_id === missing && rewrite_id === missing
                    end
                end
            end
            for (name, value) in sm_written_cases()
                id = DE.find_object(db, DE.root_id, name)
                header, payload, attrs = sm_raw_scalar(db, id)
                @test header[1] == 1
                loaded = try
                    DE.load_scalar(db, id)
                catch e
                    e
                end
                verify_siblings(name, loaded)
                # Recorded facts about Julia's own writer and loader.
                if value isa Union{Int128,UInt128}
                    @test header[4] == 16 && payload == sm_le(value) && !haskey(attrs, "jtype") && loaded == value
                elseif value isa ComplexF16
                    @test header == (1, 5, 0, 4) && payload == sm_le(value) && !haskey(attrs, "jtype")
                elseif value isa Symbol
                    @test header[2] == 6 && attrs == Dict("jtype" => "Symbol") && loaded == value
                elseif value isa Union{Date,DateTime}
                    @test header == (1, 4, 0, 8) && attrs == Dict("jtype" => string(nameof(typeof(value))))
                elseif value isa Rational || value isa Complex{<:Integer} || value isa Complex{<:Rational} || value isa Irrational
                    @test header[2] == (value isa Complex ? 5 : 4) && attrs == Dict("jtype" => string(typeof(value)))
                end
            end
            @test DE.load_scalar(db, DE.find_object(db, DE.root_id, "sm_rational_int32")) == Rational{Int32}(1, 3)
            @test DE.load_scalar(db, DE.find_object(db, DE.root_id, "sm_rational_3_2p62")) == 3 // 4611686018427387649
            @test DE.load_scalar(db, DE.find_object(db, DE.root_id, "sm_datetime_year_300k")) == DateTime(300000, 6, 1, 12, 0, 0, 4)
            @test DE.load_scalar(db, DE.find_object(db, DE.root_id, "sm_string_embedded_nul")) == "a"
            @test DE.load_scalar(db, DE.find_object(db, DE.root_id, "sm_inj_rational_int8_on_int8_min")) == Rational{Int8}(-128, 1)
            @test DE.load_scalar(db, DE.find_object(db, DE.root_id, "sm_inj_rat8_tenth")) == Rational{Int8}(1, 10)
            @test DE.load_scalar(db, DE.find_object(db, DE.root_id, "sm_inj_rat128_tiny")) == 408 // Int128(150324684338411231602781414279658602777)
            for (name, kind, freq, payload, jtype) in sm_raw_cases()
                id = DE.find_object(db, DE.root_id, name)
                header, stored, attrs = sm_raw_scalar(db, id)
                @test header == (1, Int(kind), freq, length(payload)) && stored == payload
                @test attrs == Dict("jtype" => jtype)
                loaded = try
                    DE.load_scalar(db, id)
                catch e
                    e
                end
                verify_siblings(name, loaded)
            end
        end
    end
end

if action in ("generate", "generate-empty", "generate-scalars", "generate-quarterly", "generate-annual", "generate-halfyearly", "generate-int64", "generate-strings", "generate-dates", "generate-calendar", "generate-widths", "generate-fileops", "generate-calendar-series", "generate-series-elements", "generate-represented-elements", "generate-foreign-markers", "generate-arrays-unit", "generate-matrices-text", "generate-tensors", "generate-catalogs", "generate-workspace", "generate-text-arrays", "generate-represented-mvtseries", "generate-scalar-markers")
    layout = Dict{String,Any}(
        "enums" => sizeof.([C.class_t, C.type_t, C.frequency_t, C.axis_type_t]),
    )
    for T in (C.object_t, C.axis_t, C.tseries_t, C.mvtseries_t, C.ndtseries_t, C.scalar_t)
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
