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
using TimeSeriesEcon
using Test, SHA, TOML, Pkg, Dates

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
elseif !(action in ("verify", "verify-empty", "verify-scalars", "verify-quarterly", "verify-annual", "verify-halfyearly", "verify-int64", "verify-strings", "verify-dates", "verify-calendar", "verify-widths", "verify-fileops", "verify-calendar-series", "verify-series-elements", "generate-series-elements", "verify-represented-elements", "verify-wheel"))
    error("Unknown action; use generate/verify, generate-empty/verify-empty, generate-scalars/verify-scalars, generate-quarterly/verify-quarterly, generate-annual/verify-annual, generate-halfyearly/verify-halfyearly, generate-int64/verify-int64, generate-strings/verify-strings, generate-dates/verify-dates, generate-calendar/verify-calendar, generate-widths/verify-widths, generate-fileops/verify-fileops, generate-calendar-series/verify-calendar-series, generate-series-elements/verify-series-elements, generate-represented-elements/verify-represented-elements or verify-wheel.")
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

if action in ("generate", "generate-empty", "generate-scalars", "generate-quarterly", "generate-annual", "generate-halfyearly", "generate-int64", "generate-strings", "generate-dates", "generate-calendar", "generate-widths", "generate-fileops", "generate-calendar-series", "generate-series-elements", "generate-represented-elements")
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
