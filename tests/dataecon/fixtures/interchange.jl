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
elseif !(action in ("verify", "verify-empty", "verify-scalars", "verify-quarterly", "verify-annual", "verify-halfyearly", "verify-int64", "verify-strings", "verify-dates", "verify-calendar", "verify-widths", "verify-fileops", "verify-calendar-series", "verify-wheel"))
    error("Unknown action; use generate/verify, generate-empty/verify-empty, generate-scalars/verify-scalars, generate-quarterly/verify-quarterly, generate-annual/verify-annual, generate-halfyearly/verify-halfyearly, generate-int64/verify-int64, generate-strings/verify-strings, generate-dates/verify-dates, generate-calendar/verify-calendar, generate-widths/verify-widths, generate-fileops/verify-fileops, generate-calendar-series/verify-calendar-series or verify-wheel.")
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

if action in ("generate", "generate-empty", "generate-scalars", "generate-quarterly", "generate-annual", "generate-halfyearly", "generate-int64", "generate-strings", "generate-dates", "generate-calendar", "generate-widths", "generate-fileops", "generate-calendar-series")
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
