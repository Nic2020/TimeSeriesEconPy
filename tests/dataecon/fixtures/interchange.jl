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
elseif !(action in ("verify", "verify-empty", "verify-scalars", "verify-quarterly", "verify-annual", "verify-halfyearly", "verify-int64", "verify-strings", "verify-dates", "verify-wheel"))
    error("Unknown action; use generate/verify, generate-empty/verify-empty, generate-scalars/verify-scalars, generate-quarterly/verify-quarterly, generate-annual/verify-annual, generate-halfyearly/verify-halfyearly, generate-int64/verify-int64, generate-strings/verify-strings, generate-dates/verify-dates or verify-wheel.")
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

if action in ("generate", "generate-empty", "generate-scalars", "generate-quarterly", "generate-annual", "generate-halfyearly", "generate-int64", "generate-strings", "generate-dates")
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
