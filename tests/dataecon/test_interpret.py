"""Finite marker interpretation: tokens, precedence, routes, value rules and kernels.

Pure NumPy tests (no native build). Expected conversion results are the pinned
Julia reference's own loads, spelled out as recorded bytes or values; the
kernels must reproduce Julia's routes (Float16 through Float32 for integers
up to 64 bits, Float16 through Float64 for 128-bit integers, nearest-even
rounding, plotting values for year/period dates and durations) rather than a
generic NumPy cast.
"""

import math

import numpy as np
import pytest

from tsecon import MIT, Duration, TSeries, mm
from tsecon.dataecon import COMPLEXF16, INT128, UINT128, StoredElement, StoredSeries
from tsecon.dataecon import _interpret as interp
from tsecon.dataecon._codec import MAX_BYTES, decode_series, encode_series
from tsecon.frequencies import BDaily, Daily, HalfYearly, Monthly, Quarterly, Unit, Weekly, Yearly

ANCHOR = mm(2024, 11)
I8, I16, I32, I64 = (np.dtype(f"<i{n}") for n in (1, 2, 4, 8))
U8, U16, U32, U64 = (np.dtype(f"<u{n}") for n in (1, 2, 4, 8))
F16, F32, F64 = (np.dtype(f"<f{n}") for n in (2, 4, 8))
C64, C128 = np.dtype("<c8"), np.dtype("<c16")
BOOL = np.dtype("?")
MONTHLY_CODES = [-(1 << 63), (1 << 63) - 1, 2**54 + 2**30 + 1, -1, -13, -12]


def target(spec):
    """A Target from a dtype, a StoredElement or a token."""
    if isinstance(spec, StoredElement):
        return spec.target
    if isinstance(spec, str):
        resolved = interp.resolve_token(spec)
        assert resolved is not None, spec
        return resolved
    return interp.numeric_target(np.dtype(spec))


def words(*items):
    return interp.pack_words(list(items))


def bits(array, dtype):
    return np.asarray(array).view(np.dtype(f"<u{np.dtype(dtype).itemsize}")).tolist()


def numeric(dtype, values, marker):
    return StoredSeries(ANCHOR, np.array(values, dtype=dtype), StoredElement.numeric(dtype, marker))


# ---- tokens and precedence ------------------------------------------------


class TestTokens:
    def test_finite_vocabulary(self):
        assert len(interp.ACTIVE_TOKENS) == 14 + 3 + 64 + 5
        for alias, name in interp.ALIASES.items():
            assert interp.resolve_token(alias) == interp.resolve_token(name)
        assert interp.resolve_token("MIT{Weekly{7}}") == interp.date_target("date", Weekly(7))
        assert interp.resolve_token("Duration{Unit}") == interp.date_target("duration", Unit())
        assert interp.resolve_token("Int128") == interp.wide_target("int128")
        assert interp.resolve_token("Bool").is_bool
        assert interp.resolve_token("Int64").julia_name == "Int64"

    @pytest.mark.parametrize(
        "text",
        [
            "",
            " Int64",
            "Int64 ",
            "Base.Int64",
            "(Int64)",
            "Int64;",
            "MIT{Quarterly}",
            "MIT{ Monthly}",
            "Rational{Int64}",
            "Complex{Int64}",
            "Complex{Float128}",
            "Date",
            "Symbol",
            "Any",
            "Real",
            "Vector{Int64}",
            'error("x")',
            "1 + 1",
            "NoSuchType",
        ],
    )
    def test_text_outside_the_table_is_never_evaluated(self, text):
        assert interp.resolve_token(text) is None

    def test_marker_text_rules(self):
        interp.check_marker_text(None, "element")
        interp.check_marker_text("", "element")
        with pytest.raises(TypeError):
            interp.check_marker_text(b"Int64", "element")
        with pytest.raises(ValueError):
            interp.check_marker_text("Int64\0", "element")
        with pytest.raises(ValueError):
            StoredElement.numeric(I16, marker="Int64\0")


class TestObjectPrecedence:
    def test_identity_spellings_match_axis_and_base(self):
        base = target(I16)
        for token in (
            "TSeries",
            " TSeries",
            "TimeSeriesEcon.TSeries",
            "AbstractVector",
            "Any",
            "TSeries{Monthly}",
            "TSeries{Monthly, Int16}",
            "TSeries{Monthly,Int16}",
            "TSeries{Monthly, Int16, Vector{Int16}}",
        ):
            assert interp.object_interpretation(token, Monthly(), base, 2) == "identity"
        wide = target(INT128)
        assert (
            interp.object_interpretation("TSeries{Daily, Int128, Vector{Int128}}", Daily(), wide, 1)
            == "identity"
        )
        dated = target(StoredElement.date(Yearly(12)))
        assert (
            interp.object_interpretation("TSeries{Daily, MIT{Yearly{12}}}", Daily(), dated, 1)
            == "identity"
        )

    @pytest.mark.parametrize(
        "token",
        [
            "TSeries{Monthly, Int64}",
            "TSeries{Quarterly{3}, Int16}",
            "TSeries{Quarterly{3}}",
            "TSeries{Monthly, Int16, Vector{Float64}}",
            "TSeries{Monthly,Int16,Vector{Int16}}",
            "TSeries ",
            "Base.TSeries",
            "AbstractArray",
            "Real",
            "MVTSeries",
            "Symbol",
            "",
        ],
    )
    def test_other_object_tokens_are_refused_without_fallback(self, token):
        with pytest.raises(TypeError):
            interp.object_interpretation(token, Monthly(), target(I16), 2)

    def test_vector_tokens_apply_to_empty_numeric_bases_only(self):
        assert interp.object_interpretation("Vector", Monthly(), target(I64), 0) == "vector"
        assert (
            interp.object_interpretation("Vector{Float64}", Monthly(), target(C128), 0) == "vector"
        )
        assert interp.vector_dtype("Vector", target(C128)) == C128
        assert interp.vector_dtype("Vector{Float64}", target(C128)) == F64
        # Every exact element name in the four verified spellings, Bool and the
        # wide carriers included, plus the bare Array and Vector{Any}.
        for element, dtype in (("Int8", np.dtype("i1")), ("Bool", np.dtype("?"))):
            for token in (
                f"Vector{{{element}}}",
                f"Array{{{element}}}",
                f"Array{{{element},1}}",
                f"Array{{{element}, 1}}",
            ):
                assert interp.object_interpretation(token, Monthly(), target(I64), 0) == "vector"
                assert interp.vector_dtype(token, target(I64)) == dtype
        assert interp.vector_dtype("Vector{Int128}", target(I64)) == INT128.dtype
        assert interp.vector_dtype("Vector{ComplexF16}", target(I64)) == COMPLEXF16.dtype
        assert interp.vector_dtype("Array", target(INT128)) == INT128.dtype
        assert interp.vector_dtype("Vector{Any}", target(I64)) == np.dtype(object)
        assert interp.object_interpretation("Vector", Monthly(), target(INT128), 0) == "vector"
        with pytest.raises(ValueError, match="empty payload only"):
            interp.object_interpretation("Vector", Monthly(), target(I64), 1)
        with pytest.raises(ValueError, match="empty payload only"):
            interp.object_interpretation("Array{Int8, 1}", Monthly(), target(I64), 3)
        with pytest.raises(TypeError):
            interp.object_interpretation(
                "Vector", Monthly(), target(StoredElement.date(Monthly())), 0
            )
        for token in ("Vector{Int}", "Vector{Complex{Float16}}", "Array{Int8,2}", "Vector{Real}"):
            with pytest.raises(TypeError):
                interp.object_interpretation(token, Monthly(), target(I64), 0)


# ---- routes ---------------------------------------------------------------


class TestRoutes:
    @pytest.mark.parametrize(
        ("source", "token"),
        [
            (I8, "Duration{Monthly}"),
            (U64, "Duration{Daily}"),
            (INT128, "Duration{Monthly}"),
            (F64, "MIT{Monthly}"),
            (C128, "MIT{Monthly}"),
            (COMPLEXF16, "MIT{Daily}"),
            (F16, "Duration{Monthly}"),
            (StoredElement.date(Monthly()), "MIT{Quarterly{3}}"),
            (StoredElement.date(Monthly()), "Duration{Monthly}"),
            (StoredElement.duration(Monthly()), "MIT{Monthly}"),
            (StoredElement.date(Monthly()), "Int8"),
            (StoredElement.date(Daily()), "Int32"),
            (StoredElement.duration(Unit()), "UInt64"),
            (StoredElement.date(Monthly()), "Int128"),
            (StoredElement.date(Monthly()), "UInt128"),
            (StoredElement.date(Monthly()), "Float16"),
            (StoredElement.duration(Weekly(7)), "ComplexF16"),
        ],
    )
    def test_routes_julia_has_no_method_for(self, source, token):
        with pytest.raises(TypeError):
            interp.check_route(target(source), target(token))

    @pytest.mark.parametrize(
        ("source", "token"),
        [
            (I8, "MIT{Monthly}"),
            (U8, "MIT{Daily}"),
            (U64, "MIT{Monthly}"),
            (INT128, "MIT{Yearly{12}}"),
            (UINT128, "MIT{Unit}"),
            (I64, "Duration{Monthly}"),
            (I64, "MIT{Weekly{3}}"),
            (StoredElement.date(Monthly()), "Bool"),
            (StoredElement.date(Monthly()), "Int64"),
            (StoredElement.date(Monthly()), "Int"),
            (StoredElement.duration(Daily()), "Float32"),
            (StoredElement.date(Quarterly(3)), "Float64"),
            (StoredElement.date(HalfYearly(6)), "ComplexF32"),
            (StoredElement.duration(BDaily()), "ComplexF64"),
            (F64, "ComplexF16"),
            (C128, "Int8"),
            (COMPLEXF16, "Float64"),
            (INT128, "UInt128"),
            (UINT128, "Float16"),
            (I16, "Int64"),
        ],
    )
    def test_routes_julia_converts(self, source, token):
        interp.check_route(target(source), target(token))


# ---- value rules ----------------------------------------------------------


class TestValueRules:
    @pytest.mark.parametrize(
        ("dtype", "values", "token"),
        [
            (I64, [128], "Int8"),
            (I64, [-129], "Int8"),
            (I64, [256], "UInt8"),
            (I64, [-1], "UInt8"),
            (I64, [-1], "UInt64"),
            (U64, [1 << 63], "Int64"),
            (F64, [2.5], "Int8"),
            (F64, [math.nan], "Int64"),
            (F64, [math.inf], "Int64"),
            (F64, [2.0**63], "Int64"),
            (F64, [-(2.0**63) - 2048], "Int64"),
            (F64, [2.0**127], "Int128"),
            (F64, [-1.0], "UInt128"),
            (F64, [2.0**128], "UInt128"),
            (F16, [65504.0], "Int16"),
            (F16, [0.5], "Int16"),
            (C128, [1 + 1j], "Float64"),
            (C128, [1.5 + 0j], "Int64"),
            (C128, [2 + 0j], "Bool"),
            (C128, [0 + 1j], "Bool"),
            (F64, [math.nan], "Bool"),
            (F64, [2.0], "Bool"),
            (I16, [2], "Bool"),
            (U64, [1 << 63], "MIT{Monthly}"),
        ],
    )
    def test_inexact_or_out_of_range_values_are_refused_without_allocation(
        self, dtype, values, token
    ):
        array = np.array(values, dtype=dtype)
        with pytest.raises(ValueError, match="observation 0"):
            interp.check_values(array, target(dtype), target(token))

    @pytest.mark.parametrize(
        ("element", "items", "token"),
        [
            (INT128, [1 << 63], "Int64"),
            (INT128, [-(1 << 63) - 1], "Int64"),
            (INT128, [-1], "UInt64"),
            (INT128, [-1], "UInt128"),
            (INT128, [1 << 63], "MIT{Monthly}"),
            (UINT128, [1 << 63], "Int64"),
            (UINT128, [1 << 127], "Int128"),
            (UINT128, [1 << 64], "Bool"),
            (INT128, [128], "Int8"),
            (INT128, [-129], "Int8"),
            (COMPLEXF16, [complex(1.5, 0)], "Int64"),
            (COMPLEXF16, [complex(1, 1)], "Float64"),
        ],
    )
    def test_wide_value_rules(self, element, items, token):
        series = StoredSeries.from_list(ANCHOR, element, items)
        with pytest.raises(ValueError, match="observation 0"):
            interp.check_values(series.values, target(element), target(token))

    @pytest.mark.parametrize(
        ("dtype", "values", "token"),
        [
            (I64, [-128, 127], "Int8"),
            (I64, [255, 0], "UInt8"),
            (I64, [(1 << 63) - 1], "UInt64"),
            (U64, [(1 << 63) - 1, 0], "Int64"),
            (F64, [-0.0, 0.0, 1.0, -(2.0**63), 2.0**63 - 1024], "Int64"),
            (F64, [-(2.0**127), 2.0**100, -0.0], "Int128"),
            (F64, [2.0**64, 2.0**127, 0.0], "UInt128"),
            (F16, [65504.0], "Int32"),
            (C128, [complex(1.0, -0.0), complex(0.0, -0.0)], "Int8"),
            (C128, [complex(1.0, -0.0), complex(0.0, -0.0)], "Bool"),
            (F64, [-0.0, 1.0], "Bool"),
            (U64, [(1 << 63) - 1], "MIT{Monthly}"),
            (I8, [-1, -128], "MIT{Monthly}"),
        ],
    )
    def test_exact_values_are_accepted(self, dtype, values, token):
        interp.check_values(np.array(values, dtype=dtype), target(dtype), target(token))

    def test_dated_bool_and_int64_rules(self):
        date = StoredElement.date(Monthly())
        interp.check_values(np.array([0, 1], dtype="<i8"), target(date), target("Bool"))
        with pytest.raises(ValueError, match="zero or one"):
            interp.check_values(np.array([2], dtype="<i8"), target(date), target("Bool"))
        with pytest.raises(ValueError):
            interp.check_values(
                np.array([-1], dtype="<i8"), target(StoredElement.duration(Daily())), target("Bool")
            )
        codes = np.array([-(1 << 63), (1 << 63) - 1], dtype="<i8")
        interp.check_values(codes, target(date), target("Int64"))

    def test_output_capacity_is_checked_before_allocation(self):
        interp.check_output_capacity(MAX_BYTES // 16, target(INT128))
        with pytest.raises(ValueError, match="nothing was allocated"):
            interp.check_output_capacity(MAX_BYTES // 16 + 1, target(INT128))
        with pytest.raises(ValueError):
            interp.check_output_capacity(MAX_BYTES, target(F64))


# ---- kernels against the recorded Julia loads -----------------------------


class TestRounding:
    def test_round_to_precision_nearest_even(self):
        r = interp.round_to_precision
        assert r(0, 24) == 0.0
        assert r(2**54 + 2**30 + 1, 24) == 2.0**54 + 2.0**31
        assert r(2**54 + 2**30 - 1, 24) == 2.0**54
        assert r(2**54 + 2**30, 24) == 2.0**54  # tie to even
        assert r(2**54 + 3 * 2**30, 24) == 2.0**54 + 2.0**32  # tie to even (odd neighbour up)
        assert r(-(2**54 + 2**30 + 1), 24) == -(2.0**54 + 2.0**31)
        assert r(2**127 - 1, 24) == 2.0**127
        assert r(2**128 - 1, 24) == 2.0**128
        assert r(2**100 + 2**76 + 1, 24) == 2.0**100 + 2.0**77
        assert r(2**100 + 2**76 - 1, 24) == 2.0**100

    def test_int64_midpoints_to_float32_match_julia(self):
        # Naive conversion through Python float collapses both to 2**54.
        values = np.array([2**54 + 2**30 + 1, 2**54 + 2**30 - 1], dtype="<i8")
        out = interp.convert_values(values, target(I64), target(F32))
        assert bits(out, F32) == [0x5A800001, 0x5A800000]
        halves = interp.convert_values(values, target(I64), target(F16))
        assert bits(halves, F16) == [0x7C00, 0x7C00]

    def test_uint64_above_2_63_rounds_exactly(self):
        values = np.array(
            [2**63 + 2**10 + 1, 2**63 + 2**10 - 1, 2**63 + 2**39 + 2**38 + 1, 2**64 - 1, 2**63],
            dtype="<u8",
        )
        assert bits(interp.convert_values(values, target(U64), target(F64)), F64) == [
            0x43E0000000000001,
            0x43E0000000000000,
            0x43E0000018000000,
            0x43F0000000000000,
            0x43E0000000000000,
        ]
        assert bits(interp.convert_values(values, target(U64), target(F32)), F32) == [
            0x5F000000,
            0x5F000000,
            0x5F000001,
            0x5F800000,
            0x5F000000,
        ]
        assert bits(interp.convert_values(values, target(U64), target(F16)), F16) == [0x7C00] * 5

    def test_int128_midpoints_use_the_dedicated_route(self):
        values = words(2**100 + 2**76 + 1, 2**100 + 2**76 - 1)
        out = interp.convert_values(values, target(INT128), target(F32))
        assert bits(out, F32) == [0x71800001, 0x71800000]
        assert bits(interp.convert_values(values, target(INT128), target(F64)), F64) == [
            0x4630000010000000,
            0x4630000010000000,
        ]

    def test_int128_and_uint128_endpoints(self):
        signed = words(-(1 << 127), (1 << 127) - 1, 2**127 - 2**103, 2**127 - 2**103 - 1)
        assert bits(interp.convert_values(signed, target(INT128), target(F32)), F32) == [
            0xFF000000,
            0x7F000000,
            0x7EFFFFFF,
            0x7EFFFFFF,
        ]
        assert bits(interp.convert_values(signed, target(INT128), target(F64)), F64)[:2] == [
            0xC7E0000000000000,
            0x47E0000000000000,
        ]
        assert bits(interp.convert_values(signed, target(INT128), target(F16)), F16) == [
            0xFC00,
            0x7C00,
            0x7C00,
            0x7C00,
        ]
        unsigned = words((1 << 128) - 1, 1 << 127)
        assert bits(interp.convert_values(unsigned, target(UINT128), target(F32)), F32) == [
            0x7F800000,
            0x7F000000,
        ]
        assert bits(interp.convert_values(unsigned, target(UINT128), target(F64)), F64) == [
            0x47F0000000000000,
            0x47E0000000000000,
        ]

    def test_float_narrowing_rounds_and_overflows_like_julia(self):
        values = np.array(
            [
                -0.0,
                5e-8,
                6.103515625e-05,
                65504.0,
                65519.99999,
                65520.0,
                -65520.0,
                2.0**-25,
                2.0**-24,
                3.0 * 2.0**-25,
                1.1,
            ],
            dtype="<f8",
        )
        assert bits(interp.convert_values(values, target(F64), target(F16)), F16) == [
            0x8000,
            0x0001,
            0x0400,
            0x7BFF,
            0x7BFF,
            0x7C00,
            0xFC00,
            0x0000,
            0x0001,
            0x0002,
            0x3C66,
        ]
        halves = interp.convert_values(values[:3], target(F64), target(COMPLEXF16))
        assert halves.dtype == COMPLEXF16.dtype
        assert bits(halves["real"], F16) == [0x8000, 0x0001, 0x0400]
        assert bits(halves["imag"], F16) == [0, 0, 0]
        singles = np.array(
            [-0.0, 3.4028235677973366e38, 3.4028235677973362e38, 1e-46, 2.0**-150, 3.0 * 2.0**-150],
            dtype="<f8",
        )
        assert bits(interp.convert_values(singles, target(F64), target(F32)), F32) == [
            0x80000000,
            0x7F800000,
            0x7F7FFFFF,
            0x00000000,
            0x00000000,
            0x00000002,
        ]
        subnormal = np.array([1.4e-45, -1.4e-45, 6.0e-8, 6.5e-5], dtype="<f4")
        assert bits(interp.convert_values(subnormal, target(F32), target(F16)), F16) == [
            0x0000,
            0x8000,
            0x0001,
            0x0443,
        ]

    def test_nan_and_inf_survive_without_payload_promises(self):
        values = np.array([0x7FC00055, 0xFFC00055, 0x7F800000], dtype="<u4").view("<f4")
        out = interp.convert_values(values, target(F32), target(F16))
        assert np.isnan(out[:2]).all()
        assert np.isposinf(out[2])
        wide = interp.convert_values(values, target(F32), target(F64))
        assert np.isnan(wide[:2]).all()
        assert np.signbit(wide[1])
        assert np.isposinf(wide[2])
        halves = np.array([0x7E55, 0xFE55, 0x7C01], dtype="<u2").view("<f2")
        assert np.isnan(interp.convert_values(halves, target(F16), target(F32))).all()

    def test_complex_components_and_positive_zero_imaginary(self):
        source = np.array([complex(1.0, -0.0), complex(-0.0, -0.0)], dtype="<c16")
        assert interp.convert_values(source, target(C128), target(I8)).tolist() == [1, 0]
        assert interp.convert_values(source, target(C128), target(BOOL)).tolist() == [True, False]
        out = interp.convert_values(source, target(C128), target(F64))
        assert out.tolist() == [1.0, 0.0]
        assert np.signbit(out[1])
        out = interp.convert_values(source, target(C128), target(C64))
        assert np.signbit(out.imag).tolist() == [True, True]
        real = np.array([-0.0, 2.0], dtype="<f8")
        out = interp.convert_values(real, target(F64), target(C64))
        assert np.signbit(out.real).tolist() == [True, False]
        assert not np.signbit(out.imag).any()
        halves = interp.convert_values(real, target(F64), target(COMPLEXF16))
        assert bits(halves["real"], F16) == [0x8000, 0x4000]
        assert bits(halves["imag"], F16) == [0, 0]

    def test_complexf16_source_to_integers_and_widening(self):
        source = StoredSeries.from_list(
            ANCHOR, COMPLEXF16, [complex(1, 0), complex(2, 0), complex(-0.0, 0)]
        ).values
        assert interp.convert_values(source, target(COMPLEXF16), target(I8)).tolist() == [1, 2, 0]
        assert interp.convert_values(source, target(COMPLEXF16), target(I64)).tolist() == [1, 2, 0]
        wide = interp.convert_values(source, target(COMPLEXF16), target(C128))
        assert wide.tolist() == [1 + 0j, 2 + 0j, 0j]
        assert np.signbit(wide[2].real)

    def test_integer_widening_and_narrowing_bytes(self):
        assert (
            interp.convert_values(np.array([1, 2], dtype="<i2"), target(I16), target(I64))
            .tobytes()
            .hex()
            == "01000000000000000200000000000000"
        )
        assert interp.convert_values(
            np.array([-1, 7], dtype="<i2"), target(I16), target(I64)
        ).tolist() == [-1, 7]
        wide = interp.convert_values(np.array([-1, 3], dtype="<i8"), target(I64), target(INT128))
        assert interp.unpack_words(wide, True) == [-1, 3]
        unsigned = interp.convert_values(np.array([3], dtype="<u2"), target(U16), target(UINT128))
        assert interp.unpack_words(unsigned, False) == [3]
        from_float = interp.convert_values(
            np.array([2.0**100, -0.0], dtype="<f8"), target(F64), target(INT128)
        )
        assert interp.unpack_words(from_float, True) == [2**100, 0]
        narrow = interp.convert_values(words(-1, 127), target(INT128), target(I8))
        assert narrow.tolist() == [-1, 127]
        as_uint = interp.convert_values(words((1 << 64) - 1), target(INT128), target(U64))
        assert as_uint.tolist() == [(1 << 64) - 1]

    def test_integer_sources_into_dates_keep_raw_codes(self):
        codes = interp.convert_values(
            np.array([-1, -128], dtype="i1"), target(I8), target("MIT{Monthly}")
        )
        assert codes.dtype == np.dtype("<i8")
        assert codes.tolist() == [-1, -128]
        codes = interp.convert_values(
            words(-(1 << 63), (1 << 63) - 1), target(INT128), target("MIT{Monthly}")
        )
        assert codes.tolist() == [-(1 << 63), (1 << 63) - 1]
        codes = interp.convert_values(
            np.array([(1 << 63) - 1], dtype="<u8"), target(U64), target("MIT{Daily}")
        )
        assert codes.tolist() == [(1 << 63) - 1]

    @pytest.mark.parametrize(
        ("element", "codes", "token", "expected"),
        [
            # Monthly: MIT uses year + (period - 1) / 12; Duration uses code / 12.
            # Recorded Julia bytes for [typemin, typemax, 2**54 + 2**30 + 1, -1, -13, -12].
            (
                StoredElement.date(Monthly()),
                MONTHLY_CODES,
                "Float64",
                [
                    0xC3A5555555555555,
                    0x43A5555555555555,
                    0x431555556AAAAAAB,
                    0xBFB5555555555558,
                    0xBFF1555555555556,
                    0xBFF0000000000000,
                ],
            ),
            (
                StoredElement.duration(Monthly()),
                MONTHLY_CODES,
                "Float64",
                [
                    0xC3A5555555555555,
                    0x43A5555555555555,
                    0x431555556AAAAAAB,
                    0xBFB5555555555555,
                    0xBFF1555555555555,
                    0xBFF0000000000000,
                ],
            ),
            (
                StoredElement.date(Monthly()),
                MONTHLY_CODES,
                "Float32",
                [0xDD2AAAAB, 0x5D2AAAAB, 0x58AAAAAB, 0xBDAAAAAB, 0xBF8AAAAB, 0xBF800000],
            ),
            (
                StoredElement.date(Quarterly(3)),
                [-1, -5, 7],
                "Float32",
                [0xBE800000, 0xBFA00000, 0x3FE00000],
            ),
            (
                StoredElement.duration(Quarterly(3)),
                [-1, -5, 7],
                "Float64",
                [0xBFD0000000000000, 0xBFF4000000000000, 0x3FFC000000000000],
            ),
            (StoredElement.date(HalfYearly(6)), [-1, -3], "Float32", [0xBF000000, 0xBFC00000]),
            (
                StoredElement.date(Yearly(12)),
                [-1, -13],
                "Float64",
                [0xBFF0000000000000, 0xC02A000000000000],
            ),
            # Unit and calendar codes convert directly (sitofp), Weekly included.
            (StoredElement.date(Weekly(7)), [-1, -13], "Float32", [0xBF800000, 0xC1500000]),
            (
                StoredElement.duration(BDaily()),
                [-1, -13],
                "Float64",
                [0xBFF0000000000000, 0xC02A000000000000],
            ),
            (
                StoredElement.date(Unit()),
                [-(1 << 63), (1 << 63) - 1, -1],
                "Float32",
                [0xDF000000, 0x5F000000, 0xBF800000],
            ),
            (
                StoredElement.date(Daily()),
                [2**54 + 2**30 + 1, 2**54 + 2**30 - 1],
                "Float32",
                [0x5A800001, 0x5A800000],
            ),
            (
                StoredElement.duration(Daily()),
                [2**54 + 2**30 + 1, 2**54 + 2**30 - 1],
                "Float64",
                [0x4350000010000000, 0x4350000010000000],
            ),
        ],
    )
    def test_plotting_conversions_follow_julia_operation_order(
        self, element, codes, token, expected
    ):
        values = np.array(codes, dtype="<i8")
        out = interp.convert_values(values, target(element), target(token))
        dtype = F32 if token == "Float32" else F64
        assert bits(out, dtype) == expected
        complex_token = "ComplexF32" if token == "Float32" else "ComplexF64"
        as_complex = interp.convert_values(values, target(element), target(complex_token))
        assert bits(as_complex.real, dtype) == expected
        assert not as_complex.imag.any()

    def test_dated_bool_and_int64_outputs(self):
        codes = np.array([0, 1], dtype="<i8")
        date = target(StoredElement.date(Yearly(12)))
        assert interp.convert_values(codes, date, target(BOOL)).tolist() == [False, True]
        assert interp.convert_values(codes, date, target(I64)).tolist() == [0, 1]


# ---- container behaviour --------------------------------------------------


class TestStoredSeriesInterpretation:
    def test_numeric_descriptor_requires_a_foreign_marker(self):
        element = StoredElement.numeric("<i2", marker="Int64")
        assert element.dtype == I16
        assert element.native_kind == 1
        assert element.julia_name == "Int16"
        assert element.written_marker(2) == "Int64"
        with pytest.raises(TypeError, match="Boolean values belong"):
            StoredElement.numeric("?", marker="Int64")
        with pytest.raises(TypeError):
            StoredElement.numeric(np.dtype([("lo", "<u8"), ("hi", "<u8")]), marker="Int64")
        with pytest.raises(TypeError, match="belong in a TSeries"):
            StoredSeries(ANCHOR, np.array([1], dtype="<i2"), StoredElement.numeric("<i2"))
        identity = numeric(I16, [1], "Int16")
        assert identity.element.marker == "Int16"
        assert identity.to_interpreted().values.tolist() == [1]
        with pytest.raises(TypeError, match="Boolean TSeries"):
            numeric(I16, [1], "Bool")
        with pytest.raises(TypeError, match="never evaluated"):
            numeric(I16, [1], "Rational{Int64}")
        with pytest.raises(TypeError, match="from_list"):
            StoredSeries.from_list(ANCHOR, StoredElement.numeric("<i2", marker="Int64"), [1])

    def test_redundant_markers_on_represented_carriers_are_canonicalized(self):
        series = StoredSeries.from_list(ANCHOR, INT128.with_marker("Int128"), [1])
        assert series.element == INT128
        dates = StoredSeries.from_list(
            ANCHOR, StoredElement.date(Monthly()).with_marker("MIT{Monthly}"), [ANCHOR]
        )
        assert dates.element.marker is None
        alias = StoredSeries(
            ANCHOR, np.array([1], dtype="<i8"), StoredElement.numeric("<i8", "Int")
        )
        assert alias.element.marker == "Int"
        assert alias.to_interpreted().values.tolist() == [1]

    def test_preserved_numeric_series_and_explicit_interpretation(self):
        series = numeric(I16, [1, 2], "Int64")
        assert series.element.marker == "Int64"
        assert series.active_marker == "Int64"
        assert series.tolist() == [1, 2]
        result = series.to_interpreted()
        assert isinstance(result, TSeries)
        assert result.values.dtype == I64
        assert result.values.tolist() == [1, 2]
        assert result.firstdate == ANCHOR
        assert result.values.flags.owndata
        assert not np.shares_memory(result.values, series.values)
        assert series.values.dtype == I16  # storage unchanged
        assert series == numeric(I16, [1, 2], "Int64")
        assert series != numeric(I16, [1, 2], "Int32")
        assert "marker='Int64'" in repr(series.element)

    def test_lossy_interpretations_are_explicit_only(self):
        series = numeric(I64, [9007199254740993], "Float64")
        assert series.tolist() == [9007199254740993]
        assert series.to_interpreted().values.tolist() == [9007199254740992.0]
        halves = numeric(F64, [1.1, 65520.0], "ComplexF16")
        interpreted = halves.to_interpreted()
        assert isinstance(interpreted, StoredSeries)
        assert interpreted.element == COMPLEXF16
        assert interpreted.values.tobytes().hex() == "663c0000007c0000"
        with pytest.raises(ValueError):
            StoredSeries.from_list(
                ANCHOR, COMPLEXF16, [complex(1.1, 0)]
            )  # construction stays exact
        wide = StoredSeries.from_list(ANCHOR, INT128.with_marker("Float64"), [2**100 + 1])
        assert wide.tolist() == [2**100 + 1]
        assert wide.to_interpreted().values.tolist() == [2.0**100]

    def test_numeric_into_dates_and_dates_into_numbers(self):
        as_dates = numeric(I64, [1, 2], "MIT{Monthly}").to_interpreted()
        assert isinstance(as_dates, StoredSeries)
        assert as_dates.element == StoredElement.date(Monthly())
        assert as_dates.tolist() == [MIT(Monthly(), 1), MIT(Monthly(), 2)]
        spans = numeric(I64, [-5], "Duration{Daily}").to_interpreted()
        assert spans.tolist() == [Duration(Daily(), -5)]
        flags = StoredSeries.from_list(
            ANCHOR,
            StoredElement.date(Monthly()).with_bool_marker(),
            [MIT(Monthly(), 0), MIT(Monthly(), 1)],
        )
        assert flags.to_bool().values.tolist() == [False, True]
        assert flags.to_interpreted().values.dtype == BOOL
        assert flags.tolist() == [MIT(Monthly(), 0), MIT(Monthly(), 1)]
        with pytest.raises(ValueError, match="zero or one"):
            StoredSeries.from_list(
                ANCHOR, StoredElement.date(Monthly()).with_bool_marker(), [MIT(Monthly(), 2)]
            )
        floats = StoredSeries.from_list(
            ANCHOR, StoredElement.date(Monthly()).with_marker("Float64"), [MIT(Monthly(), -1)]
        ).to_interpreted()
        assert floats.values.tolist() == [-0.08333333333333337]
        with pytest.raises(TypeError):
            StoredSeries.from_list(
                ANCHOR, StoredElement.date(Monthly()).with_marker("Float16"), [MIT(Monthly(), 0)]
            )

    def test_mutation_is_revalidated_against_the_marker(self):
        series = numeric(I16, [1, 2], "Int8")
        series.values[1] = 300
        with pytest.raises(ValueError, match="observation 1"):
            series.validate()
        with pytest.raises(ValueError):
            series.to_interpreted()
        assert series.element.marker == "Int8"
        assert series.values.tolist() == [1, 300]
        series.values[1] = -128
        series.validate()
        series.values.resize((0,), refcheck=False)
        with pytest.raises(TypeError, match="no stored width"):
            series.validate()

    def test_explicit_interpretation_validates_and_converts_one_snapshot(self, monkeypatch):
        series = numeric(I64, [1], "Int8")
        original = interp.check_output_capacity

        def mutate_live_after_snapshot(length, target):
            original(length, target)
            series.values[0] = 300

        monkeypatch.setattr(interp, "check_output_capacity", mutate_live_after_snapshot)
        result = series.to_interpreted()
        assert result.values.tolist() == [1]
        assert series.values.tolist() == [300]

    @pytest.mark.parametrize("dtype", [np.longdouble, np.clongdouble])
    def test_numeric_descriptor_rejects_long_double_on_every_platform(self, dtype):
        assert np.dtype(dtype).char in ("g", "G")
        with pytest.raises(TypeError, match="native-endian series dtypes"):
            StoredElement.numeric(dtype, "Int64")

    def test_empty_numeric_sources_keep_the_kind_default_carrier(self):
        empty = StoredSeries(
            ANCHOR, np.empty(0, dtype="<f8"), StoredElement.numeric("<f8", "Int16")
        )
        result = empty.to_interpreted()
        assert isinstance(result, TSeries)
        assert result.values.dtype == I16
        assert len(result) == 0
        dated = StoredSeries(
            ANCHOR, np.empty(0, dtype="<i8"), StoredElement.numeric("<i8", "MIT{Monthly}")
        )
        assert dated.to_interpreted() == StoredSeries.from_list(
            ANCHOR, StoredElement.date(Monthly()), []
        )
        wide = StoredSeries(
            ANCHOR, np.empty(0, dtype="<i8"), StoredElement.numeric("<i8", "UInt128")
        )
        assert wide.to_interpreted() == StoredSeries.from_list(ANCHOR, UINT128, [])
        with pytest.raises(TypeError, match="no stored width"):
            StoredSeries(ANCHOR, np.empty(0, dtype="<i2"), StoredElement.numeric("<i2", "Int64"))
        with pytest.raises(ValueError, match="no storable width"):
            StoredSeries(ANCHOR, np.empty(0, dtype=INT128.dtype), INT128.with_marker("Float64"))
        with pytest.raises(TypeError, match="cannot load an empty"):
            StoredSeries(
                ANCHOR, np.empty(0, dtype="<i8"), StoredElement.date(Monthly()).with_bool_marker()
            )

    @pytest.mark.parametrize(
        ("kind", "dtype", "marker", "target"),
        [
            (1, I64, "Int", I64),
            (2, U64, "UInt", U64),
            (5, C128, "Complex{Float16}", COMPLEXF16.dtype),
        ],
    )
    def test_new_empty_aliases_preserve_literal_marker(self, kind, dtype, marker, target):
        result = decode_series(32, ANCHOR.value, b"", kind, 0, 0, marker)
        assert isinstance(result, StoredSeries)
        assert result.values.dtype == dtype
        assert result.element.marker == marker
        assert encode_series(result).marker == marker
        assert result.to_interpreted().values.dtype == target

    def test_whole_object_markers_take_precedence(self):
        series = StoredSeries(
            ANCHOR,
            np.array([1, 2], dtype="<i2"),
            StoredElement.numeric("<i2", "NoSuchElement"),
            object_marker="TSeries",
        )
        assert series.object_marker == "TSeries"
        assert series.active_marker is None
        assert series.element.marker == "NoSuchElement"  # retained as data, never interpreted
        result = series.to_interpreted()
        assert isinstance(result, TSeries)
        assert result.values.dtype == I16
        assert result.values.tolist() == [1, 2]
        assert series != StoredSeries(
            ANCHOR, np.array([1, 2], dtype="<i2"), StoredElement.numeric("<i2", "Int64")
        )
        assert "object_marker='TSeries'" in repr(series)
        bypassed = StoredSeries(
            ANCHOR,
            np.array([0, 1], dtype="i1"),
            StoredElement.numeric("i1", "Bool"),
            object_marker="TSeries{Monthly, Int8}",
        )
        assert bypassed.to_interpreted().values.dtype == I8
        with pytest.raises(TypeError):
            bypassed.to_bool()
        wide = StoredSeries.from_list(ANCHOR, INT128, [2**100])
        identity = StoredSeries(
            ANCHOR,
            wide.values,
            INT128.with_marker("NoSuchElement"),
            object_marker="TSeries{Monthly, Int128, Vector{Int128}}",
        )
        assert identity.to_interpreted() == wide
        dated = StoredSeries(
            MIT(Daily(), 739191),
            np.array([2024], dtype="<i8"),
            StoredElement.date(Yearly(12)),
            object_marker="TSeries",
        )
        assert dated.to_interpreted() == StoredSeries(
            MIT(Daily(), 739191), np.array([2024], dtype="<i8"), StoredElement.date(Yearly(12))
        )
        with pytest.raises(TypeError):
            StoredSeries(
                ANCHOR,
                np.array([1], dtype="i1"),
                StoredElement.numeric("i1"),
                object_marker="TSeries{Monthly, Int16}",
            )
        with pytest.raises(TypeError):
            StoredSeries(
                ANCHOR, np.array([1], dtype="i1"), StoredElement.numeric("i1"), object_marker=""
            )

    def test_vector_object_markers_give_empty_arrays(self):
        empty = StoredSeries(
            ANCHOR,
            np.empty(0, dtype="<f8"),
            StoredElement.numeric("<f8", "Int16"),
            object_marker="Vector",
        )
        result = empty.to_interpreted()
        assert isinstance(result, np.ndarray)
        assert result.dtype == F64
        assert result.shape == (0,)
        typed = StoredSeries(
            ANCHOR,
            np.empty(0, dtype="<i8"),
            StoredElement.numeric("<i8"),
            object_marker="Vector{Float64}",
        )
        assert typed.to_interpreted().dtype == F64
        # The wider verified spellings: every element name, the bare Array,
        # Vector{Any} (object dtype) and the wide carriers on a wide base.
        for token, dtype in (
            ("Array{Bool, 1}", np.dtype("?")),
            ("Vector{Int128}", INT128.dtype),
            ("Array{ComplexF16}", COMPLEXF16.dtype),
            ("Vector{Any}", np.dtype(object)),
            ("Array", np.dtype("<i8")),
        ):
            wider = StoredSeries(
                ANCHOR, np.empty(0, dtype="<i8"), StoredElement.numeric("<i8"), object_marker=token
            )
            result = wider.to_interpreted()
            assert (result.dtype, result.shape) == (dtype, (0,)), token
        on_wide = StoredSeries(
            ANCHOR, np.empty(0, dtype=INT128.dtype), INT128, object_marker="Vector{Float64}"
        )
        assert on_wide.to_interpreted().dtype == F64
        with pytest.raises(ValueError, match="empty payload only"):
            StoredSeries(
                ANCHOR, np.array([1.0]), StoredElement.numeric("<f8"), object_marker="Vector"
            )
        with pytest.raises(TypeError):
            StoredSeries(
                ANCHOR,
                np.empty(0, dtype="<i8"),
                StoredElement.date(Monthly()),
                object_marker="Vector",
            )

    def test_output_capacity_guard_precedes_allocation(self, monkeypatch):
        series = numeric(I8, [1, 2, 3, 4], "Int128")
        monkeypatch.setattr(interp, "MAX_BYTES", 32)
        with pytest.raises(ValueError, match="nothing was allocated"):
            series.to_interpreted()
        series.validate()  # storage itself stays valid
