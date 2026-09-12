"""Stored representation of date, duration, 128-bit and ComplexF16 series elements.

The module under test is a representation layer that is not yet connected to
DataEcon file operations. Expected bytes and metadata come from the pinned
Julia reference's own stores, spelled out here independently of the code.
"""

import math
import warnings

import numpy as np
import pytest

from tsecon import MIT, Duration, TSeries, mm
from tsecon.dataecon._codec import MAX_BYTES
from tsecon.dataecon._represented import (
    CODE_DTYPE,
    COMPLEXF16,
    COMPLEXF16_DTYPE,
    INT128,
    INT128_DTYPE,
    UINT128,
    StoredElement,
    StoredSeries,
    julia_frequency_name,
)
from tsecon.frequencies import BDaily, Daily, HalfYearly, Monthly, Quarterly, Unit, Weekly, Yearly

ANCHOR = mm(2024, 11)  # code 24298
# Payloads as stored by the pinned Julia reference (TimeSeriesEcon.jl DataEcon).
INT128_ENDPOINTS_HEX = (
    "00000000000000000000000000000080"
    "ffffffffffffffffffffffffffffffff"
    "00000000000000000000000000000000"
    "ffffffffffffffffffffffffffffff7f"
)
INT128_WORDS = [
    1 << 64,
    -(1 << 64) - 1,
    (0x0123456789ABCDEF << 64) | 0x0FEDCBA987654321,
    -(1 << 127) + 1,
    (1 << 127) - 2,
]
UINT128_WORDS_HEX = (
    "00000000000000000100000000000000"
    "00000000000000000000000000000080"
    "feffffffffffffffffffffffffffffff"
)
COMPLEXF16_BITS_HEX = "0080003d01000180557e007e007c000000fcff7bff7bfffb"
COMPLEXF16_REAL_BITS = [0x8000, 0x0001, 0x7E55, 0x7C00, 0xFC00, 0x7BFF]
COMPLEXF16_IMAG_BITS = [0x3D00, 0x8001, 0x7E00, 0x0000, 0x7BFF, 0xFBFF]
MIT_FULL_RANGE_HEX = "0000000000000080ffffffffffffffff0000000000000000ffffffffffffff7f"
FB_INT128_01_HEX = "01" + "00" * 15 + "00" * 16
FB_COMPLEXF16_1_HEX = "003c0000"
WB_UINT128_01_HEX = "00" * 16 + "01" + "00" * 15
WB_COMPLEXF16_01_NEGZERO_IMAG_HEX = "00000000003c0080"

DATE_TOKENS = {
    Unit(): "Unit",
    Daily(): "Daily",
    BDaily(): "BDaily",
    Monthly(): "Monthly",
    **{Weekly(day): f"Weekly{{{day}}}" for day in range(1, 8)},
    **{Quarterly(month): f"Quarterly{{{month}}}" for month in range(1, 4)},
    **{HalfYearly(month): f"HalfYearly{{{month}}}" for month in range(1, 7)},
    **{Yearly(month): f"Yearly{{{month}}}" for month in range(1, 13)},
}


def carrier(hex_payload, dtype):
    return np.frombuffer(bytes.fromhex(hex_payload), dtype=dtype).copy()


def bits(array, field):
    return array[field].view("<u2").tolist()


class TestStoredElement:
    def test_kinds_dtypes_and_native_codes(self):
        assert INT128.dtype == np.dtype([("lo", "<u8"), ("hi", "<u8")])
        assert UINT128.dtype == INT128.dtype
        assert COMPLEXF16.dtype == np.dtype([("real", "<f2"), ("imag", "<f2")])
        assert (INT128.itemsize, UINT128.itemsize, COMPLEXF16.itemsize) == (16, 16, 4)
        assert (INT128.native_kind, UINT128.native_kind, COMPLEXF16.native_kind) == (1, 2, 5)
        assert (INT128.native_frequency, COMPLEXF16.native_frequency) == (0, 0)
        assert [e.julia_name for e in (INT128, UINT128, COMPLEXF16)] == [
            "Int128",
            "UInt128",
            "ComplexF16",
        ]
        date = StoredElement.date(Yearly(12))
        duration = StoredElement.duration(Daily())
        assert date.dtype == np.dtype("<i8")
        assert duration.dtype == CODE_DTYPE
        assert (date.native_kind, date.native_frequency) == (3, 268)
        assert (duration.native_kind, duration.native_frequency) == (1, 12)
        assert StoredElement.date(Unit()).native_frequency == 11
        assert StoredElement.duration(Weekly(7)).native_frequency == 23

    def test_sixty_four_date_and_duration_tokens(self):
        assert len(DATE_TOKENS) == 32
        tokens = set()
        for frequency, name in DATE_TOKENS.items():
            assert julia_frequency_name(frequency) == name
            assert StoredElement.date(frequency).julia_name == f"MIT{{{name}}}"
            assert StoredElement.duration(frequency).julia_name == f"Duration{{{name}}}"
            tokens.add(StoredElement.date(frequency).julia_name)
            tokens.add(StoredElement.duration(frequency).julia_name)
        assert len(tokens) == 64
        assert {
            "MIT{Monthly}",
            "Duration{Weekly{7}}",
            "MIT{Quarterly{3}}",
            "Duration{Unit}",
        } <= tokens

    def test_invalid_descriptors(self):
        with pytest.raises(TypeError):
            StoredElement("float128")
        with pytest.raises(TypeError):
            StoredElement("date")
        with pytest.raises(TypeError):
            StoredElement("date", 32)
        with pytest.raises(TypeError):
            StoredElement("int128", Monthly())
        with pytest.raises(TypeError):
            StoredElement("int128", marker="Int64")
        with pytest.raises(TypeError):
            StoredElement("date", Monthly(), marker="Bool")
        with pytest.raises(TypeError):
            StoredElement.date(Monthly()).with_bool_marker()

    def test_bool_marker_and_written_marker(self):
        marked = INT128.with_bool_marker()
        assert marked == StoredElement("int128", None, "Bool")
        assert marked != INT128
        assert marked.is_wide
        assert marked.julia_name == "Int128"
        assert marked.written_marker(0) == "Bool"
        assert marked.written_marker(3) == "Bool"
        assert INT128.written_marker(0) == "Int128"
        assert INT128.written_marker(3) is None
        assert StoredElement.date(Monthly()).written_marker(0) == "MIT{Monthly}"
        assert StoredElement.duration(Monthly()).written_marker(2) is None
        with pytest.raises((AttributeError, TypeError)):
            marked.marker = None


class TestConstructionAndOwnership:
    def test_copy_by_default_owns_writable_contiguous(self):
        source = carrier(INT128_ENDPOINTS_HEX, INT128_DTYPE)
        series = StoredSeries(ANCHOR, source, INT128)
        assert not np.shares_memory(series.values, source)
        assert series.values.flags.owndata
        assert series.values.flags.writeable
        assert series.values.flags.c_contiguous
        source["lo"][0] = 7
        assert series.values.tobytes() == bytes.fromhex(INT128_ENDPOINTS_HEX)

    def test_explicit_sharing_of_compatible_writable_array(self):
        source = carrier(COMPLEXF16_BITS_HEX, COMPLEXF16_DTYPE)
        series = StoredSeries(ANCHOR, source, COMPLEXF16, copy=False)
        assert np.shares_memory(series.values, source)
        series.values["imag"][3] = np.float16(2.0)
        assert bits(source, "imag")[3] == 0x4000

    def test_read_only_and_strided_input_are_copied_even_without_copy(self):
        readonly = carrier(MIT_FULL_RANGE_HEX, CODE_DTYPE)
        readonly.flags.writeable = False
        series = StoredSeries(ANCHOR, readonly, StoredElement.date(Monthly()), copy=False)
        assert not np.shares_memory(series.values, readonly)
        assert series.values.flags.writeable
        strided = carrier(INT128_ENDPOINTS_HEX, INT128_DTYPE)[::2]
        assert not strided.flags.c_contiguous
        series = StoredSeries(ANCHOR, strided, INT128, copy=False)
        assert series.values.flags.c_contiguous
        assert not np.shares_memory(series.values, strided)
        assert series.values.tobytes() == strided.tobytes()
        view = carrier(INT128_ENDPOINTS_HEX, INT128_DTYPE)[1:3]
        shared = StoredSeries(ANCHOR, view, INT128, copy=False)
        assert np.shares_memory(shared.values, view.base)

    def test_wrong_dtype_ndim_and_input_type_are_refused(self):
        with pytest.raises(TypeError, match="no implicit conversion"):
            StoredSeries(ANCHOR, np.array([1, 2], dtype="<i4"), StoredElement.date(Monthly()))
        with pytest.raises(TypeError):
            StoredSeries(ANCHOR, np.array([1, 2], dtype="<i8"), INT128)
        with pytest.raises(TypeError):
            StoredSeries(ANCHOR, np.zeros(2, dtype=[("a", "<u8"), ("b", "<u8")]), INT128)
        with pytest.raises(TypeError):
            StoredSeries(ANCHOR, np.zeros(2, dtype=[("lo", ">u8"), ("hi", ">u8")]), INT128)
        with pytest.raises(TypeError):
            StoredSeries(ANCHOR, np.zeros(2, dtype="<c8"), COMPLEXF16)
        with pytest.raises(ValueError, match="one-dimensional"):
            StoredSeries(ANCHOR, np.zeros((2, 1), dtype=INT128_DTYPE), INT128)
        with pytest.raises(TypeError, match="from_list"):
            StoredSeries(ANCHOR, [24298, 24299], StoredElement.date(Monthly()))
        with pytest.raises(TypeError):
            StoredSeries(ANCHOR, np.zeros(2, dtype=INT128_DTYPE), "int128")

    def test_anchor_must_be_a_supported_series_axis(self):
        values = np.zeros(1, dtype=CODE_DTYPE)
        with pytest.raises(TypeError):
            StoredSeries(24298, values, StoredElement.date(Monthly()))
        with pytest.raises(TypeError):
            StoredSeries(MIT(Unit(), 3), values, StoredElement.date(Monthly()))
        series = StoredSeries(MIT(Daily(), 739191), values, StoredElement.date(Yearly(12)))
        assert series.frequency == Daily()
        assert series.element.frequency == Yearly(12)

    def test_capacity_is_checked_before_any_copy(self):
        too_long = np.broadcast_to(np.zeros(1, dtype=INT128_DTYPE), (MAX_BYTES // 16 + 1,))
        with pytest.raises(ValueError, match="byte limit"):
            StoredSeries(ANCHOR, too_long, INT128)

    def test_capacity_boundary_with_a_small_limit(self, monkeypatch):
        monkeypatch.setattr("tsecon.dataecon._represented.MAX_BYTES", 64)
        assert len(StoredSeries(ANCHOR, np.zeros(4, dtype=INT128_DTYPE), INT128)) == 4
        assert len(StoredSeries(ANCHOR, np.zeros(16, dtype=COMPLEXF16_DTYPE), COMPLEXF16)) == 16
        with pytest.raises(ValueError, match="byte limit"):
            StoredSeries(ANCHOR, np.zeros(5, dtype=INT128_DTYPE), INT128)
        grown = StoredSeries(ANCHOR, np.zeros(8, dtype=CODE_DTYPE), StoredElement.date(Monthly()))
        grown.values.resize((9,), refcheck=False)
        with pytest.raises(ValueError, match="byte limit"):
            grown.validate()

    def test_read_only_anchor_descriptor_and_repr(self):
        series = StoredSeries(
            ANCHOR, np.zeros(2, dtype=CODE_DTYPE), StoredElement.duration(Daily())
        )
        with pytest.raises(AttributeError):
            series.firstdate = mm(2025, 1)
        with pytest.raises(AttributeError):
            series.element = INT128
        with pytest.raises(AttributeError):
            series.values = np.zeros(2, dtype=CODE_DTYPE)
        assert "length=2" in repr(series)
        assert "duration" in repr(series)
        with pytest.raises(TypeError):
            hash(series)


class TestAxis:
    def test_lastdate_and_len(self):
        series = StoredSeries(ANCHOR, np.zeros(3, dtype=CODE_DTYPE), StoredElement.date(Monthly()))
        assert len(series) == 3
        assert series.lastdate == mm(2025, 1)
        assert series.lastdate == TSeries(ANCHOR, np.zeros(3)).lastdate

    def test_empty_lastdate_is_the_synthetic_endpoint(self):
        empty = StoredSeries(ANCHOR, np.zeros(0, dtype=INT128_DTYPE), INT128)
        assert len(empty) == 0
        assert empty.lastdate == mm(2024, 10)
        assert empty.lastdate.value == 24297
        assert empty.lastdate == TSeries(ANCHOR, np.zeros(0)).lastdate
        assert empty.tolist() == []


class TestRevalidation:
    def test_in_place_shape_change_is_detected(self):
        series = StoredSeries(ANCHOR, np.zeros(4, dtype=CODE_DTYPE), StoredElement.date(Monthly()))
        series.values.shape = (2, 2)
        with pytest.raises(ValueError, match="one-dimensional"):
            len(series)
        with pytest.raises(ValueError):
            series.tolist()
        with pytest.raises(ValueError):
            series.validate()

    def test_in_place_dtype_change_is_detected(self):
        series = StoredSeries(ANCHOR, carrier(INT128_ENDPOINTS_HEX, INT128_DTYPE), INT128)
        series.values.dtype = np.dtype([("a", "<u8"), ("b", "<u8")])
        with pytest.raises(TypeError, match="carrier dtype"):
            series.tolist()
        with pytest.raises(TypeError):
            _ = series.lastdate
        other = StoredSeries(ANCHOR, carrier(INT128_ENDPOINTS_HEX, INT128_DTYPE), INT128)
        with pytest.raises(TypeError):
            series == other  # noqa: B015

    def test_in_place_stride_change_is_detected(self):
        series = StoredSeries.from_list(ANCHOR, INT128, [0, 1])
        with warnings.catch_warnings():
            # NumPy 2.x deprecates assigning strides; the mutation still happens.
            warnings.simplefilter("ignore", DeprecationWarning)
            series.values.strides = (0,)
        assert not series.values.flags.c_contiguous
        with pytest.raises(ValueError, match="C-contiguous"):
            series.validate()
        with pytest.raises(ValueError):
            series.tolist()
        with pytest.raises(ValueError):
            len(series)

    def test_clearing_the_writeable_flag_is_allowed(self):
        series = StoredSeries.from_list(ANCHOR, INT128.with_bool_marker(), [0, 1])
        series.values.flags.writeable = False
        series.validate()
        assert series.tolist() == [0, 1]
        assert series.to_bool().values.tolist() == [False, True]
        assert len(series) == 2

    def test_equality_covers_element_anchor_and_bytes(self):
        a = StoredSeries(ANCHOR, carrier(INT128_ENDPOINTS_HEX, INT128_DTYPE), INT128)
        b = StoredSeries(ANCHOR, carrier(INT128_ENDPOINTS_HEX, INT128_DTYPE), INT128)
        assert a == b
        assert a != StoredSeries(ANCHOR, carrier(INT128_ENDPOINTS_HEX, INT128_DTYPE), UINT128)
        assert a != StoredSeries(mm(2024, 12), carrier(INT128_ENDPOINTS_HEX, INT128_DTYPE), INT128)
        b.values["lo"][2] = 1
        assert a != b
        assert a.__eq__(TSeries(ANCHOR, np.zeros(4))) is NotImplemented


class TestDatesAndDurations:
    def test_full_int64_codes_round_trip_without_packing(self):
        element = StoredElement.date(Monthly())
        series = StoredSeries(ANCHOR, carrier(MIT_FULL_RANGE_HEX, CODE_DTYPE), element)
        values = series.tolist()
        assert values == [
            MIT(Monthly(), -(1 << 63)),
            MIT(Monthly(), -1),
            MIT(Monthly(), 0),
            MIT(Monthly(), (1 << 63) - 1),
        ]
        again = StoredSeries.from_list(ANCHOR, element, values)
        assert again == series
        assert again.values.tobytes() == bytes.fromhex(MIT_FULL_RANGE_HEX)

    def test_python_written_dates_and_cross_axis_durations(self):
        dates = StoredSeries.from_list(
            ANCHOR, StoredElement.date(Monthly()), [mm(2024, 11), mm(2024, 12)]
        )
        assert dates.values.tobytes().hex() == "ea5e000000000000eb5e000000000000"
        durations = StoredSeries.from_list(
            MIT(Yearly(12), 2024),
            StoredElement.duration(Daily()),
            [Duration(Daily(), -5), Duration(Daily(), 0), Duration(Daily(), 7)],
        )
        assert (
            durations.values.tobytes().hex() == "fbffffffffffffff00000000000000000700000000000000"
        )
        assert durations.frequency == Yearly(12)
        assert durations.element.native_frequency == 12
        assert durations.tolist() == [
            Duration(Daily(), -5),
            Duration(Daily(), 0),
            Duration(Daily(), 7),
        ]
        annual_on_daily = StoredSeries.from_list(
            MIT(Daily(), 739191),
            StoredElement.date(Yearly(12)),
            [MIT(Yearly(12), 2024), MIT(Yearly(12), 2025)],
        )
        assert annual_on_daily.values.tobytes().hex() == "e807000000000000e907000000000000"

    def test_from_list_rejects_wrong_type_frequency_and_range(self):
        element = StoredElement.date(Monthly())
        with pytest.raises(TypeError):
            StoredSeries.from_list(ANCHOR, element, [Duration(Monthly(), 1)])
        with pytest.raises(TypeError):
            StoredSeries.from_list(ANCHOR, element, [MIT(Quarterly(3), 1)])
        with pytest.raises(TypeError):
            StoredSeries.from_list(ANCHOR, element, [24298])
        with pytest.raises(ValueError):
            StoredSeries.from_list(ANCHOR, element, [MIT(Monthly(), 1 << 63)])
        with pytest.raises(TypeError):
            StoredSeries.from_list(ANCHOR, StoredElement.duration(Daily()), [MIT(Daily(), 1)])


class TestWideIntegers:
    def test_int128_endpoints_and_word_order(self):
        series = StoredSeries(ANCHOR, carrier(INT128_ENDPOINTS_HEX, INT128_DTYPE), INT128)
        assert series.tolist() == [-(1 << 127), -1, 0, (1 << 127) - 1]
        words = StoredSeries.from_list(ANCHOR, INT128, INT128_WORDS)
        assert words.values.tobytes()[:16] == bytes(8) + b"\x01" + bytes(7)
        assert words.tolist() == INT128_WORDS
        assert words.values["lo"].tolist()[1] == (1 << 64) - 1
        assert words.values["hi"].tolist()[1] == (1 << 64) - 2

    def test_uint128_endpoints(self):
        series = StoredSeries(ANCHOR, carrier(UINT128_WORDS_HEX, INT128_DTYPE), UINT128)
        assert series.tolist() == [1 << 64, 1 << 127, (1 << 128) - 2]
        assert (
            StoredSeries.from_list(ANCHOR, UINT128, [0, 1, (1 << 128) - 1]).values.tobytes().hex()
            == "00" * 16 + "01" + "00" * 15 + "ff" * 16
        )
        assert (
            StoredSeries(ANCHOR, carrier(INT128_ENDPOINTS_HEX, INT128_DTYPE), UINT128).tolist()[0]
            == 1 << 127
        )

    def test_from_list_rejects_out_of_range_and_non_int(self):
        with pytest.raises(ValueError):
            StoredSeries.from_list(ANCHOR, INT128, [1 << 127])
        with pytest.raises(ValueError):
            StoredSeries.from_list(ANCHOR, UINT128, [-1])
        for bad in (1.0, np.int64(1), True, "1"):
            with pytest.raises(TypeError):
                StoredSeries.from_list(ANCHOR, INT128, [bad])


class TestComplexF16:
    def test_bit_patterns_tolist_and_widening(self):
        series = StoredSeries(ANCHOR, carrier(COMPLEXF16_BITS_HEX, COMPLEXF16_DTYPE), COMPLEXF16)
        assert bits(series.values, "real") == COMPLEXF16_REAL_BITS
        assert bits(series.values, "imag") == COMPLEXF16_IMAG_BITS
        values = series.tolist()
        assert values[0] == complex(-0.0, 1.25)
        assert math.copysign(1.0, values[0].real) == -1.0
        assert values[1] == complex(2.0**-24, -(2.0**-24))
        assert math.isnan(values[2].real)
        assert math.isnan(values[2].imag)
        assert values[3] == complex(math.inf, 0.0)
        assert values[4] == complex(-math.inf, 65504.0)
        assert values[5] == complex(65504.0, -65504.0)
        wide = series.to_complex64()
        assert isinstance(wide, TSeries)
        assert wide.values.dtype == np.dtype("<c8")
        assert wide.firstdate == ANCHOR
        assert np.signbit(wide.values[0].real)
        assert wide.values[1] == np.complex64(complex(2.0**-24, -(2.0**-24)))
        assert np.isposinf(wide.values[3].real)
        assert wide.values[5] == np.complex64(complex(65504.0, -65504.0))
        with pytest.raises(TypeError):
            StoredSeries(ANCHOR, np.zeros(1, dtype=INT128_DTYPE), INT128).to_complex64()

    def test_from_complex_exact_representability_rules(self):
        series = StoredSeries.from_list(
            ANCHOR,
            COMPLEXF16,
            [
                complex(-0.0, 1.25),
                complex(math.inf, -math.inf),
                complex(math.nan, 65504.0),
                complex(0.0, -0.0),
            ],
        )
        assert bits(series.values, "real") == [0x8000, 0x7C00, 0x7E00, 0x0000]
        assert bits(series.values, "imag") == [0x3D00, 0xFC00, 0x7BFF, 0x8000]
        for inexact in (
            complex(0.1, 0.0),
            complex(0.0, 1e-9),
            complex(65520.0, 0.0),
            complex(70000.0, 0.0),
            complex(2049.0, 0.0),
        ):
            with pytest.raises(ValueError):
                StoredSeries.from_list(ANCHOR, COMPLEXF16, [inexact])
        for bad in (1.0, 1, np.complex64(1), (1.0, 2.0), (np.float16(1), 2.0)):
            with pytest.raises(TypeError):
                StoredSeries.from_list(ANCHOR, COMPLEXF16, [bad])

    def test_float16_pairs_preserve_every_bit_including_nan_payloads(self):
        pairs = [
            (np.array([r], dtype="<u2").view("<f2")[0], np.array([i], dtype="<u2").view("<f2")[0])
            for r, i in zip(COMPLEXF16_REAL_BITS, COMPLEXF16_IMAG_BITS, strict=True)
        ]
        series = StoredSeries.from_list(ANCHOR, COMPLEXF16, pairs)
        assert series.values.tobytes() == bytes.fromhex(COMPLEXF16_BITS_HEX)
        # tolist/from_list is a value round trip: the NaN payload becomes canonical.
        again = StoredSeries.from_list(ANCHOR, COMPLEXF16, series.tolist())
        assert bits(again.values, "real")[2] == 0x7E00
        assert bits(again.values, "real")[:2] == COMPLEXF16_REAL_BITS[:2]


class TestBoolMarkerPreservation:
    def test_marked_wide_carriers_preserve_bytes_and_convert_explicitly(self):
        int128 = StoredSeries(
            ANCHOR, carrier(FB_INT128_01_HEX, INT128_DTYPE), INT128.with_bool_marker()
        )
        assert int128.values.tobytes() == bytes.fromhex(FB_INT128_01_HEX)
        assert int128.tolist() == [1, 0]
        flags = int128.to_bool()
        assert isinstance(flags, TSeries)
        assert flags.values.dtype == np.dtype(bool)
        assert flags.values.tolist() == [True, False]
        assert flags.firstdate == ANCHOR
        uint128 = StoredSeries(
            ANCHOR, carrier(WB_UINT128_01_HEX, INT128_DTYPE), UINT128.with_bool_marker()
        )
        assert uint128.to_bool().values.tolist() == [False, True]
        complex16 = StoredSeries(
            ANCHOR, carrier(FB_COMPLEXF16_1_HEX, COMPLEXF16_DTYPE), COMPLEXF16.with_bool_marker()
        )
        assert complex16.to_bool().values.tolist() == [True]
        assert complex16.tolist() == [1 + 0j]
        signed_zero = StoredSeries(
            ANCHOR,
            carrier(WB_COMPLEXF16_01_NEGZERO_IMAG_HEX, COMPLEXF16_DTYPE),
            COMPLEXF16.with_bool_marker(),
        )
        assert signed_zero.to_bool().values.tolist() == [False, True]
        assert StoredSeries(
            ANCHOR, carrier("00800000", COMPLEXF16_DTYPE), COMPLEXF16.with_bool_marker()
        ).to_bool().values.tolist() == [False]
        assert int128.element.written_marker(len(int128)) == "Bool"
        assert int128 != StoredSeries(ANCHOR, carrier(FB_INT128_01_HEX, INT128_DTYPE), INT128)

    @pytest.mark.parametrize(
        ("hex_payload", "element"),
        [
            ("02" + "00" * 15, UINT128),
            ("00" * 8 + "01" + "00" * 7, UINT128),
            ("ff" * 16, UINT128),
            ("00" * 8 + "01" + "00" * 7, INT128),
            ("ff" * 16, INT128),
            ("0000003c", COMPLEXF16),
            ("007e0000", COMPLEXF16),
            ("003c557e", COMPLEXF16),
            ("007c0000", COMPLEXF16),
            ("00400000", COMPLEXF16),
            ("01000000", COMPLEXF16),
        ],
    )
    def test_invalid_marked_values_are_refused_at_construction(self, hex_payload, element):
        with pytest.raises(ValueError, match="exact zeros and ones"):
            StoredSeries(ANCHOR, carrier(hex_payload, element.dtype), element.with_bool_marker())
        with pytest.raises(ValueError):
            StoredSeries.from_list(
                ANCHOR,
                element.with_bool_marker(),
                StoredSeries(ANCHOR, carrier(hex_payload, element.dtype), element).tolist(),
            )

    def test_mutation_into_invalid_marked_carrier_is_detected_not_normalized(self):
        series = StoredSeries(
            ANCHOR, carrier(FB_INT128_01_HEX, INT128_DTYPE), INT128.with_bool_marker()
        )
        series.values["hi"][1] = 1
        assert len(series) == 2
        assert series.tolist() == [1, 1 << 64]
        with pytest.raises(ValueError, match="observation 1"):
            series.to_bool()
        with pytest.raises(ValueError):
            series.validate()
        assert series.element.marker == "Bool"
        assert series.values.tobytes() == bytes.fromhex(
            "01" + "00" * 15 + "00" * 8 + "01" + "00" * 7
        )

    @pytest.mark.parametrize(
        ("element", "items"),
        [(INT128, [0, 1]), (UINT128, [1]), (COMPLEXF16, [0j, 1 + 0j])],
    )
    def test_emptied_marked_carrier_is_refused_by_validate_and_to_bool(self, element, items):
        series = StoredSeries.from_list(ANCHOR, element.with_bool_marker(), items)
        series.values.resize((0,), refcheck=False)
        assert len(series) == 0
        assert series.tolist() == []
        with pytest.raises(ValueError, match="no storable width"):
            series.validate()
        with pytest.raises(ValueError, match="no storable width"):
            series.to_bool()
        assert series.element.marker == "Bool"

    def test_unmarked_carriers_have_no_bool_conversion_and_empties_are_refused(self):
        with pytest.raises(TypeError):
            StoredSeries(ANCHOR, carrier(FB_INT128_01_HEX, INT128_DTYPE), INT128).to_bool()
        with pytest.raises(ValueError, match="no storable width"):
            StoredSeries(ANCHOR, np.zeros(0, dtype=INT128_DTYPE), UINT128.with_bool_marker())
        with pytest.raises(ValueError):
            StoredSeries.from_list(ANCHOR, COMPLEXF16.with_bool_marker(), [])

    def test_from_list_with_marked_descriptor(self):
        series = StoredSeries.from_list(ANCHOR, INT128.with_bool_marker(), [0, 1, 0])
        assert series.values.tobytes().hex() == "00" * 16 + "01" + "00" * 15 + "00" * 16
        assert series.to_bool().values.tolist() == [False, True, False]
        assert StoredSeries.from_list(
            ANCHOR, COMPLEXF16.with_bool_marker(), [complex(-0.0, 0.0), 1 + 0j]
        ).to_bool().values.tolist() == [False, True]
