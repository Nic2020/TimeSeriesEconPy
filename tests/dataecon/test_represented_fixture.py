"""Julia-written represented series elements: provenance, values, bytes and controls.

Every object in ``julia_represented_elements.daec`` is claimed by exactly one
expectation below, so the coverage test fails if the fixture gains or loses
an object without a matching assertion. Expected bytes and metadata are
spelled out from the pinned Julia reference's own stores, independently of
the code under test; the Julia load outcome recorded by the generator is
noted next to each control.
"""

import hashlib
import importlib.util
import sqlite3
import tomllib
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from tsecon import MIT, Duration, TSeries, mm
from tsecon.dataecon import (
    COMPLEXF16,
    INT128,
    UINT128,
    DataEconError,
    StoredElement,
    StoredSeries,
    open_dataecon,
)
from tsecon.frequencies import BDaily, Daily, HalfYearly, Monthly, Quarterly, Unit, Weekly, Yearly

NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built",
)
FIXTURE = Path(__file__).parent / "fixtures/julia_represented_elements.daec"
ANCHOR = mm(2024, 11)  # code 24298
CODES = [-(1 << 63), -1, 0, (1 << 63) - 1]
FAMILIES = {
    11: Unit(),
    12: Daily(),
    13: BDaily(),
    32: Monthly(),
    **{16 + d: Weekly(d) for d in range(1, 8)},
    **{64 + m: Quarterly(m) for m in range(1, 4)},
    **{128 + m: HalfYearly(m) for m in range(1, 7)},
    **{256 + m: Yearly(m) for m in range(1, 13)},
}
# Payload bytes as the pinned Julia reference stores them.
INT128_ENDPOINTS_HEX = (
    "00000000000000000000000000000080ffffffffffffffffffffffffffffffff"
    "00000000000000000000000000000000ffffffffffffffffffffffffffffff7f"
)
INT128_WORDS_HEX = (
    "00000000000000000100000000000000fffffffffffffffffeffffffffffffff"
    "21436587a9cbed0fefcdab896745230101000000000000000000000000000080"
    "feffffffffffffffffffffffffffff7f"
)
UINT128_ENDPOINTS_HEX = (
    "00000000000000000000000000000000" + "00000000000000000100000000000000" + "ff" * 16
)
UINT128_WORDS_HEX = (
    "00000000000000000100000000000000"
    "00000000000000000000000000000080"
    "feffffffffffffffffffffffffffffff"
)
COMPLEXF16_BITS_HEX = "0080003d01000180557e007e007c000000fcff7bff7bfffb"
CODES_HEX = "0000000000000080ffffffffffffffff0000000000000000ffffffffffffff7f"


def float16_pairs(real_bits, imag_bits):
    return [
        (
            np.array([r], dtype="<u2").view("<f2")[0],
            np.array([i], dtype="<u2").view("<f2")[0],
        )
        for r, i in zip(real_bits, imag_bits, strict=True)
    ]


def shared_cases() -> dict[str, tuple[StoredSeries, str]]:
    """Objects Julia wrote through its own writer: expected container and payload hex."""
    cases: dict[str, tuple[StoredSeries, str]] = {}
    codes = np.array(CODES, dtype="<i8")
    for code, frequency in FAMILIES.items():
        for tag, kind in (("mit", "date"), ("dur", "duration")):
            element = StoredElement(kind, frequency)
            cases[f"rs_{tag}_{code}"] = (StoredSeries(ANCHOR, codes, element), CODES_HEX)
            cases[f"rs_{tag}_{code}_empty"] = (StoredSeries(ANCHOR, codes[:0], element), "")
    cases["rs_mit_268_on_daily"] = (
        StoredSeries.from_list(
            MIT(Daily(), 739191),
            StoredElement.date(Yearly(12)),
            [MIT(Yearly(12), 2024), MIT(Yearly(12), 2025)],
        ),
        "e807000000000000e907000000000000",
    )
    cases["rs_dur_12_on_yearly"] = (
        StoredSeries.from_list(
            MIT(Yearly(12), 2024),
            StoredElement.duration(Daily()),
            [Duration(Daily(), -5), Duration(Daily(), 0), Duration(Daily(), 7)],
        ),
        "fbffffffffffffff00000000000000000700000000000000",
    )
    cases["rs_int128_endpoints"] = (
        StoredSeries.from_list(ANCHOR, INT128, [-(1 << 127), -1, 0, (1 << 127) - 1]),
        INT128_ENDPOINTS_HEX,
    )
    cases["rs_int128_words"] = (
        StoredSeries.from_list(
            ANCHOR,
            INT128,
            [
                1 << 64,
                -(1 << 64) - 1,
                (0x0123456789ABCDEF << 64) | 0x0FEDCBA987654321,
                -(1 << 127) + 1,
                (1 << 127) - 2,
            ],
        ),
        INT128_WORDS_HEX,
    )
    cases["rs_uint128_endpoints"] = (
        StoredSeries.from_list(ANCHOR, UINT128, [0, 1 << 64, (1 << 128) - 1]),
        UINT128_ENDPOINTS_HEX,
    )
    cases["rs_uint128_words"] = (
        StoredSeries.from_list(ANCHOR, UINT128, [1 << 64, 1 << 127, (1 << 128) - 2]),
        UINT128_WORDS_HEX,
    )
    cases["rs_complexf16_bits"] = (
        StoredSeries.from_list(
            ANCHOR,
            COMPLEXF16,
            float16_pairs(
                [0x8000, 0x0001, 0x7E55, 0x7C00, 0xFC00, 0x7BFF],
                [0x3D00, 0x8001, 0x7E00, 0x0000, 0x7BFF, 0xFBFF],
            ),
        ),
        COMPLEXF16_BITS_HEX,
    )
    for element in (INT128, UINT128, COMPLEXF16):
        cases[f"rs_{element.julia_name}_empty"] = (
            StoredSeries.from_list(ANCHOR, element, []),
            "",
        )
    for name in ("rs_mit_32", "rs_int128_words", "rs_complexf16_bits"):
        cases[f"{name}_rewrite"] = cases[name]
    return cases


SHARED = shared_cases()
# Wide carriers with a Bool marker, stored through the C ABI at their own width,
# and Julia's canonical rewrite of the loaded Boolean series.
BOOL_CASES = {
    "rs_bool_int128": (INT128, [0, 1], "00" * 16 + "01" + "00" * 15),
    "rs_bool_uint128": (UINT128, [0, 1], "00" * 16 + "01" + "00" * 15),
    "rs_bool_complexf16": (COMPLEXF16, [0j, 1 + 0j], "00000000003c0000"),
}
CANONICAL_BOOL = {
    "rs_bool_int128_converted": [False, True],
    "rs_bool_uint128_converted": [False, True],
    "rs_bool_complexf16_converted": [False, True],
    "fb_int16_01_rewrite": [False, True, False],
    "fb_float64_01_rewrite": [False, True],
    "wb_uint128_01_rewrite": [False, True],
    "wb_int128_010_rewrite": [False, True, False],
    "wb_complexf16_01_negzero_imag_rewrite": [False, True],
}


def marked(element, hex_payload):
    return StoredSeries(
        ANCHOR,
        np.frombuffer(bytes.fromhex(hex_payload), dtype=element.dtype).copy(),
        element.with_bool_marker(),
    )


# Reference-only controls: Python outcome, then the pinned Julia loader's outcome
# as asserted by the generator (documentation of the deliberate differences).
# A tuple names an accepted read: ("stored", expected container) for a
# StoredSeries, ("bool", values) for a Boolean TSeries.
CONTROLS: dict[str, tuple[object, str]] = {
    # Empty date/duration elements need no marker in Python; Julia cannot load them.
    "ctl_mit_32_empty_unmarked": (
        ("stored", StoredSeries(ANCHOR, np.zeros(0, "<i8"), StoredElement.date(Monthly()))),
        "MethodError",
    ),
    "ctl_dur_32_empty_unmarked": (
        ("stored", StoredSeries(ANCHOR, np.zeros(0, "<i8"), StoredElement.duration(Monthly()))),
        "MethodError",
    ),
    # Widths and kinds that never carry an element frequency.
    "ctl_dur_32_width4": (ValueError, "MethodError"),
    "ctl_mit_32_width16": (ValueError, "MethodError"),
    "ctl_unsigned_elfreq_32": (TypeError, "MethodError"),
    "ctl_float_elfreq_32": (TypeError, "MethodError"),
    "ctl_complex_elfreq_32": (TypeError, "MethodError"),
    # Exact marker accepted; contradictory markers refused without evaluation.
    "ctl_mit_32_marked_nonempty": (
        (
            "stored",
            StoredSeries.from_list(
                ANCHOR, StoredElement.date(Monthly()), [mm(2024, 11), mm(2024, 12)]
            ),
        ),
        "MIT{Monthly}",
    ),
    "ctl_mit_32_marker_quarterly": (TypeError, "MethodError"),
    "ctl_mit_32_marker_int64": (TypeError, "Int64"),
    "ctl_mit_32_marker_float64": (TypeError, "Float64"),
    "ctl_mit_32_marker_bool": (TypeError, "Bool (temporary boundary)"),
    "ctl_dur_32_marker_bool": (TypeError, "Bool (temporary boundary)"),
    "ctl_dur_32_marker_int64": (TypeError, "Int64"),
    "ctl_int128_marker_uint128": (TypeError, "UInt128"),
    "ctl_int128_marker_float64": (TypeError, "Float64"),
    "ctl_uint128_marker_int128": (TypeError, "Int128"),
    "ctl_complexf16_marker_float16": (TypeError, "Float16"),
    "ctl_kind1_empty_marker_uint128": (TypeError, "UInt128[]"),
    "ctl_kind1_empty_marker_complexf16": (TypeError, "ComplexF16[]"),
    "ctl_kind5_empty_marker_int128": (TypeError, "Int128[]"),
    # Noncanonical element frequency codes: a deliberate compatibility restriction.
    "ctl_mit_elfreq_14": (TypeError, "ErrorException"),
    "ctl_mit_elfreq_15": (TypeError, "ErrorException"),
    "ctl_mit_elfreq_33": (TypeError, "ErrorException"),
    "ctl_dur_elfreq_14": (TypeError, "ErrorException"),
    "ctl_mit_elfreq_16": (TypeError, "MIT{Weekly{0}}"),
    "ctl_mit_elfreq_64": (TypeError, "MIT{Quarterly{0}}"),
    "ctl_mit_elfreq_128": (TypeError, "MIT{HalfYearly{0}}"),
    "ctl_mit_elfreq_256": (TypeError, "MIT{Yearly{0}}"),
    **{f"ctl_mit_elfreq_{16 + d}": (TypeError, f"MIT{{Weekly{{{d}}}}}") for d in range(8, 16)},
    "ctl_dur_elfreq_16": (TypeError, "Duration{Weekly{0}}"),
    "ctl_dur_elfreq_24": (TypeError, "Duration{Weekly{8}}"),
    "ctl_dur_elfreq_64": (TypeError, "Duration{Quarterly{0}}"),
    # Foreign Bool markers on ordinary carriers convert exactly as Julia does.
    "fb_int16_01": (("bool", [False, True, False]), "Bool"),
    "fb_int16_2": (ValueError, "InexactError"),
    "fb_int16_neg1": (ValueError, "InexactError"),
    "fb_int64_01": (("bool", [False, True]), "Bool"),
    "fb_int128_01": (("stored", marked(INT128, "01" + "00" * 15 + "00" * 16)), "Bool"),
    "fb_uint8_01": (("bool", [False, True, True]), "Bool"),
    "fb_uint8_2": (ValueError, "InexactError"),
    "fb_uint64_1": (("bool", [True]), "Bool"),
    "fb_float16_01": (("bool", [False, True]), "Bool"),
    "fb_float32_half": (ValueError, "InexactError"),
    "fb_float64_01": (("bool", [False, True]), "Bool"),
    "fb_float64_negzero": (("bool", [False]), "Bool"),
    "fb_float64_nan": (ValueError, "InexactError"),
    "fb_float64_two": (ValueError, "InexactError"),
    "fb_complexf64_10": (("bool", [True, False]), "Bool"),
    "fb_complexf64_0i": (ValueError, "InexactError"),
    "fb_complexf16_1": (("stored", marked(COMPLEXF16, "003c0000")), "Bool"),
    "fb_int16_empty": (("bool", []), "Bool[]"),
    "fb_uint8_empty": (("bool", []), "Bool[]"),
    "fb_float64_empty": (("bool", []), "Bool[]"),
    "fb_complex_empty": (("bool", []), "Bool[]"),
    # Wide carriers with a Bool marker are preserved when their values are valid.
    "wb_uint128_01": (("stored", marked(UINT128, "00" * 16 + "01" + "00" * 15)), "Bool"),
    "wb_uint128_2": (ValueError, "InexactError"),
    "wb_uint128_hiword": (ValueError, "InexactError"),
    "wb_uint128_max": (ValueError, "InexactError"),
    "wb_int128_hiword": (ValueError, "InexactError"),
    "wb_int128_neg1": (ValueError, "InexactError"),
    "wb_int128_010": (
        ("stored", marked(INT128, "00" * 16 + "01" + "00" * 15 + "00" * 16)),
        "Bool",
    ),
    "wb_complexf16_negzero": (("stored", marked(COMPLEXF16, "00800000")), "Bool"),
    "wb_complexf16_01_negzero_imag": (("stored", marked(COMPLEXF16, "00000000003c0080")), "Bool"),
    "wb_complexf16_imag_one": (ValueError, "InexactError"),
    "wb_complexf16_nan": (ValueError, "InexactError"),
    "wb_complexf16_nan_payload_imag": (ValueError, "InexactError"),
    "wb_complexf16_inf": (ValueError, "InexactError"),
    "wb_complexf16_two": (ValueError, "InexactError"),
    "wb_complexf16_subnormal": (ValueError, "InexactError"),
    # Empty Bool-marked payloads carry no width: empty Boolean series.
    "wb_kind1_empty": (("bool", []), "Bool[]"),
    "wb_kind2_empty": (("bool", []), "Bool[]"),
    "wb_kind5_empty": (("bool", []), "Bool[]"),
    # Other foreign markers stay refused: planned parity work, not an exclusion.
    "fm_int16_as_int64": (TypeError, "Int64"),
    "fm_int64_as_float64": (TypeError, "Float64"),
    "fm_float64_as_int8": (TypeError, "InexactError"),
    "fm_int64_as_int128": (TypeError, "Int128"),
    "fm_int64_as_complexf16": (TypeError, "ComplexF16"),
    "fm_int64_as_mit": (TypeError, "MIT{Monthly}"),
    "fm_int64_unknown": (TypeError, "UndefVarError"),
}
OBJECT_COUNT = len(SHARED) + len(BOOL_CASES) + len(CANONICAL_BOOL) + len(CONTROLS)


def raw_rows(path, names=None):
    """Element kind, element frequency, axis, payload and marker per object name."""
    with closing(sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)) as sql:
        rows = sql.execute(
            "SELECT o.name,t.eltype,t.elfreq,a.frequency,a.data,a.length,t.value,"
            "(SELECT value FROM attributes x WHERE x.id=o.id AND x.name='jeltype') "
            "FROM objects o JOIN tseries t ON t.id=o.id JOIN axes a ON a.id=t.axis_id"
        ).fetchall()
    return {
        row[0]: (*row[1:6], row[6] or b"", row[7]) for row in rows if not names or row[0] in names
    }


def test_fixture_provenance_and_coverage():
    provenance = tomllib.loads(FIXTURE.with_suffix(".toml").read_text(encoding="utf-8"))
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    assert provenance["timeseriesecon_sha"] == "fc0a0d01aed4903ea6c64b12d78d0bdf68468df6"
    assert provenance["native_source_sha"] == "1a108688a044380f808bebf64079e32dbb9cd1a4"
    assert provenance["native_version"] == "0.4.0"
    assert provenance["layout"]["tseries_t"]["offsets"][2] == 36
    rows = raw_rows(FIXTURE)
    claimed = SHARED.keys() | BOOL_CASES.keys() | CANONICAL_BOOL.keys() | CONTROLS.keys()
    assert len(claimed) == OBJECT_COUNT == 238
    assert set(rows) == claimed
    assert "ctl_date_elfreq_none" not in rows  # refused by the native library
    for name, (expected, hex_payload) in SHARED.items():
        element = expected.element
        marker = element.julia_name if len(expected) == 0 else None
        first = expected.firstdate
        axis = next(code for code, f in FAMILIES.items() if f == first.frequency)
        assert rows[name] == (
            element.native_kind,
            element.native_frequency,
            axis,
            first.value,
            len(expected),
            bytes.fromhex(hex_payload),
            marker,
        ), name
    for name, (element, _, hex_payload) in BOOL_CASES.items():
        payload = bytes.fromhex(hex_payload)
        assert rows[name] == (element.native_kind, 0, 32, 24298, 2, payload, "Bool"), name
    for name, values in CANONICAL_BOOL.items():
        assert rows[name] == (1, 0, 32, 24298, len(values), bytes(values), "Bool"), name


@NATIVE
@pytest.mark.parametrize("name", sorted(SHARED))
def test_julia_written_represented_series_read_and_rewrite_identically(name, tmp_path):
    expected, hex_payload = SHARED[name]
    with open_dataecon(FIXTURE) as db:
        result = db.read_series(name)
    assert isinstance(result, StoredSeries)
    assert result == expected
    assert result.values.tobytes().hex() == hex_payload
    assert result.values.flags.owndata
    assert result.values.flags.writeable
    assert result.lastdate == expected.lastdate
    path = tmp_path / "rewrite.daec"
    with open_dataecon(path, "a") as db:
        db.write_series(name, result)
    assert raw_rows(path)[name] == raw_rows(FIXTURE, {name})[name]


@NATIVE
@pytest.mark.parametrize("name", sorted(BOOL_CASES))
def test_wide_bool_controls_preserve_and_convert(name, tmp_path):
    element, items, hex_payload = BOOL_CASES[name]
    with open_dataecon(FIXTURE) as db:
        result = db.read_series(name)
        converted = db.read_series(f"{name}_converted")
    assert result == StoredSeries.from_list(ANCHOR, element.with_bool_marker(), items)
    assert result.values.tobytes().hex() == hex_payload
    assert result.tolist() == items
    assert isinstance(converted, TSeries)
    assert converted.values.dtype == np.dtype(bool)
    assert converted.values.tolist() == [False, True]
    path = tmp_path / "bool.daec"
    with open_dataecon(path, "a") as db:
        db.write_series(name, result)
        db.write_series(f"{name}_converted", result.to_bool())
    fixture_rows = raw_rows(FIXTURE, {name, f"{name}_converted"})
    assert raw_rows(path) == fixture_rows


@NATIVE
@pytest.mark.parametrize("name", sorted(CANONICAL_BOOL))
def test_julia_canonical_bool_rewrites_read_as_bool(name):
    with open_dataecon(FIXTURE) as db:
        result = db.read_series(name)
    assert isinstance(result, TSeries)
    assert result.values.dtype == np.dtype(bool)
    assert result.values.tolist() == CANONICAL_BOOL[name]
    assert result.firstdate == ANCHOR


@NATIVE
@pytest.mark.parametrize("name", sorted(CONTROLS))
def test_control_outcomes(name):
    outcome, _julia = CONTROLS[name]
    with open_dataecon(FIXTURE) as db:
        if isinstance(outcome, type):
            with pytest.raises(outcome):
                db.read_series(name)
            return
        result = db.read_series(name)
    kind, expected = outcome
    if kind == "stored":
        assert isinstance(result, StoredSeries)
        assert result == expected
        if expected.element.marker is not None:
            assert result.to_bool().values.dtype == np.dtype(bool)
    else:
        assert isinstance(result, TSeries)
        assert result.values.dtype == np.dtype(bool)
        assert result.values.tolist() == expected
        assert result.firstdate == ANCHOR


@NATIVE
def test_native_refused_date_without_element_frequency_is_absent():
    with open_dataecon(FIXTURE) as db, pytest.raises(DataEconError) as info:
        db.read_series("ctl_date_elfreq_none")
    assert info.value.code == -989
