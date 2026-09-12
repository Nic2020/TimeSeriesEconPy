"""MIT date and Duration scalar codec rules, native interchange and guards.

Dates are validated as dates: each frequency has a reliable native code range.
Durations are counts of periods and only need to fit a signed 64-bit integer.
"""

import hashlib
import importlib.util
import shutil
import sqlite3
import struct
import tomllib
from contextlib import closing
from pathlib import Path

import pytest

from tsecon import (
    MIT,
    BDaily,
    Daily,
    Duration,
    HalfYearly,
    Monthly,
    Quarterly,
    Unit,
    Weekly,
    Yearly,
    mit2yp,
)
from tsecon.dataecon import DataEconError, open_dataecon
from tsecon.dataecon._codec import (
    decode_scalar,
    encode_scalar,
    validate_date_code,
    validate_scalar_metadata,
)

FIXTURES = Path(__file__).parent / "fixtures"
NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built",
)
MAXIMUM = 2**31 - 1
MINIMUMS = {12: -393600, 4: -131200, 2: -65600, 1: -(2**31)}
# Label, frequency and native code for every supported year/period family.
FAMILIES = (
    [("m", Monthly(), 32)]
    + [(f"q{a}", Quarterly(a), 64 + a) for a in (1, 2, 3)]
    + [(f"h{a}", HalfYearly(a), 128 + a) for a in range(1, 7)]
    + [(f"y{a}", Yearly(a), 256 + a) for a in range(1, 13)]
)
DURATION_VALUES = [
    ("zero", 0),
    ("one", 1),
    ("negative_one", -1),
    ("pow53_plus_one", 2**53 + 1),
    ("max", 2**63 - 1),
    ("min", -(2**63)),
]
# Unit and calendar controls stored by Julia in the date fixture; their full
# contract and regressions are in test_calendar.py.
UNIT_CALENDAR_CONTROLS = {
    "ctl_unit_mit": MIT(Unit(), 5),
    "ctl_unit_duration": Duration(Unit(), 3),
    "ctl_daily_mit": MIT(Daily(), 738900),
    "ctl_daily_duration": Duration(Daily(), 14),
    "ctl_bdaily_mit": MIT(BDaily(), 527786),
    "ctl_weekly7_mit": MIT(Weekly(7), 105557),
}


def date_codes(frequency):
    ppy = frequency.periods_per_year
    return [
        ("typical", 2024 * ppy),
        ("cross_year", 2024 * ppy + ppy - 1),
        ("negative_one", -1),
        ("zero", 0),
        ("minimum", MINIMUMS[ppy]),
        ("maximum", MAXIMUM),
    ]


def all_cases():
    for label, frequency, code in FAMILIES:
        for suffix, value in date_codes(frequency):
            yield f"mit_{label}_{suffix}", MIT(frequency, value), 3, code
        for suffix, value in DURATION_VALUES:
            yield f"dur_{label}_{suffix}", Duration(frequency, value), 1, code


def assert_exact(actual, expected):
    assert type(actual) is type(expected)
    assert actual == expected
    assert actual.frequency == expected.frequency
    assert actual.value == expected.value


@pytest.mark.parametrize(("name", "value", "kind", "code"), list(all_cases()))
def test_date_and_duration_codec_is_exact(name, value, kind, code):
    encoded = encode_scalar(value)
    assert encoded[:2] == (kind, code)
    assert encoded[2] == struct.pack("<q", value.value)
    assert_exact(decode_scalar(*encoded), value)


@pytest.mark.parametrize(("label", "frequency", "code"), FAMILIES)
def test_date_bounds_agree_with_core_decomposition(label, frequency, code):
    ppy = frequency.periods_per_year
    minimum = MINIMUMS[ppy]
    validate_date_code(code, minimum)
    validate_date_code(code, MAXIMUM)
    assert mit2yp(MIT(frequency, minimum)) == (minimum // ppy, 1)
    for bad in (minimum - 1, MAXIMUM + 1, 2**40, -(2**40), 2**63, -(2**63)):
        with pytest.raises(ValueError, match="reliable native date range"):
            validate_date_code(code, bad)
        with pytest.raises(ValueError, match="reliable native date range"):
            encode_scalar(MIT(frequency, bad))
        if -(2**63) <= bad <= 2**63 - 1:
            with pytest.raises(ValueError, match="reliable native date range"):
                decode_scalar(3, code, struct.pack("<q", bad))


@pytest.mark.parametrize(("label", "frequency", "code"), FAMILIES)
def test_duration_never_uses_date_bounds(label, frequency, code):
    for value in (MINIMUMS[frequency.periods_per_year] - 1, MAXIMUM + 1, 2**40, -(2**40)):
        kind, freq, payload = encode_scalar(Duration(frequency, value))
        assert (kind, freq) == (1, code)
        assert_exact(decode_scalar(kind, freq, payload), Duration(frequency, value))
    for value in (2**63, -(2**63) - 1):
        with pytest.raises(ValueError, match="signed 64-bit"):
            encode_scalar(Duration(frequency, value))


@pytest.mark.parametrize(
    ("value", "kind", "code"),
    [
        (MIT(Unit(), 5), 3, 11),
        (MIT(Daily(), 738900), 3, 12),
        (MIT(BDaily(), 527786), 3, 13),
        (MIT(Weekly(7), 105557), 3, 23),
        (MIT(Weekly(3), 105558), 3, 19),
        (Duration(Unit(), 3), 1, 11),
        (Duration(Daily(), 14), 1, 12),
        (Duration(Weekly(), 1), 1, 23),
    ],
)
def test_unit_and_calendar_frequencies_share_the_scalar_codec(value, kind, code):
    encoded = encode_scalar(value)
    assert encoded[:2] == (kind, code)
    assert_exact(decode_scalar(*encoded), value)


def test_fiscal_anchor_identity_is_preserved_in_codec():
    q1 = encode_scalar(MIT(Quarterly(1), 8096))
    q3 = encode_scalar(MIT(Quarterly(3), 8096))
    assert q1[2] == q3[2]
    assert q1[1] != q3[1]
    assert decode_scalar(*q1) != decode_scalar(*q3)
    assert decode_scalar(*q3) == MIT(Quarterly(), 8096)
    assert decode_scalar(*encode_scalar(Duration(Yearly(6), 2))) == Duration(Yearly(6), 2)
    assert decode_scalar(*encode_scalar(Duration(Yearly(6), 2))) != Duration(Yearly(), 2)


def test_date_duration_and_int64_kinds_are_distinct():
    assert encode_scalar(MIT(Monthly(), 5))[:2] == (3, 32)
    assert encode_scalar(Duration(Monthly(), 5))[:2] == (1, 32)
    assert encode_scalar(5)[:2] == (1, 0)
    assert type(decode_scalar(1, 0, struct.pack("<q", 5))) is int
    assert type(decode_scalar(1, 32, struct.pack("<q", 5))) is Duration
    assert type(decode_scalar(3, 32, struct.pack("<q", 5))) is MIT


@pytest.mark.parametrize(
    "metadata",
    [
        (1, 3, 0, 8),  # a date needs a frequency
        (1, 3, 14, 8),  # unused codes between business daily and weekly
        (1, 3, 15, 8),
        (1, 3, 16, 8),  # Sunday alias never written by Julia (loads as Weekly{0})
        (1, 3, 24, 8),  # weekly anchor 8
        (1, 3, 31, 8),  # weekly anchor 15
        (1, 3, 33, 8),  # monthly alias never written by Julia; Julia fails to load it
        (1, 3, 64, 8),  # bare quarterly
        (1, 3, 128, 8),  # bare half-yearly
        (1, 3, 256, 8),  # bare yearly
        (1, 3, 269, 8),  # yearly anchor 13
        (1, 3, 192, 8),  # mixed bits
        (1, 1, 16, 8),  # weekly alias duration
        (1, 1, 192, 8),
        (1, 4, 32, 8),  # Float64 with a frequency
        (1, 2, 32, 8),
        (2, 3, 32, 8),
    ],
)
def test_date_metadata_guard_rejects_unsupported_frequencies(metadata):
    with pytest.raises(TypeError):
        validate_scalar_metadata(metadata)


@pytest.mark.parametrize("nbytes", [0, 4, 16, -8])
@pytest.mark.parametrize("kind", [1, 3])
def test_date_metadata_guard_rejects_other_widths(kind, nbytes):
    with pytest.raises(ValueError, match="eight"):
        validate_scalar_metadata((1, kind, 32, nbytes))


def test_date_metadata_guard_accepts_every_family():
    for _, _, code in FAMILIES:
        validate_scalar_metadata((1, 3, code, 8))
        validate_scalar_metadata((1, 1, code, 8))


@NATIVE
def test_julia_date_fixture_values_and_controls():
    fixture = FIXTURES / "julia_date_scalars.daec"
    provenance = tomllib.loads(fixture.with_suffix(".toml").read_text(encoding="utf-8"))
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    cases = list(all_cases())
    with open_dataecon(fixture) as db:
        results = [(db.read_scalar(name), value) for name, value, _, _ in cases]
        for label, frequency, _ in FAMILIES:
            if frequency.periods_per_year > 1:
                # Julia stored the code intact but misdates it on load; Python rejects it.
                with pytest.raises(ValueError, match="reliable native date range"):
                    db.read_scalar(f"ctl_mit_{label}_below_minimum")
        for name, expected in UNIT_CALENDAR_CONTROLS.items():
            assert_exact(db.read_scalar(name), expected)
        # Julia's Quarterly{4} collapses to native code 65: Python reads Quarterly(1).
        assert_exact(db.read_scalar("ctl_quarterly4_mit"), MIT(Quarterly(1), 8096))
        for name, error in {
            "native_date_no_freq": TypeError,
            "native_date_four_bytes": ValueError,
            "native_date_mixed_bits": TypeError,
            "native_date_bare_quarterly": TypeError,
            "native_duration_four_bytes": ValueError,
        }.items():
            with pytest.raises(error):
                db.read_scalar(name)
    for actual, expected in results:
        assert_exact(actual, expected)
    with closing(sqlite3.connect(fixture)) as conn:
        for name, value, kind, code in cases:
            row = conn.execute(
                "SELECT o.class,o.type,s.frequency,s.value FROM objects o JOIN scalars s "
                "USING(id) WHERE o.name=?",
                (name,),
            ).fetchone()
            assert row == (1, kind, code, struct.pack("<q", value.value))
        assert conn.execute("SELECT count(*) FROM attributes WHERE id!=0").fetchone() == (0,)


@NATIVE
def test_date_roundtrip_storage_and_shared_namespace(tmp_path):
    path = tmp_path / "dates.daec"
    cases = list(all_cases())
    with open_dataecon(path, "a") as db:
        for name, value, _, _ in cases:
            db.write_scalar(name, value)
        db.write_scalar("count", 24288)
        db.write_scalar("text", "2024M1")
        results = [(db.read_scalar(name), value) for name, value, _, _ in cases]
        with pytest.raises(DataEconError) as caught:
            db.write_scalar("mit_m_typical", MIT(Monthly(), 1))
        assert caught.value.code == -985
        with pytest.raises(DataEconError):
            db.write_scalar("count", MIT(Monthly(), 24288))
        with pytest.raises(DataEconError):
            db.read_series("mit_m_typical")
        with pytest.raises(ValueError, match="reliable native date range"):
            db.write_scalar("bad", MIT(Monthly(), -393601))
        with pytest.raises(ValueError, match="reliable native date range"):
            db.write_scalar("bad", MIT(HalfYearly(1), MAXIMUM + 1))
        with pytest.raises(ValueError, match="reliable native date range"):
            db.write_scalar("bad", MIT(Daily(), 11979955))
        with pytest.raises(DataEconError) as caught:
            db.read_scalar("bad")
        assert caught.value.code == -989
        assert type(db.read_scalar("count")) is int
        assert type(db.read_scalar("text")) is str
    for actual, expected in results:
        assert_exact(actual, expected)
    with pytest.raises(ValueError, match="closed"):
        db.write_scalar("later", MIT(Monthly(), 1))
    with open_dataecon(path) as db:
        with pytest.raises(ValueError, match="read-only"):
            db.write_scalar("readonly", Duration(Monthly(), 1))
        assert_exact(db.read_scalar("dur_y12_min"), Duration(Yearly(12), -(2**63)))
    with closing(sqlite3.connect(path)) as conn:
        for name, value, kind, code in cases:
            assert conn.execute(
                "SELECT o.class,o.type,s.frequency,s.value FROM objects o JOIN scalars s "
                "USING(id) WHERE o.name=?",
                (name,),
            ).fetchone() == (1, kind, code, struct.pack("<q", value.value))
        assert conn.execute("SELECT count(*) FROM attributes WHERE id!=0").fetchone() == (0,)


@NATIVE
@pytest.mark.parametrize(
    ("kind", "frequency", "payload", "error"),
    [
        (3, 0, struct.pack("<q", 24288), TypeError),
        (3, 16, struct.pack("<q", 24288), TypeError),
        (3, 192, struct.pack("<q", 24288), TypeError),
        (3, 32, struct.pack("<q", -393601), ValueError),
        (3, 65, struct.pack("<q", -131201), ValueError),
        (3, 129, struct.pack("<q", -65601), ValueError),
        (3, 257, struct.pack("<q", -(2**31) - 1), ValueError),
        (3, 32, struct.pack("<q", 2**31), ValueError),
        (3, 32, bytes(4), ValueError),
        (1, 14, struct.pack("<q", 1), TypeError),
        (1, 32, bytes(4), ValueError),
        (3, "32", struct.pack("<q", 24288), TypeError),
    ],
)
def test_date_backend_validates_before_storage(tmp_path, kind, frequency, payload, error):
    with open_dataecon(tmp_path / "invalid.daec", "a") as db:
        with pytest.raises(error):
            db._handle.write_scalar("bad", kind, frequency, payload)
        with pytest.raises(DataEconError) as caught:
            db.read_scalar("bad")
        assert caught.value.code == -989
        db.write_scalar("good", MIT(Monthly(), -393600))
        assert_exact(db.read_scalar("good"), MIT(Monthly(), -393600))


@NATIVE
@pytest.mark.parametrize(
    ("sql", "error"),
    [
        ("UPDATE scalars SET value=NULL", ValueError),
        ("UPDATE scalars SET value=zeroblob(4)", ValueError),
        (f"UPDATE scalars SET value=X'{struct.pack('<q', -393601).hex()}'", ValueError),
        (f"UPDATE scalars SET value=X'{struct.pack('<q', 2**31).hex()}'", ValueError),
        ("UPDATE scalars SET frequency=16", TypeError),
        ("UPDATE scalars SET frequency=0", TypeError),
        ("UPDATE scalars SET frequency=64", TypeError),
        ("UPDATE objects SET type=4", TypeError),
        ("UPDATE objects SET type=6", TypeError),
    ],
)
def test_date_malformed_storage(tmp_path, sql, error):
    path = tmp_path / "malformed.daec"
    shutil.copyfile(FIXTURES / "julia_date_scalars.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(f"{sql} WHERE id=(SELECT id FROM objects WHERE name='mit_m_typical')")
    with open_dataecon(path, "a") as db:
        with pytest.raises(error):
            db.read_scalar("mit_m_typical")
        assert_exact(db.read_scalar("mit_m_zero"), MIT(Monthly(), 0))
        db.write_scalar("good", Duration(Monthly(), 3))
        assert_exact(db.read_scalar("good"), Duration(Monthly(), 3))


@NATIVE
def test_duration_malformed_storage_keeps_frequency_semantics(tmp_path):
    path = tmp_path / "malformed.daec"
    shutil.copyfile(FIXTURES / "julia_date_scalars.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        for name, frequency in (("dur_m_one", 0), ("dur_m_max", 67)):
            conn.execute(
                "UPDATE scalars SET frequency=? WHERE id=(SELECT id FROM objects WHERE name=?)",
                (frequency, name),
            )
    with open_dataecon(path) as db:
        # Type 1 without a frequency is a plain Int64; with another supported
        # frequency it is that frequency's Duration. The format cannot tell more.
        assert type(db.read_scalar("dur_m_one")) is int
        assert_exact(db.read_scalar("dur_m_max"), Duration(Quarterly(3), 2**63 - 1))


@NATIVE
@pytest.mark.parametrize("key", ["jtype", "jeltype"])
@pytest.mark.parametrize("value", ["MIT{Monthly}", "Duration{Monthly}", "error(123)", None])
def test_date_rejects_all_reconstruction_attributes(tmp_path, key, value):
    path = tmp_path / "attribute.daec"
    shutil.copyfile(FIXTURES / "julia_date_scalars.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "INSERT INTO attributes SELECT id,?,? FROM objects WHERE name IN "
            "('mit_m_typical','dur_m_one')",
            (key, value),
        )
    with open_dataecon(path) as db:
        with pytest.raises(TypeError, match="reconstruction"):
            db.read_scalar("mit_m_typical")
        with pytest.raises(TypeError, match="reconstruction"):
            db.read_scalar("dur_m_one")
        assert_exact(db.read_scalar("mit_m_zero"), MIT(Monthly(), 0))
