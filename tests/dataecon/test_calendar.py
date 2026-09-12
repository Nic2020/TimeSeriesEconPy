"""Unit and calendar (Daily, BDaily, Weekly) MIT date and Duration scalar rules.

Unit codes are Julia's plain signed 64-bit pass-through with no date bound.
Calendar codes must lie inside the exact native windows (daily from 1 March
-32800, business daily and weekly from the last week of December -32800, all
through December 32800) and are confirmed by a native decode/encode round trip.
Durations stay signed 64-bit counts. Nothing here converts through datetime.
"""

import datetime as dt
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
    Monthly,
    Unit,
    Weekly,
    bdaily,
    daily,
    mit_to_date,
    weekly,
)
from tsecon.dataecon import DataEconError, open_dataecon
from tsecon.dataecon._codec import (
    decode_scalar,
    encode_scalar,
    scalar_frequency,
    scalar_frequency_code,
    validate_date_code,
    validate_scalar_metadata,
)

FIXTURES = Path(__file__).parent / "fixtures"
NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built",
)
INT64 = (-(2**63), 2**63 - 1)
WINDOWS = {12: (-11980259, 11979954), 13: (-8557114, 8557110)}
WINDOWS.update(dict.fromkeys(range(17, 24), (-1711422, 1711422)))
FAMILIES = [("d", Daily(), 12), ("b", BDaily(), 13)] + [
    (f"w{day}", Weekly(day), 16 + day) for day in range(1, 8)
]
# Julia-generated codes; the fixture test checks them against the Julia file.
CASES = {
    "d": (
        ("typical", 738900),  # 2024-01-15
        ("year_end", 739251),  # 2024-12-31
        ("year_start", 739252),  # 2025-01-01
        ("leap_day", 738945),  # 2024-02-29
        ("first_day", 1),  # 0001-01-01
        ("year_zero", -199),  # 0000-06-15
        ("negative_year", -2130),  # -0005-03-03
        ("negative_one", -1),
        ("zero", 0),
        ("minimum", -11980259),  # -32800-03-01
        ("maximum", 11979954),  # 32800-12-31
        ("py_max_year", 3652059),  # 9999-12-31
        ("beyond_py_year", 3652062),  # 10000-01-03
    ),
    "b": (
        ("typical", 527786),
        ("year_end", 528037),
        ("year_start", 528038),
        ("leap_day", 527819),
        ("first_day", 1),
        ("year_zero", -141),
        ("negative_year", -1520),
        ("negative_one", -1),
        ("zero", 0),
        ("minimum", -8557114),  # -32800-12-25
        ("maximum", 8557110),  # 32800-12-29
        ("py_max_year", 2608615),
        ("beyond_py_year", 2608616),
    ),
}
_WEEKLY = {
    # end day: (typical, year_end, year_start, leap_day, year_zero, negative_year,
    #           py_max_year, beyond_py_year)
    1: (105558, 105609, 105609, 105565, -27, -303, 521724, 521724),
    2: (105558, 105608, 105609, 105565, -27, -303, 521724, 521724),
    3: (105558, 105608, 105608, 105565, -27, -303, 521724, 521724),
    4: (105558, 105608, 105608, 105564, -28, -303, 521724, 521724),
    5: (105558, 105608, 105608, 105564, -28, -304, 521723, 521724),
    6: (105558, 105608, 105608, 105564, -28, -304, 521723, 521724),
    7: (105558, 105608, 105608, 105564, -28, -304, 521723, 521724),
}
for _day, (_t, _ye, _ys, _ld, _yz, _ny, _pm, _bp) in _WEEKLY.items():
    CASES[f"w{_day}"] = (
        ("typical", _t),
        ("year_end", _ye),
        ("year_start", _ys),
        ("leap_day", _ld),
        ("first_day", 1),
        ("year_zero", _yz),
        ("negative_year", _ny),
        ("negative_one", -1),
        ("zero", 0),
        ("minimum", -1711422),
        ("maximum", 1711422),
        ("py_max_year", _pm),
        ("beyond_py_year", _bp),
    )
UNIT_VALUES = [
    ("min", -(2**63)),
    ("neg_pow40", -(2**40)),
    ("below_int32", -(2**31) - 1),
    ("int32_min", -(2**31)),
    ("negative_one", -1),
    ("zero", 0),
    ("five", 5),
    ("int32_max", 2**31 - 1),
    ("beyond_int32", 2**31),
    ("pow40", 2**40),
    ("max", 2**63 - 1),
]
DURATION_VALUES = [
    ("zero", 0),
    ("one", 1),
    ("negative_one", -1),
    ("pow53_plus_one", 2**53 + 1),
    ("max", 2**63 - 1),
    ("min", -(2**63)),
]


def all_cases():
    for label, frequency, code in FAMILIES:
        for suffix, value in CASES[label]:
            yield f"mit_{label}_{suffix}", MIT(frequency, value), 3, code
        for suffix, value in DURATION_VALUES:
            yield f"dur_{label}_{suffix}", Duration(frequency, value), 1, code
    for suffix, value in UNIT_VALUES:
        yield f"mit_u_{suffix}", MIT(Unit(), value), 3, 11
        yield f"dur_u_{suffix}", Duration(Unit(), value), 1, 11


def assert_exact(actual, expected):
    assert type(actual) is type(expected)
    assert actual == expected
    assert actual.frequency == expected.frequency
    assert actual.value == expected.value


def test_frequency_code_table():
    assert scalar_frequency_code(Unit()) == 11
    assert scalar_frequency_code(Daily()) == 12
    assert scalar_frequency_code(BDaily()) == 13
    for day in range(1, 8):
        assert scalar_frequency_code(Weekly(day)) == 16 + day
        assert scalar_frequency(16 + day) == Weekly(day)
    assert scalar_frequency(11) == Unit()
    for code in (14, 15, 16, 24, 31):
        with pytest.raises(TypeError):
            scalar_frequency(code)


@pytest.mark.parametrize(("name", "value", "kind", "code"), list(all_cases()))
def test_codec_is_exact(name, value, kind, code):
    encoded = encode_scalar(value)
    assert encoded[:2] == (kind, code)
    assert encoded[2] == struct.pack("<q", value.value)
    assert_exact(decode_scalar(*encoded), value)


@pytest.mark.parametrize(("label", "frequency", "code"), FAMILIES)
def test_calendar_window_bounds(label, frequency, code):
    lo, hi = WINDOWS[code]
    validate_date_code(code, lo)
    validate_date_code(code, hi)
    for bad in (lo - 1, hi + 1, 2**31, -(2**31), 2**40, -(2**40), *INT64):
        with pytest.raises(ValueError, match="reliable native date range"):
            validate_date_code(code, bad)
        with pytest.raises(ValueError, match="reliable native date range"):
            encode_scalar(MIT(frequency, bad))
        with pytest.raises(ValueError, match="reliable native date range"):
            decode_scalar(3, code, struct.pack("<q", bad))


@pytest.mark.parametrize(("label", "frequency", "code"), FAMILIES)
def test_calendar_durations_never_use_date_bounds(label, frequency, code):
    lo, hi = WINDOWS[code]
    for value in (lo - 1, hi + 1, 2**40, -(2**40), *INT64):
        kind, freq, payload = encode_scalar(Duration(frequency, value))
        assert (kind, freq) == (1, code)
        assert_exact(decode_scalar(kind, freq, payload), Duration(frequency, value))
    for value in (2**63, -(2**63) - 1):
        with pytest.raises(ValueError, match="signed 64-bit"):
            encode_scalar(Duration(frequency, value))


def test_unit_codes_are_unbounded_int64_pass_through():
    for value in (*INT64, 0, 2**31, -(2**31) - 1):
        for cls in (MIT, Duration):
            kind, freq, payload = encode_scalar(cls(Unit(), value))
            assert freq == 11
            assert payload == struct.pack("<q", value)
            assert_exact(decode_scalar(kind, freq, payload), cls(Unit(), value))
    for value in (2**63, -(2**63) - 1):
        with pytest.raises(ValueError, match="64-bit"):
            encode_scalar(MIT(Unit(), value))
        with pytest.raises(ValueError, match="64-bit"):
            encode_scalar(Duration(Unit(), value))


def test_python_core_dates_agree_with_the_fixture_codes():
    # Inside datetime's years the core's date conversions reproduce the Julia codes.
    expected = {
        ("d", "typical"): dt.date(2024, 1, 15),
        ("d", "leap_day"): dt.date(2024, 2, 29),
        ("d", "py_max_year"): dt.date(9999, 12, 31),
        ("b", "typical"): dt.date(2024, 1, 15),
        ("b", "year_end"): dt.date(2024, 12, 31),
        ("b", "first_day"): dt.date(1, 1, 1),
        ("w3", "typical"): dt.date(2024, 1, 17),
        ("w7", "leap_day"): dt.date(2024, 3, 3),
        ("w1", "first_day"): dt.date(1, 1, 1),
    }
    for (label, suffix), date in expected.items():
        frequency = {lbl: freq for lbl, freq, _ in FAMILIES}[label]
        code = dict(CASES[label])[suffix]
        assert mit_to_date(MIT(frequency, code)) == date
        if isinstance(frequency, Daily):
            assert daily(date) == MIT(frequency, code)
        elif isinstance(frequency, BDaily):
            assert bdaily(date) == MIT(frequency, code)
        else:
            assert weekly(date, frequency.end_day) == MIT(frequency, code)


def test_weekly_end_day_identity_is_preserved():
    monday = encode_scalar(MIT(Weekly(1), 105558))
    sunday = encode_scalar(MIT(Weekly(7), 105558))
    assert monday[2] == sunday[2]
    assert monday[1] != sunday[1]
    assert decode_scalar(*monday) != decode_scalar(*sunday)
    assert decode_scalar(*sunday) == MIT(Weekly(), 105558)
    assert decode_scalar(*encode_scalar(Duration(Weekly(3), 2))) == Duration(Weekly(3), 2)
    assert decode_scalar(*encode_scalar(Duration(Weekly(3), 2))) != Duration(Weekly(), 2)


@pytest.mark.parametrize("metadata", [(1, 3, code, 8) for code in (11, 12, 13, *range(17, 24))])
def test_metadata_guard_accepts_unit_and_calendar(metadata):
    validate_scalar_metadata(metadata)
    validate_scalar_metadata((1, 1, metadata[2], 8))


@pytest.mark.parametrize("code", [14, 15, 16, 24, 31])
def test_metadata_guard_rejects_unwritten_calendar_codes(code):
    with pytest.raises(TypeError):
        validate_scalar_metadata((1, 3, code, 8))
    with pytest.raises(TypeError):
        validate_scalar_metadata((1, 1, code, 8))


@pytest.mark.parametrize("nbytes", [0, 4, 16])
@pytest.mark.parametrize("code", [11, 12, 13, 23])
def test_metadata_guard_rejects_other_widths(code, nbytes):
    with pytest.raises(ValueError, match="eight"):
        validate_scalar_metadata((1, 3, code, nbytes))


@NATIVE
def test_julia_fixture_values_and_controls():
    fixture = FIXTURES / "julia_unit_calendar_scalars.daec"
    provenance = tomllib.loads(fixture.with_suffix(".toml").read_text(encoding="utf-8"))
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    cases = list(all_cases())
    with open_dataecon(fixture) as db:
        results = [(db.read_scalar(name), value) for name, value, _, _ in cases]
        # Julia stored these below-window codes with a warning; Python rejects them
        # whether Julia later misdates them (daily, business daily) or not (weekly).
        for label in ("d", "b", "w7"):
            with pytest.raises(ValueError, match="reliable native date range"):
                db.read_scalar(f"ctl_mit_{label}_below_window")
        # Julia's Weekly{8} collapses to native code 17: Python reads Weekly(1).
        assert_exact(db.read_scalar("ctl_weekly8_mit"), MIT(Weekly(1), 105557))
        for name, error in {
            "native_date_weekly16": TypeError,
            "native_date_freq14": TypeError,
            "native_date_weekly24": TypeError,
            "native_date_unit_four_bytes": ValueError,
            "native_date_daily_above_maximum": ValueError,
            "native_date_daily_int32_wrap": ValueError,
            "native_duration_weekly16": TypeError,
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
def test_roundtrip_storage_and_shared_namespace(tmp_path):
    path = tmp_path / "calendar.daec"
    cases = list(all_cases())
    with open_dataecon(path, "a") as db:
        for name, value, _, _ in cases:
            db.write_scalar(name, value)
        db.write_scalar("count", 738900)
        db.write_scalar("month", MIT(Monthly(), 24288))
        results = [(db.read_scalar(name), value) for name, value, _, _ in cases]
        with pytest.raises(DataEconError) as caught:
            db.write_scalar("mit_d_typical", MIT(Daily(), 1))
        assert caught.value.code == -985
        with pytest.raises(DataEconError):
            db.write_scalar("count", MIT(Unit(), 1))
        with pytest.raises(DataEconError):
            db.read_series("mit_u_five")
        for bad in (MIT(Daily(), -11980260), MIT(BDaily(), 8557111), MIT(Weekly(1), 2**31)):
            with pytest.raises(ValueError, match="reliable native date range"):
                db.write_scalar("bad", bad)
        with pytest.raises(ValueError, match="64-bit"):
            db.write_scalar("bad", MIT(Unit(), 2**63))
        with pytest.raises(DataEconError) as caught:
            db.read_scalar("bad")
        assert caught.value.code == -989
        assert type(db.read_scalar("count")) is int
        assert_exact(db.read_scalar("month"), MIT(Monthly(), 24288))
    for actual, expected in results:
        assert_exact(actual, expected)
    with pytest.raises(ValueError, match="closed"):
        db.write_scalar("later", MIT(Unit(), 1))
    with open_dataecon(path) as db:
        with pytest.raises(ValueError, match="read-only"):
            db.write_scalar("readonly", Duration(Daily(), 1))
        assert_exact(db.read_scalar("mit_u_min"), MIT(Unit(), -(2**63)))
        assert_exact(db.read_scalar("mit_w7_maximum"), MIT(Weekly(7), 1711422))
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
        (3, 16, struct.pack("<q", 105557), TypeError),
        (3, 14, struct.pack("<q", 738900), TypeError),
        (3, 24, struct.pack("<q", 105557), TypeError),
        (3, 12, struct.pack("<q", -11980260), ValueError),
        (3, 12, struct.pack("<q", 11979955), ValueError),
        (3, 13, struct.pack("<q", -8557115), ValueError),
        (3, 13, struct.pack("<q", 8557111), ValueError),
        (3, 23, struct.pack("<q", -1711423), ValueError),
        (3, 17, struct.pack("<q", 1711423), ValueError),
        (3, 12, struct.pack("<q", 2**32 + 738900), ValueError),
        (3, 11, bytes(4), ValueError),
        (3, 12, bytes(16), ValueError),
        (1, 16, struct.pack("<q", 2), TypeError),
        (1, 11, bytes(4), ValueError),
        (3, "11", struct.pack("<q", 5), TypeError),
    ],
)
def test_backend_validates_before_storage(tmp_path, kind, frequency, payload, error):
    with open_dataecon(tmp_path / "invalid.daec", "a") as db:
        with pytest.raises(error):
            db._handle.write_scalar("bad", kind, frequency, payload)
        with pytest.raises(DataEconError) as caught:
            db.read_scalar("bad")
        assert caught.value.code == -989
        db.write_scalar("good", MIT(Daily(), -11980259))
        assert_exact(db.read_scalar("good"), MIT(Daily(), -11980259))


@NATIVE
@pytest.mark.parametrize(
    ("sql", "error"),
    [
        ("UPDATE scalars SET value=NULL", ValueError),
        ("UPDATE scalars SET value=zeroblob(4)", ValueError),
        (f"UPDATE scalars SET value=X'{struct.pack('<q', -11980260).hex()}'", ValueError),
        (f"UPDATE scalars SET value=X'{struct.pack('<q', 11979955).hex()}'", ValueError),
        (f"UPDATE scalars SET value=X'{struct.pack('<q', 2**31).hex()}'", ValueError),
        ("UPDATE scalars SET frequency=16", TypeError),
        ("UPDATE scalars SET frequency=14", TypeError),
        ("UPDATE scalars SET frequency=0", TypeError),
        ("UPDATE objects SET type=4", TypeError),
        ("UPDATE objects SET type=6", TypeError),
    ],
)
def test_malformed_storage(tmp_path, sql, error):
    path = tmp_path / "malformed.daec"
    shutil.copyfile(FIXTURES / "julia_unit_calendar_scalars.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(f"{sql} WHERE id=(SELECT id FROM objects WHERE name='mit_d_typical')")
    with open_dataecon(path, "a") as db:
        with pytest.raises(error):
            db.read_scalar("mit_d_typical")
        assert_exact(db.read_scalar("mit_d_zero"), MIT(Daily(), 0))
        db.write_scalar("good", Duration(BDaily(), 3))
        assert_exact(db.read_scalar("good"), Duration(BDaily(), 3))


@NATIVE
def test_unit_codes_are_never_range_checked_on_read(tmp_path):
    path = tmp_path / "unit.daec"
    shutil.copyfile(FIXTURES / "julia_unit_calendar_scalars.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        # A daily code far outside its window becomes a valid Unit code.
        conn.execute(
            "UPDATE scalars SET frequency=11 WHERE id=(SELECT id FROM objects WHERE name=?)",
            ("mit_d_typical",),
        )
        conn.execute(
            "UPDATE scalars SET value=? WHERE id=(SELECT id FROM objects WHERE name=?)",
            (struct.pack("<q", 2**62 + 3), "mit_u_five"),
        )
    with open_dataecon(path) as db:
        assert_exact(db.read_scalar("mit_d_typical"), MIT(Unit(), 738900))
        assert_exact(db.read_scalar("mit_u_five"), MIT(Unit(), 2**62 + 3))


@NATIVE
def test_duration_frequency_semantics_follow_the_stored_code(tmp_path):
    path = tmp_path / "duration.daec"
    shutil.copyfile(FIXTURES / "julia_unit_calendar_scalars.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        for name, frequency in (("dur_d_one", 0), ("dur_b_max", 19), ("dur_u_five", 12)):
            conn.execute(
                "UPDATE scalars SET frequency=? WHERE id=(SELECT id FROM objects WHERE name=?)",
                (frequency, name),
            )
    with open_dataecon(path) as db:
        assert type(db.read_scalar("dur_d_one")) is int
        assert_exact(db.read_scalar("dur_b_max"), Duration(Weekly(3), 2**63 - 1))
        assert_exact(db.read_scalar("dur_u_five"), Duration(Daily(), 5))


@NATIVE
@pytest.mark.parametrize("key", ["jtype", "jeltype"])
@pytest.mark.parametrize("value", ["MIT{Daily}", "MIT{Unit}", "error(123)", None])
def test_rejects_all_reconstruction_attributes(tmp_path, key, value):
    path = tmp_path / "attribute.daec"
    shutil.copyfile(FIXTURES / "julia_unit_calendar_scalars.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "INSERT INTO attributes SELECT id,?,? FROM objects WHERE name IN "
            "('mit_d_typical','mit_u_five','dur_w7_one')",
            (key, value),
        )
    with open_dataecon(path) as db:
        for name in ("mit_d_typical", "mit_u_five", "dur_w7_one"):
            with pytest.raises(TypeError, match="reconstruction"):
                db.read_scalar(name)
        assert_exact(db.read_scalar("mit_d_zero"), MIT(Daily(), 0))
