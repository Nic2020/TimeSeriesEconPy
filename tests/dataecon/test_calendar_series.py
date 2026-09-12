"""Daily, business-daily and weekly Float64 TSeries interchange rules.

The native axis stores the packed calendar code of the first observation and
the length. Python requires every observation date (the anchor alone for an
empty series) to lie inside the verified native window of its frequency and
to survive the native decode/encode round trip. Nothing here converts codes
through ``datetime``; the core constructors are used only inside its years.
"""

import datetime as dt
import hashlib
import importlib.util
import shutil
import sqlite3
import tomllib
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from tsecon import MIT, BDaily, Daily, TSeries, Unit, Weekly, bdaily, daily, mm, weekly
from tsecon.dataecon import DataEconError, open_dataecon
from tsecon.dataecon._codec import (
    decode_series,
    encode_series,
    series_frequency,
    validate_metadata,
)
from tsecon.mit import mit_to_date

NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built; configure TSECON_DATAECON_ROOT",
)
FIXTURES = Path(__file__).parent / "fixtures"
WINDOWS = {12: (-11980259, 11979954), 13: (-8557114, 8557110)}
WINDOWS.update(dict.fromkeys(range(17, 24), (-1711422, 1711422)))
FAMILIES = [("d", Daily(), 12), ("b", BDaily(), 13)] + [
    (f"w{day}", Weekly(day), 16 + day) for day in range(1, 8)
]
VALUES = [1.25, -2.5, 0.0, 4.75]
LONG = list(0.25 * np.arange(1, 1001))


def at(frequency, date):
    """The core's own code for a date (inside datetime's years only)."""
    if isinstance(frequency, Daily):
        return daily(date).value
    if isinstance(frequency, BDaily):
        return bdaily(date).value
    return weekly(date, frequency.end_day).value


def cases(frequency, code):
    """The Julia fixture's cases: (suffix, first code, values)."""
    lo, hi = WINDOWS[code]
    return [
        ("cross_year", at(frequency, dt.date(2024, 12, 30)), VALUES),
        ("leap", at(frequency, dt.date(2024, 2, 28)), VALUES[:3]),
        ("weekend", at(frequency, dt.date(2024, 1, 12)), VALUES[:2]),
        ("negative", -1, VALUES[:2]),
        ("zero", 0, VALUES[:1]),
        ("minimum", lo, VALUES[:1]),
        ("maximum", hi, VALUES[:1]),
        ("beyond_py_year", at(frequency, dt.date(9999, 12, 31)), [*VALUES, 8.5]),
        # Only the first date is packed: trailing codes may pass the window.
        ("last_beyond_maximum", hi, VALUES[:2]),
        ("long_span", hi - 499, LONG),
        ("empty", at(frequency, dt.date(2024, 1, 15)), []),
        ("empty_minimum", lo, []),
        ("empty_maximum", hi, []),
    ]


def assert_series(actual, frequency, first, values):
    """Check frequency, dates and owning values independently of byte encoding."""
    assert actual.frequency == frequency
    assert actual.firstdate == MIT(frequency, first)
    assert actual.lastdate == MIT(frequency, first + len(values) - 1)
    assert actual.values.dtype == np.float64
    assert actual.values.flags.owndata
    np.testing.assert_array_equal(actual.values, values)


def test_series_frequency_table():
    assert series_frequency(12) == Daily()
    assert series_frequency(13) == BDaily()
    for day in range(1, 8):
        assert series_frequency(16 + day) == Weekly(day)
    for code in (11, 14, 15, 16, 24, 31, 0, -1, 2**40):
        with pytest.raises(TypeError):
            series_frequency(code)
        with pytest.raises(TypeError):
            validate_metadata((2, 12, 4, 0, 1, 1, code, 0, 8))


def test_fixture_codes_agree_with_the_core_calendar():
    # Values the Julia writer produced; the core's daily/bdaily/weekly reproduce them.
    assert at(Daily(), dt.date(2024, 12, 30)) == 739250
    assert at(BDaily(), dt.date(2024, 1, 12)) == 527785
    assert at(Weekly(3), dt.date(2024, 12, 30)) == 105608
    assert at(Weekly(1), dt.date(2024, 12, 30)) == 105608
    assert at(Daily(), dt.date(9999, 12, 31)) == 3652059
    assert at(BDaily(), dt.date(9999, 12, 31)) == 2608615
    assert mit_to_date(MIT(BDaily(), 527786)) == dt.date(2024, 1, 15)


@pytest.mark.parametrize(("label", "frequency", "code"), FAMILIES)
def test_codec_roundtrip_snapshots_and_owns(label, frequency, code):
    for _suffix, first, values in cases(frequency, code):
        source = np.array(values, dtype=np.float64)
        encoded = encode_series(TSeries(MIT(frequency, first), source))
        assert encoded[:2] == (code, first)
        assert encoded[2] == source.tobytes()
        source[:] = -1
        assert_series(decode_series(*encoded), frequency, first, values)
    values = np.arange(8, dtype=np.float64)
    encoded = encode_series(TSeries(MIT(frequency, 100), values[::2]))
    result = decode_series(*encoded)
    assert_series(result, frequency, 100, [0, 2, 4, 6])
    result.values[0] = 99
    assert np.frombuffer(encoded[2], dtype=np.float64)[0] == 0


@pytest.mark.parametrize(("label", "frequency", "code"), FAMILIES)
def test_window_bounds_precede_encoding(label, frequency, code):
    lo, hi = WINDOWS[code]
    for first, length in ((lo - 1, 0), (lo - 1, 1), (hi + 1, 0), (hi + 1, 1), (hi + 1, 2)):
        with pytest.raises(ValueError, match="date range"):
            encode_series(TSeries(MIT(frequency, first), np.ones(length)))
        with pytest.raises(ValueError, match="date range"):
            validate_metadata((2, 12, 4, 0, 1, length, code, first, 8 * length))
    for first in (2**31, -(2**31), 2**40, -(2**40), 2**32 + 738900):
        with pytest.raises(ValueError, match="date range"):
            encode_series(TSeries(MIT(frequency, first), np.ones(1)))
    # Trailing codes are implicit: the first date alone must lie in the window.
    # The payload limit bounds the span at 16,777,216 values, so the last code
    # never exceeds hi + 16,777,215 (inside the signed 32-bit range).
    limit = 16_777_216
    for first, length in ((lo, 1), (hi, 1), (lo, 0), (hi, 0), (hi, 2), (hi - 3, 5), (hi, limit)):
        validate_metadata((2, 12, 4, 0, 1, length, code, first, 8 * length))
    assert hi + limit - 1 < 2**31 - 1
    with pytest.raises(ValueError, match="oversized"):
        validate_metadata((2, 12, 4, 0, 1, limit + 1, code, hi, 8 * (limit + 1)))


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.complex128, ">f8"])
@pytest.mark.parametrize("length", [0, 2])
def test_calendar_dtype_is_not_coerced(dtype, length):
    with pytest.raises(TypeError):
        encode_series(TSeries(MIT(Daily(), 738900), np.ones(length, dtype=dtype)))


@pytest.mark.parametrize("length", [0, 2])
def test_unit_series_remain_rejected(length):
    with pytest.raises(TypeError):
        encode_series(TSeries(MIT(Unit(), 5), np.ones(length)))


@NATIVE
@pytest.mark.parametrize(("label", "frequency", "code"), FAMILIES)
def test_julia_fixture_and_python_roundtrips(tmp_path, label, frequency, code):
    fixture = FIXTURES / "julia_calendar_series.daec"
    provenance = tomllib.loads(fixture.with_suffix(".toml").read_text(encoding="utf-8"))
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    path = tmp_path / "calendar.daec"
    loaded = []
    with open_dataecon(fixture) as reference, open_dataecon(path, "a") as db:
        for suffix, first, values in cases(frequency, code):
            name = f"cs_{label}_{suffix}"
            source = reference.read_series(name)
            assert_series(source, frequency, first, values)
            db.write_series(name, source)
            loaded.append((db.read_series(name), first, values))
            if values:
                source.values[0] = 99
        # Python's marker-free empty encoding, written natively by Julia's probe.
        native_empty = reference.read_series(f"cs_{label}_native_empty")
        assert_series(native_empty, frequency, at(frequency, dt.date(2024, 1, 15)), [])
        # Julia stored these; Python rejects them instead of misdating or trusting them.
        for name in (
            f"ctl_{label}_below_window",
            f"cs_{label}_native_first_above_maximum",
            f"cs_{label}_native_int32_wrap",
        ):
            with pytest.raises(ValueError, match="date range"):
                reference.read_series(name)
        with pytest.raises(DataEconError) as caught:
            db.write_series(f"cs_{label}_zero", TSeries(MIT(frequency, 0), np.ones(1)))
        assert caught.value.code == -985
        with pytest.raises(DataEconError) as caught:
            db.write_scalar(f"cs_{label}_zero", 1.25)
        assert caught.value.code == -985
    for result, first, values in loaded:
        assert_series(result, frequency, first, values)
    with open_dataecon(path) as db:
        for suffix, first, values in cases(frequency, code):
            assert_series(db.read_series(f"cs_{label}_{suffix}"), frequency, first, values)
        with pytest.raises(ValueError, match="read-only"):
            db.write_series("readonly", loaded[0][0])
    with pytest.raises(ValueError, match="closed"):
        db.read_series(f"cs_{label}_zero")
    with closing(sqlite3.connect(path)) as conn:
        for suffix, first, values in cases(frequency, code):
            assert conn.execute(
                "SELECT o.class,o.type,t.eltype,t.elfreq,a.ax_type,a.length,a.frequency,a.data,"
                "length(t.value),t.value IS NULL FROM objects o JOIN tseries t USING(id) "
                "JOIN axes a ON t.axis_id=a.id WHERE o.name=?",
                (f"cs_{label}_{suffix}",),
            ).fetchone() == (
                2,
                12,
                4,
                0,
                1,
                len(values),
                code,
                first,
                8 * len(values) if values else None,
                not values,
            )
        assert conn.execute("SELECT count(*) FROM attributes WHERE id != 0").fetchone() == (0,)


@NATIVE
def test_julia_fixture_controls():
    with open_dataecon(FIXTURES / "julia_calendar_series.daec") as db:
        # Julia's Weekly{8} collapsed to native code 17.
        assert_series(db.read_series("ctl_weekly8_series"), Weekly(1), 105557, [1.25])
        for name in ("native_axis_weekly16", "native_axis_weekly24", "native_axis_freq14"):
            with pytest.raises(TypeError):
                db.read_series(name)
        with pytest.raises(TypeError, match="reconstruction"):
            db.read_series("ctl_empty_float32_daily")
    with closing(sqlite3.connect(FIXTURES / "julia_calendar_series.daec")) as conn:
        # Julia marks its own empty series; Python writes no marker (checked above).
        assert conn.execute(
            "SELECT count(*) FROM attributes JOIN objects USING(id) WHERE objects.name LIKE "
            "'cs_%_empty%' AND objects.name NOT LIKE '%native%' AND attributes.name='jeltype' "
            "AND value='Float64'"
        ).fetchone() == (27,)


@NATIVE
@pytest.mark.parametrize(
    ("frequency", "first", "payload", "exception"),
    [
        (11, 5, bytes(8), TypeError),
        (14, 738900, b"", TypeError),
        (16, 105557, b"", TypeError),
        (24, 105557, b"", TypeError),
        (31, 105557, b"", TypeError),
        ("12", 738900, b"", TypeError),
        (12, 738900.0, b"", ValueError),
        (12, -11980260, b"", ValueError),
        (12, 11979955, b"", ValueError),
        (13, -8557115, bytes(8), ValueError),
        (13, 8557111, bytes(16), ValueError),
        (23, -1711423, b"", ValueError),
        (17, 1711423, bytes(16), ValueError),
        (12, 2**31, b"", ValueError),
        (12, 2**32 + 738900, b"", ValueError),
        (12, 2**40, b"", ValueError),
        (12, 738900, bytes(7), ValueError),
    ],
)
def test_backend_validates_before_storage(tmp_path, frequency, first, payload, exception):
    path = tmp_path / "invalid.daec"
    with open_dataecon(path, "a") as db:
        with pytest.raises(exception):
            db._handle.write("bad", frequency, first, payload)
        with pytest.raises(DataEconError) as caught:
            db.read_series("bad")
        assert caught.value.code == -989
        db.write_series("good", TSeries(MIT(Daily(), -11980259), np.ones(1)))
        assert_series(db.read_series("good"), Daily(), -11980259, [1])
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT count(*) FROM axes").fetchone() == (1,)


AXIS = "UPDATE axes SET {} WHERE id=(SELECT axis_id FROM tseries WHERE id=:id)"


@NATIVE
@pytest.mark.parametrize(
    ("sql", "error"),
    [
        (AXIS.format("data=-11980260"), ValueError),
        (AXIS.format("data=11979955"), ValueError),
        (AXIS.format("data=4295706196"), ValueError),
        (AXIS.format("length=5"), ValueError),
        (AXIS.format("frequency=16"), TypeError),
        (AXIS.format("frequency=11"), TypeError),
        (AXIS.format("frequency=14"), TypeError),
        (AXIS.format("ax_type=0"), TypeError),
        ("UPDATE tseries SET eltype=1 WHERE id=:id", TypeError),
        ("UPDATE tseries SET value=zeroblob(24) WHERE id=:id", ValueError),
        ("INSERT INTO attributes VALUES(:id,'jtype','TSeries')", TypeError),
        ("INSERT INTO attributes VALUES(:id,'jeltype','Float64')", TypeError),
    ],
)
def test_malformed_storage_rejected_before_pointer_use(tmp_path, sql, error):
    # cs_d_cross_year is four daily values from 30 December 2024 (code 739250).
    path = tmp_path / "malformed.daec"
    shutil.copyfile(FIXTURES / "julia_calendar_series.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        (oid,) = conn.execute("SELECT id FROM objects WHERE name='cs_d_cross_year'").fetchone()
        conn.execute(sql, {"id": oid})
    with open_dataecon(path, "a") as db:
        with pytest.raises(error):
            db.read_series("cs_d_cross_year")
        assert_series(db.read_series("cs_d_leap"), Daily(), 738944, VALUES[:3])
        db.write_series("good", TSeries(MIT(BDaily(), 527785), np.ones(2)))
        assert_series(db.read_series("good"), BDaily(), 527785, [1, 1])


@NATIVE
def test_empty_calendar_series_accept_only_the_exact_marker(tmp_path):
    path = tmp_path / "marker.daec"
    shutil.copyfile(FIXTURES / "julia_calendar_series.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "UPDATE attributes SET value='Float32' WHERE id=(SELECT id FROM objects "
            "WHERE name='cs_w7_empty')"
        )
    with open_dataecon(path) as db:
        with pytest.raises(TypeError, match="reconstruction"):
            db.read_series("cs_w7_empty")
        empty = db.read_series("cs_w6_empty")
        assert empty.lastdate == empty.firstdate - 1
        assert empty.values.shape == (0,)
        assert empty.values.flags.owndata


@NATIVE
def test_overwrite_between_calendar_series_and_scalars(tmp_path):
    day = TSeries(daily("2024-12-30"), np.array(VALUES))
    week = TSeries(weekly("2024-12-30", 3), np.array(VALUES))
    with open_dataecon(tmp_path / "overwrite.daec", "a") as db:
        db.write_scalar("x", 1)
        db.write_series("x", day, overwrite=True)
        assert_series(db.read_series("x"), Daily(), 739250, VALUES)
        db.write_series("x", week, overwrite=True)
        assert_series(db.read_series("x"), Weekly(3), 105608, VALUES)
        db.write_scalar("x", 2.5, overwrite=True)
        assert db.read_scalar("x") == 2.5
        with pytest.raises(DataEconError):
            db.read_series("x")
        # Validation still precedes the delete: the original survives a bad overwrite.
        db.write_series("y", day)
        with pytest.raises(ValueError, match="date range"):
            db.write_series("y", TSeries(MIT(Daily(), 11979955), np.ones(1)), overwrite=True)
        assert_series(db.read_series("y"), Daily(), 739250, VALUES)
        # A trailing observation past the window is allowed, as in Julia.
        db.write_series("y", TSeries(MIT(Daily(), 11979954), np.ones(2)), overwrite=True)
        assert_series(db.read_series("y"), Daily(), 11979954, [1, 1])


@NATIVE
def test_weekly_end_day_identity_and_business_days(tmp_path):
    with open_dataecon(tmp_path / "weeks.daec", "a") as db:
        for day in range(1, 8):
            db.write_series(f"w{day}", TSeries(MIT(Weekly(day), 105558), np.ones(2)))
        db.write_series("friday", TSeries(bdaily("2024-01-12"), np.array([1.0, 2.0])))
        series = [db.read_series(f"w{day}") for day in range(1, 8)]
        friday = db.read_series("friday")
    assert len({s.firstdate for s in series}) == 7
    assert [mit_to_date(s.firstdate).isoformat() for s in series] == [
        "2024-01-15",
        "2024-01-16",
        "2024-01-17",
        "2024-01-18",
        "2024-01-19",
        "2024-01-20",
        "2024-01-21",
    ]
    # Business days skip the weekend without any holiday calendar.
    assert mit_to_date(friday.lastdate) == dt.date(2024, 1, 15)
    assert friday.lastdate.value - friday.firstdate.value == 1


@NATIVE
def test_years_beyond_datetime_roundtrip_as_codes(tmp_path):
    start = MIT(Daily(), 3652059)  # 31 December 9999
    with open_dataecon(tmp_path / "far.daec", "a") as db:
        db.write_series("far", TSeries(start, np.array([*VALUES, 8.5])))
        db.write_series("max", TSeries(MIT(BDaily(), 8557110), np.ones(1)))
        far = db.read_series("far")
        assert_series(db.read_series("max"), BDaily(), 8557110, [1])
        # Trailing codes past every family's window, from the anchor at its maximum.
        for label, frequency, code in FAMILIES:
            hi = WINDOWS[code][1]
            db.write_series(f"trail_{label}", TSeries(MIT(frequency, hi), np.arange(3000.0)))
            assert_series(db.read_series(f"trail_{label}"), frequency, hi, np.arange(3000.0))
    assert_series(far, Daily(), 3652059, [*VALUES, 8.5])
    assert mit_to_date(far.firstdate) == dt.date(9999, 12, 31)
    # The core's datetime conversion stops at year 9999; the adapter does not use it.
    with pytest.raises((ValueError, OverflowError)):
        mit_to_date(far.lastdate)


@NATIVE
def test_calendar_and_year_period_series_share_the_namespace(tmp_path):
    with open_dataecon(tmp_path / "mixed.daec", "a") as db:
        db.write_series("m", TSeries(mm(2024, 1), np.ones(3)))
        db.write_series("d", TSeries(daily("2024-01-01"), np.ones(3)))
        with pytest.raises(DataEconError) as caught:
            db.write_series("m", TSeries(daily("2024-01-01"), np.ones(3)))
        assert caught.value.code == -985
        assert db.read_series("m").frequency == mm(2024, 1).frequency
        assert db.read_series("d").frequency == Daily()
