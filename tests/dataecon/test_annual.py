"""Annual fiscal endings, native date bounds and Julia interchange."""

import hashlib
import importlib.util
import shutil
import sqlite3
import tomllib
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from tsecon import MIT, TSeries, Yearly
from tsecon.dataecon import DataEconError, open_dataecon
from tsecon.dataecon._codec import decode_series, encode_series, validate_metadata
from tsecon.mit import mit_to_date

NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built; configure TSECON_DATAECON_ROOT",
)
FIXTURE = Path(__file__).parent / "fixtures/julia_annual.daec"
CASES = (
    ("cross_year", 2024, [1.25, -2.5, 0.0, 4.75]),
    ("negative", -1, [1.25, -2.5]),
    ("zero", 0, [1.25]),
    ("minimum", -(2**31), [1.25]),
    ("maximum", 2**31 - 1, [1.25]),
    ("empty", 2024, []),
    ("empty_minimum", -(2**31), []),
    ("empty_maximum", 2**31 - 1, []),
)


def assert_annual(actual, month, first, values):
    """Require owned values and the complete fiscal date identity."""
    assert actual.frequency == Yearly(month)
    assert actual.firstdate == MIT(Yearly(month), first)
    assert actual.lastdate == MIT(Yearly(month), first + len(values) - 1)
    assert actual.values.dtype == np.float64
    assert actual.values.flags.owndata
    np.testing.assert_array_equal(actual.values, values)


@pytest.mark.parametrize("month", range(1, 13))
def test_codec_annual_snapshot_and_empty_limits(month):
    values = np.arange(8, dtype=np.float64)
    encoded = encode_series(TSeries(MIT(Yearly(month), 2024), values[::2]))
    assert encoded[:2] == (256 + month, 2024)
    values[:] = -1
    assert_annual(decode_series(*encoded), month, 2024, [0, 2, 4, 6])
    for year in (-(2**31), -32801, 0, 2**31 - 1):
        encoded = encode_series(TSeries(MIT(Yearly(month), year), np.empty(0)))
        assert_annual(decode_series(*encoded), month, year, [])


@pytest.mark.parametrize("code", [256, 269, 288, 512, -1, 2**40])
def test_annual_frequency_allowlist(code):
    with pytest.raises(TypeError):
        validate_metadata((2, 12, 4, 0, 1, 1, code, 2024, 8))


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.complex128, ">f8"])
@pytest.mark.parametrize("length", [0, 2])
def test_annual_dtype_is_not_coerced(dtype, length):
    with pytest.raises(TypeError):
        encode_series(TSeries(MIT(Yearly(6), 2024), np.ones(length, dtype=dtype)))


@pytest.mark.parametrize(
    ("year", "length"),
    [(-(2**31) - 1, 0), (-(2**31) - 1, 1), (2**31, 0), (2**31, 1), (2**31 - 1, 2)],
)
def test_annual_bounds_before_snapshot(year, length):
    with pytest.raises(ValueError, match="date range"):
        encode_series(TSeries(MIT(Yearly(6), year), np.ones(length)))


@NATIVE
@pytest.mark.parametrize("month", range(1, 13))
def test_julia_fixture_and_python_round_trip(tmp_path, month):
    provenance = tomllib.loads(FIXTURE.with_suffix(".toml").read_text())
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    path = tmp_path / "annual.daec"
    results = []
    with open_dataecon(FIXTURE) as reference, open_dataecon(path, "a") as db:
        for suffix, year, values in CASES:
            name = f"y{month}_{suffix}"
            source = reference.read_series(name)
            assert_annual(source, month, year, values)
            db.write_series(name, source)
            results.append((db.read_series(name), year, values))
            if values:
                source.values[0] = 99
        for suffix, year in (
            ("native_empty", 2024),
            ("native_empty_minimum", -(2**31)),
            ("native_empty_maximum", 2**31 - 1),
        ):
            assert_annual(reference.read_series(f"y{month}_{suffix}"), month, year, [])
        with pytest.raises(DataEconError) as caught:
            db.write_series(f"y{month}_empty", results[0][0])
        assert caught.value.code == -985
        with pytest.raises(DataEconError) as caught:
            db.write_scalar(f"y{month}_empty", 1.25)
        assert caught.value.code == -985
    for result, year, values in results:
        assert_annual(result, month, year, values)
    with open_dataecon(path) as db:
        for suffix, year, values in CASES:
            assert_annual(db.read_series(f"y{month}_{suffix}"), month, year, values)
        with pytest.raises(ValueError, match="read-only"):
            db.write_series("readonly", results[0][0])
    with pytest.raises(ValueError, match="closed"):
        db.read_series(f"y{month}_empty")
    with closing(sqlite3.connect(path)) as conn:
        for suffix, year, values in CASES:
            assert conn.execute(
                "SELECT o.class,o.type,t.eltype,t.elfreq,a.ax_type,a.length,a.frequency,a.data,"
                "t.value FROM objects o JOIN tseries t USING(id) "
                "JOIN axes a ON t.axis_id=a.id WHERE o.name=?",
                (f"y{month}_{suffix}",),
            ).fetchone() == (
                2,
                12,
                4,
                0,
                1,
                len(values),
                256 + month,
                year,
                np.array(values, dtype="<f8").tobytes() if values else None,
            )
        assert conn.execute("SELECT count(*) FROM attributes WHERE id != 0").fetchone() == (0,)


@NATIVE
def test_all_year_ends_stay_distinct_and_february_handles_leap_year():
    with open_dataecon(FIXTURE) as db:
        series = [db.read_series(f"y{month}_cross_year") for month in range(1, 13)]
    assert len({s.firstdate for s in series}) == 12
    assert [mit_to_date(s.firstdate).month for s in series] == list(range(1, 13))
    assert mit_to_date(series[1].firstdate).isoformat() == "2024-02-29"
    assert mit_to_date(series[1].lastdate).isoformat() == "2027-02-28"
    assert mit_to_date(series[5].firstdate, ref="begin").isoformat() == "2023-07-01"


@NATIVE
@pytest.mark.parametrize(
    ("frequency", "first", "payload", "exception"),
    [
        (256, 2024, b"", TypeError),
        (269, 2024, b"", TypeError),
        (262, 2024.0, b"", ValueError),
        (262, "2024", b"", ValueError),
        (262, -(2**31) - 1, b"", ValueError),
        (262, 2**31, b"", ValueError),
        (262, 10**30, b"", ValueError),
        (262, 2**31 - 1, bytes(16), ValueError),
        (262, 2024, bytes(7), ValueError),
    ],
)
def test_native_validation_creates_no_partial_axis(tmp_path, frequency, first, payload, exception):
    path = tmp_path / "invalid.daec"
    with open_dataecon(path, "a") as db:
        db.write_series("good", TSeries(MIT(Yearly(6), 2024), np.ones(1)))
        with pytest.raises(exception):
            db._handle.write("bad", frequency, first, payload)
        with pytest.raises(DataEconError) as caught:
            db.read_series("bad")
        assert caught.value.code == -989
        assert_annual(db.read_series("good"), 6, 2024, [1])
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT count(*) FROM axes").fetchone() == (1,)


@NATIVE
@pytest.mark.parametrize(
    ("assignment", "exception"),
    [
        ("frequency=256", TypeError),
        ("frequency=269", TypeError),
        ("data=-2147483649", ValueError),
        ("data=2147483648", ValueError),
        ("data=2147483647", ValueError),
        ("length=1000000000", ValueError),
    ],
)
def test_bad_native_metadata_leaves_other_fiscal_endings_readable(tmp_path, assignment, exception):
    path = tmp_path / "malformed.daec"
    shutil.copyfile(FIXTURE, path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "UPDATE axes SET " + assignment + " WHERE id=(SELECT axis_id FROM tseries "
            "WHERE id=(SELECT id FROM objects WHERE name='y6_cross_year'))"
        )
    with open_dataecon(path) as db:
        with pytest.raises(exception):
            db.read_series("y6_cross_year")
        assert_annual(db.read_series("y12_cross_year"), 12, 2024, [1.25, -2.5, 0, 4.75])


@NATIVE
@pytest.mark.parametrize("payload", [None, bytes(31)])
def test_annual_malformed_payload_rejected(tmp_path, payload):
    path = tmp_path / "payload.daec"
    shutil.copyfile(FIXTURE, path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "UPDATE tseries SET value=? WHERE id=(SELECT id FROM objects "
            "WHERE name='y6_cross_year')",
            (payload,),
        )
    with open_dataecon(path) as db:
        with pytest.raises(ValueError):
            db.read_series("y6_cross_year")
        assert_annual(db.read_series("y12_cross_year"), 12, 2024, [1.25, -2.5, 0, 4.75])


@NATIVE
@pytest.mark.parametrize(
    ("suffix", "key", "marker"),
    [
        ("empty", "jeltype", "Float32"),
        ("empty", "jeltype", None),
        ("empty", "jtype", "TSeries"),
        ("empty", "jtype", None),
        ("cross_year", "jeltype", "Float64"),
        ("empty", "jeltype", "error(123)"),
    ],
)
def test_annual_reconstruction_markers_rejected(tmp_path, suffix, key, marker):
    path = tmp_path / "marker.daec"
    shutil.copyfile(FIXTURE, path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO attributes SELECT id,?,? FROM objects WHERE name=?",
            (key, marker, f"y6_{suffix}"),
        )
    with open_dataecon(path) as db:
        with pytest.raises(TypeError, match="reconstruction"):
            db.read_series(f"y6_{suffix}")
        assert_annual(db.read_series("y12_empty"), 12, 2024, [])
