"""Quarterly fiscal anchors, native bounds and independent Julia interchange."""

import hashlib
import importlib.util
import shutil
import sqlite3
import tomllib
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from tsecon import MIT, Quarterly, TSeries
from tsecon.dataecon import DataEconError, open_dataecon
from tsecon.dataecon._codec import decode_series, encode_series, validate_metadata
from tsecon.mit import mit_to_date

NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built; configure TSECON_DATAECON_ROOT",
)
FIXTURES = Path(__file__).parent / "fixtures"
CASES = (
    ("cross_year", 8099, [1.25, -2.5, 0.0, 4.75]),
    ("negative", -1, [1.25, -2.5]),
    ("zero", 0, [1.25]),
    ("minimum", -131200, [1.25]),
    ("maximum", 2147483647, [1.25]),
    ("empty", 8096, []),
    ("empty_minimum", -131200, []),
    ("empty_maximum", 2147483647, []),
)


def assert_series(actual, anchor, code, values):
    """Check dates, frequency and owning values independently of byte encoding."""
    assert actual.frequency == Quarterly(anchor)
    assert actual.firstdate == MIT(Quarterly(anchor), code)
    assert actual.lastdate == MIT(Quarterly(anchor), code + len(values) - 1)
    assert actual.values.dtype == np.float64
    assert actual.values.flags.owndata
    np.testing.assert_array_equal(actual.values, values)


@pytest.mark.parametrize("anchor", [1, 2, 3])
def test_codec_preserves_anchor_and_snapshots_strides(anchor):
    values = np.arange(8, dtype=np.float64)
    source = TSeries(MIT(Quarterly(anchor), 8099), values[::2])
    freq, first, payload = encode_series(source)
    assert (freq, first) == (64 + anchor, 8099)
    values[:] = -1
    result = decode_series(freq, first, payload)
    assert_series(result, anchor, 8099, [0, 2, 4, 6])
    result.values[0] = 99
    assert np.frombuffer(payload, dtype=np.float64)[0] == 0


@pytest.mark.parametrize("frequency", [0, 33, 64, 68, 96, 128, 135, 256, -1, 2**40])
def test_noncanonical_frequency_rejected(frequency):
    with pytest.raises(TypeError):
        validate_metadata((2, 12, 4, 0, 1, 1, frequency, 8099, 8))


@pytest.mark.parametrize("anchor", [1, 2, 3])
@pytest.mark.parametrize(
    ("code", "length"), [(-131201, 0), (-131201, 1), (2**31, 0), (2**31, 1), (2**31 - 1, 2)]
)
def test_quarterly_bounds_precede_encoding(anchor, code, length):
    with pytest.raises(ValueError, match="date range"):
        encode_series(TSeries(MIT(Quarterly(anchor), code), np.ones(length)))


@pytest.mark.parametrize("dtype", [np.float32, np.int64, np.complex128, ">f8"])
@pytest.mark.parametrize("length", [0, 2])
def test_quarterly_dtype_is_not_coerced(dtype, length):
    with pytest.raises(TypeError):
        encode_series(TSeries(MIT(Quarterly(1), 8099), np.ones(length, dtype=dtype)))


@NATIVE
@pytest.mark.parametrize("anchor", [1, 2, 3])
def test_julia_fixture_and_python_roundtrips(tmp_path, anchor):
    fixture = FIXTURES / "julia_quarterly.daec"
    provenance = tomllib.loads(fixture.with_suffix(".toml").read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    path = tmp_path / "quarters.daec"
    loaded = []
    with open_dataecon(fixture) as reference, open_dataecon(path, "a") as db:
        for suffix, code, values in CASES:
            name = f"q{anchor}_{suffix}"
            source = reference.read_series(name)
            assert_series(source, anchor, code, values)
            db.write_series(name, source)
            result = db.read_series(name)
            loaded.append((result, code, values))
            if values:
                source.values[0] = 99
        assert_series(reference.read_series(f"q{anchor}_native_empty"), anchor, 8096, [])
        with pytest.raises(DataEconError) as caught:
            db.write_scalar(f"q{anchor}_cross_year", 1.25)
        assert caught.value.code == -985
        with pytest.raises(DataEconError) as caught:
            db.write_series(
                f"q{anchor}_cross_year", TSeries(MIT(Quarterly(anchor), 8099), np.ones(1))
            )
        assert caught.value.code == -985
    for result, code, values in loaded:
        assert_series(result, anchor, code, values)
    with open_dataecon(path) as db:
        for suffix, code, values in CASES:
            assert_series(db.read_series(f"q{anchor}_{suffix}"), anchor, code, values)
        with pytest.raises(ValueError, match="read-only"):
            db.write_series("readonly", loaded[0][0])
    with pytest.raises(ValueError, match="closed"):
        db.read_series(f"q{anchor}_cross_year")
    with closing(sqlite3.connect(path)) as conn:
        for suffix, code, values in CASES:
            assert conn.execute(
                "SELECT o.class,o.type,t.eltype,t.elfreq,a.ax_type,a.length,a.frequency,a.data,"
                "length(t.value),t.value IS NULL FROM objects o JOIN tseries t USING(id) "
                "JOIN axes a ON t.axis_id=a.id WHERE o.name=?",
                (f"q{anchor}_{suffix}",),
            ).fetchone() == (
                2,
                12,
                4,
                0,
                1,
                len(values),
                64 + anchor,
                code,
                8 * len(values) if values else None,
                not values,
            )
        assert conn.execute("SELECT count(*) FROM attributes WHERE id != 0").fetchone() == (0,)


@NATIVE
def test_same_integer_dates_keep_distinct_fiscal_anchors(tmp_path):
    with open_dataecon(tmp_path / "anchors.daec", "a") as db:
        for anchor in (1, 2, 3):
            db.write_series(str(anchor), TSeries(MIT(Quarterly(anchor), 8099), np.ones(4)))
        series = [db.read_series(str(anchor)) for anchor in (1, 2, 3)]
    assert len({s.firstdate for s in series}) == 3
    assert [mit_to_date(s.firstdate).isoformat() for s in series] == [
        "2024-10-31",
        "2024-11-30",
        "2024-12-31",
    ]
    assert [mit_to_date(s.lastdate).isoformat() for s in series] == [
        "2025-07-31",
        "2025-08-31",
        "2025-09-30",
    ]


@NATIVE
@pytest.mark.parametrize(
    ("frequency", "first", "payload", "exception"),
    [
        (64, 8096, b"", TypeError),
        (68, 8096, b"", TypeError),
        ("65", 8096, b"", TypeError),
        (65, 8096.0, b"", ValueError),
        (65, True, b"", ValueError),
        (65, 10**30, b"", ValueError),
        (65, -131201, b"", ValueError),
        (65, 2**31, b"", ValueError),
        (65, 2**31 - 1, bytes(16), ValueError),
        (65, 8096, bytes(7), ValueError),
    ],
)
def test_native_input_guards_leave_no_partial_axis(tmp_path, frequency, first, payload, exception):
    path = tmp_path / "invalid.daec"
    with open_dataecon(path, "a") as db:
        with pytest.raises(exception):
            db._handle.write("bad", frequency, first, payload)
        with pytest.raises(DataEconError) as caught:
            db.read_series("bad")
        assert caught.value.code == -989
        db.write_series("good", TSeries(MIT(Quarterly(1), 8099), np.ones(1)))
        assert_series(db.read_series("good"), 1, 8099, [1])
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT count(*) FROM axes").fetchone() == (1,)


@NATIVE
@pytest.mark.parametrize(
    ("sql", "exception"),
    [
        (
            "UPDATE axes SET frequency=64 "
            "WHERE id=(SELECT axis_id FROM tseries "
            "WHERE id=(SELECT id FROM objects WHERE name='q1_cross_year'))",
            TypeError,
        ),
        (
            "UPDATE axes SET data=-131201 "
            "WHERE id=(SELECT axis_id FROM tseries "
            "WHERE id=(SELECT id FROM objects WHERE name='q1_cross_year'))",
            ValueError,
        ),
        (
            "UPDATE axes SET data=2147483648 "
            "WHERE id=(SELECT axis_id FROM tseries "
            "WHERE id=(SELECT id FROM objects WHERE name='q1_cross_year'))",
            ValueError,
        ),
        (
            "UPDATE axes SET data=2147483647 "
            "WHERE id=(SELECT axis_id FROM tseries "
            "WHERE id=(SELECT id FROM objects WHERE name='q1_cross_year'))",
            ValueError,
        ),
        (
            "UPDATE tseries SET value=zeroblob(31) "
            "WHERE id=(SELECT id FROM objects WHERE name='q1_cross_year')",
            ValueError,
        ),
        (
            "UPDATE tseries SET value=NULL "
            "WHERE id=(SELECT id FROM objects WHERE name='q1_cross_year')",
            ValueError,
        ),
    ],
)
def test_malformed_native_records_do_not_poison_other_objects(tmp_path, sql, exception):
    path = tmp_path / "malformed.daec"
    shutil.copyfile(FIXTURES / "julia_quarterly.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(sql)
    with open_dataecon(path) as db:
        with pytest.raises(exception):
            db.read_series("q1_cross_year")
        assert_series(db.read_series("q2_cross_year"), 2, 8099, [1.25, -2.5, 0, 4.75])


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
def test_quarterly_reconstruction_markers_rejected(tmp_path, suffix, key, marker):
    path = tmp_path / "marker.daec"
    shutil.copyfile(FIXTURES / "julia_quarterly.daec", path)
    name = f"q1_{suffix}"
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO attributes SELECT id,?,? FROM objects WHERE name=?",
            (key, marker, name),
        )
    with open_dataecon(path) as db:
        with pytest.raises(TypeError, match="reconstruction"):
            db.read_series(name)
        assert_series(db.read_series("q2_empty"), 2, 8096, [])
