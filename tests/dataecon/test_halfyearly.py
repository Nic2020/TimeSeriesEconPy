"""Half-yearly fiscal endings, native date bounds and Julia interchange."""

import hashlib
import importlib.util
import shutil
import sqlite3
import tomllib
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from tsecon import MIT, HalfYearly, TSeries
from tsecon.dataecon import DataEconError, open_dataecon
from tsecon.dataecon._codec import decode_series, encode_series, validate_metadata
from tsecon.mit import mit_to_date

NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built; configure TSECON_DATAECON_ROOT",
)
FIXTURE = Path(__file__).parent / "fixtures/julia_halfyearly.daec"
MINIMUM = -65600
CASES = (
    ("cross_year", 4048, [1.25, -2.5, 0.0, 4.75]),
    ("negative", -1, [1.25, -2.5]),
    ("zero", 0, [1.25]),
    ("minimum", MINIMUM, [1.25]),
    ("maximum", 2**31 - 1, [1.25]),
    ("empty", 4048, []),
    ("empty_minimum", MINIMUM, []),
    ("empty_maximum", 2**31 - 1, []),
)


def assert_half(actual, month, first, values):
    """Require owned values and the complete fiscal date identity."""
    assert actual.frequency == HalfYearly(month)
    assert actual.firstdate == MIT(HalfYearly(month), first)
    assert actual.lastdate == MIT(HalfYearly(month), first + len(values) - 1)
    assert actual.values.dtype == np.float64
    assert actual.values.flags.owndata
    np.testing.assert_array_equal(actual.values, values)


@pytest.mark.parametrize("month", range(1, 7))
def test_codec_halfyearly_snapshot_and_empty_limits(month):
    values = np.arange(8, dtype=np.float64)
    encoded = encode_series(TSeries(MIT(HalfYearly(month), 4049), values[::2]))
    assert encoded[:2] == (128 + month, 4049)
    values[:] = -1
    assert_half(decode_series(*encoded), month, 4049, [0, 2, 4, 6])
    for code in (MINIMUM, MINIMUM + 1, -1, 0, 2**31 - 1):
        encoded = encode_series(TSeries(MIT(HalfYearly(month), code), np.empty(0)))
        assert_half(decode_series(*encoded), month, code, [])


@pytest.mark.parametrize("code", [128, 135, 136, 160, 192, 384, -1, 2**40])
def test_halfyearly_frequency_allowlist(code):
    with pytest.raises(TypeError):
        validate_metadata((2, 12, 4, 0, 1, 1, code, 4048, 8))


@pytest.mark.parametrize("dtype", [object, "S4", "U4", ">f8"])
@pytest.mark.parametrize("length", [0, 2])
def test_halfyearly_dtype_is_not_coerced(dtype, length):
    with pytest.raises(TypeError):
        encode_series(TSeries(MIT(HalfYearly(3), 4048), np.ones(length, dtype=dtype)))


@pytest.mark.parametrize("month", range(1, 7))
@pytest.mark.parametrize(
    ("code", "length"),
    [
        (MINIMUM - 1, 0),
        (MINIMUM - 1, 1),
        (-(2**31), 0),
        (-(2**31), 1),
        (-(2**31) - 1, 0),
        (2**31, 0),
        (2**31, 1),
        (2**31 - 1, 2),
    ],
)
def test_halfyearly_bounds_before_snapshot(month, code, length):
    with pytest.raises(ValueError, match="date range"):
        encode_series(TSeries(MIT(HalfYearly(month), code), np.ones(length)))


def test_halfyearly_lower_bound_does_not_change_other_frequencies():
    # Annual keeps the full signed 32-bit range; quarterly keeps its own limit.
    validate_metadata((2, 12, 4, 0, 1, 1, 262, -(2**31), 8))
    validate_metadata((2, 12, 4, 0, 1, 1, 66, -131200, 8))
    validate_metadata((2, 12, 4, 0, 1, 1, 66, MINIMUM - 1, 8))
    with pytest.raises(ValueError, match="quarterly"):
        validate_metadata((2, 12, 4, 0, 1, 1, 66, -131201, 8))
    with pytest.raises(ValueError, match="half-yearly"):
        validate_metadata((2, 12, 4, 0, 1, 1, 131, MINIMUM - 1, 8))


@NATIVE
@pytest.mark.parametrize("month", range(1, 7))
def test_julia_fixture_and_python_round_trip(tmp_path, month):
    provenance = tomllib.loads(FIXTURE.with_suffix(".toml").read_text())
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    path = tmp_path / "halfyearly.daec"
    results = []
    with open_dataecon(FIXTURE) as reference, open_dataecon(path, "a") as db:
        for suffix, code, values in CASES:
            name = f"h{month}_{suffix}"
            source = reference.read_series(name)
            assert_half(source, month, code, values)
            db.write_series(name, source)
            results.append((db.read_series(name), code, values))
            if values:
                source.values[0] = 99
        for suffix, code in (
            ("native_empty", 4048),
            ("native_empty_minimum", MINIMUM),
            ("native_empty_maximum", 2**31 - 1),
        ):
            assert_half(reference.read_series(f"h{month}_{suffix}"), month, code, [])
        with pytest.raises(DataEconError) as caught:
            db.write_series(f"h{month}_empty", results[0][0])
        assert caught.value.code == -985
        with pytest.raises(DataEconError) as caught:
            db.write_scalar(f"h{month}_empty", 1.25)
        assert caught.value.code == -985
    for result, code, values in results:
        assert_half(result, month, code, values)
    with open_dataecon(path) as db:
        for suffix, code, values in CASES:
            assert_half(db.read_series(f"h{month}_{suffix}"), month, code, values)
        with pytest.raises(ValueError, match="read-only"):
            db.write_series("readonly", results[0][0])
    with pytest.raises(ValueError, match="closed"):
        db.read_series(f"h{month}_empty")
    with closing(sqlite3.connect(path)) as conn:
        for suffix, code, values in CASES:
            assert conn.execute(
                "SELECT o.class,o.type,t.eltype,t.elfreq,a.ax_type,a.length,a.frequency,a.data,"
                "t.value FROM objects o JOIN tseries t USING(id) "
                "JOIN axes a ON t.axis_id=a.id WHERE o.name=?",
                (f"h{month}_{suffix}",),
            ).fetchone() == (
                2,
                12,
                4,
                0,
                1,
                len(values),
                128 + month,
                code,
                np.array(values, dtype="<f8").tobytes() if values else None,
            )
        assert conn.execute("SELECT count(*) FROM attributes WHERE id != 0").fetchone() == (0,)


@NATIVE
@pytest.mark.parametrize("month", range(1, 7))
def test_julia_below_minimum_anchor_is_rejected_not_misdated(month):
    # Julia stores code -65601 intact but its loader decodes another date.
    with open_dataecon(FIXTURE) as db:
        with pytest.raises(ValueError, match="half-yearly date range"):
            db.read_series(f"h{month}_below_minimum")
        assert_half(db.read_series(f"h{month}_minimum"), month, MINIMUM, [1.25])
    with closing(sqlite3.connect(FIXTURE)) as conn:
        assert conn.execute(
            "SELECT a.data FROM objects o JOIN tseries t USING(id) JOIN axes a "
            "ON t.axis_id=a.id WHERE o.name=?",
            (f"h{month}_below_minimum",),
        ).fetchone() == (MINIMUM - 1,)


@NATIVE
def test_all_half_year_ends_stay_distinct_and_february_handles_leap_year():
    with open_dataecon(FIXTURE) as db:
        series = [db.read_series(f"h{month}_cross_year") for month in range(1, 7)]
    assert len({s.firstdate for s in series}) == 6
    assert [mit_to_date(s.firstdate).month for s in series] == list(range(1, 7))
    assert [mit_to_date(s.firstdate + 1).month for s in series] == list(range(7, 13))
    assert mit_to_date(series[1].firstdate).isoformat() == "2024-02-29"
    assert mit_to_date(series[1].firstdate + 2).isoformat() == "2025-02-28"
    assert mit_to_date(series[2].firstdate, ref="begin").isoformat() == "2023-10-01"
    assert mit_to_date(series[2].lastdate).isoformat() == "2025-09-30"
    assert mit_to_date(series[5].firstdate, ref="begin").isoformat() == "2024-01-01"
    assert mit_to_date(series[5].lastdate).isoformat() == "2025-12-31"


@NATIVE
@pytest.mark.parametrize(
    ("frequency", "first", "payload", "exception"),
    [
        (128, 4048, b"", TypeError),
        (135, 4048, b"", TypeError),
        (134, 4048.0, b"", ValueError),
        (134, "4048", b"", ValueError),
        (134, -65601, b"", ValueError),
        (134, -65602, b"", ValueError),
        (134, -(2**31), b"", ValueError),
        (134, 2**31, b"", ValueError),
        (134, 10**30, b"", ValueError),
        (134, 2**31 - 1, bytes(16), ValueError),
        (134, 4048, bytes(7), ValueError),
    ],
)
def test_native_validation_creates_no_partial_axis(tmp_path, frequency, first, payload, exception):
    path = tmp_path / "invalid.daec"
    with open_dataecon(path, "a") as db:
        db.write_series("good", TSeries(MIT(HalfYearly(6), 4048), np.ones(1)))
        with pytest.raises(exception):
            db._handle.write("bad", frequency, first, payload, False, 4, 0, len(payload) // 8, None)
        with pytest.raises(DataEconError) as caught:
            db.read_series("bad")
        assert caught.value.code == -989
        assert_half(db.read_series("good"), 6, 4048, [1])
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT count(*) FROM axes").fetchone() == (1,)


@NATIVE
@pytest.mark.parametrize(
    ("assignment", "exception"),
    [
        ("frequency=128", TypeError),
        ("frequency=135", TypeError),
        ("frequency=192", TypeError),
        ("data=-65601", ValueError),
        ("data=-2147483648", ValueError),
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
            "WHERE id=(SELECT id FROM objects WHERE name='h3_cross_year'))"
        )
    with open_dataecon(path) as db:
        with pytest.raises(exception):
            db.read_series("h3_cross_year")
        assert_half(db.read_series("h6_cross_year"), 6, 4048, [1.25, -2.5, 0, 4.75])


@NATIVE
@pytest.mark.parametrize("payload", [None, bytes(31)])
def test_halfyearly_malformed_payload_rejected(tmp_path, payload):
    path = tmp_path / "payload.daec"
    shutil.copyfile(FIXTURE, path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "UPDATE tseries SET value=? WHERE id=(SELECT id FROM objects "
            "WHERE name='h3_cross_year')",
            (payload,),
        )
    with open_dataecon(path) as db:
        with pytest.raises(ValueError):
            db.read_series("h3_cross_year")
        assert_half(db.read_series("h6_cross_year"), 6, 4048, [1.25, -2.5, 0, 4.75])


@NATIVE
@pytest.mark.parametrize(
    ("suffix", "key", "marker"),
    [
        ("empty", "jeltype", "Float128"),
        ("empty", "jeltype", None),
        ("empty", "jtype", "TSeries"),
        ("empty", "jtype", None),
        ("cross_year", "jeltype", "Float64"),
        ("empty", "jeltype", "error(123)"),
    ],
)
def test_halfyearly_reconstruction_markers_rejected(tmp_path, suffix, key, marker):
    path = tmp_path / "marker.daec"
    shutil.copyfile(FIXTURE, path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO attributes SELECT id,?,? FROM objects WHERE name=?",
            (key, marker, f"h3_{suffix}"),
        )
    with open_dataecon(path) as db:
        with pytest.raises(TypeError, match="reconstruction"):
            db.read_series(f"h3_{suffix}")
        assert_half(db.read_series("h6_empty"), 6, 4048, [])
