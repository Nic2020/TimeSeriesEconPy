"""Exact numeric/Boolean series storage, reconstruction and failure residue."""

import hashlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
import tomllib
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from tsecon import MIT, TSeries, mm
from tsecon.dataecon import open_dataecon
from tsecon.dataecon._codec import MAX_BYTES, decode_series, encode_series, validate_metadata
from tsecon.frequencies import BDaily, Daily, HalfYearly, Monthly, Quarterly, Weekly, Yearly

NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built",
)
TYPES = ["i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8", "c8", "c16", "?"]
TOKENS = [
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float16",
    "Float32",
    "Float64",
    "ComplexF32",
    "ComplexF64",
    "Bool",
]
FREQUENCIES = [
    Monthly(),
    Daily(),
    BDaily(),
    *[Weekly(d) for d in range(1, 8)],
    *[Quarterly(m) for m in range(1, 4)],
    *[HalfYearly(m) for m in range(1, 7)],
    *[Yearly(m) for m in range(1, 13)],
]
FIXTURE = Path(__file__).parent / "fixtures/julia_series_elements.daec"


def test_fixture_provenance():
    provenance = tomllib.loads(FIXTURE.with_suffix(".toml").read_text(encoding="utf-8"))
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    assert provenance["timeseriesecon_sha"] == "fc0a0d01aed4903ea6c64b12d78d0bdf68468df6"
    assert provenance["native_source_sha"] == "1a108688a044380f808bebf64079e32dbb9cd1a4"
    with closing(sqlite3.connect(f"{FIXTURE.resolve().as_uri()}?mode=ro", uri=True)) as sql:
        assert sql.execute("SELECT count(*) FROM objects WHERE class=2").fetchone() == (92,)


def values_for(dtype):
    dt = np.dtype(dtype)
    if dt.kind in "iu":
        info = np.iinfo(dt)
        return np.array(
            [info.min, -1, 0, info.max] if dt.kind == "i" else [0, 1, info.max], dtype=dt
        )
    if dt.kind == "b":
        return np.array([False, True, False])
    if dt.kind == "c":
        return np.array([complex(-0.0, 1.25), complex(2.5, -3)], dtype=dt)
    bits = {
        2: [0x8000, 1, 0x3D00, 0x7E55, 0x7C00],
        4: [0x80000000, 1, 0x3FA00000, 0x7FC00055, 0x7F800000],
        8: [0x8000000000000000, 1, 0x3FF4000000000000, 0x7FF8000000000055, 0x7FF0000000000000],
    }
    return np.array(bits[dt.itemsize], dtype=f"u{dt.itemsize}").view(dt)


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("empty", [False, True])
def test_exact_codec_snapshot_and_owning_result(dtype, empty):
    values = values_for(dtype)
    values = values[:0] if empty else values[::-1]
    expected = values.tobytes()
    encoded = encode_series(TSeries(mm(2024, 11), values))
    values[...] = 0
    result = decode_series(*encoded)
    assert result.values.dtype == np.dtype(dtype)
    assert result.values.tobytes() == expected
    assert result.values.flags.owndata
    assert result.values.flags.writeable
    assert result.firstdate == mm(2024, 11)


@pytest.mark.parametrize("dtype", [np.longdouble, np.clongdouble])
@pytest.mark.parametrize("length", [0, 2])
def test_long_double_series_are_rejected_on_every_platform(dtype, length):
    # dtype equality alone aliases these to f8/c16 on Windows. The original
    # character survives construction and identifies the unsupported dtype.
    series = TSeries(mm(2024, 1), np.ones(length, dtype=dtype))
    assert series.values.dtype.char in ("g", "G")
    with pytest.raises(TypeError, match="supported native-endian dtype"):
        encode_series(series)


def test_noncanonical_boolean_array_bytes_are_canonicalized_without_mutation():
    # Retain the array view: scalar extraction would canonicalize before encoding.
    raw = np.array([0, 1, 2, 127, 128, 255], dtype=np.uint8)
    source = raw.view(np.bool_)[::-1]
    encoded = encode_series(TSeries(mm(2024, 1), source))
    assert encoded.payload == b"\1\1\1\1\1\0"
    assert encoded.marker == "Bool"
    assert raw.tobytes() == b"\0\1\2\x7f\x80\xff"
    with pytest.raises(ValueError, match="zero and one"):
        decode_series(32, 24288, b"\2", 1, 1, "Bool")


@pytest.mark.parametrize(
    ("element", "nbytes", "length", "marker"),
    [
        (1, 8, 1, "Bool"),
        (2, 1, 1, "Bool"),
        (4, 0, 0, "Int16"),
        (1, 0, 0, "Int128"),
        (5, 0, 0, "ComplexF16"),
        (1, 1, 1, "Int8"),
        (4, 0, 0, " Float64"),
        (4, 0, 0, "Float64()"),
    ],
)
def test_reconstruction_allowlist(element, nbytes, length, marker):
    with pytest.raises(TypeError):
        decode_series(32, 24288, bytes(nbytes), element, length, marker)


@pytest.mark.parametrize(("kind", "width"), [(1, 1), (2, 2), (4, 4), (5, 16)])
def test_width_dependent_capacity_without_allocating(kind, width):
    length = MAX_BYTES // width
    validate_metadata((2, 12, kind, 0, 1, length, 12, 11979954, MAX_BYTES))
    with pytest.raises(ValueError):
        validate_metadata((2, 12, kind, 0, 1, length + 1, 12, 11979954, MAX_BYTES + width))
    with pytest.raises(ValueError):
        validate_metadata((2, 12, kind, 0, 1, 2, 268, 2**31 - 1, 2 * width))


@NATIVE
@pytest.mark.parametrize("dtype", TYPES)
def test_julia_fixture_and_native_rewrite(dtype, tmp_path):
    token = TOKENS[TYPES.index(dtype)]
    expected = values_for(dtype)
    with open_dataecon(FIXTURE) as db:
        nonempty = db.read_series(f"es_{token}")
        empty = db.read_series(f"es_{token}_empty")
        default = db.read_series(f"es_{token}_unmarked")
    assert nonempty.values.dtype == expected.dtype
    assert nonempty.values.tobytes() == expected.tobytes()
    assert empty.values.dtype == expected.dtype
    kind = expected.dtype.kind
    assert default.values.dtype == np.dtype(
        {"i": "i8", "u": "u8", "f": "f8", "c": "c16", "b": "i8"}[kind]
    )
    for result in (nonempty, empty, default):
        assert result.firstdate == mm(2024, 11)
        assert result.values.flags.owndata
    path = tmp_path / "rewrite.daec"
    with open_dataecon(path, "a") as db:
        db.write_series("nonempty", nonempty)
        db.write_series("empty", empty)
    with open_dataecon(path) as db:
        for name, before in (("nonempty", nonempty), ("empty", empty)):
            after = db.read_series(name)
            assert after.values.dtype == before.values.dtype
            assert after.values.tobytes() == before.values.tobytes()
    with closing(sqlite3.connect(path)) as sql:
        rows = sql.execute(
            "SELECT o.name,t.eltype,t.elfreq,a.frequency,a.data,hex(t.value) "
            "FROM objects o JOIN tseries t ON o.id=t.id JOIN axes a ON a.id=t.axis_id"
        ).fetchall()
        for name, element, elfreq, frequency, first, payload in rows:
            assert (element, elfreq, frequency, first) == (
                {"i": 1, "u": 2, "f": 4, "c": 5, "b": 1}[kind],
                0,
                32,
                24298,
            )
            assert payload.lower() == (expected.tobytes().hex() if name == "nonempty" else "")


@NATIVE
@pytest.mark.parametrize("frequency", FREQUENCIES)
@pytest.mark.parametrize("dtype", ["i1", "u8", "f2", "c8", "?"])
def test_widths_over_all_existing_axis_families(frequency, dtype, tmp_path):
    path = tmp_path / "axis.daec"
    values = values_for(dtype)
    first = MIT(frequency, 100)
    with open_dataecon(path, "a") as db:
        for empty in (False, True):
            source = TSeries(first, values[:0] if empty else values)
            db.write_series("s", source, overwrite=True)
            actual = db.read_series("s")
            assert actual.firstdate == first
            assert actual.values.dtype == values.dtype
            assert actual.values.tobytes() == source.values.tobytes()


@NATIVE
@pytest.mark.parametrize(
    "name",
    [
        "bad_bool",
        "foreign_bool",
        "unsupported_Int128",
        "unsupported_UInt128",
        "unsupported_ComplexF16",
        "unknown_marker",
        "wrong_marker",
    ],
)
def test_fixture_rejections(name):
    with open_dataecon(FIXTURE) as db, pytest.raises((TypeError, ValueError)):
        db.read_series(name)


@NATIVE
def test_invalid_overwrite_keeps_original(tmp_path):
    path = tmp_path / "preserve.daec"
    with open_dataecon(path, "a") as db:
        db.write_scalar("keep", 7)
        with pytest.raises(ValueError, match="zero and one"):
            db._handle.write("keep", 32, 24288, b"\2", True, 1, 1, "Bool")
        assert db.read_scalar("keep") == 7


MARKER_FAULT = """
import json,sys,numpy as np
from tsecon import TSeries,mm
from tsecon.dataecon import open_dataecon,DataEconError
path,kind,overwrite=sys.argv[1:]
db=open_dataecon(path,'a')
values=np.array([True,False]) if kind=='bool' else np.array([],dtype=np.int16)
report={}
try:
    db.write_series('target',TSeries(mm(2024,11),values),overwrite=overwrite=='yes')
except DataEconError as e:
    report['error']=[e.code,e.operation]
else:
    raise AssertionError('marker injection did not fail')
result=db.read_series('target')
report['dtype']=str(result.values.dtype)
report['values']=result.values.tolist()
try:
    db.close()
except DataEconError as e:
    report['close']=e.code
report['closed']=db.closed
print(json.dumps(report))
"""


@NATIVE
@pytest.mark.parametrize("kind", ["bool", "empty"])
@pytest.mark.parametrize("overwrite", ["yes", "no"])
def test_marker_failure_leaves_readable_wrong_dtype(tmp_path, kind, overwrite):
    path = tmp_path / "marker-failure.daec"
    with open_dataecon(path, "a") as db:
        db.write_scalar("other", 9)
        if overwrite == "yes":
            db.write_scalar("target", 7)
    with closing(sqlite3.connect(path)) as sql, sql:
        sql.execute(
            "CREATE TRIGGER fail_marker BEFORE INSERT ON attributes WHEN NEW.name='jeltype' "
            "BEGIN SELECT RAISE(ABORT,'injected marker failure'); END"
        )
    run = subprocess.run(
        [sys.executable, "-c", MARKER_FAULT, str(path), kind, overwrite],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    report = json.loads(run.stdout)
    # Positive native error codes pass through SQLite: 19 is SQLITE_CONSTRAINT.
    assert report["error"][0] == 19
    assert "unmarked object may read as a different dtype" in report["error"][1]
    assert report["dtype"] == ("int8" if kind == "bool" else "int64")
    assert report["values"] == ([1, 0] if kind == "bool" else [])
    assert report["close"] == 19
    assert report["closed"]
    with closing(sqlite3.connect(path)) as sql:
        assert sql.execute("SELECT count(*) FROM attributes WHERE name='jeltype'").fetchone() == (
            0,
        )
        assert sql.execute(
            "SELECT hex(t.value) FROM tseries t JOIN objects o ON o.id=t.id WHERE o.name='target'"
        ).fetchone() == (("0100" if kind == "bool" else ""),)
    with open_dataecon(path) as db:
        assert db.read_scalar("other") == 9
        assert db.read_series("target").values.dtype == np.dtype(report["dtype"])
