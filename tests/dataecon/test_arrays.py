# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sqlite3
import subprocess
import sys
import tomllib
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from tsecon import TSeries
from tsecon.dataecon import StoredSeries, open_dataecon
from tsecon.dataecon._codec import MAX_BYTES, decode_array, encode_array, validate_array_metadata
from tsecon.dataecon._metadata import _SCALAR_FREQUENCIES
from tsecon.frequencies import Monthly, Unit
from tsecon.mit import MIT
from tsecon.mitrange import MITRange

FIXTURES = Path(__file__).parent / "fixtures"
NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built; configure TSECON_DATAECON_ROOT",
)
DTYPES = [
    "i1",
    "<i2",
    "<i4",
    "<i8",
    "u1",
    "<u2",
    "<u4",
    "<u8",
    "<f2",
    "<f4",
    "<f8",
    "<c8",
    "<c16",
    "?",
]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("length", [0, 3])
def test_array_codec_preserves_dtype_bytes_and_ownership(dtype, length):
    source = (
        np.array([False, True, False][:length], dtype=bool)
        if dtype == "?"
        else np.arange(length, dtype=dtype)
    )
    if np.dtype(dtype).kind == "c" and length:
        source[0] = complex(-0.0, -0.0)
    encoded = encode_array(source[::-1])
    metadata = (
        2,
        encoded.object_type,
        encoded.element,
        encoded.element_frequency,
        encoded.axis_type,
        encoded.length,
        encoded.frequency,
        encoded.first,
        len(encoded.payload),
    )
    loaded = decode_array(metadata, encoded.payload, encoded.marker, encoded.object_marker)
    assert isinstance(loaded, np.ndarray)
    assert loaded.dtype == source.dtype
    assert loaded.flags.owndata
    assert loaded.flags.writeable
    assert loaded.flags.c_contiguous
    assert loaded.tobytes() == source[::-1].tobytes()


def test_array_float_and_complex_bits_are_not_hidden_by_equality():
    source = np.array(
        [complex(-0.0, 1.0), complex(np.nan, -0.0), complex(2.0, np.nan)], dtype="<c16"
    )
    encoded = encode_array(source)
    assert encoded.payload == source.tobytes()
    changed = bytearray(encoded.payload)
    changed[16:24] = np.float64(3.0).tobytes()
    assert bytes(changed) != encoded.payload


@pytest.mark.parametrize(
    "value", [range(0, 5), range(3, 8), range(-3, 2), range(1, 6, 2), range(1, 0)]
)
def test_lossy_integer_ranges_are_refused_before_native_write(value):
    with pytest.raises(ValueError, match=r"preserves only.*length"):
        encode_array(value)


def test_range_codec_preserves_only_lossless_forms():
    integer = encode_array(range(1, 6))
    assert integer[:8] == (11, 0, 0, 0, b"", 0, 0, 5)
    assert decode_array((2, 11, 0, 0, 0, 5, 0, 0, 0), b"", None, None) == range(1, 6)

    source = MITRange(MIT(Monthly(), -3), MIT(Monthly(), 1))
    dated = encode_array(source)
    assert dated[:8] == (11, 1, 32, -3, b"", 0, 0, 5)
    assert decode_array((2, 11, 0, 0, 1, 5, 32, -3, 0), b"", None, None) == source

    empty = MITRange(MIT(Unit(), 5), MIT(Unit(), 4))
    encoded = encode_array(empty)
    assert encoded.marker == "MIT{Unit}"
    assert decode_array((2, 11, 0, 0, 1, 0, 11, 5, 0), b"", encoded.marker, None) == empty


@pytest.mark.parametrize(("code", "frequency"), sorted(_SCALAR_FREQUENCIES.items()))
def test_mit_ranges_cover_every_canonical_frequency(code, frequency):
    source = MITRange(MIT(frequency, 0), MIT(frequency, 1))
    encoded = encode_array(source)
    assert (encoded.object_type, encoded.axis_type, encoded.frequency, encoded.first) == (
        11,
        1,
        code,
        0,
    )


def test_range_steps_and_unit_overflow_are_refused():
    with pytest.raises(ValueError, match="unit-step"):
        encode_array(MITRange(MIT(Unit(), 1), MIT(Unit(), 5), 2))
    with pytest.raises(ValueError, match="endpoint"):
        encode_array(MITRange(MIT(Unit(), 2**63 - 1), MIT(Unit(), 2**63)))


def test_array_metadata_capacity_and_shape_guards_need_no_large_allocation():
    validate_array_metadata((2, 10, 1, 0, 0, MAX_BYTES, 0, 0, MAX_BYTES))
    with pytest.raises(ValueError, match="oversized"):
        validate_array_metadata((2, 10, 1, 0, 0, MAX_BYTES + 1, 0, 0, MAX_BYTES + 1))
    with pytest.raises(ValueError, match="element width"):
        validate_array_metadata((2, 10, 1, 0, 0, 2, 0, 0, 3))
    with pytest.raises(TypeError, match="plain axis"):
        validate_array_metadata((2, 10, 1, 0, 1, 2, 11, 0, 2))


@NATIVE
@pytest.mark.parametrize(
    ("name", "statement", "error"),
    [
        (
            "array_Int16",
            "UPDATE axes SET ax_type=1 WHERE id=(SELECT axis_id FROM tseries WHERE id=:id)",
            TypeError,
        ),
        ("array_Int16", "UPDATE tseries SET value=x'00' WHERE id=:id", ValueError),
        ("array_Int16", "INSERT INTO attributes VALUES(:id,'jeltype','Bool')", TypeError),
        ("array_Bool", "UPDATE tseries SET value=x'020100' WHERE id=:id", ValueError),
        ("array_Int16", "INSERT INTO attributes VALUES(:id,'jtype','TSeries')", TypeError),
        ("array_Int16", "INSERT INTO attributes VALUES(:id,'jtype','Vector{Any}')", TypeError),
        ("range_int", "UPDATE tseries SET eltype=1 WHERE id=:id", TypeError),
        ("range_int", "INSERT INTO attributes VALUES(:id,'jeltype','Int64')", TypeError),
    ],
)
def test_malformed_array_metadata_and_markers_are_refused(tmp_path, name, statement, error):
    path = tmp_path / "malformed.daec"
    shutil.copyfile(FIXTURES / "julia_arrays_unit.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        (oid,) = conn.execute("SELECT id FROM objects WHERE name=?", (name,)).fetchone()
        conn.execute(statement, {"id": oid})
    with open_dataecon(path) as db, pytest.raises(error):
        db.read_array(name)


@NATIVE
def test_julia_fixture_vectors_ranges_and_unit_series(tmp_path):  # noqa: PLR0912, PLR0915
    fixture = FIXTURES / "julia_arrays_unit.daec"
    provenance = tomllib.loads(fixture.with_suffix(".toml").read_text(encoding="utf-8"))
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    with closing(sqlite3.connect(fixture)) as conn:
        stored_names = {name for (name,) in conn.execute("SELECT name FROM objects WHERE id != 0")}
    assert len(stored_names) == 163
    visited: set[str] = set()
    path = tmp_path / "arrays-unit.daec"
    with open_dataecon(fixture) as reference, open_dataecon(path, "a") as output:
        for dtype in DTYPES:
            token = {
                "?": "Bool",
                "<c8": "ComplexF32",
                "<c16": "ComplexF64",
            }.get(dtype, np.dtype(dtype).name.title().replace("Uint", "UInt"))
            for suffix in ("", "_empty"):
                name = f"array_{token}{suffix}"
                visited.add(name)
                value = reference.read_array(name)
                output.write_array(name, value)
                again = output.read_array(name)
                assert isinstance(value, np.ndarray)
                assert isinstance(again, np.ndarray)
                assert value.dtype == again.dtype
                assert value.tobytes() == again.tobytes()
                unit_name = f"unit_{token}{suffix}"
                visited.add(unit_name)
                unit = reference.read_series(unit_name)
                output.write_series(unit_name, unit)
                unit_again = output.read_series(unit_name)
                assert isinstance(unit, TSeries)
                assert isinstance(unit_again, TSeries)
                assert unit.firstdate == unit_again.firstdate
                assert unit.values.dtype == unit_again.values.dtype
                assert unit.values.tobytes() == unit_again.values.tobytes()
        for name in (
            "range_int",
            "range_int_empty",
            "range_monthly",
            "range_unit",
            "range_unit_empty",
        ):
            visited.add(name)
            value = reference.read_array(name)
            output.write_array(name, value)
            assert output.read_array(name) == value
        for code in sorted(_SCALAR_FREQUENCIES):
            if code == 11:
                label = "u"
            elif code == 12:
                label = "d"
            elif code == 13:
                label = "b"
            elif 17 <= code <= 23:
                label = f"w{code - 16}"
            elif code == 32:
                label = "m"
            elif 65 <= code <= 67:
                label = f"q{code - 64}"
            elif 129 <= code <= 134:
                label = f"h{code - 128}"
            else:
                label = f"y{code - 256}"
            for part in ("all", "min", "max"):
                name = f"range_{part}_{label}"
                visited.add(name)
                value = reference.read_array(name)
                output.write_array(name, value)
                assert output.read_array(name) == value
        for name in ("unit_min", "unit_max", "unit_max_span"):
            visited.add(name)
            value = reference.read_series(name)
            output.write_series(name, value)
            again = output.read_series(name)
            assert isinstance(value, TSeries)
            assert isinstance(again, TSeries)
            assert value.firstdate == again.firstdate
            assert value.values.dtype == again.values.dtype
            assert value.values.tobytes() == again.values.tobytes()
        for name in ("unit_mit_monthly", "unit_int128", "unit_complexf16"):
            visited.add(name)
            value = reference.read_series(name)
            assert isinstance(value, StoredSeries)
            output.write_series(name, value)
            assert output.read_series(name) == value
    assert visited == stored_names
    with closing(sqlite3.connect(path)) as conn:
        output_names = {name for (name,) in conn.execute("SELECT name FROM objects WHERE id != 0")}
    assert output_names == stored_names


@NATIVE
def test_write_array_refuses_before_overwrite_deletion(tmp_path):
    path = tmp_path / "overwrite.daec"
    with open_dataecon(path, "a") as db:
        db.write_array("kept", np.array([1, 2], dtype=np.int16))
        with pytest.raises(ValueError, match="preserves only"):
            db.write_array("kept", range(3, 5), overwrite=True)
        assert db.read_array("kept").tobytes() == np.array([1, 2], dtype=np.int16).tobytes()


@NATIVE
def test_read_outputs_survive_close_and_do_not_alias(tmp_path):
    path = tmp_path / "ownership.daec"
    source = np.array([1, 2, 3], dtype=np.int16)
    with open_dataecon(path, "a") as db:
        db.write_array("values", source)
        loaded = db.read_array("values")
        source[0] = 99
    assert isinstance(loaded, np.ndarray)
    assert loaded.tolist() == [1, 2, 3]
    loaded[0] = -7
    with open_dataecon(path) as db:
        assert db.read_array("values").tolist() == [1, 2, 3]


ARRAY_MARKER_FAULT = """
import json,sys,numpy as np
from tsecon.dataecon import DataEconError,open_dataecon
db=open_dataecon(sys.argv[1],'a')
report={}
try:
    db.write_array('target',np.empty(0,dtype=np.int16),overwrite=True)
except DataEconError as e:
    report['error']=[e.code,e.operation]
else:
    raise AssertionError('marker injection did not fail')
value=db.read_array('target')
report['dtype']=str(value.dtype)
try:
    db.close()
except DataEconError as e:
    report['close']=e.code
report['closed']=db.closed
print(json.dumps(report))
"""


@NATIVE
def test_array_marker_failure_leaves_documented_unmarked_residue(tmp_path):
    path = tmp_path / "marker-residue.daec"
    with open_dataecon(path, "a") as db:
        db.write_scalar("target", 7)
    with closing(sqlite3.connect(path)) as sql, sql:
        sql.execute(
            "CREATE TRIGGER fail_array_marker BEFORE INSERT ON attributes "
            "WHEN NEW.name='jeltype' BEGIN SELECT RAISE(ABORT,'injected'); END"
        )
    run = subprocess.run(
        [sys.executable, "-c", ARRAY_MARKER_FAULT, str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    report = json.loads(run.stdout)
    assert report["error"][0] == 19
    assert "unmarked object may read differently" in report["error"][1]
    assert report["dtype"] == "int64"
    assert report["close"] == 19
    assert report["closed"]
