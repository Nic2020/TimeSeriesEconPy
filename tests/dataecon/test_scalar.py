"""Float64 scalar codecs, native interchange and representation guards.

Int64, string and date/duration scalar coverage lives in ``test_int64.py``,
``test_string.py`` and ``test_dates.py``.
"""

import hashlib
import importlib.util
import math
import shutil
import sqlite3
import struct
import tomllib
from contextlib import closing
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from tsecon import TSeries, mm
from tsecon.dataecon import DataEconError, _codec, open_dataecon
from tsecon.dataecon._codec import decode_scalar, encode_scalar, validate_scalar_metadata

FIXTURES = Path(__file__).parent / "fixtures"
NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built",
)
CASES = [
    ("scalar_finite", 1.25),
    ("scalar_negative", -2.5),
    ("scalar_zero", 0.0),
    ("scalar_negative_zero", -0.0),
    ("scalar_nan", float("nan")),
    ("scalar_positive_inf", float("inf")),
    ("scalar_negative_inf", float("-inf")),
]


def assert_scalar(actual, expected):
    assert type(actual) is float
    if math.isnan(expected):
        assert math.isnan(actual)
    else:
        assert actual == expected
        assert math.copysign(1.0, actual) == math.copysign(1.0, expected)


@pytest.mark.parametrize(("name", "value"), CASES)
@pytest.mark.parametrize("constructor", [float, np.float64])
def test_scalar_codec(name, value, constructor):
    kind, frequency, payload = encode_scalar(constructor(value))
    assert (kind, frequency) == (4, 0)
    assert len(payload) == 8
    assert_scalar(decode_scalar(kind, frequency, payload), value)


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        Decimal("1.25"),
        np.array(1.25),
        np.array([1.25]),
        np.array(1.25, dtype=np.float32),
        b"1.25",
        None,
    ],
)
def test_scalar_codec_rejects_implicit_coercion(value):
    # np.float32/np.int32/complex are their own exact-width kinds; see test_widths.py.
    with pytest.raises(TypeError):
        encode_scalar(value)


def test_scalar_codec_does_not_invoke_custom_conversion():
    class CustomFloat(float):
        def __float__(self):
            pytest.fail("Implicit conversion called")

    with pytest.raises(TypeError):
        encode_scalar(CustomFloat(1.25))


def test_scalar_codec_uses_type_identity_not_custom_equality():
    class PretendFloat(type):
        __hash__ = None

        def __eq__(cls, other):
            return True

    class Value(metaclass=PretendFloat):
        def __float__(self):
            pytest.fail("Implicit conversion called")

    with pytest.raises(TypeError):
        encode_scalar(Value())


@pytest.mark.parametrize("length", [-1, 0, 1, 3, 7, 9, 16, 2**62])
def test_scalar_size_guard(length):
    # 2 and 4 bytes are Float16/Float32 since the width slice.
    with pytest.raises(ValueError):
        validate_scalar_metadata((1, 4, 0, length))


@pytest.mark.parametrize("metadata", [(2, 4, 0, 8), (1, 7, 0, 8), (1, 3, 0, 8), (1, 4, 32, 8)])
def test_scalar_type_guard(metadata):
    with pytest.raises(TypeError):
        validate_scalar_metadata(metadata)


@pytest.mark.parametrize(
    "operation", [lambda: encode_scalar(1.25), lambda: decode_scalar(4, 0, bytes(8))]
)
def test_scalar_codec_rejects_big_endian(monkeypatch, operation):
    monkeypatch.setattr(_codec.sys, "byteorder", "big")
    with pytest.raises(RuntimeError, match="little-endian"):
        operation()


@NATIVE
@pytest.mark.parametrize("payload", [b"", bytes(7), bytes(9)])
def test_scalar_backend_validates_payload_before_storage(tmp_path, payload):
    with open_dataecon(tmp_path / "invalid-write.daec", "a") as db:
        with pytest.raises(ValueError, match="eight"):
            db._handle.write_scalar("bad", 4, 0, payload)
        with pytest.raises(DataEconError) as caught:
            db.read_scalar("bad")
        assert caught.value.code == -989
        db.write_scalar("good", 1.25)
        assert db.read_scalar("good") == 1.25


@NATIVE
def test_julia_scalar_fixture_and_abi():
    fixture = FIXTURES / "julia_float64_scalars.daec"
    provenance = tomllib.loads(fixture.with_suffix(".toml").read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    with open_dataecon(fixture) as db:
        results = [(db.read_scalar(name), value) for name, value in CASES]
        # The Float32 control is a supported width since the width slice: it must
        # come back as an exact NumPy float32, never widened to a Python float.
        narrow = db.read_scalar("scalar_float32")
        assert type(narrow) is np.float32
        assert narrow == np.float32(1.25)
        # The Int64 control is a supported representation since Int64 support landed.
        integer = db.read_scalar("scalar_integer")
        assert type(integer) is int
        assert integer == 7
    for actual, expected in results:
        assert_scalar(actual, expected)
    native = importlib.import_module("tsecon.dataecon._native")
    layout = provenance["layout"]["scalar_t"]
    assert native.abi_layout()["scalar_t"] == (layout["size"], tuple(layout["offsets"]))


@NATIVE
def test_scalar_roundtrip_metadata_and_owner_guards(tmp_path):
    path = tmp_path / "scalars.daec"
    with open_dataecon(path, "a") as db:
        for name, value in CASES:
            db.write_scalar(name, np.float64(value))
        results = [(db.read_scalar(name), value) for name, value in CASES]
        with pytest.raises(DataEconError) as caught:
            db.write_scalar("scalar_finite", 2.0)
        assert caught.value.code == -985
        db.write_series("series", TSeries(mm(2024, 1), np.array([1.25])))
        with pytest.raises(DataEconError):
            db.write_scalar("series", 1.25)
        with pytest.raises(DataEconError):
            db.write_series("scalar_finite", TSeries(mm(2024, 1), np.empty(0)))
        with pytest.raises(DataEconError):
            db.read_scalar("series")
        with pytest.raises(DataEconError):
            db.read_series("scalar_finite")
        with pytest.raises(TypeError):
            db.write_scalar("invalid", True)
        with pytest.raises(DataEconError) as caught:
            db.read_scalar("invalid")
        assert caught.value.code == -989
        assert db.read_scalar("scalar_finite") == 1.25
    for actual, expected in results:
        assert_scalar(actual, expected)
    with pytest.raises(ValueError, match="closed"):
        db.read_scalar("scalar_finite")
    with pytest.raises(ValueError, match="closed"):
        db.write_scalar("new", 1.25)
    with open_dataecon(path) as db:
        with pytest.raises(ValueError, match="read-only"):
            db.write_scalar("readonly", 1.25)
        assert db.read_scalar("scalar_finite") == 1.25
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute(
            "SELECT o.class,o.type,s.frequency,s.value FROM objects o JOIN scalars s USING(id) "
            "WHERE o.name='scalar_finite'"
        ).fetchone() == (1, 4, 0, struct.pack("<d", 1.25))
        assert conn.execute("SELECT count(*) FROM attributes WHERE id!=0").fetchone() == (0,)


@NATIVE
@pytest.mark.parametrize(
    ("sql", "error"),
    [
        ("UPDATE scalars SET value=NULL", ValueError),
        ("UPDATE scalars SET value=zeroblob(7)", ValueError),
        ("UPDATE scalars SET value=zeroblob(9)", ValueError),
        ("UPDATE scalars SET frequency=32", TypeError),
        # Types 1 and 2 would read the same eight bytes as Int64/UInt64: the
        # format cannot tell a retyped Float64 apart. Type 7 and date codes fail.
        ("UPDATE objects SET type=7 WHERE class=1", TypeError),
        ("UPDATE objects SET type=3 WHERE class=1", TypeError),
    ],
)
def test_scalar_malformed_storage(tmp_path, sql, error):
    path = tmp_path / "malformed.daec"
    shutil.copyfile(FIXTURES / "julia_float64_scalars.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(sql)
    with open_dataecon(path, "a") as db:
        with pytest.raises(error):
            db.read_scalar("scalar_finite")
        db.write_scalar("good", 2.5)
        assert db.read_scalar("good") == 2.5


@NATIVE
@pytest.mark.parametrize("key", ["jtype", "jeltype"])
@pytest.mark.parametrize("value", ["Float64", "error(123)", None])
def test_scalar_rejects_all_reconstruction_attributes(tmp_path, key, value):
    path = tmp_path / "attribute.daec"
    shutil.copyfile(FIXTURES / "julia_float64_scalars.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "INSERT INTO attributes SELECT id,?,? FROM objects WHERE name='scalar_finite'",
            (key, value),
        )
    with open_dataecon(path) as db:
        with pytest.raises(TypeError, match="reconstruction"):
            db.read_scalar("scalar_finite")
        assert db.read_scalar("scalar_negative") == -2.5
