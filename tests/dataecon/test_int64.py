"""Int64 scalar codec exactness, native interchange and representation guards."""

import hashlib
import importlib.util
import shutil
import sqlite3
import struct
import tomllib
from contextlib import closing
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from tsecon import MIT, Duration, Monthly
from tsecon.dataecon import DataEconError, open_dataecon
from tsecon.dataecon._codec import decode_scalar, encode_scalar, validate_scalar_metadata

FIXTURES = Path(__file__).parent / "fixtures"
NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built",
)
# Mirrors the Julia fixture and the installed-wheel checker.
CASES = [
    ("int_zero", 0),
    ("int_one", 1),
    ("int_negative_one", -1),
    ("int_seven", 7),
    ("int_negative_seven", -7),
    ("int_pow53", 2**53),
    ("int_pow53_plus_one", 2**53 + 1),
    ("int_pow53_minus_one", 2**53 - 1),
    ("int_negative_pow53_minus_one", -(2**53 + 1)),
    ("int_pow62_plus_one", 2**62 + 1),
    ("int_max", 2**63 - 1),
    ("int_min", -(2**63)),
    ("int_min_plus_one", -(2**63) + 1),
    ("int_max_minus_one", 2**63 - 2),
]
NATIVE_CASES = [
    ("int_native_pow53_plus_one", 2**53 + 1),
    ("int_native_min", -(2**63)),
    ("int_native_max", 2**63 - 1),
    ("int_native_negative_one", -1),
]
# Julia-written neighbours of Int64 that Python must refuse to coerce.
CONTROLS = {
    "ctl_bool_true": ValueError,
    "ctl_int32": ValueError,
    "ctl_int128": ValueError,
    "ctl_uint64": TypeError,
    "ctl_rational": TypeError,
}
# Neighbours that later slices made supported; they must never read as int.
SUPPORTED_CONTROLS = {
    "ctl_duration_monthly": Duration(Monthly(), 3),
    "ctl_mit_monthly": MIT(Monthly(), 24288),
    "ctl_string": "7",
}


def assert_int(actual, expected):
    assert type(actual) is int
    assert actual == expected


@pytest.mark.parametrize(("name", "value"), CASES)
@pytest.mark.parametrize("constructor", [int, np.int64])
def test_int64_codec_is_exact(name, value, constructor):
    kind, frequency, payload = encode_scalar(constructor(value))
    assert (kind, frequency) == (1, 0)
    assert payload == struct.pack("<q", value) == np.int64(value).tobytes()
    assert_int(decode_scalar(kind, frequency, payload), value)


@pytest.mark.parametrize(
    "value", [2**53 + 1, -(2**53 + 1), 2**62 + 1, 2**63 - 1, -(2**63) + 1, 2**63 - 2]
)
def test_int64_codec_does_not_round_through_float(value):
    assert int(float(value)) != value
    kind, frequency, payload = encode_scalar(value)
    assert_int(decode_scalar(kind, frequency, payload), value)


def test_int64_and_float64_kinds_are_distinct():
    assert encode_scalar(1)[0] == 1
    assert encode_scalar(1.0)[0] == 4
    assert encode_scalar(1)[2] != encode_scalar(1.0)[2]
    assert type(decode_scalar(4, 0, struct.pack("<d", 1.0))) is float
    assert type(decode_scalar(1, 0, struct.pack("<q", 1))) is int


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        np.bool_(True),
        np.int8(7),
        np.int16(7),
        np.int32(7),
        np.intc(7),
        np.uint8(7),
        np.uint32(7),
        np.uint64(7),
        np.float32(7),
        Fraction(7),
        Decimal(7),
        7j,
        b"7",
        np.array(7),
        np.array([7]),
        None,
    ],
)
def test_int64_codec_rejects_other_representations(value):
    with pytest.raises(TypeError):
        encode_scalar(value)


@pytest.mark.parametrize("value", [2**63, -(2**63) - 1, 2**64, -(2**100), 10**30])
def test_int64_codec_rejects_out_of_range_python_int(value):
    with pytest.raises(ValueError, match="signed 64-bit"):
        encode_scalar(value)


def test_int64_codec_does_not_invoke_custom_conversion():
    class CustomInt(int):
        def __index__(self):
            raise AssertionError("Implicit conversion called")

        def __int__(self):
            raise AssertionError("Implicit conversion called")

    class Indexable:
        def __index__(self):
            raise AssertionError("Implicit conversion called")

        def __int__(self):
            raise AssertionError("Implicit conversion called")

    for value in (CustomInt(7), Indexable()):
        with pytest.raises(TypeError):
            encode_scalar(value)


@pytest.mark.parametrize(
    "metadata",
    [(1, 1, 11, 8), (1, 1, 12, 8), (1, 1, 192, 8), (1, 3, 0, 8), (1, 2, 0, 8), (2, 1, 0, 8)],
)
def test_int64_metadata_guard_rejects_unsupported_duration_date_unsigned_and_class(metadata):
    with pytest.raises(TypeError):
        validate_scalar_metadata(metadata)


@pytest.mark.parametrize("nbytes", [0, 1, 2, 4, 7, 9, 16, -8, 2**62])
def test_int64_metadata_guard_rejects_other_widths(nbytes):
    with pytest.raises(ValueError, match="eight"):
        validate_scalar_metadata((1, 1, 0, nbytes))
    if 0 <= nbytes <= 16:
        with pytest.raises(ValueError, match="eight"):
            decode_scalar(1, 0, bytes(nbytes))


def test_int64_metadata_guard_accepts_exact_encoding():
    validate_scalar_metadata((1, 1, 0, 8))
    validate_scalar_metadata((1, 4, 0, 8))


@NATIVE
def test_julia_int64_fixture_values_and_controls():
    fixture = FIXTURES / "julia_int64_scalars.daec"
    provenance = tomllib.loads(fixture.with_suffix(".toml").read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    with open_dataecon(fixture) as db:
        results = [(db.read_scalar(name), value) for name, value in CASES + NATIVE_CASES]
        for name, error in CONTROLS.items():
            with pytest.raises(error):
                db.read_scalar(name)
        for name, expected in SUPPORTED_CONTROLS.items():
            loaded = db.read_scalar(name)
            assert type(loaded) is type(expected)
            assert loaded == expected
        assert_int(db.read_scalar("int_max"), 2**63 - 1)
    for actual, expected in results:
        assert_int(actual, expected)
    with closing(sqlite3.connect(fixture)) as conn:
        for name, value in CASES + NATIVE_CASES:
            row = conn.execute(
                "SELECT o.class,o.type,s.frequency,s.value FROM objects o JOIN scalars s "
                "USING(id) WHERE o.name=?",
                (name,),
            ).fetchone()
            assert row == (1, 1, 0, struct.pack("<q", value))
        assert conn.execute(
            "SELECT count(*) FROM attributes WHERE id!=0 AND id NOT IN "
            "(SELECT id FROM objects WHERE name='ctl_rational')"
        ).fetchone() == (0,)
    native = importlib.import_module("tsecon.dataecon._native")
    layout = provenance["layout"]["scalar_t"]
    assert native.abi_layout()["scalar_t"] == (layout["size"], tuple(layout["offsets"]))


@NATIVE
def test_int64_roundtrip_storage_and_shared_namespace(tmp_path):
    path = tmp_path / "integers.daec"
    with open_dataecon(path, "a") as db:
        for name, value in CASES:
            db.write_scalar(name, value)
            db.write_scalar(f"{name}_np", np.int64(value))
        db.write_scalar("rate", 1.25)
        results = [(db.read_scalar(name), value) for name, value in CASES]
        results += [(db.read_scalar(f"{name}_np"), value) for name, value in CASES]
        with pytest.raises(DataEconError) as caught:
            db.write_scalar("int_seven", 7.0)
        assert caught.value.code == -985
        with pytest.raises(DataEconError):
            db.write_scalar("rate", 1)
        with pytest.raises(DataEconError):
            db.read_series("int_seven")
        with pytest.raises(ValueError, match="signed 64-bit"):
            db.write_scalar("too_big", 2**63)
        with pytest.raises(DataEconError) as caught:
            db.read_scalar("too_big")
        assert caught.value.code == -989
        assert type(db.read_scalar("rate")) is float
    for actual, expected in results:
        assert_int(actual, expected)
    with pytest.raises(ValueError, match="closed"):
        db.write_scalar("later", 1)
    with open_dataecon(path) as db:
        with pytest.raises(ValueError, match="read-only"):
            db.write_scalar("readonly", 1)
        assert_int(db.read_scalar("int_min"), -(2**63))
    with closing(sqlite3.connect(path)) as conn:
        for name, value in CASES:
            assert conn.execute(
                "SELECT o.class,o.type,s.frequency,s.value FROM objects o JOIN scalars s "
                "USING(id) WHERE o.name=?",
                (name,),
            ).fetchone() == (1, 1, 0, struct.pack("<q", value))
        assert conn.execute("SELECT count(*) FROM attributes WHERE id!=0").fetchone() == (0,)


@NATIVE
@pytest.mark.parametrize(
    ("kind", "payload", "error"),
    [
        (2, bytes(8), TypeError),
        (3, bytes(8), TypeError),
        (1, bytes(4), ValueError),
        (1, b"", ValueError),
    ],
)
def test_int64_backend_validates_kind_and_width_before_storage(tmp_path, kind, payload, error):
    with open_dataecon(tmp_path / "invalid.daec", "a") as db:
        with pytest.raises(error):
            db._handle.write_scalar("bad", kind, 0, payload)
        with pytest.raises(DataEconError) as caught:
            db.read_scalar("bad")
        assert caught.value.code == -989
        db.write_scalar("good", -1)
        assert_int(db.read_scalar("good"), -1)


@NATIVE
@pytest.mark.parametrize(
    ("sql", "error"),
    [
        ("UPDATE scalars SET value=NULL", ValueError),
        ("UPDATE scalars SET value=zeroblob(4)", ValueError),
        ("UPDATE scalars SET value=zeroblob(16)", ValueError),
        ("UPDATE scalars SET frequency=11", TypeError),
        ("UPDATE objects SET type=2", TypeError),
        ("UPDATE objects SET type=3", TypeError),
    ],
)
def test_int64_malformed_storage(tmp_path, sql, error):
    path = tmp_path / "malformed.daec"
    shutil.copyfile(FIXTURES / "julia_int64_scalars.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(f"{sql} WHERE id=(SELECT id FROM objects WHERE name='int_seven')")
    with open_dataecon(path, "a") as db:
        with pytest.raises(error):
            db.read_scalar("int_seven")
        assert_int(db.read_scalar("int_negative_seven"), -7)
        db.write_scalar("good", 2**53 + 1)
        assert_int(db.read_scalar("good"), 2**53 + 1)


@NATIVE
@pytest.mark.parametrize("key", ["jtype", "jeltype"])
@pytest.mark.parametrize("value", ["Int64", "Duration{Monthly}", "error(123)", None])
def test_int64_rejects_all_reconstruction_attributes(tmp_path, key, value):
    path = tmp_path / "attribute.daec"
    shutil.copyfile(FIXTURES / "julia_int64_scalars.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "INSERT INTO attributes SELECT id,?,? FROM objects WHERE name='int_seven'",
            (key, value),
        )
    with open_dataecon(path) as db:
        with pytest.raises(TypeError, match="reconstruction"):
            db.read_scalar("int_seven")
        assert_int(db.read_scalar("int_negative_seven"), -7)
