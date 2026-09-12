"""Boolean writes use Julia's unmarked Int8 scalar representation."""

import importlib.util
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from tsecon.dataecon import DataEconError, open_dataecon
from tsecon.dataecon._codec import decode_scalar, encode_scalar

NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built",
)
VALUES = [False, True, np.bool_(False), np.bool_(True)]


@pytest.mark.parametrize("value", VALUES)
def test_bool_encodes_as_int8_and_decodes_without_guessing(value):
    encoded = encode_scalar(value)
    assert encoded == (1, 0, b"\x01" if value else b"\x00")
    restored = decode_scalar(*encoded)
    assert type(restored) is np.int8
    assert restored == int(value)
    assert encode_scalar(restored) == encoded
    # The file cannot distinguish a Boolean from an Int8 with the same byte.
    assert type(decode_scalar(1, 0, b"\x02")) is np.int8
    assert decode_scalar(1, 0, b"\x02") == 2


@pytest.mark.parametrize("raw", [2, 127, 128, 255])
def test_boolean_extracted_from_nonzero_array_bytes_stores_one(raw):
    # NumPy canonicalizes the Boolean when extracting the scalar from the array.
    value = np.frombuffer(bytes([raw]), dtype=np.bool_)[0]
    assert type(value) is np.bool_
    assert bool(value)
    assert encode_scalar(value) == (1, 0, b"\x01")


class Truthy:
    def __bool__(self):
        raise AssertionError("must not invoke arbitrary truth conversion")


@pytest.mark.parametrize("value", [Truthy(), np.array(True), np.array([False])])
def test_truth_convertible_objects_are_not_boolean_scalars(value):
    with pytest.raises(TypeError):
        encode_scalar(value)


@NATIVE
def test_julia_boolean_fixture_python_writes_and_raw_metadata(tmp_path):
    fixture = Path(__file__).parent / "fixtures/julia_numeric_widths.daec"
    path = tmp_path / "bool.daec"
    with open_dataecon(fixture) as julia, open_dataecon(path, "a") as db:
        for name, value in zip(("false", "true", "np_false", "np_true"), VALUES, strict=True):
            reference = julia.read_scalar("wctl_bool_true" if value else "wctl_bool_false")
            assert type(reference) is np.int8
            db.write_scalar(name, value)
            actual = db.read_scalar(name)
            assert type(actual) is np.int8
            assert actual == reference
            db.write_scalar(name + "_rewrite", actual)
        with pytest.raises(DataEconError):
            db.write_scalar("true", False)
        db.write_scalar("true", False, overwrite=True)
        assert db.read_scalar("true") == np.int8(0)
    with closing(sqlite3.connect(path)) as conn:
        rows = conn.execute(
            "SELECT o.name,o.class,o.type,s.frequency,length(s.value),hex(s.value) "
            "FROM objects o JOIN scalars s USING(id) ORDER BY o.name"
        ).fetchall()
        assert len(rows) == 8
        for name, cls, kind, frequency, width, payload in rows:
            assert (cls, kind, frequency, width) == (1, 1, 0, 1)
            # The original "true" object was overwritten with False above.
            expected = "01" if name in {"true_rewrite", "np_true", "np_true_rewrite"} else "00"
            assert payload == expected
        assert conn.execute("SELECT count(*) FROM attributes WHERE id != 0").fetchone() == (0,)
    with open_dataecon(path) as db, pytest.raises(ValueError, match="read-only"):
        db.write_scalar("no", True)
    with pytest.raises(ValueError, match="closed"):
        db.write_scalar("no", True)
