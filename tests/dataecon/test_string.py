"""String scalar codec rules, native interchange and payload guards."""

import hashlib
import importlib.util
import shutil
import sqlite3
import tomllib
from contextlib import closing
from pathlib import Path

import pytest

from tsecon.dataecon import DataEconError, _codec, open_dataecon
from tsecon.dataecon._codec import decode_scalar, encode_scalar, validate_scalar_metadata

FIXTURES = Path(__file__).parent / "fixtures"
NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built",
)
# Mirrors the Julia fixture and the installed-wheel checker.
CASES = [
    ("str_ascii", "hello"),
    ("str_empty", ""),
    ("str_digits", "7"),
    ("str_space", " "),
    ("str_latin", "héllo wörld"),
    ("str_cjk", "日本語"),
    ("str_emoji", "🙂 ok"),
    ("str_combining", "é"),
    ("str_whitespace", "a\nb\tc\r\n"),
    ("str_delimiter", "a‖b"),
    ("str_slash", "a/b"),
    ("str_quote", 'say "hi"'),
    ("str_expression", "error(123)"),
    ("str_long", "x" * 1000),
    ("str_long_utf8", "é" * 500),
]
# Julia-written payloads outside the supported subset: Python refuses them
# instead of truncating or guessing (a lossless representation is future work).
REJECTED = {
    "ctl_str_embedded_nul": ValueError,  # Julia loads "a"; Python never truncates
    "ctl_str_invalid_utf8": ValueError,  # Julia returns invalid bytes unchecked
    "ctl_symbol": TypeError,  # jtype="Symbol" reconstruction marker
    "native_str_no_terminator": ValueError,
    "native_str_empty_payload": ValueError,
    "native_str_with_frequency": TypeError,
}


def assert_str(actual, expected):
    assert type(actual) is str
    assert actual == expected


@pytest.mark.parametrize(("name", "value"), CASES)
def test_string_codec_round_trips_utf8_with_terminator(name, value):
    kind, frequency, payload = encode_scalar(value)
    assert (kind, frequency) == (6, 0)
    assert payload == value.encode("utf-8") + b"\0"
    assert len(payload) == len(value.encode("utf-8")) + 1
    assert_str(decode_scalar(kind, frequency, payload), value)


@pytest.mark.parametrize("value", ["a\0b", "abc\0", "\0", "\0abc"])
def test_string_codec_rejects_embedded_nul(value):
    with pytest.raises(ValueError, match="NUL"):
        encode_scalar(value)


@pytest.mark.parametrize("value", ["\udc80", "ab\ud800"])
def test_string_codec_rejects_lone_surrogates(value):
    with pytest.raises(ValueError, match="UTF-8"):
        encode_scalar(value)


def test_string_codec_size_limit(monkeypatch):
    monkeypatch.setattr(_codec, "MAX_BYTES", 8)
    assert encode_scalar("abcdefg")[2] == b"abcdefg\0"
    with pytest.raises(ValueError, match="size limit"):
        encode_scalar("abcdefgh")
    with pytest.raises(ValueError, match="size limit"):
        validate_scalar_metadata((1, 6, 0, 9))
    validate_scalar_metadata((1, 6, 0, 8))


@pytest.mark.parametrize(
    "value",
    [b"hi", bytearray(b"hi"), memoryview(b"hi"), ["hi"], ("hi",), None, 7, 1.5],
)
def test_string_codec_rejects_non_str(value):
    if isinstance(value, (int, float)):
        encode_scalar(value)  # numeric kinds, not strings
        return
    with pytest.raises(TypeError):
        encode_scalar(value)


def test_string_codec_rejects_subclass_without_conversion():
    class CustomStr(str):
        def encode(self, *args, **kwargs):
            raise AssertionError("Implicit conversion called")

    with pytest.raises(TypeError):
        encode_scalar(CustomStr("hi"))


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b"hi", "NUL-terminated"),
        (b"h\0i\0", "NUL-terminated"),
        (b"\0\0", "NUL-terminated"),
        (b"f\xffo\0", "UTF-8"),
        (b"\xed\xa0\x80\0", "UTF-8"),
    ],
)
def test_string_decode_rejects_malformed_payloads(payload, match):
    with pytest.raises(ValueError, match=match):
        decode_scalar(6, 0, payload)


def test_string_decode_accepts_only_nul_as_empty():
    assert_str(decode_scalar(6, 0, b"\0"), "")


@pytest.mark.parametrize("metadata", [(1, 6, 32, 3), (1, 6, 11, 1), (2, 6, 0, 3), (1, 7, 0, 3)])
def test_string_metadata_guard_rejects_frequency_class_and_other_type(metadata):
    with pytest.raises(TypeError):
        validate_scalar_metadata(metadata)


@pytest.mark.parametrize("nbytes", [0, -1, 2**62])
def test_string_metadata_guard_rejects_bad_lengths(nbytes):
    with pytest.raises(ValueError):
        validate_scalar_metadata((1, 6, 0, nbytes))


def test_string_metadata_guard_accepts_any_terminated_length():
    for nbytes in (1, 2, 1001, _codec.MAX_BYTES):
        validate_scalar_metadata((1, 6, 0, nbytes))


@NATIVE
def test_julia_string_fixture_values_and_controls():
    fixture = FIXTURES / "julia_string_scalars.daec"
    provenance = tomllib.loads(fixture.with_suffix(".toml").read_text(encoding="utf-8"))
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    with open_dataecon(fixture) as db:
        results = [(db.read_scalar(name), value) for name, value in CASES]
        for name, error in REJECTED.items():
            with pytest.raises(error):
                db.read_scalar(name)
        assert_str(db.read_scalar("native_str_only_nul"), "")
    for actual, expected in results:
        assert_str(actual, expected)
    with closing(sqlite3.connect(fixture)) as conn:
        for name, value in CASES:
            row = conn.execute(
                "SELECT o.class,o.type,s.frequency,s.value FROM objects o JOIN scalars s "
                "USING(id) WHERE o.name=?",
                (name,),
            ).fetchone()
            assert row == (1, 6, 0, value.encode("utf-8") + b"\0")
        assert conn.execute(
            "SELECT name,value FROM attributes WHERE id="
            "(SELECT id FROM objects WHERE name='ctl_symbol')"
        ).fetchall() == [("jtype", "Symbol")]
        assert conn.execute(
            "SELECT count(*) FROM attributes WHERE id!=0 AND id NOT IN "
            "(SELECT id FROM objects WHERE name='ctl_symbol')"
        ).fetchone() == (0,)
        assert conn.execute(
            "SELECT value FROM scalars WHERE id="
            "(SELECT id FROM objects WHERE name='ctl_str_embedded_nul')"
        ).fetchone() == (b"a\0b\0",)


@NATIVE
def test_string_roundtrip_storage_and_shared_namespace(tmp_path):
    path = tmp_path / "strings.daec"
    with open_dataecon(path, "a") as db:
        for name, value in CASES:
            db.write_scalar(name, value)
        db.write_scalar("rate", 1.25)
        db.write_scalar("count", 7)
        results = [(db.read_scalar(name), value) for name, value in CASES]
        with pytest.raises(DataEconError) as caught:
            db.write_scalar("str_ascii", "again")
        assert caught.value.code == -985
        with pytest.raises(DataEconError):
            db.write_scalar("rate", "1.25")
        with pytest.raises(DataEconError):
            db.read_series("str_ascii")
        with pytest.raises(ValueError, match="NUL"):
            db.write_scalar("bad", "a\0b")
        with pytest.raises(DataEconError) as caught:
            db.read_scalar("bad")
        assert caught.value.code == -989
        assert db.read_scalar("rate") == 1.25
        assert db.read_scalar("count") == 7
    for actual, expected in results:
        assert_str(actual, expected)
    with pytest.raises(ValueError, match="closed"):
        db.write_scalar("later", "x")
    with open_dataecon(path) as db:
        with pytest.raises(ValueError, match="read-only"):
            db.write_scalar("readonly", "x")
        assert_str(db.read_scalar("str_cjk"), "日本語")
    with closing(sqlite3.connect(path)) as conn:
        for name, value in CASES:
            assert conn.execute(
                "SELECT o.class,o.type,s.frequency,s.value FROM objects o JOIN scalars s "
                "USING(id) WHERE o.name=?",
                (name,),
            ).fetchone() == (1, 6, 0, value.encode("utf-8") + b"\0")
        assert conn.execute("SELECT count(*) FROM attributes WHERE id!=0").fetchone() == (0,)


@NATIVE
@pytest.mark.parametrize(
    ("kind", "frequency", "payload", "error"),
    [
        (6, 32, b"hi\0", TypeError),
        (6, 0, b"", ValueError),
        (7, 0, b"hi\0", TypeError),
    ],
)
def test_string_backend_validates_metadata_before_storage(
    tmp_path, kind, frequency, payload, error
):
    with open_dataecon(tmp_path / "invalid.daec", "a") as db:
        with pytest.raises(error):
            db._handle.write_scalar("bad", kind, frequency, payload)
        with pytest.raises(DataEconError) as caught:
            db.read_scalar("bad")
        assert caught.value.code == -989
        db.write_scalar("good", "ok")
        assert_str(db.read_scalar("good"), "ok")


@NATIVE
@pytest.mark.parametrize(
    ("sql", "error"),
    [
        ("UPDATE scalars SET value=NULL", ValueError),
        ("UPDATE scalars SET value=X'6869'", ValueError),  # terminator removed
        ("UPDATE scalars SET value=X'680069'", ValueError),  # interior NUL, no terminator
        ("UPDATE scalars SET value=X'ff00'", ValueError),  # invalid UTF-8
        ("UPDATE scalars SET frequency=32", TypeError),
        ("UPDATE objects SET type=7", TypeError),
        ("UPDATE objects SET type=4", ValueError),  # eight-byte rule for numeric kinds
    ],
)
def test_string_malformed_storage(tmp_path, sql, error):
    path = tmp_path / "malformed.daec"
    shutil.copyfile(FIXTURES / "julia_string_scalars.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(f"{sql} WHERE id=(SELECT id FROM objects WHERE name='str_ascii')")
    with open_dataecon(path, "a") as db:
        with pytest.raises(error):
            db.read_scalar("str_ascii")
        assert_str(db.read_scalar("str_latin"), "héllo wörld")
        db.write_scalar("good", "still fine")
        assert_str(db.read_scalar("good"), "still fine")


@NATIVE
@pytest.mark.parametrize("key", ["jtype", "jeltype"])
@pytest.mark.parametrize("value", ["String", "Symbol", "error(123)", None])
def test_string_rejects_all_reconstruction_attributes(tmp_path, key, value):
    path = tmp_path / "attribute.daec"
    shutil.copyfile(FIXTURES / "julia_string_scalars.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "INSERT INTO attributes SELECT id,?,? FROM objects WHERE name='str_ascii'",
            (key, value),
        )
    with open_dataecon(path) as db:
        with pytest.raises(TypeError, match="reconstruction"):
            db.read_scalar("str_ascii")
        assert_str(db.read_scalar("str_digits"), "7")
