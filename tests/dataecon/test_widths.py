"""Narrow, unsigned and complex numeric scalar widths: codec exactness and interchange.

Julia stores Float16/32, Int8/16/32, UInt8..UInt64 and ComplexF32/F64 at their
own width with no marker and reloads them by (type, nbytes). Python accepts the
exact NumPy scalar classes (plus Python ``complex``) and returns the sized NumPy
scalar, or the built-in for the eight-byte Int64/Float64 and sixteen-byte
ComplexF64 encodings. Int128, UInt128 and ComplexF16 have no NumPy scalar type
and stay rejected; a Julia ``Bool`` is byte-identical to ``Int8`` and reads as
``np.int8``, exactly as Julia reloads it.
"""

import hashlib
import importlib.util
import math
import shutil
import sqlite3
import struct
import subprocess
import sys
import tomllib
from contextlib import closing
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from tsecon.dataecon import DataEconError, _codec, open_dataecon
from tsecon.dataecon._codec import decode_scalar, encode_scalar, validate_scalar_metadata

FIXTURES = Path(__file__).parent / "fixtures"
NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built",
)
NAN, INF = float("nan"), float("inf")


def _signed(cls):
    info = np.iinfo(cls)
    return [
        ("zero", cls(0)),
        ("one", cls(1)),
        ("negative_one", cls(-1)),
        ("seven", cls(7)),
        ("min", cls(info.min)),
        ("max", cls(info.max)),
        ("min_plus_one", cls(info.min + 1)),
        ("max_minus_one", cls(info.max - 1)),
    ]


def _unsigned(cls):
    info = np.iinfo(cls)
    return [
        ("zero", cls(0)),
        ("one", cls(1)),
        ("seven", cls(7)),
        ("max", cls(info.max)),
        ("max_minus_one", cls(info.max - 1)),
        ("high_bit", cls(info.max // 2 + 1)),
    ]


def _floats(cls):
    info = np.finfo(cls)
    return [
        ("zero", cls(0.0)),
        ("negative_zero", cls(-0.0)),
        (
            "one_and_half" if cls is np.float16 else "one_quarter",
            cls(1.5 if cls is np.float16 else 1.25),
        ),
        ("tenth", cls(0.1)),
        ("max", cls(info.max)),
        ("negative_max", cls(-info.max)),
        ("min_normal", cls(info.tiny)),
        ("min_subnormal", cls(info.smallest_subnormal)),
        ("nan", cls(NAN)),
        ("inf", cls(INF)),
        ("negative_inf", cls(-INF)),
        # Precision loss happens in the caller, before storage: 2049 -> 2048 in
        # Float16 and 16777217 -> 16777216 in Float32, exactly as in Julia.
        (
            "two_pow_11_plus_one" if cls is np.float16 else "two_pow_24_plus_one",
            cls(2049 if cls is np.float16 else 16777217),
        ),
        ("pi", cls(math.pi)),
    ]


def _complexes(cls):
    part = np.float32 if cls is np.complex64 else np.float64
    info = np.finfo(part)
    return [
        ("plain", cls(complex(1.5, -2.25))),
        ("negative_zero_real", cls(complex(-0.0, 0.0))),
        ("negative_zero_imag", cls(complex(0.0, -0.0))),
        ("nan_real", cls(complex(NAN, 1.0))),
        ("inf_imag", cls(complex(1.0, -INF))),
        ("max_min", cls(complex(float(info.max), float(info.tiny)))),
        ("subnormal", cls(complex(float(info.smallest_subnormal), 0.0))),
        ("tenths", cls(complex(0.1, 0.2))),
    ]


# Mirrors the Julia fixture and the installed-wheel checker (names and values).
GROUPS = {
    "f16": _floats(np.float16),
    "f32": _floats(np.float32),
    "i8": _signed(np.int8),
    "i16": _signed(np.int16),
    "i32": _signed(np.int32),
    "u8": _unsigned(np.uint8),
    "u16": _unsigned(np.uint16),
    "u32": _unsigned(np.uint32),
    "u64": _unsigned(np.uint64),
    "c32": _complexes(np.complex64),
    "c64": _complexes(np.complex128),
}
CASES = [
    (f"w_{group}_{suffix}", value) for group, cases in GROUPS.items() for suffix, value in cases
]
# Native type code and byte width per group.
KINDS = {
    "f16": (4, 2), "f32": (4, 4), "i8": (1, 1), "i16": (1, 2), "i32": (1, 4),
    "u8": (2, 1), "u16": (2, 2), "u32": (2, 4), "u64": (2, 8), "c32": (5, 8), "c64": (5, 16),
}  # fmt: skip
# Read results: the sized NumPy class, or the built-in for ComplexF64.
RESULT_TYPES = {
    "f16": np.float16, "f32": np.float32, "i8": np.int8, "i16": np.int16, "i32": np.int32,
    "u8": np.uint8, "u16": np.uint16, "u32": np.uint32, "u64": np.uint64,
    "c32": np.complex64, "c64": complex,
}  # fmt: skip
# Julia-written controls in the fixture: unresolved widths, Bool and the marker experiment.
UNRESOLVED_CONTROLS = (
    "wctl_int128",
    "wctl_int128_max",
    "wctl_uint128",
    "wctl_uint128_max",
    "wctl_c16",
)
BOOL_CONTROLS = {"wctl_bool_true": np.int8(1), "wctl_bool_false": np.int8(0)}


def group_of(name):
    return name.split("_")[1]


def expected_bytes(value):
    if type(value) is complex:
        return struct.pack("<dd", value.real, value.imag)
    return value.tobytes()


def assert_same(actual, expected, group):
    """Require the exact result class and identical bits (NaN payloads included)."""
    assert type(actual) is RESULT_TYPES[group]
    assert expected_bytes(actual) == expected_bytes(expected)


@pytest.mark.parametrize(("name", "value"), CASES)
def test_width_codec_is_exact(name, value):
    group = group_of(name)
    kind, frequency, payload = encode_scalar(value)
    assert (kind, frequency, len(payload)) == (KINDS[group][0], 0, KINDS[group][1])
    assert payload == value.tobytes()
    assert_same(decode_scalar(kind, frequency, payload), value, group)


def test_width_codec_never_widens_narrow_floats():
    for value in (np.float16(0.1), np.float32(0.1)):
        _, _, payload = encode_scalar(value)
        assert len(payload) == value.itemsize
        assert payload != struct.pack("<d", 0.1)[: len(payload)]
        assert float(decode_scalar(4, 0, payload)) == float(value) != 0.1
    assert float(np.float16(2049)) == 2048
    assert float(np.float32(16777217)) == 16777216


def _alias_result(kind, width):
    if width == 8:
        return int if kind == 1 else np.uint64
    return np.dtype(f"<{'i' if kind == 1 else 'u'}{width}").type


# C-named integer classes reached through their dtype type codes, which every
# supported NumPy version defines (NumPy 1.26 has no np.long/np.ulong attribute).
C_NAMED_CODES = [("i", 1), ("I", 2), ("l", 1), ("L", 2), ("q", 1), ("Q", 2), ("p", 1), ("P", 2)]


@pytest.mark.parametrize(("code", "kind"), C_NAMED_CODES)
def test_width_codec_accepts_c_named_aliases_by_width(code, kind):
    # np.intc/np.uintc (Windows) and np.longlong/np.ulonglong (Linux/macOS) are
    # classes distinct from the sized names; acceptance is platform-independent
    # and reads return the sized class (or int/np.uint64) for that width.
    cls = np.dtype(code).type
    width = np.dtype(code).itemsize
    value = cls(7)
    code_out, frequency, payload = encode_scalar(value)
    assert (code_out, frequency, len(payload)) == (kind, 0, width)
    assert payload == value.tobytes()
    loaded = decode_scalar(code_out, frequency, payload)
    assert type(loaded) is _alias_result(kind, width)
    assert int(loaded) == 7


@pytest.mark.parametrize(
    "name", ["intc", "uintc", "longlong", "ulonglong", "long", "ulong", "int_", "uint", "intp"]
)
def test_width_codec_accepts_every_named_integer_class_this_numpy_defines(name):
    # Look in the module dictionary: on NumPy 1.26, getattr(np, "long") goes
    # through a __getattr__ that warns before raising AttributeError.
    cls = np.__dict__.get(name)
    if cls is None:
        pytest.skip(f"np.{name} is not defined by NumPy {np.__version__}")
    kind = 1 if np.dtype(cls).kind == "i" else 2
    width = np.dtype(cls).itemsize
    code_out, _, payload = encode_scalar(cls(7))
    assert (code_out, len(payload)) == (kind, width)
    assert type(decode_scalar(code_out, 0, payload)) is _alias_result(kind, width)


def test_width_class_table_is_discovered_from_type_codes():
    # The table must cover exactly the classes behind the numeric type codes,
    # with their true widths, independent of which attribute names exist.
    expected = {}
    for code in "bhilqpBHILQPefdFD":
        dtype = np.dtype(code)
        expected[dtype.type] = ({"i": 1, "u": 2, "f": 4, "c": 5}[dtype.kind], dtype.itemsize)
    assert expected == _codec._NUMPY_CLASSES
    for cls, entry in expected.items():
        assert _codec._numpy_class(cls) == entry
    assert _codec._numpy_class(np.bool_) is None
    assert _codec._numpy_class(np.longdouble) is None
    assert _codec._numpy_class(float) is None


def test_width_codec_imports_without_numpy_2_only_names():
    # NumPy 1.26 (the declared floor) has no np.long/np.ulong. Strip them from
    # the live NumPy module in a subprocess before importing the adapter; the
    # import and the alias paths must not depend on those attribute names.
    program = """
import numpy as np
for name in ("long", "ulong"):
    if name in np.__dict__:
        delattr(np, name)
import tsecon.dataecon
from tsecon.dataecon._codec import encode_scalar, decode_scalar
for code in "bhilqpBHILQPefdFD":
    cls = np.dtype(code).type
    kind, frequency, payload = encode_scalar(cls(1))
    assert len(payload) == np.dtype(code).itemsize, code
    assert decode_scalar(kind, frequency, payload) == 1, code
print("numpy-names-ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", program], check=True, timeout=120, capture_output=True, text=True
    )
    assert "numpy-names-ok" in result.stdout


@pytest.mark.parametrize("value", [8 + 3j, complex(-0.0, NAN), np.complex128(1.5 - 2.25j)])
def test_python_complex_and_complex128_are_complexf64(value):
    kind, frequency, payload = encode_scalar(value)
    assert (kind, frequency, len(payload)) == (5, 0, 16)
    assert payload == struct.pack("<dd", value.real, value.imag)
    loaded = decode_scalar(kind, frequency, payload)
    assert type(loaded) is complex
    assert struct.pack("<dd", loaded.real, loaded.imag) == payload


def test_uint64_above_int64_max_never_goes_through_signed_or_float():
    value = np.uint64(2**64 - 1)
    kind, _, payload = encode_scalar(value)
    assert kind == 2
    assert payload == b"\xff" * 8
    assert struct.unpack("<q", payload)[0] == -1
    loaded = decode_scalar(kind, 0, payload)
    assert type(loaded) is np.uint64
    assert int(loaded) == 2**64 - 1
    # The same bytes under the signed code are an Int64 -1: the type code decides.
    signed = decode_scalar(1, 0, payload)
    assert type(signed) is int
    assert signed == -1


class _Sub32(np.float32):
    pass


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        np.bool_(True),
        _Sub32(1.5),
        np.array(np.float16(1.5)),
        np.array([np.int8(1)]),
        np.str_("1.5"),
        np.datetime64("2024-01-01"),
        np.timedelta64(1, "D"),
        Fraction(1, 2),
        Decimal("1.5"),
        b"\x01",
        None,
    ],
)
def test_width_codec_rejects_other_representations(value):
    with pytest.raises(TypeError):
        encode_scalar(value)


class _Pretend(type):
    # A metaclass that makes the subclass hash and compare equal to np.float32:
    # a dictionary or set membership test on type(value) would accept it.
    def __hash__(cls):
        return hash(np.float32)

    def __eq__(cls, other):
        return other is np.float32


class _Spoofed(np.float32, metaclass=_Pretend):
    def tobytes(self, *args, **kwargs):
        raise RuntimeError("CUSTOM_CONVERSION_INVOKED")

    def __float__(self):
        raise RuntimeError("CUSTOM_CONVERSION_INVOKED")


def test_width_codec_rejects_metaclass_spoofed_subclass_without_conversion():
    value = _Spoofed(1.25)
    assert type(value) == np.float32  # noqa: E721 - the spoof is the point
    assert type(value) in {np.float32}
    assert type(value) is not np.float32
    with pytest.raises(TypeError, match="exact NumPy"):
        encode_scalar(value)


def test_width_codec_does_not_invoke_custom_conversion():
    class Indexable:
        def __index__(self):
            raise AssertionError("Implicit conversion called")

        def __int__(self):
            raise AssertionError("Implicit conversion called")

        def __float__(self):
            raise AssertionError("Implicit conversion called")

        def __complex__(self):
            raise AssertionError("Implicit conversion called")

    with pytest.raises(TypeError):
        encode_scalar(Indexable())


@pytest.mark.parametrize(
    ("kind", "nbytes"),
    [
        (1, 1),
        (1, 2),
        (1, 4),
        (1, 8),
        (2, 1),
        (2, 2),
        (2, 4),
        (2, 8),
        (4, 2),
        (4, 4),
        (4, 8),
        (5, 8),
        (5, 16),
    ],
)
def test_width_metadata_guard_accepts_supported_widths(kind, nbytes):
    validate_scalar_metadata((1, kind, 0, nbytes))


@pytest.mark.parametrize(
    ("kind", "nbytes"),
    [
        (1, 16),
        (2, 16),
        (5, 4),
        (1, 3),
        (1, 0),
        (2, 0),
        (2, 3),
        (2, 32),
        (4, 1),
        (4, 0),
        (4, 16),
        (5, 2),
        (5, 12),
        (5, 32),
        (5, 0),
        (1, -1),
        (2, 2**62),
    ],
)
def test_width_metadata_guard_rejects_other_widths(kind, nbytes):
    # Int128/UInt128/ComplexF16 are Julia-supported widths without a NumPy type;
    # the rest have no Julia loader method either.
    with pytest.raises(ValueError, match="width"):
        validate_scalar_metadata((1, kind, 0, nbytes))
    if 0 <= nbytes <= 32:
        with pytest.raises(ValueError, match="width"):
            decode_scalar(kind, 0, bytes(nbytes))


@pytest.mark.parametrize(
    "metadata",
    [
        (1, 2, 32, 8),
        (1, 2, 11, 4),
        (1, 4, 32, 8),
        (1, 4, 11, 4),
        (1, 5, 32, 16),
        (1, 5, 67, 8),
        (1, 7, 0, 8),
        (2, 2, 0, 8),
    ],
)
def test_width_metadata_guard_rejects_frequencies_other_types_and_classes(metadata):
    with pytest.raises(TypeError):
        validate_scalar_metadata(metadata)


def test_width_codec_rejects_big_endian(monkeypatch):
    monkeypatch.setattr(_codec.sys, "byteorder", "big")
    with pytest.raises(RuntimeError, match="little-endian"):
        encode_scalar(np.float16(1.5))
    with pytest.raises(RuntimeError, match="little-endian"):
        decode_scalar(4, 0, bytes(2))


@NATIVE
def test_julia_width_fixture_values_and_controls():
    fixture = FIXTURES / "julia_numeric_widths.daec"
    provenance = tomllib.loads(fixture.with_suffix(".toml").read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    with open_dataecon(fixture) as db:
        results = [(db.read_scalar(name), value, group_of(name)) for name, value in CASES]
        # Native no-attribute encodings written directly through the C ABI.
        native = {name: db.read_scalar(name) for name in NATIVE_CASES}
        for name in UNRESOLVED_CONTROLS:
            with pytest.raises(ValueError, match="width"):
                db.read_scalar(name)
        for name, expected in BOOL_CONTROLS.items():
            loaded = db.read_scalar(name)
            assert type(loaded) is np.int8
            assert loaded == expected
        with pytest.raises(TypeError, match="reconstruction"):
            db.read_scalar("wctl_marker_true")
        with pytest.raises(TypeError, match="reconstruction"):
            db.read_scalar("wctl_complex_int")
    for actual, expected, group in results:
        assert_same(actual, expected, group)
    for name, expected in NATIVE_CASES.items():
        assert_same(native[name], expected, group_of(name))
    with closing(sqlite3.connect(fixture)) as conn:
        for name, value in CASES:
            kind, width = KINDS[group_of(name)]
            row = conn.execute(
                "SELECT o.class,o.type,s.frequency,s.value FROM objects o JOIN scalars s "
                "USING(id) WHERE o.name=?",
                (name,),
            ).fetchone()
            assert row == (1, kind, 0, expected_bytes(value)), name
            assert len(row[3]) == width
        assert conn.execute(
            "SELECT o.name FROM attributes a JOIN objects o USING(id) WHERE a.id!=0 ORDER BY o.name"
        ).fetchall() == [("wctl_complex_int",), ("wctl_marker_true",)]
        assert conn.execute(
            "SELECT o.type,length(s.value) FROM objects o JOIN scalars s USING(id) "
            "WHERE o.name IN ('wctl_bool_true','wctl_int8_one')"
        ).fetchall() == [(1, 1), (1, 1)]
    native_module = importlib.import_module("tsecon.dataecon._native")
    layout = provenance["layout"]["scalar_t"]
    assert native_module.abi_layout()["scalar_t"] == (layout["size"], tuple(layout["offsets"]))


# Native (C-ABI) objects in the fixture and their expected exact values.
NATIVE_CASES = {
    "wn_f16_tenth": np.float16(0.1),
    "wn_f16_nan": np.float16(NAN),
    "wn_f32_tenth": np.float32(0.1),
    "wn_f32_negative_zero": np.float32(-0.0),
    "wn_i8_min": np.int8(-128),
    "wn_i16_max": np.int16(32767),
    "wn_i32_min": np.int32(-(2**31)),
    "wn_u8_max": np.uint8(255),
    "wn_u16_max": np.uint16(65535),
    "wn_u32_max": np.uint32(2**32 - 1),
    "wn_u64_max": np.uint64(2**64 - 1),
    "wn_u64_high_bit": np.uint64(2**63),
    "wn_c32_plain": np.complex64(1.5 - 2.25j),
    "wn_c64_plain": complex(8.0, 3.0),
    "wn_c64_negative_zero": complex(-0.0, -0.0),
}


@NATIVE
def test_width_roundtrip_storage_and_shared_namespace(tmp_path):
    path = tmp_path / "widths.daec"
    with open_dataecon(path, "a") as db:
        for name, value in CASES:
            db.write_scalar(name, value)
        db.write_scalar("py_complex", 8 + 3j)
        db.write_scalar("count", 7)
        db.write_scalar("rate", 1.25)
        results = [(db.read_scalar(name), value, group_of(name)) for name, value in CASES]
        assert type(db.read_scalar("py_complex")) is complex
        assert type(db.read_scalar("count")) is int
        assert type(db.read_scalar("rate")) is float
        with pytest.raises(DataEconError) as caught:
            db.write_scalar("w_i8_seven", np.int16(7))
        assert caught.value.code == -985
        with pytest.raises(DataEconError):
            db.read_series("w_u64_max")
        with pytest.raises(TypeError):
            db.write_scalar("flag", True)
        with pytest.raises(DataEconError) as caught:
            db.read_scalar("flag")
        assert caught.value.code == -989
    for actual, expected, group in results:
        assert_same(actual, expected, group)
    with pytest.raises(ValueError, match="closed"):
        db.write_scalar("later", np.int8(1))
    with open_dataecon(path) as db:
        with pytest.raises(ValueError, match="read-only"):
            db.write_scalar("readonly", np.uint8(1))
        assert_same(db.read_scalar("w_f16_tenth"), np.float16(0.1), "f16")
    with closing(sqlite3.connect(path)) as conn:
        for name, value in CASES:
            kind, _ = KINDS[group_of(name)]
            assert conn.execute(
                "SELECT o.class,o.type,s.frequency,s.value FROM objects o JOIN scalars s "
                "USING(id) WHERE o.name=?",
                (name,),
            ).fetchone() == (1, kind, 0, value.tobytes())
        assert conn.execute(
            "SELECT o.type,s.value FROM objects o JOIN scalars s USING(id) "
            "WHERE o.name='py_complex'"
        ).fetchone() == (5, struct.pack("<dd", 8.0, 3.0))
        assert conn.execute("SELECT count(*) FROM attributes WHERE id!=0").fetchone() == (0,)


@NATIVE
@pytest.mark.parametrize(
    ("kind", "payload", "error"),
    [
        (7, bytes(8), TypeError),
        (2, bytes(16), ValueError),
        (1, bytes(3), ValueError),
        (4, bytes(1), ValueError),
        (5, bytes(4), ValueError),
        (5, b"", ValueError),
    ],
)
def test_width_backend_validates_kind_and_width_before_storage(tmp_path, kind, payload, error):
    with open_dataecon(tmp_path / "invalid.daec", "a") as db:
        with pytest.raises(error):
            db._handle.write_scalar("bad", kind, 0, payload)
        with pytest.raises(DataEconError) as caught:
            db.read_scalar("bad")
        assert caught.value.code == -989
        db.write_scalar("good", np.uint16(65535))
        assert_same(db.read_scalar("good"), np.uint16(65535), "u16")


@NATIVE
@pytest.mark.parametrize(
    ("name", "sql", "error"),
    [
        ("w_i8_seven", "UPDATE scalars SET value=NULL", ValueError),
        ("w_i8_seven", "UPDATE scalars SET value=zeroblob(3)", ValueError),
        ("w_i8_seven", "UPDATE scalars SET value=zeroblob(16)", ValueError),
        ("w_u32_seven", "UPDATE scalars SET frequency=32", TypeError),
        ("w_f32_tenth", "UPDATE scalars SET frequency=11", TypeError),
        ("w_c32_plain", "UPDATE scalars SET value=zeroblob(4)", ValueError),
        ("w_c64_plain", "UPDATE objects SET type=7", TypeError),
        ("w_u64_max", "UPDATE objects SET type=3", TypeError),
    ],
)
def test_width_malformed_storage(tmp_path, name, sql, error):
    path = tmp_path / "malformed.daec"
    shutil.copyfile(FIXTURES / "julia_numeric_widths.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(f"{sql} WHERE id=(SELECT id FROM objects WHERE name=?)", (name,))
    with open_dataecon(path, "a") as db:
        with pytest.raises(error):
            db.read_scalar(name)
        assert_same(db.read_scalar("w_i16_min"), np.int16(-32768), "i16")
        db.write_scalar("good", np.float16(-0.0))
        assert_same(db.read_scalar("good"), np.float16(-0.0), "f16")


@NATIVE
@pytest.mark.parametrize("key", ["jtype", "jeltype"])
@pytest.mark.parametrize("value", ["Float16", "UInt8", "Bool", "error(123)", None])
def test_width_rejects_all_reconstruction_attributes(tmp_path, key, value):
    path = tmp_path / "attribute.daec"
    shutil.copyfile(FIXTURES / "julia_numeric_widths.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "INSERT INTO attributes SELECT id,?,? FROM objects WHERE name='w_u8_seven'",
            (key, value),
        )
    with open_dataecon(path) as db:
        with pytest.raises(TypeError, match="reconstruction"):
            db.read_scalar("w_u8_seven")
        assert_same(db.read_scalar("w_u8_max"), np.uint8(255), "u8")
