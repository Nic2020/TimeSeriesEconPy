"""StoredScalar: the stored form, its constructors, accessors and finite interpretation.

The pure-Python contract is checked without the native extension: representation
validation, the constructors that build Julia's own storage (checking that the
pinned loader would rebuild the requested value), the family accessors, and the
interpretation table against every row of ``julia_scalar_routes.toml`` (754
payload/marker pairs loaded by the pinned Julia). Native tests then cover the
write/read routes, marker residue on a failed attribute write, overwrite
preservation, owned results after closure and Workspace dispatch.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import math
import sqlite3
import struct
import subprocess
import sys
import tomllib
from contextlib import closing
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from tsecon import MIT, Duration, Monthly, Quarterly, Workspace
from tsecon.dataecon import (
    DataEconError,
    IntegerComplex,
    StoredScalar,
    open_dataecon,
    open_dataecon_memory,
)
from tsecon.dataecon import _exact as ex
from tsecon.dataecon._codec import decode_scalar, encode_marked_scalar, encode_scalar
from tsecon.dataecon._scalars import SPELLINGS, is_stored_form
from tsecon.dataecon._workspace import classify

NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built",
)
FIXTURES = Path(__file__).parent / "fixtures"
ROUTES = tomllib.loads((FIXTURES / "julia_scalar_routes.toml").read_text(encoding="utf-8"))
F64 = ex.float64_bits


def f64(value: float) -> StoredScalar:
    return StoredScalar(F64(value), 4)


def i64(value: int, marker: str | None = None) -> StoredScalar:
    return StoredScalar(struct.pack("<q", value), 1, 0, marker)


# ---- representation ---------------------------------------------------------


def test_fields_validation_equality_and_repr():
    stored = StoredScalar(F64(0.5), 4, 0, "Rational{Int64}")
    assert stored == StoredScalar(F64(0.5), 4, 0, "Rational{Int64}")
    assert stored != StoredScalar(F64(0.5), 4)
    assert stored != StoredScalar(F64(0.5), 4, 0, "Rational{Int32}")
    assert hash(stored) == hash(StoredScalar(F64(0.5), 4, 0, "Rational{Int64}"))
    assert repr(stored) == "StoredScalar(Float64, 000000000000e03f, marker='Rational{Int64}')"
    assert repr(StoredScalar(b"x" * 40 + b"\0", 6)) == "StoredScalar(String, 41 bytes)"
    assert (stored.nbytes, stored.is_text, stored.is_wide, stored.active_marker) == (
        8,
        False,
        False,
        "Rational{Int64}",
    )
    with pytest.raises(AttributeError):
        stored.marker = None  # type: ignore[misc]
    with pytest.raises(TypeError, match="bytes"):
        StoredScalar(bytearray(8), 4)
    with pytest.raises(TypeError, match="native integer codes"):
        StoredScalar(bytes(8), 4.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="width"):
        StoredScalar(bytes(3), 4)
    with pytest.raises(TypeError):
        StoredScalar(bytes(8), 7)
    with pytest.raises(ValueError, match="NUL"):
        StoredScalar(bytes(8), 4, 0, "Int\x0064")
    with pytest.raises(TypeError, match="string or None"):
        StoredScalar(bytes(8), 4, 0, b"Int64")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reliable native date range"):
        StoredScalar(struct.pack("<q", 2**40), 3, 32, "Symbol")


def test_element_descriptor_and_julia_name():
    assert StoredScalar(bytes(2), 4).julia_name == "Float16"
    assert StoredScalar(bytes(16), 1).element.julia_name == "Int128"
    assert StoredScalar(bytes(16), 2).julia_name == "UInt128"
    assert StoredScalar(bytes(4), 5).julia_name == "ComplexF16"
    assert StoredScalar(bytes(8), 3, 32).element.kind == "date"
    assert StoredScalar(bytes(8), 3, 32).julia_name == "MIT{Monthly}"
    assert StoredScalar(bytes(8), 1, 67).julia_name == "Duration{Quarterly{3}}"
    assert StoredScalar(b"a\0", 6).element is None
    assert StoredScalar(b"a\0", 6).julia_name == "String"


@pytest.mark.parametrize(
    ("kind", "payload", "marker", "stored"),
    [
        (4, bytes(8), None, False),
        (4, bytes(8), "Float64", True),
        (1, bytes(16), None, True),
        (2, bytes(16), None, True),
        (5, bytes(4), None, True),
        (5, bytes(8), None, False),
        (6, b"ok\0", None, False),
        (6, b"ok", None, True),
        (6, b"o\0k\0", None, True),
        (6, b"\xff\0", None, True),
        (6, b"ok\0", "Symbol", True),
    ],
)
def test_decode_scalar_returns_the_stored_form_exactly_when_needed(kind, payload, marker, stored):
    value = decode_scalar(kind, 0, payload, marker)
    assert is_stored_form(kind, payload, marker) is stored
    assert (type(value) is StoredScalar) is stored
    if stored:
        assert value == StoredScalar(payload, kind, 0, marker)


# ---- constructors: Julia's own storage, checked against its reconstruction -----


def test_wide_integer_constructors():
    assert StoredScalar.int128(-1) == StoredScalar(b"\xff" * 16, 1)
    assert StoredScalar.int128(2**127 - 1).to_int() == 2**127 - 1
    assert StoredScalar.int128(-(2**127)).to_interpreted() == -(2**127)
    assert StoredScalar.uint128(2**128 - 1).to_interpreted() == 2**128 - 1
    with pytest.raises(ValueError, match="Int128"):
        StoredScalar.int128(2**127)
    with pytest.raises(ValueError, match="UInt128"):
        StoredScalar.uint128(-1)
    with pytest.raises(ValueError):
        StoredScalar.int128(True)  # bool is not an exact int
    with pytest.raises(ValueError):
        StoredScalar.uint128(np.int64(1))


def test_complexf16_constructor_keeps_bits():
    plain = StoredScalar.complexf16(complex(1.5, -2.25))
    assert plain.payload == np.float16(1.5).tobytes() + np.float16(-2.25).tobytes()
    assert plain.to_interpreted() == complex(1.5, -2.25)
    assert type(plain.to_complex64()) is np.complex64
    negzero = StoredScalar.complexf16(complex(-0.0, 0.0))
    assert negzero.payload == b"\x00\x80\x00\x00"
    assert math.copysign(1.0, negzero.to_complex().real) == -1.0
    assert math.copysign(1.0, float(negzero.to_complex64().real)) == -1.0
    special = StoredScalar.complexf16(complex(math.nan, math.inf))
    assert math.isnan(special.to_complex().real)
    assert special.to_complex().imag == math.inf
    wide = special.to_complex64()
    assert math.isnan(float(wide.real))
    assert float(wide.imag) == math.inf
    pair = StoredScalar.complexf16((np.float16(65504.0), np.float16(-0.0)))
    assert pair.to_complex() == complex(65504.0, 0.0)
    with pytest.raises(ValueError, match="float16"):
        StoredScalar.complexf16(complex(0.1, 0))
    with pytest.raises(ValueError, match="float16"):
        StoredScalar.complexf16(complex(70000.0, 0))
    with pytest.raises(TypeError):
        StoredScalar.complexf16((1.5, 2.5))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        StoredScalar.complexf16(np.complex64(1))  # type: ignore[arg-type]


def test_text_constructors():
    assert StoredScalar.symbol("gdp") == StoredScalar(b"gdp\0", 6, 0, "Symbol")
    assert StoredScalar.symbol("").to_interpreted() == ""
    assert StoredScalar.symbol("é🙂").to_interpreted() == "é🙂"
    assert StoredScalar.text("abc", "SubString{String}").to_interpreted() == "abc"
    assert StoredScalar.text("abc").to_interpreted() == "abc"
    raw = StoredScalar.raw_text(b"f\xffo")
    assert raw == StoredScalar(b"f\xffo\0", 6)
    with pytest.raises(ValueError, match="UTF-8"):
        raw.to_str()
    with pytest.raises(ValueError, match="allow_nul"):
        StoredScalar.raw_text(b"a\0b")
    assert StoredScalar.raw_text(b"a\0b", allow_nul=True).payload == b"a\0b\0"
    assert StoredScalar.raw_text(b"", marker="Symbol") == StoredScalar(b"\0", 6, 0, "Symbol")
    with pytest.raises(ValueError, match="NUL"):
        StoredScalar.symbol("a\0b")
    with pytest.raises(ValueError, match="surrogates"):
        StoredScalar.symbol("\ud800")
    with pytest.raises(TypeError):
        StoredScalar.symbol(b"x")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        StoredScalar.raw_text("x")  # type: ignore[arg-type]


def test_fraction_constructor_requires_julia_to_rebuild_the_fraction():
    half = StoredScalar.fraction(Fraction(1, 2))
    assert half == StoredScalar(F64(0.5), 4, 0, "Rational{Int64}")
    assert half.to_fraction() == Fraction(1, 2)
    # Exact even though 2**54 is beyond 2**53 (Julia reloads it), and refused
    # for 2**-100 even though the float is exact (the denominator exceeds Int64).
    assert StoredScalar.fraction(Fraction(2**54)).to_fraction() == Fraction(2**54)
    with pytest.raises(ValueError, match="Julia raises InexactError"):
        StoredScalar.fraction(Fraction(1, 2**100))
    with pytest.raises(ValueError, match="exact=False"):
        StoredScalar.fraction(Fraction(1, 3))
    lossy = StoredScalar.fraction(Fraction(1, 3), exact=False)
    assert lossy.payload == F64(1 / 3)
    assert lossy.to_fraction() == Fraction(6004799503160661, 18014398509481984)
    assert StoredScalar.fraction(Fraction(1, 3), parameter="Int32").to_fraction() == Fraction(1, 3)
    assert StoredScalar.fraction(Fraction(3, 2**62), exact=False).to_fraction() == Fraction(
        3, 4611686018427387649
    )  # Julia's own reconstruction above 2**53 partial quotients
    with pytest.raises(ValueError, match="OverflowError"):
        StoredScalar.fraction(Fraction(-1, 2), parameter="UInt8")
    with pytest.raises(ValueError, match="InexactError"):
        StoredScalar.fraction(Fraction(1000), parameter="Int8")
    with pytest.raises(ValueError, match="InexactError"):
        StoredScalar.fraction(Fraction(1000), parameter="Int8", exact=False)
    # Julia's writer stores float(num)/float(den), not the correctly rounded quotient.
    assert StoredScalar.fraction(Fraction(2**62, 3), exact=False).payload == F64(float(2**62) / 3.0)
    with pytest.raises(TypeError, match="Fraction"):
        StoredScalar.fraction(0.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="parameter"):
        StoredScalar.fraction(Fraction(1, 2), parameter="Bool")


def test_date_and_datetime_constructors():
    day = StoredScalar.date(dt.date(2024, 3, 15))
    assert day == StoredScalar(F64(1710460800.0), 4, 0, "Date")
    assert day.to_date() == dt.date(2024, 3, 15)
    assert day.to_datetime64() == np.datetime64("2024-03-15")
    stamp = StoredScalar.datetime(dt.datetime(2024, 3, 15, 13, 45, 30, 123000))
    assert stamp == StoredScalar(F64(1710510330.123), 4, 0, "DateTime")
    assert stamp.to_datetime() == dt.datetime(2024, 3, 15, 13, 45, 30, 123000)
    assert stamp.to_datetime64() == np.datetime64("2024-03-15T13:45:30.123")
    assert StoredScalar.date(dt.date(1, 1, 1)).to_calendar() == (1, 1, 1)
    assert StoredScalar.datetime(dt.datetime(9999, 12, 31, 23, 59, 59, 999000)).to_calendar() == (
        9999,
        12,
        31,
        23,
        59,
        59,
        999,
    )
    with pytest.raises(TypeError, match="not a datetime"):
        StoredScalar.date(dt.datetime(2024, 3, 15))
    with pytest.raises(TypeError, match="time zone"):
        StoredScalar.datetime(dt.datetime(2024, 3, 15, tzinfo=dt.UTC))
    with pytest.raises(ValueError, match="millisecond"):
        StoredScalar.datetime(dt.datetime(2024, 3, 15, 0, 0, 0, 1))
    with pytest.raises(TypeError):
        StoredScalar.datetime(dt.date(2024, 3, 15))  # type: ignore[arg-type]
    # datetime.date/datetime inputs are routed by exact type in the codec.
    assert encode_marked_scalar(dt.date(2024, 3, 15)) == (4, 0, F64(1710460800.0), "Date")
    assert encode_marked_scalar(dt.datetime(2024, 3, 15)) == (4, 0, F64(1710460800.0), "DateTime")


def test_calendar_constructors_cover_every_proleptic_year():
    assert StoredScalar.calendar_date(0, 1, 1).to_calendar() == (0, 1, 1)
    assert StoredScalar.calendar_date(-1000, 6, 30).to_datetime64() == np.datetime64("-1000-06-30")
    assert StoredScalar.calendar_date(300000, 1, 1).to_datetime64() == np.datetime64("300000-01-01")
    assert StoredScalar.calendar_datetime(10000, 1, 1, 0, 0, 0, 7).to_calendar() == (
        10000,
        1,
        1,
        0,
        0,
        0,
        7,
    )
    with pytest.raises(ValueError, match="outside Python's datetime range"):
        StoredScalar.calendar_date(10000, 1, 1).to_interpreted()
    # Beyond 2**53 milliseconds Julia's Float64 storage loses the millisecond:
    # the write is refused rather than stored with a different reload.
    with pytest.raises(ValueError, match="would not reload"):
        StoredScalar.calendar_datetime(300000, 6, 1, 12, 0, 0, 5)
    assert StoredScalar.calendar_datetime(300000, 6, 1, 12, 0, 0, 4).to_calendar() == (
        300000,
        6,
        1,
        12,
        0,
        0,
        4,
    )
    with pytest.raises(ValueError, match="Month"):
        StoredScalar.calendar_date(2024, 13, 1)
    with pytest.raises(ValueError, match="Day"):
        StoredScalar.calendar_date(2023, 2, 29)
    with pytest.raises(ValueError, match="Time of day"):
        StoredScalar.calendar_datetime(2024, 1, 1, 24)
    with pytest.raises(TypeError, match="exact Python int"):
        StoredScalar.calendar_date(2024.0, 1, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Int64 millisecond range"):
        StoredScalar.calendar_date(300_000_000_000_000, 1, 1)


def test_integer_complex_constructor_and_pair():
    pair = IntegerComplex(2**54, 1)
    assert pair.real == 2**54
    assert StoredScalar.integer_complex(pair) == StoredScalar(
        struct.pack("<dd", 2.0**54, 1.0), 5, 0, "Complex{Int64}"
    )
    assert StoredScalar.integer_complex(2**54, 1).to_integer_complex() == pair
    assert StoredScalar.integer_complex(127, -128, parameter="Int8").marker == "Complex{Int8}"
    assert StoredScalar.integer_complex(1, 0, parameter="Bool").to_interpreted() == IntegerComplex(
        1
    )
    with pytest.raises(ValueError, match="not exactly representable in Float64"):
        StoredScalar.integer_complex(2**53 + 1, 0)
    with pytest.raises(ValueError, match="InexactError"):
        StoredScalar.integer_complex(128, 0, parameter="Int8")
    with pytest.raises(ValueError, match="InexactError"):
        StoredScalar.integer_complex(0, 2, parameter="Bool")
    with pytest.raises(ValueError, match="Float64 range"):
        StoredScalar.integer_complex(10**400, 0, parameter="Int128")
    with pytest.raises(TypeError, match="exact Python int"):
        StoredScalar.integer_complex(1.0, 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="not both"):
        StoredScalar.integer_complex(pair, 1)
    with pytest.raises(TypeError, match="parameter"):
        StoredScalar.integer_complex(1, 0, parameter="Float64")
    assert complex(IntegerComplex(3, -4)) == complex(3, -4)
    assert IntegerComplex(3).to_complex() == 3 + 0j
    with pytest.raises(ValueError, match="imaginary part"):
        IntegerComplex(0, 2**53 + 1).to_complex()
    with pytest.raises(TypeError):
        IntegerComplex(1.0, 0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        IntegerComplex(True, 0)
    assert encode_marked_scalar(IntegerComplex(2, 3)) == (
        5,
        0,
        struct.pack("<dd", 2.0, 3.0),
        "Complex{Int64}",
    )


# ---- accessors are per family ----------------------------------------------


def test_accessors_refuse_other_families():
    text = StoredScalar(b"ab\0", 6)
    number = f64(1.5)
    with pytest.raises(TypeError, match="to_int"):
        text.to_int()
    with pytest.raises(TypeError, match="to_float"):
        text.to_float()
    with pytest.raises(TypeError, match="to_complex"):
        number.to_complex()
    with pytest.raises(TypeError, match="to_complex64"):
        StoredScalar(bytes(16), 5).to_complex64()
    with pytest.raises(TypeError, match="to_str"):
        number.to_str()
    with pytest.raises(TypeError, match="Rational"):
        number.to_fraction()
    with pytest.raises(TypeError, match="Complex"):
        number.to_integer_complex()
    with pytest.raises(TypeError, match="Date"):
        number.to_calendar()
    with pytest.raises(TypeError, match="Date"):
        number.to_datetime64()
    with pytest.raises(TypeError, match="use to_datetime"):
        StoredScalar(F64(0.0), 4, 0, "DateTime").to_date()
    with pytest.raises(TypeError, match="use to_date"):
        StoredScalar(F64(0.0), 4, 0, "Date").to_datetime()
    assert text.to_bytes() == b"ab\0"
    assert StoredScalar(struct.pack("<q", 24288), 3, 32).to_int() == 24288
    assert StoredScalar(struct.pack("<q", -5), 1, 32).to_int() == -5
    assert StoredScalar(np.float16(-0.0).tobytes(), 4).to_float() == 0.0
    assert math.copysign(1.0, StoredScalar(np.float16(-0.0).tobytes(), 4).to_float()) == -1.0
    assert StoredScalar(np.complex64(1 - 2j).tobytes(), 5).to_complex() == 1 - 2j


def test_stored_value_of_unmarked_and_any():
    assert StoredScalar(b"ok\0", 6, 0, "Any").to_interpreted() == "ok"
    assert StoredScalar(struct.pack("<q", 24288), 3, 32, "Any").to_interpreted() == MIT(
        Monthly(), 24288
    )
    assert StoredScalar(struct.pack("<q", 2), 1, 67, "Any").to_interpreted() == Duration(
        Quarterly(3), 2
    )
    assert StoredScalar(np.int8(-3).tobytes(), 1, 0, "Any").to_interpreted() == np.int8(-3)
    assert type(StoredScalar(np.int8(-3).tobytes(), 1, 0, "Any").to_interpreted()) is np.int8
    assert StoredScalar(bytes(16), 2).to_interpreted() == 0


# ---- the finite interpretation table against every probed route --------------


def canonical(value: object) -> str:  # noqa: PLR0911 - finite canonical text table
    """Julia's canonical text of a Python result (the probe's ``value`` convention)."""
    if type(value) is Fraction:
        return f"{value.numerator}//{value.denominator}"
    if type(value) is IntegerComplex:
        return f"{value.real},{value.imag}"
    if type(value) is dt.datetime:
        return (
            f"{value.year}-{value.month}-{value.day}T{value.hour}:{value.minute}:"
            f"{value.second}.{value.microsecond // 1000}"
        )
    if type(value) is dt.date:
        return f"{value.year}-{value.month}-{value.day}"
    if type(value) is float:
        return struct.pack("<d", value).hex()
    if type(value) is complex:
        return struct.pack("<dd", value.real, value.imag).hex()
    if isinstance(value, (np.floating, np.complexfloating)):
        return value.tobytes().hex()
    if type(value) is bool:
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if type(value) is MIT:
        return str(value)
    if type(value) is Duration:
        return str(value.value)
    assert type(value) is str
    return value


def julia_type_of(value: object, token: str) -> str | None:
    """Julia's loaded type name for a Python result where it can be named."""
    if type(value) is Fraction or type(value) is IntegerComplex:
        return None  # the parameter is checked through the canonical text's row type
    if type(value) is float:
        return "Float64"
    if type(value) is complex:
        return None if token in ("Complex{Float16}", "Complex") else "ComplexF64"
    if isinstance(value, np.generic):
        return {
            np.float16: "Float16",
            np.float32: "Float32",
            np.complex64: "ComplexF32",
            np.int8: "Int8",
            np.int16: "Int16",
            np.int32: "Int32",
            np.uint8: "UInt8",
            np.uint16: "UInt16",
            np.uint32: "UInt32",
            np.uint64: "UInt64",
        }[type(value)]
    return None


# Routes the reader represents differently by design (Julia's loaded value is
# still recorded in the fixture): each maps a predicate over (payload label,
# token) to the exception class and message the reader raises instead.
def _narrow_float(payload: str) -> bool:
    return payload.startswith(("f16", "f32", "c16", "c32"))


DESIGNED = [
    # Julia prints the value for a Symbol; the reader reproduces only integer text.
    (
        lambda p, t: t == "Symbol" and p.startswith(("f", "c")),
        TypeError,
        "printed form",
    ),
    # Rationalizing in Float16/Float32 arithmetic is not transcribed.
    (
        lambda p, t: t in ("Rational", "Rational{Int}") and _narrow_float(p),
        TypeError,
        "narrower float width",
    ),
    # Date/DateTime on UInt64/128-bit/narrow-float payloads: unverified promotion.
    (
        lambda p, t: (
            t in ("Date", "DateTime", "Dates.DateTime")
            and (p.startswith(("u64", "i128", "u128")) or _narrow_float(p))
        ),
        TypeError,
        "not verified",
    ),
    # Julia wraps the Int64 product of an integer unix time; the reader refuses.
    (lambda p, t: p.startswith("i64_wrap"), ValueError, "overflows Int64"),
    # BigFloat interpretation is deferred (the payload is exact, see to_float/to_int).
    (lambda p, t: t == "BigFloat", TypeError, "deferred"),
    # +/-1//0 has no Fraction; to_float() gives the signed infinity.
    (
        lambda p, t: p == "f64_inf" and t in ("Rational", "Rational{Int}"),
        ValueError,
        "no Fraction can hold",
    ),
    # A monthly code beyond the reliable native window (Julia builds the MIT).
    (
        lambda p, t: t == "MIT{Monthly}" and p == "i64_2p53p1",
        ValueError,
        "reliable native date range",
    ),
]
# Julia's loaded value is a ComplexF16 (printed as its two-byte components);
# the reader returns a Python complex whose components widen from Float16.
COMPLEXF16_RESULTS = {"ComplexF16"}


def _designed(payload: str, token: str) -> tuple[type[Exception], str] | None:
    for predicate, error, match in DESIGNED:
        if predicate(payload, token):
            return error, match
    return None


@pytest.mark.parametrize("row", ROUTES["rows"], ids=[r["name"] for r in ROUTES["rows"]])
def test_every_probed_route(row: dict) -> None:
    stored = StoredScalar(bytes.fromhex(row["hex"]), row["kind"], 0, row["token"])
    designed = _designed(row["payload"], row["token"])
    if designed is not None:
        error, match = designed
        with pytest.raises(error, match=match):
            stored.to_interpreted()
        return
    if "error" in row:
        # Pin value failures separately from unavailable conversion routes.
        error = {"InexactError": ValueError, "OverflowError": ValueError, "MethodError": TypeError}[
            row["error"]
        ]
        with pytest.raises(error):
            stored.to_interpreted()
        return
    if row["type"] in ("Date", "DateTime"):
        # Years beyond datetime's range are exact through to_calendar().
        parts = stored.to_calendar()
        text = "{}-{}-{}".format(*parts[:3])
        if row["type"] == "DateTime":
            text += "T{}:{}:{}.{}".format(*parts[3:])
        assert text == row["value"], row["name"]
        if 1 <= parts[0] <= 9999:
            assert canonical(stored.to_interpreted()) == row["value"]
        else:
            with pytest.raises(ValueError, match="outside Python's datetime range"):
                stored.to_interpreted()
        return
    value = stored.to_interpreted()
    if row["type"] in COMPLEXF16_RESULTS:
        assert type(value) is complex
        bits = np.float16(value.real).tobytes() + np.float16(value.imag).tobytes()
        assert bits.hex() == row["value"]
        return
    assert canonical(value) == row["value"], row["name"]
    named = julia_type_of(value, row["token"])
    if named is not None:
        assert named == row["type"], row["name"]
    if type(value) is Fraction:
        assert row["type"].startswith("Rational{")
    if type(value) is IntegerComplex:
        assert row["type"].startswith("Complex{")


def test_probe_provenance_and_coverage():
    assert ROUTES["pin"] == "fc0a0d01aed4903ea6c64b12d78d0bdf68468df6"
    rows = ROUTES["rows"]
    assert len(rows) == 754
    assert sum("error" in r for r in rows) == 217
    tokens = {r["token"] for r in rows}
    assert {"Rational", "Complex", "Any", "Real", "Integer", "Signed", "Unsigned"} <= tokens
    assert {"AbstractFloat", "BigInt", "Symbol", "Date", "DateTime", "Bool"} <= tokens
    assert {"Base.Float64", "Core.Int64", "TimeSeriesEcon.Int64", "Float64 "} <= tokens


@pytest.mark.parametrize(("width", "bits"), [(2, 0x7C01), (4, 0x7F800001), (8, 0x7FF0000000000001)])
@pytest.mark.parametrize("marker", ["Real", "AbstractFloat"])
def test_complex_real_projection_keeps_signaling_nan_bits(width, bits, marker):
    real = bits.to_bytes(width, "little")
    scalar = StoredScalar(real + b"\0" * width, 5, marker=marker)
    projected = scalar.to_interpreted()
    actual = struct.pack("<d", projected) if width == 8 else projected.tobytes()
    assert actual == real
    assert scalar.payload == real + b"\0" * width


@pytest.mark.parametrize("exact", [True, False])
def test_fraction_component_overflow_is_a_value_error(exact):
    value = Fraction(10**400 + 1, 10**400)
    with pytest.raises(ValueError, match="Fraction components exceed"):
        StoredScalar.fraction(value, exact=exact)


@NATIVE
def test_workspace_reports_fraction_overflow_and_preserves_existing_object():
    with open_dataecon_memory() as db:
        db.write_scalar("bad", 7)
        report = db.write_workspace(
            Workspace(bad=Fraction(10**400 + 1, 10**400), tail=2), overwrite=True
        )
        assert report.count == 1
        assert [(item.path, item.category) for item in report.skipped] == [("/bad", "invalid")]
        assert db.read_scalar("bad") == 7
        assert db.read_scalar("tail") == 2


def test_spellings_are_a_finite_verified_table():
    # Every alternative spelling maps to an exact token; the literal text is
    # preserved on the StoredScalar and only interpretation uses the table.
    assert SPELLINGS["Dates.Date"] == "Date"
    assert SPELLINGS[" Int64 "] == "Int64"
    assert i64(2, "Base.Int64").to_interpreted() == 2
    assert i64(2, "Base.Int64").marker == "Base.Int64"
    assert f64(2.0).to_interpreted() == 2.0
    assert StoredScalar(F64(2.0), 4, 0, "Float64 ").to_interpreted() == 2.0
    with pytest.raises(TypeError, match="no supported interpretation"):
        StoredScalar(F64(2.0), 4, 0, "Base.Float32").to_interpreted()
    with pytest.raises(TypeError, match="no supported interpretation"):
        StoredScalar(F64(2.0), 4, 0, "Int64\t").to_interpreted()


def test_unsupported_markers_are_opaque_and_never_echoed():
    for marker in ("NoSuchType", 'error("evaluated")', "Irrational{:π}", "Vector{Int64}", ""):
        stored = f64(1.5)
        stored = StoredScalar(stored.payload, 4, 0, marker)
        assert stored.marker == marker
        with pytest.raises(TypeError) as info:
            stored.to_interpreted()
        assert marker not in str(info.value) or marker == ""
        assert "preserved" in str(info.value)


def test_rational_routes_follow_julia_exactly():
    third = StoredScalar(F64(1 / 3), 4, 0, "Rational{Int64}")
    assert third.to_fraction() == Fraction(6004799503160661, 18014398509481984)
    assert StoredScalar(F64(1 / 3), 4, 0, "Rational{Int32}").to_fraction() == Fraction(1, 3)
    assert StoredScalar(F64(3 * 2.0**-62), 4, 0, "Rational{Int64}").to_fraction() == Fraction(
        3, 4611686018427387649
    )
    assert StoredScalar(F64(-(2.0**63)), 4, 0, "Rational{Int64}").to_float() == -(2.0**63)
    with pytest.raises(ValueError, match="InexactError"):
        StoredScalar(F64(-(2.0**63)), 4, 0, "Rational{Int64}").to_fraction()
    with pytest.raises(ValueError, match="InexactError"):
        StoredScalar(F64(math.nan), 4, 0, "Rational{Int64}").to_fraction()
    with pytest.raises(ValueError, match="OverflowError"):
        StoredScalar(F64(-0.5), 4, 0, "Rational{UInt8}").to_fraction()
    inf = StoredScalar(F64(math.inf), 4, 0, "Rational{Int64}")
    with pytest.raises(ValueError, match="to_float"):
        inf.to_fraction()
    assert inf.to_float() == math.inf
    assert StoredScalar(F64(-math.inf), 4, 0, "Rational{Int8}").to_float() == -math.inf
    # typemin loads from an integer payload (Julia's own rewrite of it then fails).
    assert StoredScalar(b"\x80", 1, 0, "Rational{Int8}").to_fraction() == Fraction(-128)
    assert i64(-128, "Rational{Int8}").to_fraction() == Fraction(-128)
    with pytest.raises(ValueError, match="InexactError"):
        StoredScalar(F64(-128.0), 4, 0, "Rational{Int8}").to_fraction()
    with pytest.raises(TypeError, match="no Rational"):
        StoredScalar(b"1//2\0", 6, 0, "Rational{Int64}").to_fraction()
    with pytest.raises(TypeError, match="no Rational"):
        StoredScalar(struct.pack("<q", 3), 3, 32, "Rational{Int64}").to_fraction()
    complex_payload = StoredScalar(struct.pack("<dd", 2.0, 0.0), 5, 0, "Rational{Int64}")
    assert complex_payload.to_fraction() == Fraction(2)
    with pytest.raises(ValueError, match="InexactError"):
        StoredScalar(struct.pack("<dd", 2.0, 1.0), 5, 0, "Rational{Int64}").to_fraction()


def test_integer_complex_routes_keep_every_bit():
    assert i64(2**53 + 1, "Complex{Int64}").to_integer_complex() == IntegerComplex(2**53 + 1)
    wide = StoredScalar((2**70).to_bytes(16, "little"), 1, 0, "Complex{Int128}")
    assert wide.to_interpreted() == IntegerComplex(2**70)
    with pytest.raises(ValueError, match="InexactError"):
        StoredScalar((2**70).to_bytes(16, "little"), 1, 0, "Complex{Int64}").to_interpreted()
    both = StoredScalar(struct.pack("<dd", 2.0**54, 1.0), 5, 0, "Complex{Int64}")
    assert both.to_interpreted() == IntegerComplex(2**54, 1)
    with pytest.raises(ValueError, match="InexactError"):
        StoredScalar(struct.pack("<dd", 1.5, 0.0), 5, 0, "Complex{Int64}").to_interpreted()
    with pytest.raises(ValueError, match="InexactError"):
        StoredScalar(struct.pack("<dd", 2.0, 0.0), 5, 0, "Complex{Bool}").to_interpreted()
    assert StoredScalar(np.complex64(2 + 0j).tobytes(), 5, 0, "Complex{Int8}").to_interpreted() == (
        IntegerComplex(2)
    )
    with pytest.raises(TypeError, match="no Complex"):
        StoredScalar(b"1+2im\0", 6, 0, "Complex{Int64}").to_interpreted()
    assert IntegerComplex(2**53 + 1, 0) != IntegerComplex(2**53, 0)


def test_date_routes():
    assert i64(1710460800, "Date").to_interpreted() == dt.date(2024, 3, 15)
    assert StoredScalar(b"\x03", 1, 0, "DateTime").to_interpreted() == dt.datetime(
        1970, 1, 1, 0, 0, 3
    )
    assert StoredScalar(b"\xfd", 1, 0, "Date").to_interpreted() == dt.date(1969, 12, 31)
    assert StoredScalar(F64(-0.0015), 4, 0, "DateTime").to_calendar() == (
        1969,
        12,
        31,
        23,
        59,
        59,
        999,
    )
    assert StoredScalar(struct.pack("<dd", 1.0, 0.0), 5, 0, "Date").to_interpreted() == dt.date(
        1970, 1, 1
    )
    with pytest.raises(ValueError, match="InexactError"):
        StoredScalar(struct.pack("<dd", 1.0, 2.0), 5, 0, "Date").to_interpreted()
    with pytest.raises(ValueError, match="InexactError"):
        StoredScalar(F64(math.nan), 4, 0, "Date").to_interpreted()
    with pytest.raises(ValueError, match="InexactError"):
        StoredScalar(F64(1e300), 4, 0, "DateTime").to_interpreted()
    with pytest.raises(TypeError, match="no Date"):
        StoredScalar(b"x\0", 6, 0, "Date").to_interpreted()
    # NumPy's NaT sentinel and its Int64 count are refused explicitly.
    nat = StoredScalar(F64((ex.MIN_INT64) / 1000.0), 4, 0, "DateTime")
    with pytest.raises(ValueError):
        nat.to_datetime64()


def test_abstract_routes_are_value_dependent():
    assert StoredScalar(np.uint8(3).tobytes(), 2, 0, "Signed").to_interpreted() == np.int8(3)
    assert type(StoredScalar(np.uint8(3).tobytes(), 2, 0, "Signed").to_interpreted()) is np.int8
    with pytest.raises(ValueError, match="outside the target range"):
        StoredScalar(np.uint64(2**64 - 1).tobytes(), 2, 0, "Signed").to_interpreted()
    with pytest.raises(ValueError, match="outside the target range"):
        StoredScalar(np.int8(-3).tobytes(), 1, 0, "Unsigned").to_interpreted()
    assert f64(2.0).to_interpreted() == 2.0
    assert StoredScalar(F64(2.0), 4, 0, "Integer").to_interpreted() == 2
    assert type(StoredScalar(F64(2.0), 4, 0, "Integer").to_interpreted()) is int
    assert StoredScalar(F64(-0.0), 4, 0, "Unsigned").to_interpreted() == np.uint64(0)
    with pytest.raises(ValueError, match="InexactError"):
        StoredScalar(F64(1.5), 4, 0, "Integer").to_interpreted()
    assert StoredScalar(bytes(15) + b"\x80", 1, 0, "AbstractFloat").to_interpreted() == -(2.0**127)
    assert StoredScalar(
        np.complex64(2 + 0j).tobytes(), 5, 0, "Real"
    ).to_interpreted() == np.float32(2)
    with pytest.raises(ValueError, match="InexactError"):
        StoredScalar(struct.pack("<dd", 1.0, 2.0), 5, 0, "Real").to_interpreted()
    for token in ("Real", "Number", "Integer", "Signed", "Unsigned", "AbstractFloat", "BigInt"):
        with pytest.raises(TypeError, match=r"no .* conversion"):
            StoredScalar(b"1\0", 6, 0, token).to_interpreted()
        with pytest.raises(TypeError, match=r"no .* conversion"):
            StoredScalar(struct.pack("<q", 1), 3, 32, token).to_interpreted()
    assert i64(2**63 - 1, "BigInt").to_interpreted() == 2**63 - 1
    assert StoredScalar(F64(1e300), 4, 0, "BigInt").to_interpreted() == int(1e300)
    with pytest.raises(ValueError, match="InexactError"):
        StoredScalar(F64(1.5), 4, 0, "BigInt").to_interpreted()


def test_element_tokens_reuse_the_series_kernels():
    assert StoredScalar(F64(1.5), 4, 0, "Float32").to_interpreted() == np.float32(1.5)
    assert i64(3, "Float64").to_interpreted() == 3.0
    assert i64(-3, "Int128").to_interpreted() == -3
    assert i64(24288, "MIT{Monthly}").to_interpreted() == MIT(Monthly(), 24288)
    assert i64(5, "Duration{Monthly}").to_interpreted() == Duration(Monthly(), 5)
    assert StoredScalar(b"\x01", 1, 0, "Bool").to_interpreted() is True
    assert StoredScalar(struct.pack("<q", 24288), 3, 32, "Int64").to_interpreted() == 24288
    assert StoredScalar(struct.pack("<q", 24288), 3, 32, "MIT{Monthly}").to_interpreted() == MIT(
        Monthly(), 24288
    )
    with pytest.raises(TypeError, match="no conversion"):
        StoredScalar(struct.pack("<q", 24288), 3, 32, "MIT{Quarterly{3}}").to_interpreted()
    with pytest.raises(ValueError, match="not exactly zero or one"):
        StoredScalar(b"\x02", 1, 0, "Bool").to_interpreted()
    with pytest.raises(ValueError, match="outside the target range"):
        i64(-3, "UInt8").to_interpreted()
    with pytest.raises(TypeError, match="integer sources only"):
        f64(2.0)._convert_element(
            __import__("tsecon.dataecon._interpret").dataecon._interpret.ACTIVE_TOKENS[
                "MIT{Monthly}"
            ]
        )
    assert StoredScalar(F64(2.0), 4, 0, "Complex{Float16}").to_interpreted() == 2 + 0j
    assert StoredScalar(F64(2.0), 4, 0, "Complex{Float32}").to_interpreted() == np.complex64(2)
    assert StoredScalar(F64(2.0), 4, 0, "Complex{Float64}").to_interpreted() == 2 + 0j


def test_codec_dispatch_by_exact_type():
    assert encode_marked_scalar(Fraction(1, 2)) == (4, 0, F64(0.5), "Rational{Int64}")
    stored = StoredScalar.symbol("x")
    assert encode_marked_scalar(stored) == (6, 0, b"x\0", "Symbol")
    assert encode_marked_scalar(1.5) == (4, 0, F64(1.5), None)
    with pytest.raises(ValueError, match="exact=False"):
        encode_marked_scalar(Fraction(1, 3))
    with pytest.raises(TypeError, match="StoredScalar"):
        encode_scalar(Fraction(1, 2))

    class Sub(Fraction):
        pass

    with pytest.raises(TypeError):
        encode_marked_scalar(Sub(1, 2))
    assert classify(Fraction(1, 2)) == "scalar"
    assert classify(dt.date(2024, 1, 1)) == "scalar"
    assert classify(IntegerComplex(1, 2)) == "scalar"
    assert classify(stored) == "scalar"


# ---- native routes ------------------------------------------------------------


def stored_row(path, name):
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute(
            "SELECT o.type, s.frequency, s.value, "
            "(SELECT group_concat(x.name || '=' || x.value, '|') FROM attributes x "
            "WHERE x.id=o.id) "
            "FROM objects o JOIN scalars s ON s.id=o.id WHERE o.name=?",
            (name,),
        ).fetchone()


@NATIVE
def test_write_routes_store_julias_form_and_read_back(tmp_path):
    path = tmp_path / "routes.daec"
    inputs = {
        "half": (Fraction(1, 2), (4, 0, F64(0.5), "jtype=Rational{Int64}")),
        "day": (dt.date(2024, 3, 15), (4, 0, F64(1710460800.0), "jtype=Date")),
        "stamp": (
            dt.datetime(2024, 3, 15, 13, 45, 30, 123000),
            (4, 0, F64(1710510330.123), "jtype=DateTime"),
        ),
        "z": (
            IntegerComplex(2**54, -1),
            (5, 0, struct.pack("<dd", 2.0**54, -1.0), "jtype=Complex{Int64}"),
        ),
        "sym": (StoredScalar.symbol("gdp"), (6, 0, b"gdp\0", "jtype=Symbol")),
        "raw": (StoredScalar.raw_text(b"f\xffo"), (6, 0, b"f\xffo\0", None)),
        "big": (
            StoredScalar.int128(-(2**127)),
            (1, 0, (-(2**127)).to_bytes(16, "little", signed=True), None),
        ),
        "ubig": (StoredScalar.uint128(2**128 - 1), (2, 0, b"\xff" * 16, None)),
        "c16": (
            StoredScalar.complexf16(complex(1.5, -2.25)),
            (5, 0, np.float16(1.5).tobytes() + np.float16(-2.25).tobytes(), None),
        ),
        "third32": (
            StoredScalar.fraction(Fraction(1, 3), parameter="Int32"),
            (4, 0, F64(1 / 3), "jtype=Rational{Int32}"),
        ),
        "year0": (StoredScalar.calendar_date(0, 1, 1), (4, 0, F64(-62167219200.0), "jtype=Date")),
        "opaque": (
            StoredScalar(F64(1.5), 4, 0, "Irrational{:π}"),
            (4, 0, F64(1.5), "jtype=Irrational{:π}"),
        ),
        "plain": (2.5, (4, 0, F64(2.5), None)),
    }
    with open_dataecon(path, "w") as db:
        for name, (value, _) in inputs.items():
            db.write_scalar(name, value)
    for name, (_, row) in inputs.items():
        assert stored_row(path, name) == row, name
    with open_dataecon(path) as db:
        assert db.read_scalar("half").to_fraction() == Fraction(1, 2)
        assert db.read_scalar("day").to_date() == dt.date(2024, 3, 15)
        assert db.read_scalar("stamp").to_datetime() == dt.datetime(2024, 3, 15, 13, 45, 30, 123000)
        assert db.read_scalar("z").to_integer_complex() == IntegerComplex(2**54, -1)
        assert db.read_scalar("sym").to_interpreted() == "gdp"
        assert db.read_scalar("raw") == StoredScalar.raw_text(b"f\xffo")
        assert db.read_scalar("big").to_int() == -(2**127)
        assert db.read_scalar("ubig").to_int() == 2**128 - 1
        assert db.read_scalar("c16").to_complex() == complex(1.5, -2.25)
        assert db.read_scalar("third32").to_fraction() == Fraction(1, 3)
        assert db.read_scalar("year0").to_calendar() == (0, 1, 1)
        assert db.read_scalar("opaque").marker == "Irrational{:π}"
        assert db.read_scalar("plain") == 2.5
        assert db.read_object("half").to_fraction() == Fraction(1, 2)
        kept = db.read_scalar("sym")
    assert kept.to_interpreted() == "gdp"  # owned result after closure


@NATIVE
def test_refusals_happen_before_any_native_call(tmp_path):
    path = tmp_path / "refused.daec"
    with open_dataecon(path, "w") as db:
        db.write_scalar("keep", 1)
        for value, error in (
            (Fraction(1, 3), ValueError),
            (dt.datetime(2024, 1, 1, tzinfo=dt.UTC), TypeError),
            (dt.datetime(2024, 1, 1, 0, 0, 0, 1), ValueError),
            (IntegerComplex(2**53 + 1, 0), ValueError),
            (2**64, ValueError),
            (b"bytes", TypeError),
        ):
            with pytest.raises(error):
                db.write_scalar("bad", value)
            assert not db.exists("bad")
        with open_dataecon(path) as reader, pytest.raises(ValueError, match="read-only"):
            reader.write_scalar("nope", Fraction(1, 2))
        assert [o.name for o in db.list_objects()] == ["keep"]


@NATIVE
def test_overwrite_validates_before_the_original_is_deleted(tmp_path):
    path = tmp_path / "overwrite.daec"
    with open_dataecon(path, "a") as db:
        db.write_scalar("x", StoredScalar.symbol("old"))
        with pytest.raises(ValueError):
            db.write_scalar("x", Fraction(1, 3), overwrite=True)
        assert db.read_scalar("x") == StoredScalar.symbol("old")
        with pytest.raises(DataEconError):
            db.write_scalar("x", Fraction(1, 2))
        db.write_scalar("x", Fraction(1, 2), overwrite=True)
        assert db.read_scalar("x").to_fraction() == Fraction(1, 2)
        assert db.get_attributes("x") == {"jtype": "Rational{Int64}"}
        db.write_scalar("x", 7, overwrite=True)
        assert db.read_scalar("x") == 7
        assert db.get_attributes("x") == {}


MARKER_FAULT = r"""
import json, sys
from fractions import Fraction
from tsecon.dataecon import open_dataecon, DataEconError
path = sys.argv[1]
db = open_dataecon(path, "a")
report = {}
try:
    db.write_scalar("half", Fraction(1, 2))
except DataEconError as exc:
    report["error"] = [exc.code, exc.operation]
else:
    raise AssertionError("injection did not fail")
# The value is stored without its marker and reads as the plain family.
report["read"] = db.read_scalar("half")
report["attributes"] = db.get_attributes("half")
try:
    db.close()
except DataEconError as exc:
    report["close"] = exc.code
report["closed"] = db.closed
print(json.dumps(report))
"""


@NATIVE
def test_failed_marker_write_leaves_the_plain_value_readable(tmp_path):
    # As for series markers: the failed attribute statement leaves the native
    # handle unable to close cleanly, so the sequence runs in a subprocess and
    # the residue is inspected from the file afterward.
    path = tmp_path / "residue.daec"
    with open_dataecon(path, "a") as db:
        db.write_scalar("first", 1)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "CREATE TRIGGER fail_marker BEFORE INSERT ON attributes WHEN NEW.name = 'jtype' "
            "BEGIN SELECT RAISE(ABORT, 'injected attribute failure'); END"
        )
    run = subprocess.run(
        [sys.executable, "-c", MARKER_FAULT, str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    report = json.loads(run.stdout)
    assert report["error"][0] == 19
    assert "write_scalar marker" in report["error"][1]
    assert "no rollback" in report["error"][1]
    assert report["read"] == 0.5
    assert report["attributes"] == {}
    assert report["close"] == 19
    assert report["closed"] is True
    assert stored_row(path, "half") == (4, 0, F64(0.5), None)
    with open_dataecon(path) as db:
        assert db.read_scalar("first") == 1


@NATIVE
def test_jeltype_on_a_scalar_is_ignored_like_julia(tmp_path):
    path = tmp_path / "jeltype.daec"
    with open_dataecon(path, "a") as db:
        db.write_scalar("x", 1.5)
        db.write_scalar("y", Fraction(1, 2))
        ids = {name: db.object_id(name) for name in ("x", "y")}
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.executemany(
            "INSERT INTO attributes VALUES (?, 'jeltype', 'Float32')", [(i,) for i in ids.values()]
        )
    with open_dataecon(path) as db:
        assert db.read_scalar("x") == 1.5
        assert db.read_scalar("y") == StoredScalar(F64(0.5), 4, 0, "Rational{Int64}")
        assert db.get_attributes("x") == {"jeltype": "Float32"}


@NATIVE
def test_workspace_round_trip_of_every_new_family():
    ws = Workspace(
        half=Fraction(1, 2),
        day=dt.date(2024, 1, 2),
        stamp=dt.datetime(2024, 1, 2, 3, 4, 5),
        z=IntegerComplex(1, -2),
        sym=StoredScalar.symbol("s"),
        wide=StoredScalar.uint128(2**64),
        nested=Workspace(third=StoredScalar.fraction(Fraction(1, 3), exact=False)),
    )
    with open_dataecon_memory() as db:
        assert db.write_workspace(ws).ok
        loaded, report = db.read_workspace()
        assert report.ok
        assert report.count == 8
        assert loaded.half.to_fraction() == Fraction(1, 2)
        assert loaded.day.to_date() == dt.date(2024, 1, 2)
        assert loaded.stamp.to_datetime() == dt.datetime(2024, 1, 2, 3, 4, 5)
        assert loaded.z.to_integer_complex() == IntegerComplex(1, -2)
        assert loaded.sym == StoredScalar.symbol("s")
        assert loaded.wide.to_int() == 2**64
        assert loaded.nested.third.to_fraction() == Fraction(6004799503160661, 18014398509481984)
        strict = db.read_workspace(strict=True)
        assert strict.report == report


@NATIVE
def test_malformed_wide_and_marked_rows_stay_refused(tmp_path):
    path = tmp_path / "malformed.daec"
    with open_dataecon(path, "a") as db:
        db.write_scalar("a", StoredScalar.int128(1))
        db.write_scalar("b", Fraction(1, 2))
        db.write_scalar("c", StoredScalar.symbol("s"))
        ids = {name: db.object_id(name) for name in "abc"}
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("UPDATE scalars SET value = zeroblob(12) WHERE id = ?", (ids["a"],))
        conn.execute("UPDATE scalars SET frequency = 32 WHERE id = ?", (ids["b"],))
        conn.execute("UPDATE attributes SET value = NULL WHERE id = ?", (ids["c"],))
    with open_dataecon(path) as db:
        with pytest.raises(ValueError, match="width"):
            db.read_scalar("a")
        with pytest.raises(TypeError, match="no frequency"):
            db.read_scalar("b")
        with pytest.raises(TypeError, match="NULL reconstruction attribute"):
            db.read_scalar("c")
        report = db.read_workspace().report
        assert [s.category for s in report.skipped] == ["invalid", "unsupported", "unsupported"]
