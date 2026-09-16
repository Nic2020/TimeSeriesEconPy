"""The remaining reconstruction routes against ``julia_marker_routes.toml``.

Every row of the fixture records what the pinned Julia loader built (type and
canonical text) or raised for a raw payload stored with an injected marker:
scalar routes over date, duration, wide and narrow payloads; Julia's printed
``Symbol`` forms (calendar dates in any year and at extreme codes); ``Rational{T}``
of Float16/Float32 bit patterns; the remaining ``Date``/``DateTime`` payload
families (with Julia's wrapping Int64 arithmetic); element tokens on series
and plain arrays (the dated complexes and ``Char`` included); empty-only
tokens; text whole-object markers (the printed ``Symbol`` of a String array
with Julia's escaping); wider ``Bool`` markers on plain arrays; and the
whole-object ``Symbol`` of numeric containers. The pure-Python interpretation
is checked row by row without the native extension.

The differences are listed explicitly with their reason: a date-kind scalar
code outside the native codec's verified window is refused at construction
(the established calendar restriction; Julia's own value there is the C
codec's wrapped artefact); ``±1//0`` has no Fraction (the carrier keeps it);
a ``datetime.date``/``datetime.datetime`` result needs years 1..9999
(``to_calendar()`` has every component); a Float16 unix time whose product is
``-Inf`` reaches an undefined ``unsafe_trunc`` in Julia; an invalid ``Char``
above U+10FFFF has no ``str`` character; a character this Python's Unicode
tables leave unassigned cannot be printed the way Julia's tables decide; and
the whole-object ``Symbol`` of a dated container is display text that depends
on the loading session's ``LINES``/``COLUMNS``.
"""

from __future__ import annotations

import datetime as dt
import math
import struct
import tomllib
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tsecon import MIT, Duration, Monthly
from tsecon.dataecon import (
    DatedComplex,
    IntegerComplex,
    RationalComplex,
    StoredArray,
    StoredElement,
    StoredMVTSeries,
    StoredScalar,
    StoredSeries,
    StoredText,
    _printed,
)
from tsecon.dataecon import _exact as ex
from tsecon.dataecon._codec import series_dtype
from tsecon.dataecon._metadata import _SCALAR_FREQUENCIES, julia_frequency_name
from tsecon.dataecon._scalars import validate_date_code
from tsecon.mvtseries import MVTSeries
from tsecon.tseries import TSeries

FIXTURES = Path(__file__).parent / "fixtures"
ROUTES = tomllib.loads((FIXTURES / "julia_marker_routes.toml").read_text(encoding="utf-8"))
ROWS = ROUTES["rows"]
BY_SECTION: dict[str, list[dict]] = {}
for _row in ROWS:
    BY_SECTION.setdefault(_row["section"], []).append(_row)

# Julia's failure classes: a value the loader refuses is a ValueError, a route
# it lacks (or text it cannot resolve) a TypeError. ``CodePointError`` is
# ``Char(::UInt32)`` refusing a code point. The ``ArgumentError`` rows are the
# native codec's own failure on a date code outside its window (refused at
# construction here) and ``Symbol`` of UInt8 bytes containing NUL (ValueError).
JULIA_ERRORS = {
    "InexactError": ValueError,
    "OverflowError": ValueError,
    "CodePointError": ValueError,
    "MethodError": TypeError,
    "ArgumentError": TypeError,
    "UndefVarError": TypeError,
}


def ids(section: str) -> list[str]:
    return [r["name"] for r in BY_SECTION[section]]


def canonical(value: object) -> str:  # noqa: PLR0911 - finite canonical text table
    """Julia's canonical text of a Python result (the probe's ``value`` convention)."""
    if type(value) is Fraction:
        return f"{value.numerator}//{value.denominator}"
    if type(value) is IntegerComplex:
        return f"{value.real},{value.imag}"
    if type(value) in (RationalComplex, DatedComplex):
        return f"{canonical(value.real)},{canonical(value.imag)}"  # type: ignore[union-attr]
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
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if type(value) is MIT:
        return _printed.mit_string(value.value, value.frequency)
    if type(value) is Duration:
        return str(value.value)
    assert type(value) is str
    return value


def char_text(value: object) -> str:
    """The probe's ``U+xxxx`` text of a loaded Char."""
    assert isinstance(value, str)
    assert len(value) == 1
    return f"U+{ord(value):x}"


def same_decimal(value: Decimal, text: str) -> bool:
    expected = Decimal(text)
    if value.is_nan() or expected.is_nan():
        return value.is_nan() and expected.is_nan()
    return value == expected


def stored_of(row: dict) -> StoredScalar:
    return StoredScalar(bytes.fromhex(row["hex"]), row["kind"], row["frequency"], row["token"])


def outside_native_window(row: dict) -> bool:
    """Whether a date-kind scalar code lies outside the native codec's verified window.

    Such a scalar is refused at construction (the established calendar
    restriction); the pinned loader's value there comes from the C codec's
    Int32/UInt32 unpacking, a wrapped artefact it warns about, not from the
    stored code.
    """
    if row["kind"] != 3:
        return False
    code = int.from_bytes(bytes.fromhex(row["hex"]), "little", signed=True)
    try:
        validate_date_code(row["frequency"], code)
    except ValueError:
        return True
    return False


def float16_undefined(raw: bytes, kind: int) -> bool:
    """Whether Julia's ``unix2datetime`` of this element reaches ``unsafe_trunc(Int64, -Inf16)``.

    The Float16 product ``Float16(1000) * x`` overflows to ``-Inf``, which
    Julia's Float16 range check admits; the conversion of an infinite value
    is undefined and the reader refuses it. Every Int64 wrap on the way is
    Julia's defined arithmetic and is reproduced instead.
    """
    if (kind, len(raw)) not in ((4, 2), (5, 4)):
        return False
    with np.errstate(all="ignore"):
        product = np.float16(1000) * np.frombuffer(raw[:2], dtype="<f2")[0]
    return bool(np.isinf(product) and product < 0)


def check_scalar_row(  # noqa: PLR0911, PLR0912, PLR0915 - one branch per family
    row: dict,
) -> None:
    token = row["token"]
    if outside_native_window(row):
        with pytest.raises(ValueError, match="reliable native date range"):
            stored_of(row)
        return
    stored = stored_of(row)
    if "error" in row:
        with pytest.raises(JULIA_ERRORS[row["error"]]) as info:
            stored.to_interpreted()
        assert "NoSuchType" not in str(info.value)
        return
    loaded_type = row["type"]
    if loaded_type in ("Date", "DateTime"):
        if float16_undefined(bytes.fromhex(row["hex"]), row["kind"]):
            with pytest.raises(ValueError, match="undefined conversion"):
                stored.to_interpreted()
            return
        parts = stored.to_calendar()
        text = "{}-{}-{}".format(*parts[:3])
        if loaded_type == "DateTime":
            text += "T{}:{}:{}.{}".format(*parts[3:])
        assert text == row["value"], row["name"]
        if 1 <= parts[0] <= 9999:
            assert canonical(stored.to_interpreted()) == row["value"]
        else:
            with pytest.raises(ValueError, match="outside Python's datetime range"):
                stored.to_interpreted()
        return
    if loaded_type.startswith("Rational{") and row["value"].endswith("//0"):
        with pytest.raises(ValueError, match="no Fraction can hold"):
            stored.to_interpreted()
        assert math.isinf(stored.to_float() if stored.kind == 4 else stored.to_complex().real)
        return
    if loaded_type.startswith("Complex{Rational{") and "//0," in row["value"] + ",":
        with pytest.raises(ValueError, match="no Fraction"):
            stored.to_interpreted()
        return
    if loaded_type == "Char":
        if int(row["value"][2:], 16) > 0x10FFFF:
            with pytest.raises(ValueError, match="invalid Char"):
                stored.to_interpreted()
            return
        value = stored.to_interpreted()
        assert type(value) is str
        assert char_text(value) == row["value"], row["name"]
        assert stored.to_char() == value
        return
    value = stored.to_interpreted()
    if type(value) is Decimal:
        assert loaded_type == "BigFloat"
        assert same_decimal(value, row["value"]), row["name"]
        return
    if loaded_type == "ComplexF16":
        assert type(value) is complex
        bits = np.float16(value.real).tobytes() + np.float16(value.imag).tobytes()
        assert bits.hex() == row["value"]
        return
    assert canonical(value) == row["value"], row["name"]
    if loaded_type == "Symbol":
        assert type(value) is str
        assert token == "Symbol"
        assert stored.to_printed() == value
    if type(value) is Fraction:
        assert loaded_type.startswith("Rational{")
    if type(value) is IntegerComplex:
        assert loaded_type.startswith("Complex{")
        assert not loaded_type.startswith(("Complex{Rational", "Complex{MIT", "Complex{Duration"))
    if type(value) is RationalComplex:
        assert loaded_type.startswith("Complex{Rational{")
    if type(value) is DatedComplex:
        assert loaded_type.startswith(("Complex{MIT{", "Complex{Duration{"))
        assert value.imag.value == 0
        assert stored.to_dated_complex() == value
    if type(value) is MIT:
        assert loaded_type == "MIT{" + _julia_frequency(value) + "}"
    if type(value) is Duration:
        assert loaded_type == "Duration{" + _julia_frequency(value) + "}"
    if type(value) is int:
        assert loaded_type in ("Int64", "Int128", "UInt128", "BigInt")


def _julia_frequency(value: MIT | Duration) -> str:
    return julia_frequency_name(value.frequency)


@pytest.mark.parametrize("row", BY_SECTION["scalar"], ids=ids("scalar"))
def test_scalar_routes(row: dict) -> None:
    check_scalar_row(row)


@pytest.mark.parametrize("row", BY_SECTION["symbol"], ids=ids("symbol"))
def test_printed_symbol_forms(row: dict) -> None:
    check_scalar_row(row)


@pytest.mark.parametrize("row", BY_SECTION["rational"], ids=ids("rational"))
def test_narrow_float_rationals(row: dict) -> None:
    check_scalar_row(row)


@pytest.mark.parametrize("row", BY_SECTION["calendar"], ids=ids("calendar"))
def test_remaining_calendar_payloads(row: dict) -> None:
    check_scalar_row(row)


@pytest.mark.parametrize("row", BY_SECTION["char"], ids=ids("char"))
def test_scalar_char_routes(row: dict) -> None:
    check_scalar_row(row)


def test_probe_provenance_and_coverage():
    assert ROUTES["pin"] == "fc0a0d01aed4903ea6c64b12d78d0bdf68468df6"
    assert len(ROWS) == 11021
    assert {r["section"] for r in ROWS} == {
        "scalar",
        "symbol",
        "rational",
        "calendar",
        "series",
        "empty",
        "text",
        "bool",
        "object",
        "char",
    }
    frequencies = {r["frequency"] for r in BY_SECTION["symbol"] if r["payload"].startswith("mit_")}
    assert frequencies == set(_SCALAR_FREQUENCIES)
    assert sum("error" in r for r in ROWS) == 3965
    # The listed differences, counted; every other loaded row is reproduced.
    assert sum(outside_native_window(r) for r in BY_SECTION["symbol"]) == 143
    undefined = [
        r["name"]
        for r in BY_SECTION["calendar"]
        if "error" not in r and float16_undefined(bytes.fromhex(r["hex"]), r["kind"])
    ]
    assert len(undefined) == 8


def test_printed_forms_are_julias_rules():
    assert ex.julia_float_string(1e5, 8) == "100000.0"
    assert ex.julia_float_string(1e6, 8) == "1.0e6"
    assert ex.julia_float_string(1234567.0, 8) == "1.234567e6"
    assert ex.julia_float_string(1e-4, 8) == "0.0001"
    assert ex.julia_float_string(1e-5, 8) == "1.0e-5"
    assert ex.julia_float_string(-0.0, 8) == "-0.0"
    assert ex.julia_float_string(np.float32(0.1), 4) == "0.1"
    assert ex.julia_float_string(np.float32(1e10), 4) == "1.0e10"
    assert ex.julia_float_string(np.float32(1e10), 4, show=True) == "1.0f10"
    assert ex.julia_float_string(np.float32(0.5), 4, show=True) == "0.5"
    assert ex.julia_float_string(np.float32(1e10), 4, typed=True) == "1.0f10"
    assert ex.julia_float_string(np.float16(0.3333), 2, typed=True) == "Float16(0.3333)"
    assert ex.julia_float_string(np.float16(1e4), 2, show=True) == "1.0e4"
    assert ex.julia_complex_string(1.5, -2.0, 8) == "1.5 - 2.0im"
    assert ex.julia_complex_string(math.inf, math.nan, 8) == "Inf + NaN*im"
    assert ex.julia_complex_string(np.float32(1), np.float32(0), 4) == "1.0f0 + 0.0f0im"
    assert StoredScalar(struct.pack("<q", 24288), 3, 32, "Symbol").to_printed() == "2024M1"
    assert StoredScalar(struct.pack("<q", 5), 1, 32, "Symbol").to_printed() == "5"
    # Calendar dates print in any proleptic year, as Julia's Date does.
    assert StoredScalar(struct.pack("<q", 0), 3, 12, "Symbol").to_printed() == "0000-12-31"
    assert StoredScalar(struct.pack("<q", -366), 3, 12, "Symbol").to_printed() == "-0001-12-31"
    assert StoredScalar(struct.pack("<q", 3652060), 3, 12, "Symbol").to_printed() == "10000-01-01"


def test_julia_date_arithmetic_wraps_like_int64():
    daily = _SCALAR_FREQUENCIES[12]
    assert _printed.mit_string(739252, daily) == "2025-01-01"
    assert _printed.mit_string(2**62, daily) == "0001-30141738682531948-3689348814741910326"
    assert _printed.mit_string(-(2**40), daily) == "-3010360589-01-18"
    assert _printed.mit_string(2**63 - 1, daily) == "0000--60283477365063881--7378697629483820653"
    weekly = _SCALAR_FREQUENCIES[23]
    assert _printed.mit_string(105604, weekly) == "2024-12-08"
    assert _printed.mit_string(2**63 - 1, weekly) == "0000--60283477365063881--7378697629483820659"
    bdaily = _SCALAR_FREQUENCIES[13]
    assert _printed.mit_string(528037, bdaily) == "2024-12-31"
    with pytest.raises(ValueError, match="InexactError"):
        _printed.mit_string(2**63 - 1, bdaily)
    assert _printed.mit_string(7, _SCALAR_FREQUENCIES[11]) == "7U"
    assert _printed.mit_string(-1, _SCALAR_FREQUENCIES[32]) == "-1M12"
    assert _printed.julia_yearmonthday(0) == (0, 12, 31)
    assert _printed.julia_yearmonthday(1) == (1, 1, 1)


def test_unix_times_wrap_like_julias_int64():
    # Int64(1000) * n wraps, and so does UNIXEPOCH + ms: the rows of the
    # calendar section pin the exact dates; here the arithmetic itself.
    assert ex.rata_die_ms_from_unix_integer(0) == ex.UNIX_EPOCH_MS
    assert ex.rata_die_ms_from_unix_integer(2**63 - 1) == ex.wrap_int64(
        ex.UNIX_EPOCH_MS + ex.wrap_int64(1000 * (2**63 - 1))
    )
    assert ex.rata_die_ms_from_unix_integer(9223372036854775) == ex.wrap_int64(
        ex.UNIX_EPOCH_MS + 9223372036854775000
    )
    # 1000 * (2**63 + 1) wraps modulo 2**64 to 1000 before the Int64 check.
    assert ex.rata_die_ms_from_unix_uint64(2**63 + 1) == ex.UNIX_EPOCH_MS + 1000
    with pytest.raises(ValueError, match="InexactError"):
        ex.rata_die_ms_from_unix_uint64(2**64 - 1)
    assert ex.rata_die_ms_from_unix_wide_integer(-5, signed=True) == ex.UNIX_EPOCH_MS - 5000
    with pytest.raises(ValueError, match="InexactError"):
        ex.rata_die_ms_from_unix_wide_integer(2**70, signed=True)
    assert ex.rata_die_ms_from_unix_seconds(9.223372036854e15) == ex.wrap_int64(
        ex.UNIX_EPOCH_MS + math.trunc(1000.0 * 9.223372036854e15)
    )
    with pytest.raises(ValueError, match="InexactError"):
        ex.rata_die_ms_from_unix_seconds(9.223372036854775e15)
    with pytest.raises(ValueError, match="undefined conversion"):
        ex.rata_die_ms_from_unix_narrow(np.float16(-1000.0), 2)
    with pytest.raises(ValueError, match="InexactError"):
        ex.rata_die_ms_from_unix_narrow(np.float16(1000.0), 2)
    with pytest.raises(ValueError, match="InexactError"):
        ex.rata_die_ms_from_unix_narrow(np.float32(1e20), 4)


def test_rational_complex_pair_and_constructor():
    pair = RationalComplex(Fraction(1, 2), Fraction(-2))
    assert pair.to_complex() == complex(0.5, -2.0)
    assert complex(pair) == complex(0.5, -2.0)
    with pytest.raises(TypeError, match="Fraction"):
        RationalComplex(1, Fraction(2))  # type: ignore[arg-type]
    stored = StoredScalar.rational_complex(Fraction(1, 2), Fraction(-2))
    assert stored.marker == "Complex{Rational{Int64}}"
    assert stored.to_interpreted() == pair
    assert StoredScalar.rational_complex(pair, parameter="Int8").to_rational_complex() == pair
    with pytest.raises(ValueError, match="exact=False"):
        StoredScalar.rational_complex(Fraction(1, 3), Fraction(0))
    lossy = StoredScalar.rational_complex(Fraction(1, 3), Fraction(0), exact=False)
    assert lossy.to_rational_complex().real == Fraction(6004799503160661, 18014398509481984)
    with pytest.raises(ValueError, match="OverflowError"):
        StoredScalar.rational_complex(Fraction(-1, 3), Fraction(0), parameter="UInt8", exact=False)
    with pytest.raises(TypeError, match="not both"):
        StoredScalar.rational_complex(pair, Fraction(1))
    with pytest.raises(TypeError, match="Unknown Rational parameter"):
        StoredScalar.rational_complex(Fraction(1), parameter="Bool")
    with pytest.raises(ValueError, match="storage range"):
        StoredScalar.rational_complex(Fraction(10**400), Fraction(0))
    inf = StoredScalar(struct.pack("<dd", math.inf, 1.0), 5, 0, "Complex{Rational{Int64}}")
    with pytest.raises(ValueError, match="no Fraction"):
        inf.to_rational_complex()
    assert inf.to_complex() == complex(math.inf, 1.0)
    with pytest.raises(TypeError, match="Complex\\{Rational"):
        StoredScalar(struct.pack("<d", 1.0), 4, 0, "Rational{Int64}").to_rational_complex()


def test_dated_complex_pair_and_routes():
    monthly = Monthly()
    pair = DatedComplex(MIT(monthly, 24288), MIT(monthly, 0))
    assert pair.frequency == monthly
    with pytest.raises(TypeError, match="both be MIT"):
        DatedComplex(MIT(monthly, 1), Duration(monthly, 0))
    with pytest.raises(TypeError, match="both be MIT"):
        DatedComplex(1, 2)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="one frequency"):
        DatedComplex(MIT(monthly, 1), MIT(_SCALAR_FREQUENCIES[12], 0))
    stored = StoredScalar(struct.pack("<q", 24288), 3, 32, "Complex")
    assert stored.to_interpreted() == pair
    assert stored.to_dated_complex() == pair
    explicit = StoredScalar(struct.pack("<q", 24288), 3, 32, "Complex{MIT{Monthly}}")
    assert explicit.to_interpreted() == pair
    assert StoredScalar(b"\x05", 1, 0, "Complex{MIT{Monthly}}").to_interpreted() == DatedComplex(
        MIT(monthly, 5), MIT(monthly, 0)
    )
    duration = StoredScalar(struct.pack("<q", -5), 1, 32, "Complex").to_interpreted()
    assert duration == DatedComplex(Duration(monthly, -5), Duration(monthly, 0))
    with pytest.raises(TypeError, match=r"no Complex\{Duration\{Monthly\}\} conversion"):
        StoredScalar(b"\x05", 1, 0, "Complex{Duration{Monthly}}").to_interpreted()
    with pytest.raises(TypeError, match=r"no Complex\{MIT\{Monthly\}\} conversion"):
        StoredScalar(struct.pack("<d", 5.0), 4, 0, "Complex{MIT{Monthly}}").to_interpreted()
    with pytest.raises(ValueError, match="InexactError"):
        StoredScalar((2**70).to_bytes(16, "little"), 1, 0, "Complex{MIT{Monthly}}").to_interpreted()
    with pytest.raises(TypeError, match="to_dated_complex applies"):
        StoredScalar(struct.pack("<q", 5), 1, 0, "Complex").to_dated_complex()
    # A container of dated complexes: the pair carrier, its Python objects and no storage.
    codes = np.array([24288, 24289], dtype="<i8")
    series = StoredSeries(ANCHOR, codes, StoredElement.date(monthly).with_marker("Complex"))
    carrier = series.to_interpreted()
    assert isinstance(carrier, StoredSeries)
    assert carrier.element == StoredElement.dated_complex(monthly, "MIT")
    assert carrier.element.julia_name == "Complex{MIT{Monthly}}"
    assert carrier.tolist() == [
        DatedComplex(MIT(monthly, 24288), MIT(monthly, 0)),
        DatedComplex(MIT(monthly, 24289), MIT(monthly, 0)),
    ]
    with pytest.raises(TypeError, match="no DataEcon storage"):
        carrier.element.written_marker(2)
    with pytest.raises(TypeError, match="Dated complex elements need"):
        StoredElement.dated_complex(None, "MIT")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Unknown dated complex parameter"):
        StoredElement.dated_complex(monthly, "Date")


def test_char_routes_and_their_limits():
    assert StoredScalar(struct.pack("<q", 97), 1, 0, "Char").to_interpreted() == "a"
    assert StoredScalar(struct.pack("<q", 0xD800), 1, 0, "Char").to_char() == "\ud800"
    assert StoredScalar(struct.pack("<d", 97.0), 4, 0, "Char").to_char() == "a"
    assert StoredScalar(struct.pack("<dd", 97.0, 0.0), 5, 0, "Char").to_char() == "a"
    with pytest.raises(ValueError, match="InexactError"):
        StoredScalar(struct.pack("<d", 97.5), 4, 0, "Char").to_char()
    with pytest.raises(ValueError, match="InexactError"):
        StoredScalar(struct.pack("<q", -1), 1, 0, "Char").to_char()
    with pytest.raises(ValueError, match="CodePointError"):
        StoredScalar(struct.pack("<q", 0x200000), 1, 0, "Char").to_char()
    with pytest.raises(ValueError, match=r"invalid Char U\+110000"):
        StoredScalar(struct.pack("<q", 0x110000), 1, 0, "Char").to_char()
    with pytest.raises(TypeError, match="no Char conversion"):
        StoredScalar(struct.pack("<q", 97), 3, 32, "Char").to_interpreted()
    with pytest.raises(TypeError, match="no Char conversion"):
        StoredScalar(b"a\0", 6, 0, "Char").to_interpreted()
    chars = StoredArray(np.array([97, 0x1F642], dtype="<i8"), StoredElement.numeric("<i8", "Char"))
    loaded = chars.to_interpreted()
    assert isinstance(loaded, np.ndarray)
    assert loaded.dtype == np.dtype(object)
    assert loaded.tolist() == ["a", "\U0001f642"]
    nul = StoredArray(np.array([0], dtype="<i8"), StoredElement.numeric("<i8", "Char"))
    assert nul.to_interpreted().tolist() == ["\0"]
    with pytest.raises(ValueError, match="invalid Char"):
        StoredArray(
            np.array([97, 0x110000], dtype="<i8"), StoredElement.numeric("<i8", "Char")
        ).to_interpreted()
    with pytest.raises(TypeError, match="plain arrays load it"):
        StoredSeries(
            ANCHOR, np.array([97], dtype="<i8"), StoredElement.numeric("<i8", "Char")
        ).to_interpreted()


def test_bigfloat_constructor_and_decimal():
    tenth = Decimal(1) / Decimal(10)
    stored = StoredScalar.bigfloat(Decimal(1) / Decimal(8))
    assert stored.marker == "BigFloat"
    assert stored.to_float() == 0.125
    assert stored.to_interpreted() == Decimal("0.125")
    exact_tenth = Decimal("0.1000000000000000055511151231257827021181583404541015625")
    assert StoredScalar.bigfloat(exact_tenth).to_decimal() == exact_tenth
    with pytest.raises(ValueError, match="not exactly representable"):
        StoredScalar.bigfloat(tenth)
    assert StoredScalar.bigfloat(2**53).to_decimal() == Decimal(2**53)
    assert StoredScalar.bigfloat(math.inf).to_decimal() == Decimal("Infinity")
    assert StoredScalar.bigfloat(math.nan).to_decimal().is_nan()
    with pytest.raises(ValueError, match="not exactly representable"):
        StoredScalar.bigfloat(Decimal("0.1"))
    with pytest.raises(ValueError, match="not exactly representable"):
        StoredScalar.bigfloat(2**53 + 1)
    with pytest.raises(ValueError, match="beyond the Float64 range"):
        StoredScalar.bigfloat(10**400)
    with pytest.raises(TypeError, match="Decimal, float or int"):
        StoredScalar.bigfloat("0.1")  # type: ignore[arg-type]
    narrow = StoredScalar(np.float32(0.1).tobytes(), 4, 0, "BigFloat")
    assert narrow.to_decimal() == Decimal("0.100000001490116119384765625")
    with pytest.raises(ValueError, match="InexactError"):
        StoredScalar(struct.pack("<dd", 1.0, 2.0), 5, 0, "BigFloat").to_decimal()
    with pytest.raises(TypeError, match="no BigFloat"):
        StoredScalar(struct.pack("<q", 3), 3, 32, "BigFloat").to_decimal()


def test_integer_payload_dates_skip_the_native_window():
    huge = StoredScalar(struct.pack("<q", 2**62), 1, 0, "MIT{Monthly}").to_interpreted()
    assert type(huge) is MIT
    assert huge.value == 2**62
    daily = StoredScalar(struct.pack("<q", -(2**40)), 1, 0, "MIT{Daily}").to_interpreted()
    assert daily.value == -(2**40)
    assert StoredScalar(b"\x05", 1, 0, "MIT{Unit}").to_interpreted() == MIT(
        _SCALAR_FREQUENCIES[11], 5
    )
    with pytest.raises(TypeError, match="Int64 sources only"):
        StoredScalar(b"\x05", 1, 0, "Duration{Monthly}").to_interpreted()


# ---- containers: element tokens on series and plain arrays --------------------

ANCHOR = MIT(Monthly(), 24288)
_WIDTHS = {
    "Int8": 1,
    "UInt8": 1,
    "Int16": 2,
    "UInt16": 2,
    "Int32": 4,
    "UInt32": 4,
    "Float16": 2,
    "Float32": 4,
    "ComplexF16": 4,
}


def _itemsize(row: dict) -> int:
    base = row["base"].split("_")[0].split("{")[0]
    if base == "MITx":
        return 8
    return _WIDTHS.get(base, 16 if base in ("ComplexF64", "UInt128", "Int128") else 8)


def _shape(row: dict, raw: bytes) -> tuple[int, ...] | None:
    if row["container"] in ("matrix", "mvtseries"):
        return (2, 2) if raw else (0, 2)
    if row["container"] == "tensor":
        return (1, 2, 2) if raw else (0, 2, 2)
    return None


def container_of(row: dict) -> StoredSeries | StoredArray | StoredMVTSeries | np.ndarray[Any, Any]:
    """Build the stored container of a series/empty/bool row exactly as the codec would."""
    raw = bytes.fromhex(row["hex"])
    length = len(raw) // _itemsize(row)
    element = series_dtype(row["kind"], row["frequency"], length, len(raw), row["token"])
    if not isinstance(element, StoredElement):
        if not length:
            # The codec returns a typed empty array for a same-kind element token.
            return np.empty(_shape(row, raw) or 0, dtype=element)
        element = StoredElement.numeric(element, row["token"])
    values = np.frombuffer(raw, dtype=element.dtype).copy()
    shape = _shape(row, raw)
    if row["container"] == "tseries":
        return StoredSeries(ANCHOR, values, element)
    if row["container"] == "vector":
        return StoredArray(values, element)
    assert shape is not None
    shaped = values.reshape(shape, order="F").copy()
    if row["container"] == "mvtseries":
        return StoredMVTSeries(ANCHOR, ["c1", "c2"], shaped, element)
    return StoredArray(shaped, element)


def element_type(loaded_type: str) -> str:
    """The element type named in Julia's loaded container type."""
    if loaded_type.startswith(("TSeries{", "MVTSeries{")):
        inner = loaded_type[loaded_type.index(",") + 2 :]
        depth = 0
        for index, char in enumerate(inner):
            depth += char == "{"
            depth -= char == "}"
            if char == "," and depth == 0:
                return inner[:index]
        return inner.rstrip("}")
    if loaded_type.startswith("Vector{"):
        return loaded_type[len("Vector{") : -1]
    if loaded_type.startswith("Matrix{"):
        return loaded_type[len("Matrix{") : -1]
    assert loaded_type.startswith("Array{")
    return loaded_type[len("Array{") : loaded_type.rindex(",")]


def interpreted_elements(value: object) -> list[object]:
    """Column-major elements of an interpreted container, as Python objects."""
    if isinstance(value, (StoredSeries, StoredArray, StoredMVTSeries)):
        if value.element.kind == "complexf16":
            return [row.tobytes() for row in value.values.reshape(-1, order="F")]
        if isinstance(value, StoredSeries):
            return value.tolist()
        return list(np.array(value.tolist(), dtype=object).reshape(-1, order="F"))
    if isinstance(value, StoredText):
        return [item.decode() for item in value.values]
    array = value.values if isinstance(value, (TSeries, MVTSeries)) else value
    return list(np.asarray(array).reshape(-1, order="F"))


def canonical_element(item: object) -> str:
    if isinstance(item, np.datetime64):
        text = np.datetime_as_string(item)
        negative = text.startswith("-")
        date, _, clock = text.lstrip("-").partition("T")
        y, m, d = (int(part) for part in date.split("-"))
        out = f"{-y if negative else y}-{m}-{d}"
        if clock:
            h, mi, s = clock.split(":")
            sec, _, ms = s.partition(".")
            out += f"T{int(h)}:{int(mi)}:{int(sec)}.{int(ms or 0)}"
        return out
    if isinstance(item, Decimal):
        return "NaN" if item.is_nan() else str(item)
    if isinstance(item, bytes):
        return item.hex()
    return canonical(item)


def compare_elements(items: list[object], text: str, name: str, *, chars: bool = False) -> None:
    expected = text.split(";") if text else []
    assert len(items) == len(expected), name
    for item, want in zip(items, expected, strict=True):
        if isinstance(item, Decimal):
            assert same_decimal(item, want), name
        elif chars:
            assert char_text(item) == want, name
        else:
            assert canonical_element(item) == want, name


def _expect_refusal(row: dict, loaded: str) -> tuple[type[Exception], str] | None:
    """The listed differences on containers: the reader raises where Julia loads."""
    raw = bytes.fromhex(row["hex"])
    size = _itemsize(row)
    if loaded in ("Date", "DateTime") and any(
        float16_undefined(raw[start : start + size], row["kind"])
        for start in range(0, len(raw), size)
    ):
        return ValueError, "undefined conversion"
    if loaded.startswith(("Rational{", "Complex{Rational{")) and "//0" in row["value"]:
        return ValueError, "no Fraction"
    if loaded == "Char" and any(int(v[2:], 16) > 0x10FFFF for v in row["value"].split(";")):
        return ValueError, "invalid Char"
    return None


@pytest.mark.parametrize("row", BY_SECTION["series"], ids=ids("series"))
def test_series_and_array_routes(row: dict) -> None:
    dated = row["container"] in ("tseries", "mvtseries")
    if "error" in row:
        error: Any = JULIA_ERRORS[row["error"]]
        if dated and row["token"] in ("Date", "DateTime", "Symbol", "Char"):
            # Julia converts the elements (and may fail on a value) before the
            # dated constructor rejects the non-numeric result; the reader
            # refuses the route first.
            error = (TypeError, ValueError)
        with pytest.raises(error) as info:
            container_of(row).to_interpreted()
        assert "NoSuchType" not in str(info.value)
        return
    loaded = element_type(row["type"])
    refusal = _expect_refusal(row, loaded)
    if refusal is not None and refusal[1] == "no Fraction":
        # ±1//0 is preserved in the exact carrier; only the Fraction conversion raises.
        carrier = container_of(row).to_interpreted()
        assert isinstance(carrier, (StoredSeries, StoredArray, StoredMVTSeries))
        with pytest.raises(ValueError, match="no Fraction"):
            carrier.tolist()
        return
    if refusal is not None:
        with pytest.raises(refusal[0], match=refusal[1]):
            container_of(row).to_interpreted()
        return
    value = container_of(row).to_interpreted()
    compare_elements(interpreted_elements(value), row["value"], row["name"], chars=loaded == "Char")
    if loaded == "Symbol":
        assert isinstance(value, StoredText)
        assert value.marker == "Symbol"
    elif loaded == "Char":
        assert isinstance(value, np.ndarray)
        assert value.dtype == np.dtype(object)
    elif isinstance(value, (StoredSeries, StoredArray, StoredMVTSeries)):
        assert value.element.julia_name == loaded, row["name"]
        if loaded.startswith(("Complex{MIT{", "Complex{Duration{")):
            assert value.element.kind == "datedcomplex"
            assert all(type(item) is DatedComplex for item in interpreted_elements(value))
    elif loaded in ("BigInt", "BigFloat"):
        array = value.values if isinstance(value, (TSeries, MVTSeries)) else value
        assert np.asarray(array).dtype == object
    elif loaded in ("Date", "DateTime"):
        assert np.asarray(value).dtype == np.dtype("<M8[D]" if loaded == "Date" else "<M8[ms]")


EMPTY_DTYPES = {
    "Date": "<M8[D]",
    "DateTime": "<M8[ms]",
    "Symbol": "<U1",
    "String": "<U1",
    "Char": object,
    "Int8": "<i1",
    "Float32": "<f4",
}


@pytest.mark.parametrize("row", BY_SECTION["empty"], ids=ids("empty"))
def test_empty_only_routes(row: dict) -> None:
    if "error" in row:
        with pytest.raises(JULIA_ERRORS[row["error"]]):
            container_of(row).to_interpreted()
        return
    loaded = element_type(row["type"])
    if row["token"] == "Bool":
        # An empty Bool marker keeps its established empty Boolean result through the codec.
        assert series_dtype(row["kind"], 0, 0, 0, "Bool") == np.dtype("?")
        return
    container = container_of(row)
    value = container if isinstance(container, np.ndarray) else container.to_interpreted()
    if isinstance(value, (StoredSeries, StoredArray, StoredMVTSeries)):
        assert value.element.julia_name == loaded
        assert len(value) == 0
        return
    array = np.asarray(value.values if isinstance(value, (TSeries, MVTSeries)) else value)
    assert array.size == 0
    if loaded in EMPTY_DTYPES:
        assert array.dtype == np.dtype(EMPTY_DTYPES[loaded])
    elif loaded in ("Int64", "UInt64", "Float64", "ComplexF64"):
        assert array.dtype.kind in "iufc"
    else:
        assert array.dtype == np.dtype(object), (row["name"], loaded, array.dtype)


# ---- text: whole-object and element markers ------------------------------------

_TEXT_SHAPES: dict[str, tuple[int, ...] | None] = {
    "ab": (2,),
    "empty": (0,),
    "unicode": (2,),
    "escapes": None,
    "unassigned": (3,),
    "m22": (2, 2),
    "m02": (0, 2),
    "m12": (1, 2),
    "m21": (2, 1),
    "m20": (2, 0),
    "m00": (0, 0),
    "m22e": (2, 2),
    "t221": (2, 2, 1),
    "t022": (0, 2, 2),
    "t222": (2, 2, 2),
    "t212": (2, 1, 2),
    "t1114": (1, 1, 1, 4),
    "t11221": (1, 1, 2, 2, 1),
    "t202": (2, 0, 2),
    "t0000": (0, 0, 0, 0),
}


def text_container(row: dict) -> StoredText:
    raw = bytes.fromhex(row["hex"])
    elements = tuple(raw.split(b"\0")[:-1]) if raw else ()
    shape = _TEXT_SHAPES[row["base"]] or (len(elements),)
    return StoredText(elements, row["eltoken"] or None, shape, row["token"] or None)


def _probe_text(item: bytes) -> str:
    """The probe's text of one loaded string: itself, or ``bytes:<hex>`` when not UTF-8."""
    try:
        return item.decode("utf-8")
    except UnicodeDecodeError:
        return "bytes:" + item.hex()


@pytest.mark.parametrize("row", BY_SECTION["text"], ids=ids("text"))
def test_text_marker_routes(row: dict) -> None:
    stored = text_container(row)
    invalid = any(_probe_text(item).startswith("bytes:") for item in stored.values)
    if "error" in row:
        with pytest.raises(JULIA_ERRORS[row["error"]]):
            stored.to_interpreted()
        return
    if row["type"] == "Symbol":
        # The whole-object Symbol is Julia's printed String array, escaping included;
        # a character this Python's tables leave unassigned is refused, since
        # Julia's tables decide whether it prints raw (U+2FFC, U+1FAE9) or escaped (U+0378).
        if row["base"] == "unassigned":
            with pytest.raises(ValueError, match="unassigned in this Python's Unicode"):
                stored.to_interpreted()
            return
        assert stored.to_interpreted() == row["value"], row["name"]
        return
    expected = row["value"].split(";") if row["value"] else []
    if invalid:
        # Julia keeps the invalid bytes in its String/Symbol; decoding to str raises
        # here, while the Symbol element route keeps the bytes in a StoredText.
        with pytest.raises(ValueError):
            stored.tolist()
        if element_type(row["type"]) == "Symbol":
            value = stored.to_interpreted()
            assert isinstance(value, StoredText)
            assert [_probe_text(item) for item in value.values] == expected
        else:
            with pytest.raises(ValueError):
                stored.to_interpreted()
        assert [_probe_text(item) for item in stored.values] == expected
        return
    plain = stored.tolist()  # stored-text semantics are untouched by any marker
    value = stored.to_interpreted()
    loaded = element_type(row["type"])
    if loaded == "Symbol":
        assert isinstance(value, StoredText)
        assert value.marker == "Symbol"
        assert [item.decode() for item in value.values] == expected
        return
    if isinstance(value, list):
        assert value == expected == plain
        return
    flat = list(np.asarray(value, dtype=object).reshape(-1, order="F"))
    assert [str(item) for item in flat] == expected, row["name"]
    if loaded in ("Any", "AbstractString", "SubString{String}", "Union{String, Symbol}"):
        assert np.asarray(value).dtype == object
    else:
        assert loaded in ("String", "Char", "Int64", "Vector{String}")


def test_julia_string_escaping_rules():
    repr_of = _printed.julia_string_repr
    assert repr_of(b"") == '""'
    assert repr_of(b'a"b') == '"a\\"b"'
    assert repr_of(b"c\\d") == '"c\\\\d"'
    assert repr_of(b"$x") == '"\\$x"'
    assert repr_of(b"\t\n\r\x07\x08\x0b\x0c\x1b\x7f\x01") == '"\\t\\n\\r\\a\\b\\v\\f\\e\\x7f\\x01"'
    assert repr_of(b"\x000") == '"\\x000"'
    assert repr_of(b"\x00a") == '"\\0a"'
    assert repr_of("\u00e9\U0001f642\u65e5\u672c \u00a0".encode()) == (
        '"\u00e9\U0001f642\u65e5\u672c \u00a0"'
    )
    assert repr_of("\u200b\u2028".encode()) == '"\\u200b\\u2028"'
    assert repr_of("\u0085".encode()) == '"\\u85"'
    assert repr_of("\u00851".encode()) == '"\\u00851"'
    assert repr_of("\u0085g".encode()) == '"\\u85g"'
    assert repr_of("\U000e0001".encode()) == '"\\Ue0001"'
    assert repr_of("\U000e0001a".encode()) == '"\\U000e0001a"'
    assert repr_of(b"\xed\xa0\x80") == '"\\ud800"'
    assert repr_of(b"\xed\xa0\x801") == '"\\ud8001"'
    assert repr_of(b"\xff\xc0\x80\xe2\x82A\x80A") == '"\\xff\\xc0\\x80\\xe2\\x82A\\x80A"'
    assert repr_of(b"\xf8\x88\x80\x80\x80") == '"\\xf8\\x88\\x80\\x80\\x80"'
    assert repr_of(b"\xf4\x90\x80\x80") == '"\\U110000"'
    assert repr_of(b"\xf4\x90\x80\x80b") == '"\\U00110000b"'
    assert (
        repr_of(b"\xe0\x80\x80\xf0\x80\x80\x80\xc2") == '"\\xe0\\x80\\x80\\xf0\\x80\\x80\\x80\\xc2"'
    )
    with pytest.raises(ValueError, match=r"U\+0378 is unassigned"):
        repr_of("\u0378".encode())
    assert _printed.julia_chars(b"a\xc3\xa9\xff") == [
        (0x61, b"a"),
        (0xE9, b"\xc3\xa9"),
        (-1, b"\xff"),
    ]


def test_julia_array_layout_rules():
    cells = np.array(["1", "-2", "3", "4"], dtype=object)
    layout = _printed.array_string
    assert layout(cells, "", "Int64") == "[1, -2, 3, 4]"
    assert layout(cells.reshape((2, 2), order="F"), "Int8", "Int8") == "Int8[1 3; -2 4]"
    assert layout(cells.reshape((4, 1)), "", "Int64") == "[1; -2; 3; 4;;]"
    assert layout(cells.reshape((1, 4)), "", "Int64") == "[1 -2 3 4]"
    assert layout(cells.reshape((2, 2, 1), order="F"), "", "Int64") == "[1 3; -2 4;;;]"
    assert layout(cells.reshape((1, 2, 2), order="F"), "", "Int64") == "[1 -2;;; 3 4]"
    assert layout(cells.reshape((1, 1, 1, 4), order="F"), "", "Int64") == "[1;;;; -2;;;; 3;;;; 4]"
    assert (
        layout(cells.reshape((1, 1, 2, 2, 1), order="F"), "", "Int64")
        == "[1;;; -2;;;; 3;;; 4;;;;;]"
    )
    empty = np.empty(0, dtype=object)
    assert layout(empty, "", "Float64") == "Float64[]"
    assert layout(empty.reshape((0, 2)), "", "String") == "Matrix{String}(undef, 0, 2)"
    assert layout(empty.reshape((2, 0, 2)), "", "Int64") == "Array{Int64, 3}(undef, 2, 0, 2)"
    assert _printed.text_symbol((b"a", b"b"), (2,)) == '["a", "b"]'
    assert _printed.text_symbol((b"a", b"b", b"c", b"d"), (2, 2)) == '["a" "c"; "b" "d"]'
    assert _printed.byte_symbol(b"ab") == "ab"
    with pytest.raises(ValueError, match="NUL"):
        _printed.byte_symbol(b"a\0b")
    with pytest.raises(ValueError, match="not UTF-8"):
        _printed.byte_symbol(b"\xff")


# ---- object: whole-object Symbol on numeric containers ---------------------------


def object_container(row: dict) -> StoredSeries | StoredArray | StoredMVTSeries:
    raw = bytes.fromhex(row["hex"])
    size = _itemsize(row)
    length = len(raw) // size
    marker = row["eltoken"] or None
    element = series_dtype(row["kind"], row["frequency"], length, len(raw), marker)
    if not isinstance(element, StoredElement):
        element = StoredElement.numeric(element, marker)
    dims = tuple(row["dims"])
    values = np.frombuffer(raw, dtype=element.dtype).copy()
    if row["container"] == "vector":
        return StoredArray(values, element, object_marker=row["token"])
    if row["container"] == "tseries":
        return StoredSeries(ANCHOR, values, element, object_marker=row["token"])
    shaped = (
        values.reshape(dims, order="F").copy() if values.size else np.empty(dims, element.dtype)
    )
    if row["container"] == "mvtseries":
        return StoredMVTSeries(ANCHOR, ["c1", "c2"], shaped, element, object_marker=row["token"])
    return StoredArray(shaped, element, object_marker=row["token"])


@pytest.mark.parametrize("row", BY_SECTION["object"], ids=ids("object"))
def test_whole_object_symbol_on_numeric_containers(row: dict) -> None:
    stored = object_container(row)
    if row["container"] in ("tseries", "mvtseries"):
        # Julia's display text: the same stored object printed differently under
        # LINES=8,COLUMNS=30 (the rows with an env field), so nothing is reconstructed.
        with pytest.raises(TypeError, match="LINES/COLUMNS"):
            stored.to_interpreted()
        return
    if "error" in row:
        if row["token"] == "Symbol":
            # Symbol(::Vector{UInt8}) names the symbol with the bytes; NUL is refused.
            assert row["base"].startswith("UInt8")
            assert row["container"] == "vector"
            with pytest.raises(ValueError, match="NUL"):
                stored.to_interpreted()
            return
        with pytest.raises(JULIA_ERRORS[row["error"]]):
            stored.to_interpreted()
        return
    assert stored.active_marker == row["token"]
    if row["value"].startswith("bytes:"):
        # Julia's Symbol keeps the invalid bytes as its name; no str holds them.
        with pytest.raises(ValueError, match="not UTF-8"):
            stored.to_interpreted()
        return
    value = stored.to_interpreted()
    assert type(value) is str
    assert value == row["value"], row["name"]


def test_display_text_depends_on_the_loading_session():
    rows = [r for r in BY_SECTION["object"] if "env" in r]
    by_name: dict[str, dict[str, str]] = {}
    for r in rows:
        by_name.setdefault(r["name"], {})[r["env"]] = r["value"]
    differing = [name for name, values in by_name.items() if len(set(values.values())) > 1]
    assert differing, "the probe recorded no display-size dependence"
    # The empty series prints without any dependence; the reader still refuses it.
    assert any(len(values) == 1 for values in by_name.values())


# ---- wider Bool markers on plain arrays ----------------------------------------


@pytest.mark.parametrize("row", BY_SECTION["bool"], ids=ids("bool"))
def test_wide_bool_plain_arrays(row: dict) -> None:
    if "error" in row:
        with pytest.raises(JULIA_ERRORS[row["error"]]):
            container_of(row).to_interpreted()
        return
    stored = container_of(row)
    assert isinstance(stored, StoredArray)
    assert stored.element.marker == "Bool"
    assert stored.active_marker == "Bool"
    flags = stored.to_interpreted()
    assert isinstance(flags, np.ndarray)
    assert flags.dtype == np.dtype("?")
    assert flags.shape == stored.shape
    assert list(flags.reshape(-1, order="F")) == [c == "true" for c in row["value"].split(";")]
    assert np.array_equal(stored.to_bool(), flags)
    # The stored width, bytes and marker are what a rewrite stores;
    # Julia's own rewrite of the loaded Bool array is the canonical Int8 form.
    assert stored.values.tobytes(order="F") == bytes.fromhex(row["hex"])
    assert row["rewrite_metadata"] == [1, 4]
    assert row["rewrite_attributes"] == {"jeltype": "Bool"}
