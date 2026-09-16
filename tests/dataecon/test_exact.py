"""Exact reconstruction arithmetic behind Julia's marker-mapped scalars.

Pure Python tests (no native build). Every expected value in ``JULIA_ROWS``
was recorded from the pinned Julia loader reading the listed payload bytes
with the listed ``jtype`` marker: ``("rational", (num, den))`` is the
``Rational{T}`` it built, ``("complex", (re, im))`` the integer ``Complex{T}``,
``("date", (y, m, d))`` / ``("datetime", (y, m, d, h, mi, s, ms))`` the
``Date``/``DateTime``, ``("int", n)`` the Int128/UInt128 value, and
``("error", name)`` the exception type it raised. The helpers must reproduce
each outcome from the bytes alone. Independent oracles cross-check the
transcriptions: Python's own ``datetime`` for calendar dates within its range,
and the exact continued fraction of the float for ``Rational{T}`` wherever its
partial quotients are exactly representable (above ``2**53`` Julia's Float64
loop rounds them, and two such cases confirmed with the pinned Julia are
pinned as literals).
"""

from __future__ import annotations

import datetime as dt
import math
import struct

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from tsecon.dataecon import _exact as ex

# (name, payload hex, (class, type, frequency, nbytes), jtype, Julia outcome)
JULIA_ROWS = [
    (
        "int128_min",
        "00000000000000000000000000000080",
        (1, 1, 0, 16),
        None,
        ("int", -170141183460469231731687303715884105728),
    ),
    (
        "int128_max",
        "ffffffffffffffffffffffffffffff7f",
        (1, 1, 0, 16),
        None,
        ("int", 170141183460469231731687303715884105727),
    ),
    ("int128_neg1", "ffffffffffffffffffffffffffffffff", (1, 1, 0, 16), None, ("int", -1)),
    (
        "uint128_max",
        "ffffffffffffffffffffffffffffffff",
        (1, 2, 0, 16),
        None,
        ("int", 340282366920938463463374607431768211455),
    ),
    ("uint128_zero", "00000000000000000000000000000000", (1, 2, 0, 16), None, ("int", 0)),
    (
        "uint128_2p64",
        "00000000000000000100000000000000",
        (1, 2, 0, 16),
        None,
        ("int", 18446744073709551616),
    ),
    ("cf16", "003e80c0", (1, 5, 0, 4), None, ("complexf16", "Float16(1.5) - Float16(2.25)im")),
    ("cf16_nan", "007e007c", (1, 5, 0, 4), None, ("complexf16", "NaN16 + Inf16*im")),
    (
        "cf16_negzero",
        "00800000",
        (1, 5, 0, 4),
        None,
        ("complexf16", "Float16(-0.0) + Float16(0.0)im"),
    ),
    ("date", "000000e0e37cd941", (1, 4, 0, 8), "Date", ("date", (2024, 3, 15))),
    ("date_epoch", "0000000000000000", (1, 4, 0, 8), "Date", ("date", (1970, 1, 1))),
    ("date_before_epoch", "000000000018f5c0", (1, 4, 0, 8), "Date", ("date", (1969, 12, 31))),
    ("date_year0", "000000f8e8f22cc2", (1, 4, 0, 8), "Date", ("date", (0, 1, 1))),
    ("date_negative_year", "0000006577d135c2", (1, 4, 0, 8), "Date", ("date", (-1000, 6, 30))),
    ("date_year1", "000000ee23ef2cc2", (1, 4, 0, 8), "Date", ("date", (1, 1, 1))),
    ("date_year9999", "00000078f97f4d42", (1, 4, 0, 8), "Date", ("date", (9999, 12, 31))),
    ("date_year10000", "0000c020fa7f4d42", (1, 4, 0, 8), "Date", ("date", (10000, 1, 1))),
    ("date_year_300k", "000072a3811ba142", (1, 4, 0, 8), "Date", ("date", (300000, 1, 1))),
    (
        "datetime",
        "0000803e147dd941",
        (1, 4, 0, 8),
        "DateTime",
        ("datetime", (2024, 3, 15, 13, 45, 30, 0)),
    ),
    (
        "datetime_ms",
        "3bdf873e147dd941",
        (1, 4, 0, 8),
        "DateTime",
        ("datetime", (2024, 3, 15, 13, 45, 30, 123)),
    ),
    (
        "datetime_ms_999",
        "9eefff3f387dd941",
        (1, 4, 0, 8),
        "DateTime",
        ("datetime", (2024, 3, 15, 23, 59, 59, 999)),
    ),
    (
        "datetime_epoch",
        "0000000000000000",
        (1, 4, 0, 8),
        "DateTime",
        ("datetime", (1970, 1, 1, 0, 0, 0, 0)),
    ),
    (
        "datetime_negative_ms",
        "fca9f1d24d6250bf",
        (1, 4, 0, 8),
        "DateTime",
        ("datetime", (1969, 12, 31, 23, 59, 59, 999)),
    ),
    (
        "datetime_year0",
        "7dfffff7e8f22cc2",
        (1, 4, 0, 8),
        "DateTime",
        ("datetime", (0, 1, 1, 0, 0, 0, 1)),
    ),
    (
        "datetime_year9999",
        "dfffbf20fa7f4d42",
        (1, 4, 0, 8),
        "DateTime",
        ("datetime", (9999, 12, 31, 23, 59, 59, 999)),
    ),
    (
        "datetime_year10000",
        "e500c020fa7f4d42",
        (1, 4, 0, 8),
        "DateTime",
        ("datetime", (10000, 1, 1, 0, 0, 0, 7)),
    ),
    (
        "datetime_year_300k",
        "02808b35831ba142",
        (1, 4, 0, 8),
        "DateTime",
        ("datetime", (300000, 6, 1, 12, 0, 0, 4)),
    ),
    (
        "datetime_year_neg300k",
        "fe7f48e36555a1c2",
        (1, 4, 0, 8),
        "DateTime",
        ("datetime", (-300000, 6, 1, 12, 0, 0, 4)),
    ),
    ("rational_half", "000000000000e03f", (1, 4, 0, 8), "Rational{Int64}", ("rational", (1, 2))),
    (
        "rational_third",
        "555555555555d53f",
        (1, 4, 0, 8),
        "Rational{Int64}",
        ("rational", (6004799503160661, 18014398509481984)),
    ),
    (
        "rational_neg_third",
        "555555555555d5bf",
        (1, 4, 0, 8),
        "Rational{Int64}",
        ("rational", (-6004799503160661, 18014398509481984)),
    ),
    ("rational_dyadic", "000000000000d83f", (1, 4, 0, 8), "Rational{Int64}", ("rational", (3, 8))),
    ("rational_int", "0000000000001c40", (1, 4, 0, 8), "Rational{Int64}", ("rational", (7, 1))),
    ("rational_zero", "0000000000000000", (1, 4, 0, 8), "Rational{Int64}", ("rational", (0, 1))),
    ("rational_inf", "000000000000f07f", (1, 4, 0, 8), "Rational{Int64}", ("rational", (1, 0))),
    ("rational_neginf", "000000000000f0ff", (1, 4, 0, 8), "Rational{Int64}", ("rational", (-1, 0))),
    (
        "rational_big_num",
        "555555555555b543",
        (1, 4, 0, 8),
        "Rational{Int64}",
        ("rational", (1537228672809129216, 1)),
    ),
    (
        "rational_2p53",
        "0000000000004043",
        (1, 4, 0, 8),
        "Rational{Int64}",
        ("rational", (9007199254740992, 1)),
    ),
    ("rational_int32", "555555555555d53f", (1, 4, 0, 8), "Rational{Int32}", ("rational", (1, 3))),
    ("rational_int8", "000000000000e83f", (1, 4, 0, 8), "Rational{Int8}", ("rational", (3, 4))),
    ("rational_uint8", "000000000000e83f", (1, 4, 0, 8), "Rational{UInt8}", ("rational", (3, 4))),
    (
        "rational_int128",
        "555555555555d53f",
        (1, 4, 0, 8),
        "Rational{Int128}",
        ("rational", (6004799503160661, 18014398509481984)),
    ),
    (
        "rational_int128_big",
        "5555555555551546",
        (1, 4, 0, 8),
        "Rational{Int128}",
        ("rational", (422550200076076443709319675904, 1)),
    ),
    (
        "rational_int16_small",
        "000000000000503f",
        (1, 4, 0, 8),
        "Rational{Int16}",
        ("rational", (1, 1024)),
    ),
    (
        "complex_int",
        "000000000000f03f0000000000000040",
        (1, 5, 0, 16),
        "Complex{Int64}",
        ("complex", (1, 2)),
    ),
    (
        "complex_int_neg",
        "00000000000008c00000000000000000",
        (1, 5, 0, 16),
        "Complex{Int64}",
        ("complex", (-3, 0)),
    ),
    (
        "complex_int_zero",
        "00000000000000000000000000000000",
        (1, 5, 0, 16),
        "Complex{Int64}",
        ("complex", (0, 0)),
    ),
    (
        "complex_int8",
        "000000000000f03f00000000000000c0",
        (1, 5, 0, 16),
        "Complex{Int8}",
        ("complex", (1, -2)),
    ),
    (
        "complex_uint8",
        "000000000000f03f0000000000000040",
        (1, 5, 0, 16),
        "Complex{UInt8}",
        ("complex", (1, 2)),
    ),
    (
        "complex_int128_big",
        "0000000000005044000000000000f03f",
        (1, 5, 0, 16),
        "Complex{Int128}",
        ("complex", (1180591620717411303424, 1)),
    ),
    (
        "complex_int_2p53",
        "00000000000040430000000000000000",
        (1, 5, 0, 16),
        "Complex{Int64}",
        ("complex", (9007199254740992, 0)),
    ),
    (
        "complex_bool",
        "000000000000f03f0000000000000000",
        (1, 5, 0, 16),
        "Complex{Bool}",
        ("complex", (1, 0)),
    ),
    (
        "complex_int64_max",
        "000000000000e043000000000000e0c3",
        (1, 5, 0, 16),
        "Complex{Int64}",
        ("error", "InexactError"),
    ),
    (
        "complex_float16",
        "003c0040",
        (1, 5, 0, 4),
        None,
        ("complexf16", "Float16(1.0) + Float16(2.0)im"),
    ),
    (
        "complex_int_2p54",
        "0000000000005043000000000000f03f",
        (1, 5, 0, 16),
        "Complex{Int64}",
        ("complex", (18014398509481984, 1)),
    ),
    (
        "complex_int_3x2p61",
        "000000000000d843000000000000b0c3",
        (1, 5, 0, 16),
        "Complex{Int64}",
        ("complex", (6917529027641081856, -1152921504606846976)),
    ),
    (
        "complex_int_2p53p1_control",
        "00000000000040430000000000000000",
        (1, 5, 0, 16),
        "Complex{Int64}",
        ("complex", (9007199254740992, 0)),
    ),
    (
        "complex_int128_2p100",
        "00000000000030460000000000005044",
        (1, 5, 0, 16),
        "Complex{Int128}",
        ("complex", (1267650600228229401496703205376, 1180591620717411303424)),
    ),
    (
        "complex_int128_2p127_neg",
        "000000000000d0c70000000000000000",
        (1, 5, 0, 16),
        "Complex{Int128}",
        ("complex", (-85070591730234615865843651857942052864, 0)),
    ),
    (
        "complex_int8_127",
        "0000000000c05f4000000000000060c0",
        (1, 5, 0, 16),
        "Complex{Int8}",
        ("complex", (127, -128)),
    ),
    ("inj_date_on_int64", "808ff36500000000", (1, 1, 0, 8), "Date", ("date", (2024, 3, 15))),
    (
        "inj_datetime_on_float",
        "3bdf873e147dd941",
        (1, 4, 0, 8),
        "DateTime",
        ("datetime", (2024, 3, 15, 13, 45, 30, 123)),
    ),
    (
        "inj_rational_on_int64",
        "0700000000000000",
        (1, 1, 0, 8),
        "Rational{Int64}",
        ("rational", (7, 1)),
    ),
    (
        "inj_rational_on_float_nan",
        "000000000000f87f",
        (1, 4, 0, 8),
        "Rational{Int64}",
        ("error", "InexactError"),
    ),
    (
        "inj_rational_on_float_huge",
        "9c7500883ce4377e",
        (1, 4, 0, 8),
        "Rational{Int64}",
        ("error", "InexactError"),
    ),
    (
        "inj_rational_int8_on_float",
        "000000000000e03f",
        (1, 4, 0, 8),
        "Rational{Int8}",
        ("rational", (1, 2)),
    ),
    (
        "inj_rational_int8_overflow",
        "0000000000408f40",
        (1, 4, 0, 8),
        "Rational{Int8}",
        ("error", "InexactError"),
    ),
    (
        "inj_complex_int_on_float",
        "0000000000000040",
        (1, 4, 0, 8),
        "Complex{Int64}",
        ("complex", (2, 0)),
    ),
    (
        "inj_complex_int_on_float_frac",
        "000000000000f83f",
        (1, 4, 0, 8),
        "Complex{Int64}",
        ("error", "InexactError"),
    ),
    (
        "inj_date_on_complex",
        "000000000000f03f0000000000000000",
        (1, 5, 0, 16),
        "Date",
        ("date", (1970, 1, 1)),
    ),
    (
        "inj_wide_complex_int128",
        "05000000000000000000000000000000",
        (1, 1, 0, 16),
        "Complex{Int64}",
        ("complex", (5, 0)),
    ),
    (
        "rat64_2p_neg100",
        "000000000000b039",
        (1, 4, 0, 8),
        "Rational{Int64}",
        ("error", "InexactError"),
    ),
    (
        "rat64_2p_neg62",
        "000000000000103c",
        (1, 4, 0, 8),
        "Rational{Int64}",
        ("rational", (1, 4611686018427387904)),
    ),
    (
        "rat64_2p_neg63",
        "000000000000003c",
        (1, 4, 0, 8),
        "Rational{Int64}",
        ("error", "InexactError"),
    ),
    (
        "rat64_2p54",
        "0000000000005043",
        (1, 4, 0, 8),
        "Rational{Int64}",
        ("rational", (18014398509481984, 1)),
    ),
    (
        "rat64_3x2p61",
        "000000000000d843",
        (1, 4, 0, 8),
        "Rational{Int64}",
        ("rational", (6917529027641081856, 1)),
    ),
    ("rat64_2p63", "000000000000e043", (1, 4, 0, 8), "Rational{Int64}", ("error", "InexactError")),
    (
        "rat64_neg2p63",
        "000000000000e0c3",
        (1, 4, 0, 8),
        "Rational{Int64}",
        ("error", "InexactError"),
    ),
    (
        "rat64_subnormal",
        "0100000000000000",
        (1, 4, 0, 8),
        "Rational{Int64}",
        ("error", "InexactError"),
    ),
    (
        "rat64_tenth",
        "9a9999999999b93f",
        (1, 4, 0, 8),
        "Rational{Int64}",
        ("rational", (3602879701896397, 36028797018963968)),
    ),
    ("rat8_1_128", "000000000000803f", (1, 4, 0, 8), "Rational{Int8}", ("error", "InexactError")),
    ("rat8_1_64", "000000000000903f", (1, 4, 0, 8), "Rational{Int8}", ("rational", (1, 64))),
    ("rat8_127", "0000000000c05f40", (1, 4, 0, 8), "Rational{Int8}", ("rational", (127, 1))),
    ("rat8_128", "0000000000006040", (1, 4, 0, 8), "Rational{Int8}", ("error", "InexactError")),
    ("rat8_neg128", "00000000000060c0", (1, 4, 0, 8), "Rational{Int8}", ("error", "InexactError")),
    ("rat8_neg127", "0000000000c05fc0", (1, 4, 0, 8), "Rational{Int8}", ("rational", (-127, 1))),
    (
        "ratu8_neg_half",
        "000000000000e0bf",
        (1, 4, 0, 8),
        "Rational{UInt8}",
        ("error", "OverflowError"),
    ),
    (
        "rat64_neg2p63_edge",
        "ffffffffffffdfc3",
        (1, 4, 0, 8),
        "Rational{Int64}",
        ("rational", (-9223372036854774784, 1)),
    ),
    ("ratu8_half", "000000000000e03f", (1, 4, 0, 8), "Rational{UInt8}", ("rational", (1, 2))),
    (
        "rat128_2p_neg100",
        "000000000000b039",
        (1, 4, 0, 8),
        "Rational{Int128}",
        ("rational", (1, 1267650600228229401496703205376)),
    ),
    (
        "rat128_2p_neg127",
        "0000000000000038",
        (1, 4, 0, 8),
        "Rational{Int128}",
        ("error", "InexactError"),
    ),
    (
        "rat128_2p126",
        "000000000000d047",
        (1, 4, 0, 8),
        "Rational{Int128}",
        ("rational", (85070591730234615865843651857942052864, 1)),
    ),
    (
        "rat64_on_int64_max",
        "ffffffffffffff7f",
        (1, 1, 0, 8),
        "Rational{Int64}",
        ("rational", (9223372036854775807, 1)),
    ),
    (
        "rat64_on_int128_big",
        "00000000000000004000000000000000",
        (1, 1, 0, 16),
        "Rational{Int64}",
        ("error", "InexactError"),
    ),
    (
        "cint_on_int64_2p53p1",
        "0100000000002000",
        (1, 1, 0, 8),
        "Complex{Int64}",
        ("complex", (9007199254740993, 0)),
    ),
    (
        "cint_on_float_2p53p1",
        "0000000000004043",
        (1, 4, 0, 8),
        "Complex{Int64}",
        ("complex", (9007199254740992, 0)),
    ),
    (
        "cint_on_int128_2p70",
        "00000000000000004000000000000000",
        (1, 1, 0, 16),
        "Complex{Int64}",
        ("error", "InexactError"),
    ),
    (
        "cint128_on_int128_2p70",
        "00000000000000004000000000000000",
        (1, 1, 0, 16),
        "Complex{Int128}",
        ("complex", (1180591620717411303424, 0)),
    ),
    (
        "cint_on_complex_2p53p1",
        "0000000000004043000000000000f03f",
        (1, 5, 0, 16),
        "Complex{Int64}",
        ("complex", (9007199254740992, 1)),
    ),
]


def _payload(header, hexstr):
    """The decoded payload of a row by native type code and width."""
    _, kind, _, nbytes = header
    raw = bytes.fromhex(hexstr)
    assert len(raw) == nbytes
    if kind == 4 and nbytes == 8:
        return "float", struct.unpack("<d", raw)[0]
    if kind == 5 and nbytes == 16:
        return "complex", struct.unpack("<dd", raw)
    if kind == 1 and nbytes == 8:
        return "int", struct.unpack("<q", raw)[0]
    if kind in (1, 2) and nbytes == 16:
        return "int", int.from_bytes(raw, "little", signed=kind == 1)
    return "bytes", raw


def _unmarked_outcome(header, value):
    if header[3] == 16:
        signed = header[1] == 1
        return "int", ex.int128_from_bytes(
            value.to_bytes(16, "little", signed=signed), signed=signed
        )
    return "complexf16", ex.complexf16_from_bytes(value)


def _rational_outcome(family, value, parameter):
    if family == "float":
        return "rational", ex.rationalize(value, parameter)
    return "rational", ex.rational_from_integer(value, parameter)


def _complex_outcome(family, value, parameter):
    if family == "float":
        return "complex", ex.integer_complex_from_floats(value, 0.0, parameter)
    if family == "complex":
        return "complex", ex.integer_complex_from_floats(*value, parameter)
    return "complex", ex.integer_complex_from_integer(value, parameter)


def _date_outcome(family, value, marker):
    if family == "int":
        rata = ex.rata_die_ms_from_unix_integer(value)
    else:
        if family == "complex":
            assert value[1] == 0.0
            value = value[0]
        rata = ex.rata_die_ms_from_unix_seconds(value)
    if marker == "Date":
        return "date", ex.date_from_rata_die_ms(rata)
    return "datetime", ex.civil_from_rata_die_ms(rata)


def _julia_outcome(header, hexstr, marker):
    """What the helpers say Julia loads for a row (an ``("error", type)`` on refusal)."""
    family, value = _payload(header, hexstr)
    try:
        if marker is None:
            return _unmarked_outcome(header, value)
        if (parameter := ex.rational_parameter(marker)) is not None:
            return _rational_outcome(family, value, parameter)
        if (parameter := ex.integer_complex_parameter(marker)) is not None:
            return _complex_outcome(family, value, parameter)
        assert marker in ("Date", "DateTime")
        return _date_outcome(family, value, marker)
    except ValueError as error:
        message = str(error)
        for name in ("OverflowError", "InexactError"):
            if name in message:
                return "error", name
        raise


@pytest.mark.parametrize("row", JULIA_ROWS, ids=[row[0] for row in JULIA_ROWS])
def test_helpers_reproduce_julia_outcomes(row):
    name, hexstr, header, marker, expected = row
    kind, outcome = _julia_outcome(header, hexstr, marker)
    if expected[0] == "complexf16":
        # Julia prints the components; compare their bit patterns instead.
        real, imag = outcome
        assert real.tobytes() + imag.tobytes() == bytes.fromhex(hexstr)
        assert ex.complexf16_to_bytes(real, imag) == bytes.fromhex(hexstr)
        return
    assert (kind, outcome) == expected, name


@pytest.mark.parametrize(
    "row",
    [row for row in JULIA_ROWS if row[3] in ("Date", "DateTime") and row[0].startswith("date")],
)
def test_julia_written_dates_are_reproduced_bit_for_bit(row):
    # Julia's own writer produced these bytes from a Date/DateTime; the encode
    # path rebuilds the same Float64 from the calendar components.
    name, hexstr, _, marker, (_, civil) = row
    rata = ex.rata_die_ms_from_civil(*civil)
    assert ex.float64_bits(ex.unix_seconds_from_rata_die_ms(rata)) == bytes.fromhex(hexstr), name
    if marker == "Date":
        assert ex.date_from_rata_die_ms(rata) == civil


# ---- Rational{T}: Julia's rationalize against the exact continued fraction ----


def _exact_continued_fraction(x: float, low: int, high: int) -> tuple[tuple[int, int], int]:
    """The convergents of the exact ratio of ``|x|`` in Python integers.

    Returns the last convergent whose quotients and products fit ``T`` and the
    largest partial quotient that was pushed. Where every pushed quotient is
    exactly representable in Float64 (at most ``2**53``) Julia's float loop
    computes the same quotients, so the two must agree there; larger
    quotients round in Julia's loop and the results may legitimately differ.
    """
    p, q = (-1 if x < 0 else 1), 0
    pp, qq = 0, 1
    numerator, denominator = abs(x).as_integer_ratio()
    largest = 0
    while denominator:
        a, remainder = divmod(numerator, denominator)
        if not low <= a <= high:
            break
        np_, nq = a * p + pp, a * q + qq
        if not (low <= np_ <= high and low <= nq <= high):
            break
        largest = max(largest, a)
        p, pp = np_, p
        q, qq = nq, q
        numerator, denominator = denominator, remainder
    return (p, q), largest


def _exact_or_error(x: float, parameter: str):
    try:
        return ex.rationalize(x, parameter)
    except ValueError as error:
        return "OverflowError" if "OverflowError" in str(error) else "InexactError"


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=True),
    st.sampled_from(sorted(ex.RATIONAL_PARAMETERS)),
)
@example(1 / 3, "Int32")
@example(1 / 3, "Int64")
@example(0.1, "Int16")
@example(2.0**-1074, "Int128")
@example(-128.0, "Int8")
@example(-(2.0**63), "Int64")
@example(float(2**63 - 1024), "Int64")
@example(0.5, "UInt8")
@example(-0.0, "UInt64")
@example(3 * 2.0**-62, "Int64")
@example(2.7141251072346146e-36, "Int128")
def test_rationalize_reproduces_the_float_and_matches_the_exact_expansion_when_it_can(x, parameter):
    low, high = ex.RATIONAL_PARAMETERS[parameter]
    result = _exact_or_error(x, parameter)
    (exact, largest) = _exact_continued_fraction(x, low, high)
    if result == "OverflowError":
        assert low == 0
        assert x < 0
        return
    if result == "InexactError":
        # Julia refuses exactly when its best convergent does not reproduce x;
        # the exact expansion cannot do better when its quotients were exact.
        if largest <= 2**53:
            assert float(exact[0]) / float(exact[1]) != x if exact[1] else True
        return
    num, den = result
    assert float(num) / float(den) == x  # Julia's acceptance rule
    assert math.gcd(num, den) == 1 or (num, den) in ((0, 1),)
    if largest <= 2**53:
        assert (num, den) == exact


def test_partial_quotients_above_two_to_the_53_follow_julia_not_the_exact_fraction():
    # Confirmed with the pinned Julia: Rational{Int64}(3 * 2.0^-62) is
    # 3//4611686018427387649 (the exact fraction is 3//2^62 = 3//4611686018427387904)
    # and Rational{Int128}(2.7141251072346146e-36) is the rational below.
    assert ex.rationalize(3 * 2.0**-62, "Int64") == (3, 4611686018427387649)
    assert (3 * 2.0**-62).as_integer_ratio() == (3, 4611686018427387904)
    assert ex.rationalize(2.7141251072346146e-36, "Int128") == (
        408,
        150324684338411231602781414279658602777,
    )
    assert ex.rationalize(2.0**-62, "Int64") == (1, 4611686018427387904)  # a power of two is exact


@pytest.mark.parametrize(
    ("x", "parameter", "expected"),
    [
        (0.1, "Int32", (1, 10)),
        (1 / 3, "Int16", (1, 3)),
        (1 / 3, "Int8", (1, 3)),
        (0.1, "Int8", (1, 10)),  # the convergent 1//10 reproduces 0.1 exactly in Float64
        (0.3, "Int8", (3, 10)),  # Float64(3) / Float64(10) is the double 0.3, so 3//10 passes
        (0.7, "Int8", (7, 10)),
        (-0.0, "UInt8", (0, 1)),
        (0.0, "Int64", (0, 1)),
        (-math.inf, "Int8", (-1, 0)),
        (math.inf, "UInt16", (1, 0)),
        (-math.inf, "UInt16", "OverflowError"),
        (float(2**64 - 2**11), "UInt64", (2**64 - 2**11, 1)),
        (2.0**64, "UInt64", "InexactError"),
        (2.0**127, "Int128", "InexactError"),
        (-(2.0**127), "Int128", "InexactError"),
    ],
)
def test_rational_boundaries(x, parameter, expected):
    assert _exact_or_error(x, parameter) == expected


def test_rational_from_integer_uses_the_parameter_range_including_typemin():
    assert ex.rational_from_integer(-128, "Int8") == (-128, 1)
    assert ex.rational_from_integer(2**127 - 1, "Int128") == (2**127 - 1, 1)
    with pytest.raises(ValueError, match="InexactError"):
        ex.rational_from_integer(128, "Int8")
    with pytest.raises(ValueError, match="InexactError"):
        ex.rational_from_integer(-1, "UInt8")


def test_parameter_tokens_are_exact_spellings_from_the_finite_tables():
    assert ex.rational_parameter("Rational{Int64}") == "Int64"
    assert ex.rational_parameter("Rational{UInt128}") == "UInt128"
    assert ex.integer_complex_parameter("Complex{Bool}") == "Bool"
    for token in (
        "Rational",
        "Rational{Bool}",
        "Rational{Int}",
        "Rational{ Int64}",
        "Rational{Int64} ",
        "Base.Rational{Int64}",
        "Rational{Float64}",
    ):
        assert ex.rational_parameter(token) is None, token
    for token in (
        "Complex{Float64}",
        "ComplexF64",
        "Complex{Int}",
        "Complex{Int64",
        "Complex{Rational{Int64}}",
    ):
        assert ex.integer_complex_parameter(token) is None, token
    assert set(ex.RATIONAL_PARAMETERS) == set(ex.INTEGER_PARAMETERS)
    assert set(ex.COMPLEX_PARAMETERS) == set(ex.INTEGER_PARAMETERS) | {"Bool"}
    assert ex.INTEGER_PARAMETERS["Int8"] == (-128, 127)
    assert ex.INTEGER_PARAMETERS["UInt128"] == (0, 2**128 - 1)


# ---- Complex{T} and T(x) ---------------------------------------------------


@pytest.mark.parametrize(
    ("value", "parameter", "expected"),
    [
        (-0.0, "Int64", 0),
        (-128.0, "Int8", -128),
        (127.0, "Int8", 127),
        (128.0, "Int8", "InexactError"),
        (-129.0, "Int8", "InexactError"),
        (1.5, "Int64", "InexactError"),
        (math.nan, "Int64", "InexactError"),
        (math.inf, "UInt8", "InexactError"),
        (-1.0, "UInt8", "InexactError"),
        (2.0**63, "Int64", "InexactError"),
        (-(2.0**63), "Int64", -(2**63)),
        (2.0**64, "UInt64", "InexactError"),
        (2.0**64 - 2.0**11, "UInt64", 2**64 - 2**11),
        (2.0**127, "Int128", "InexactError"),
        (-(2.0**127), "Int128", -(2**127)),
        (2.0**128, "UInt128", "InexactError"),
        (1.0, "Bool", 1),
        (0.0, "Bool", 0),
        (2.0, "Bool", "InexactError"),
    ],
)
def test_integer_from_float_follows_julia_bounds(value, parameter, expected):
    if expected == "InexactError":
        with pytest.raises(ValueError, match="InexactError"):
            ex.integer_from_float(value, parameter)
    else:
        assert ex.integer_from_float(value, parameter) == expected


def test_integer_complex_components_are_checked_independently():
    assert ex.integer_complex_from_floats(-0.0, -0.0, "Int64") == (0, 0)
    assert ex.integer_complex_from_floats(127.0, -128.0, "Int8") == (127, -128)
    with pytest.raises(ValueError, match="InexactError"):
        ex.integer_complex_from_floats(1.0, 0.5, "Int64")
    with pytest.raises(ValueError, match="InexactError"):
        ex.integer_complex_from_floats(1.0, -1.0, "UInt8")
    assert ex.integer_complex_from_integer(2**53 + 1, "Int64") == (2**53 + 1, 0)
    assert ex.integer_complex_from_integer(2**70, "Int128") == (2**70, 0)
    with pytest.raises(ValueError, match="InexactError"):
        ex.integer_complex_from_integer(2**70, "Int64")


# ---- calendar -------------------------------------------------------------


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.integers(min_value=1, max_value=dt.date.max.toordinal()))
@example(1)
@example(dt.date.max.toordinal())
@example(ex.UNIX_EPOCH_MS // ex.MS_PER_DAY)
def test_calendar_agrees_with_python_datetime_inside_its_range(ordinal):
    date = dt.date.fromordinal(ordinal)
    assert ex.year_month_day(ordinal) == (date.year, date.month, date.day)
    assert ex.total_days(date.year, date.month, date.day) == ordinal


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.integers(min_value=-(10**9), max_value=10**9))
@example(0)
@example(-1)
@example(-366)
def test_calendar_round_trips_outside_pythons_range(days):
    year, month, day = ex.year_month_day(days)
    assert 1 <= month <= 12
    assert 1 <= day <= ex.days_in_month(year, month)
    assert ex.total_days(year, month, day) == days


def test_year_zero_is_a_leap_year_and_the_epoch_constant_matches_julia():
    assert ex.is_leap_year(0)
    assert ex.days_in_month(0, 2) == 29
    assert not ex.is_leap_year(-100)
    assert ex.is_leap_year(-400)
    assert ex.total_days(1970, 1, 1) * ex.MS_PER_DAY == ex.UNIX_EPOCH_MS
    assert ex.total_days(1970, 1, 1) == dt.date(1970, 1, 1).toordinal()
    assert ex.total_days(0, 12, 31) == 0
    assert ex.year_month_day(0) == (0, 12, 31)


def test_invalid_calendar_components_are_refused():
    with pytest.raises(ValueError):
        ex.total_days(2024, 13, 1)
    with pytest.raises(ValueError):
        ex.total_days(2023, 2, 29)
    with pytest.raises(ValueError):
        ex.rata_die_ms_from_civil(2024, 1, 1, 24)
    with pytest.raises(ValueError):
        ex.rata_die_ms_from_civil(2024, 1, 1, 0, 0, 0, 1000)


def test_unix_seconds_truncate_toward_zero_like_julia():
    # trunc(Int64, 1000 * x): -0.0015 s is -1 ms, +0.0015 s is 1 ms.
    assert ex.civil_from_rata_die_ms(ex.rata_die_ms_from_unix_seconds(-0.0015)) == (
        1969,
        12,
        31,
        23,
        59,
        59,
        999,
    )
    assert ex.civil_from_rata_die_ms(ex.rata_die_ms_from_unix_seconds(0.0015)) == (
        1970,
        1,
        1,
        0,
        0,
        0,
        1,
    )
    assert ex.rata_die_ms_from_unix_integer(0) == ex.UNIX_EPOCH_MS
    assert ex.date_from_rata_die_ms(ex.rata_die_ms_from_unix_integer(-1)) == (1969, 12, 31)


@pytest.mark.parametrize(
    "seconds", [math.nan, math.inf, -math.inf, 2.0**63 / 1000, 1e300, -(2.0**63) / 1000 - 1e6]
)
def test_unix_seconds_outside_int64_milliseconds_are_refused(seconds):
    with pytest.raises(ValueError):
        ex.rata_die_ms_from_unix_seconds(seconds)


def test_integer_unix_seconds_wrap_like_julias_int64():
    # Int64(1000) * n and UNIXEPOCH + ms wrap: Julia's defined modular
    # arithmetic, reproduced exactly (the pinned rows of julia_scalar_routes
    # and julia_marker_routes hold the resulting dates).
    wrapped = ex.rata_die_ms_from_unix_integer(2**63 // 1000 + 1)
    assert wrapped == ex.wrap_int64(ex.UNIX_EPOCH_MS + ex.wrap_int64(1000 * (2**63 // 1000 + 1)))
    assert wrapped < 0
    assert ex.rata_die_ms_from_unix_integer(-(2**63) // 1000 - 1) == ex.wrap_int64(
        ex.UNIX_EPOCH_MS + ex.wrap_int64(1000 * (-(2**63) // 1000 - 1))
    )
    assert ex.wrap_int64(2**63) == -(2**63)
    assert ex.wrap_int64(-(2**63) - 1) == 2**63 - 1
    # Just inside: the product fits, the epoch offset still fits.
    assert ex.rata_die_ms_from_unix_integer(2**63 // 1000 - 62135683201) > 0


def test_writer_side_seconds_follow_julia_float_division():
    # datetime2unix: Float64(ms - UNIXEPOCH) / 1000.0, so the millisecond
    # count is converted once (exactly below 2**53) and divided once.
    rata = ex.rata_die_ms_from_civil(2024, 3, 15, 13, 45, 30, 123)
    assert ex.unix_seconds_from_rata_die_ms(rata) == float(rata - ex.UNIX_EPOCH_MS) / 1000.0
    assert ex.unix_seconds_from_rata_die_ms(ex.UNIX_EPOCH_MS) == 0.0
    # Beyond 2**53 milliseconds Float64(ms) itself rounds: an odd count one
    # past 2**53 comes back one millisecond earlier (Julia's own loss), while
    # the even count round-trips.
    even = ex.UNIX_EPOCH_MS + 2**53
    odd = even + 1

    def round_trip(rata):
        return ex.rata_die_ms_from_unix_seconds(ex.unix_seconds_from_rata_die_ms(rata))

    assert round_trip(even) == even
    assert round_trip(odd) == even
    assert ex.civil_from_rata_die_ms(odd)[-1] == 993
    assert ex.civil_from_rata_die_ms(round_trip(odd))[-1] == 992


# ---- width-preserving bytes -----------------------------------------------


@pytest.mark.parametrize("value", [0, 1, -1, 2**64, -(2**127), 2**127 - 1])
def test_int128_bytes_round_trip(value):
    assert ex.int128_from_bytes(ex.int128_to_bytes(value, signed=True), signed=True) == value


@pytest.mark.parametrize("value", [0, 1, 2**64, 2**128 - 1])
def test_uint128_bytes_round_trip(value):
    assert ex.int128_from_bytes(ex.int128_to_bytes(value, signed=False), signed=False) == value


def test_int128_bytes_refuse_wrong_widths_and_ranges():
    with pytest.raises(ValueError):
        ex.int128_from_bytes(b"\x00" * 8, signed=True)
    for value, signed in ((2**127, True), (-1, False), (2**128, False), (True, True), (1.0, True)):
        with pytest.raises(ValueError):
            ex.int128_to_bytes(value, signed=signed)


def test_complexf16_bytes_keep_nan_payloads_and_signed_zeros():
    raw = bytes.fromhex("017e0080")  # NaN with payload bit 1, then -0.0
    real, imag = ex.complexf16_from_bytes(raw)
    assert math.isnan(float(real))
    assert real.tobytes() == bytes.fromhex("017e")
    assert float(imag) == 0.0
    assert math.copysign(1.0, float(imag)) == -1.0
    assert ex.complexf16_to_bytes(real, imag) == raw
    with pytest.raises(ValueError):
        ex.complexf16_from_bytes(b"\x00" * 8)
    with pytest.raises(TypeError):
        ex.complexf16_to_bytes(1.5, np.float16(0))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ex.complexf16_to_bytes(np.float32(1.5), np.float16(0))  # type: ignore[arg-type]


def test_float64_bits_round_trip_including_nan_payloads():
    raw = bytes.fromhex("0100000000000ff8")  # a quiet NaN with a payload bit
    assert ex.float64_bits(ex.float64_from_bits(raw)) == raw
    assert ex.float64_from_bits(ex.float64_bits(-0.0)) == 0.0
    assert math.copysign(1.0, ex.float64_from_bits(ex.float64_bits(-0.0))) == -1.0
    with pytest.raises(ValueError):
        ex.float64_from_bits(b"\x00" * 4)
