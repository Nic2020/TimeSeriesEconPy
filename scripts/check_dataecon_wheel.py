# SPDX-License-Identifier: MIT
"""Require installed-wheel DataEcon provenance and interchange.

When native support is not required, verify core-only wheels omit the extension.
Run after the wheel has been installed, from outside its source package.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import re
import struct
import subprocess
import sys
import sysconfig
from fractions import Fraction
from pathlib import Path

import numpy as np
from build_dataecon_windows import HEADER_SHA256, SOURCE_COMMIT, SOURCE_SHA256

import tsecon
import tsecon.dataecon as de
from tsecon import (
    MIT,
    BDaily,
    Daily,
    Duration,
    HalfYearly,
    Monthly,
    Quarterly,
    Unit,
    Weekly,
    Yearly,
    bdaily,
    daily,
    mm,
    qq,
    weekly,
)
from tsecon.frequencies import Frequency


def validate_provenance(package: Path) -> dict:
    """Require the pinned source manifest and matching packaged DLL and notices."""
    manifest = json.loads((package / "_binary/build-info.json").read_text(encoding="utf-8"))
    if (
        manifest["source_commit"] != SOURCE_COMMIT
        or manifest["source_archive_sha256"] != SOURCE_SHA256
        or manifest["outputs"]["include/daec.h"] != HEADER_SHA256
        or manifest["dataecon_version"] != "0.4.0"
    ):
        raise ValueError("Unexpected DataEcon source/header provenance in the installed wheel.")
    if manifest.get("linkage") == "static-hidden":
        if (
            manifest["platform"] not in ("linux", "darwin")
            or not manifest["compiler"]
            or not re.fullmatch(r"[0-9a-f]{64}", manifest["outputs"]["lib/libdaec.a"])
        ):
            raise ValueError("Missing static native build provenance.")
    else:
        dll = package / "_binary/libdaec.dll"
        if hashlib.sha256(dll.read_bytes()).hexdigest() != manifest["outputs"]["bin/libdaec.dll"]:
            raise ValueError("Installed DataEcon DLL does not match its build manifest.")
        if not manifest["dependencies"] or not manifest["windows_sdk_versions"]:
            raise ValueError("Native dependency or SDK provenance is missing.")
    for notice in ("DATAECON_LICENSE.txt", "SQLITE_NOTICE.txt"):
        if not (package / notice).read_text(encoding="utf-8").strip():
            raise ValueError(f"Missing DataEcon wheel notice: {notice}")
    return manifest


def audit_static_extension(extension: Path) -> str:
    """Inspect the installed (repaired) extension's dependencies and public symbols."""
    if sys.platform == "linux":
        dependencies = subprocess.check_output(["readelf", "-d", str(extension)], text=True)
        symbols = subprocess.check_output(["nm", "-D", "--defined-only", str(extension)], text=True)
    else:
        dependencies = subprocess.check_output(["otool", "-L", str(extension)], text=True)
        symbols = subprocess.check_output(["nm", "-gU", str(extension)], text=True)
    if re.search(r"lib(?:daec|sqlite)", dependencies, re.I):
        raise ValueError(
            "Static DataEcon extension depends on an external DataEcon/SQLite library."
        )
    if re.search(r"\b_?(?:de_\w+|sqlite3\w*|_open)\s*$", symbols, re.M):
        raise ValueError("Private DataEcon/SQLite symbols escaped into the public symbol table.")
    if "PyInit__native" not in symbols:
        raise ValueError("DataEcon Python entry point is missing.")
    return dependencies


def loaded_dll_path() -> Path:
    """Inspect the Windows loader; no alternate DataEcon binding is used here."""
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
    kernel.GetModuleHandleW.restype = ctypes.c_void_p
    kernel.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    kernel.GetModuleFileNameW.restype = ctypes.c_uint32
    handle = kernel.GetModuleHandleW("libdaec.dll")
    buffer = ctypes.create_unicode_buffer(32768)
    if not handle:
        raise RuntimeError("DataEcon DLL was not loaded.")
    length = kernel.GetModuleFileNameW(handle, buffer, len(buffer))
    if not 0 < length < len(buffer):
        raise RuntimeError("Cannot determine the loaded DataEcon DLL path.")
    return Path(buffer.value).resolve()


def check_extension_abis(package: Path) -> None:
    """Reject stale interpreter-specific extension files."""
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    pattern = "*.pyd" if sys.platform == "win32" else "*.so"
    stale = [path.name for path in package.parent.rglob(pattern) if not path.name.endswith(suffix)]
    if stale:
        raise ValueError(f"Wheel contains extensions for another Python ABI: {stale}")


SCALAR_CASES = {
    "scalar_finite": 1.25,
    "scalar_negative": -2.5,
    "scalar_zero": 0.0,
    "scalar_negative_zero": -0.0,
    "scalar_nan": float("nan"),
    "scalar_positive_inf": float("inf"),
    "scalar_negative_inf": float("-inf"),
}


# Exact Int64 values, including both signed endpoints and values around 2**53
# that a floating-point detour would change.
INT64_CASES = {
    "int_zero": 0,
    "int_one": 1,
    "int_negative_one": -1,
    "int_seven": 7,
    "int_negative_seven": -7,
    "int_pow53": 2**53,
    "int_pow53_plus_one": 2**53 + 1,
    "int_pow53_minus_one": 2**53 - 1,
    "int_negative_pow53_minus_one": -(2**53 + 1),
    "int_pow62_plus_one": 2**62 + 1,
    "int_max": 2**63 - 1,
    "int_min": -(2**63),
    "int_min_plus_one": -(2**63) + 1,
    "int_max_minus_one": 2**63 - 2,
}


def _width_cases() -> dict:
    """Exact NumPy widths mirroring the Julia verifier's `width_cases` names and values."""
    nan, inf = float("nan"), float("inf")
    cases: dict = {}

    def signed(prefix, cls):
        info = np.iinfo(cls)
        for suffix, value in (
            ("zero", 0),
            ("one", 1),
            ("negative_one", -1),
            ("seven", 7),
            ("min", info.min),
            ("max", info.max),
            ("min_plus_one", info.min + 1),
            ("max_minus_one", info.max - 1),
        ):
            cases[f"w_{prefix}_{suffix}"] = cls(value)

    def unsigned(prefix, cls):
        info = np.iinfo(cls)
        for suffix, value in (
            ("zero", 0),
            ("one", 1),
            ("seven", 7),
            ("max", info.max),
            ("max_minus_one", info.max - 1),
            ("high_bit", info.max // 2 + 1),
        ):
            cases[f"w_{prefix}_{suffix}"] = cls(value)

    def floats(prefix, cls):
        info = np.finfo(cls)
        half = cls is np.float16
        for suffix, value in (
            ("zero", 0.0),
            ("negative_zero", -0.0),
            ("one_and_half" if half else "one_quarter", 1.5 if half else 1.25),
            ("tenth", 0.1),
            ("max", info.max),
            ("negative_max", -info.max),
            ("min_normal", info.tiny),
            ("min_subnormal", info.smallest_subnormal),
            ("nan", nan),
            ("inf", inf),
            ("negative_inf", -inf),
            ("two_pow_11_plus_one" if half else "two_pow_24_plus_one", 2049 if half else 16777217),
            ("pi", math.pi),
        ):
            cases[f"w_{prefix}_{suffix}"] = cls(value)

    def complexes(prefix, cls):
        info = np.finfo(np.float32 if cls is np.complex64 else np.float64)
        for suffix, value in (
            ("plain", complex(1.5, -2.25)),
            ("negative_zero_real", complex(-0.0, 0.0)),
            ("negative_zero_imag", complex(0.0, -0.0)),
            ("nan_real", complex(nan, 1.0)),
            ("inf_imag", complex(1.0, -inf)),
            ("max_min", complex(float(info.max), float(info.tiny))),
            ("subnormal", complex(float(info.smallest_subnormal), 0.0)),
            ("tenths", complex(0.1, 0.2)),
        ):
            cases[f"w_{prefix}_{suffix}"] = cls(value)

    floats("f16", np.float16)
    floats("f32", np.float32)
    signed("i8", np.int8)
    signed("i16", np.int16)
    signed("i32", np.int32)
    unsigned("u8", np.uint8)
    unsigned("u16", np.uint16)
    unsigned("u32", np.uint32)
    unsigned("u64", np.uint64)
    complexes("c32", np.complex64)
    complexes("c64", np.complex128)
    cases["w_pyc_plain"] = complex(8.0, 3.0)  # Python complex is stored as ComplexF64
    return cases


# Narrow, unsigned and complex widths; Julia must load each at its own width.
WIDTH_CASES = _width_cases()


# UTF-8 strings stored with a NUL terminator; Julia must load each as a String.
STRING_CASES = {
    "str_ascii": "hello",
    "str_empty": "",
    "str_digits": "7",
    "str_space": " ",
    "str_latin": "héllo wörld",
    "str_cjk": "日本語",
    "str_emoji": "🙂 ok",
    "str_combining": "é",
    "str_whitespace": "a\nb\tc\r\n",
    "str_delimiter": "a‖b",
    "str_slash": "a/b",
    "str_quote": 'say "hi"',
    "str_expression": "error(123)",
    "str_long": "x" * 1000,
    "str_long_utf8": "é" * 500,
}
# MIT date and Duration scalars over every year/period frequency and anchor.
# Reliable minimum codes: -32800 * periods_per_year, except annual (full Int32).
DATE_FAMILIES = (
    [("m", Monthly())]
    + [(f"q{a}", Quarterly(a)) for a in (1, 2, 3)]
    + [(f"h{a}", HalfYearly(a)) for a in range(1, 7)]
    + [(f"y{a}", Yearly(a)) for a in range(1, 13)]
)
DATE_MINIMUMS = {12: -393600, 4: -131200, 2: -65600, 1: -(2**31)}
DURATION_VALUES = (
    ("zero", 0),
    ("one", 1),
    ("negative_one", -1),
    ("pow53_plus_one", 2**53 + 1),
    ("max", 2**63 - 1),
    ("min", -(2**63)),
)


# Unit and calendar scalars. Unit codes are plain signed 64-bit integers; calendar
# codes are the native rata-die style codes of the period's end date. The window
# endpoints (`minimum`/`maximum`) and the year-10000 cases lie outside Python's
# datetime range and are handled as integer codes only. Dates in comments.
UNIT_CASES = {
    "min": -(2**63),
    "neg_pow40": -(2**40),
    "below_int32": -(2**31) - 1,
    "int32_min": -(2**31),
    "negative_one": -1,
    "zero": 0,
    "five": 5,
    "int32_max": 2**31 - 1,
    "beyond_int32": 2**31,
    "pow40": 2**40,
    "max": 2**63 - 1,
}
CALENDAR_FAMILIES = [("d", Daily()), ("b", BDaily())] + [
    (f"w{day}", Weekly(day)) for day in range(1, 8)
]
CALENDAR_CASES = {
    "d": (
        ("typical", 738900),  # 2024-01-15
        ("year_end", 739251),  # 2024-12-31
        ("year_start", 739252),  # 2025-01-01
        ("leap_day", 738945),  # 2024-02-29
        ("first_day", 1),  # 0001-01-01
        ("year_zero", -199),  # 0000-06-15
        ("negative_year", -2130),  # -0005-03-03
        ("negative_one", -1),  # 0000-12-30
        ("zero", 0),  # 0000-12-31
        ("minimum", -11980259),  # -32800-03-01
        ("maximum", 11979954),  # 32800-12-31
        ("py_max_year", 3652059),  # 9999-12-31
        ("beyond_py_year", 3652062),  # 10000-01-03
    ),
    "b": (
        ("typical", 527786),  # 2024-01-15
        ("year_end", 528037),  # 2024-12-31
        ("year_start", 528038),  # 2025-01-01
        ("leap_day", 527819),  # 2024-02-29
        ("first_day", 1),  # 0001-01-01
        ("year_zero", -141),  # 0000-06-15
        ("negative_year", -1520),  # -0005-03-03
        ("negative_one", -1),  # 0000-12-28
        ("zero", 0),  # 0000-12-29
        ("minimum", -8557114),  # -32800-12-25
        ("maximum", 8557110),  # 32800-12-29
        ("py_max_year", 2608615),  # 9999-12-31
        ("beyond_py_year", 2608616),  # 10000-01-03
    ),
    "w1": (
        ("typical", 105558),  # week ending 2024-01-15
        ("year_end", 105609),  # 2025-01-06
        ("year_start", 105609),  # 2025-01-06
        ("leap_day", 105565),  # 2024-03-04
        ("first_day", 1),  # 0001-01-01
        ("year_zero", -27),  # 0000-06-19
        ("negative_year", -303),  # -0005-03-06
        ("negative_one", -1),  # 0000-12-18
        ("zero", 0),  # 0000-12-25
        ("minimum", -1711422),  # -32800-12-25
        ("maximum", 1711422),  # 32800-12-25
        ("py_max_year", 521724),  # 10000-01-03
        ("beyond_py_year", 521724),  # 10000-01-03
    ),
    "w2": (
        ("typical", 105558),  # 2024-01-16
        ("year_end", 105608),  # 2024-12-31
        ("year_start", 105609),  # 2025-01-07
        ("leap_day", 105565),  # 2024-03-05
        ("first_day", 1),  # 0001-01-02
        ("year_zero", -27),  # 0000-06-20
        ("negative_year", -303),  # -0005-03-07
        ("negative_one", -1),  # 0000-12-19
        ("zero", 0),  # 0000-12-26
        ("minimum", -1711422),  # -32800-12-26
        ("maximum", 1711422),  # 32800-12-26
        ("py_max_year", 521724),  # 10000-01-04
        ("beyond_py_year", 521724),  # 10000-01-04
    ),
    "w3": (
        ("typical", 105558),  # 2024-01-17
        ("year_end", 105608),  # 2025-01-01
        ("year_start", 105608),  # 2025-01-01
        ("leap_day", 105565),  # 2024-03-06
        ("first_day", 1),  # 0001-01-03
        ("year_zero", -27),  # 0000-06-21
        ("negative_year", -303),  # -0005-03-08
        ("negative_one", -1),  # 0000-12-20
        ("zero", 0),  # 0000-12-27
        ("minimum", -1711422),  # -32800-12-27
        ("maximum", 1711422),  # 32800-12-27
        ("py_max_year", 521724),  # 10000-01-05
        ("beyond_py_year", 521724),  # 10000-01-05
    ),
    "w4": (
        ("typical", 105558),  # 2024-01-18
        ("year_end", 105608),  # 2025-01-02
        ("year_start", 105608),  # 2025-01-02
        ("leap_day", 105564),  # 2024-02-29
        ("first_day", 1),  # 0001-01-04
        ("year_zero", -28),  # 0000-06-15
        ("negative_year", -303),  # -0005-03-09
        ("negative_one", -1),  # 0000-12-21
        ("zero", 0),  # 0000-12-28
        ("minimum", -1711422),  # -32800-12-28
        ("maximum", 1711422),  # 32800-12-28
        ("py_max_year", 521724),  # 10000-01-06
        ("beyond_py_year", 521724),  # 10000-01-06
    ),
    "w5": (
        ("typical", 105558),  # 2024-01-19
        ("year_end", 105608),  # 2025-01-03
        ("year_start", 105608),  # 2025-01-03
        ("leap_day", 105564),  # 2024-03-01
        ("first_day", 1),  # 0001-01-05
        ("year_zero", -28),  # 0000-06-16
        ("negative_year", -304),  # -0005-03-03
        ("negative_one", -1),  # 0000-12-22
        ("zero", 0),  # 0000-12-29
        ("minimum", -1711422),  # -32800-12-29
        ("maximum", 1711422),  # 32800-12-29
        ("py_max_year", 521723),  # 9999-12-31
        ("beyond_py_year", 521724),  # 10000-01-07
    ),
    "w6": (
        ("typical", 105558),  # 2024-01-20
        ("year_end", 105608),  # 2025-01-04
        ("year_start", 105608),  # 2025-01-04
        ("leap_day", 105564),  # 2024-03-02
        ("first_day", 1),  # 0001-01-06
        ("year_zero", -28),  # 0000-06-17
        ("negative_year", -304),  # -0005-03-04
        ("negative_one", -1),  # 0000-12-23
        ("zero", 0),  # 0000-12-30
        ("minimum", -1711422),  # -32800-12-30
        ("maximum", 1711422),  # 32800-12-30
        ("py_max_year", 521723),  # 10000-01-01
        ("beyond_py_year", 521724),  # 10000-01-08
    ),
    "w7": (
        ("typical", 105558),  # 2024-01-21
        ("year_end", 105608),  # 2025-01-05
        ("year_start", 105608),  # 2025-01-05
        ("leap_day", 105564),  # 2024-03-03
        ("first_day", 1),  # 0001-01-07
        ("year_zero", -28),  # 0000-06-18
        ("negative_year", -304),  # -0005-03-05
        ("negative_one", -1),  # 0000-12-24
        ("zero", 0),  # 0000-12-31
        ("minimum", -1711422),  # -32800-12-31
        ("maximum", 1711422),  # 32800-12-31
        ("py_max_year", 521723),  # 10000-01-02
        ("beyond_py_year", 521724),  # 10000-01-09
    ),
}


def date_codes(frequency) -> tuple[tuple[str, int], ...]:
    """Return the same per-frequency date codes the Julia verifier expects."""
    ppy = frequency.periods_per_year
    return (
        ("typical", 2024 * ppy),
        ("cross_year", 2024 * ppy + ppy - 1),
        ("negative_one", -1),
        ("zero", 0),
        ("minimum", DATE_MINIMUMS[ppy]),
        ("maximum", 2**31 - 1),
    )


def scalar_cases() -> dict:
    """All scalar objects written to the interchange output, keyed by name."""
    cases = {**SCALAR_CASES, **INT64_CASES, **WIDTH_CASES, **STRING_CASES}
    for label, frequency in DATE_FAMILIES:
        for suffix, code in date_codes(frequency):
            cases[f"mit_{label}_{suffix}"] = MIT(frequency, code)
        for suffix, value in DURATION_VALUES:
            cases[f"dur_{label}_{suffix}"] = Duration(frequency, value)
    for label, frequency in CALENDAR_FAMILIES:
        for suffix, code in CALENDAR_CASES[label]:
            cases[f"mit_{label}_{suffix}"] = MIT(frequency, code)
        for suffix, value in DURATION_VALUES:
            cases[f"dur_{label}_{suffix}"] = Duration(frequency, value)
    for suffix, value in UNIT_CASES.items():
        cases[f"mit_u_{suffix}"] = MIT(Unit(), value)
        cases[f"dur_u_{suffix}"] = Duration(Unit(), value)
    return cases


QUARTERLY_CASES = (
    ("cross_year", 8099, [1.25, -2.5, 0.0, 4.75]),
    ("negative", -1, [1.25, -2.5]),
    ("zero", 0, [1.25]),
    ("minimum", -131200, [1.25]),
    ("maximum", 2147483647, [1.25]),
    ("empty", 8096, []),
    ("empty_minimum", -131200, []),
    ("empty_maximum", 2147483647, []),
)


ANNUAL_CASES = (
    ("cross_year", 2024, [1.25, -2.5, 0.0, 4.75]),
    ("negative", -1, [1.25, -2.5]),
    ("zero", 0, [1.25]),
    ("minimum", -(2**31), [1.25]),
    ("maximum", 2**31 - 1, [1.25]),
    ("empty", 2024, []),
    ("empty_minimum", -(2**31), []),
    ("empty_maximum", 2**31 - 1, []),
)


HALFYEARLY_CASES = (
    ("cross_year", 4048, [1.25, -2.5, 0.0, 4.75]),
    ("negative", -1, [1.25, -2.5]),
    ("zero", 0, [1.25]),
    ("minimum", -65600, [1.25]),
    ("maximum", 2**31 - 1, [1.25]),
    ("empty", 4048, []),
    ("empty_minimum", -65600, []),
    ("empty_maximum", 2**31 - 1, []),
)
# Name prefix, frequency constructor, fiscal anchors and cases written per target.
DATED_GROUPS = (
    ("q", Quarterly, (1, 2, 3), QUARTERLY_CASES),
    ("y", Yearly, range(1, 13), ANNUAL_CASES),
    ("h", HalfYearly, range(1, 7), HALFYEARLY_CASES),
)
# Calendar-frequency series (Daily 12, BDaily 13, Weekly 17..23): the Julia
# verifier expects these exact first codes, which the core's own calendar
# reproduces from the dates (30 December 2024, 28 February 2024, Friday 12
# January 2024, 15 January 2024, 31 December 9999) and the verified windows.
CALENDAR_WINDOWS = {12: (-11980259, 11979954), 13: (-8557114, 8557110)}
CALENDAR_WINDOWS.update(dict.fromkeys(range(17, 24), (-1711422, 1711422)))


def _calendar_dates(frequency) -> tuple[int, ...]:
    """First codes from the core's own calendar; Julia derives them independently."""
    dates = (
        dt.date(2024, 12, 30),
        dt.date(2024, 2, 28),
        dt.date(2024, 1, 12),
        dt.date(2024, 1, 15),
        dt.date(9999, 12, 31),
    )
    if isinstance(frequency, Daily):
        return tuple(daily(d).value for d in dates)
    if isinstance(frequency, BDaily):
        return tuple(bdaily(d).value for d in dates)
    return tuple(weekly(d, frequency.end_day).value for d in dates)


_CALENDAR_DATES = {label: _calendar_dates(frequency) for label, frequency in CALENDAR_FAMILIES}
_KNOWN_CODES = (_CALENDAR_DATES["d"][0], _CALENDAR_DATES["b"][2], _CALENDAR_DATES["w3"][0])
if _KNOWN_CODES != (739250, 527785, 105608):
    raise ValueError("Core calendar codes differ from the Julia fixture's known values.")


def calendar_series_cases(label: str, code: int) -> tuple[tuple[str, int, list[float]], ...]:
    """The calendar-series cases the Julia verifier expects for one family."""
    cross_year, leap, weekend, mid, far = _CALENDAR_DATES[label]
    lo, hi = CALENDAR_WINDOWS[code]
    values = [1.25, -2.5, 0.0, 4.75]
    return (
        ("cross_year", cross_year, values),
        ("leap", leap, values[:3]),
        ("weekend", weekend, values[:2]),
        ("negative", -1, values[:2]),
        ("zero", 0, values[:1]),
        ("minimum", lo, values[:1]),
        ("maximum", hi, values[:1]),
        ("beyond_py_year", far, [*values, 8.5]),
        # Only the first date is packed: trailing codes may pass the window.
        ("last_beyond_maximum", hi, values[:2]),
        ("long_span", hi - 499, list(0.25 * np.arange(1, 1001))),
        ("empty", mid, []),
        ("empty_minimum", lo, []),
        ("empty_maximum", hi, []),
    )


CALENDAR_SERIES_GROUPS = tuple(
    (label, frequency, code, calendar_series_cases(label, code))
    for (label, frequency), code in zip(CALENDAR_FAMILIES, (12, 13, *range(17, 24)), strict=True)
)


def series_cases() -> list[tuple[str, MIT, list[float]]]:
    """Every dated series written to the interchange output: name, start and values."""
    dated = [
        (f"{prefix}{anchor}_{suffix}", MIT(frequency(anchor), code), values)
        for prefix, frequency, anchors, cases in DATED_GROUPS
        for anchor in anchors
        for suffix, code, values in cases
    ]
    calendar = [
        (f"cs_{label}_{suffix}", MIT(frequency, first), values)
        for label, frequency, _, cases in CALENDAR_SERIES_GROUPS
        for suffix, first, values in cases
    ]
    return dated + calendar


def check_scalar_value(actual: object, expected: object) -> None:
    """Require the exact Python type, numerical classification and the sign of zero.

    Int64, string, MIT and Duration expectations require the exact type and an
    equal value; a floating-point detour would change integers beyond 2**53
    and is a failure, not a tolerance.
    """
    if type(expected) in (int, str, MIT, Duration):
        if type(actual) is not type(expected) or actual != expected:
            raise TypeError(f"Scalar read returned {actual!r} instead of {expected!r}.")
        return
    if type(expected) in (complex, np.complex128):
        # ComplexF64 (written as complex or np.complex128) returns a built-in
        # complex; compare component bits so NaN payloads and -0.0 count.
        if (
            type(actual) is not complex
            or np.complex128(actual).tobytes() != np.complex128(expected).tobytes()
        ):
            raise TypeError(f"Scalar read returned {actual!r} instead of {expected!r}.")
        return
    if isinstance(expected, np.generic):
        # Narrow/unsigned/complex64 widths return the exact NumPy class with identical bits.
        if type(actual) is not type(expected) or actual.tobytes() != expected.tobytes():
            raise TypeError(f"Scalar read returned {actual!r} instead of {expected!r}.")
        return
    if type(actual) is not float:
        raise TypeError("Scalar read did not return a Python float.")
    np.testing.assert_equal(actual, expected)
    if expected == 0.0 and np.signbit(actual) != np.signbit(expected):
        raise ValueError("Scalar read changed the sign of zero.")


FILEOPS_SERIES = tsecon.TSeries(mm(2024, 1), np.array([1.0, 2.0, 3.0]))


def write_file_operation_objects(db: de.DataEconFile) -> None:
    """Overwrite (Int64 -> string), delete, and replace a scalar by a series; Julia verifies."""
    db.write_scalar("fo_overwritten", 1)
    db.write_scalar("fo_overwritten", "two", overwrite=True)
    db.write_scalar("fo_deleted", 1.5)
    db.delete("fo_deleted")
    db.write_scalar("fo_series_overwritten", 1)
    db.write_series("fo_series_overwritten", FILEOPS_SERIES, overwrite=True)
    if db.is_empty():
        raise ValueError("Combined output reported an empty root catalog.")


def check_file_operation_objects(db: de.DataEconFile) -> None:
    """Re-read the file-operation outcomes through a read-only owner."""
    if db.read_scalar("fo_overwritten") != "two" or db.is_empty():
        raise ValueError("Overwrite did not replace the scalar in the combined output.")
    try:
        db.read_scalar("fo_deleted")
    except de.DataEconError as error:
        if error.code != -989:
            raise
    else:
        raise ValueError("Deleted object is still readable.")
    np.testing.assert_array_equal(
        db.read_series("fo_series_overwritten").values, FILEOPS_SERIES.values
    )


PRIMARY_OUTPUT = re.compile(r"cp\d+\.daec")


def file_operations_output(output: Path) -> Path:
    """Auxiliary file-operations output for a primary interchange output.

    The workflow's Julia steps discover the primary output by globbing
    ``*.daec`` in the interchange directory and require exactly one match, so
    every auxiliary file lives in the ``fileops`` subdirectory, which that
    glob does not descend into. The Julia verifier derives the same path.
    """
    return output.parent / "fileops" / f"{output.stem}-fileops.daec"


def check_discovery_contract(output_dir: Path) -> None:
    """Require that only primary ``cpXYZ.daec`` outputs sit in the discovery directory."""
    others = sorted(
        p.name for p in output_dir.glob("*.daec") if not PRIMARY_OUTPUT.fullmatch(p.name)
    )
    if others:
        raise ValueError(f"Non-primary outputs would break Julia output discovery: {others}")


def write_file_operations(output: Path) -> Path:
    """Exercise truncation and in-memory databases; Julia verifies the auxiliary file."""
    sibling = file_operations_output(output)
    if sibling.exists():
        raise FileExistsError(f"Use a fresh interchange output directory: {sibling}")
    sibling.parent.mkdir(parents=True, exist_ok=True)
    with de.open_dataecon(sibling, "a") as db:
        db.write_scalar("junk", 1)
        db.write_series("junk_series", FILEOPS_SERIES)
        if db.is_empty():
            raise ValueError("Populated file reported an empty root catalog.")
    with de.open_dataecon(sibling, "w") as db:  # "w" truncates the existing file
        if not db.is_empty():
            raise ValueError("Mode 'w' did not truncate the existing file.")
        db.write_scalar("after_truncate", 42)
        db.write_series("after_truncate_series", FILEOPS_SERIES)
    with de.open_dataecon(sibling) as db:
        if db.is_empty() or db.read_scalar("after_truncate") != 42:
            raise ValueError("Truncated file lost its new content.")
        try:
            db.read_scalar("junk")
        except de.DataEconError as error:
            if error.code != -989:
                raise
        else:
            raise ValueError("Truncation left the old content in place.")
    with de.open_dataecon_memory() as memory:
        memory.write_scalar("m", 1.5)
        memory.write_scalar("m", 2.5, overwrite=True)
        memory.truncate()
        if not memory.is_empty() or memory.path != ":memory:":
            raise ValueError("In-memory database did not behave as an empty scratch file.")
        memory.write_scalar("m", np.uint8(7))
        check_scalar_value(memory.read_scalar("m"), np.uint8(7))
    if output.with_name(":memory:").exists() or Path(":memory:").exists():
        raise ValueError("An in-memory database created a file on disk.")
    return sibling


def series_element_cases() -> list[tuple[str, MIT, np.ndarray]]:
    """Exact dtype, empty-marker and axis cases for the Julia series verifier."""
    cases = []
    names = (
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
    )
    dtypes = ("i1", "i2", "i4", "i8", "u1", "u2", "u4", "u8", "f2", "f4", "f8", "c8", "c16", "?")
    for name, dtype in zip(names, dtypes, strict=True):
        values = series_element_values(np.dtype(dtype))
        cases.extend(
            ((f"es_{name}", mm(2024, 11), values), (f"es_{name}_empty", mm(2024, 11), values[:0]))
        )
    families = [
        (32, tsecon.Monthly()),
        (12, tsecon.Daily()),
        (13, tsecon.BDaily()),
        *[(16 + d, tsecon.Weekly(d)) for d in range(1, 8)],
        *[(64 + m, tsecon.Quarterly(m)) for m in range(1, 4)],
        *[(128 + m, tsecon.HalfYearly(m)) for m in range(1, 7)],
        *[(256 + m, tsecon.Yearly(m)) for m in range(1, 13)],
    ]
    for code, frequency in families:
        cases.append(
            (f"es_axis_{code}", MIT(frequency, 100), np.array([-7, 0, 23], dtype=np.int16))
        )
        maximum = {12: 11979954, 13: 8557110}.get(code, 1711422 if 17 <= code <= 23 else None)
        if maximum is not None:
            cases.append(
                (f"es_trailing_{code}", MIT(frequency, maximum), np.array([-1, 1], dtype=np.int8))
            )
    return cases


def series_element_values(dtype: np.dtype) -> np.ndarray:
    """Build independent integer extrema and floating-point bit patterns."""
    if dtype.kind in "iu":
        info = np.iinfo(dtype)
        return np.array(
            [info.min, -1, 0, info.max] if dtype.kind == "i" else [0, 1, info.max], dtype=dtype
        )
    if dtype.kind == "b":
        return np.array([False, True, False])
    if dtype.kind == "c":
        return np.array([complex(-0.0, 1.25), complex(2.5, -3)], dtype=dtype)
    bits = {
        2: [0x8000, 1, 0x3D00, 0x7E55, 0x7C00],
        4: [0x80000000, 1, 0x3FA00000, 0x7FC00055, 0x7F800000],
        8: [0x8000000000000000, 1, 0x3FF4000000000000, 0x7FF8000000000055, 0x7FF0000000000000],
    }
    return np.array(bits[dtype.itemsize], dtype=f"u{dtype.itemsize}").view(dtype)


def check_series_elements(db: de.DataEconFile) -> None:
    """Check exact bytes, dtype, axis and ownership in the installed wheel."""
    for name, first, expected in series_element_cases():
        result = db.read_series(name)
        if (
            result.firstdate != first
            or result.values.dtype != expected.dtype
            or result.values.tobytes() != expected.tobytes()
            or not result.values.flags.owndata
        ):
            raise ValueError(f"Numeric/Boolean series interchange changed {name}.")


def write_series_elements(db: de.DataEconFile) -> None:
    """Write typed series into the primary interchange file."""
    for name, first, values in series_element_cases():
        db.write_series(name, tsecon.TSeries(first, values))


# Represented series elements: the Julia verifier's `represented_cases` (MIT and
# Duration elements over all 32 element frequencies with both Int64 endpoints,
# the two cross-axis cases, Int128/UInt128/ComplexF16 endpoints, word and bit
# cases and marked empties), written from StoredSeries containers.
REPRESENTED_ANCHOR = mm(2024, 11)
REPRESENTED_CODES = np.array([-(2**63), -1, 0, 2**63 - 1], dtype="<i8")
ELEMENT_FAMILIES = (
    [(11, Unit()), (12, Daily()), (13, BDaily()), (32, Monthly())]
    + [(16 + d, Weekly(d)) for d in range(1, 8)]
    + [(64 + m, Quarterly(m)) for m in range(1, 4)]
    + [(128 + m, HalfYearly(m)) for m in range(1, 7)]
    + [(256 + m, Yearly(m)) for m in range(1, 13)]
)
REPRESENTED_REWRITES = ("rs_mit_32", "rs_int128_words", "rs_complexf16_bits")
# Wide carriers with a preserved Bool marker and their explicit conversions.
REPRESENTED_BOOL = (
    ("rs_bool_int128", de.INT128, [0, 1]),
    ("rs_bool_uint128", de.UINT128, [0, 1]),
    ("rs_bool_complexf16", de.COMPLEXF16, [0j, 1 + 0j]),
)


def _float16_pairs(real_bits, imag_bits):
    return [
        (np.array([r], dtype="<u2").view("<f2")[0], np.array([i], dtype="<u2").view("<f2")[0])
        for r, i in zip(real_bits, imag_bits, strict=True)
    ]


def represented_cases() -> list[tuple[str, de.StoredSeries]]:
    """Every represented series written to the interchange output, keyed by name."""
    cases: list[tuple[str, de.StoredSeries]] = []
    for code, frequency in ELEMENT_FAMILIES:
        for tag, kind in (("mit", "date"), ("dur", "duration")):
            element = de.StoredElement(kind, frequency)
            cases.append(
                (
                    f"rs_{tag}_{code}",
                    de.StoredSeries(REPRESENTED_ANCHOR, REPRESENTED_CODES, element),
                )
            )
            cases.append(
                (
                    f"rs_{tag}_{code}_empty",
                    de.StoredSeries(REPRESENTED_ANCHOR, REPRESENTED_CODES[:0], element),
                )
            )
    cases.append(
        (
            "rs_mit_268_on_daily",
            de.StoredSeries.from_list(
                daily(dt.date(2024, 11, 1)),
                de.StoredElement.date(Yearly(12)),
                [MIT(Yearly(12), 2024), MIT(Yearly(12), 2025)],
            ),
        )
    )
    cases.append(
        (
            "rs_dur_12_on_yearly",
            de.StoredSeries.from_list(
                MIT(Yearly(12), 2024),
                de.StoredElement.duration(Daily()),
                [Duration(Daily(), -5), Duration(Daily(), 0), Duration(Daily(), 7)],
            ),
        )
    )
    int128_words = [
        2**64,
        -(2**64) - 1,
        (0x0123456789ABCDEF << 64) | 0x0FEDCBA987654321,
        -(2**127) + 1,
        2**127 - 2,
    ]
    for name, element, items in (
        ("rs_int128_endpoints", de.INT128, [-(2**127), -1, 0, 2**127 - 1]),
        ("rs_int128_words", de.INT128, int128_words),
        ("rs_uint128_endpoints", de.UINT128, [0, 2**64, 2**128 - 1]),
        ("rs_uint128_words", de.UINT128, [2**64, 2**127, 2**128 - 2]),
        (
            "rs_complexf16_bits",
            de.COMPLEXF16,
            _float16_pairs(
                [0x8000, 0x0001, 0x7E55, 0x7C00, 0xFC00, 0x7BFF],
                [0x3D00, 0x8001, 0x7E00, 0x0000, 0x7BFF, 0xFBFF],
            ),
        ),
        ("rs_Int128_empty", de.INT128, []),
        ("rs_UInt128_empty", de.UINT128, []),
        ("rs_ComplexF16_empty", de.COMPLEXF16, []),
    ):
        cases.append((name, de.StoredSeries.from_list(REPRESENTED_ANCHOR, element, items)))
    return cases


def write_represented_series(db: de.DataEconFile) -> None:
    """Write represented containers, rewrite read results and convert wide Bool carriers."""
    for name, series in represented_cases():
        db.write_series(name, series)
    for name in REPRESENTED_REWRITES:
        db.write_series(f"{name}_rewrite", db.read_series(name))
    for name, element, items in REPRESENTED_BOOL:
        marked = de.StoredSeries.from_list(REPRESENTED_ANCHOR, element.with_bool_marker(), items)
        db.write_series(name, marked)
        preserved = db.read_series(name)
        if preserved != marked:
            raise ValueError(f"Wide Bool carrier {name} did not round-trip its bytes and marker.")
        db.write_series(f"{name}_converted", preserved.to_bool())


def check_represented_series(db: de.DataEconFile) -> None:
    """Re-read the represented objects: exact container equality and ownership."""
    expected = dict(represented_cases())
    expected.update({f"{name}_rewrite": expected[name] for name in REPRESENTED_REWRITES})
    for name, series in expected.items():
        result = db.read_series(name)
        if (
            not isinstance(result, de.StoredSeries)
            or result != series
            or result.values.tobytes() != series.values.tobytes()
            or not result.values.flags.owndata
            or not result.values.flags.writeable
        ):
            raise ValueError(f"Represented series interchange changed {name}.")
    for name, element, items in REPRESENTED_BOOL:
        result = db.read_series(name)
        converted = db.read_series(f"{name}_converted")
        if (
            not isinstance(result, de.StoredSeries)
            or result.element != element.with_bool_marker()
            or result.tolist() != items
            or result.to_bool().values.tolist() != [False, True]
            or not isinstance(converted, tsecon.TSeries)
            or converted.values.dtype != np.dtype(bool)
            or converted.values.tolist() != [False, True]
        ):
            raise ValueError(f"Wide Bool interchange changed {name}.")


# Foreign reconstruction markers: the Julia verifier's `foreign_shared_cases`, written
# from preserved StoredSeries containers (stored kind, bytes and exact marker text)
# together with their explicit interpretation as "<name>_interpreted".
def _numeric(dtype, values, marker, object_marker=None):
    return de.StoredSeries(
        REPRESENTED_ANCHOR,
        np.array(values, dtype=dtype),
        de.StoredElement.numeric(dtype, marker),
        object_marker=object_marker,
    )


def _dated(kind, frequency, codes, marker, object_marker=None):
    return de.StoredSeries(
        REPRESENTED_ANCHOR,
        np.array(codes, dtype="<i8"),
        de.StoredElement(kind, frequency, marker),
        object_marker=object_marker,
    )


def foreign_cases() -> list[tuple[str, de.StoredSeries]]:
    """Every preserved foreign-marker container written to the interchange output."""
    int128 = de.INT128
    return [
        ("fx_int16_as_int64", _numeric("<i2", [1, 2], "Int64")),
        ("fx_int64_precision_as_float64", _numeric("<i8", [2**53 + 1], "Float64")),
        (
            "fx_int128_as_float64",
            de.StoredSeries.from_list(
                REPRESENTED_ANCHOR, int128.with_marker("Float64"), [2**100 + 1]
            ),
        ),
        ("fx_mit_monthly_as_bool", _dated("date", Monthly(), [0, 1], "Bool")),
        ("fx_duration_monthly_as_bool", _dated("duration", Monthly(), [0, 1], "Bool")),
        ("fx_int64_as_mit_monthly", _numeric("<i8", [1, 2], "MIT{Monthly}")),
        ("fx_int64_as_duration_daily", _numeric("<i8", [-5, 7], "Duration{Daily}")),
        ("fx_float64_as_complexf16", _numeric("<f8", [1.1, 65520.0], "ComplexF16")),
        (
            "fx_float64_as_float16",
            _numeric("<f8", [-0.0, 5e-8, 65504.0, 65520.0, 2.0**-25, 3.0 * 2.0**-25], "Float16"),
        ),
        (
            "fx_int64_midpoints_as_float32",
            _numeric("<i8", [2**54 + 2**30 + 1, 2**54 + 2**30 - 1], "Float32"),
        ),
        (
            "fx_uint64_midpoints_as_float64",
            _numeric("<u8", [2**63 + 2**10 + 1, 2**63 + 2**10 - 1, 2**64 - 1], "Float64"),
        ),
        (
            "fx_int128_midpoints_as_float32",
            de.StoredSeries.from_list(
                REPRESENTED_ANCHOR,
                int128.with_marker("Float32"),
                [2**100 + 2**76 + 1, 2**100 + 2**76 - 1],
            ),
        ),
        (
            "fx_uint128_max_as_float32",
            de.StoredSeries.from_list(
                REPRESENTED_ANCHOR, de.UINT128.with_marker("Float32"), [2**128 - 1, 2**127]
            ),
        ),
        (
            "fx_mit_monthly_as_float64",
            _dated("date", Monthly(), [-1, -13, -(2**63)], "Float64"),
        ),
        ("fx_duration_monthly_as_float64", _dated("duration", Monthly(), [-1, -13], "Float64")),
        ("fx_mit_daily_as_float32", _dated("date", Daily(), [2**54 + 2**30 + 1, -1], "Float32")),
        ("fx_mit_quarterly_as_complexf32", _dated("date", Quarterly(3), [-1, 7], "ComplexF32")),
        ("fx_duration_yearly_as_int64", _dated("duration", Yearly(12), [-(2**63), 3], "Int64")),
        (
            "fx_complexf64_negzero_as_int8",
            _numeric("<c16", [complex(1.0, -0.0), complex(0.0, -0.0)], "Int8"),
        ),
        (
            "fx_complexf16_as_int64",
            de.StoredSeries.from_list(
                REPRESENTED_ANCHOR, de.COMPLEXF16.with_marker("Int64"), [1 + 0j, 2 + 0j]
            ),
        ),
        (
            "fx_uint128_as_int128",
            de.StoredSeries.from_list(
                REPRESENTED_ANCHOR, de.UINT128.with_marker("Int128"), [2**127 - 1, 0]
            ),
        ),
        ("fx_int64_as_uint128", _numeric("<i8", [2**63 - 1, 0], "UInt128")),
        ("fx_int16_alias_as_int", _numeric("<i2", [3, 4], "Int")),
        ("fx_float64_empty_as_int16", _numeric("<f8", [], "Int16")),
        ("fx_int64_empty_as_mit_monthly", _numeric("<i8", [], "MIT{Monthly}")),
        ("fx_complexf64_empty_as_uint128", _numeric("<c16", [], "UInt128")),
        (
            "fx_outer_tseries_inactive_eltype",
            _numeric("<i2", [1, 2], "NoSuchElement", "TSeries"),
        ),
        (
            "fx_outer_tseries_bypasses_bool",
            _numeric("i1", [0, 1], "Bool", "TSeries{Monthly, Int8}"),
        ),
        (
            "fx_outer_identity_int128",
            de.StoredSeries(
                REPRESENTED_ANCHOR,
                de.StoredSeries.from_list(REPRESENTED_ANCHOR, int128, [2**100]).values,
                int128,
                object_marker="TSeries{Monthly, Int128, Vector{Int128}}",
            ),
        ),
        (
            "fx_outer_identity_mit",
            _dated("date", Monthly(), [1], "Bool", "TSeries{Monthly, MIT{Monthly}}"),
        ),
        ("fx_outer_vector_empty", _numeric("<f8", [], "Int16", "Vector")),
        ("fx_outer_vector_float64_empty", _numeric("<i8", [], None, "Vector{Float64}")),
    ]


def write_foreign_markers(db: de.DataEconFile) -> None:
    """Write each preserved container and its explicit interpretation."""
    for name, series in foreign_cases():
        db.write_series(name, series)
        result = db.read_series(name)
        if result != series:
            raise ValueError(f"Foreign marker {name} did not round-trip its storage and markers.")
        interpreted = result.to_interpreted()
        if isinstance(interpreted, np.ndarray):
            # The Vector object markers interpret to an empty array; Julia's own
            # writer stores such a value as an empty typed series.
            interpreted = tsecon.TSeries(series.firstdate, interpreted)
        db.write_series(f"{name}_interpreted", interpreted)


ARRAY_DTYPES = (
    np.dtype("i1"),
    np.dtype("<i2"),
    np.dtype("<i4"),
    np.dtype("<i8"),
    np.dtype("u1"),
    np.dtype("<u2"),
    np.dtype("<u4"),
    np.dtype("<u8"),
    np.dtype("<f2"),
    np.dtype("<f4"),
    np.dtype("<f8"),
    np.dtype("<c8"),
    np.dtype("<c16"),
    np.dtype("?"),
)


def array_token(dtype: np.dtype) -> str:
    """Return the exact Julia spelling used by the installed verifier."""
    if dtype.kind == "b":
        return "Bool"
    if dtype.kind == "c":
        return "ComplexF32" if dtype.itemsize == 8 else "ComplexF64"
    return dtype.name.title().replace("Uint", "UInt")


def array_values(dtype: np.dtype) -> np.ndarray:  # noqa: PLR0911 - finite dtype table
    """Representative values including exact floating bit patterns."""
    if dtype.kind == "i":
        info = np.iinfo(dtype)
        return np.array([info.min, -1, 0, info.max], dtype=dtype)
    if dtype.kind == "u":
        info = np.iinfo(dtype)
        return np.array([0, 1, info.max], dtype=dtype)
    if dtype == np.dtype("<f2"):
        return np.array([0x8000, 1, 0x3D00, 0x7E55, 0x7C00], dtype="<u2").view(dtype)
    if dtype == np.dtype("<f4"):
        return np.array([0x80000000, 1, 0x3FA00000, 0x7FC00055, 0x7F800000], dtype="<u4").view(
            dtype
        )
    if dtype == np.dtype("<f8"):
        return np.array(
            [0x8000000000000000, 1, 0x3FF4000000000000, 0x7FF8000000000055, 0x7FF0000000000000],
            dtype="<u8",
        ).view(dtype)
    if dtype.kind == "c":
        return np.array([complex(-0.0, 1.25), complex(2.5, -3.0)], dtype=dtype)
    return np.array([False, True, False], dtype=bool)


def range_families() -> list[tuple[str, Frequency, int, int]]:
    """Every canonical MIT frequency with its reliable first-code window."""
    families: list[tuple[str, Frequency, int, int]] = [
        ("u", Unit(), -(2**63), 2**63 - 1),
        ("d", Daily(), -11980259, 11979954),
        ("b", BDaily(), -8557114, 8557110),
        ("m", Monthly(), -393600, 2**31 - 1),
    ]
    families.extend((f"w{day}", Weekly(day), -1711422, 1711422) for day in range(1, 8))
    families.extend((f"q{month}", Quarterly(month), -131200, 2**31 - 1) for month in range(1, 4))
    families.extend((f"h{month}", HalfYearly(month), -65600, 2**31 - 1) for month in range(1, 7))
    families.extend((f"y{month}", Yearly(month), -(2**31), 2**31 - 1) for month in range(1, 13))
    return families


def write_arrays_unit(db: de.DataEconFile) -> None:
    """Write every supported plain vector dtype, lossless ranges and Unit series."""
    for dtype in ARRAY_DTYPES:
        token = array_token(dtype)
        values = array_values(dtype)
        db.write_array(f"array_{token}", values)
        db.write_array(f"array_{token}_empty", np.empty(0, dtype=dtype))
        db.write_series(f"unit_{token}", tsecon.TSeries(MIT(Unit(), -2), values))
        db.write_series(
            f"unit_{token}_empty", tsecon.TSeries(MIT(Unit(), -2), np.empty(0, dtype=dtype))
        )
    db.write_array("range_int", range(1, 6))
    db.write_array("range_int_empty", range(1, 1))
    db.write_array("range_monthly", tsecon.MITRange(MIT(Monthly(), -3), MIT(Monthly(), 1)))
    db.write_array("range_unit", tsecon.MITRange(MIT(Unit(), -3), MIT(Unit(), 1)))
    db.write_array("range_unit_empty", tsecon.MITRange(MIT(Unit(), 5), MIT(Unit(), 4)))
    for label, frequency, minimum, maximum in range_families():
        db.write_array(f"range_all_{label}", tsecon.MITRange(MIT(frequency, 0), MIT(frequency, 1)))
        db.write_array(
            f"range_min_{label}", tsecon.MITRange(MIT(frequency, minimum), MIT(frequency, minimum))
        )
        db.write_array(
            f"range_max_{label}", tsecon.MITRange(MIT(frequency, maximum), MIT(frequency, maximum))
        )
    db.write_series("unit_min", tsecon.TSeries(MIT(Unit(), -(2**63)), [1.0]))
    db.write_series("unit_max", tsecon.TSeries(MIT(Unit(), 2**63 - 1), [1.0]))
    db.write_series("unit_max_span", tsecon.TSeries(MIT(Unit(), 2**63 - 2), [1.0, 2.0]))
    db.write_series(
        "unit_mit_monthly",
        de.StoredSeries.from_list(
            MIT(Unit(), -2),
            de.StoredElement.date(Monthly()),
            [MIT(Monthly(), -1), MIT(Monthly(), 0)],
        ),
    )
    db.write_series(
        "unit_int128",
        de.StoredSeries.from_list(MIT(Unit(), -2), de.INT128, [-(2**127), 2**127 - 1]),
    )
    db.write_series(
        "unit_complexf16",
        de.StoredSeries.from_list(MIT(Unit(), -2), de.COMPLEXF16, [complex(-0.0, 1.25)]),
    )


def check_arrays_unit(db: de.DataEconFile) -> None:  # noqa: PLR0912 - finite ABI matrix
    """Check owning byte-exact reads before Julia independently verifies the file."""
    for dtype in ARRAY_DTYPES:
        token = array_token(dtype)
        for suffix, expected in (("", array_values(dtype)), ("_empty", np.empty(0, dtype=dtype))):
            values = db.read_array(f"array_{token}{suffix}")
            if values.dtype != dtype or values.tobytes() != expected.tobytes():
                raise ValueError(f"Plain array {token}{suffix} changed dtype or bytes.")
            if not values.flags.owndata or not values.flags.writeable:
                raise ValueError(f"Plain array {token}{suffix} is not owning and writable.")
            series = db.read_series(f"unit_{token}{suffix}")
            if (
                series.firstdate != MIT(Unit(), -2)
                or series.values.dtype != dtype
                or series.values.tobytes() != expected.tobytes()
                or not series.values.flags.owndata
            ):
                raise ValueError(f"Unit series {token}{suffix} changed its representation.")
    expected_ranges = {
        "range_int": range(1, 6),
        "range_int_empty": range(1, 1),
        "range_monthly": tsecon.MITRange(MIT(Monthly(), -3), MIT(Monthly(), 1)),
        "range_unit": tsecon.MITRange(MIT(Unit(), -3), MIT(Unit(), 1)),
        "range_unit_empty": tsecon.MITRange(MIT(Unit(), 5), MIT(Unit(), 4)),
    }
    for name, expected in expected_ranges.items():
        if db.read_array(name) != expected:
            raise ValueError(f"Range {name} changed its values.")
    for label, frequency, minimum, maximum in range_families():
        for part, first, length in (("all", 0, 2), ("min", minimum, 1), ("max", maximum, 1)):
            expected = tsecon.MITRange(MIT(frequency, first), MIT(frequency, first + length - 1))
            if db.read_array(f"range_{part}_{label}") != expected:
                raise ValueError(f"MIT range {part}/{label} changed its frequency or codes.")
    represented = {
        "unit_mit_monthly": [MIT(Monthly(), -1), MIT(Monthly(), 0)],
        "unit_int128": [-(2**127), 2**127 - 1],
        "unit_complexf16": [complex(-0.0, 1.25)],
    }
    for name, expected in represented.items():
        value = db.read_series(name)
        if not isinstance(value, de.StoredSeries) or value.firstdate != MIT(Unit(), -2):
            raise ValueError(f"Represented Unit series {name} lost its stored form.")
        actual = value.tolist()
        if name == "unit_complexf16":
            if actual[0].imag != 1.25 or not np.signbit(actual[0].real):
                raise ValueError("ComplexF16 Unit series lost a component or signed zero.")
        elif actual != expected:
            raise ValueError(f"Represented Unit series {name} changed values.")


def check_foreign_markers(db: de.DataEconFile) -> None:
    """Re-read the preserved objects: exact equality, ownership, unchanged interpretation."""
    for name, series in foreign_cases():
        result = db.read_series(name)
        if (
            not isinstance(result, de.StoredSeries)
            or result != series
            or result.values.tobytes() != series.values.tobytes()
            or result.element.marker != series.element.marker
            or result.object_marker != series.object_marker
            or not result.values.flags.owndata
            or not result.values.flags.writeable
        ):
            raise ValueError(f"Foreign marker interchange changed {name}.")
        interpreted = result.to_interpreted()
        again = db.read_series(f"{name}_interpreted")
        if isinstance(interpreted, np.ndarray):
            expected_dtype = interpreted.dtype
            if not isinstance(again, tsecon.TSeries) or again.values.dtype != expected_dtype:
                raise ValueError(f"Vector interpretation of {name} changed.")
            continue
        if type(again) is not type(interpreted) or again.values.dtype != interpreted.values.dtype:
            raise ValueError(f"Interpretation of {name} changed its type.")
        if isinstance(interpreted, de.StoredSeries):
            if again != interpreted:
                raise ValueError(f"Interpretation of {name} changed its stored value.")
        elif again.values.tobytes() != interpreted.values.tobytes():
            raise ValueError(f"Interpretation of {name} changed its values.")


def matrix_values(dtype: np.dtype) -> np.ndarray:
    """Six values shaped 2x3 column-major, matching the Julia fixture exactly.

    These are the same six values `matrix_text_values` builds in the fixture
    generator, so Julia's verifier compares like with like: signed zeros, NaN
    payloads, infinities, subnormals and both integer extremes.
    """
    kind = np.dtype(dtype).kind
    if kind == "b":
        flat = np.array([False, True, False, True, True, False])
    elif kind == "i":
        info = np.iinfo(dtype)
        flat = np.array([info.min, -1, 0, info.max, 7, 9], dtype=dtype)
    elif kind == "u":
        info = np.iinfo(dtype)
        flat = np.array([0, 1, info.max, 7, 9, 11], dtype=dtype)
    elif kind == "f":
        width = np.dtype(dtype).itemsize
        bits = {2: "<u2", 4: "<u4", 8: "<u8"}[width]
        raw = {
            2: [0x8000, 0x0001, 0x3D00, 0x7E55, 0x7C00, 0x0002],
            4: [0x80000000, 1, 0x3FA00000, 0x7FC00055, 0x7F800000, 2],
            8: [
                0x8000000000000000,
                1,
                0x3FF4000000000000,
                0x7FF8000000000055,
                0x7FF0000000000000,
                2,
            ],
        }[width]
        flat = np.array(raw, dtype=bits).view(dtype)
    else:
        component = "<f4" if np.dtype(dtype).itemsize == 8 else "<f8"
        real = np.array([-0.0, 2.5, 0.0, 1.0, 3.0, 5.0], dtype=component)
        imag = np.array([1.25, -3.0, -0.0, 2.0, 4.0, 6.0], dtype=component)
        flat = np.empty(6, dtype=dtype)
        flat.real, flat.imag = real, imag
    return np.asarray(flat).reshape((2, 3), order="F").copy()


def matrix_axis_families() -> list[tuple[str, Frequency, int, int]]:
    """The axis families the MVTSeries fixture anchors at both endpoints."""
    return [
        ("u", Unit(), -(2**63), 2**63 - 1),
        ("d", Daily(), -11980259, 11979954),
        ("b", BDaily(), -8557114, 8557110),
        ("w7", Weekly(7), -1711422, 1711422),
        ("m", Monthly(), -393600, 2147483647),
        ("q1", Quarterly(1), -131200, 2147483647),
        ("h1", HalfYearly(1), -65600, 2147483647),
        ("y1", Yearly(1), -(2**31), 2**31 - 1),
    ]


def write_matrices_text(db: de.DataEconFile) -> None:
    """Write plain matrices, structure markers, MVTSeries and packed text vectors."""
    for dtype in ARRAY_DTYPES:
        token = array_token(dtype)
        db.write_array(f"matrix_{token}", matrix_values(dtype))
        db.write_array(f"matrix_{token}_empty", np.empty((0, 0), dtype=dtype))
    db.write_array("matrix_zero_rows", np.empty((0, 3), dtype=np.float64))
    db.write_array("matrix_zero_cols", np.empty((3, 0), dtype=np.float64))
    db.write_array("matrix_1x1", np.array([[2.5]], dtype=np.float64))
    db.write_array(
        "matrix_mit",
        de.StoredArray(np.array([[-1, 1], [0, 2]], dtype="<i8"), de.StoredElement.date(Monthly())),
    )
    db.write_array(
        "matrix_duration",
        de.StoredArray(
            np.array([[1, 3], [2, 4]], dtype="<i8"), de.StoredElement.duration(Monthly())
        ),
    )
    db.write_array("matrix_int128", stored_matrix_128(de.INT128, [-(2**127), -1, 0, 2**127 - 1]))
    db.write_array("matrix_uint128", stored_matrix_128(de.UINT128, [0, 1, 2, 2**128 - 1]))
    db.write_array("matrix_complexf16", stored_complexf16_matrix())
    db.write_array(
        "matrix_diagonal",
        de.StoredArray(
            np.array([[3.0, 0.0], [0.0, 5.0]]),
            de.StoredElement.numeric("<f8"),
            object_marker="Diagonal",
        ),
    )
    db.write_array(
        "matrix_symmetric",
        de.StoredArray(
            np.array([[1.0, 3.0], [3.0, 4.0]]),
            de.StoredElement.numeric("<f8"),
            object_marker="Symmetric",
        ),
    )
    db.write_array(
        "matrix_hermitian",
        de.StoredArray(
            np.array([[1 + 0j, 2 + 1j], [2 - 1j, 4 + 0j]], dtype="<c16"),
            de.StoredElement.numeric("<c16"),
            object_marker="Hermitian",
        ),
    )
    db.write_series(
        "mvts_float64",
        tsecon.MVTSeries(mm(2024, 1), ("a", "b"), np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])),
    )
    db.write_series(
        "mvts_int64",
        tsecon.MVTSeries(
            MIT(Quarterly(3), 8080), ("x", "y"), np.array([[1, 3], [2, 4]], dtype=np.int64)
        ),
    )
    db.write_series(
        "mvts_bool",
        tsecon.MVTSeries(mm(2024, 1), ("a", "b"), np.array([[True, False], [False, True]])),
    )
    db.write_series(
        "mvts_one_col", tsecon.MVTSeries(mm(2024, 1), ("only",), np.array([[1.0], [2.0]]))
    )
    db.write_series(
        "mvts_unicode",
        tsecon.MVTSeries(
            mm(2024, 1), ("a\u00e9", "\U0001f642"), np.array([[1.0, 3.0], [2.0, 4.0]])
        ),
    )
    db.write_series(
        "mvts_empty_name",
        tsecon.MVTSeries(mm(2024, 1), ("", "b"), np.array([[1.0, 3.0], [2.0, 4.0]])),
    )
    db.write_series("mvts_zero_rows", tsecon.MVTSeries(mm(2024, 1), ("a", "b"), np.empty((0, 2))))
    for label, frequency, minimum, maximum in matrix_axis_families():
        for suffix, code in (("min", minimum), ("max", maximum)):
            db.write_series(
                f"mvts_{suffix}_{label}",
                tsecon.MVTSeries(MIT(frequency, code), ("a",), np.array([[1.0]])),
            )
    db.write_array("text_ascii", ["alpha", "", "z"])
    db.write_array("text_symbol", de.StoredText.from_list(["alpha", "", "z"], "Symbol"))
    db.write_array("text_empty", [])
    db.write_array("text_multibyte", ["\u00e9", "\U0001f642", "a\u00e9"])
    db.write_array(
        "text_multibyte_symbol",
        de.StoredText.from_list(["\u00e9", "\U0001f642", "a\u00e9"], "Symbol"),
    )
    db.write_array("text_one_empty_string", [""])


def stored_matrix_128(element: de.StoredElement, values: list[int]) -> de.StoredArray:
    """Pack four 128-bit values column-major into a 2x2 carrier.

    Both words are unsigned: a negative Int128's low word does not fit a signed
    64-bit integer, and the carrier stores the two-word bit pattern either way.
    """
    words: list[int] = []
    for value in values:
        raw = value & ((1 << 128) - 1)
        words.extend([raw & ((1 << 64) - 1), raw >> 64])
    packed = np.array(words, dtype="<u8").view(element.dtype)
    return de.StoredArray(packed.reshape((2, 2), order="F").copy(), element)


def stored_complexf16_matrix() -> de.StoredArray:
    """A 1x2 ComplexF16 carrier matching the Julia fixture's bit patterns."""
    parts = np.array([0x8000, 0x3D00, 0x4000, 0x4200], dtype="<u2").view(de.COMPLEXF16.dtype)
    return de.StoredArray(parts.reshape((1, 2), order="F").copy(), de.COMPLEXF16)


def check_matrices_text(db: de.DataEconFile) -> None:  # noqa: PLR0912 - finite ABI matrix
    """Read every object written by :func:`write_matrices_text` back."""
    for dtype in ARRAY_DTYPES:
        token = array_token(dtype)
        expected = matrix_values(dtype)
        actual = db.read_array(f"matrix_{token}")
        if not isinstance(actual, np.ndarray) or actual.dtype != np.dtype(dtype):
            raise TypeError(f"matrix_{token} must read back as its own dtype.")
        if actual.tobytes(order="C") != expected.tobytes(order="C"):
            raise ValueError(f"matrix_{token} values changed across the round trip.")
        if not (actual.flags["C_CONTIGUOUS"] and actual.flags["WRITEABLE"]):
            raise ValueError(f"matrix_{token} must own writable contiguous storage.")
        empty = db.read_array(f"matrix_{token}_empty")
        if empty.shape != (0, 0) or empty.dtype != np.dtype(dtype):
            raise ValueError(f"matrix_{token}_empty lost its shape or dtype.")
    if db.read_array("matrix_zero_rows").shape != (0, 3):
        raise ValueError("A zero-row matrix must keep its column count.")
    if db.read_array("matrix_zero_cols").shape != (3, 0):
        raise ValueError("A zero-column matrix must keep its row count.")
    for name, token, expected in (
        ("matrix_diagonal", "Diagonal", [[3.0, 0.0], [0.0, 5.0]]),
        ("matrix_symmetric", "Symmetric", [[1.0, 3.0], [3.0, 4.0]]),
    ):
        value = db.read_array(name)
        if not isinstance(value, de.StoredArray) or value.object_marker != token:
            raise TypeError(f"{name} must preserve its structural marker.")
        if value.to_interpreted().tolist() != expected:
            raise ValueError(f"{name} did not reconstruct Julia's dense value.")
    for name in (
        "matrix_mit",
        "matrix_duration",
        "matrix_int128",
        "matrix_uint128",
        "matrix_complexf16",
    ):
        value = db.read_array(name)
        if not isinstance(value, de.StoredArray) or value.ndim != 2:
            raise TypeError(f"{name} must read back as a two-dimensional StoredArray.")
    for name, first, names, expected in (
        ("mvts_float64", mm(2024, 1), ("a", "b"), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
        ("mvts_int64", MIT(Quarterly(3), 8080), ("x", "y"), [[1, 3], [2, 4]]),
        ("mvts_one_col", mm(2024, 1), ("only",), [[1.0], [2.0]]),
        ("mvts_unicode", mm(2024, 1), ("a\u00e9", "\U0001f642"), [[1.0, 3.0], [2.0, 4.0]]),
        ("mvts_empty_name", mm(2024, 1), ("", "b"), [[1.0, 3.0], [2.0, 4.0]]),
    ):
        value = db.read_series(name)
        if not isinstance(value, tsecon.MVTSeries):
            raise TypeError(f"{name} must read back as an MVTSeries.")
        if value.firstdate != first or tuple(value.columns) != names:
            raise ValueError(f"{name} lost its anchor or column names.")
        if value.values.tolist() != expected:
            raise ValueError(f"{name} values changed across the round trip.")
    if db.read_series("mvts_bool").values.dtype != np.dtype(bool):
        raise TypeError("A Boolean MVTSeries must read back with a bool dtype.")
    if db.read_series("mvts_zero_rows").shape != (0, 2):
        raise ValueError("A zero-row MVTSeries must keep its columns.")
    for label, frequency, minimum, maximum in matrix_axis_families():
        for suffix, code in (("min", minimum), ("max", maximum)):
            value = db.read_series(f"mvts_{suffix}_{label}")
            if value.firstdate != MIT(frequency, code):
                raise ValueError(f"mvts_{suffix}_{label} lost its axis anchor.")
    if db.read_array("text_ascii") != ["alpha", "", "z"]:
        raise ValueError("An ASCII text vector must read back as plain strings.")
    if db.read_array("text_multibyte") != ["\u00e9", "\U0001f642", "a\u00e9"]:
        raise ValueError("A multibyte text vector must read back intact.")
    if db.read_array("text_empty") != []:
        raise ValueError("An empty text vector must read back empty.")
    if db.read_array("text_one_empty_string") != [""]:
        raise ValueError("One empty string is not an empty vector.")
    for name in ("text_symbol", "text_multibyte_symbol"):
        value = db.read_array(name)
        if not isinstance(value, de.StoredText) or value.marker != "Symbol":
            raise TypeError(f"{name} must preserve its Symbol marker.")


def tensor_values(dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    """The six matrix values laid out column-major in a three- to five-dimensional shape."""
    flat = matrix_values(dtype).reshape(-1, order="F")
    return flat.reshape(shape, order="F").copy()


def tensor_run(shape: tuple[int, ...]) -> np.ndarray:
    """The column-major run 1..24 in a shape whose bytes never change."""
    return np.arange(1, 25, dtype="<i8").reshape(shape, order="F").copy()


def stored_words_128(
    element: de.StoredElement, values: list[int], shape: tuple[int, ...]
) -> de.StoredArray:
    """Pack 128-bit values column-major into a carrier of any supported rank."""
    words: list[int] = []
    for value in values:
        raw = value & ((1 << 128) - 1)
        words.extend([raw & ((1 << 64) - 1), raw >> 64])
    packed = np.array(words, dtype="<u8").view(element.dtype)
    return de.StoredArray(packed.reshape(shape, order="F").copy(), element)


def stored_complexf16_tensor(shape: tuple[int, ...]) -> de.StoredArray:
    """The fixture's two ComplexF16 bit patterns in a three- or four-dimensional carrier."""
    parts = np.array([0x8000, 0x3D00, 0x4000, 0x4200], dtype="<u2").view(de.COMPLEXF16.dtype)
    return de.StoredArray(parts.reshape(shape, order="F").copy(), de.COMPLEXF16)


def bit_array(values: np.ndarray, token: str) -> de.StoredArray:
    """A Julia BitArray as Python preserves it: Int8 zero/one, Bool element, rank token."""
    carrier = np.asarray(values, dtype=bool).astype(np.int8)
    element = de.StoredElement.numeric(np.dtype("i1"), "Bool")
    return de.StoredArray(carrier, element, object_marker=token)


def write_tensors(db: de.DataEconFile) -> None:
    """Write three- to five-dimensional plain arrays over every supported element."""
    for dtype in ARRAY_DTYPES:
        token = array_token(dtype)
        db.write_array(f"tensor_{token}", tensor_values(dtype, (1, 2, 3)))
        db.write_array(f"tensor5_{token}", tensor_values(dtype, (3, 1, 2, 1, 1)))
        db.write_array(f"tensor_{token}_empty", np.empty((0, 2, 3), dtype=dtype))
    for dtype in (np.dtype("<i8"), np.dtype("<f8"), np.dtype("?"), np.dtype("u1")):
        db.write_array(f"tensor4_{array_token(dtype)}", tensor_values(dtype, (1, 2, 3, 1)))
    db.write_array("tensor_empty_2x0x3", np.empty((2, 0, 3), dtype=np.float64))
    db.write_array("tensor_empty_2x3x0", np.empty((2, 3, 0), dtype=np.float64))
    db.write_array("tensor_empty_rank5", np.empty((0, 0, 0, 0, 0), dtype=np.float64))
    db.write_array("tensor_empty_rank4_bool", np.empty((0, 1, 1, 1), dtype=bool))
    for shape in ((2, 3, 4), (4, 3, 2), (2, 2, 2, 3), (2, 2, 2, 3, 1)):
        db.write_array("tensor_run_" + "x".join(map(str, shape)), tensor_run(shape))
    db.write_array("tensor_1x1x1", np.array([[[2.5]]], dtype=np.float64))
    db.write_array("tensor_1x1x1x1x1", np.array([42], dtype="<i8").reshape((1, 1, 1, 1, 1)))
    db.write_array(
        "tensor_mit",
        de.StoredArray(
            np.array([-1, 0, 1, 2], dtype="<i8").reshape((2, 1, 2), order="F"),
            de.StoredElement.date(Monthly()),
        ),
    )
    db.write_array(
        "tensor_mit_unit5",
        de.StoredArray(
            np.array([-(2**63), 2**63 - 1], dtype="<i8").reshape((1, 2, 1, 1, 1)),
            de.StoredElement.date(Unit()),
        ),
    )
    db.write_array(
        "tensor_duration_q1",
        de.StoredArray(
            np.array([1, -2, 2**63 - 1, 0], dtype="<i8").reshape((1, 2, 2), order="F"),
            de.StoredElement.duration(Quarterly(1)),
        ),
    )
    db.write_array(
        "tensor_int128", stored_words_128(de.INT128, [-(2**127), -1, 0, 2**127 - 1], (2, 1, 2))
    )
    db.write_array("tensor_uint128", stored_words_128(de.UINT128, [0, 1, 2, 2**128 - 1], (1, 2, 2)))
    db.write_array("tensor_complexf16", stored_complexf16_tensor((1, 1, 2)))
    db.write_array("tensor_complexf16_4d", stored_complexf16_tensor((2, 1, 1, 1)))
    db.write_array(
        "tensor_bitarray", bit_array(tensor_values(np.dtype("?"), (1, 2, 3)), "BitArray{3}")
    )
    db.write_array("bit_vector", bit_array(np.array([True, False, True]), "BitVector"))
    db.write_array("bit_matrix", bit_array(np.array([[True, False], [False, True]]), "BitMatrix"))
    cube = np.arange(1, 9, dtype="<i8").reshape((2, 2, 2), order="F")
    db.write_array(
        "tensor_marked_float64",
        de.StoredArray(cube, de.StoredElement.numeric(np.dtype("<i8"), "Float64")),
    )
    db.write_array(
        "tensor_object_float64",
        de.StoredArray(
            cube, de.StoredElement.numeric(np.dtype("<i8"), None), object_marker="Array{Float64,3}"
        ),
    )
    db.write_array(
        "tensor_object_identity",
        de.StoredArray(
            cube, de.StoredElement.numeric(np.dtype("<i8"), None), object_marker="Array"
        ),
    )
    db.write_array(
        "tensor_marked_bool_wide",
        de.StoredArray(
            np.array([0, 0, 1, 0], dtype="<u8").view(de.INT128.dtype).reshape((1, 2, 1)),
            de.INT128.with_bool_marker(),
        ),
    )


def check_tensors(db: de.DataEconFile) -> None:  # noqa: PLR0912 - finite ABI matrix
    """Read every object written by :func:`write_tensors` back."""
    for dtype in ARRAY_DTYPES:
        token = array_token(dtype)
        for name, shape in ((f"tensor_{token}", (1, 2, 3)), (f"tensor5_{token}", (3, 1, 2, 1, 1))):
            expected = tensor_values(dtype, shape)
            actual = db.read_array(name)
            if not isinstance(actual, np.ndarray) or actual.dtype != np.dtype(dtype):
                raise TypeError(f"{name} must read back as its own dtype.")
            if actual.shape != shape or actual.tobytes(order="C") != expected.tobytes(order="C"):
                raise ValueError(f"{name} values or shape changed across the round trip.")
            if not (actual.flags["C_CONTIGUOUS"] and actual.flags["WRITEABLE"]):
                raise ValueError(f"{name} must own writable contiguous storage.")
        empty = db.read_array(f"tensor_{token}_empty")
        if empty.shape != (0, 2, 3) or empty.dtype != np.dtype(dtype):
            raise ValueError(f"tensor_{token}_empty lost its shape or dtype.")
    for name, shape in (
        ("tensor_empty_2x0x3", (2, 0, 3)),
        ("tensor_empty_2x3x0", (2, 3, 0)),
        ("tensor_empty_rank5", (0, 0, 0, 0, 0)),
        ("tensor_empty_rank4_bool", (0, 1, 1, 1)),
    ):
        if db.read_array(name).shape != shape:
            raise ValueError(f"{name} lost its stored shape.")
    for shape in ((2, 3, 4), (4, 3, 2), (2, 2, 2, 3), (2, 2, 2, 3, 1)):
        name = "tensor_run_" + "x".join(map(str, shape))
        actual = db.read_array(name)
        if actual.shape != shape or not np.array_equal(actual, tensor_run(shape)):
            raise ValueError(f"{name} did not keep the column-major run.")
    if db.read_array("tensor_1x1x1").tolist() != [[[2.5]]]:
        raise ValueError("A 1x1x1 tensor changed.")
    if db.read_array("tensor_1x1x1x1x1").shape != (1, 1, 1, 1, 1):
        raise ValueError("A rank-5 singleton lost its shape.")
    for name, ndim in (
        ("tensor_mit", 3),
        ("tensor_mit_unit5", 5),
        ("tensor_duration_q1", 3),
        ("tensor_int128", 3),
        ("tensor_uint128", 3),
        ("tensor_complexf16", 3),
        ("tensor_complexf16_4d", 4),
    ):
        value = db.read_array(name)
        if not isinstance(value, de.StoredArray) or value.ndim != ndim:
            raise TypeError(f"{name} must read back as a {ndim}-dimensional StoredArray.")
    for name, token, expected in (
        ("tensor_bitarray", "BitArray{3}", tensor_values(np.dtype("?"), (1, 2, 3))),
        ("bit_vector", "BitVector", np.array([True, False, True])),
        ("bit_matrix", "BitMatrix", np.array([[True, False], [False, True]])),
    ):
        value = db.read_array(name)
        if not isinstance(value, de.StoredArray) or value.object_marker != token:
            raise TypeError(f"{name} must preserve Julia's {token} token.")
        if not np.array_equal(value.to_interpreted(), expected):
            raise ValueError(f"{name} did not interpret to its Boolean values.")
    cube = np.arange(1, 9, dtype="<i8").reshape((2, 2, 2), order="F")
    for name, marker, object_marker in (
        ("tensor_marked_float64", "Float64", None),
        ("tensor_object_float64", None, "Array{Float64,3}"),
        ("tensor_object_identity", None, "Array"),
    ):
        value = db.read_array(name)
        if not isinstance(value, de.StoredArray):
            raise TypeError(f"{name} must preserve its marker in a StoredArray.")
        if value.element.marker != marker or value.object_marker != object_marker:
            raise ValueError(f"{name} lost a reconstruction marker.")
        interpreted = value.to_interpreted()
        if interpreted.shape != (2, 2, 2) or not np.array_equal(interpreted, cube):
            raise ValueError(f"{name} did not interpret to Julia's values.")
    wide = db.read_array("tensor_marked_bool_wide")
    if not isinstance(wide, de.StoredArray) or wide.element.marker != "Bool":
        raise TypeError("A wide Bool tensor must keep its carrier and marker.")
    if wide.to_bool().tolist() != [[[False], [True]]]:
        raise ValueError("A wide Bool tensor did not convert to its flags.")


# Catalogs, nested paths and attributes: the tree Julia's generate-catalogs
# writes at the root of its fixture, written here under /catalogs so the wheel
# output keeps its root-level objects. Julia's verify-wheel reads it back.
CATALOG_ORDER_NAMES = ["b", "B", "a", "1", " sp", "_u", "\u00e4", "Z"]
CATALOG_ATTRIBUTES = [
    ("note", "second"),
    ("empty", ""),
    ("", "empty name"),
    ("unicod\u00e9/\u540d ", "v\u00e4lue \U0001f642\nline\ttab"),
    ("delim", "a\u2016b"),
    ("unit", "a\x1fb"),
    ("k\u20163", "v3"),
    ("run\x1e\x1f\x1f", "r"),
    ("long", "x" * 5000),
]
CATALOG_DATES = np.array([24288, 24289], dtype="<i8")


def write_catalogs(db: de.DataEconFile) -> None:
    prefix = "/catalogs"
    db.new_catalog(prefix)
    db.new_catalog(f"{prefix}/cat")
    db.new_catalog(f"{prefix}/cat/sub")
    db.new_catalog(f"{prefix}/cat/sub/deep")
    db.new_catalog(f"{prefix}/cat/\u65e5\u672c\u8a9e")
    db.write_scalar(f"{prefix}/cat/scalar", 1.5)
    db.write_scalar(f"{prefix}/cat/sub/text", "vintage \u2016 3")
    db.write_array(f"{prefix}/cat/sub/vector", np.array([1, 2, 3], dtype=np.int32))
    db.write_array(f"{prefix}/cat/sub/matrix", np.array([[1.0, 2.0], [3.0, 4.0]]))
    db.write_array(
        f"{prefix}/cat/sub/deep/tensor",
        np.arange(1, 25, dtype=np.int64).reshape((2, 3, 4), order="F"),
    )
    db.write_series(
        f"{prefix}/cat/sub/deep/series", tsecon.TSeries(mm(2024, 1), np.array([1.0, 2.0, 3.0]))
    )
    db.write_series(
        f"{prefix}/cat/sub/mvtseries",
        tsecon.MVTSeries(qq(2024, 1), ("p", "q"), np.array([[1.0, 2.0], [3.0, 4.0]])),
    )
    db.write_series(f"{prefix}/cat/sub/bools", tsecon.TSeries(mm(2024, 1), np.array([True, False])))
    db.write_array(f"{prefix}/cat/sub/text_vector", ["a", "b"])
    db.write_series(
        f"{prefix}/cat/\u65e5\u672c\u8a9e/dates",
        de.StoredSeries(mm(2024, 1), CATALOG_DATES, de.StoredElement.date(Monthly())),
    )
    db.write_array(
        f"{prefix}/cat/\u65e5\u672c\u8a9e/date_array",
        de.StoredArray(CATALOG_DATES.reshape((1, 2)), de.StoredElement.date(Monthly())),
    )
    for name in CATALOG_ORDER_NAMES:
        db.write_scalar(f"{prefix}/cat/{name}", 1)
    db.write_scalar(f"{prefix}/cat/scalar_parent", 2)
    try:
        db.write_scalar(f"{prefix}/cat/scalar_parent/child", 3)
    except ValueError:
        pass
    else:
        raise ValueError("A scalar must not accept children.")
    db.set_attribute(f"{prefix}/cat/scalar", "note", "first")
    for name, value in CATALOG_ATTRIBUTES:
        db.set_attribute(f"{prefix}/cat/scalar", name, value)
    db.set_attribute(f"{prefix}/cat", "owner", "julia")
    db.set_attribute("/", "root_note", "hello")


def check_catalogs(db: de.DataEconFile) -> None:
    """Read every object written by :func:`write_catalogs` back."""
    prefix = "/catalogs"
    if db.catalog_size(f"{prefix}/cat") != 4 + len(CATALOG_ORDER_NAMES):
        raise ValueError("Unexpected catalog size.")
    listed = [entry.name for entry in db.list_objects(f"{prefix}/cat")]
    if listed != sorted(listed, key=lambda n: n.encode("utf-8")) or len(listed) != 12:
        raise ValueError("Catalog listing is not in UTF-8 byte order.")
    paths = [entry.path for entry in db.list_objects(prefix, recursive=True)]
    if f"{prefix}/cat/sub/deep/tensor" not in paths or f"{prefix}/cat/scalar_parent/child" in paths:
        raise ValueError("Recursive listing is wrong.")
    if (
        db.read_scalar(f"{prefix}/cat/scalar") != 1.5
        or db.read_scalar(f"{prefix}/cat/sub/text") != "vintage \u2016 3"
    ):
        raise ValueError("Nested scalar mismatch.")
    np.testing.assert_array_equal(
        db.read_array(f"{prefix}/cat/sub/deep/tensor"),
        np.arange(1, 25, dtype=np.int64).reshape((2, 3, 4), order="F"),
    )
    series = db.read_series(f"{prefix}/cat/sub/deep/series")
    if series.firstdate != mm(2024, 1) or series.values.tolist() != [1.0, 2.0, 3.0]:
        raise ValueError("Nested series mismatch.")
    dates = db.read_series(f"{prefix}/cat/\u65e5\u672c\u8a9e/dates")
    if not isinstance(dates, de.StoredSeries) or dates.values.tolist() != CATALOG_DATES.tolist():
        raise ValueError("Nested represented series mismatch.")
    if db.get_attributes(f"{prefix}/cat/scalar") != dict(CATALOG_ATTRIBUTES):
        raise ValueError("Attribute enumeration mismatch.")
    if db.get_attribute(f"{prefix}/cat/sub/bools", "jeltype") != "Bool":
        raise ValueError("Marker attribute not readable.")
    if (
        db.get_attributes(f"{prefix}/cat") != {"owner": "julia"}
        or db.get_attribute("/", "root_note") != "hello"
    ):
        raise ValueError("Catalog/root attributes mismatch.")
    info = db.object_info(f"{prefix}/cat/sub/deep/tensor")
    if info.depth != 5 or info.kind != "array" or db.object_path(info.id) != info.path:
        raise ValueError("Object info mismatch.")


# Workspace interchange: the mixed Workspace Julia's generate-workspace writes
# with writedb at the root of its fixture, written here with write_workspace
# under /workspace. Julia's verify-wheel reads it back with readdb; the three
# marked scalars of the Julia tree (Symbol, Rational, Date) have no Python
# writer yet and are absent here.
WORKSPACE_ORDER = [("b", 1), ("a", 2), ("B", 3), ("\u00e4", 4), ("_", 5), ("10", 6), ("9", 7)]


def workspace_tree() -> tsecon.Workspace:
    ws = tsecon.Workspace()
    ws.f = 1.5
    ws.i = 7
    ws.i8 = np.int8(-3)
    ws.u = np.uint64(2**63)
    ws.c = 1.0 + 2.0j
    ws.s = "vintage \u2016 3"
    ws.b = True
    ws.d = qq(2020, 1)
    ws.dur = Duration(Monthly(), 2)
    ws.ts = tsecon.TSeries(mm(2024, 1), np.array([1.0, 2.0, 3.0]))
    ws.tsi = tsecon.TSeries(qq(2020, 1), np.array([1, 2], dtype=np.int64))
    ws.tsb = tsecon.TSeries(tsecon.yy(2020), np.array([True, False]))
    ws.tse = tsecon.TSeries(mm(2024, 1), np.array([], dtype=np.float64))
    ws.tsd = de.StoredSeries(mm(2024, 1), CATALOG_DATES, de.StoredElement.date(Monthly()))
    ws.mv = tsecon.MVTSeries(qq(2024, 1), ("p", "q"), np.array([[1.0, 2.0], [3.0, 4.0]]))
    ws.mat = np.array([[1.0, 2.0], [3.0, 4.0]])
    ws.v = np.array([1, 2, 3], dtype=np.int32)
    ws.vs = ["a", "b"]
    ws.ur = range(1, 6)
    ws.mr = tsecon.MITRange(qq(2020, 1), qq(2020, 4))
    ws.t3 = np.arange(1, 25, dtype=np.int64).reshape((2, 3, 4), order="F")
    ws.nested = tsecon.Workspace(x=1, deeper=tsecon.Workspace(y=2.0))
    ws.nested.deeper["\u65e5\u672c\u8a9e"] = 9
    ws.empty = tsecon.Workspace()
    ws.order = tsecon.Workspace()
    for name, value in WORKSPACE_ORDER:
        ws.order[name] = value
    return ws


def write_workspace_tree(db: de.DataEconFile) -> None:
    db.new_catalog("/workspace")
    report = db.write_workspace(workspace_tree(), "/workspace")
    if not report.ok or report.count != 35:
        raise ValueError(f"write_workspace did not store the whole tree: {report}")
    db.set_attribute("/workspace/f", "note", "user attribute")


def check_workspace_tree(db: de.DataEconFile) -> None:  # noqa: PLR0912 - finite inventory
    """Read the tree written by :func:`write_workspace_tree` back as a Workspace."""
    expected = workspace_tree()
    loaded = db.read_workspace("/workspace")
    ws, report = loaded.workspace, loaded.report
    if not report.ok or report.count != 35 or report.path != "/workspace":
        raise ValueError(f"read_workspace did not load the whole tree: {report}")
    if list(ws.keys()) != sorted(expected.keys(), key=lambda k: k.encode("utf-8")):
        raise ValueError("Workspace keys are not in UTF-8 byte order.")
    for name in ("f", "i", "c", "s", "d", "dur"):
        if ws[name] != expected[name] or type(ws[name]) is not type(expected[name]):
            raise ValueError(f"Workspace scalar {name} mismatch.")
    if ws.i8 != np.int8(-3) or ws.u != np.uint64(2**63) or ws.b != np.int8(1):
        raise ValueError("Workspace NumPy/Boolean scalar mismatch.")
    for name in ("ts", "tsi", "tsb", "tse"):
        got, want = ws[name], expected[name]
        if not isinstance(got, tsecon.TSeries) or got.firstdate != want.firstdate:
            raise ValueError(f"Workspace series {name} mismatch.")
        if got.values.dtype != want.values.dtype or got.values.tolist() != want.values.tolist():
            raise ValueError(f"Workspace series {name} values mismatch.")
    if not isinstance(ws.tsd, de.StoredSeries) or ws.tsd.values.tolist() != CATALOG_DATES.tolist():
        raise ValueError("Workspace represented series mismatch.")
    if not isinstance(ws.mv, tsecon.MVTSeries) or list(ws.mv.columns) != ["p", "q"]:
        raise ValueError("Workspace MVTSeries mismatch.")
    np.testing.assert_array_equal(ws.mv.values, expected.mv.values)
    np.testing.assert_array_equal(ws.mat, expected.mat)
    np.testing.assert_array_equal(ws.v, expected.v)
    np.testing.assert_array_equal(ws.t3, expected.t3)
    if ws.v.dtype != np.int32 or ws.t3.dtype != np.int64 or ws.t3.shape != (2, 3, 4):
        raise ValueError("Workspace array dtype/shape mismatch.")
    if ws.vs != ["a", "b"] or ws.ur != range(1, 6) or list(ws.mr) != list(expected.mr):
        raise ValueError("Workspace text/range mismatch.")
    if ws.nested.x != 1 or ws.nested.deeper.y != 2.0 or ws.nested.deeper["\u65e5\u672c\u8a9e"] != 9:
        raise ValueError("Nested Workspace mismatch.")
    if list(ws.nested.keys()) != ["deeper", "x"] or len(ws.empty) != 0:
        raise ValueError("Nested Workspace order or empty catalog mismatch.")
    if list(ws.order.keys()) != ["10", "9", "B", "_", "a", "b", "\u00e4"]:
        raise ValueError("Workspace byte order mismatch.")
    if [ws.order[n] for n, _ in WORKSPACE_ORDER] != [v for _, v in WORKSPACE_ORDER]:
        raise ValueError("Workspace order values mismatch.")
    if db.get_attributes("/workspace/b") != {} or db.get_attributes("/workspace/tsb") != {
        "jeltype": "Bool"
    }:
        raise ValueError("Workspace markers mismatch.")
    if db.get_attribute("/workspace/f", "note") != "user attribute":
        raise ValueError("User attribute is not independent of the Workspace.")
    if db.read_object("/workspace/f") != 1.5 or not isinstance(
        db.read_object("/workspace/ts"), tsecon.TSeries
    ):
        raise ValueError("read_object dispatch mismatch.")


# ---- text matrices and tensors ----------------------------------------------

TEXT_M23 = np.array([["r1c1", "r1c2", "r1c3"], ["r2c1", "r2c2", "r2c3"]])
TEXT_MULTI22 = np.array([["\u00e9", "a\u00e9"], ["\U0001f642", "z"]])
TEXT_T223 = np.array([f"e{i}" for i in range(1, 13)]).reshape((2, 2, 3), order="F")
TEXT_H24 = np.array([f"h{i}" for i in range(1, 25)])


def text_array_inventory() -> list[tuple[str, np.ndarray | de.StoredText]]:
    """The fixture's text-array inventory in its Python input forms (same names as Julia's)."""
    symbols = np.array([["alpha", "b"], ["", "d"]])
    substrings = np.array([["ab", "bc"]])
    return [
        ("text2_2x3", TEXT_M23),
        ("text2_3x2", TEXT_M23.T),
        ("text2_blank_2x2", np.array([["", ""], ["", ""]])),
        ("text2_symbol_2x2", de.StoredText.from_numpy(symbols, "Symbol")),
        ("text2_substring_1x2", de.StoredText.from_numpy(substrings, "SubString{String}")),
        ("text2_empty_0x0", np.empty((0, 0), dtype=str)),
        ("text2_empty_0x3", np.empty((0, 3), dtype=str)),
        ("text2_empty_3x0", np.empty((3, 0), dtype=str)),
        ("text2_symbol_empty_0x2", de.StoredText((), "Symbol", (0, 2))),
        ("text3_2x2x3", TEXT_T223),
        ("text3_3x2x4", TEXT_H24.reshape((3, 2, 4), order="F")),
        ("text3_4x3x2", TEXT_H24.reshape((4, 3, 2), order="F")),
        (
            "text4_1x2x3x1",
            np.array([f"f{i}" for i in range(1, 7)]).reshape((1, 2, 3, 1), order="F"),
        ),
        (
            "text5_2x1x2x1x1",
            np.array([f"g{i}" for i in range(1, 5)]).reshape((2, 1, 2, 1, 1), order="F"),
        ),
        (
            "text3_symbol_1x2x3",
            de.StoredText.from_numpy(
                np.array(["a", "bb", "ccc", "", "e", "f"]).reshape((1, 2, 3), order="F"), "Symbol"
            ),
        ),
        (
            "text5_symbol_1x1x2x1x1",
            de.StoredText.from_numpy(np.array(["p", "q"]).reshape((1, 1, 2, 1, 1)), "Symbol"),
        ),
        (
            "text3_substring_1x1x2",
            de.StoredText.from_numpy(substrings.reshape((1, 1, 2)), "SubString{String}"),
        ),
        ("text2_multibyte_2x2", TEXT_MULTI22),
        ("text2_multibyte_symbol_2x2", de.StoredText.from_numpy(TEXT_MULTI22, "Symbol")),
        ("text3_multibyte_1x2x2", TEXT_MULTI22.reshape((1, 2, 2), order="F")),
        (
            "text5_multibyte_symbol_2x1x1x1x2",
            de.StoredText.from_numpy(TEXT_MULTI22.reshape((2, 1, 1, 1, 2), order="F"), "Symbol"),
        ),
        ("text3_empty_0x2x3", np.empty((0, 2, 3), dtype=str)),
        ("text3_empty_marked_0x2x3", np.empty((0, 2, 3), dtype=str)),
        ("text5_symbol_empty_0x0x0x0x0", de.StoredText((), "Symbol", (0, 0, 0, 0, 0))),
        ("text4_empty_1x0x1x0", np.empty((1, 0, 1, 0), dtype=str)),
        ("text2_string_marker_2x3", de.StoredText.from_numpy(TEXT_M23, "String")),
        ("text3_string_marker_2x2x3", de.StoredText.from_numpy(TEXT_T223, "String")),
        ("text2_abstract_2x3", de.StoredText.from_numpy(TEXT_M23, "AbstractString")),
    ]


def text_workspace() -> tsecon.Workspace:
    return tsecon.Workspace(
        tm=TEXT_M23,
        ts=de.StoredText.from_numpy(np.array([["alpha", "b"], ["", "d"]]), "Symbol"),
        tt=TEXT_T223,
        t5=de.StoredText.from_numpy(np.array(["p", "q"]).reshape((1, 1, 2, 1, 1)), "Symbol"),
        tv=["a", "b"],
    )


def write_text_arrays(db: de.DataEconFile) -> None:
    """Write the text matrices and tensors the Julia fixture stores, under the same names."""
    for name, value in text_array_inventory():
        db.write_array(name, value)
    db.new_catalog("/text_ws")
    report = db.write_workspace(text_workspace(), "/text_ws")
    if not report.ok or report.count != 5:
        raise ValueError(f"write_workspace did not store the text Workspace: {report}")


def _check_text_value(name: str, actual: object, expected: np.ndarray | de.StoredText) -> None:
    if isinstance(expected, de.StoredText):
        if not isinstance(actual, de.StoredText) or actual != expected:
            raise ValueError(f"Text array {name} lost its marker, bytes or shape.")
        if actual.tolist() != expected.tolist():
            raise ValueError(f"Text array {name} decodes differently.")
        return
    if not isinstance(actual, np.ndarray) or actual.dtype.kind != "U":
        raise ValueError(f"Text array {name} did not read back as a NumPy str_ array.")
    if actual.shape != expected.shape or actual.tolist() != expected.tolist():
        raise ValueError(f"Text array {name} changed shape or values.")
    if not actual.flags.owndata or not actual.flags.c_contiguous:
        raise ValueError(f"Text array {name} does not own C-contiguous storage.")


def check_text_arrays(db: de.DataEconFile) -> None:
    """Read every text array written by :func:`write_text_arrays`, including the Workspace."""
    for name, expected in text_array_inventory():
        _check_text_value(name, db.read_array(name), expected)
        markers = db.get_attributes(name)
        if isinstance(expected, de.StoredText):
            if markers != {"jeltype": expected.marker}:
                raise ValueError(f"Text array {name} marker mismatch: {markers}.")
        elif markers != ({"jeltype": "String"} if expected.size == 0 else {}):
            raise ValueError(f"Text array {name} marker mismatch: {markers}.")
    loaded = db.read_workspace("/text_ws")
    if not loaded.report.ok or loaded.report.count != 5:
        raise ValueError(f"read_workspace did not load the text Workspace: {loaded.report}")
    for name, expected in text_workspace().items():
        if name == "tv":
            if loaded.workspace.tv != ["a", "b"]:
                raise ValueError("Text Workspace vector mismatch.")
            continue
        _check_text_value(f"text_ws/{name}", loaded.workspace[name], expected)


# Represented MVTSeries and structure-marked matrices: the inventory Julia's
# generate-represented-mvtseries stores, rebuilt from Python's own forms
# (StoredMVTSeries carriers and the StoredArray structure constructors). Julia's
# verify-wheel reads it back with the same expectations as the reference fixture.

RMV_ANCHOR = mm(2024, 1)  # code 24288
RMV_Q = de.StoredElement.date(Quarterly(3))
RMV_Y6 = de.StoredElement.duration(Yearly(6))
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1


def _q(code: int) -> MIT:
    return MIT(Quarterly(3), code)


def _y6(code: int) -> Duration:
    return Duration(Yearly(6), code)


def _mvts_128(element: de.StoredElement, rows: list[list[int]]) -> np.ndarray:
    """Pack Python integers into a rows-by-columns 128-bit carrier."""
    flat = [value for row in rows for value in row]
    packed = de.StoredSeries.from_list(RMV_ANCHOR, element, flat).values
    return packed.reshape((len(rows), len(rows[0]) if rows else 0))


def _cf16_carrier(bits: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    """A ComplexF16 carrier from exact (real, imag) float16 bit patterns, row-major."""
    out = np.empty(len(bits), dtype=de.COMPLEXF16.dtype)
    out["real"] = np.array([r for r, _ in bits], dtype="<u2").view(np.float16)
    out["imag"] = np.array([i for _, i in bits], dtype="<u2").view(np.float16)
    return out.reshape(shape)


def represented_mvtseries_inventory() -> list[tuple[str, de.StoredMVTSeries | tsecon.MVTSeries]]:
    """The fixture's MVTSeries inventory in Python's input forms (same names as Julia's)."""
    inventory: list[tuple[str, de.StoredMVTSeries | tsecon.MVTSeries]] = [
        (
            "rmv_mit_q_on_m",
            de.StoredMVTSeries.from_list(
                RMV_ANCHOR,
                ("a", "b", "c"),
                RMV_Q,
                [[_q(-1), _q(1), _q(INT64_MIN)], [_q(0), _q(2), _q(INT64_MAX)]],
            ),
        ),
        (
            "rmv_dur_y6_on_d",
            de.StoredMVTSeries.from_list(
                MIT(Daily(), 738000),
                ("x", "y"),
                RMV_Y6,
                [[_y6(1), _y6(INT64_MIN)], [_y6(-2), _y6(INT64_MAX)], [_y6(3), _y6(0)]],
            ),
        ),
        (
            "rmv_mit_unit_on_unit",
            de.StoredMVTSeries.from_list(
                MIT(Unit(), -5),
                ("a", "b"),
                de.StoredElement.date(Unit()),
                [
                    [MIT(Unit(), INT64_MIN), MIT(Unit(), 1)],
                    [MIT(Unit(), 0), MIT(Unit(), INT64_MAX)],
                ],
            ),
        ),
        (
            "rmv_mit_w7_on_b",
            de.StoredMVTSeries.from_list(
                MIT(BDaily(), 500000),
                ("a",),
                de.StoredElement.date(Weekly(7)),
                [[MIT(Weekly(7), -1711422)], [MIT(Weekly(7), 1711422)], [MIT(Weekly(7), 7)]],
            ),
        ),
        (
            "rmv_int128",
            de.StoredMVTSeries(
                RMV_ANCHOR,
                ("a", "b"),
                _mvts_128(de.INT128, [[-(2**127), 2**127 - 1], [-1, 7], [0, 2**70]]),
                de.INT128,
            ),
        ),
        (
            "rmv_uint128",
            de.StoredMVTSeries(
                RMV_ANCHOR,
                ("a", "b", "c"),
                _mvts_128(de.UINT128, [[0, 2, 2**70], [1, 2**128 - 1, 5]]),
                de.UINT128,
            ),
        ),
        (
            "rmv_complexf16",
            de.StoredMVTSeries(
                RMV_ANCHOR,
                ("a", "b"),
                _cf16_carrier(
                    [(0x8000, 0x3D00), (0x7E55, 0xFC00), (0x4000, 0x4200), (0x0000, 0x8000)], (2, 2)
                ),
                de.COMPLEXF16,
            ),
        ),
        (
            "rmv_mit_one_row",
            de.StoredMVTSeries.from_list(
                RMV_ANCHOR, ("a", "b", "c"), RMV_Q, [[_q(1), _q(2), _q(3)]]
            ),
        ),
        (
            "rmv_mit_unicode_names",
            de.StoredMVTSeries.from_list(
                RMV_ANCHOR,
                ("a\u00e9", "\U0001f642", ""),
                RMV_Q,
                [[_q(1), _q(3), _q(5)], [_q(2), _q(4), _q(6)]],
            ),
        ),
        (
            "rmv_int128_zero_rows",
            de.StoredMVTSeries.from_list(RMV_ANCHOR, ("a", "b"), de.INT128, []),
        ),
        (
            "rmv_complexf16_zero_rows",
            de.StoredMVTSeries.from_list(RMV_ANCHOR, ("a", "b"), de.COMPLEXF16, []),
        ),
        ("rmv_dur_zero_rows", de.StoredMVTSeries.from_list(RMV_ANCHOR, ("a", "b"), RMV_Y6, [])),
    ]
    for label, frequency, low, high in matrix_axis_families():
        for suffix, code in (("min", low), ("max", high)):
            inventory.append(
                (
                    f"rmv_mit_{suffix}_{label}",
                    de.StoredMVTSeries.from_list(
                        MIT(frequency, code), ("a", "b"), RMV_Q, [[_q(1), _q(2)]]
                    ),
                )
            )
    int_rows = np.array([[1, 2], [0, 1]], dtype="<i8")
    inventory.extend(
        [
            (
                "rmv_bool_on_int128",
                de.StoredMVTSeries(
                    RMV_ANCHOR,
                    ("a", "b"),
                    _mvts_128(de.INT128, [[1, 0], [0, 1]]),
                    de.INT128.with_bool_marker(),
                ),
            ),
            (
                "rmv_bool_on_int64_01",
                tsecon.MVTSeries(RMV_ANCHOR, ["a", "b"], np.array([[True, False], [False, True]])),
            ),
            (
                "rmv_float64_on_int64",
                de.StoredMVTSeries(
                    RMV_ANCHOR, ("a", "b"), int_rows, de.StoredElement.numeric("<i8", "Float64")
                ),
            ),
            (
                "rmv_mit_m_on_int64",
                de.StoredMVTSeries(
                    RMV_ANCHOR,
                    ("a", "b"),
                    int_rows,
                    de.StoredElement.numeric("<i8", "MIT{Monthly}"),
                ),
            ),
            (
                "rmv_identity_object",
                de.StoredMVTSeries(
                    RMV_ANCHOR, ("a", "b"), int_rows, RMV_Q, object_marker="MVTSeries"
                ),
            ),
            (
                "rmv_identity_param",
                de.StoredMVTSeries(
                    RMV_ANCHOR,
                    ("a", "b"),
                    _mvts_128(de.INT128, [[1, 2], [0, 1]]),
                    de.INT128,
                    object_marker="MVTSeries{Monthly, Int128}",
                ),
            ),
            (
                "rmv_empty_mit_marked",
                de.StoredMVTSeries.from_list(
                    RMV_ANCHOR, ("a", "b"), de.StoredElement.date(Monthly()), []
                ),
            ),
        ]
    )
    return inventory


def structure_inventory() -> list[tuple[str, de.StoredArray]]:
    """The fixture's structure-marked matrices, built with the Python constructors."""
    a = np.array([[1.0, 3.0], [20.0, 4.0]])
    h = np.array([[1 + 9j, 2 - 1j], [20 + 20j, 4 - 9j]])
    signs = np.frombuffer(
        np.array(
            [0x8000000000000000, 0x4020000000000000, 0x7FF8000000000055, 0x8000000000000000],
            dtype="<u8",
        ).tobytes(),
        dtype="<f8",
    ).reshape((2, 2), order="F")
    csigns = np.array(
        [[complex(-0.0, -0.0), complex(-0.0, 7.0)], [complex(8.0, 8.0), complex(-0.0, -0.0)]]
    )
    i128 = de.StoredArray(_mvts_128(de.INT128, [[-(2**127), 3], [20, 2**127 - 1]]), de.INT128)
    c16 = de.StoredArray(
        _cf16_carrier(
            [(0x3C00, 0x4880), (0x4000, 0xBC00), (0x4D00, 0x4D00), (0x4400, 0x0000)], (2, 2)
        ),
        de.COMPLEXF16,
    )
    mitm = de.StoredArray(
        np.array([[1, 3], [20, 4]], dtype="<i8"), de.StoredElement.date(Monthly())
    )
    return [
        ("str_sym_u_f64", de.StoredArray.symmetric(a, "U")),
        ("str_sym_l_f64", de.StoredArray.symmetric(a, "L")),
        ("str_herm_u_c64", de.StoredArray.hermitian(h, "U")),
        ("str_herm_l_c64", de.StoredArray.hermitian(h, "L")),
        ("str_diag_vec_f64", de.StoredArray.diagonal(np.array([3.0, -0.0, 5.0]))),
        ("str_sym_l_signs", de.StoredArray.symmetric(signs, "L")),
        ("str_herm_u_signs", de.StoredArray.hermitian(csigns, "U")),
        ("str_sym_u_i128", de.StoredArray.symmetric(i128, "U")),
        ("str_herm_l_c16", de.StoredArray.hermitian(c16, "L")),
        ("str_diag_mit_m", de.StoredArray.diagonal(mitm)),
        ("str_sym_l_bool", de.StoredArray.symmetric(np.array([[True, True], [False, False]]), "L")),
        ("str_diag_f16", de.StoredArray.diagonal(np.array([[1, 3], [20, 4]], dtype=np.float16))),
        ("str_sym_empty_f64", de.StoredArray.symmetric(np.empty((0, 0)))),
        ("str_diag_empty_f64", de.StoredArray.diagonal(np.empty(0))),
        ("str_herm_1x1_c64", de.StoredArray.hermitian(np.array([[complex(2.0, 5.0)]]))),
    ]


def write_represented_mvtseries(db: de.DataEconFile) -> None:
    """Write the represented MVTSeries and structure matrices under the fixture's names."""
    for name, value in represented_mvtseries_inventory():
        db.write_series(name, value)
    for name, value in structure_inventory():
        db.write_array(name, value)


def _expected_mvtseries_markers(expected: de.StoredMVTSeries | tsecon.MVTSeries) -> dict:
    if isinstance(expected, tsecon.MVTSeries):
        return {"jeltype": "Bool"} if expected.values.dtype.kind == "b" else {}
    markers = {}
    token = expected.element.written_marker(expected.values.size)
    if token is not None:
        markers["jeltype"] = token
    if expected.object_marker is not None:
        markers["jtype"] = expected.object_marker
    return markers


def check_represented_mvtseries(db: de.DataEconFile) -> None:  # noqa: PLR0912 - finite inventory
    """Read every object written by :func:`write_represented_mvtseries` back exactly."""
    for name, expected in represented_mvtseries_inventory():
        actual = db.read_series(name)
        if type(actual) is not type(expected):
            raise ValueError(f"Represented MVTSeries {name} read back as {type(actual).__name__}.")
        if isinstance(actual, tsecon.MVTSeries):
            if (
                actual.firstdate != expected.firstdate
                or actual.column_names != expected.column_names
                or actual.values.dtype != expected.values.dtype
                or not np.array_equal(actual.values, expected.values)
            ):
                raise ValueError(f"MVTSeries {name} changed on the way back.")
        elif actual != expected:
            raise ValueError(f"Represented MVTSeries {name} changed on the way back.")
        if isinstance(actual, de.StoredMVTSeries):
            if not actual.values.flags.owndata or not actual.values.flags.c_contiguous:
                raise ValueError(f"Represented MVTSeries {name} does not own its carrier.")
            # The ComplexF16 row carries a NaN payload, which never compares
            # equal although its bytes already did above; compare its printed
            # form. (MIT codes beyond datetime's range have no repr, so the
            # other kinds compare their values directly.)
            same = expected.element.kind == "complexf16"
            actual_list, expected_list = actual.tolist(), expected.tolist()
            if (
                (repr(actual_list) != repr(expected_list))
                if same
                else (actual_list != expected_list)
            ):
                raise ValueError(f"Represented MVTSeries {name} converts differently.")
        markers = db.get_attributes(name)
        if markers != _expected_mvtseries_markers(expected):
            raise ValueError(f"Represented MVTSeries {name} marker mismatch: {markers}.")
    for name, expected in structure_inventory():
        actual = db.read_array(name)
        if not isinstance(actual, de.StoredArray) or actual != expected:
            raise ValueError(f"Structure matrix {name} changed on the way back.")
        dense = actual.to_interpreted()
        values = dense.values if isinstance(dense, de.StoredArray) else dense
        if values.tobytes(order="F") != expected.values.tobytes(order="F"):
            raise ValueError(f"Structure matrix {name} is not stored in its dense form.")
        if db.get_attribute(name, "jtype") != expected.object_marker:
            raise ValueError(f"Structure matrix {name} lost its wrapper marker.")


# ---- marker-mapped, wide and exceptional scalars ------------------------------

SCALAR_MARKER_FIXTURE = "julia_scalar_markers.daec"
SCALAR_MARKER_CASES = 151


def scalar_marker_inventory() -> list[tuple[str, object]]:
    """Python inputs for the scalar forms Julia's own writer produces (verify-wheel names).

    Each value is written through ``write_scalar``; the Julia verifier loads
    the object with the pinned loader and compares it with the value the
    Python side promised (``float(p//q)`` plus a ``Rational`` marker, unix
    seconds plus ``Date``/``DateTime``, two Float64 plus an integer ``Complex``
    marker, the sixteen- and four-byte widths, ``Symbol`` and raw text).
    """
    return [
        ("smw_half", Fraction(1, 2)),
        ("smw_2p54", Fraction(2**54)),
        ("smw_third_lossy", de.StoredScalar.fraction(Fraction(1, 3), exact=False)),
        ("smw_third_int32", de.StoredScalar.fraction(Fraction(1, 3), parameter="Int32")),
        # 3/2**62 is not reconstructible under any parameter (Julia rounds the
        # partial quotient above 2**53): the explicit lossy write records that.
        (
            "smw_3_2p62_lossy",
            de.StoredScalar.fraction(Fraction(3, 2**62), parameter="Int128", exact=False),
        ),
        ("smw_u8_three_quarters", de.StoredScalar.fraction(Fraction(3, 4), parameter="UInt8")),
        ("smw_day", dt.date(2024, 3, 15)),
        ("smw_stamp", dt.datetime(2024, 3, 15, 13, 45, 30, 123000)),
        ("smw_stamp_negative_ms", dt.datetime(1969, 12, 31, 23, 59, 59, 999000)),
        ("smw_year0", de.StoredScalar.calendar_date(0, 1, 1)),
        ("smw_year300k", de.StoredScalar.calendar_date(300000, 1, 1)),
        ("smw_neg300k", de.StoredScalar.calendar_datetime(-300000, 6, 1, 12, 0, 0, 4)),
        ("smw_z", de.IntegerComplex(2**54, 1)),
        ("smw_z_neg", de.IntegerComplex(-3, 0)),
        ("smw_z8", de.StoredScalar.integer_complex(127, -128, parameter="Int8")),
        ("smw_z128", de.StoredScalar.integer_complex(2**100, 2**70, parameter="Int128")),
        ("smw_zbool", de.StoredScalar.integer_complex(1, 0, parameter="Bool")),
        ("smw_sym", de.StoredScalar.symbol("gdp")),
        ("smw_sym_empty", de.StoredScalar.symbol("")),
        ("smw_sym_multibyte", de.StoredScalar.symbol("\u00e9\U0001f642")),
        ("smw_substring", de.StoredScalar.text("hello", "SubString{String}")),
        ("smw_raw", de.StoredScalar.raw_text(b"f\xffo")),
        ("smw_raw_nul", de.StoredScalar.raw_text(b"a\0b", allow_nul=True)),
        ("smw_i128_min", de.StoredScalar.int128(-(2**127))),
        ("smw_i128_max", de.StoredScalar.int128(2**127 - 1)),
        ("smw_u128_max", de.StoredScalar.uint128(2**128 - 1)),
        ("smw_u128_2p64", de.StoredScalar.uint128(2**64)),
        ("smw_cf16", de.StoredScalar.complexf16(complex(1.5, -2.25))),
        ("smw_cf16_special", de.StoredScalar.complexf16(complex(math.nan, math.inf))),
        ("smw_cf16_negzero", de.StoredScalar.complexf16(complex(-0.0, 0.0))),
        # Opaque markers are preserved on write; Julia's loader fails on them.
        (
            "smw_opaque_irrational",
            de.StoredScalar(struct.pack("<d", 3.141592653589793), 4, 0, "Irrational{:\u03c0}"),
        ),
        ("smw_opaque_unknown", de.StoredScalar(struct.pack("<d", 1.5), 4, 0, "NoSuchType")),
    ]


def reference_scalar_markers(fixture: Path) -> dict[str, de.StoredScalar]:
    """Every ``sm_*`` case of the reference fixture, read in its stored form."""
    path = fixture.parent / SCALAR_MARKER_FIXTURE
    if not path.is_file():
        raise FileNotFoundError(f"The scalar-marker reference fixture is missing: {path}")
    siblings = ("_type", "_value", "_julia", "_rewrite", "_error")
    with de.open_dataecon(path) as db:
        names = sorted(
            info.name
            for info in db.list_objects()
            if info.name.startswith("sm_") and not info.name.endswith(siblings)
        )
        cases = {name: db.read_scalar(name) for name in names}
    if len(cases) != SCALAR_MARKER_CASES or any(
        not isinstance(value, de.StoredScalar) for value in cases.values()
    ):
        raise ValueError("The scalar-marker reference fixture inventory changed.")
    return cases


def write_scalar_markers(db: de.DataEconFile, fixture: Path) -> None:
    """Write the Python-built scalar forms and rewrite every reference case verbatim."""
    for name, value in scalar_marker_inventory():
        db.write_scalar(name, value)
    for name, stored in reference_scalar_markers(fixture).items():
        db.write_scalar(f"smrw_{name}", stored)


def check_scalar_markers(db: de.DataEconFile, fixture: Path) -> None:
    """Read every scalar written by :func:`write_scalar_markers` back exactly."""
    expected_markers = {
        "smw_half": {"jtype": "Rational{Int64}"},
        "smw_day": {"jtype": "Date"},
        "smw_stamp": {"jtype": "DateTime"},
        "smw_z": {"jtype": "Complex{Int64}"},
        "smw_sym": {"jtype": "Symbol"},
        "smw_raw": {},
        "smw_i128_min": {},
    }
    for name, value in scalar_marker_inventory():
        actual = db.read_scalar(name)
        if not isinstance(actual, de.StoredScalar):
            raise ValueError(f"Scalar {name} did not read back in its stored form.")
        if isinstance(value, de.StoredScalar):
            if actual != value:
                raise ValueError(f"Scalar {name} changed on the way back.")
        elif actual.marker is None or actual.to_interpreted() != value:
            raise ValueError(f"Scalar {name} does not interpret to its input.")
        if name in expected_markers and db.get_attributes(name) != expected_markers[name]:
            raise ValueError(f"Scalar {name} marker mismatch: {db.get_attributes(name)}.")
    assert db.read_scalar("smw_half").to_fraction() == Fraction(1, 2)
    assert db.read_scalar("smw_third_lossy").to_fraction() == Fraction(
        6004799503160661, 18014398509481984
    )
    assert db.read_scalar("smw_neg300k").to_calendar() == (-300000, 6, 1, 12, 0, 0, 4)
    assert db.read_scalar("smw_z128").to_integer_complex() == de.IntegerComplex(2**100, 2**70)
    assert db.read_scalar("smw_cf16_negzero").to_complex64() == np.complex64(0)
    for name, stored in reference_scalar_markers(fixture).items():
        actual = db.read_scalar(f"smrw_{name}")
        if actual != stored:
            raise ValueError(f"Rewritten reference scalar {name} changed on the way back.")
        markers = {} if stored.marker is None else {"jtype": stored.marker}
        if db.get_attributes(f"smrw_{name}") != markers:
            raise ValueError(f"Rewritten reference scalar {name} lost or gained a marker.")


# ---- extended markers: exact carriers, abstract/empty/text/Bool routes -------

EXTENDED_ANCHOR = mm(2024, 1)


def _f64(value: float) -> bytes:
    return struct.pack("<d", value)


def extended_scalar_inventory() -> list[tuple[str, de.StoredScalar, object]]:
    """Python scalar forms of the remaining routes: (name, stored form, interpreted value)."""
    from decimal import Decimal  # noqa: PLC0415 - checker-only import

    rc = de.RationalComplex(Fraction(1, 2), Fraction(-2))
    return [
        ("xm_rc", de.StoredScalar.rational_complex(rc), rc),
        (
            "xm_rc8",
            de.StoredScalar.rational_complex(Fraction(1, 3), Fraction(0), parameter="Int8"),
            de.RationalComplex(Fraction(1, 3), Fraction(0)),
        ),
        ("xm_bigfloat", de.StoredScalar.bigfloat(Decimal(1) / Decimal(8)), Decimal("0.125")),
        ("xm_sym_float", de.StoredScalar(_f64(1.5), 4, 0, "Symbol"), "1.5"),
        ("xm_sym_f32big", de.StoredScalar(np.float32(1e10).tobytes(), 4, 0, "Symbol"), "1.0e10"),
        (
            "xm_sym_c64",
            de.StoredScalar(struct.pack("<dd", 1.5, -2.0), 5, 0, "Symbol"),
            "1.5 - 2.0im",
        ),
        ("xm_sym_mit", de.StoredScalar(struct.pack("<q", 24288), 3, 32, "Symbol"), "2024M1"),
        (
            "xm_rat_f32",
            de.StoredScalar(np.float32(0.1).tobytes(), 4, 0, "Rational{Int64}"),
            Fraction(13421773, 134217728),
        ),
        (
            "xm_rat_f16",
            de.StoredScalar(np.float16(1 / 3).tobytes(), 4, 0, "Rational{Int8}"),
            Fraction(1, 3),
        ),
        ("xm_date_u64", de.StoredScalar(struct.pack("<Q", 5), 2, 0, "Date"), dt.date(1970, 1, 1)),
        (
            "xm_datetime_i128",
            de.StoredScalar((1710460800).to_bytes(16, "little", signed=True), 1, 0, "DateTime"),
            dt.datetime(2024, 3, 15),
        ),
        (
            "xm_mit_big",
            de.StoredScalar(struct.pack("<q", 2**62), 1, 0, "MIT{Monthly}"),
            tsecon.MIT(tsecon.Monthly(), 2**62),
        ),
        ("xm_abs_dur", de.StoredScalar(struct.pack("<q", 5), 1, 32, "AbstractFloat"), 5 / 12),
        # Printed calendar Symbols in any
        # year, the dated complexes, Char and Julia's wrapping unix time.
        ("xm_sym_daily0", de.StoredScalar(struct.pack("<q", 0), 3, 12, "Symbol"), "0000-12-31"),
        (
            "xm_sym_daily_big",
            de.StoredScalar(struct.pack("<q", 3652060), 3, 12, "Symbol"),
            "10000-01-01",
        ),
        (
            "xm_dated_complex",
            de.StoredScalar(struct.pack("<q", 24288), 3, 32, "Complex"),
            de.DatedComplex(tsecon.MIT(tsecon.Monthly(), 24288), tsecon.MIT(tsecon.Monthly(), 0)),
        ),
        (
            "xm_dated_complex_i8",
            de.StoredScalar(b"\x05", 1, 0, "Complex{MIT{Daily}}"),
            de.DatedComplex(tsecon.MIT(tsecon.Daily(), 5), tsecon.MIT(tsecon.Daily(), 0)),
        ),
        ("xm_char", de.StoredScalar(struct.pack("<q", 0x1F642), 1, 0, "Char"), "\U0001f642"),
        ("xm_char_f64", de.StoredScalar(_f64(97.0), 4, 0, "Char"), "a"),
    ]


def extended_container_inventory() -> list[tuple[str, object]]:
    """Python-written containers of the remaining routes (Julia loads each one)."""
    rational_values, rational_element = de.rational_storage(
        [Fraction(1, 2), Fraction(-3, 4), Fraction(5)]
    )
    intcomplex_values, intcomplex_element = de.integer_complex_storage(
        [de.IntegerComplex(2**54, 1), de.IntegerComplex(-3, 0)]
    )
    rc_values, rc_element = de.rational_complex_storage(
        np.array(
            [
                [
                    de.RationalComplex(Fraction(1, 2), Fraction(-2)),
                    de.RationalComplex(Fraction(1, 3)),
                ],
                [de.RationalComplex(Fraction(0)), de.RationalComplex(Fraction(-1, 8), Fraction(7))],
            ],
            dtype=object,
        ),
        parameter="Int8",
    )
    mvt_values, mvt_element = de.rational_storage(
        np.array([[Fraction(1, 2), Fraction(1)], [Fraction(-1, 4), Fraction(3)]], dtype=object)
    )
    tensor_values, tensor_element = de.rational_storage(
        np.array([[[Fraction(2**70), Fraction(1, 2)], [Fraction(-1), Fraction(0)]]], dtype=object),
        parameter="Int128",
    )
    return [
        ("xm_rational_series", de.StoredSeries(EXTENDED_ANCHOR, rational_values, rational_element)),
        ("xm_intcomplex_vector", de.StoredArray(intcomplex_values, intcomplex_element)),
        ("xm_rc_matrix", de.StoredArray(rc_values, rc_element)),
        (
            "xm_rational_mvt",
            de.StoredMVTSeries(EXTENDED_ANCHOR, ("a", "b"), mvt_values, mvt_element),
        ),
        ("xm_rational_tensor", de.StoredArray(tensor_values, tensor_element)),
        (
            "xm_abstract_signed",
            de.StoredSeries(
                EXTENDED_ANCHOR,
                np.array([0, 1, 2], dtype="<u1"),
                de.StoredElement.numeric("<u1", "Signed"),
            ),
        ),
        (
            "xm_abstract_real_c",
            de.StoredSeries(
                EXTENDED_ANCHOR,
                np.array([1, 2], dtype="<c8"),
                de.StoredElement.numeric("<c8", "Real"),
            ),
        ),
        (
            "xm_union",
            de.StoredSeries(
                EXTENDED_ANCHOR,
                np.array([3, -4], dtype="<i8"),
                de.StoredElement.numeric("<i8", "Union{Int64,Float64}"),
            ),
        ),
        (
            "xm_bigint",
            de.StoredArray(np.array([1e300, 2.0]), de.StoredElement.numeric("<f8", "BigInt")),
        ),
        (
            "xm_bigfloat_series",
            de.StoredSeries(
                EXTENDED_ANCHOR,
                np.array([0.1], dtype="<f4"),
                de.StoredElement.numeric("<f4", "BigFloat"),
            ),
        ),
        (
            "xm_date_array",
            de.StoredArray(
                np.array([0, 86400], dtype="<i8"), de.StoredElement.numeric("<i8", "Date")
            ),
        ),
        (
            "xm_datetime_matrix",
            de.StoredArray(
                np.array([[1710460800.5], [-1.0]]), de.StoredElement.numeric("<f8", "DateTime")
            ),
        ),
        (
            "xm_symbol_array",
            de.StoredArray(np.array([0.5, 1e10]), de.StoredElement.numeric("<f8", "Symbol")),
        ),
        (
            "xm_empty_date",
            de.StoredSeries(
                EXTENDED_ANCHOR, np.empty(0, dtype="<f8"), de.StoredElement.numeric("<f8", "Date")
            ),
        ),
        (
            "xm_empty_real",
            de.StoredSeries(
                EXTENDED_ANCHOR, np.empty(0, dtype="<f8"), de.StoredElement.numeric("<f8", "Real")
            ),
        ),
        (
            "xm_empty_rational",
            de.StoredArray(
                np.empty(0, dtype="<i8"), de.StoredElement.numeric("<i8", "Rational{Int64}")
            ),
        ),
        ("xm_text_obj", de.StoredText((b"a", b"b"), "Symbol", (2,), "Vector{String}")),
        ("xm_text_any", de.StoredText((b"a", b"b"), None, (2,), "Vector{Any}")),
        ("xm_text_empty_symbol", de.StoredText((), None, (0,), "Vector{Symbol}")),
        (
            "xm_bool_i64",
            de.StoredArray(
                np.array([[0, 1], [1, 0]], dtype="<i8"), de.StoredElement.numeric("<i8", "Bool")
            ),
        ),
        (
            "xm_opaque_series",
            de.StoredSeries(
                EXTENDED_ANCHOR, np.array([1.0]), de.StoredElement.numeric("<f8", "NoSuchType")
            ),
        ),
        ("xm_opaque_text", de.StoredText((b"a",), None, (1,), "NoSuchType")),
        # The printed whole-object
        # Symbol of text and numeric arrays, the dated complexes, Char elements
        # and the wrapped unix time of an Int64 element.
        ("xm_text_symbol", de.StoredText((b'a"b', b"\xc3\xa9"), "Symbol", (2,), "Symbol")),
        (
            "xm_text_symbol_matrix",
            de.StoredText((b"a", b"b", b"c", b"d"), None, (2, 2), "Symbol"),
        ),
        (
            "xm_array_symbol",
            de.StoredArray(
                np.array([[1, 2], [3, 4]], dtype="<i1"),
                de.StoredElement.numeric("<i1"),
                object_marker="Symbol",
            ),
        ),
        (
            "xm_array_symbol_f32",
            de.StoredArray(
                np.array([0.5, 1e10], dtype="<f4"),
                de.StoredElement.numeric("<f4", "Bool"),
                object_marker="Symbol",
            ),
        ),
        (
            "xm_bytes_symbol",
            de.StoredArray(
                np.array([97, 98], dtype="<u1"),
                de.StoredElement.numeric("<u1"),
                object_marker="Symbol",
            ),
        ),
        (
            "xm_dated_complex_series",
            de.StoredSeries(
                EXTENDED_ANCHOR,
                np.array([24288, 24289], dtype="<i8"),
                de.StoredElement.date(tsecon.Monthly()).with_marker("Complex"),
            ),
        ),
        (
            "xm_char_array",
            de.StoredArray(
                np.array([97, 0x1F642], dtype="<i8"), de.StoredElement.numeric("<i8", "Char")
            ),
        ),
        (
            "xm_opaque_object",
            de.StoredSeries(
                EXTENDED_ANCHOR,
                np.array([3], dtype="<i8"),
                de.StoredElement.numeric("<i8"),
                object_marker="TSeries{Monthly, Int16}",
            ),
        ),
    ]


def write_extended_markers(db: de.DataEconFile) -> None:
    """Write the extended-marker scalars and containers under the verifier's names."""
    for name, stored, _ in extended_scalar_inventory():
        db.write_scalar(name, stored)
    for name, value in extended_container_inventory():
        if isinstance(value, (de.StoredSeries, de.StoredMVTSeries)):
            db.write_series(name, value)
        else:
            db.write_array(name, value)
    db.write_scalar(
        "xm_datetime_wrap", de.StoredScalar(struct.pack("<q", 9223372036854775), 1, 0, "DateTime")
    )


def check_extended_markers(db: de.DataEconFile) -> None:  # noqa: PLR0912, PLR0915 - finite inventory
    """Read every extended-marker object back exactly and check its interpretation."""
    for name, stored, interpreted in extended_scalar_inventory():
        actual = db.read_scalar(name)
        if actual != stored:
            raise ValueError(f"Scalar {name} changed on the way back.")
        value = actual.to_interpreted()
        if value != interpreted or type(value) is not type(interpreted):
            raise ValueError(f"Scalar {name} interprets to {value!r}, not {interpreted!r}.")
    for name, expected in extended_container_inventory():
        reader = (
            db.read_series
            if isinstance(expected, (de.StoredSeries, de.StoredMVTSeries))
            else db.read_array
        )
        actual = reader(name)
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"Container {name} changed on the way back.")
        if name.startswith("xm_opaque"):
            try:
                actual.to_interpreted()
            except TypeError:
                continue
            raise ValueError(f"Opaque marker {name} was interpreted.")
        value = actual.to_interpreted()
        if name == "xm_rational_series" and value.tolist() != [
            Fraction(1, 2),
            Fraction(-3, 4),
            Fraction(5),
        ]:
            raise ValueError("xm_rational_series does not interpret to its fractions.")
        if name == "xm_intcomplex_vector" and value.tolist() != [
            de.IntegerComplex(2**54, 1),
            de.IntegerComplex(-3, 0),
        ]:
            raise ValueError("xm_intcomplex_vector does not interpret to its pairs.")
        if name == "xm_rc_matrix" and value.tolist()[1][1] != de.RationalComplex(
            Fraction(-1, 8), Fraction(7)
        ):
            raise ValueError("xm_rc_matrix does not interpret to its pairs.")
        if name == "xm_rational_tensor" and value.tolist()[0][0][0] != Fraction(2**70):
            raise ValueError("xm_rational_tensor lost its Int128 numerator.")
        if name == "xm_abstract_signed" and value.values.dtype != np.dtype("<i1"):
            raise ValueError("xm_abstract_signed did not interpret to Int8.")
        if name == "xm_bigint" and value.tolist() != [int(1e300), 2]:
            raise ValueError("xm_bigint did not interpret to exact integers.")
        if name == "xm_date_array" and value.tolist() != [
            np.datetime64("1970-01-01").item(),
            np.datetime64("1970-01-02").item(),
        ]:
            raise ValueError("xm_date_array did not interpret to dates.")
        if name == "xm_symbol_array" and value.tolist() != ["0.5", "1.0e10"]:
            raise ValueError("xm_symbol_array did not interpret to Julia's printed forms.")
        if name == "xm_text_obj" and value != ["a", "b"]:
            raise ValueError("xm_text_obj did not interpret to plain strings.")
        if name == "xm_bool_i64" and value.tolist() != [[False, True], [True, False]]:
            raise ValueError("xm_bool_i64 did not interpret to Booleans.")
        if name == "xm_empty_date" and value.dtype != np.dtype("<M8[D]"):
            raise ValueError("xm_empty_date did not interpret to an empty datetime64[D].")
        if name == "xm_text_symbol" and value != '["a\\"b", "\u00e9"]':
            raise ValueError(f"xm_text_symbol printed {value!r}.")
        if name == "xm_text_symbol_matrix" and value != '["a" "c"; "b" "d"]':
            raise ValueError(f"xm_text_symbol_matrix printed {value!r}.")
        if name == "xm_array_symbol" and value != "Int8[1 2; 3 4]":
            raise ValueError(f"xm_array_symbol printed {value!r}.")
        if name == "xm_array_symbol_f32" and value != "Float32[0.5, 1.0f10]":
            raise ValueError(f"xm_array_symbol_f32 printed {value!r}.")
        if name == "xm_bytes_symbol" and value != "ab":
            raise ValueError(f"xm_bytes_symbol named {value!r}.")
        if name == "xm_dated_complex_series" and value.tolist() != [
            de.DatedComplex(tsecon.MIT(tsecon.Monthly(), 24288), tsecon.MIT(tsecon.Monthly(), 0)),
            de.DatedComplex(tsecon.MIT(tsecon.Monthly(), 24289), tsecon.MIT(tsecon.Monthly(), 0)),
        ]:
            raise ValueError("xm_dated_complex_series did not interpret to its pairs.")
        if name == "xm_char_array" and value.tolist() != ["a", "\U0001f642"]:
            raise ValueError("xm_char_array did not interpret to its characters.")
    # A wrapped Int64 unix time: Julia's defined arithmetic, exact through to_calendar().
    wrapped = db.read_scalar("xm_datetime_wrap")
    if wrapped.to_calendar() != (-292275055, 5, 16, 16, 47, 3, 384):
        raise ValueError(f"xm_datetime_wrap did not wrap like Julia: {wrapped.to_calendar()!r}.")
    attributes = db.get_attributes("xm_text_obj")
    if attributes != {"jeltype": "Symbol", "jtype": "Vector{String}"}:
        raise ValueError(f"xm_text_obj markers changed: {attributes}.")
    if db.get_attributes("xm_bool_i64") != {"jeltype": "Bool"}:
        raise ValueError("xm_bool_i64 lost its Bool marker.")
    if db.get_attributes("xm_opaque_object") != {"jtype": "TSeries{Monthly, Int16}"}:
        raise ValueError("xm_opaque_object lost its whole-object marker.")


def write_interchange(  # noqa: PLR0915 - one write set per interchange family
    output_dir: Path, series: tsecon.TSeries, fixture: Path
) -> Path:
    """Write and reopen series and scalars for separate Julia verification."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"cp{sys.version_info.major}{sys.version_info.minor}.daec"
    if output.exists():
        raise FileExistsError(f"Use a fresh interchange output directory: {output}")
    with de.open_dataecon(output, "a") as db:
        db.write_series("sample", series)
        write_scalar_markers(db, fixture)
        write_series_elements(db)
        write_represented_series(db)
        write_foreign_markers(db)
        write_arrays_unit(db)
        write_matrices_text(db)
        write_tensors(db)
        write_catalogs(db)
        write_workspace_tree(db)
        write_text_arrays(db)
        write_represented_mvtseries(db)
        write_extended_markers(db)
        for name, value in (
            ("bool_false", False),
            ("bool_true", True),
            ("bool_numpy_false", np.bool_(False)),
            ("bool_numpy_true", np.bool_(True)),
        ):
            db.write_scalar(name, value)
        for name, value in scalar_cases().items():
            db.write_scalar(name, value)
        for name, anchor in (("empty", mm(2024, 1)), ("empty_later", mm(2025, 7))):
            db.write_series(name, tsecon.TSeries(anchor, np.empty(0, dtype=np.float64)))
        for name, start, values in series_cases():
            db.write_series(name, tsecon.TSeries(start, np.array(values, dtype=np.float64)))
        write_file_operation_objects(db)
    write_file_operations(output)
    check_discovery_contract(output_dir)
    with de.open_dataecon(output) as db:
        check_scalar_markers(db, fixture)
        check_series_elements(db)
        check_represented_series(db)
        check_foreign_markers(db)
        check_arrays_unit(db)
        check_matrices_text(db)
        check_tensors(db)
        check_catalogs(db)
        check_workspace_tree(db)
        check_text_arrays(db)
        check_represented_mvtseries(db)
        check_extended_markers(db)
        np.testing.assert_array_equal(db.read_series("sample").values, series.values)
        for name, expected in (
            ("bool_false", 0),
            ("bool_true", 1),
            ("bool_numpy_false", 0),
            ("bool_numpy_true", 1),
        ):
            actual = db.read_scalar(name)
            if type(actual) is not np.int8 or actual != expected:
                raise ValueError("Boolean scalar storage must read back as Int8 zero/one.")
        check_file_operation_objects(db)
        empties = [
            (db.read_series(name), anchor)
            for name, anchor in (("empty", mm(2024, 1)), ("empty_later", mm(2025, 7)))
        ]
        scalars = [(db.read_scalar(name), value) for name, value in scalar_cases().items()]
        dated = [(db.read_series(name), start, values) for name, start, values in series_cases()]
    for actual, start, values in dated:
        np.testing.assert_array_equal(actual.values, values)
        if (
            actual.firstdate != start
            or actual.lastdate != start + len(values) - 1
            or actual.frequency != start.frequency
            or actual.values.dtype != np.float64
            or not actual.values.flags.owndata
        ):
            raise ValueError("Series interchange lost dates, frequency, dtype or ownership.")
    for actual, expected in scalars:
        check_scalar_value(actual, expected)
    for empty, anchor in empties:
        if (
            empty.firstdate != anchor
            or empty.lastdate != anchor - 1
            or empty.values.shape != (0,)
            or empty.values.dtype != np.float64
            or not empty.values.flags.owndata
        ):
            raise ValueError("Empty series lost its date anchor, dtype or ownership.")
    return output


def check(fixture: Path, output_dir: Path) -> None:
    """Check installed provenance and generate an output for separate Julia verification."""
    package = Path(de.__file__).resolve().parent
    if not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(
            "Wheel check imported the source checkout instead of the installed package."
        )
    if "tsecon.dataecon._native" in sys.modules:
        raise RuntimeError("Core/package import unexpectedly loaded DataEcon eagerly.")
    native = importlib.util.find_spec("tsecon.dataecon._native")
    if os.environ.get("TSECON_REQUIRE_DATAECON") != "1":
        if native is not None or (package / "_binary").exists():
            raise RuntimeError("This platform should still have a core-only DataEcon installation.")
        print("Core-only wheel: import succeeds and DataEcon native artifacts are absent.")
        return
    if native is None:
        raise RuntimeError("Wheel must enable and include DataEcon native support.")
    manifest = validate_provenance(package)
    if sys.platform != "win32" and (
        manifest.get("linkage") != "static-hidden" or manifest["platform"] != sys.platform
    ):
        raise ValueError("Unexpected native linkage/platform.")
    check_extension_abis(package)
    with de.open_dataecon(fixture) as db:
        series = db.read_series("sample")
    if sys.platform == "win32" and loaded_dll_path() != (package / "_binary/libdaec.dll").resolve():
        raise RuntimeError("DataEcon loaded a DLL outside the installed wheel.")
    if sys.platform != "win32":
        print(audit_static_extension(Path(native.origin)))
    if series.firstdate != mm(2024, 1) or series.lastdate != mm(2024, 4):
        raise ValueError("Julia fixture date metadata changed.")
    if not series.values.flags.owndata:
        raise ValueError("Loaded series does not own its data.")
    np.testing.assert_array_equal(series.values, [1.25, -2.5, 0.0, 4.75])
    output = write_interchange(output_dir, series, fixture)
    print(f"DataEcon installed-wheel check passed for Python {sys.version.split()[0]}.")
    print(f"Package: {Path(tsecon.__file__).parent}")
    if sys.platform == "win32":
        print(f"Native dependencies: {', '.join(manifest['dependencies'])}")
    print(f"Julia verification input: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=os.environ.get(
            "TSECON_DATAECON_OUTPUT_DIR",
            Path(__file__).resolve().parents[1] / "build/dataecon-interchange",
        ),
        help="Defaults to TSECON_DATAECON_OUTPUT_DIR or the project build directory.",
    )
    args = parser.parse_args()
    check(args.fixture, args.output_dir)
