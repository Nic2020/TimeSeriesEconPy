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
import subprocess
import sys
import sysconfig
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
    weekly,
)


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


def write_interchange(output_dir: Path, series: tsecon.TSeries) -> Path:
    """Write and reopen series and scalars for separate Julia verification."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"cp{sys.version_info.major}{sys.version_info.minor}.daec"
    if output.exists():
        raise FileExistsError(f"Use a fresh interchange output directory: {output}")
    with de.open_dataecon(output, "a") as db:
        db.write_series("sample", series)
        write_series_elements(db)
        write_represented_series(db)
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
        check_series_elements(db)
        check_represented_series(db)
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
    output = write_interchange(output_dir, series)
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
