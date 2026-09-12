# SPDX-License-Identifier: MIT
"""Require installed-wheel DataEcon provenance and interchange.

When native support is not required, verify core-only wheels omit the extension.
Run after the wheel has been installed, from outside its source package.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
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
    mm,
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
    cases = {**SCALAR_CASES, **INT64_CASES, **STRING_CASES}
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
    if type(actual) is not float:
        raise TypeError("Scalar read did not return a Python float.")
    np.testing.assert_equal(actual, expected)
    if expected == 0.0 and np.signbit(actual) != np.signbit(expected):
        raise ValueError("Scalar read changed the sign of zero.")


def write_interchange(output_dir: Path, series: tsecon.TSeries) -> Path:
    """Write and reopen series and scalars for separate Julia verification."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"cp{sys.version_info.major}{sys.version_info.minor}.daec"
    if output.exists():
        raise FileExistsError(f"Use a fresh interchange output directory: {output}")
    with de.open_dataecon(output, "a") as db:
        db.write_series("sample", series)
        for name, value in scalar_cases().items():
            db.write_scalar(name, value)
        for name, anchor in (("empty", mm(2024, 1)), ("empty_later", mm(2025, 7))):
            db.write_series(name, tsecon.TSeries(anchor, np.empty(0, dtype=np.float64)))
        for prefix, frequency, anchors, cases in DATED_GROUPS:
            for anchor in anchors:
                for suffix, code, values in cases:
                    db.write_series(
                        f"{prefix}{anchor}_{suffix}",
                        tsecon.TSeries(
                            MIT(frequency(anchor), code), np.array(values, dtype=np.float64)
                        ),
                    )
    with de.open_dataecon(output) as db:
        np.testing.assert_array_equal(db.read_series("sample").values, series.values)
        empties = [
            (db.read_series(name), anchor)
            for name, anchor in (("empty", mm(2024, 1)), ("empty_later", mm(2025, 7)))
        ]
        scalars = [(db.read_scalar(name), value) for name, value in scalar_cases().items()]
        dated = [
            (db.read_series(f"{prefix}{anchor}_{suffix}"), MIT(frequency(anchor), code), values)
            for prefix, frequency, anchors, cases in DATED_GROUPS
            for anchor in anchors
            for suffix, code, values in cases
        ]
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
