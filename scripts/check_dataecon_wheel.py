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
from tsecon import MIT, HalfYearly, Quarterly, Yearly, mm


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


def check_scalar_value(actual: float | int, expected: float | int) -> None:
    """Require the exact Python type, numerical classification and the sign of zero.

    Int64 expectations require an exact ``int``; a floating-point detour would
    change the values beyond 2**53 and is a failure, not a tolerance.
    """
    if type(expected) is int:
        if type(actual) is not int or actual != expected:
            raise TypeError(f"Int64 scalar read returned {actual!r} instead of {expected!r}.")
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
        for name, value in {**SCALAR_CASES, **INT64_CASES}.items():
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
        scalars = [
            (db.read_scalar(name), value) for name, value in {**SCALAR_CASES, **INT64_CASES}.items()
        ]
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
