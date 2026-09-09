"""Ensure wheel gates reject missing native support and incorrect provenance."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checker(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    return load_script("wheel_checker", ROOT / "scripts/check_dataecon_wheel.py")


@pytest.fixture
def package(tmp_path, checker):
    binary = tmp_path / "_binary"
    binary.mkdir()
    dll = binary / "libdaec.dll"
    dll.write_bytes(b"synthetic DLL contents for hash validation only")
    manifest = {
        "source_commit": checker.SOURCE_COMMIT,
        "source_archive_sha256": checker.SOURCE_SHA256,
        "dataecon_version": "0.4.0",
        "dependencies": ["KERNEL32.dll"],
        "windows_sdk_versions": ["10.0.26100.0"],
        "outputs": {
            "include/daec.h": checker.HEADER_SHA256,
            "bin/libdaec.dll": hashlib.sha256(dll.read_bytes()).hexdigest(),
        },
    }
    (binary / "build-info.json").write_text(json.dumps(manifest))
    for name in ("DATAECON_LICENSE.txt", "SQLITE_NOTICE.txt"):
        (tmp_path / name).write_text("synthetic test notice\n")
    return tmp_path


def test_matching_provenance(checker, package):
    assert checker.validate_provenance(package)["source_commit"] == checker.SOURCE_COMMIT


def test_changed_dll_rejected(checker, package):
    (package / "_binary/libdaec.dll").write_bytes(b"different DLL contents")
    with pytest.raises(ValueError, match="does not match"):
        checker.validate_provenance(package)


@pytest.mark.parametrize(
    "missing",
    [
        "_binary/build-info.json",
        "_binary/libdaec.dll",
        "DATAECON_LICENSE.txt",
        "SQLITE_NOTICE.txt",
    ],
)
def test_missing_wheel_artifact_fails(checker, package, missing):
    (package / missing).unlink()
    with pytest.raises(FileNotFoundError):
        checker.validate_provenance(package)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("source_commit", "unexpected-source"),
        ("source_archive_sha256", "unexpected-archive"),
        ("dataecon_version", "0.5.0"),
        ("dependencies", []),
        ("windows_sdk_versions", []),
    ],
)
def test_wrong_or_missing_build_provenance(checker, package, key, value):
    path = package / "_binary/build-info.json"
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        checker.validate_provenance(package)


def test_required_native_support_cannot_skip(monkeypatch):
    guard = load_script("native_guard", ROOT / "tests/dataecon/conftest.py")
    monkeypatch.setenv("TSECON_REQUIRE_DATAECON", "1")
    monkeypatch.setattr(guard.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(pytest.UsageError, match=r"required.*not built"):
        guard.pytest_configure(None)


def test_optional_native_support_remains_optional(monkeypatch):
    guard = load_script("native_guard", ROOT / "tests/dataecon/conftest.py")
    monkeypatch.delenv("TSECON_REQUIRE_DATAECON", raising=False)
    monkeypatch.setattr(guard.importlib.util, "find_spec", lambda _: None)
    guard.pytest_configure(None)
