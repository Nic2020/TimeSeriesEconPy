"""Ensure wheel gates reject missing native support and incorrect provenance."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("system", ["linux", "darwin"])
@pytest.mark.parametrize("fault", [None, "dependency", "symbol", "entry"])
def test_static_extension_link_audit(checker, monkeypatch, system, fault):
    monkeypatch.setattr(checker.sys, "platform", system)
    dependencies = "libc.so.6" if system == "linux" else "/usr/lib/libSystem.B.dylib"
    symbols = "00000000 T PyInit__native\n"
    if fault == "dependency":
        dependencies += "\nlibsqlite3.so.0"
    elif fault == "symbol":
        symbols += "00000001 T _sqlite3_open\n"
    elif fault == "entry":
        symbols = ""
    monkeypatch.setattr(
        checker.subprocess,
        "check_output",
        lambda command, **kwargs: symbols if command[0] == "nm" else dependencies,
    )
    if fault:
        with pytest.raises(ValueError):
            checker.audit_static_extension(Path("synthetic-extension.so"))
    else:
        assert checker.audit_static_extension(Path("synthetic-extension.so")) == dependencies


@pytest.mark.parametrize(
    ("system", "machine", "deployment"),
    [
        ("linux", "aarch64", None),
        ("darwin", "x86_64", "11.0"),
        ("darwin", "arm64", None),
        ("darwin", "arm64", "14.0"),
        ("win32", "AMD64", None),
    ],
)
def test_unix_builder_rejects_undeclared_target(monkeypatch, system, machine, deployment):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    builder = load_script("unix_builder", ROOT / "scripts/build_dataecon_unix.py")
    with pytest.raises(RuntimeError, match="Supported targets"):
        builder.target_flags(system, machine, deployment)


@pytest.mark.parametrize("system", ["linux", "darwin"])
def test_static_manifest_preserves_input_provenance(checker, package, system):
    path = package / "_binary/build-info.json"
    manifest = json.loads(path.read_text())
    manifest.update(linkage="static-hidden", platform=system, compiler=["test compiler"])
    manifest["outputs"]["lib/libdaec.a"] = "a" * 64
    path.write_text(json.dumps(manifest))
    (package / "_binary/libdaec.dll").unlink()
    assert checker.validate_provenance(package)["linkage"] == "static-hidden"
    manifest["outputs"]["lib/libdaec.a"] = "invalid-hash"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance"):
        checker.validate_provenance(package)


@pytest.mark.parametrize("system", ["linux", "darwin"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_static_link_inputs_verified_before_packaging(monkeypatch, tmp_path, system, corrupt):
    hook = load_script("build_hook", ROOT / "hatch_build.py")
    native_root = tmp_path / "native"
    (native_root / "include").mkdir(parents=True)
    (native_root / "lib").mkdir()
    header = native_root / "include/daec.h"
    archive = native_root / "lib/libdaec.a"
    header.write_bytes(b"test header")
    archive.write_bytes(b"test archive")
    manifest = {
        "linkage": "static-hidden",
        "platform": system,
        "outputs": {
            str(path.relative_to(native_root)).replace("\\", "/"): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in (header, archive)
        },
    }
    (native_root / "build-info.json").write_text(json.dumps(manifest))
    if corrupt:
        archive.write_bytes(b"changed archive")
    monkeypatch.setattr(hook.sys, "platform", system)
    monkeypatch.setattr(hook, "SRC_PKG", tmp_path / "package")
    extension = SimpleNamespace(
        include_dirs=[],
        extra_objects=[],
        depends=[],
        extra_compile_args=[],
        libraries=[],
        extra_link_args=[],
    )
    if corrupt:
        with pytest.raises(ValueError, match="differs from manifest"):
            hook._configure_static_dataecon(extension, native_root)
        assert not (tmp_path / "package").exists()
    else:
        hook._configure_static_dataecon(extension, native_root)
        assert extension.extra_objects == [str(archive)]
        assert "daec" not in extension.libraries
        assert (tmp_path / "package/dataecon/_binary/build-info.json").is_file()
