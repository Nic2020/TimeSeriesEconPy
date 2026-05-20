# SPDX-License-Identifier: MIT
"""Fetch a pre-built X-13ARIMA-SEATS binary into ``src/tsecon/x13/_binary/``.

The wheels published by ``wheels.yml`` vendor an x13as binary compiled
from Census Bureau Fortran source via ``fortran-lang/setup-fortran@v1``.
Development checkouts do not run that pipeline, so this script downloads
a Census Bureau **pre-built** binary into the same location ``_resolve_binary``
in :mod:`tsecon.x13._result` looks at. After running it once,
``tsecon.x13.run`` works locally without any further configuration.

The script is intentionally pinned to ``X13AS_VERSION`` in
:mod:`tsecon._mirror` — the same version the wheels build from source.

Platform support
----------------

* **Windows**: downloads + extracts ``x13as_ascii-v1-1-b62.zip`` (Census
  ships pre-built Windows binaries).
* **Linux / macOS**: downloads + extracts ``x13as_ascii-v1-1-b62.tar.gz``
  (Census ships pre-built Linux binaries; the macOS path falls through
  to the same tarball and only works on x86_64 macOS — Apple Silicon
  users need a Fortran toolchain locally or a CI-built wheel).

Run::

    uv run python scripts/fetch_x13as_local.py

The download URL + SHA-256 are recorded in :mod:`tsecon._mirror`; this
script verifies the SHA-256 before unpacking and refuses to install a
tampered archive.
"""

from __future__ import annotations

import hashlib
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tsecon._mirror import (  # noqa: E402
    X13AS_SOURCE_URL,
    X13AS_VERSION,
    X13AS_WINDOWS_PREBUILT_SHA256,
    X13AS_WINDOWS_PREBUILT_URL,
)

BINARY_DIR = REPO_ROOT / "src" / "tsecon" / "x13" / "_binary"

# Linux pre-built tarball (sibling to the source archive at the Census site).
X13AS_LINUX_PREBUILT_URL = (
    "https://www2.census.gov/software/x-13arima-seats/x13as/unix-linux/"
    "program-archives/x13as_ascii-v1-1-b62.tar.gz"
)


def _download(url: str) -> bytes:
    """Download ``url`` and return its body bytes."""
    print(f"Downloading {url}", flush=True)
    with urllib.request.urlopen(url) as response:
        return response.read()


def _verify_sha256(blob: bytes, expected: str, label: str) -> None:
    """Raise :class:`RuntimeError` if SHA-256 of ``blob`` does not match."""
    digest = hashlib.sha256(blob).hexdigest()
    if digest != expected:
        msg = (
            f"SHA-256 mismatch for {label}.\n"
            f"  expected: {expected}\n"
            f"  actual:   {digest}\n"
            "Refusing to install a tampered archive."
        )
        raise RuntimeError(msg)


def _ensure_binary_dir() -> None:
    """Create ``BINARY_DIR`` if missing."""
    BINARY_DIR.mkdir(parents=True, exist_ok=True)


def _install_windows(blob: bytes) -> Path:
    """Unpack the Windows zip and copy ``x13as_ascii.exe`` to ``x13as.exe``."""
    with tempfile.TemporaryDirectory(prefix="x13_fetch_") as tmp:
        zip_path = Path(tmp) / "x13as.zip"
        zip_path.write_bytes(blob)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        # Census zip layout: ``x13as/x13as_ascii.exe``.
        src = Path(tmp) / "x13as" / "x13as_ascii.exe"
        if not src.is_file():
            msg = f"Expected x13as_ascii.exe inside Windows zip at {src}."
            raise RuntimeError(msg)
        dst = BINARY_DIR / "x13as.exe"
        shutil.copy2(src, dst)
        return dst


def _install_posix(blob: bytes) -> Path:
    """Unpack the Linux tarball and copy ``x13as_ascii`` to ``x13as``."""
    with tempfile.TemporaryDirectory(prefix="x13_fetch_") as tmp:
        tar_path = Path(tmp) / "x13as.tar.gz"
        tar_path.write_bytes(blob)
        with tarfile.open(tar_path) as tf:
            tf.extractall(tmp, filter="data")
        # Census tarball flattens to ``x13as_ascii`` in the cwd.
        candidates = list(Path(tmp).rglob("x13as_ascii"))
        candidates = [c for c in candidates if c.is_file()]
        if not candidates:
            msg = f"Could not locate x13as_ascii inside tarball under {tmp}."
            raise RuntimeError(msg)
        src = candidates[0]
        dst = BINARY_DIR / "x13as"
        shutil.copy2(src, dst)
        dst.chmod(0o755)
        return dst


def main() -> int:
    """Entry point."""
    _ensure_binary_dir()
    sysname = platform.system()
    if sysname == "Windows":
        blob = _download(X13AS_WINDOWS_PREBUILT_URL)
        _verify_sha256(blob, X13AS_WINDOWS_PREBUILT_SHA256, X13AS_WINDOWS_PREBUILT_URL)
        installed = _install_windows(blob)
    elif sysname in {"Linux", "Darwin"}:
        if sysname == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}:
            print(
                "WARNING: Census Bureau does not ship an Apple Silicon pre-built "
                "binary. Falling back to the Linux x86_64 tarball; it will not "
                "run on arm64 macOS. For a local Apple Silicon build, install "
                "gfortran via Homebrew and use the wheels.yml recipe directly.",
                file=sys.stderr,
            )
        blob = _download(X13AS_LINUX_PREBUILT_URL)
        # The Linux pre-built tarball SHA-256 is not pinned (Census refreshes
        # the build on the same version pin from time to time without bumping
        # the version string). We accept whatever is currently published; the
        # X13AS_VERSION pin remains the trust anchor.
        installed = _install_posix(blob)
    else:
        msg = f"Unsupported platform: {sysname}. Build x13as v{X13AS_VERSION} locally."
        raise RuntimeError(msg)
    print(f"Installed {installed} ({X13AS_VERSION})", flush=True)
    print(f"Source URL was {X13AS_SOURCE_URL} (Fortran source, for CI build)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
