# SPDX-License-Identifier: MIT
"""Mirror metadata: which TimeSeriesEcon.jl commit this version reflects.

The CI workflow ``.github/workflows/upstream-sync.yml`` reads this constant,
diffs the pinned SHA against the upstream default branch, and opens / updates a
tracking issue when there are unported changes.

When porting an upstream change, update ``MIRRORS_JULIA_SHA`` to the new
commit and close the associated tracking issue.
"""

from __future__ import annotations

from typing import Final

MIRRORS_JULIA_SHA: Final[str] = "fc0a0d01aed4903ea6c64b12d78d0bdf68468df6"
"""Commit SHA of ``bankofcanada/TimeSeriesEcon.jl`` mirrored by this release."""

MIRRORS_JULIA_DATE: Final[str] = "2025-12-22"
"""ISO-8601 date of ``MIRRORS_JULIA_SHA``, for human reference."""

MIRRORS_JULIA_REPO: Final[str] = "bankofcanada/TimeSeriesEcon.jl"
"""GitHub ``owner/name`` of the mirrored upstream repository."""

X13AS_VERSION: Final[str] = "v1-1-b62"
"""Pinned X-13ARIMA-SEATS version used by the wheels build (M2.6).

The Census Bureau source archive at
``https://www2.census.gov/software/x-13arima-seats/x13as/unix-linux/program-archives/x13as_asciisrc-v1-1-b62.tar.gz``
is downloaded by ``wheels.yml`` and compiled in each per-platform job via
``fortran-lang/setup-fortran@v1``. The same version is fetched by
``scripts/fetch_x13as_local.py`` for local development.

Bumped from ``v1-1-b60`` (decision 24 sub-decision 2's original pin) to
``v1-1-b62`` in M2.6 (session 54) after Census Bureau dropped b60 from the
public archive. Yggdrasil still ships b60; the fidelity test compensates by
running both the Julia and Python sides through the same locally-resolved
binary via ``getoption("x13path")``.
"""

X13AS_SOURCE_URL: Final[str] = (
    "https://www2.census.gov/software/x-13arima-seats/x13as/"
    "unix-linux/program-archives/x13as_asciisrc-v1-1-b62.tar.gz"
)
"""Census Bureau source tarball URL for ``X13AS_VERSION``."""

X13AS_SOURCE_SHA256: Final[str] = "82796c84b54891474df1a6ffb16be7bf7bd4a992e3679158645c888e6cddda12"
"""SHA-256 of ``X13AS_SOURCE_URL`` (verified 2026-05-20).

Lifted from the live Census archive listing; re-check on any version bump.
``wheels.yml``'s download step pipes the response through ``sha256sum -c``
against this constant so a Census-side replacement cannot silently change
the binary inputs.
"""

X13AS_WINDOWS_PREBUILT_URL: Final[str] = (
    "https://www2.census.gov/software/x-13arima-seats/x13as/"
    "windows/program-archives/x13as_ascii-v1-1-b62.zip"
)
"""Census Bureau pre-built Windows zip for ``X13AS_VERSION``.

Used by ``scripts/fetch_x13as_local.py`` on Windows so a Fortran toolchain
is not required for local development. Not used by ``wheels.yml`` (the CI
build compiles from source for parity with the Linux / macOS jobs).
"""

X13AS_WINDOWS_PREBUILT_SHA256: Final[str] = (
    "c6bd65132a3219555d00abf794649e75c133f23c0db066ed03c0b5ca30e694c2"
)
"""SHA-256 of ``X13AS_WINDOWS_PREBUILT_URL`` (verified 2026-05-20)."""
