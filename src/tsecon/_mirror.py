# SPDX-License-Identifier: MIT
"""Mirror metadata: which TimeSeriesEcon.jl commit this version reflects.

The CI workflow ``.github/workflows/upstream-sync.yml`` reads this constant,
diffs the pinned SHA against the upstream default branch, and opens / updates a
tracking issue when there are unported changes.

When porting an upstream change, update ``MIRRORS_JULIA_SHA`` to the new
commit and close the associated tracking issue. See
``claude_files/decisions/09_version_mirror_tracking.md`` for the rationale.
"""

from __future__ import annotations

from typing import Final

MIRRORS_JULIA_SHA: Final[str] = "fc0a0d01aed4903ea6c64b12d78d0bdf68468df6"
"""Commit SHA of ``bankofcanada/TimeSeriesEcon.jl`` mirrored by this release."""

MIRRORS_JULIA_DATE: Final[str] = "2025-12-22"
"""ISO-8601 date of ``MIRRORS_JULIA_SHA``, for human reference."""

MIRRORS_JULIA_REPO: Final[str] = "bankofcanada/TimeSeriesEcon.jl"
"""GitHub ``owner/name`` of the mirrored upstream repository."""
