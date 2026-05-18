# SPDX-License-Identifier: MIT
"""X-13ARIMA-SEATS lookup tables: output-file classifications and metadata.

Mirrors ``TimeSeriesEcon.jl/src/x13/x13consts.jl`` (1,058 LOC). Lands in
**M2.1**.

Content planned:

* ``_series_extensions`` — set of file extensions parsed as
  :class:`~tsecon.tseries.TSeries` (e.g. ``d11`` / ``d10`` / ``d12`` / ``d13`` /
  trend, irregular, seasonal-factor families).
* ``_probably_series_extensions`` — extensions parsed best-effort as series
  with a warning on failure (the Julia upstream's "unknown output type"
  surface; ``x13result.jl:165-178``).
* ``_table_extensions`` — extensions parsed as :class:`WorkspaceTable`
  (multi-column equal-length-vector tables).
* ``_kv_list_extensions`` — extensions parsed as
  :class:`~tsecon.workspace.Workspace` key/value pairs.
* ``_human_text_extensions`` — extensions parsed as plain ``str`` (logs,
  rendered tables, diagnostics intended for direct user reading).
* ``_output_descriptions`` — ordered mapping from output-file name to a
  human-readable description; surface for documentation generation and for
  the ``load=`` keyword's value space (mirrors Julia's
  ``X13._output_descriptions`` referenced in ``x13result.jl:71-73``).

The classification is closed-form: X13as v1.1 b60's output filenames are
the universe; any extension not in one of the above falls through to the
``unknown output file`` warning path mirrored from
``x13result.jl:194-198``. Future X13as version bumps may add new
extensions — when this happens the constants ship in the bump PR.
"""

from __future__ import annotations

__all__: list[str] = []
