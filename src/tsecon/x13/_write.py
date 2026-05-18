# SPDX-License-Identifier: MIT
"""X-13ARIMA-SEATS spec serializer: :class:`X13spec` -> ``.spc`` text.

Mirrors ``TimeSeriesEcon.jl/src/x13/x13write.jl`` (294 LOC). Lands in **M2.4**
alongside :func:`tsecon.x13._spec.validateX13spec`.

Content planned:

* :func:`x13write` (``x13write.jl:1-...``) — top-level emitter that walks
  the :class:`~tsecon.x13._spec.X13spec` and emits each populated
  sub-spec's text block.
* :func:`_emit_block` (private) — one X13as block (e.g. ``series { ...
  }`` / ``x11 { ... }`` / ``arima { model = ... }``) with the canonical
  X13as indentation + ``=`` alignment.
* :func:`_format_var` (private) — :class:`~tsecon.x13._spec.X13var`
  dispatcher; emits the ``.spc``-grammar token for each leaf type.
* :func:`impose_line_length` (private) — line-wrap long argument lists
  (the X13as binary rejects spec lines >132 columns; mirrors Julia's
  ``impose_line_length!``).

The Julia version writes a ``spec.spc`` text file directly to disk; the
Python port returns the text as a ``str`` so that callers that don't
need the file (testing, dry-run validation) skip the I/O. The
``X13.run`` entry calls :func:`x13write` then writes the returned text
to ``<tempdir>/spec.spc`` itself.

Edge cases that need explicit tests in M2.4:

* Long :func:`tsecon.x13._spec.regression` ``variables=...`` lists —
  the line-wrapping is the most fragile part of the serializer (X13as
  is whitespace-sensitive in obscure ways; Julia's
  ``impose_line_length!`` was the source of two bugs in the upstream's
  test suite history).
* Empty sub-specs — the spec object can carry a ``transform=transform()``
  with no arguments; the serializer should emit ``transform { }``
  rather than omit the block (the Julia upstream's invariant).
* Comment lines and blank lines in spec strings — must survive the
  round-trip (some users hand-write spec strings; the validation
  path strips comments but the write path preserves them).
"""

from __future__ import annotations

__all__: list[str] = []
