# SPDX-License-Identifier: MIT
r"""X-13ARIMA-SEATS spec serializer: :class:`X13spec` -> ``.spc`` text.

Mirrors ``TimeSeriesEcon.jl/src/x13/x13write.jl`` (294 LOC). Lands in
**M2.4** alongside :func:`tsecon.x13._spec.validateX13spec`.

Surface:

* :func:`x13write` (``x13write.jl:50``) — top-level emitter that walks
  the :class:`~tsecon.x13._spec.X13spec` and emits each populated
  sub-spec's text block.
* :func:`impose_line_length` (``x13write.jl:8``) — line-wrap helper
  that splits long argument lines so the X-13as binary does not reject
  the spec on its 132-column line limit. Mutates the input list in
  place (mirrors Julia's ``impose_line_length!`` ``!``-suffixed name).
* :func:`emit_block` (private) — render one ``<name> { ... }`` block
  from a populated sub-spec dataclass; the writer's main per-block
  worker.

The Julia version writes a ``spec.spc`` text file directly to disk; the
Python port returns the text as a :class:`str` so that callers that do
not need the file (testing, dry-run validation) skip the I/O. The
``X13.run`` entry (M2.5) calls :func:`x13write` then writes the
returned text to ``<tempdir>/spec.spc`` itself, mirroring Julia's
``open(...) do f; println(f, spec.string); end`` block.

Line-wrap quirks (mirrors Julia ``x13write.jl:8-47``):

* ``_text_len(s)`` treats each ``\t`` as eight columns (Julia
  ``_length``); the writer never emits tabs, but the wrapper handles
  them for round-trip fidelity with hand-edited spec strings.
* ``" + "`` is preferred as a split point over plain spaces because
  ``print=(table1 + table2 + …)`` lists are the most common
  long-argument shape (the Julia upstream's
  ``x13write_plus(::Vector{Symbol})`` produces them).
* When the leading whitespace-only fragment of a continuation line
  cannot fit a single un-splittable token, raise :exc:`ValueError`
  rather than recurse infinitely (Julia ``ArgumentError``).
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from tsecon.frequencies import Monthly
from tsecon.mit import MIT, mit2yp
from tsecon.mitrange import MITRange
from tsecon.mvtseries import MVTSeries
from tsecon.tseries import TSeries
from tsecon.x13._consts import _ORDERED_MONTH_NAMES
from tsecon.x13._spec import (
    _X13SPEC_SUBSPEC_FIELDS,
    ArimaModel,
    ArimaSpec,
    Span,
    X13arima,
    X13automdl,
    X13check,
    X13default,
    X13estimate,
    X13force,
    X13forecast,
    X13history,
    X13identify,
    X13metadata,
    X13outlier,
    X13pickmdl,
    X13regression,
    X13seats,
    X13series,
    X13slidingspans,
    X13spec,
    X13spectrum,
    X13transform,
    X13var,
    X13x11,
    X13x11regression,
    validateX13spec,
)

__all__ = [
    "impose_line_length",
    "x13write",
]


# ---------------------------------------------------------------------------
# Line-wrap helper
# ---------------------------------------------------------------------------


# The X-13as binary rejects spec lines longer than 132 columns. Inside a
# block each line is indented by 8 spaces ("        "), so the effective
# argument-content budget is 132 - 8 = 124. Mirrors Julia
# ``impose_line_length!`` default ``limit = 132 - 8`` (``x13write.jl:8``).
_DEFAULT_LINE_LIMIT: int = 132 - 8

# Continuation lines emitted by the wrapper carry the same 8-space indent
# (so the wrapped output mirrors the visual indent of the spec block).
_CONTINUATION_INDENT: str = "        "


def _text_len(s: str) -> int:
    """Length of ``s`` with tabs counted as 8 columns.

    Mirrors Julia ``_length`` (``x13write.jl:48``). The writer never
    emits tabs itself; the helper exists for round-trip fidelity with
    hand-authored or upstream-derived spec strings.
    """
    return len(s) + 7 * s.count("\t")


def impose_line_length(
    s: list[str],
    limit: int = _DEFAULT_LINE_LIMIT,
    delve: bool = True,
) -> None:
    r"""Wrap any over-limit lines in ``s`` in place.

    Mirrors Julia ``impose_line_length!`` (``x13write.jl:8-47``). Walks
    the list; for each line longer than ``limit`` columns, splits at
    the last ``" + "`` (when present) or otherwise the last space that
    keeps the prefix within budget, then inserts the suffix — prefixed
    with 8 spaces of continuation indent — at the next position.
    Recurses into ``"\n"``-joined sub-lines when ``delve=True`` (so
    embedded newlines, e.g. inside ``metadata`` blocks, get wrapped on
    their own terms before being rejoined).

    Raises :exc:`ValueError` when a continuation line cannot be split
    further (single un-splittable token over budget) — surfaces the
    Julia ``ArgumentError`` from ``x13write.jl:38-41``.
    """
    counter = 0
    while counter < len(s):
        line = s[counter]
        if delve and "\n" in line:
            sub_lines = line.split("\n")
            impose_line_length(sub_lines, limit, False)
            s[counter] = "\n".join(sub_lines)
            counter += 1
            continue
        if _text_len(line) <= limit:
            counter += 1
            continue

        splitchar = " + " if " + " in line else " "
        parts = line.split(splitchar)
        # Find the largest prefix that fits the budget. Julia computes a
        # running cumulative length and stops the first time it exceeds
        # the limit, then keeps everything up to ``i - 1`` on the
        # current line. The ``best_split_index == 0`` case is left
        # alone deliberately — Julia recurses through it until the
        # next-iteration "continuation is 8 whitespace columns wide"
        # guard fires and raises (see below).
        cum = 0
        best_split_index = 2
        for i, part in enumerate(parts, start=1):
            cum += _text_len(part) + _text_len(splitchar)
            if cum > limit:
                best_split_index = i - 1
                break

        s1 = splitchar.join(parts[:best_split_index]) + splitchar
        s2 = _CONTINUATION_INDENT + splitchar.join(parts[best_split_index:])
        if len(s1) == 8 and s1.strip() == "" and len(s2) > limit:
            msg = (
                f"Could not split the following line into components "
                f"shorter than {limit}. Please shorten the argument "
                f"length:\n{s2}"
            )
            raise ValueError(msg)
        s[counter] = s1
        s.insert(counter + 1, s2)
        counter += 1


# ---------------------------------------------------------------------------
# Value rendering
# ---------------------------------------------------------------------------


# Fields whose Julia upstream type is ``::String`` (rather than the more
# common ``::Symbol``). Values for these fields render with double quotes
# in .spc output; everything else gets rendered as a bare token. Mirrors
# the Julia per-method dispatch on ``x13write(::String)`` vs
# ``x13write(::Symbol)``.
_QUOTED_STRING_FIELDS: frozenset[str] = frozenset(
    {
        "file",
        "format",
        "name",
        "title",
        "umfile",
        "umformat",
        "umname",
    }
)

# Fields whose Julia upstream type is ``::Vector{String}`` (the rare
# multi-line quoted form). Only ``x11.title`` and ``x11regression.umname``
# meet this shape; everything else with a Python ``list[str]`` payload is
# Julia ``Vector{Symbol}`` and renders space-separated, unquoted.
_QUOTED_STRING_LIST_FIELDS: frozenset[str] = frozenset({"title", "umname"})


def _emit_value(val: Any, *, field: str | None = None) -> str:  # noqa: PLR0911, PLR0912
    """Render a single spec-field value to its ``.spc`` token.

    Mirrors the family of ``x13write(::T)`` methods at
    ``x13write.jl:235-289``. Dispatches on Python type rather than
    Julia's multimethod table.

    The optional ``field`` argument carries the name of the dataclass
    field being rendered, so the dispatcher can apply the
    quoted-string-vs-bare-symbol distinction without overloading the
    types module.
    """
    # bool first — bool ⊂ int in Python; reverse order would route True
    # / False through the int branch and emit ``1`` / ``0`` instead of
    # ``yes`` / ``no``.
    if isinstance(val, bool):
        return "yes" if val else "no"
    if isinstance(val, str):
        if field in _QUOTED_STRING_FIELDS:
            return f'"{val}"'
        return val
    if isinstance(val, X13var):
        return str(val)
    if isinstance(val, ArimaSpec):
        return _emit_arima_spec(val)
    if isinstance(val, ArimaModel):
        # ``Vector{ArimaSpec}`` renders as concatenated ``(p d q)(P D Q)``
        # forms — no separator. Mirrors Julia ``x13write(::Vector{ArimaSpec})``
        # at ``x13write.jl:244``.
        return "".join(_emit_arima_spec(s) for s in val.specs)
    if isinstance(val, MIT):
        return _emit_mit(val)
    if isinstance(val, MITRange):
        return f"({_emit_mit(val.first())}, {_emit_mit(val.last())})"
    if isinstance(val, Span):
        return _emit_span(val)
    if isinstance(val, TSeries):
        return _emit_tseries(val)
    if isinstance(val, MVTSeries):
        return _emit_mvtseries(val)
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        return str(val)
    if val is None:
        # Mirrors Julia ``x13write(::Missing) = ""`` (``x13write.jl:271``).
        # Used by ``arima.ar`` / ``arima.ma`` fixed-flag tuples to render
        # "no initial value" entries as empty strings inside the
        # parenthesised list.
        return ""
    if isinstance(val, list):
        return _emit_list(val, field=field)
    if isinstance(val, tuple):
        return _emit_list(list(val), field=field)
    msg = f"x13write: no .spc rendering for value of type {type(val).__name__}: {val!r}"
    raise TypeError(msg)


def _emit_pickmdl_list(models: list[ArimaModel]) -> str:
    """Render a list of :class:`ArimaModel` for the pickmdl ``models`` field.

    Mirrors Julia ``x13write(::Vector{ArimaModel})`` (``x13write.jl:245-253``).
    Every model except the last gets a trailing ``" *"`` (default flag)
    or ``" X"`` (non-default) suffix; the last model gets no suffix.
    """
    lines: list[str] = []
    for m in models[:-1]:
        suffix = " *" if m.default else " X"
        lines.append(f"{_emit_value(m)}{suffix}")
    lines.append(_emit_value(models[-1]))
    return "\n".join(lines)


def _emit_arima_spec(val: ArimaSpec) -> str:
    """Render an :class:`ArimaSpec` as ``(p d q)`` or ``(p d q)period``.

    Mirrors Julia ``x13write(::ArimaSpec)`` (``x13write.jl:243``).
    Tuple-shaped p/d/q components serialize as space-separated lists
    inside square brackets, e.g. ``([2 3] 0 0)`` for
    ``ArimaSpec((2, 3), 0, 0)``.
    """

    def _pdq(x: int | tuple[int, ...] | X13default) -> str:
        if isinstance(x, X13default):
            return "0"
        if isinstance(x, tuple):
            return "[" + " ".join(str(k) for k in x) + "]"
        return str(x)

    body = f"({_pdq(val.p)} {_pdq(val.d)} {_pdq(val.q)})"
    if isinstance(val.period, X13default) or val.period == 0:
        return body
    return f"{body}{val.period}"


def _emit_mit(val: MIT) -> str:
    """Render an MIT to ``year.period`` (or ``year`` for Yearly).

    Mirrors ``x13write.jl:286-295``. Monthly MITs render as
    ``year.<month-abbr>`` (``2020.jul``); other frequencies use the
    bare period number.
    """
    year_, period_ = mit2yp(val)
    if isinstance(val.frequency, Monthly):
        return f"{year_}.{_ORDERED_MONTH_NAMES[period_ - 1]}"
    return f"{year_}.{period_}"


def _emit_span(val: Span) -> str:
    """Render a :class:`Span` as ``(b, e)``.

    Mirrors ``x13write.jl:282``. :data:`None` endpoints render as the
    empty string (Julia ``missing``) — this is the on-disk shape the
    binary expects for ``"open"`` endpoints.
    """
    b = _emit_value(val.b) if val.b is not None else ""
    e = _emit_value(val.e) if val.e is not None else ""
    return f"({b}, {e})"


def _emit_tseries(val: TSeries) -> str:
    """Render a TSeries data argument as ``(v1 v2 v3 …)``.

    Mirrors ``x13write.jl:254``.
    """
    return "(" + " ".join(_emit_number(v) for v in val.values) + ")"


def _emit_mvtseries(val: MVTSeries) -> str:
    """Render an MVTSeries as the multi-column ``.spc`` data block.

    Mirrors ``x13write.jl:255-268``. A single-column MVTSeries collapses
    to the TSeries shape; multi-column MVTSeries iterate row-by-row,
    joining row values with 8-space spacing.
    """
    if val.shape[1] == 1:
        first_col = val.column_names[0]
        return _emit_tseries(val.columns[first_col])
    rows: list[str] = []
    for row in val.values:
        rows.append("        ".join(_emit_number(v) for v in row))
    body = "\n        ".join(rows)
    return f"(        {body}        )"


def _emit_number(x: Any) -> str:
    """Render a single scalar value (used inside data lists)."""
    if isinstance(x, (np.floating, float)):
        return str(float(x))
    if isinstance(x, (np.integer, int)):
        return str(int(x))
    msg = f"x13write: cannot render numeric value of type {type(x).__name__}: {x!r}"
    raise TypeError(msg)


def _emit_list(val: list[Any], *, field: str | None = None) -> str:
    """Render a list of values as ``(v1, v2, …)`` (or the variant per element type).

    Mirrors ``x13write.jl:272-275``. The Julia upstream has three
    parallel signatures:

    * ``Vector{Symbol|X13var}`` → space-separated, bare tokens.
    * ``Vector{String}`` → newline-separated, each entry quoted.
    * Generic ``Vector{<:Any>`` → comma-and-space-separated.

    The Python port dispatches by inspecting element type at runtime
    PLUS the field name — ``list[str]`` belonging to a field in
    :data:`_QUOTED_STRING_LIST_FIELDS` renders in the quoted form;
    everything else stays bare (the Julia ``Vector{Symbol}`` default).
    Empty lists render as ``()``.
    """
    if not val:
        return "()"
    is_strings = all(isinstance(v, str) and not isinstance(v, X13var) for v in val)
    if is_strings:
        if field in _QUOTED_STRING_LIST_FIELDS:
            body = "\n        ".join(f'"{v}"' for v in val)
            return f"({body})"
        body = " ".join(val)
        return f"({body})"
    if all(isinstance(v, (str, X13var)) for v in val):
        body = " ".join(_emit_value(v) for v in val)
        return f"({body})"
    body = ", ".join(_emit_value(v) for v in val)
    return f"({body})"


def _emit_alt(val: list[str] | bool) -> str:
    """Render a value via the Julia ``x13write_alt`` overloads.

    Mirrors ``x13write_alt`` (``x13write.jl:237`` & ``240``). Used for
    the handful of fields whose .spc form diverges from the generic
    list / bool rendering:

    * ``Vector{Symbol}`` → ``"a,b,c"`` (comma-separated quoted string).
    * ``Bool`` → ``1`` / ``0`` (numeric, no ``yes``/``no``).
    """
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, list):
        return '"' + ",".join(str(v) for v in val) + '"'
    msg = (  # type: ignore[unreachable]
        f"x13write_alt: unsupported value of type {type(val).__name__}"
    )
    raise TypeError(msg)


def _emit_plus(val: list[str] | str) -> str:
    """Render a ``print=`` list with ``+`` separators.

    Mirrors ``x13write_plus(::Vector{Symbol})`` (``x13write.jl:238``).
    ``print=(table1 + table2 + …)`` is the X-13as grammar for unioning
    selected tables; the impose_line_length helper prefers splitting
    here when the list overflows the column budget.

    Bare-string scalar values fall through to the parenthesised
    one-element form to mirror Julia's ``Symbol`` dispatch
    (``print=(:default)`` → ``print = (default)``).
    """
    if isinstance(val, str):
        return f"({val})"
    body = " + ".join(str(v) for v in val)
    return f"({body})"


# ---------------------------------------------------------------------------
# Per-spec block emission
# ---------------------------------------------------------------------------


# Names emitted as keys in the .spc — the field-name → spec-keyword
# rewrite (Julia ``x13write.jl:133`` for ``func`` and the keys-at-end
# handling).
_KEY_RENAMES: dict[str, str] = {
    "func": "function",
    "lambda_": "lambda",  # Python reserved-word rename in force()
}

_KEYS_AT_END: frozenset[str] = frozenset({"ma", "ar", "b", "aictest"})
_KEYS_ALT: frozenset[str] = frozenset({"printphtrf", "tabtables"})
_KEYS_PLUS: frozenset[str] = frozenset({"print"})

# Spec types whose serialization runs through the generic emitter
# (``_emit_generic_block``). The series + metadata blocks have
# specialised emitters because their key structure diverges.
_SPEC_BLOCK_NAMES: dict[type, str] = {
    X13arima: "arima",
    X13automdl: "automdl",
    X13check: "check",
    X13estimate: "estimate",
    X13force: "force",
    X13forecast: "forecast",
    X13history: "history",
    X13identify: "identify",
    X13outlier: "outlier",
    X13pickmdl: "pickmdl",
    X13regression: "regression",
    X13seats: "seats",
    X13slidingspans: "slidingspans",
    X13spectrum: "spectrum",
    X13transform: "transform",
    X13x11: "x11",
    X13x11regression: "x11regression",
}


def _is_set(val: Any) -> bool:
    """Return True iff a sub-spec field carries a non-default value."""
    return not isinstance(val, X13default)


def _block_field_iter(spec: Any) -> Iterable[tuple[str, Any]]:
    """Iterate (field-name, value) pairs in container declaration order."""
    for f in spec.__dataclass_fields__:
        yield f, getattr(spec, f)


def _emit_fixed_values(spec: Any, key: str, val: list[Any]) -> str:
    """Render a list of values with their ``fix<key>`` overrides interleaved.

    Mirrors Julia ``x13write_fixed_values`` (``x13write.jl:224-232``).
    Used for the ``ar`` / ``ma`` / ``b`` fields on the arima /
    regression / x11regression specs: ``fixar=[True, False, True]``
    appends ``f`` / blank / ``f`` to each entry, producing
    ``(0.3f, 0.2, 0.5f)``.
    """
    fix_attr = f"fix{key}"
    fix_val = getattr(spec, fix_attr, None)
    if fix_val is None or isinstance(fix_val, X13default):
        return _emit_value(val, field=key)
    pieces: list[str] = []
    for v, f in zip(val, fix_val, strict=False):
        suffix = "f" if f else ""
        pieces.append(f"{_emit_value(v, field=key)}{suffix}")
    return "(" + ",".join(pieces) + ")"


def _emit_generic_block(  # noqa: PLR0912
    spec: Any,
    name: str,
    *,
    test: bool,
    outfolder: str | None,
) -> str:
    """Render a generic sub-spec block ``<name> { …key = value… }``.

    Mirrors Julia ``x13write(::Union{X13arima, …})`` (``x13write.jl:120-175``).
    """
    parts: list[str] = []
    keys_at_end: list[str] = []
    for key, val in _block_field_iter(spec):
        if test and key in ("print", "save", "savelog"):
            continue
        if key in ("fixar", "fixma", "fixb"):
            continue
        if not _is_set(val):
            continue
        out_key = _KEY_RENAMES.get(key, key)
        if key in _KEYS_ALT:
            parts.append(f"{out_key} = {_emit_alt(val)}")
            continue
        if key in _KEYS_PLUS:
            parts.append(f"{out_key} = {_emit_plus(val)}")
            continue
        if isinstance(spec, X13pickmdl) and key == "models":
            if outfolder is not None and len(outfolder):
                mdl_string = _emit_pickmdl_list(val) + "\n"
                mdl_path = Path(outfolder) / "pickmdl.mdl"
                mdl_path.write_text(mdl_string, encoding="utf-8")
                parts.append(f'file = "{os.fspath(mdl_path)}"')
            else:
                parts.append(f"{out_key} = {_emit_pickmdl_list(val)}")
            continue
        if key in _KEYS_AT_END:
            keys_at_end.append(key)
            continue
        parts.append(f"{out_key} = {_emit_value(val, field=key)}")

    for key in keys_at_end:
        val = getattr(spec, key)
        if key in ("ma", "ar", "b"):
            parts.append(f"{key} = {_emit_fixed_values(spec, key, val)}")
        else:
            parts.append(f"{key} = {_emit_value(val, field=key)}")

    impose_line_length(parts)
    if not parts:
        return f"{name} {{ }}"
    body = "\n        ".join(parts)
    return f"{name} {{\n        {body}\n}}"


def _emit_series_block(
    spec: X13series,
    *,
    test: bool,
) -> str:
    """Render the ``series { ... }`` block.

    Mirrors Julia ``x13write(::X13series)`` (``x13write.jl:177-194``).
    Skips the ``print``/``save``/``savelog`` fields when ``test=True``
    (the test-mode round-trip path skips outputs that depend on
    binary-side defaults).
    """
    parts: list[str] = []
    for key, val in _block_field_iter(spec):
        if test and key in ("print", "save", "savelog"):
            continue
        if not _is_set(val):
            continue
        if key in _KEYS_PLUS:
            parts.append(f"{key} = {_emit_plus(val)}")
            continue
        parts.append(f"{key} = {_emit_value(val, field=key)}")
    impose_line_length(parts)
    if not parts:
        return "series { }"
    body = "\n        ".join(parts)
    return f"series {{\n        {body}\n}}"


def _emit_metadata_block(spec: X13metadata) -> str:
    """Render the ``metadata { ... }`` block.

    Mirrors Julia ``x13write(::X13metadata)`` (``x13write.jl:196-218``).
    The Julia upstream stores `(key, value)` pairs in the `entries`
    field and renders them as either a single ``key = "k"`` /
    ``value = "v"`` pair (single-entry case) or two parenthesised lists
    of quoted strings (multi-entry case).
    """
    keys_vec = [p[0] for p in spec.entries]
    vals_vec = [p[1] for p in spec.entries]
    parts: list[str] = []
    if len(keys_vec) == 1:
        # Both key and value are ``::String`` in Julia (always quoted).
        parts.append(f'key = "{keys_vec[0]}"')
        parts.append(f'value = "{vals_vec[0]}"')
    else:
        parts.append("key = (")
        for k in keys_vec:
            parts.append(f'        "{k}"')
        parts.append(")")
        parts.append("value = (")
        for v in vals_vec:
            parts.append(f'        "{v}"')
        parts.append(")")
    impose_line_length(parts)
    body = "\n        ".join(parts)
    return f"metadata {{\n        {body}\n}}"


# ---------------------------------------------------------------------------
# Top-level x13write entry
# ---------------------------------------------------------------------------


def x13write(
    spec: X13spec,
    *,
    test: bool = False,
    outfolder: str | None = None,
) -> str:
    r"""Serialize an :class:`X13spec` to ``.spc`` text.

    Mirrors Julia ``x13write(::X13spec)`` (``x13write.jl:50-80``). Runs
    :func:`validateX13spec` first; on success, walks ``series`` followed
    by each populated sub-spec field in declaration order, emits each
    block, and joins them with newlines.

    ``test=True`` skips the ``print`` / ``save`` / ``savelog`` fields
    on each block — useful for round-trip and validation tests that
    don't want to lock the per-binary output defaults.

    The rendered text is stored on :attr:`X13spec.string` as a
    side-effect (mirrors Julia ``spec.string = join(s, "\n")``); the
    function additionally returns the text so callers can pipe it to
    a file or to the binary directly.

    ``outfolder`` is the directory the M2.5 binary runner created;
    used by the ``pickmdl`` block to emit a side-file (``pickmdl.mdl``)
    rather than inline the model list. If :data:`None`, the model list
    is inlined.
    """
    validateX13spec(spec)
    if not isinstance(spec.series, X13series):
        msg = "x13write: spec.series must be an X13series; build via newspec(...)."
        raise ValueError(msg)

    blocks: list[str] = [_emit_series_block(spec.series, test=test)]
    for fname in _X13SPEC_SUBSPEC_FIELDS:
        val = getattr(spec, fname)
        if not _is_set(val):
            continue
        if isinstance(val, X13metadata):
            blocks.append(_emit_metadata_block(val))
            continue
        block_name = _SPEC_BLOCK_NAMES.get(type(val))
        if block_name is None:
            msg = f"x13write: no block emitter for sub-spec type {type(val).__name__}."
            raise TypeError(msg)
        blocks.append(_emit_generic_block(val, block_name, test=test, outfolder=outfolder))

    text = "\n".join(blocks)
    spec.string = text
    return text
