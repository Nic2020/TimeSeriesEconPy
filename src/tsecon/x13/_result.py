# SPDX-License-Identifier: MIT
"""X-13ARIMA-SEATS result containers, output parsers, and binary runner.

Mirrors ``TimeSeriesEcon.jl/src/x13/x13result.jl`` (1,035 LOC) and the
result-side surface from ``X13.jl`` (``WorkspaceTable``,
``X13ResultWorkspace``). Lands in **M2.5** (session 53).

Surface
-------

* :class:`WorkspaceTable` — :class:`~tsecon.workspace.Workspace` subclass
  that prints as a column-aligned table (every entry is an equal-length
  list / array). Mirrors Julia ``WorkspaceTable`` at ``X13.jl:12-22``.
* :class:`X13ResultWorkspace` — :class:`~tsecon.workspace.Workspace`
  subclass with lazy attribute / key access: an entry stored as an
  :class:`X13lazy` is materialised into the parsed object on first
  access and written back into ``_c`` so subsequent accesses are
  cheap. Mirrors Julia ``X13ResultWorkspace`` at ``X13.jl:24-35`` +
  ``x13result.jl:49-56``.
* :class:`X13lazy` — frozen 3-tuple ``(file, ext, frequency)`` placeholder
  for an output that has not been parsed yet. Mirrors
  ``x13result.jl:43-47``.
* :class:`X13result` — top-level result container with ``spec`` /
  ``outfolder`` / ``series`` / ``tables`` / ``text`` / ``other`` /
  ``stdout`` / ``errors`` / ``warnings`` / ``notes`` fields. Mirrors
  ``x13result.jl:23-40``. The temp ``outfolder`` is removed by a
  :func:`weakref.finalize` callback when the result is garbage-collected
  (mirrors Julia's ``finalizer`` at ``x13result.jl:37-38``); the
  callback uses an exponential-backoff retry loop on Windows
  :class:`PermissionError` (locked in decision 24 sub-decision 4).
* :func:`run` — execute the X-13 binary on a populated :class:`X13spec`
  (or a raw spec string + frequency) and parse the resulting output
  directory. Mirrors ``x13result.jl:76-237``.
* :func:`loadresult` — :class:`X13lazy` to parsed-object dispatcher.
  Mirrors ``x13result.jl:240-285``.
* Eight per-format readers: :func:`x13read_series`,
  :func:`x13read_workspace_table`, :func:`x13read_key_values`,
  :func:`x13read_udg`, :func:`x13read_estimates`, :func:`x13read_model`,
  :func:`x13read_identify`, :func:`x13read_seatsseries`,
  :func:`x13read_err`.

The Julia ``run`` accepts ``load::Union{Symbol, Vector{Symbol}}``; the
Python port accepts ``load: str | Sequence[str] = "none"`` with the
sentinel values ``"none"`` / ``"all"`` matching Julia's ``:none`` /
``:all``.

Cleanup
-------

Two cleanup paths exist (mirroring the Julia upstream):

* **Per-result cleanup.** Each :class:`X13result` registers a
  :func:`weakref.finalize` callback at construction that removes its
  ``outfolder`` when the result is GC'd. The callback retries
  :func:`shutil.rmtree` with an exponential backoff
  (``0.1 / 0.2 / 0.4`` s) on Windows :class:`PermissionError` to let
  antivirus / Defender release file handles; on the final retry it
  falls back to ``ignore_errors=True`` so a stuck cleanup never
  raises.
* **Stale-folder sweep.** :func:`tsecon.x13.cleanup` (in :mod:`_x13`)
  removes ``x13_*`` folders from :func:`tempfile.gettempdir` that the
  process leaked (e.g. due to a hard kill before the finalizer ran).

Concurrency note
----------------

The Julia upstream wraps the X-13 binary invocation in
``cd(spec.folder) do ... end`` (``x13result.jl:108``), which changes
the process-wide cwd and does not play well with concurrent runs. The
Python port passes ``cwd=spec.folder`` to
:func:`subprocess.run` instead — per-subprocess scoping, so concurrent
:func:`run` calls from threads are safe.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import warnings
import weakref
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from tsecon._options import getoption
from tsecon.frequencies import Frequency, YPFrequency
from tsecon.mit import MIT
from tsecon.mitrange import MITRange
from tsecon.mvtseries import MVTSeries
from tsecon.tseries import TSeries
from tsecon.workspace import Workspace
from tsecon.x13._consts import (
    _HUMAN_TEXT_EXTENSIONS,
    _KV_LIST_EXTENSIONS,
    _MONTHS_AND_QUARTERS,
    _PROBABLY_SERIES_EXTENSIONS,
    _SERIES_EXTENSIONS,
    _TABLE_EXTENSIONS,
)
from tsecon.x13._spec import ArimaModel, ArimaSpec, X13default, X13series, X13spec
from tsecon.x13._write import x13write

__all__ = [
    "WorkspaceTable",
    "X13ResultWorkspace",
    "X13lazy",
    "X13result",
    "loadresult",
    "run",
    "x13read_err",
    "x13read_estimates",
    "x13read_identify",
    "x13read_key_values",
    "x13read_model",
    "x13read_seatsseries",
    "x13read_series",
    "x13read_udg",
    "x13read_workspace_table",
]


# ---------------------------------------------------------------------------
# WorkspaceTable
# ---------------------------------------------------------------------------


class WorkspaceTable(Workspace):
    """A :class:`Workspace` whose entries are equal-length vectors.

    Mirrors Julia ``WorkspaceTable`` at ``X13.jl:12-22``. The X-13 binary
    emits a number of multi-column tabular outputs (autocorrelation
    plots, sliding-spans summaries, R/I/O tables, …) where the natural
    Python representation is "a dict of equal-length lists". The
    underlying storage is the inherited ``_c`` dict; what differs is
    :meth:`__repr__`, which prints as a column-aligned table rather
    than the keyed list :class:`Workspace` produces.

    Construction mirrors the parent: ``WorkspaceTable()``,
    ``WorkspaceTable(a=[1, 2], b=[3, 4])``,
    ``WorkspaceTable({"a": [1, 2]})``. No validation runs at construction
    that columns have equal lengths — the X-13 readers always produce
    balanced columns, and unbalanced columns simply render with blank
    cells past their length (mirrors Julia).
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return _format_workspace_table(self)

    def __str__(self) -> str:
        return _format_workspace_table(self)


def _format_workspace_table(w: WorkspaceTable) -> str:
    """Render a WorkspaceTable as a column-aligned table.

    Mirrors Julia ``Base.show(::IO, ::MIME"text/plain", ::WorkspaceTable)``
    at ``x13result.jl:747-877``. The Python version drops the screen-size
    truncation logic — Python repr is typically captured to a string and
    formatted by the consumer, so we just print everything.
    """
    if len(w) == 0:
        return "Empty WorkspaceTable"
    headers = list(w._c.keys())
    cols: list[list[str]] = []
    for v in w._c.values():
        if isinstance(v, np.ndarray):
            cols.append([_compact_str(x) for x in v.tolist()])
        elif isinstance(v, (list, tuple)):
            cols.append([_compact_str(x) for x in v])
        else:
            cols.append([_compact_str(v)])
    numrows = max((len(c) for c in cols), default=0)
    col_widths = [
        max(len(headers[i]), max((len(s) for s in cols[i]), default=0)) for i in range(len(headers))
    ]
    lines: list[str] = []
    lines.append(" ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers))))
    lines.append(" ".join("-" * col_widths[i] for i in range(len(headers))))
    for r in range(numrows):
        row_parts: list[str] = []
        for i, col in enumerate(cols):
            cell = col[r] if r < len(col) else ""
            if i < len(headers):
                row_parts.append(cell.rjust(col_widths[i]))
        lines.append(" ".join(row_parts))
    return "\n".join(lines)


def _compact_str(v: Any) -> str:
    """Compact rendering of one cell value (mirrors Julia ``:compact=>true``)."""
    if isinstance(v, float):
        if np.isnan(v):
            return "NaN"
        return f"{v:g}"
    return str(v)


# ---------------------------------------------------------------------------
# X13lazy + X13ResultWorkspace
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class X13lazy:
    """A placeholder for an X-13 output that has not been parsed yet.

    Mirrors Julia ``X13lazy`` at ``x13result.jl:43-47``. Holds the path
    to the output file, the file extension (which selects the parser),
    and the data frequency (needed for :func:`x13read_series` /
    :func:`x13read_seatsseries`). :class:`X13ResultWorkspace` swaps the
    instance for the parsed object on first attribute / key access.
    """

    file: str
    ext: str
    frequency: Frequency

    def __repr__(self) -> str:
        return self.file

    def __str__(self) -> str:
        return self.file


class X13ResultWorkspace(Workspace):
    """A :class:`Workspace` that materialises :class:`X13lazy` entries on first access.

    Mirrors Julia ``X13ResultWorkspace`` at ``X13.jl:24-35`` +
    ``Base.getproperty(::X13ResultWorkspace)`` at ``x13result.jl:49-56``.
    Accessing an entry that is an :class:`X13lazy` triggers a call to
    :func:`loadresult`; the parsed object is written back into the
    underlying ``_c`` dict, so subsequent accesses skip the parse.

    Both ``ws.foo`` (attribute) and ``ws["foo"]`` (key) access paths go
    through the same materialisation. Subset access (``ws[["a", "b"]]``)
    returns a plain :class:`Workspace` per the parent contract; lazy
    entries inside such a subset are *not* materialised eagerly.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires when normal lookup fails; route to _c
        # ourselves rather than calling super().__getattr__ (which would
        # double-route through this method on subclasses).
        try:
            val = object.__getattribute__(self, "_c")[name]
        except KeyError as e:
            msg = f"X13ResultWorkspace has no member {name!r}"
            raise AttributeError(msg) from e
        if isinstance(val, X13lazy):
            val = loadresult(val)
            self._c[name] = val
        return val

    def __getitem__(self, key: Any) -> Any:
        val = super().__getitem__(key)
        # Subset access returns a Workspace, never an X13lazy.
        if isinstance(val, X13lazy) and isinstance(key, str):
            val = loadresult(val)
            self._c[key] = val
        return val


# ---------------------------------------------------------------------------
# X13result
# ---------------------------------------------------------------------------


class X13result:
    """Top-level container for one X-13 run's outputs.

    Mirrors Julia ``X13result`` at ``x13result.jl:23-40``. Construct via
    :func:`run`; direct construction is documented but considered
    internal.

    Attributes
    ----------
    spec
        The :class:`X13spec` that was run.
    outfolder
        Absolute path to the temp directory the run produced.
        Auto-removed when this :class:`X13result` is garbage-collected
        (via :func:`weakref.finalize` — see module docstring).
    series
        :class:`X13ResultWorkspace` of TSeries / MVTSeries outputs.
    tables
        :class:`X13ResultWorkspace` of :class:`WorkspaceTable` outputs
        (multi-column equal-length tables: acf, pcf, R/I/O, …).
    text
        :class:`X13ResultWorkspace` of plain-string outputs (the
        ``.out`` echo of the input spec, summaries, ``.err`` content).
    other
        :class:`X13ResultWorkspace` for everything else — UDG diagnostic
        codes, model / estimate / identify outputs.
    stdout
        The captured stdout from the X-13 binary.
    errors / warnings / notes
        Lists of strings parsed from the ``.err`` channel by
        :func:`x13read_err`.
    """

    __slots__ = (
        "__weakref__",
        "_cleanup_handle",
        "errors",
        "notes",
        "other",
        "outfolder",
        "series",
        "spec",
        "stdout",
        "tables",
        "text",
        "warnings",
    )

    def __init__(self, spec: X13spec, outfolder: str, stdout: str) -> None:
        self.spec = spec
        self.outfolder = outfolder
        self.stdout = stdout
        self.series = X13ResultWorkspace()
        self.tables = X13ResultWorkspace()
        self.text = X13ResultWorkspace()
        self.other = X13ResultWorkspace()
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []
        # Register the cleanup. The finalize callback only sees its args,
        # not the X13result instance, which is what allows GC of the
        # X13result to actually trigger the callback (a closure over self
        # would keep self alive forever).
        self._cleanup_handle = weakref.finalize(self, _cleanup_outfolder, outfolder)

    def __repr__(self) -> str:
        return _format_x13result(self)

    def __str__(self) -> str:
        return _format_x13result(self)


def _format_x13result(r: X13result) -> str:
    """Render X13result as a multi-line summary.

    Mirrors Julia ``Base.show(::IO, ::MIME"text/plain", ::X13result)`` at
    ``x13result.jl:884-944``. The Python version skips the
    screen-size truncation logic for the same reason :func:`_format_workspace_table`
    does.
    """
    lines = ["X13 results"]
    fields: list[tuple[str, str]] = [
        ("spec", "X13 spec"),
        ("outfolder", repr(r.outfolder)),
        ("series", f"X13ResultWorkspace with {len(r.series)} TSeries/MVTSeries"),
        ("tables", f"X13ResultWorkspace with {len(r.tables)} tables"),
        ("text", f"X13ResultWorkspace with {len(r.text)} entries"),
        ("other", f"X13ResultWorkspace with {len(r.other)} entries"),
        ("stdout", f"{len(r.stdout)}-byte String"),
        ("errors", f"Vector{{String}} ({len(r.errors)})"),
        ("warnings", f"Vector{{String}} ({len(r.warnings)})"),
        ("notes", f"Vector{{String}} ({len(r.notes)})"),
    ]
    width = max(len(k) for k, _ in fields)
    for k, v in fields:
        lines.append(f"  {k.rjust(width)} ⇒ {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

# Backoff schedule for the per-result cleanup retry loop. Locked in
# decision 24 sub-decision 4 (M2.0 follow-up): three attempts at
# 0.1 / 0.2 / 0.4 s — long enough to clear typical Windows
# antivirus / Defender handle-release latency, short enough that the
# total worst-case (~0.7 s) is invisible to interactive use.
_CLEANUP_BACKOFF_S: Final[tuple[float, ...]] = (0.1, 0.2, 0.4)


def _cleanup_outfolder(outfolder: str) -> None:
    """Remove the X-13 temp directory, retrying on Windows PermissionError.

    Module-level (not a method) so :func:`weakref.finalize` can register
    it without capturing the owning :class:`X13result` in a closure —
    a closure would keep the result alive forever and the finalizer
    would never fire.
    """
    if not outfolder or not os.path.isdir(outfolder):
        return
    import time  # noqa: PLC0415  - deferred to keep import time cheap

    for delay in _CLEANUP_BACKOFF_S:
        try:
            shutil.rmtree(outfolder)
            return
        except PermissionError:
            time.sleep(delay)
        except FileNotFoundError:
            return
    # Final attempt — swallow any remaining error so a stuck cleanup
    # never raises from the finalizer (where exceptions are printed but
    # not propagated, and would just clutter user terminals).
    shutil.rmtree(outfolder, ignore_errors=True)


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------


_COLNAME_BAD_CHARS: Final[re.Pattern[str]] = re.compile(r"[\s\-\.]+")


def _sanitize_colname(s: str) -> str:
    """Replace whitespace / hyphen / period runs with a single underscore.

    Mirrors Julia ``_sanitize_colname`` at ``x13result.jl:736-738``.
    """
    return _COLNAME_BAD_CHARS.sub("_", s)


def _tryparse_int(s: str) -> int | None:
    """Try parsing ``s`` as an int; return ``None`` on failure."""
    try:
        return int(s)
    except (ValueError, TypeError):
        return None


def _tryparse_float(s: str, default: float | None = None) -> float | None:
    """Try parsing ``s`` as a float; return ``default`` on failure.

    Mirrors Julia ``_tryparse(Float64, s, default)`` at
    ``x13result.jl:741-745``.
    """
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def _add_layers(ws: Workspace) -> None:
    """Convert dotted keys into nested :class:`Workspace` entries.

    Mirrors Julia ``_add_layers!`` at ``x13result.jl:707-732``. The
    X-13 UDG output uses dotted keys like ``k.other.udf.roots.ar.01``
    to encode a hierarchy; this helper walks the workspace and rewrites
    such keys as nested workspaces.
    """
    keys_added: list[str] = []
    for key in list(ws._c.keys()):
        dot = key.find(".")
        if dot == -1:
            continue
        trunk = key[:dot]
        leaf = key[dot + 1 :]
        if trunk not in ws._c:
            ws._c[trunk] = Workspace()
            keys_added.append(trunk)
        elif not isinstance(ws._c[trunk], Workspace):
            trunk = trunk + "_"
            ws._c[trunk] = Workspace()
            keys_added.append(trunk)
        ws._c[trunk]._c[leaf] = ws._c[key]
        del ws._c[key]
    for k in keys_added:
        _add_layers(ws._c[k])


# ---------------------------------------------------------------------------
# Parsers — text channel
# ---------------------------------------------------------------------------


def x13read_err(
    file: str | os.PathLike[str],
    warnings_out: list[str],
    notes_out: list[str],
    errors_out: list[str],
) -> None:
    r"""Parse the X-13 ``.err`` output file into warnings / notes / errors lists.

    Mirrors Julia ``x13read_err`` at ``x13result.jl:287-313``. Each
    diagnostic begins with ``" WARNING:"`` / ``" ERROR:"`` / ``" NOTE:"``
    (note the leading space); continuation lines are appended to the
    last diagnostic with ``\n`` separators. The three output lists are
    mutated in place to mirror the Julia ``push!`` shape.
    """
    text = Path(file).read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    collected: list[str] = []
    for line in lines:
        if len(line) >= 11 and (
            line.startswith(" WARNING:") or line.startswith(" ERROR:") or line.startswith(" NOTE:")
        ):
            collected.append(line)
        elif collected and line:
            # Continuation lines extend the last diagnostic (Julia
            # ``x13result.jl:296-301``); the Julia version also appends
            # bare trailing-newline empty lines, but those carry no
            # information so the Python port skips them — matches the
            # Julia *intent* without the trailing-whitespace artefact.
            collected[-1] = collected[-1] + "\n" + line
    for line in collected:
        if line.startswith(" WARNING:"):
            warnings_out.append(line[10:])
        elif line.startswith(" ERROR:"):
            errors_out.append(line[8:])
        elif line.startswith(" NOTE:"):
            notes_out.append(line[7:])


def x13read_key_values(  # noqa: PLR0912, PLR0915 - tracks Julia's per-value-type ladder
    lines: Sequence[str],
    separator: re.Pattern[str] | str = re.compile(r"[\t:]"),
) -> Workspace:
    r"""Parse an X-13 key/value output file into a :class:`Workspace`.

    Mirrors Julia ``x13read_key_values`` at ``x13result.jl:316-390``. The
    key is the text before the first separator match; the value is the
    stripped text after. Values are tried (in order) as: an X-13 date
    (only for ``key == "date"``), an integer, a float, a vector of
    floats (whitespace-separated), the booleans ``"yes"`` / ``"no"``.
    Otherwise the raw string is kept.

    The default separator is the Julia upstream's ``r"[\t\:]"`` —
    either a tab or a colon. The ``"udg"`` parser passes ``": "``
    instead (mirrors the upstream regex / string switch).

    After all keys are loaded, :func:`_add_layers` rewrites dotted keys
    into nested workspaces.
    """
    ws = Workspace()
    sep_re: re.Pattern[str] = (
        re.compile(re.escape(separator)) if isinstance(separator, str) else separator
    )
    for line in lines:
        if not line.strip():
            continue
        m = sep_re.search(line)
        if m is None and isinstance(separator, str) and separator == ": ":
            # Mirror Julia's fallback to a bare colon when ": " missed.
            m = re.compile(":").search(line)
        if m is None:
            warnings.warn(f"Could not parse: {line}", stacklevel=2)
            continue
        split_point = m.start()
        key = line[:split_point]
        val_raw = line[split_point + 1 :].strip()
        value: Any = val_raw
        found = False
        if key == "date":
            try:
                # Julia: Dates.DateFormat("u d, y") — abbreviated month name,
                # day, year. e.g. "May 20, 2026".
                value = _parse_x13_date(val_raw)
                found = True
            except ValueError:
                pass
        if not found:
            iv = _tryparse_int(val_raw)
            if iv is not None:
                value = iv
                found = True
        if not found:
            fv = _tryparse_float(val_raw)
            if fv is not None:
                value = fv
                found = True
        if not found:
            splitval = re.split(r"[\t\s]+", val_raw.replace("*******", "NaN"))
            if len(splitval) > 1:
                parsed: list[float] = []
                ok = True
                for v in splitval:
                    if not v.strip():
                        continue
                    fv2 = _tryparse_float(v.strip())
                    if fv2 is None:
                        ok = False
                        break
                    parsed.append(fv2)
                if ok and parsed:
                    value = parsed
                    found = True
        if not found and val_raw == "no":
            value = False
            found = True
        if not found and val_raw == "yes":
            value = True
            found = True
        ws._c[key] = value
    _add_layers(ws)
    return ws


def _parse_x13_date(s: str) -> str:
    """Parse an X-13 ``"Mon D, YYYY"`` date into an ISO-format string.

    Mirrors Julia ``Dates.Date(s, "u d, y")``. We return a string
    (``"YYYY-MM-DD"``) rather than a :class:`datetime.date` because:
    (a) the X-13 output dates are pure metadata (the date the binary
    was built / run), not used for downstream computation; (b) keeping
    them as strings sidesteps a calendar-vs-frequency surface area
    expansion we don't need at this milestone.
    """
    months = (
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    )
    # Normalise consecutive whitespace.
    parts = re.sub(r"\s+", " ", s).strip().split(" ")
    if len(parts) != 3:
        msg = f"could not parse X-13 date {s!r}"
        raise ValueError(msg)
    mon_raw, day_raw, year_raw = parts
    mon_l = mon_raw[:3].lower()
    try:
        mon = months.index(mon_l) + 1
    except ValueError as e:
        msg = f"could not parse X-13 month {mon_raw!r}"
        raise ValueError(msg) from e
    day = int(day_raw.rstrip(","))
    year = int(year_raw)
    return f"{year:04d}-{mon:02d}-{day:02d}"


def x13read_udg(file: str | os.PathLike[str]) -> Workspace:
    """Parse a UDG (Census-Bureau-defined diagnostic) output file.

    Mirrors Julia ``x13read_udg`` at ``x13result.jl:392-395``. Reads
    the file as text, splits on newlines, and dispatches to
    :func:`x13read_key_values` with separator ``": "``.
    """
    text = Path(file).read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    return x13read_key_values(lines, separator=": ")


# ---------------------------------------------------------------------------
# Parsers — table channel
# ---------------------------------------------------------------------------


def x13read_workspace_table(  # noqa: PLR0912 - mirrors Julia's per-ext branch ladder
    lines: Sequence[str], *, ext: str = "nospecialrules"
) -> WorkspaceTable:
    """Parse an X-13 multi-column tabular output into a :class:`WorkspaceTable`.

    Mirrors Julia ``x13read_workspace_table`` at ``x13result.jl:398-453``.
    The first line is a tab-separated header row; subsequent lines are
    tab-separated value rows. Two special-case extensions:

    * ``ext="acm"`` — the header row is missing the ``"lag"`` column;
      it's inserted at position 1 (0-indexed).
    * ``ext="rog"`` — the rate-of-growth table uses a different header
      layout that is rebuilt from line 2.
    """
    work_lines = list(lines)
    if work_lines and not work_lines[-1].strip():
        work_lines = work_lines[:-1]
    if not work_lines:
        return WorkspaceTable()
    headers = [_sanitize_colname(h) for h in work_lines[0].strip().split("\t")]
    if ext == "acm":
        headers.insert(1, "lag")
    elif ext == "rog":
        # Rebuild lines: replace ``\s+:`` with ``:`` and ``\s\s+`` with ``\t``.
        rog_lines = [
            re.sub(r"\s\s+", "\t", re.sub(r"\s+:", ":", line)).strip() for line in work_lines[1:]
        ]
        if not rog_lines:
            return WorkspaceTable()
        headers = ["measure"] + [_sanitize_colname(h) for h in rog_lines[0].split("\t")]
        work_lines = work_lines[:1] + rog_lines
    nrows = len(work_lines) - 2
    nrows = max(nrows, 0)
    columns: list[list[str]] = [["" for _ in range(nrows)] for _ in headers]
    for i, line in enumerate(work_lines[2:]):
        if not line.strip():
            continue
        for j, val in enumerate(line.split("\t")):
            if j >= len(headers):
                if not val.strip():
                    continue
                continue
            if i < nrows:
                columns[j][i] = val
    parsed_columns: list[Any] = []
    for col in columns:
        all_int = all(_tryparse_int(v) is not None for v in col)
        if col and all_int:
            parsed_columns.append([int(v) for v in col])
            continue
        all_float = all(_tryparse_float(v) is not None for v in col)
        if col and all_float:
            parsed_columns.append([float(v) for v in col])
            continue
        parsed_columns.append(list(col))
    ws = WorkspaceTable()
    for header, col_values in zip(headers, parsed_columns, strict=False):
        ws._c[header] = col_values
    return ws


# ---------------------------------------------------------------------------
# Parsers — series channel
# ---------------------------------------------------------------------------


def _parse_period_string(period_string: str, freq: Frequency) -> MIT:
    """Parse the per-row period token at the head of an X-13 series line.

    Mirrors Julia ``x13result.jl:473-481`` (and the parallel block at
    ``x13result.jl:515-518`` for seatsseries). Three cases:

    * For Monthly, a 3-letter lowercase month abbreviation (``"jan"``…)
      yields ``MIT.from_yp(Monthly, 1, month)``; the Julia code maps it
      this way because some legacy tables use month-only headers.
    * Otherwise the last two characters are the period (1-based), the
      preceding characters are the year.
    """
    if not isinstance(freq, YPFrequency):
        msg = f"X-13 series output requires a YP frequency, got {type(freq).__name__}."
        raise ValueError(msg)
    if len(period_string) > 2:
        head = period_string[:3].lower()
        if head in _MONTHS_AND_QUARTERS:
            p = _MONTHS_AND_QUARTERS[head]
            y = 1
        else:
            try:
                p = int(period_string[-2:])
                y = int(period_string[:-2])
            except ValueError as e:
                msg = f"Period string has an unexpected format: {period_string!r}."
                raise ValueError(msg) from e
        return MIT.from_yp(freq, y, p)
    msg = f"Period string has an unexpected format: {period_string!r}."
    raise ValueError(msg)


def x13read_series(file: str | os.PathLike[str], freq: Frequency) -> TSeries | MVTSeries:
    """Parse an X-13 tabular series output file.

    Mirrors Julia ``x13read_series`` at ``x13result.jl:455-493``. The
    first line is a tab-separated header row beginning with a date
    column; line 2 is dashes; lines 3+ are tab-separated rows where the
    first column is the period token and the remainder are float
    values.

    Returns
    -------
    * :class:`MVTSeries` when there are 2+ data columns,
    * :class:`TSeries` when there is exactly 1 data column,
    * empty :class:`TSeries` over the inferred range when the header
        row has no data columns past the date column (mirrors the Julia
        ``length(headers) == 0`` branch).
    """
    text = Path(file).read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    if len(lines) < 3:
        msg = f"X-13 series file too short to parse: {file!s} ({len(lines)} lines)."
        raise ValueError(msg)
    headers_tokens = lines[0].split("\t")[1:]
    headers = [_sanitize_colname(h) for h in headers_tokens]
    data_lines = [line for line in lines[2:] if line.strip()]
    if not data_lines:
        msg = f"X-13 series file has no data rows: {file!s}."
        raise ValueError(msg)
    first_row = data_lines[0].split("\t")
    lastcol = len(first_row)
    if lastcol > len(headers) + 1 and not first_row[-1].strip():
        lastcol -= 1
    n = len(data_lines)
    if headers:
        vals = np.empty((n, len(headers)), dtype=np.float64)
        for i, line in enumerate(data_lines):
            parts = line.split("\t")[1:lastcol]
            for j, v in enumerate(parts):
                if j >= len(headers):
                    break
                fv = _tryparse_float(v.strip(), default=float("nan"))
                vals[i, j] = float("nan") if fv is None else fv
    else:
        vals = np.empty((n, 0), dtype=np.float64)
    first_period = first_row[0]
    start = _parse_period_string(first_period, freq)
    if len(headers) > 1:
        return MVTSeries(start, headers, vals)
    if len(headers) == 0:
        end = MIT(start.frequency, start.value + n - 1)
        return TSeries(MITRange(start, end))
    return TSeries(start, vals[:, 0])


def x13read_seatsseries(  # noqa: PLR0912 - mirrors Julia's column / header / period branch ladder
    lines: Sequence[str], freq: Frequency
) -> TSeries | MVTSeries:
    r"""Parse a SEATS-format series file (whitespace-delimited, indented).

    Mirrors Julia ``x13read_seatsseries`` at ``x13result.jl:495-530``.
    SEATS output uses ``\s\s+`` as the column delimiter (two or more
    consecutive whitespace chars) and offsets the header row by 1-2
    lines depending on the layout.
    """
    delim = re.compile(r"\s\s+")
    if not lines:
        msg = "X-13 SEATS output is empty."
        raise ValueError(msg)
    headers_line = 1  # 0-indexed; Julia ``headers_line = 2`` is 1-indexed
    if not delim.search(lines[headers_line]):
        headers_line += 1
    if headers_line >= len(lines):
        msg = "X-13 SEATS output has no header row."
        raise ValueError(msg)
    header_parts = delim.split(lines[headers_line])
    headers = [_sanitize_colname(h) for h in header_parts[2:]]
    data_lines = list(lines[headers_line + 1 : -1])
    data_lines = [line for line in data_lines if line.strip()]
    if not data_lines:
        msg = "X-13 SEATS output has no data rows."
        raise ValueError(msg)
    first_row = delim.split(data_lines[0])
    lastcol = len(first_row)
    if lastcol > len(headers) + 1 and not first_row[-1]:
        lastcol -= 1
    n = len(data_lines)
    if headers:
        vals = np.empty((n, len(headers)), dtype=np.float64)
        for i, line in enumerate(data_lines):
            parts = delim.split(line)[1:lastcol]
            for j, v in enumerate(parts):
                if j >= len(headers):
                    break
                fv = _tryparse_float(v.strip(), default=float("nan"))
                vals[i, j] = float("nan") if fv is None else fv
    else:
        vals = np.empty((n, 0), dtype=np.float64)
    # SEATS dates are ``P - Y`` (period-dash-year) — note the order is
    # reversed from x13read_series ("YYYYPP" → year then period).
    date_token = delim.split(data_lines[0])[0]
    pieces = date_token.split("-")
    if len(pieces) <= 1:
        msg = f"SEATS period token has unexpected format: {date_token!r}."
        raise ValueError(msg)
    p = int(pieces[0].strip())
    y = int(pieces[1].strip())
    if not isinstance(freq, YPFrequency):
        msg = f"SEATS series output requires a YP frequency, got {type(freq).__name__}."
        raise ValueError(msg)
    start = MIT.from_yp(freq, y, p)
    if len(headers) > 1:
        return MVTSeries(start, headers, vals)
    if len(headers) == 0:
        end = MIT(start.frequency, start.value + n - 1)
        return TSeries(MITRange(start, end))
    return TSeries(start, vals[:, 0])


# ---------------------------------------------------------------------------
# Parsers — estimates / model / identify
# ---------------------------------------------------------------------------


def x13read_estimates(file: str | os.PathLike[str]) -> Workspace:
    """Parse the X-13 ``.est`` (estimates) output file.

    Mirrors Julia ``x13read_estimates`` at ``x13result.jl:532-560``.
    Walks the file looking for the section markers ``$arima:``,
    ``$regression:``, ``$arima$estimates:``, ``$regression$estimates:``,
    ``$variance:``; dispatches each section to the right sub-parser.
    """
    text = Path(file).read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    indices: dict[str, int] = {}
    for i, line in enumerate(lines):
        if line == "$arima:":
            indices["arima"] = i
        elif line == "$regression:":
            indices["regression"] = i
        elif line == "$arima$estimates:":
            indices["arimaestimates"] = i
        elif line == "$regression$estimates:":
            indices["regressionestimates"] = i
        elif line == "$variance:":
            indices["variance"] = i
    res = Workspace()
    # Julia ``lines[arimaestimates+1:variance-1]`` is inclusive-inclusive
    # (Julia ranges); the Python equivalent is exclusive on the upper bound,
    # so ``[start:variance]`` (not ``variance-1``) keeps the same elements.
    if "arima" in indices and "arimaestimates" in indices and "variance" in indices:
        res._c["arima"] = x13read_workspace_table(
            lines[indices["arimaestimates"] + 1 : indices["variance"]]
        )
        res._c["variance"] = x13read_key_values(lines[indices["variance"] + 1 :])
    elif "regression" in indices and "regressionestimates" in indices and "variance" in indices:
        res._c["regression"] = x13read_workspace_table(
            lines[indices["regressionestimates"] + 1 : indices["variance"]]
        )
        res._c["variance"] = x13read_key_values(lines[indices["variance"] + 1 :])
    return res


def x13read_identify(file: str | os.PathLike[str]) -> Workspace:
    """Parse the X-13 ``.iac`` / ``.ipc`` (identify) output files.

    Mirrors Julia ``x13read_identify`` at ``x13result.jl:562-577``. The
    file contains 1+ pages, each beginning with a ``$diff = N`` /
    ``$sdiff = M`` pair followed by an autocorrelation /
    partial-autocorrelation table.
    """
    text = Path(file).read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    page_locs: list[int] = []
    for i, line in enumerate(lines):
        if len(line) > 4 and line[:5] == "$diff":
            page_locs.append(i)
    res = Workspace()
    for i, loc in enumerate(page_locs):
        diff_label = lines[loc].replace("$", "").replace("= ", "").strip()
        sdiff_label = lines[loc + 1].replace("$", "").replace("= ", "").strip()
        sym = f"{diff_label}{sdiff_label}"
        # Same inclusive-vs-exclusive Julia/Python translation as
        # ``x13read_estimates`` above: ``page_locs[i+1] - 1`` in Julia
        # (last line of the current page) becomes ``page_locs[i+1]`` in
        # Python (exclusive on the next page's start).
        end_idx = page_locs[i + 1] if i + 1 < len(page_locs) else len(lines)
        res._c[sym] = x13read_workspace_table(lines[loc + 2 : end_idx])
    return res


def x13read_model(file: str | os.PathLike[str]) -> Workspace:
    """Parse the X-13 ``.mdl`` (model) output file.

    Mirrors Julia ``x13read_model`` at ``x13result.jl:579-613``. The
    file contains zero, one, or both of an ``arima{model=...}`` block
    and a ``regression{...}`` block; each is dispatched to
    :func:`_x13read_model_block`.
    """
    text = Path(file).read_text(encoding="utf-8", errors="replace")
    lines = text.split("\n")
    indices: dict[str, int] = {}
    for i, line in enumerate(lines):
        s = line.strip()
        if s in {"arima{model=", "arima{"}:
            indices["arima"] = i
        elif s == "regression{":
            indices["regression"] = i
    res = Workspace()
    if "arima" in indices:
        end = len(lines)
        if "regression" in indices:
            end = (
                len(lines)
                if indices["arima"] > indices["regression"]
                else indices["regression"] - 1
            )
        res._c["arima"] = _x13read_model_block(lines[indices["arima"] : end])
    if "regression" in indices:
        end = len(lines)
        if "arima" in indices:
            end = len(lines) if indices["regression"] > indices["arima"] else indices["arima"] - 1
        res._c["regression"] = _x13read_model_block(lines[indices["regression"] : end])
    return res


def _x13read_model_block(  # noqa: PLR0912, PLR0915 - tracks Julia's per-key parse ladder
    lines: Sequence[str],
) -> Workspace:
    """Parse one arima{...} or regression{...} block.

    Mirrors Julia ``_x13read_model`` at ``x13result.jl:615-704``.
    """
    res = Workspace()
    indices: dict[str, int] = {}
    for i, line in enumerate(lines):
        s = line.strip()
        if s in {"arima{model=", "model="}:
            indices["model"] = i
        elif s == "ar  =(":
            indices["ar"] = i
        elif s == "ma  =(":
            indices["ma"] = i
        elif s == "regression{":
            indices["regression"] = i
        elif s in {"variables=(", "regression{variables=("}:
            indices["variables"] = i
        elif s == "b=(":
            indices["b"] = i
    ordered = sorted(indices, key=indices.__getitem__)
    for i, key in enumerate(ordered):
        if key == "model":
            model_line = lines[indices[key] + 1]
            arima_specs: list[ArimaSpec] = []
            for piece in model_line.split("("):
                if not piece.strip():
                    continue
                cleaned = piece.replace(")", " ").replace(",", " ").strip()
                tokens = [t for t in cleaned.split(" ") if t]
                if "[" not in piece:
                    arima_specs.append(ArimaSpec(*[int(t) for t in tokens]))
                else:
                    parsed_vals: list[int | list[int]] = []
                    for tok in tokens:
                        if tok.startswith("["):
                            parsed_vals.append([int(tok[1:-1])])
                        else:
                            parsed_vals.append(int(tok))
                    arima_specs.append(ArimaSpec(*parsed_vals))  # type: ignore[arg-type]
            res._c["model"] = ArimaModel(specs=list(arima_specs))
        elif key in {"ar", "ma", "b", "variables"}:
            if key != ordered[-1]:
                next_key = ordered[i + 1]
                val_lines = lines[indices[key] + 1 : indices[next_key] - 1]
            else:
                val_lines = lines[indices[key] + 1 : -2]
            values = [v.strip() for v in val_lines if v.strip() and v.strip() not in {")", "}"}]
            if key in {"ar", "ma", "b"}:
                fixed = [False] * len(values)
                parsed_floats: list[float] = [0.0] * len(values)
                for idx, v in enumerate(values):
                    if v.endswith("f"):
                        parsed_floats[idx] = float(v[:-1])
                        fixed[idx] = True
                    else:
                        parsed_floats[idx] = float(v)
                res._c[f"fix{key}"] = fixed
                res._c[key] = parsed_floats
            else:
                res._c[key] = list(values)
    return res


# ---------------------------------------------------------------------------
# loadresult dispatch
# ---------------------------------------------------------------------------


def loadresult(lazy_or_file: X13lazy | str | os.PathLike[str], /, *args: Any) -> Any:
    """Materialise an :class:`X13lazy` into the parsed object.

    Mirrors Julia ``loadresult`` at ``x13result.jl:240-285``. Two
    calling conventions:

    * ``loadresult(X13lazy(file, ext, freq))`` — the usual shape, used
      by :class:`X13ResultWorkspace` when an attribute / key access
      hits a lazy entry.
    * ``loadresult(file, ext, freq)`` — same as above but with the
      three arguments passed positionally. Mirrors Julia's two-arity
      overload.
    """
    if isinstance(lazy_or_file, X13lazy):
        file = lazy_or_file.file
        ext = lazy_or_file.ext
        freq = lazy_or_file.frequency
    else:
        file = os.fspath(lazy_or_file)
        if len(args) != 2:
            msg = "loadresult(file, ext, freq) requires three positional arguments."
            raise TypeError(msg)
        ext, freq = args
    return _dispatch_loadresult(file, ext, freq)


def _dispatch_loadresult(  # noqa: PLR0911 - one return per extension class is the clearest shape
    file: str, ext: str, freq: Frequency
) -> Any:
    """Dispatch the per-extension parser table behind :func:`loadresult`."""
    if ext in _SERIES_EXTENSIONS:
        return x13read_series(file, freq)
    if ext in _TABLE_EXTENSIONS:
        text = Path(file).read_text(encoding="utf-8", errors="replace")
        return x13read_workspace_table(text.split("\n"), ext=ext)
    if ext == "udg":
        return x13read_udg(file)
    if ext in _KV_LIST_EXTENSIONS:
        text = Path(file).read_text(encoding="utf-8", errors="replace")
        return x13read_key_values(text.split("\n"), separator=re.compile(r"\s+"))
    if ext == "est":
        return x13read_estimates(file)
    if ext == "mdl":
        return x13read_model(file)
    if ext in {"ipc", "iac"}:
        return x13read_identify(file)
    if ext == "tbs":
        text = Path(file).read_text(encoding="utf-8", errors="replace")
        lines = text.split("\n")
        if len(lines) > 2:
            return x13read_seatsseries(lines, freq)
        return None
    if ext == "rog":
        text = Path(file).read_text(encoding="utf-8", errors="replace")
        return x13read_workspace_table(text.split("\n"), ext="rog")
    if ext in _HUMAN_TEXT_EXTENSIONS:
        return Path(file).read_text(encoding="utf-8", errors="replace")
    if ext not in {"txt", "log"}:
        warnings.warn(
            f"Encountered unknown output file extension {ext!r}. "
            "Contents below.\n"
            "================================================================\n"
            f"{Path(file).read_text(encoding='utf-8', errors='replace')}\n"
            f"{file}",
            stacklevel=2,
        )
    return None


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _normalise_load(load: str | Sequence[str]) -> tuple[bool, set[str]]:
    """Normalise the ``load=`` argument to ``(is_all, set_of_keys)``.

    Mirrors Julia ``_load = load isa Symbol ? Set([load]) : Set(load)``
    at ``x13result.jl:91``.
    """
    if isinstance(load, str):
        if load == "none":
            return False, set()
        if load == "all":
            return True, set()
        return False, {load}
    return False, set(load)


def _resolve_binary() -> str | None:
    """Return the path to the X-13 binary, or :data:`None` if not available.

    Resolution order (mirrors Julia ``x13result.jl:103-107``):

    1. If :func:`tsecon.getoption` ``"x13path"`` is non-empty, use that.
    2. Else look for the bundled binary at ``tsecon/x13/_binary/x13as``
       (lands in M2.6 alongside the wheels matrix; until then this
       check returns :data:`None`).
    3. Otherwise :data:`None` — :func:`run` raises a clear error.
    """
    x13path = getoption("x13path")
    if isinstance(x13path, str) and x13path:
        return x13path
    # M2.6: in-wheel binary lookup. The path layout:
    #   tsecon/x13/_binary/x13as            (Linux / macOS)
    #   tsecon/x13/_binary/x13as.exe        (Windows)
    here = Path(__file__).parent / "_binary"
    for cand in ("x13as.exe", "x13as"):
        p = here / cand
        if p.is_file():
            return str(p)
    return None


def run(
    spec: X13spec | str,
    freq: Frequency | None = None,
    *,
    verbose: bool = True,
    allow_errors: bool = False,
    load: str | Sequence[str] = "none",
) -> X13result:
    """Run X-13ARIMA-SEATS on a :class:`X13spec` or raw spec string.

    Mirrors Julia ``X13.run`` (two methods at ``x13result.jl:76-88``):

    * ``run(spec, *, ...)`` — pass an :class:`X13spec`; the spec is
      serialised via :func:`tsecon.x13.x13write` and written to
      ``<outfolder>/spec.spc``.
    * ``run(specstring, freq, *, ...)`` — pass a pre-formatted
      ``.spc`` text plus the frequency the binary should assume; the
      spec text is written verbatim.

    Parameters
    ----------
    spec
        An :class:`X13spec` (the usual shape) or a raw spec string.
    freq
        Required when ``spec`` is a string; ignored otherwise.
    verbose
        Echo warnings / notes from the X-13 ``.err`` channel after the
        run finishes. Default :data:`True`.
    allow_errors
        If :data:`False` (default), raise :exc:`RuntimeError` when the
        X-13 binary reports errors. If :data:`True`, emit them as
        warnings and return the result anyway.
    load
        Which output tables to eagerly parse. ``"none"`` (default) and
        ``"all"`` mirror Julia's ``:none`` / ``:all``; otherwise pass
        a single key (``"d11"``) or a sequence of keys
        (``("d11", "d12")``). Unrecognised keys are silently skipped
        (mirrors Julia ``intersect(_load, keys(res.series))``).
    """
    if isinstance(spec, str):
        if freq is None:
            msg = "run(specstring, freq=...): freq is required for the string overload."
            raise TypeError(msg)
        return _run_specstring(spec, freq, verbose=verbose, allow_errors=allow_errors, load=load)
    if not isinstance(spec, X13spec):
        msg = f"run() expects an X13spec or a spec string; got {type(spec).__name__}."  # type: ignore[unreachable]
        raise TypeError(msg)
    x13write(spec)
    if isinstance(spec.folder, X13default) or not spec.folder:
        spec.folder = tempfile.mkdtemp(prefix="x13_")
    folder = spec.folder
    assert isinstance(folder, str)
    Path(folder, "spec.spc").write_text(str(spec.string), encoding="utf-8")
    return _run_internal(spec, verbose=verbose, allow_errors=allow_errors, load=load)


def _run_specstring(
    specstring: str,
    freq: Frequency,
    *,
    verbose: bool,
    allow_errors: bool,
    load: str | Sequence[str],
) -> X13result:
    """Execute the ``run(specstring, freq, ...)`` overload's body."""
    from tsecon.x13._spec import newspec  # noqa: PLC0415 - avoid import cycle

    if not isinstance(freq, YPFrequency):
        msg = f"run(specstring, freq, ...): freq must be a YPFrequency, got {type(freq).__name__}."
        raise TypeError(msg)
    placeholder = TSeries(MIT.from_yp(freq, 1, 1), np.array([0.0]))
    spec = newspec(placeholder)
    spec.string = specstring
    spec.folder = tempfile.mkdtemp(prefix="x13_")
    folder = spec.folder
    Path(folder, "spec.spc").write_text(specstring, encoding="utf-8")
    return _run_internal(spec, verbose=verbose, allow_errors=allow_errors, load=load)


def _run_internal(  # noqa: PLR0912 - sequential phase ladder matches Julia _run
    spec: X13spec,
    *,
    verbose: bool,
    allow_errors: bool,
    load: str | Sequence[str],
) -> X13result:
    """Shared run body — invokes the binary and parses the output dir."""
    is_load_all, load_keys = _normalise_load(load)
    folder = spec.folder
    if not isinstance(folder, str):
        msg = "_run_internal: spec.folder must be a string at this point."
        raise TypeError(msg)
    gpath = Path(folder, "graphics")
    gpath.mkdir(exist_ok=True)

    binary = _resolve_binary()
    if binary is None:
        msg = (
            "No X-13ARIMA-SEATS binary available. The bundled binary "
            "lands in M2.6 (alongside the wheels matrix). Until then "
            'either point setoption("x13path", "/path/to/x13as") at a '
            "user-installed binary, or skip the X-13 wrapper."
        )
        raise RuntimeError(msg)

    cmd = [binary, "-I", "spec", "-G", "graphics", "-S"]
    proc = subprocess.run(
        cmd,
        cwd=folder,
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = proc.stdout
    stderr = proc.stderr
    if stderr:
        msg = f"Running X-13 failed: {stderr}\nAdditional information may be available in {folder}."
        raise RuntimeError(msg)

    # Mirror Julia's scan of stdout for ``ERROR:`` markers (some
    # diagnostics are printed to stdout but not the .err file).
    stdout_lines = stdout.split("\n")
    for i, line in enumerate(stdout_lines):
        if "ERROR:" in line:
            err_msg = line
            for j in range(i + 1, len(stdout_lines)):
                if stdout_lines[j].startswith("     "):
                    err_msg = err_msg + "\n" + stdout_lines[j]
                else:
                    break
            if allow_errors:
                warnings.warn(err_msg, stacklevel=2)
            else:
                raise RuntimeError(err_msg)

    res = X13result(spec, folder, stdout)
    if not isinstance(spec.series, X13series):
        msg = "run(): spec.series must be an X13series."
        raise TypeError(msg)
    data = spec.series.data
    if not isinstance(data, TSeries):
        msg = "run(): spec.series.data must be a TSeries."  # type: ignore[unreachable]
        raise TypeError(msg)
    freq = data.frequency
    _populate_result(res, folder, freq, is_load_all=is_load_all, load_keys=load_keys)
    _materialise_eager(res, load_keys=load_keys, is_load_all=is_load_all)
    if verbose:
        for w in res.warnings:
            warnings.warn(w, stacklevel=2)
        # Notes are informational — emit as UserWarning so they surface
        # under the project's filterwarnings config without being a
        # test failure (since they're real X-13 binary diagnostics, not
        # python-side warnings).
    if res.errors:
        if allow_errors:
            warnings.warn("There were errors in the specification file.", stacklevel=2)
        else:
            msg = "There were errors in the specification file: " + "; ".join(res.errors)
            raise RuntimeError(msg)
    return res


def _walk_output_files(folder: str) -> Iterator[Path]:
    """Yield X-13 output files from ``folder`` and its first level of subdirs.

    Mirrors Julia ``x13result.jl:149-155``.
    """
    root = Path(folder)
    for p in sorted(root.iterdir()):
        if p.is_file():
            yield p
    for p in sorted(root.iterdir()):
        if p.is_dir():
            for q in sorted(p.iterdir()):
                if q.is_file():
                    yield q


def _populate_result(  # noqa: PLR0912 - per-extension dispatch ladder matches Julia
    res: X13result,
    folder: str,
    freq: Frequency,
    *,
    is_load_all: bool,
    load_keys: set[str],
) -> None:
    """Walk the output folder and populate ``res.{series, tables, text, other}``.

    Mirrors Julia ``x13result.jl:147-200``.
    """
    for path in _walk_output_files(folder):
        suffix = path.suffix
        ext = suffix[1:] if suffix.startswith(".") else suffix
        if not ext:
            continue
        file_str = str(path)
        if is_load_all:
            load_keys.add(ext)
        if ext in _SERIES_EXTENSIONS:
            res.series._c[ext] = X13lazy(file_str, ext, freq)
        elif ext in _PROBABLY_SERIES_EXTENSIONS:
            try:
                ts = loadresult(X13lazy(file_str, ext, freq))
                res.series._c[ext] = ts
            except (ValueError, OSError):
                warnings.warn(
                    f"Encountered an unknown output type: {ext}. "
                    "Attempted to load it as a series but failed.",
                    stacklevel=2,
                )
                continue
            warnings.warn(
                f"Encountered an unknown output type: {ext}. Loaded as a series.",
                stacklevel=2,
            )
        elif ext in _TABLE_EXTENSIONS:
            res.tables._c[ext] = X13lazy(file_str, ext, freq)
        elif ext in {"udg", "est", "mdl", "ipc", "iac"} | _KV_LIST_EXTENSIONS:
            res.other._c[ext] = X13lazy(file_str, ext, freq)
        elif ext == "err":
            x13read_err(file_str, res.warnings, res.notes, res.errors)
            res.text._c[ext] = Path(file_str).read_text(encoding="utf-8", errors="replace")
        elif ext == "tbs" or (ext.lower() == "out" and path.name.upper() in {"TABLE-S.OUT"}):
            res.series._c["tbs"] = X13lazy(file_str, "tbs", freq)
            if is_load_all:
                load_keys.add("tbs")
        elif ext == "rog" or (ext.lower() == "out" and path.name.upper() in {"ROGTABLE.OUT"}):
            res.tables._c["rog"] = X13lazy(file_str, "rog", freq)
            if is_load_all:
                load_keys.add("rog")
        elif ext in _HUMAN_TEXT_EXTENSIONS:
            res.text._c[ext] = X13lazy(file_str, ext, freq)
        elif ext not in {"txt", "log"}:
            # Genuine unknown — surface as a UserWarning so the user notices.
            warnings.warn(
                f"Encountered unknown X-13 output file extension {ext!r} at {file_str}.",
                stacklevel=2,
            )


def _materialise_eager(res: X13result, *, load_keys: set[str], is_load_all: bool) -> None:
    """Pre-load entries the caller asked for via ``load=``.

    Mirrors Julia ``x13result.jl:202-212``.
    """
    if not load_keys and not is_load_all:
        return
    for bucket in (res.series, res.tables, res.other):
        for key in list(bucket._c.keys()):
            if key in load_keys or (is_load_all and isinstance(bucket._c[key], X13lazy)):
                val = bucket._c[key]
                if isinstance(val, X13lazy):
                    bucket._c[key] = loadresult(val)


# Public re-exports for the parser module surface.

# The two consts re-exported below are imported by ``_x13.cleanup`` to
# implement the stale-folder sweep without depending on the private
# names in :mod:`tsecon.x13._consts`.

_TEMP_PREFIX: Final[str] = "x13_"
