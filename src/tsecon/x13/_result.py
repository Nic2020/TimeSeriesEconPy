# SPDX-License-Identifier: MIT
"""X-13ARIMA-SEATS result types and output parsers.

Mirrors ``TimeSeriesEcon.jl/src/x13/x13result.jl`` (1,035 LOC). Lands in
**M2.5**.

Content planned:

* :class:`X13lazy` (``x13result.jl:43-47``) — a frozen 3-tuple ``(file,
  ext, freq)`` that the :meth:`X13ResultWorkspace.__getattr__` /
  :meth:`__getitem__` materialises to a :class:`~tsecon.tseries.TSeries`
  / :class:`~tsecon.workspace.Workspace` on first access. The result is
  cached in the owning workspace's ``_c`` dict (Julia: ``x13result.jl:49-
  56``). Mirrors the eager-on-``load=`` + lazy-everything-else contract.
* :class:`X13result` (``x13result.jl:23-40``) — the top-level result
  container with ``spec`` / ``outfolder`` / ``series`` / ``tables`` /
  ``text`` / ``other`` / ``stdout`` / ``errors`` / ``warnings`` /
  ``notes`` fields. The Julia version uses a finalizer to remove
  ``outfolder`` when the result is GC'd; the Python port uses
  ``weakref.finalize`` with a :func:`shutil.rmtree` that retries on
  Windows :class:`PermissionError` (M2.0 sub-decision: standard
  exponential backoff over up to ~1 s, with
  :func:`tempfile.TemporaryDirectory(ignore_cleanup_errors=True)` as
  the fallback for the irrecoverable case).
* :func:`run` (``x13result.jl:76-237``) — the public entry. Two
  overloads:

  1. ``run(spec: X13spec, *, verbose=True, allow_errors=False,
     load: str | Sequence[str] = "none") -> X13result`` — the typed
     path; serialises ``spec`` via :func:`tsecon.x13._write.x13write`,
     writes ``spec.spc`` + ``graphics/`` under
     :func:`tempfile.mkdtemp(prefix="x13_")`, invokes the bundled
     ``x13as`` (or ``setoption("x13path")`` override) via
     :mod:`subprocess`, parses ``stdout`` + ``stderr`` + the err-file
     warning/error/note channels, walks the output directory, and
     populates the result containers with :class:`X13lazy` proxies
     (or eager parses for ``load`` entries).
  2. ``run(specstring: str, freq: type[Frequency], *, ...) -> X13result``
     — the string path; bypasses :class:`X13spec` for users who already
     have a hand-written ``.spc`` string.

* :func:`loadresult` (``x13result.jl:240-285``) — :class:`X13lazy` →
  parsed object dispatcher.
* :func:`x13read_series` / :func:`x13read_workspace_table` /
  :func:`x13read_udg` / :func:`x13read_key_values` /
  :func:`x13read_estimates` / :func:`x13read_model` /
  :func:`x13read_identify` / :func:`x13read_seatsseries` /
  :func:`x13read_err` (``x13result.jl:287-...``) — per-format parsers
  for the X13as output file dialect.

The ``load=`` argument's Python surface (locked in decision 24 pick 3):

* ``load="none"`` (default) — everything lazy.
* ``load="all"`` — every output table eagerly parsed at :func:`run`
  return.
* ``load="d11"`` — that one table eager.
* ``load=("d11", "d12")`` — those tables eager.

Mirrors Julia's ``:none`` / ``:all`` / ``:d11`` / ``[:d11, :d12]``
shapes, mapped to Python idioms (``str | Sequence[str]`` instead of
``Symbol | Vector{Symbol}``).

Implementation notes carried forward to M2.5:

* The ``outfolder`` cleanup retry pattern (M2.0 sub-decision 4): wrap
  :func:`shutil.rmtree` in a small retry loop (Python 3.11+ compatible —
  no reliance on the 3.12 ``onexc`` callback), waiting on the order of
  ``[0.1, 0.2, 0.4]`` seconds between retries to let Windows
  antivirus / Defender release the handle, then falling back to
  ``ignore_cleanup_errors=True`` after the final retry so a stuck
  cleanup never breaks user pipelines.
* The Julia upstream's ``cd(spec.folder) do ... end`` (``x13result.jl:108``)
  is a process-cwd change that doesn't play well with concurrent
  :func:`run` calls. The Python port uses ``subprocess.run(..., cwd=...)``
  to scope the cwd to the child process, allowing concurrent
  :func:`run` calls from threads without interlocking.
* The Julia upstream walks ``readdir(parent, join=false)`` then filters
  by owner UID (``X13.jl:80-87``). On Windows we drop the UID filter
  (no POSIX UID); ``cleanup`` removes any ``x13_*`` folder under
  :func:`tempfile.gettempdir` regardless of ownership. Acceptable: the
  tempdir on Windows is per-user by default.
"""

from __future__ import annotations

__all__: list[str] = []
