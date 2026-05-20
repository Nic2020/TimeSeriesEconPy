# SPDX-License-Identifier: MIT
"""Module-level conveniences for :mod:`tsecon.x13`.

Mirrors the convenience block of ``TimeSeriesEcon.jl/src/x13/X13.jl``
(``X13.jl:42-98``). The result-side types — :class:`WorkspaceTable`,
:class:`X13ResultWorkspace`, :class:`X13lazy`, :class:`X13result`,
:func:`run`, parsers — live in :mod:`tsecon.x13._result`. This sibling
hosts the three top-level convenience functions:

* :func:`cleanup` — sweep stale ``x13_*`` temp folders that the
  per-result :func:`weakref.finalize` callback missed (process hard
  kill, finalizer race at interpreter shutdown). Mirrors Julia
  ``cleanup`` at ``X13.jl:62-90``.
* :func:`deseasonalize` — run a default ``x11`` spec on a TSeries and
  return a new TSeries with the ``d11`` (final seasonally adjusted
  series) values. Mirrors Julia ``deseasonalize`` at ``X13.jl:58``.
* :func:`deseasonalize_inplace` — in-place flavour; replaces the values
  of the input TSeries in place. Mirrors Julia ``deseasonalize!`` at
  ``X13.jl:52-57``. The ``_inplace`` rename follows the existing
  TimeSeriesEconPy convention (Julia ``foo!`` → Python
  ``foo_inplace``) — see :func:`tsecon.recursive.rec` and
  :func:`tsecon.workspace.merge_inplace` for the parallel pattern.

The Julia upstream's module-level ``__init__`` (``X13.jl:92-98``) prints
an `@info` warning at import time when 5+ stale ``x13_*`` folders are
found. Importing :mod:`tsecon.x13` on every Python script start is
already slower than Julia's `using` (no in-process module cache); we
defer the leftover-folder count to the first :func:`run` call instead,
where it's emitted as a :class:`UserWarning` rather than as a print
statement.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from typing import TYPE_CHECKING, Any

from tsecon.x13._result import _TEMP_PREFIX, X13result, run

if TYPE_CHECKING:
    from tsecon.tseries import TSeries

__all__ = [
    "cleanup",
    "deseasonalize",
    "deseasonalize_inplace",
    "get_cleanup_folders",
]


def get_cleanup_folders() -> list[str]:
    """Return absolute paths of all ``x13_*`` folders in the system temp dir.

    Mirrors Julia ``get_cleanup_folders`` at ``X13.jl:76-90``. The Julia
    version filters by current-user UID; on POSIX we keep that check
    via :func:`os.stat`; on Windows we drop it (the system tempdir is
    per-user by default).
    """
    parent = tempfile.gettempdir()
    out: list[str] = []
    try:
        entries = os.listdir(parent)
    except OSError:
        return out
    my_uid: int | None = None
    if hasattr(os, "geteuid"):
        try:
            my_uid = os.geteuid()
        except OSError:
            my_uid = None
    for name in entries:
        if not name.startswith(_TEMP_PREFIX):
            continue
        full = os.path.join(parent, name)
        if not os.path.isdir(full):
            continue
        if my_uid is not None:
            try:
                if os.stat(full).st_uid != my_uid:
                    continue
            except OSError:
                continue
        out.append(full)
    return out


def cleanup() -> None:
    """Remove all stale ``x13_*`` folders from the system temp directory.

    Mirrors Julia ``cleanup`` at ``X13.jl:62-74``. Use this when a
    previous Python process was hard-killed before the per-result
    :func:`weakref.finalize` callbacks could fire, leaving X-13 temp
    folders behind. Emits a :class:`UserWarning` summarising the count
    removed (Julia prints to stdout; we use the warnings channel so
    the message routes through the project's ``filterwarnings`` config).
    """
    import shutil  # noqa: PLC0415 - keep import-time cost lean

    folders = get_cleanup_folders()
    for f in folders:
        shutil.rmtree(f, ignore_errors=True)
    warnings.warn(f"Removed {len(folders)} temporary x13 folders.", stacklevel=2)


def deseasonalize_inplace(ts: TSeries, **kwargs: Any) -> TSeries:
    """Run a default ``x11`` spec on ``ts`` and replace its values with the ``d11`` series.

    Mirrors Julia ``deseasonalize!`` at ``X13.jl:52-57``. The default
    seasonal-adjustment decomposition is multiplicative (``mode="mult"``);
    the seasonal filter defaults to the X-13 binary's automatic choice
    unless ``seasonalma=`` is passed. ``kwargs`` are forwarded to
    :func:`tsecon.x13.x11` (the spec builder).

    Returns ``ts`` for chaining (mirrors Julia's ``ts`` return).

    Raises
    ------
    RuntimeError
        If no X-13 binary is available — the bundled binary lands in
        M2.6 (alongside the wheels matrix). Until then, point
        ``setoption("x13path", ...)`` at a user-installed binary.
    """
    from tsecon.x13._spec import newspec, x11  # noqa: PLC0415 - avoid import cycle

    spec = newspec(ts, x11=x11(save="d11", **kwargs))
    res: X13result = run(spec, verbose=False)
    d11 = res.series.d11
    # d11 is a TSeries; copy its values into ``ts``'s buffer. The two
    # share a frequency by construction (the binary echoes the input's
    # frequency); guard against length mismatches in case the binary
    # produces a wider d11 (e.g. with forecasts appended).
    n = min(len(ts.values), len(d11.values))
    ts.values[:n] = d11.values[:n]
    return ts


def deseasonalize(ts: TSeries, **kwargs: Any) -> TSeries:
    """Return a new TSeries with the X-11 seasonally adjusted values.

    Mirrors Julia ``deseasonalize`` at ``X13.jl:58``. Equivalent to
    :func:`deseasonalize_inplace` on a copy of the input.
    """
    return deseasonalize_inplace(ts.copy(), **kwargs)
