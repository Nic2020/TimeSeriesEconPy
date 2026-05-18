# SPDX-License-Identifier: MIT
"""Module-level entry points for :mod:`tsecon.x13`.

Hosts :class:`WorkspaceTable`, :class:`X13ResultWorkspace`,
:func:`cleanup`, and :func:`deseasonalize`.

Mirrors ``TimeSeriesEcon.jl/src/x13/X13.jl`` (100 LOC). Lands in **M2.5**.

Surface planned:

* :class:`WorkspaceTable` — :class:`~tsecon.workspace.Workspace` subclass that
  prints as a table (every entry is an equal-length vector). Mirrors
  ``X13.jl:11-22``.
* :class:`X13ResultWorkspace` — Workspace subclass with lazy
  :func:`__getattr__` / ``__getitem__`` that materialises
  :class:`~tsecon.x13._result.X13lazy` proxies on first access and caches
  the parsed value. Mirrors ``X13.jl:24-35`` + ``x13result.jl:49-56``.
* :func:`cleanup` — remove stale ``x13_*`` folders from
  :func:`tempfile.gettempdir` owned by the current user. Mirrors
  ``X13.jl:62-90``.
* :func:`deseasonalize` / :func:`deseasonalize_inplace` — convenience
  wrappers around :func:`tsecon.x13.run` that default to ``x11(save=:d11)``
  and replace the input series' values with the resulting ``d11``. Mirrors
  ``X13.jl:42-58``.

The Julia module-level :func:`__init__` that warns on >5 leftover ``x13_*``
folders (``X13.jl:92-98``) is ported as an importlib-finder side effect
deferred to first-use rather than import-time: a no-op at ``import
tsecon.x13`` so the import cost stays cheap, with the warning emitted on
the first :func:`run` call. M2.5 records the rationale.
"""

from __future__ import annotations

__all__: list[str] = []
