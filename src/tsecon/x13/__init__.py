# SPDX-License-Identifier: MIT
"""X-13ARIMA-SEATS wrapper (subpackage).

Mirrors ``TimeSeriesEcon.jl/src/x13/``. The Julia upstream's five-file split
maps to private siblings here; this ``__init__`` will re-export the public
surface once the sub-milestones land:

==========================  ===========================  ==================
Julia file                  Python sibling                Lands in
==========================  ===========================  ==================
``X13.jl``                  :mod:`tsecon.x13._x13`        M2.5 (`cleanup`,
                                                         `deseasonalize`,
                                                         `WorkspaceTable`,
                                                         `X13ResultWorkspace`)
``x13consts.jl``            :mod:`tsecon.x13._consts`     M2.1 (output-
                                                         description tables)
``x13spec.jl``              :mod:`tsecon.x13._spec`       M2.1 (`X13var`
                                                         types,
                                                         `ArimaSpec`,
                                                         `ArimaModel`,
                                                         `X13spec`),
                                                         M2.2 (high-traffic
                                                         builders), M2.3
                                                         (rare builders),
                                                         M2.4
                                                         (`validateX13spec`)
``x13write.jl``              :mod:`tsecon.x13._write`     M2.4 (.spc serializer)
``x13result.jl``             :mod:`tsecon.x13._result`    M2.5 (`X13lazy`,
                                                         `X13result`,
                                                         :func:`run`,
                                                         readers)
==========================  ===========================  ==================

The public surface this subpackage will eventually export — locked in
[decision 24](../../../../claude_files/decisions/24_x13_kickoff.md):

* :func:`run` — execute X-13ARIMA-SEATS on a spec; returns an
  :class:`X13result`.
* :func:`deseasonalize` / :func:`deseasonalize_inplace` — convenience over
  ``run`` for the common ``d11`` extraction.
* :func:`cleanup` — remove stale ``x13_*`` temp folders the process leaked.
* :class:`WorkspaceTable`, :class:`X13ResultWorkspace`, :class:`X13result`,
  :class:`X13lazy` — result-side containers.
* :class:`X13spec`, :class:`ArimaSpec`, :class:`ArimaModel`,
  :func:`validateX13spec`, :func:`newspec` — spec-side surface.
* 19 spec builders: :func:`series`, :func:`x11`, :func:`seats`, :func:`arima`,
  :func:`automdl`, :func:`transform`, :func:`regression`, :func:`forecast`,
  :func:`outlier`, :func:`history`, :func:`identify`, :func:`check`,
  :func:`estimate`, :func:`metadata`, :func:`pickmdl`, :func:`force`,
  :func:`slidingspans`, :func:`spectrum`, :func:`x11regression`.
* 25 :class:`X13var` types: ``ao`` / ``ls`` / ``tc`` / ``so`` (point
  outliers), ``aos`` / ``lss`` / ``rp`` / ``qd`` / ``qi`` / ``tl`` (range
  outliers), ``td`` / ``tdnolpyear`` / ``td1coef`` / ``td1nolpyear`` /
  ``tdstock`` / ``tdstock1coef`` (trading-day regressors), ``easter`` /
  ``labor`` / ``thank`` / ``sceaster`` / ``easterstock`` / ``sincos``
  (calendar regressors), ``lpyear`` / ``lom`` / ``loq`` / ``seasonal``
  (leap-year / length-of-period / seasonal).

Binary distribution
-------------------

The ``x13as`` executable ships **in the wheel**. The override
``tsecon.setoption("x13path", "/path/to/x13as")`` points at a
user-installed binary (mirrors Julia's ``TimeSeriesEcon.setoption(:x13path,
"...")``). Default is ``""`` → use the bundled binary.

Sub-decisions locked in M2.0 follow-up (this session):

* **Sourcing path:** compile from Census Bureau Fortran source via
  ``fortran-lang/setup-fortran@v1`` in ``wheels.yml`` (the same GitHub
  Action the R ``x13binary`` package uses for its cross-platform
  prebuilds).
* **Version pin:** X-13ARIMA-SEATS v1.1, **build b60** — matches the
  Yggdrasil ``X13as_jll`` pin so the M2.5 numerical-fidelity test
  (Python ``d11`` vs Julia ``d11`` at 1e-10) compares binary parity, not
  binary version drift. Bumps follow the Yggdrasil cadence.
* **Wheels matrix delta:** none. The existing
  ``{ubuntu-latest, windows-latest, macos-latest} × {cp311, cp312,
  cp313}`` matrix from `wheels.yml` carries over; the new step set
  installs gfortran, downloads the Census source tarball, runs ``make
  -f makefile.gf``, and places ``x13as`` under
  :mod:`tsecon.x13._binary` for :func:`importlib.resources.files` lookup
  at runtime. arm64 wheels stay deferred until user demand surfaces
  (existing matrix already covers macOS arm64 via ``macos-latest``).

Stub modules below ship with empty ``__all__`` lists; the public surface
is wired in M2.1+. ``tsecon.x13`` is intentionally **not** re-exported
from the top-level ``tsecon/__init__.py`` until M2.1 lands at least one
public symbol; users access it via ``import tsecon.x13``.
"""

from __future__ import annotations

__all__: list[str] = []
