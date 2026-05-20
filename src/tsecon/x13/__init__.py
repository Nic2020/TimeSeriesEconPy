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

The public surface this subpackage will eventually export:

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

M2.1 landed the spec-type surface (session 49): 26 :class:`X13var` leaves,
:class:`RegimeChange`, :class:`X13default`, :class:`ArimaSpec`, and
:class:`ArimaModel` are re-exported below.

M2.2 landed the high-traffic spec builders (session 50): :class:`Span`,
the eight container dataclasses (:class:`X13series`, :class:`X13arima`,
:class:`X13automdl`, :class:`X13transform`, :class:`X13regression`,
:class:`X13forecast`, :class:`X13seats`, :class:`X13x11`), and their
eight builder functions (:func:`series`, :func:`arima`, :func:`automdl`,
:func:`transform`, :func:`regression`, :func:`forecast`, :func:`seats`,
:func:`x11`). Each builder validates its kwargs and returns a frozen
dataclass; the ``.spc`` text emission is deferred to M2.4
(:mod:`tsecon.x13._write`).

M2.3 lands the eleven rare spec builders (this session): the eleven
container dataclasses (:class:`X13check`, :class:`X13estimate`,
:class:`X13force`, :class:`X13history`, :class:`X13identify`,
:class:`X13metadata`, :class:`X13outlier`, :class:`X13pickmdl`,
:class:`X13slidingspans`, :class:`X13spectrum`, :class:`X13x11regression`)
and their eleven builder functions (:func:`check`, :func:`estimate`,
:func:`force`, :func:`history`, :func:`identify`, :func:`metadata`,
:func:`outlier`, :func:`pickmdl`, :func:`slidingspans`, :func:`spectrum`,
:func:`x11regression`). Same shape pattern as M2.2 — frozen-slotted
containers + builders with per-builder validation. Closes the full
19-builder X-13 surface; the remaining M2 work is the writer (M2.4),
result-side workspace (M2.5), and wheels distribution (M2.6).

``tsecon.x13`` is now re-exported from the top-level ``tsecon/__init__.py``
(``import tsecon; tsecon.x13.ao(2020 // 1)`` works). The other private
siblings (``_x13`` / ``_write`` / ``_result``) ship empty ``__all__`` until
their owning sub-milestone lands.
"""

from __future__ import annotations

from tsecon.x13._spec import (
    ArimaModel,
    ArimaSpec,
    RegimeChange,
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
    X13spectrum,
    X13transform,
    X13var,
    X13x11,
    X13x11regression,
    ao,
    aos,
    arima,
    automdl,
    check,
    easter,
    easterstock,
    estimate,
    force,
    forecast,
    history,
    identify,
    labor,
    lom,
    loq,
    lpyear,
    ls,
    lss,
    metadata,
    outlier,
    pickmdl,
    qd,
    qi,
    regression,
    rp,
    sceaster,
    seasonal,
    seats,
    series,
    sincos,
    slidingspans,
    so,
    spectrum,
    tc,
    td,
    td1coef,
    td1nolpyear,
    tdnolpyear,
    tdstock,
    tdstock1coef,
    thank,
    tl,
    transform,
    x11,
    x11regression,
)

__all__ = [
    "ArimaModel",
    "ArimaSpec",
    "RegimeChange",
    "Span",
    "X13arima",
    "X13automdl",
    "X13check",
    "X13default",
    "X13estimate",
    "X13force",
    "X13forecast",
    "X13history",
    "X13identify",
    "X13metadata",
    "X13outlier",
    "X13pickmdl",
    "X13regression",
    "X13seats",
    "X13series",
    "X13slidingspans",
    "X13spectrum",
    "X13transform",
    "X13var",
    "X13x11",
    "X13x11regression",
    "ao",
    "aos",
    "arima",
    "automdl",
    "check",
    "easter",
    "easterstock",
    "estimate",
    "force",
    "forecast",
    "history",
    "identify",
    "labor",
    "lom",
    "loq",
    "lpyear",
    "ls",
    "lss",
    "metadata",
    "outlier",
    "pickmdl",
    "qd",
    "qi",
    "regression",
    "rp",
    "sceaster",
    "seasonal",
    "seats",
    "series",
    "sincos",
    "slidingspans",
    "so",
    "spectrum",
    "tc",
    "td",
    "td1coef",
    "td1nolpyear",
    "tdnolpyear",
    "tdstock",
    "tdstock1coef",
    "thank",
    "tl",
    "transform",
    "x11",
    "x11regression",
]
