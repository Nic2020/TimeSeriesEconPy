# SPDX-License-Identifier: MIT
"""X-13ARIMA-SEATS wrapper (subpackage).

Mirrors ``TimeSeriesEcon.jl/src/x13/``. The Julia upstream's five-file split
maps to private siblings here; this ``__init__`` will re-export the public
surface once the sub-milestones land:

==========================  ===========================  ==================
Julia file                  Python sibling                Lands in
==========================  ===========================  ==================
``X13.jl``                  [`tsecon.x13._x13`][tsecon.x13._x13]        M2.5 (`cleanup`,
                                                         `deseasonalize`,
                                                         `WorkspaceTable`,
                                                         `X13ResultWorkspace`)
``x13consts.jl``            `tsecon.x13._consts`     M2.1 (output-
                                                         description tables)
``x13spec.jl``              [`tsecon.x13._spec`][tsecon.x13._spec]       M2.1 (`X13var`
                                                         types,
                                                         `ArimaSpec`,
                                                         `ArimaModel`,
                                                         `X13spec`),
                                                         M2.2 (high-traffic
                                                         builders), M2.3
                                                         (rare builders),
                                                         M2.4
                                                         (`validateX13spec`)
``x13write.jl``              [`tsecon.x13._write`][tsecon.x13._write]     M2.4 (.spc serializer)
``x13result.jl``             `tsecon.x13._result`    M2.5 (`X13lazy`,
                                                         `X13result`,
                                                         [`run`][tsecon.x13._result.run],
                                                         readers)
==========================  ===========================  ==================

The public surface this subpackage will eventually export:

* [`run`][tsecon.x13._result.run] — execute X-13ARIMA-SEATS on a spec; returns an
  [`X13result`][tsecon.x13._result.X13result].
* [`deseasonalize`][tsecon.x13._x13.deseasonalize] /
  [`deseasonalize_inplace`][tsecon.x13._x13.deseasonalize_inplace] — convenience over ``run`` for
  the common ``d11`` extraction.
* [`cleanup`][tsecon.x13._x13.cleanup] — remove stale ``x13_*`` temp folders the process leaked.
* [`WorkspaceTable`][tsecon.x13._result.WorkspaceTable],
  [`X13ResultWorkspace`][tsecon.x13._result.X13ResultWorkspace],
  [`X13result`][tsecon.x13._result.X13result], [`X13lazy`][tsecon.x13._result.X13lazy] — result-side
  containers.
* [`X13spec`][tsecon.x13._spec.X13spec], [`ArimaSpec`][tsecon.x13._spec.ArimaSpec],
  [`ArimaModel`][tsecon.x13._spec.ArimaModel],
  [`validateX13spec`][tsecon.x13._spec.validateX13spec], [`newspec`][tsecon.x13._spec.newspec] —
  spec-side surface.
* 19 spec builders: [`series`][tsecon.x13._spec.series], [`x11`][tsecon.x13._spec.x11],
  [`seats`][tsecon.x13._spec.seats], [`arima`][tsecon.x13._spec.arima],
  [`automdl`][tsecon.x13._spec.automdl], [`transform`][tsecon.x13._spec.transform],
  [`regression`][tsecon.x13._spec.regression], [`forecast`][tsecon.x13._spec.forecast],
  [`outlier`][tsecon.x13._spec.outlier], [`history`][tsecon.x13._spec.history],
  [`identify`][tsecon.x13._spec.identify], [`check`][tsecon.x13._spec.check],
  [`estimate`][tsecon.x13._spec.estimate], [`metadata`][tsecon.x13._spec.metadata],
  [`pickmdl`][tsecon.x13._spec.pickmdl], [`force`][tsecon.x13._spec.force],
  [`slidingspans`][tsecon.x13._spec.slidingspans], [`spectrum`][tsecon.x13._spec.spectrum],
  [`x11regression`][tsecon.x13._spec.x11regression].
* 25 [`X13var`][tsecon.x13._spec.X13var] types: ``ao`` / ``ls`` / ``tc`` / ``so`` (point
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
  `tsecon.x13._binary` for [`importlib.resources.files`][importlib.resources.files] lookup
  at runtime. arm64 wheels stay deferred until user demand surfaces
  (existing matrix already covers macOS arm64 via ``macos-latest``).

M2.1 landed the spec-type surface (session 49): 26 [`X13var`][tsecon.x13._spec.X13var] leaves,
[`RegimeChange`][tsecon.x13._spec.RegimeChange], [`X13default`][tsecon.x13._spec.X13default],
[`ArimaSpec`][tsecon.x13._spec.ArimaSpec], and [`ArimaModel`][tsecon.x13._spec.ArimaModel] are
re-exported below.

M2.2 landed the high-traffic spec builders (session 50): [`Span`][tsecon.x13._spec.Span],
the eight container dataclasses ([`X13series`][tsecon.x13._spec.X13series],
[`X13arima`][tsecon.x13._spec.X13arima], [`X13automdl`][tsecon.x13._spec.X13automdl],
[`X13transform`][tsecon.x13._spec.X13transform], [`X13regression`][tsecon.x13._spec.X13regression],
[`X13forecast`][tsecon.x13._spec.X13forecast], [`X13seats`][tsecon.x13._spec.X13seats],
[`X13x11`][tsecon.x13._spec.X13x11]), and their eight builder functions
([`series`][tsecon.x13._spec.series], [`arima`][tsecon.x13._spec.arima],
[`automdl`][tsecon.x13._spec.automdl], [`transform`][tsecon.x13._spec.transform],
[`regression`][tsecon.x13._spec.regression], [`forecast`][tsecon.x13._spec.forecast],
[`seats`][tsecon.x13._spec.seats], [`x11`][tsecon.x13._spec.x11]). Each builder validates its kwargs
and returns a frozen dataclass; the ``.spc`` text emission is deferred to M2.4
([`tsecon.x13._write`][tsecon.x13._write]).

M2.3 lands the eleven rare spec builders (this session): the eleven
container dataclasses ([`X13check`][tsecon.x13._spec.X13check],
[`X13estimate`][tsecon.x13._spec.X13estimate], [`X13force`][tsecon.x13._spec.X13force],
[`X13history`][tsecon.x13._spec.X13history], [`X13identify`][tsecon.x13._spec.X13identify],
[`X13metadata`][tsecon.x13._spec.X13metadata], [`X13outlier`][tsecon.x13._spec.X13outlier],
[`X13pickmdl`][tsecon.x13._spec.X13pickmdl], [`X13slidingspans`][tsecon.x13._spec.X13slidingspans],
[`X13spectrum`][tsecon.x13._spec.X13spectrum],
[`X13x11regression`][tsecon.x13._spec.X13x11regression]) and their eleven builder functions
([`check`][tsecon.x13._spec.check], [`estimate`][tsecon.x13._spec.estimate],
[`force`][tsecon.x13._spec.force], [`history`][tsecon.x13._spec.history],
[`identify`][tsecon.x13._spec.identify], [`metadata`][tsecon.x13._spec.metadata],
[`outlier`][tsecon.x13._spec.outlier], [`pickmdl`][tsecon.x13._spec.pickmdl],
[`slidingspans`][tsecon.x13._spec.slidingspans], [`spectrum`][tsecon.x13._spec.spectrum],
[`x11regression`][tsecon.x13._spec.x11regression]). Same shape pattern as M2.2 — frozen-slotted
containers + builders with per-builder validation. Closes the full 19-builder X-13 surface; the
remaining M2 work is the writer (M2.4), result-side workspace (M2.5), and wheels distribution
(M2.6).

M2.4 lands the [`X13spec`][tsecon.x13._spec.X13spec] aggregator (mutable + slots, mirroring
Julia's ``mutable struct``), [`newspec`][tsecon.x13._spec.newspec] factory, the
[`validateX13spec`][tsecon.x13._spec.validateX13spec] cross-spec invariant check (~50 branches:
ARIMA-source mutual exclusion, regression-variable type compatibility,
data-range containment, slidingspans bounds, soft warnings for HP
filter sample-size / modelspan-vs-span / spectrum.qcheck /
forcecal / etc.), and the ``.spc`` serializer in
[`tsecon.x13._write`][tsecon.x13._write] (top-level [`x13write`][tsecon.x13._write.x13write] +
[`impose_line_length`][tsecon.x13._write.impose_line_length] line-wrap helper). The serializer
mirrors the Julia upstream's column-budget logic (132 minus the 8-space block
indent), preferring ``" + "`` splits on ``print=`` lists when present.

M2.5 lands the result-side surface (this session): the
[`WorkspaceTable`][tsecon.x13._result.WorkspaceTable] and
[`X13ResultWorkspace`][tsecon.x13._result.X13ResultWorkspace] workspace subclasses (lazy
materialisation of [`X13lazy`][tsecon.x13._result.X13lazy] proxies on first attribute / key access),
[`X13result`][tsecon.x13._result.X13result], the [`run`][tsecon.x13._result.run] entry that invokes
the binary on a populated spec (or raw spec string), the
[`loadresult`][tsecon.x13._result.loadresult] dispatcher, the eight per-format readers
([`x13read_series`][tsecon.x13._result.x13read_series],
[`x13read_workspace_table`][tsecon.x13._result.x13read_workspace_table],
[`x13read_key_values`][tsecon.x13._result.x13read_key_values],
[`x13read_udg`][tsecon.x13._result.x13read_udg],
[`x13read_estimates`][tsecon.x13._result.x13read_estimates],
[`x13read_model`][tsecon.x13._result.x13read_model],
[`x13read_identify`][tsecon.x13._result.x13read_identify],
[`x13read_seatsseries`][tsecon.x13._result.x13read_seatsseries],
[`x13read_err`][tsecon.x13._result.x13read_err]), and the
[`deseasonalize`][tsecon.x13._x13.deseasonalize] /
[`deseasonalize_inplace`][tsecon.x13._x13.deseasonalize_inplace] convenience wrappers +
[`cleanup`][tsecon.x13._x13.cleanup] sweep. The end-to-end binary-fidelity test (Julia ``d11`` vs
Python ``d11`` to 1e-10) is deferred to M2.6 alongside the wheels-side gfortran binary; until then
[`run`][tsecon.x13._result.run] raises a clear error message when no binary is reachable via
``setoption("x13path", ...)``.

``tsecon.x13`` is re-exported from the top-level
``tsecon/__init__.py`` (``import tsecon; tsecon.x13.ao(2020 // 1)``
works); the M2.5 result surface lands alongside.
"""

from __future__ import annotations

from tsecon.x13._result import (
    WorkspaceTable,
    X13lazy,
    X13result,
    X13ResultWorkspace,
    loadresult,
    run,
    x13read_err,
    x13read_estimates,
    x13read_identify,
    x13read_key_values,
    x13read_model,
    x13read_seatsseries,
    x13read_series,
    x13read_udg,
    x13read_workspace_table,
)
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
    X13spec,
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
    newspec,
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
    validateX13spec,
    x11,
    x11regression,
)
from tsecon.x13._write import impose_line_length, x13write
from tsecon.x13._x13 import (
    cleanup,
    deseasonalize,
    deseasonalize_inplace,
    get_cleanup_folders,
)

__all__ = [
    "ArimaModel",
    "ArimaSpec",
    "RegimeChange",
    "Span",
    "WorkspaceTable",
    "X13ResultWorkspace",
    "X13arima",
    "X13automdl",
    "X13check",
    "X13default",
    "X13estimate",
    "X13force",
    "X13forecast",
    "X13history",
    "X13identify",
    "X13lazy",
    "X13metadata",
    "X13outlier",
    "X13pickmdl",
    "X13regression",
    "X13result",
    "X13seats",
    "X13series",
    "X13slidingspans",
    "X13spec",
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
    "cleanup",
    "deseasonalize",
    "deseasonalize_inplace",
    "easter",
    "easterstock",
    "estimate",
    "force",
    "forecast",
    "get_cleanup_folders",
    "history",
    "identify",
    "impose_line_length",
    "labor",
    "loadresult",
    "lom",
    "loq",
    "lpyear",
    "ls",
    "lss",
    "metadata",
    "newspec",
    "outlier",
    "pickmdl",
    "qd",
    "qi",
    "regression",
    "rp",
    "run",
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
    "validateX13spec",
    "x11",
    "x11regression",
    "x13read_err",
    "x13read_estimates",
    "x13read_identify",
    "x13read_key_values",
    "x13read_model",
    "x13read_seatsseries",
    "x13read_series",
    "x13read_udg",
    "x13read_workspace_table",
    "x13write",
]
