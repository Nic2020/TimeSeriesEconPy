# SPDX-License-Identifier: MIT
"""X-13ARIMA-SEATS spec types and spec builders.

Mirrors ``TimeSeriesEcon.jl/src/x13/x13spec.jl`` (4,137 LOC — the bulk of
M2). Lands across **M2.1 / M2.2 / M2.3 / M2.4**.

Content planned (Julia line refs are to ``x13spec.jl``):

M2.1 — :class:`X13var` dataclasses + scalar spec types
================================================================

Abstract base :class:`X13var` plus the 25 concrete leaf types the spec
builders accept as outlier / regressor arguments. Each ships ``__str__``
producing the X13as ``.spc``-grammar token; pickling / equality from
``@dataclass(frozen=True, slots=True)``.

* Point outliers (``MIT``): ``ao`` (``x13spec.jl:9-11``), ``ls``
  (``20-22``), ``tc`` (``31-33``), ``so`` (``35-37``).
* Range outliers (``MIT, MIT`` or ``MITRange``): ``aos`` (``12-18``),
  ``lss`` (``23-29``), ``rp`` (``39-45``), ``qd`` (``47-53``), ``qi``
  (``54-60``), ``tl`` (``62-68``).
* Trading-day regressors (``MIT, regimechange``): ``td``
  (``96-103``), ``tdnolpyear`` (``104-111``), ``td1coef`` (``112-119``),
  ``td1nolpyear`` (``120-127``).
* Trading-day calendar (``n: int``): ``tdstock`` (``70-72``),
  ``tdstock1coef`` (``73-75``).
* Calendar regressors (``n: int``): ``easter`` (``76-78``), ``labor``
  (``79-81``), ``thank`` (``82-84``), ``sceaster`` (``86-88``),
  ``easterstock`` (``89-91``).
* Calendar regressor (``n: list[int]``): ``sincos`` (``92-94``).
* Length-of-period / leap-year (``MIT, regimechange``): ``lpyear``
  (``129-135``), ``lom`` (``137-143``), ``loq`` (``146-152``).
* Seasonal regressor: ``seasonal`` (``154-...``).

Plus :class:`ArimaSpec` (``230``) and :class:`ArimaModel` (``243``) — the
generic ARIMA(p, d, q)(P, D, Q) builders the ``arima`` / ``pickmdl``
builders consume.

The :class:`X13default` sentinel (``x13spec.jl:4-5``) ports to a
module-level singleton ``_X13DEFAULT = X13default()`` so call-site
identity comparisons (``arg is _X13DEFAULT``) match the Julia
``isa X13default`` dispatch shape.

M2.2 — High-traffic spec builders
====================================

8 builders that every realistic spec uses:

* :func:`series` (``x13spec.jl:732``) — wraps a :class:`TSeries` into a
  :class:`X13series` carrying frequency + period info.
* :func:`x11` (``3180``), :func:`seats` (``2478``) — the two seasonal
  adjustment engines (mutually exclusive in a single spec).
* :func:`arima` (``873``), :func:`automdl` (``1034``) — explicit and
  automatic model selection.
* :func:`transform` (``2928``), :func:`regression` (``2219``) — pre-
  filtering + regressor declarations.
* :func:`forecast` (``1409``) — forecast horizon + diagnostics.

M2.3 — Rare spec builders
===========================

11 builders that round out parity:

* :func:`outlier` (``1833``), :func:`history` (``1602``),
  :func:`identify` (``1686``), :func:`check` (``1149``),
  :func:`estimate` (``1228``), :func:`metadata` (``1721``),
  :func:`pickmdl` (``1972``), :func:`force` (``1343``),
  :func:`slidingspans` (``2666``), :func:`spectrum` (``2785``),
  :func:`x11regression` (``3420``).

M2.4 — :class:`X13spec` container + validation
================================================

:class:`X13spec` (``x13spec.jl:4138``) is the per-frequency typed
container the builders accumulate into. :func:`newspec` (``578``) /
:func:`newspec_from_frequency` (``603``) are the constructors;
:func:`validateX13spec` (``3563``) is the cross-builder invariant check
that runs before :func:`run` ships the spec to the binary.

Notes on Python idiom
======================

The Julia upstream relies on multiple dispatch (``aos(x::UnitRange{<:MIT})``
vs ``aos(mit1::MIT, mit2::MIT)``) for the range-outlier dual constructor.
The Python port lands these as two staticmethod-style alternates
``aos.from_range(mr: MITRange)`` and the canonical
``aos(mit1: MIT, mit2: MIT)`` constructor (or accepts either via
``__new__`` runtime-type dispatch). Decision deferred to M2.1 commit.

The Julia builders accept ``Pair{Symbol,Any}`` / ``Vector{Pair{...}}`` for
metadata; the Python port uses ``dict[str, str]`` for :func:`metadata`
and tuple/list of dataclass instances for outlier / regressor arguments,
matching the broader tsecon pattern (no ``Pair`` analogue in Python).
"""

from __future__ import annotations

__all__: list[str] = []
