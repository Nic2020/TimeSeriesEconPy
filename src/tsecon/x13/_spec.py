# SPDX-License-Identifier: MIT
"""X-13ARIMA-SEATS spec types and spec builders.

Mirrors ``TimeSeriesEcon.jl/src/x13/x13spec.jl`` (4,137 LOC — the bulk of
M2). Lands across **M2.1 / M2.2 / M2.3 / M2.4**.

M2.1 — :class:`X13var` dataclasses + ARIMA-spec types (this session)
====================================================================

This module ships:

* :class:`X13default` sentinel + module-private :data:`_X13DEFAULT` singleton.
* :class:`RegimeChange` :class:`~enum.StrEnum` for the trading-day / seasonal
  regime-change qualifier (``both`` / ``zerobefore`` / ``zeroafter`` /
  ``neither``).
* :class:`X13var` abstract base + 26 concrete leaves the spec builders
  accept as outlier / regressor arguments. Each ``__str__`` produces the
  X13as ``.spc``-grammar token; pickling and equality come from
  ``@dataclass(frozen=True, slots=True)``.

  * Point outliers (``mit: MIT``): ``ao`` (``x13spec.jl:9-11``), ``ls``
    (``20-22``), ``tc`` (``31-33``), ``so`` (``35-37``).
  * Range outliers (``mit1, mit2: MIT``; classmethod
    :meth:`~aos.from_range`): ``aos`` (``12-18``), ``lss`` (``23-29``),
    ``rp`` (``39-45``), ``qd`` (``47-53``), ``qi`` (``54-60``), ``tl``
    (``62-68``).
  * Trading-day regressors with optional regime change
    (``mit: MIT | None, regimechange: RegimeChange``): ``td``
    (``96-103``), ``tdnolpyear`` (``104-111``), ``td1coef``
    (``112-119``), ``td1nolpyear`` (``120-127``).
  * Trading-day calendar (``n: int``): ``tdstock`` (``70-72``),
    ``tdstock1coef`` (``73-75``).
  * Calendar regressors (``n: int``): ``easter`` (``76-78``), ``labor``
    (``79-81``), ``thank`` (``82-84``), ``sceaster`` (``86-88``),
    ``easterstock`` (``89-91``).
  * Calendar regressor (``n: tuple[int, ...]``): ``sincos`` (``92-94``).
  * Length-of-period / leap-year (``mit: MIT | None, regimechange``):
    ``lpyear`` (``129-135``), ``lom`` (``137-143``), ``loq`` (``146-152``).
  * Seasonal regressor (``mit, regimechange``): ``seasonal`` (``154-161``).

  The Julia upstream's :func:`ao` etc. are types whose lowercase names are
  the X13as keywords. The Python port preserves the lowercase names (with
  a per-file ``N801`` lint ignore in ``pyproject.toml``) because the
  ``__str__`` output uses ``type(self).__name__`` — capitalising the
  classes would break .spc serialization or require a parallel name map.

  The Julia source has **26** ``X13var`` types.

* :class:`ArimaSpec` and :class:`ArimaModel` — the generic
  ``(p, d, q)(P, D, Q)[period]`` ARIMA builders the :func:`arima` /
  :func:`pickmdl` spec builders (M2.2 / M2.3) consume.

M2.2 — High-traffic spec builders (later session)
==================================================

8 builders that every realistic spec uses: :func:`series`
(``x13spec.jl:732``); :func:`x11` (``3180``), :func:`seats` (``2478``);
:func:`arima` (``873``), :func:`automdl` (``1034``); :func:`transform`
(``2928``), :func:`regression` (``2219``); :func:`forecast` (``1409``).

M2.3 — Rare spec builders (later session)
==========================================

11 builders that round out parity: :func:`outlier` (``1833``),
:func:`history` (``1602``), :func:`identify` (``1686``), :func:`check`
(``1149``), :func:`estimate` (``1228``), :func:`metadata` (``1721``),
:func:`pickmdl` (``1972``), :func:`force` (``1343``), :func:`slidingspans`
(``2666``), :func:`spectrum` (``2785``), :func:`x11regression`
(``3420``).

M2.4 — :class:`X13spec` container + validation (later session)
================================================================

:class:`X13spec` (``x13spec.jl:4138``) is the per-frequency typed
container the builders accumulate into. :func:`newspec` (``578``) /
:func:`newspec_from_frequency` (``603``) are the constructors;
:func:`validateX13spec` (``3563``) is the cross-builder invariant check
that runs before :func:`run` ships the spec to the binary.

Notes on Python idiom (M2.1)
============================

The Julia upstream's dual-constructor for range outliers
(``aos(::UnitRange{<:MIT})`` vs ``aos(mit1::MIT, mit2::MIT)``) ports as
the canonical two-argument constructor plus an :meth:`~aos.from_range`
:func:`classmethod` accepting an :class:`~tsecon.mitrange.MITRange`. This
keeps the frozen-slots dataclass shape uniform and matches how
:class:`MIT` itself surfaces alternate constructors (``MIT.from_yp``).

The Julia ``regimechange::Symbol`` field ports as a :class:`RegimeChange`
:class:`~enum.StrEnum`; users may pass either the enum member
(``RegimeChange.BOTH``) or the bare string (``"both"``), the
:meth:`__post_init__` normalises to the enum. Members are equal to their
``.value`` (``RegimeChange.BOTH == "both"`` is :data:`True`) so equality
checks remain transparent.

The Julia ``td()`` / ``td(mit::MIT)`` / ``td(mit, rc)`` constructor trio
maps to a single Python dataclass with sentinel defaults
(``mit=None, regimechange=None``); :meth:`__post_init__` resolves
``(None, None)`` to ``regimechange=NEITHER`` (no-arg form) and
``(mit, None)`` to ``regimechange=BOTH`` (mit-only form, matching
Julia's default). The same pattern applies to the seven other
regime-bearing types (``tdnolpyear``, ``td1coef``, ``td1nolpyear``,
``lpyear``, ``lom``, ``loq``, ``seasonal``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Final

import numpy as np

from tsecon.frequencies import Monthly, Yearly, ppy
from tsecon.mit import MIT, mit2yp
from tsecon.mitrange import MITRange
from tsecon.mvtseries import MVTSeries
from tsecon.tseries import TSeries
from tsecon.x13._consts import _ORDERED_MONTH_NAMES

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


# ---------------------------------------------------------------------------
# X13default sentinel
# ---------------------------------------------------------------------------


class X13default:
    """Sentinel for spec-builder fields that should be left at the X-13 default.

    Mirrors Julia's ``X13default`` struct (``x13spec.jl:4``). Spec builders
    (M2.2 / M2.3) accept :class:`X13default` instances as default arguments;
    the ``.spc`` writer (M2.4) skips fields whose value ``is _X13DEFAULT``,
    leaving the binary to apply its built-in default.

    The class is a singleton — :data:`_X13DEFAULT` is the canonical instance,
    and identity comparisons (``arg is _X13DEFAULT``) replace Julia's
    ``isa X13default`` dispatch.
    """

    __slots__ = ()
    _instance: ClassVar[X13default | None] = None

    def __new__(cls) -> X13default:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "X13default()"


_X13DEFAULT: Final[X13default] = X13default()


# ---------------------------------------------------------------------------
# RegimeChange enum
# ---------------------------------------------------------------------------


class RegimeChange(StrEnum):
    """Regime-change qualifier for ``td*``, ``lpyear``, ``lom``, ``loq``, ``seasonal``.

    Maps to X-13ARIMA-SEATS ``.spc`` tokens:

    * :attr:`NEITHER` — no regime change; the MIT is unused in serialization.
    * :attr:`BOTH` — break at ``mit``, both halves estimated.
      Token: ``td/2020.jul/``.
    * :attr:`ZEROBEFORE` — break with zero coefficient before ``mit``.
      Token: ``td//2020.jul/``.
    * :attr:`ZEROAFTER` — break with zero coefficient after ``mit``.
      Token: ``td/2020.jul//``.

    :class:`RegimeChange` is a :class:`~enum.StrEnum`; members compare equal
    to their underlying string (``RegimeChange.BOTH == "both"``). Spec
    constructors normalise plain strings to enum members via
    :meth:`__post_init__`.
    """

    BOTH = "both"
    ZEROBEFORE = "zerobefore"
    ZEROAFTER = "zeroafter"
    NEITHER = "neither"


_REGIME_START: Final[dict[RegimeChange, str]] = {
    RegimeChange.BOTH: "/",
    RegimeChange.ZEROBEFORE: "//",
    RegimeChange.ZEROAFTER: "/",
}
_REGIME_END: Final[dict[RegimeChange, str]] = {
    RegimeChange.BOTH: "/",
    RegimeChange.ZEROBEFORE: "/",
    RegimeChange.ZEROAFTER: "//",
}


# ---------------------------------------------------------------------------
# MIT → .spc string helper
# ---------------------------------------------------------------------------


def _mit_to_spc(mit: MIT) -> str:
    """Serialise an MIT to its X-13 ``.spc`` form.

    Mirrors ``TimeSeriesEcon.jl/src/x13/x13write.jl:286-293``. Returns
    ``"<year>.<month-abbr>"`` for :class:`~tsecon.frequencies.Monthly` MITs
    (e.g. ``"2020.jul"``) and ``"<year>.<period>"`` for every other
    year/period frequency (e.g. ``"2020.3"`` for Quarterly).
    """
    year_, period_ = mit2yp(mit)
    if isinstance(mit.frequency, Monthly):
        return f"{year_}.{_ORDERED_MONTH_NAMES[period_ - 1]}"
    return f"{year_}.{period_}"


# ---------------------------------------------------------------------------
# X13var base + 26 concrete leaves
# ---------------------------------------------------------------------------


class X13var:
    """Abstract base class for X-13 outlier / regressor / calendar variables.

    Each concrete subclass is a frozen-slotted dataclass that overrides
    :meth:`__str__` to produce the ``.spc``-grammar token. Subclasses use
    lowercase Julia-mirror names (``ao``, ``ls``, …); the class name is
    consumed by serialization via ``type(self).__name__`` — capitalising
    it would either break ``.spc`` output or require a parallel name map.
    """

    __slots__ = ()


# -- Point outliers ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ao(X13var):
    """Additive outlier at ``mit`` (regARIMA AO regressor).

    Mirrors ``x13spec.jl:9-11``. Serializes as ``ao<year>.<period>``,
    e.g. ``ao2020.jul`` for Monthly or ``ao2020.3`` for Quarterly.
    """

    mit: MIT

    def __str__(self) -> str:
        return f"ao{_mit_to_spc(self.mit)}"


@dataclass(frozen=True, slots=True)
class ls(X13var):
    """Level-shift outlier at ``mit``.

    Mirrors ``x13spec.jl:20-22``. Serializes as ``ls<year>.<period>``.
    """

    mit: MIT

    def __str__(self) -> str:
        return f"ls{_mit_to_spc(self.mit)}"


@dataclass(frozen=True, slots=True)
class tc(X13var):
    """Temporary-change outlier at ``mit``.

    Mirrors ``x13spec.jl:31-33``. Serializes as ``tc<year>.<period>``.
    """

    mit: MIT

    def __str__(self) -> str:
        return f"tc{_mit_to_spc(self.mit)}"


@dataclass(frozen=True, slots=True)
class so(X13var):
    """Seasonal outlier at ``mit``.

    Mirrors ``x13spec.jl:35-37``. Serializes as ``so<year>.<period>``.
    """

    mit: MIT

    def __str__(self) -> str:
        return f"so{_mit_to_spc(self.mit)}"


# -- Range outliers ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class aos(X13var):
    """Range additive-outlier from ``mit1`` to ``mit2`` (inclusive).

    Mirrors ``x13spec.jl:12-18``. Serializes as
    ``aos<year1>.<period1>-<year2>.<period2>``. Construct with
    :meth:`from_range` to bridge an :class:`~tsecon.mitrange.MITRange`.
    """

    mit1: MIT
    mit2: MIT

    @classmethod
    def from_range(cls, mr: MITRange) -> aos:
        """Construct from an :class:`~tsecon.mitrange.MITRange`.

        Mirrors the Julia inner constructor
        ``aos(x::UnitRange{<:MIT}) = new(first(x), last(x))``. The Python
        :class:`~tsecon.mitrange.MITRange` is inclusive, so its
        :meth:`~tsecon.mitrange.MITRange.first` and
        :meth:`~tsecon.mitrange.MITRange.last` match the Julia range
        endpoints.
        """
        return cls(mr.first(), mr.last())

    def __str__(self) -> str:
        return f"aos{_mit_to_spc(self.mit1)}-{_mit_to_spc(self.mit2)}"


@dataclass(frozen=True, slots=True)
class lss(X13var):
    """Range level-shift outlier from ``mit1`` to ``mit2``.

    Mirrors ``x13spec.jl:23-29``.
    """

    mit1: MIT
    mit2: MIT

    @classmethod
    def from_range(cls, mr: MITRange) -> lss:
        """Construct from an :class:`~tsecon.mitrange.MITRange`."""
        return cls(mr.first(), mr.last())

    def __str__(self) -> str:
        return f"lss{_mit_to_spc(self.mit1)}-{_mit_to_spc(self.mit2)}"


@dataclass(frozen=True, slots=True)
class rp(X13var):
    """Ramp outlier from ``mit1`` to ``mit2``.

    Mirrors ``x13spec.jl:39-45``.
    """

    mit1: MIT
    mit2: MIT

    @classmethod
    def from_range(cls, mr: MITRange) -> rp:
        """Construct from an :class:`~tsecon.mitrange.MITRange`."""
        return cls(mr.first(), mr.last())

    def __str__(self) -> str:
        return f"rp{_mit_to_spc(self.mit1)}-{_mit_to_spc(self.mit2)}"


@dataclass(frozen=True, slots=True)
class qd(X13var):
    """Quadratic-decay outlier range from ``mit1`` to ``mit2``.

    Mirrors ``x13spec.jl:47-53``.
    """

    mit1: MIT
    mit2: MIT

    @classmethod
    def from_range(cls, mr: MITRange) -> qd:
        """Construct from an :class:`~tsecon.mitrange.MITRange`."""
        return cls(mr.first(), mr.last())

    def __str__(self) -> str:
        return f"qd{_mit_to_spc(self.mit1)}-{_mit_to_spc(self.mit2)}"


@dataclass(frozen=True, slots=True)
class qi(X13var):
    """Quadratic-incline outlier range from ``mit1`` to ``mit2``.

    Mirrors ``x13spec.jl:54-60``.
    """

    mit1: MIT
    mit2: MIT

    @classmethod
    def from_range(cls, mr: MITRange) -> qi:
        """Construct from an :class:`~tsecon.mitrange.MITRange`."""
        return cls(mr.first(), mr.last())

    def __str__(self) -> str:
        return f"qi{_mit_to_spc(self.mit1)}-{_mit_to_spc(self.mit2)}"


@dataclass(frozen=True, slots=True)
class tl(X13var):
    """Temporary-level-shift outlier range from ``mit1`` to ``mit2``.

    Mirrors ``x13spec.jl:62-68``.
    """

    mit1: MIT
    mit2: MIT

    @classmethod
    def from_range(cls, mr: MITRange) -> tl:
        """Construct from an :class:`~tsecon.mitrange.MITRange`."""
        return cls(mr.first(), mr.last())

    def __str__(self) -> str:
        return f"tl{_mit_to_spc(self.mit1)}-{_mit_to_spc(self.mit2)}"


# -- Calendar / trading-day-stock regressors with integer ``n`` -------------


@dataclass(frozen=True, slots=True)
class tdstock(X13var):
    """Trading-day stock regressor with calendar-day-of-month ``n``.

    Mirrors ``x13spec.jl:70-72``. Serializes as ``tdstock[n]``.
    """

    n: int

    def __str__(self) -> str:
        return f"tdstock[{self.n}]"


@dataclass(frozen=True, slots=True)
class tdstock1coef(X13var):
    """One-coefficient trading-day stock regressor with day ``n``.

    Mirrors ``x13spec.jl:73-75``.
    """

    n: int

    def __str__(self) -> str:
        return f"tdstock1coef[{self.n}]"


@dataclass(frozen=True, slots=True)
class easter(X13var):
    """Easter holiday regressor with window length ``n`` days.

    Mirrors ``x13spec.jl:76-78``.
    """

    n: int

    def __str__(self) -> str:
        return f"easter[{self.n}]"


@dataclass(frozen=True, slots=True)
class labor(X13var):
    """U.S. Labor Day regressor with window length ``n`` days.

    Mirrors ``x13spec.jl:79-81``.
    """

    n: int

    def __str__(self) -> str:
        return f"labor[{self.n}]"


@dataclass(frozen=True, slots=True)
class thank(X13var):
    """U.S. Thanksgiving regressor with window length ``n`` days.

    Mirrors ``x13spec.jl:82-84``.
    """

    n: int

    def __str__(self) -> str:
        return f"thank[{self.n}]"


@dataclass(frozen=True, slots=True)
class sceaster(X13var):
    """Statistics-Canada Easter regressor with window length ``n`` days.

    Mirrors ``x13spec.jl:86-88``.
    """

    n: int

    def __str__(self) -> str:
        return f"sceaster[{self.n}]"


@dataclass(frozen=True, slots=True)
class easterstock(X13var):
    """Easter-stock regressor with window length ``n`` days.

    Mirrors ``x13spec.jl:89-91``.
    """

    n: int

    def __str__(self) -> str:
        return f"easterstock[{self.n}]"


# -- sincos calendar regressor (tuple of frequencies) -----------------------


@dataclass(frozen=True, slots=True)
class sincos(X13var):
    """Sine/cosine seasonal-frequency regressor with frequencies ``n``.

    Mirrors ``x13spec.jl:92-94``. Julia uses ``n::Vector{Int64}``; the
    Python port uses ``tuple[int, ...]`` so the dataclass stays frozen.
    Serializes as ``sincos[<space-separated frequencies>]``.
    """

    n: tuple[int, ...]

    def __post_init__(self) -> None:
        # Runtime check guards against users passing a list (the
        # Julia-flavored form); the type annotation says tuple, so mypy
        # marks the error path unreachable — that's by design.
        if not isinstance(self.n, tuple):
            msg = (  # type: ignore[unreachable]
                f"sincos.n must be a tuple of ints, got "
                f"{type(self.n).__name__}; pass tuple(...) explicitly."
            )
            raise TypeError(msg)

    def __str__(self) -> str:
        return f"sincos[{' '.join(str(k) for k in self.n)}]"


# -- Trading-day regressors with optional regime change ---------------------
# Implementation note: Julia's three constructors
#   td(mit, rc) → both explicit
#   td(mit)     → mit only; rc defaults to :both
#   td()        → no args; mit defaults to a placeholder 1M1, rc=:neither
# port as a single Python dataclass with sentinel ``None`` defaults; the
# `__post_init__` resolves the sentinel pair to match Julia's per-arity
# default chain. Validation raises if the result would be a regime change
# without an MIT.


_RegimeChangeArg = RegimeChange | str | None


def _resolve_regime(
    mit: MIT | None,
    regimechange: _RegimeChangeArg,
) -> RegimeChange:
    """Resolve the (mit, regimechange) sentinel pair to a concrete RegimeChange.

    Mirrors Julia's three-constructor default chain for ``td`` /
    ``tdnolpyear`` / ``td1coef`` / ``td1nolpyear`` / ``lpyear`` / ``lom`` /
    ``loq`` / ``seasonal`` in one helper.

    * ``(None, None)`` → :attr:`RegimeChange.NEITHER` (Julia ``td()``).
    * ``(mit, None)`` → :attr:`RegimeChange.BOTH` (Julia ``td(mit)``).
    * ``(_, rc)`` with explicit rc → enum-normalized rc.

    Raises :exc:`ValueError` if the resolved regime change requires an
    ``mit`` but none was provided.
    """
    if regimechange is None:
        resolved = RegimeChange.NEITHER if mit is None else RegimeChange.BOTH
    elif isinstance(regimechange, RegimeChange):
        resolved = regimechange
    else:
        resolved = RegimeChange(regimechange)
    if resolved is not RegimeChange.NEITHER and mit is None:
        msg = f"regimechange={resolved.value!r} requires an MIT argument; got mit=None."
        raise ValueError(msg)
    return resolved


def _regime_str(name: str, mit: MIT | None, regimechange: RegimeChange) -> str:
    """Render a regime-bearing X13var to its ``.spc`` token.

    Mirrors ``x13write.jl:281``. ``regimechange=NEITHER`` renders the
    bare class name (e.g. ``td``); any other regime renders the X-13
    bracketed form (``td/2020.jul/`` for BOTH, ``td//2020.jul/`` for
    ZEROBEFORE, ``td/2020.jul//`` for ZEROAFTER).
    """
    if regimechange is RegimeChange.NEITHER:
        return name
    assert mit is not None  # invariant established by _resolve_regime
    return f"{name}{_REGIME_START[regimechange]}{_mit_to_spc(mit)}{_REGIME_END[regimechange]}"


@dataclass(frozen=True, slots=True)
class td(X13var):
    """Trading-day regressor with optional regime change at ``mit``.

    Mirrors ``x13spec.jl:96-103``. Three call shapes mirror the Julia
    upstream:

    * ``td()`` — no regime change. .spc form: ``td``.
    * ``td(mit)`` — regime change at ``mit``, both halves estimated. .spc
      form: ``td/<mit>/``.
    * ``td(mit, regimechange=...)`` — explicit regime-change qualifier.
    """

    mit: MIT | None = None
    regimechange: RegimeChange = RegimeChange.NEITHER

    def __init__(
        self,
        mit: MIT | None = None,
        regimechange: _RegimeChangeArg = None,
    ) -> None:
        resolved = _resolve_regime(mit, regimechange)
        object.__setattr__(self, "mit", mit)
        object.__setattr__(self, "regimechange", resolved)

    def __str__(self) -> str:
        return _regime_str("td", self.mit, self.regimechange)


@dataclass(frozen=True, slots=True)
class tdnolpyear(X13var):
    """Trading-day regressor (no leap-year term) with optional regime change.

    Mirrors ``x13spec.jl:104-111``.
    """

    mit: MIT | None = None
    regimechange: RegimeChange = RegimeChange.NEITHER

    def __init__(
        self,
        mit: MIT | None = None,
        regimechange: _RegimeChangeArg = None,
    ) -> None:
        resolved = _resolve_regime(mit, regimechange)
        object.__setattr__(self, "mit", mit)
        object.__setattr__(self, "regimechange", resolved)

    def __str__(self) -> str:
        return _regime_str("tdnolpyear", self.mit, self.regimechange)


@dataclass(frozen=True, slots=True)
class td1coef(X13var):
    """One-coefficient trading-day regressor with optional regime change.

    Mirrors ``x13spec.jl:112-119``.
    """

    mit: MIT | None = None
    regimechange: RegimeChange = RegimeChange.NEITHER

    def __init__(
        self,
        mit: MIT | None = None,
        regimechange: _RegimeChangeArg = None,
    ) -> None:
        resolved = _resolve_regime(mit, regimechange)
        object.__setattr__(self, "mit", mit)
        object.__setattr__(self, "regimechange", resolved)

    def __str__(self) -> str:
        return _regime_str("td1coef", self.mit, self.regimechange)


@dataclass(frozen=True, slots=True)
class td1nolpyear(X13var):
    """One-coefficient trading-day regressor without leap-year term.

    Mirrors ``x13spec.jl:120-127``.
    """

    mit: MIT | None = None
    regimechange: RegimeChange = RegimeChange.NEITHER

    def __init__(
        self,
        mit: MIT | None = None,
        regimechange: _RegimeChangeArg = None,
    ) -> None:
        resolved = _resolve_regime(mit, regimechange)
        object.__setattr__(self, "mit", mit)
        object.__setattr__(self, "regimechange", resolved)

    def __str__(self) -> str:
        return _regime_str("td1nolpyear", self.mit, self.regimechange)


# -- Length-of-period / leap-year regressors --------------------------------


@dataclass(frozen=True, slots=True)
class lpyear(X13var):
    """Leap-year regressor with optional regime change.

    Mirrors ``x13spec.jl:129-135``.
    """

    mit: MIT | None = None
    regimechange: RegimeChange = RegimeChange.NEITHER

    def __init__(
        self,
        mit: MIT | None = None,
        regimechange: _RegimeChangeArg = None,
    ) -> None:
        resolved = _resolve_regime(mit, regimechange)
        object.__setattr__(self, "mit", mit)
        object.__setattr__(self, "regimechange", resolved)

    def __str__(self) -> str:
        return _regime_str("lpyear", self.mit, self.regimechange)


@dataclass(frozen=True, slots=True)
class lom(X13var):
    """Length-of-month regressor with optional regime change.

    Mirrors ``x13spec.jl:137-143``.
    """

    mit: MIT | None = None
    regimechange: RegimeChange = RegimeChange.NEITHER

    def __init__(
        self,
        mit: MIT | None = None,
        regimechange: _RegimeChangeArg = None,
    ) -> None:
        resolved = _resolve_regime(mit, regimechange)
        object.__setattr__(self, "mit", mit)
        object.__setattr__(self, "regimechange", resolved)

    def __str__(self) -> str:
        return _regime_str("lom", self.mit, self.regimechange)


@dataclass(frozen=True, slots=True)
class loq(X13var):
    """Length-of-quarter regressor with optional regime change.

    Mirrors ``x13spec.jl:146-152``.
    """

    mit: MIT | None = None
    regimechange: RegimeChange = RegimeChange.NEITHER

    def __init__(
        self,
        mit: MIT | None = None,
        regimechange: _RegimeChangeArg = None,
    ) -> None:
        resolved = _resolve_regime(mit, regimechange)
        object.__setattr__(self, "mit", mit)
        object.__setattr__(self, "regimechange", resolved)

    def __str__(self) -> str:
        return _regime_str("loq", self.mit, self.regimechange)


@dataclass(frozen=True, slots=True)
class seasonal(X13var):
    """Seasonal regressor with optional regime change.

    Mirrors ``x13spec.jl:154-161``.
    """

    mit: MIT | None = None
    regimechange: RegimeChange = RegimeChange.NEITHER

    def __init__(
        self,
        mit: MIT | None = None,
        regimechange: _RegimeChangeArg = None,
    ) -> None:
        resolved = _resolve_regime(mit, regimechange)
        object.__setattr__(self, "mit", mit)
        object.__setattr__(self, "regimechange", resolved)

    def __str__(self) -> str:
        return _regime_str("seasonal", self.mit, self.regimechange)


# ---------------------------------------------------------------------------
# ArimaSpec / ArimaModel
# ---------------------------------------------------------------------------


_OrderValue = int | tuple[int, ...] | X13default
_OrderArg = int | tuple[int, ...] | list[int] | X13default


def _coerce_order(value: _OrderArg) -> _OrderValue:
    """Coerce an ARIMA order argument to its stored shape.

    Julia accepts ``Int64`` or ``Vector{Int64}`` (the latter spells operators
    with missing lags, e.g. ``ArimaSpec([2,3],0,0)`` for
    ``(1 - Φ_2*B^2 - Φ_3*B^3)*z_t = a_t``). The Python port stores the
    list form as a :class:`tuple` so the dataclass stays frozen-hashable;
    bare ``int`` and :class:`X13default` pass through.
    """
    if isinstance(value, X13default):
        return value
    if isinstance(value, bool):
        msg = "ARIMA order must be int or tuple/list of ints, not bool."
        raise TypeError(msg)
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)):
        out = tuple(value)
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in out):
            msg = "ARIMA order elements must be int."
            raise TypeError(msg)
        return out
    # Defensive catch-all: the type signature lists every allowed branch,
    # but the runtime check protects against callers bypassing the
    # annotation (e.g. via ``Any``-typed glue code in M2.2 builders).
    msg = (  # type: ignore[unreachable]
        f"ARIMA order must be int, tuple/list of ints, or X13default; got {type(value).__name__}."
    )
    raise TypeError(msg)


@dataclass(slots=True)
class ArimaSpec:
    r"""A single ARIMA(p, d, q) operator with optional period.

    Mirrors ``x13spec.jl:216-228``. ``p``, ``d``, ``q`` may be a non-negative
    :class:`int` (e.g. ``ArimaSpec(1, 1, 1)`` for ARIMA(1,1,1)) or a
    :class:`tuple` of ints to spell an operator with missing lags
    (``ArimaSpec((2, 3), 0, 0)`` →
    :math:`(1 - \Phi_2 B^2 - \Phi_3 B^3) z_t = a_t`). ``period=0`` (the
    default) means "infer the period from spec ordering" — the ARIMA
    builder (M2.2) uses position in the spec list to assign seasonal
    periods when ``period=0``.

    Mutable to mirror Julia's ``mutable struct``: spec builders re-assign
    fields as they accumulate. Equality compares field-by-field.
    """

    p: _OrderValue = 0
    d: _OrderValue = 0
    q: _OrderValue = 0
    period: int | X13default = 0

    def __init__(
        self,
        p: _OrderArg = 0,
        d: _OrderArg = 0,
        q: _OrderArg = 0,
        period: int | X13default = 0,
    ) -> None:
        self.p = _coerce_order(p)
        self.d = _coerce_order(d)
        self.q = _coerce_order(q)
        if isinstance(period, bool) or not isinstance(period, (int, X13default)):
            msg = f"ArimaSpec period must be int or X13default; got {type(period).__name__}."
            raise TypeError(msg)
        self.period = period

    @classmethod
    def two_seasonal(
        cls,
        p: _OrderArg,
        d: _OrderArg,
        q: _OrderArg,
        P: _OrderArg,  # noqa: N803 — mirrors Julia's (p,d,q)(P,D,Q) signature
        D: _OrderArg,  # noqa: N803
        Q: _OrderArg,  # noqa: N803
    ) -> tuple[ArimaSpec, ArimaSpec]:
        """Build the ``(p,d,q)(P,D,Q)`` pair, mirroring Julia's 6-arg form.

        Returns a 2-tuple of :class:`ArimaSpec` — the nonseasonal then the
        seasonal — both with ``period=0`` (the ARIMA builder assigns the
        actual seasonal period from spec ordering).

        Mirrors ``x13spec.jl:227``
        (``ArimaSpec(p,d,q,P,D,Q) = (new(p,d,q,0), new(P,D,Q,0))``).
        """
        return cls(p, d, q, 0), cls(P, D, Q, 0)


@dataclass(slots=True)
class ArimaModel:
    """A collection of :class:`ArimaSpec` operators with a ``default`` flag.

    Mirrors ``x13spec.jl:232-242``. The ``default`` flag marks the model as
    the airline-default (``(0,1,1)(0,1,1)``) for ``automdl`` purposes; the
    :func:`automdl` builder (M2.2) consults it to decide whether to
    explicitly serialize the model in the ``.spc`` file or rely on the
    binary's built-in default.

    Use the :class:`classmethod`-style helpers below (mirroring Julia's
    six positional-only constructors) for the common ARIMA(p,d,q)[period]
    and ARIMA(p,d,q)(P,D,Q) call shapes; pass an explicit list of
    :class:`ArimaSpec` for fully custom multi-operator models.
    """

    specs: list[ArimaSpec]
    default: bool = False

    @classmethod
    def from_pdq(
        cls,
        p: _OrderArg,
        d: _OrderArg,
        q: _OrderArg,
        period: int = 0,
        *,
        default: bool = False,
    ) -> ArimaModel:
        """Build a single-operator ``(p, d, q)[period]`` model.

        Mirrors Julia's ``ArimaModel(p, d, q; default=...)`` and
        ``ArimaModel(p, d, q, period; default=...)``.
        """
        return cls([ArimaSpec(p, d, q, period)], default=default)

    @classmethod
    def from_pdq_seasonal(
        cls,
        p: _OrderArg,
        d: _OrderArg,
        q: _OrderArg,
        P: _OrderArg,  # noqa: N803
        D: _OrderArg,  # noqa: N803
        Q: _OrderArg,  # noqa: N803
        *,
        default: bool = False,
    ) -> ArimaModel:
        """Build a two-operator ``(p, d, q)(P, D, Q)`` model.

        Mirrors Julia's six-positional-argument
        ``ArimaModel(p, d, q, P, D, Q; default=...)``.
        """
        return cls([ArimaSpec(p, d, q, 0), ArimaSpec(P, D, Q, 0)], default=default)

    @classmethod
    def from_specs(
        cls,
        *specs: ArimaSpec,
        default: bool = False,
    ) -> ArimaModel:
        """Build from one or more pre-built :class:`ArimaSpec` instances.

        Mirrors Julia's ``ArimaModel(specs::ArimaSpec...; default=...)``.
        """
        if not specs:
            msg = "ArimaModel.from_specs requires at least one ArimaSpec."
            raise ValueError(msg)
        return cls(list(specs), default=default)


# ---------------------------------------------------------------------------
# Span helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Span:
    """A start/end MIT pair with optional open endpoints.

    Mirrors ``x13spec.jl:163-171``. Either endpoint may be :data:`None`
    (Julia ``missing``), meaning "use the underlying series' first or last
    observation". Both endpoints together define an inclusive range.

    The Julia upstream also accepts fuzzy ``end`` values such as ``M11`` or
    ``Q2`` (resolved to the most recent occurrence in the series); the
    Python port restricts ``b`` and ``e`` to :class:`MIT` or :data:`None`
    in M2.2. Fuzzy endings land in a later session if a builder (e.g.
    ``x11regression``) requires them.

    Construct from an :class:`~tsecon.mitrange.MITRange` via
    :meth:`from_range` (mirrors Julia's
    ``Span(x::UnitRange{<:MIT}) = new(first(x), last(x))``).
    """

    b: MIT | None = None
    e: MIT | None = None

    @classmethod
    def from_range(cls, mr: MITRange) -> Span:
        """Construct from an inclusive :class:`~tsecon.mitrange.MITRange`."""
        return cls(mr.first(), mr.last())


# ---------------------------------------------------------------------------
# X13series container + series() builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class X13series:
    """The ``series`` spec block — the time series + its load-time options.

    Mirrors ``x13spec.jl:175-199`` (``X13series{F<:Frequency}``). The Julia
    type-parameter ``F`` is dropped — the underlying :class:`~tsecon.tseries.TSeries`
    already carries its own :class:`~tsecon.frequencies.Frequency` instance.

    Constructed by :func:`series`; never construct directly outside the
    spec-builder path (validation happens in :func:`series`, not in
    ``__init__``).
    """

    appendbcst: bool | X13default
    appendfcst: bool | X13default
    comptype: str | X13default
    compwt: float | X13default
    data: TSeries
    decimals: int | X13default
    file: str | X13default
    format: str | X13default
    modelspan: MITRange | Span | X13default
    name: str | X13default
    period: int | X13default
    precision: int | X13default
    print: str | list[str] | X13default
    save: str | list[str] | X13default
    span: MITRange | Span | X13default
    start: MIT | X13default
    title: str | X13default
    type: str | X13default
    divpower: int | X13default
    missingcode: float | X13default
    missingval: float | X13default
    saveprecision: int | X13default
    trimzero: bool | str | X13default


_SERIES_PRINT_ALL: Final[list[str]] = [
    "default",
    "adjoriginal",
    "adjorigplot",
    "calendaradjorig",
    "outlieradjorig",
    "seriesplot",
]
_SERIES_SAVE_ALL: Final[list[str]] = [
    "span",
    "specfile",
    "adjoriginal",
    "calendaradjorig",
    "outlieradjorig",
    "seriesmvadj",
]


def _expand_all(
    value: str | list[str] | X13default,
    all_list: list[str],
) -> str | list[str] | X13default:
    """Mirror Julia's ``print=:all`` / ``save=:all`` expansion to a fixed list.

    Mirrors the per-builder pattern at e.g. ``x13spec.jl:814-819``:
    if the user passed ``"all"`` (scalar) or ``["all"]`` (1-element list),
    expand to the spec-specific "everything" list. Otherwise return
    ``value`` unchanged.
    """
    if isinstance(value, str) and value == "all":
        return list(all_list)
    if isinstance(value, list) and value == ["all"]:
        return list(all_list)
    return value


def _check_span_against(
    span: MITRange | Span | X13default,
    data: TSeries,
    *,
    arg_name: str = "span",
) -> None:
    """Verify a span argument lies within ``data.range``.

    Mirrors the ``span isa UnitRange`` / ``span isa Span`` containment
    checks at ``x13spec.jl:777-788``. ``X13default`` and absent endpoints
    short-circuit. Raises :exc:`ValueError` (Python idiom for
    Julia's ``ArgumentError``).
    """
    if isinstance(span, X13default):
        return
    data_first = data.range.first()
    data_last = data.range.last()
    if isinstance(span, MITRange):
        if span.first() < data_first or span.last() > data_last:
            msg = (
                f"{arg_name} ({span!r}) must be contained within the range "
                f"of the provided series ({data.range!r})."
            )
            raise ValueError(msg)
    elif isinstance(span, Span):
        if span.b is not None and span.b < data_first:
            msg = (
                f"The start of the specified {arg_name} must be on or after "
                f"the start of the provided series ({data_first!r}). "
                f"Received: {span.b!r}"
            )
            raise ValueError(msg)
        if span.e is not None and span.e > data_last:
            msg = (
                f"The end of the specified {arg_name} must be on or before "
                f"the end of the provided series ({data_last!r}). "
                f"Received: {span.e!r}"
            )
            raise ValueError(msg)


def series(
    t: TSeries,
    *,
    appendbcst: bool | X13default = _X13DEFAULT,
    appendfcst: bool | X13default = _X13DEFAULT,
    comptype: str | X13default = _X13DEFAULT,
    compwt: float | X13default = _X13DEFAULT,
    decimals: int | X13default = _X13DEFAULT,
    file: str | X13default = _X13DEFAULT,
    format: str | X13default = _X13DEFAULT,
    modelspan: MITRange | Span | X13default = _X13DEFAULT,
    name: str | X13default = _X13DEFAULT,
    period: int | X13default = _X13DEFAULT,
    precision: int | X13default = _X13DEFAULT,
    print: str | list[str] | X13default = _X13DEFAULT,
    save: str | list[str] | X13default = _X13DEFAULT,
    span: MITRange | Span | X13default = _X13DEFAULT,
    start: MIT | X13default = _X13DEFAULT,
    title: str | X13default = _X13DEFAULT,
    type: str | X13default = _X13DEFAULT,
    divpower: int | X13default = _X13DEFAULT,
    missingcode: float | X13default = _X13DEFAULT,
    missingval: float | X13default = _X13DEFAULT,
    saveprecision: int | X13default = _X13DEFAULT,
    trimzero: bool | str | X13default = _X13DEFAULT,
) -> X13series:
    """Build the ``series`` spec — required for every X-13 run.

    Mirrors ``x13spec.jl:732-822``. The ``t`` argument is the input
    :class:`~tsecon.tseries.TSeries`; all other arguments are keyword-only
    and mirror the Julia spec's named parameters one-for-one. Defaults are
    the :data:`_X13DEFAULT` sentinel — fields holding the sentinel are
    omitted by the M2.4 ``.spc`` writer, letting X-13 apply its built-in
    default.

    Validation performed:

    * ``name`` and ``title`` truncated (with a :class:`UserWarning`) to 64
      and 79 characters respectively, mirroring Julia.
    * ``period`` is auto-set to ``ppy(t)`` for non-Monthly / non-Yearly
      frequencies (Quarterly defaults to 4, etc.).
    * ``span`` (whether :class:`~tsecon.mitrange.MITRange` or
      :class:`Span`) must be contained within ``t.range``.
    * :class:`Span` ``span`` endpoints must be :class:`MIT` or
      :data:`None` — fuzzy endings (Julia's ``M11`` / ``Q2``) are
      rejected here per upstream.
    * ``divpower`` must be in ``[-9, 9]``.
    * If ``t.values`` contains ``NaN``, ``missingcode`` must be set; the
      output's ``data`` field has NaN replaced with ``missingcode`` in a
      fresh copy (input ``t`` is not mutated).
    * ``print="all"`` / ``save="all"`` expand to the upstream-defined
      "everything" lists.

    Returns an :class:`X13series` carrying the (possibly cropped /
    NaN-replaced) data and the validated field values.
    """
    if not isinstance(t, TSeries):
        # Runtime guard: the annotation says TSeries so mypy marks this path
        # unreachable. Callers that bypass the annotation (``Any``-typed glue
        # code, dynamic dispatch) still hit a sharp error rather than failing
        # mid-derivation on ``t.range``.
        msg = (  # type: ignore[unreachable]
            f"series() requires a TSeries; got {t.__class__.__name__}."
        )
        raise TypeError(msg)

    data = t.copy()
    if not isinstance(start, X13default):
        if start < data.range.first() or start > data.range.last():
            msg = f"series() start={start!r} must be within the series range {data.range!r}."
            raise ValueError(msg)
        # Crop to start..end (inclusive).
        start_idx = start - data.range.first()
        data = TSeries(start, data.values[start_idx:].copy())
        start_value: MIT | X13default = start
    else:
        start_value = data.range.first()

    if isinstance(name, str) and len(name) > 64:
        warnings.warn(
            f"Series name truncated to 64 characters. Full name: {name}",
            UserWarning,
            stacklevel=2,
        )
        name = name[:64]

    if isinstance(title, str) and len(title) > 79:
        warnings.warn(
            f"Series title truncated to 79 characters. Full title: {title}",
            UserWarning,
            stacklevel=2,
        )
        title = title[:79]

    if not isinstance(t.frequency, Monthly) and not isinstance(t.frequency, Yearly):
        period = ppy(t.frequency)

    _check_span_against(span, t, arg_name="span")
    _check_span_against(modelspan, t, arg_name="modelspan")

    if not isinstance(divpower, X13default) and (divpower < -9 or divpower > 9):
        msg = f"divpower values must be between -9 and 9 (inclusive). Received: {divpower}."
        raise ValueError(msg)

    if np.isnan(data.values).any():
        if isinstance(missingcode, X13default):
            msg = (
                "The provided tseries has NaN values but no `missingcode` "
                "was specified. Please specify a missingcode for your "
                "Series argument. I.e. missingcode = -99999.0."
            )
            raise ValueError(msg)
        data = data.copy()
        nan_mask = np.isnan(data.values)
        data.values[nan_mask] = missingcode

    print = _expand_all(print, _SERIES_PRINT_ALL)
    save = _expand_all(save, _SERIES_SAVE_ALL)

    return X13series(
        appendbcst=appendbcst,
        appendfcst=appendfcst,
        comptype=comptype,
        compwt=compwt,
        data=data,
        decimals=decimals,
        file=file,
        format=format,
        modelspan=modelspan,
        name=name,
        period=period,
        precision=precision,
        print=print,
        save=save,
        span=span,
        start=start_value,
        title=title,
        type=type,
        divpower=divpower,
        missingcode=missingcode,
        missingval=missingval,
        saveprecision=saveprecision,
        trimzero=trimzero,
    )


# ---------------------------------------------------------------------------
# X13arima container + arima() builder
# ---------------------------------------------------------------------------


_FloatOrNone = float | None
_ArArg = list[_FloatOrNone] | list[float] | X13default


@dataclass(frozen=True, slots=True)
class X13arima:
    """The ``arima`` spec block — the ARIMA(p, d, q)(P, D, Q) component.

    Mirrors ``x13spec.jl:245-252``. The ``model`` field is an
    :class:`ArimaModel` (one or more :class:`ArimaSpec` operators); the
    AR / MA initial-value vectors (``ar`` / ``ma``) and their fixed-flag
    counterparts (``fixar`` / ``fixma``) align position-by-position with
    the model's coefficient sequence. Missing initial values default to
    ``0.1`` in the binary, encoded as :data:`None` here.
    """

    model: ArimaModel
    title: str | X13default
    ar: list[_FloatOrNone] | list[float] | X13default
    ma: list[_FloatOrNone] | list[float] | X13default
    fixar: list[bool] | X13default
    fixma: list[bool] | X13default


def arima(
    model: ArimaModel | ArimaSpec | tuple[ArimaSpec, ...],
    *,
    title: str | X13default = _X13DEFAULT,
    ar: _ArArg = _X13DEFAULT,
    ma: _ArArg = _X13DEFAULT,
    fixar: list[bool] | X13default = _X13DEFAULT,
    fixma: list[bool] | X13default = _X13DEFAULT,
) -> X13arima:
    """Build the ``arima`` spec — the ARIMA part of the regARIMA model.

    Mirrors ``x13spec.jl:873-911``. Accepts any of:

    * an :class:`ArimaModel` (the primary Julia overload),
    * a single :class:`ArimaSpec` (wrapped via :meth:`ArimaModel.from_specs`),
    * a tuple of :class:`ArimaSpec` (wrapped via :meth:`ArimaModel.from_specs`).

    The two-positional-tuple form mirrors :meth:`ArimaSpec.two_seasonal`'s
    return shape, so ``arima(ArimaSpec.two_seasonal(1,1,1,0,1,1))`` ports
    Julia's ``arima(ArimaSpec(1,1,1,0,1,1)...)`` idiom.

    Validation:

    * ``fixar`` length must match ``ar`` length (if both set).
    * ``fixma`` length must match ``ma`` length (if both set).
    * ``title`` truncated to 79 chars with a :class:`UserWarning`.
    """
    if isinstance(model, ArimaSpec):
        wrapped: ArimaModel = ArimaModel.from_specs(model)
    elif isinstance(model, tuple):
        wrapped = ArimaModel.from_specs(*model)
    elif isinstance(model, ArimaModel):
        wrapped = model
    else:
        # Type annotation enumerates every branch above; this catch-all
        # guards Any-typed glue code from M2.4 (X13spec.arima = arima(...)).
        msg = (  # type: ignore[unreachable]
            f"arima() model must be ArimaModel, ArimaSpec, or tuple of "
            f"ArimaSpec; got {model.__class__.__name__}."
        )
        raise TypeError(msg)

    if (
        not isinstance(fixar, X13default)
        and not isinstance(ar, X13default)
        and len(fixar) != len(ar)
    ):
        msg = f"fixar must have the same length as ar. Provided ar={ar}, fixar={fixar}."
        raise ValueError(msg)
    if (
        not isinstance(fixma, X13default)
        and not isinstance(ma, X13default)
        and len(fixma) != len(ma)
    ):
        msg = f"fixma must have the same length as ma. Provided ma={ma}, fixma={fixma}."
        raise ValueError(msg)

    if isinstance(title, str) and len(title) > 79:
        warnings.warn(
            f"Arima title truncated to 79 characters. Full title: {title}",
            UserWarning,
            stacklevel=2,
        )
        title = title[:79]

    return X13arima(
        model=wrapped,
        title=title,
        ar=ar,
        ma=ma,
        fixar=fixar,
        fixma=fixma,
    )


# ---------------------------------------------------------------------------
# X13automdl container + automdl() builder
# ---------------------------------------------------------------------------


_IntOrNone = int | None


@dataclass(frozen=True, slots=True)
class X13automdl:
    """The ``automdl`` spec block — automatic ARIMA model selection.

    Mirrors ``x13spec.jl:254-272``. Auto-selects the ARIMA orders given
    bounds on the regular/seasonal differencing and ARMA polynomial
    degrees. Mutually exclusive with ``arima``.
    """

    diff: list[int] | X13default
    acceptdefault: bool | X13default
    checkmu: bool | X13default
    ljungboxlimit: float | X13default
    maxorder: list[_IntOrNone] | X13default
    maxdiff: list[_IntOrNone] | X13default
    mixed: bool | X13default
    print: str | list[str] | X13default
    savelog: str | list[str] | X13default
    armalimit: float | X13default
    balanced: bool | X13default
    exactdiff: bool | str | X13default
    fcstlim: int | X13default
    hrinitial: bool | X13default
    reducecv: float | X13default
    rejectfcst: bool | X13default
    urfinal: float | X13default


_AUTOMDL_DEFAULT_PRINT: Final[list[str]] = [
    "autochoice",
    "autochoicemdl",
    "autodefaulttests",
    "autofinaltests",
    "autoljungboxtest",
    "bestfivemdl",
    "header",
    "unitroottest",
    "unitroottestmdl",
]


def automdl(  # noqa: PLR0912
    *,
    diff: list[int] | X13default = _X13DEFAULT,
    acceptdefault: bool | X13default = _X13DEFAULT,
    checkmu: bool | X13default = _X13DEFAULT,
    ljungboxlimit: float | X13default = _X13DEFAULT,
    maxorder: list[_IntOrNone] | X13default = _X13DEFAULT,
    maxdiff: list[_IntOrNone] | X13default = _X13DEFAULT,
    mixed: bool | X13default = _X13DEFAULT,
    print: str | list[str] | X13default | None = None,
    savelog: str | list[str] | X13default | None = None,
    armalimit: float | X13default = _X13DEFAULT,
    balanced: bool | X13default = _X13DEFAULT,
    exactdiff: bool | str | X13default = _X13DEFAULT,
    fcstlim: int | X13default = _X13DEFAULT,
    hrinitial: bool | X13default = _X13DEFAULT,
    reducecv: float | X13default = _X13DEFAULT,
    rejectfcst: bool | X13default = _X13DEFAULT,
    urfinal: float | X13default = _X13DEFAULT,
) -> X13automdl:
    """Build the ``automdl`` spec — automatic ARIMA model selection.

    Mirrors ``x13spec.jl:1034-1141``. The ``print`` and ``savelog``
    arguments default (in upstream Julia) to specific multi-element lists,
    not :data:`_X13DEFAULT`. In Python the ``None`` sentinel triggers the
    same defaults (a non-:data:`X13default` default would otherwise be
    serialized as the *user's* choice, masking that the binary's default
    differs).

    Validation:

    * ``diff`` must be length 2; values ``∈ {0, 1, 2}`` (regular) and
      ``{0, 1}`` (seasonal).
    * ``maxdiff`` must be length 2; values ``∈ {1, 2}`` (regular) and
      ``{1}`` (seasonal). :data:`None` slots accepted.
    * ``maxorder`` must be length 2; values ``∈ {1, 2, 3, 4}`` (regular)
      and ``{1, 2}`` (seasonal). :data:`None` slots accepted.
    * ``diff`` is ignored if both ``diff`` and ``maxdiff`` are set; a
      :class:`UserWarning` flags the redundancy (upstream Julia parity).
    """
    if print is None:
        print = list(_AUTOMDL_DEFAULT_PRINT)
    if savelog is None:
        savelog = "alldiagnostics"

    if not isinstance(diff, X13default):
        if len(diff) != 2:
            msg = "The diff argument of the automdl spec must contain exactly two values."
            raise ValueError(msg)
        if diff[0] not in (0, 1, 2):
            msg = (
                f"Acceptable values for the regular differencing orders of "
                f"the automdl spec are 0, 1, and 2. Received: {diff[0]}."
            )
            raise ValueError(msg)
        if diff[1] not in (0, 1):
            msg = (
                f"Acceptable values for the seasonal differencing orders of "
                f"the automdl spec are 0 and 1. Received: {diff[1]}."
            )
            raise ValueError(msg)
        if not isinstance(maxdiff, X13default):
            warnings.warn(
                "The diff argument of the automdl spec will be ignored "
                "because a maxdiff argument is specified.",
                UserWarning,
                stacklevel=2,
            )

    if not isinstance(maxdiff, X13default):
        if len(maxdiff) != 2:
            msg = "The maxdiff argument of the automdl spec must contain exactly two values."
            raise ValueError(msg)
        if maxdiff[0] is not None and maxdiff[0] not in (1, 2):
            msg = (
                f"Acceptable values for the regular maximum differencing "
                f"orders of the automdl spec are 1 and 2. "
                f"Received: {maxdiff[0]}."
            )
            raise ValueError(msg)
        if maxdiff[1] is not None and maxdiff[1] not in (1,):
            msg = (
                f"The only acceptable value for the seasonal maximum "
                f"differencing order of the automdl spec is 1. "
                f"Received: {maxdiff[1]}."
            )
            raise ValueError(msg)

    if not isinstance(maxorder, X13default):
        if len(maxorder) != 2:
            msg = "The maxorder argument of the automdl spec must contain exactly two values."
            raise ValueError(msg)
        if maxorder[0] is not None and maxorder[0] not in (1, 2, 3, 4):
            msg = (
                f"The maximum order for the regular ARMA model must be "
                f"greater than zero and can be at most 4. "
                f"Received: {maxorder[0]}."
            )
            raise ValueError(msg)
        if maxorder[1] is not None and maxorder[1] not in (1, 2):
            msg = (
                f"The maximum order for the seasonal ARMA model can be "
                f"either 1 or 2. Received: {maxorder[1]}."
            )
            raise ValueError(msg)

    return X13automdl(
        diff=diff,
        acceptdefault=acceptdefault,
        checkmu=checkmu,
        ljungboxlimit=ljungboxlimit,
        maxorder=maxorder,
        maxdiff=maxdiff,
        mixed=mixed,
        print=print,
        savelog=savelog,
        armalimit=armalimit,
        balanced=balanced,
        exactdiff=exactdiff,
        fcstlim=fcstlim,
        hrinitial=hrinitial,
        reducecv=reducecv,
        rejectfcst=rejectfcst,
        urfinal=urfinal,
    )


# ---------------------------------------------------------------------------
# X13transform container + transform() builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class X13transform:
    """The ``transform`` spec block — pre-modeling data transformation.

    Mirrors ``x13spec.jl:467-486``. Selects between log, Box-Cox-power,
    or user-supplied prior-adjustment transformation modes.
    """

    adjust: str | X13default
    aicdiff: float | X13default
    data: TSeries | MVTSeries | X13default
    file: str | X13default
    format: str | X13default
    func: str | X13default
    mode: str | list[str] | X13default
    name: str | list[str] | X13default
    power: float | X13default
    precision: int | X13default
    print: str | list[str] | X13default
    save: str | list[str] | X13default
    savelog: str | list[str] | X13default
    start: MIT | list[MIT] | X13default
    title: str | X13default
    type: str | list[str] | X13default
    constant: float | X13default
    trimzero: bool | str | X13default


_TRANSFORM_PRINT_ALL: Final[list[str]] = [
    "aictransform",
    "seriesconstant",
    "seriesconstantplot",
    "prior",
    "permprior",
    "tempprior",
    "prioradjusted",
    "permprioradjusted",
    "prioradjustedptd",
    "permprioradjustedptd",
    "transformed",
]
_TRANSFORM_SAVE_ALL: Final[list[str]] = [
    "seriesconstant",
    "prior",
    "permprior",
    "tempprior",
    "prioradjusted",
    "permprioradjusted",
    "prioradjustedptd",
    "permprioradjustedptd",
    "transformed",
]


def transform(  # noqa: PLR0912
    *,
    adjust: str | X13default = _X13DEFAULT,
    aicdiff: float | X13default = _X13DEFAULT,
    data: TSeries | MVTSeries | X13default = _X13DEFAULT,
    file: str | X13default = _X13DEFAULT,
    format: str | X13default = _X13DEFAULT,
    func: str | X13default = _X13DEFAULT,
    mode: str | list[str] | X13default = _X13DEFAULT,
    power: float | X13default = _X13DEFAULT,
    precision: int | X13default = _X13DEFAULT,
    print: str | list[str] | X13default = _X13DEFAULT,
    save: str | list[str] | X13default = _X13DEFAULT,
    savelog: str | list[str] | X13default | None = None,
    title: str | X13default = _X13DEFAULT,
    type: str | list[str] | X13default = _X13DEFAULT,
    constant: float | X13default = _X13DEFAULT,
    trimzero: bool | str | X13default = _X13DEFAULT,
) -> X13transform:
    """Build the ``transform`` spec — pre-modeling transformation selector.

    Mirrors ``x13spec.jl:2928-3010``. Derives ``start`` and ``name`` from
    ``data`` when set (mirrors the Julia upstream's derivation block).

    Validation:

    * ``power`` and ``func`` are mutually exclusive.
    * ``adjust="lpyear"`` is only valid with a log-transform
      (``power=0.0`` or ``func="log"``).
    * ``mode`` is at most two values; ``"diff"`` is incompatible with
      ``"ratio"`` / ``"percent"`` in the same list.
    * ``title`` truncated to 79 chars with a :class:`UserWarning`.
    * ``type`` requires ``data`` set; ``type`` list length must match
      the number of provided series.
    """
    if savelog is None:
        savelog = "autotransform"

    start: MIT | list[MIT] | X13default = _X13DEFAULT
    name: str | list[str] | X13default = _X13DEFAULT
    if not isinstance(data, X13default):
        start = data.range.first()
        if isinstance(data, MVTSeries):
            names = list(data.column_names)
            name = names[0] if len(names) == 1 else names

    if not isinstance(func, X13default) and not isinstance(power, X13default):
        msg = "Either power or func can be specified, but not both."
        raise ValueError(msg)

    if not isinstance(adjust, X13default) and adjust == "lpyear":
        if not isinstance(power, X13default) and power != 0.0:
            msg = "adjust='lpyear' is only allowed when a log-transform (power=0.0) is specified."
            raise ValueError(msg)
        if not isinstance(func, X13default) and func != "log":
            msg = "adjust='lpyear' is only allowed when a log-transform (func='log') is specified."
            raise ValueError(msg)

    if isinstance(mode, list):
        if len(mode) > 2:
            msg = (
                f"Only up to two values can be included in the mode "
                f"argument. Received: {len(mode)}."
            )
            raise ValueError(msg)
        if "diff" in mode and ("ratio" in mode or "percent" in mode):
            msg = (
                f"The 'diff' mode is not compatible with the 'ratio' or "
                f"'percent' modes. Received: {mode}."
            )
            raise ValueError(msg)

    if isinstance(title, str) and len(title) > 79:
        warnings.warn(
            f"Transform title truncated to 79 characters. Full title: {title}",
            UserWarning,
            stacklevel=2,
        )
        title = title[:79]

    if not isinstance(type, X13default):
        if isinstance(data, X13default):
            msg = (
                "A user-defined prior-adjustment type is specified, but no data has been provided."
            )
            raise ValueError(msg)
        if isinstance(data, TSeries) and isinstance(type, list) and len(type) > 1:
            msg = (
                f"The number of user-defined prior adjustment types "
                f"provided ({len(type)}) must match the number of data "
                f"series provided (1)."
            )
            raise ValueError(msg)
        if isinstance(data, MVTSeries):
            ncols = data.shape[1]
            if isinstance(type, list) and len(type) != ncols:
                msg = (
                    f"The number of user-defined prior adjustment types "
                    f"provided ({len(type)}) must match the number of data "
                    f"series provided ({ncols})."
                )
                raise ValueError(msg)
            if isinstance(type, str) and ncols != 1:
                msg = (
                    f"The number of user-defined prior adjustment types "
                    f"provided (1) must match the number of data series "
                    f"provided ({ncols})."
                )
                raise ValueError(msg)

    print = _expand_all(print, _TRANSFORM_PRINT_ALL)
    save = _expand_all(save, _TRANSFORM_SAVE_ALL)

    return X13transform(
        adjust=adjust,
        aicdiff=aicdiff,
        data=data,
        file=file,
        format=format,
        func=func,
        mode=mode,
        name=name,
        power=power,
        precision=precision,
        print=print,
        save=save,
        savelog=savelog,
        start=start,
        title=title,
        type=type,
        constant=constant,
        trimzero=trimzero,
    )


# ---------------------------------------------------------------------------
# X13regression container + regression() builder
# ---------------------------------------------------------------------------


_VariableArg = str | X13var
_VariablesField = _VariableArg | list[_VariableArg] | X13default


@dataclass(frozen=True, slots=True)
class X13regression:
    """The ``regression`` spec block — regARIMA regressors / outliers.

    Mirrors ``x13spec.jl:383-407``. Carries the predefined regressor list
    (``variables``), the user-supplied regressors (``data`` /  ``user``),
    initial coefficient values (``b``) and fix-flags (``fixb``), and the
    AIC-comparison configuration (``aictest`` / ``aicdiff``).
    """

    aicdiff: list[_FloatOrNone] | list[float] | X13default
    aictest: str | list[str] | X13default
    chi2test: bool | X13default
    chi2testcv: float | X13default
    data: MVTSeries | X13default
    file: str | X13default
    format: str | X13default
    print: str | list[str] | X13default
    save: str | list[str] | X13default
    savelog: str | list[str] | X13default
    pvaictest: float | X13default
    start: MIT | X13default
    testalleaster: bool | X13default
    tlimit: float | X13default
    user: str | list[str] | X13default
    usertype: str | list[str] | X13default
    variables: _VariablesField
    b: list[float] | X13default
    fixb: list[bool] | X13default
    centeruser: str | X13default
    eastermeans: bool | X13default
    noapply: str | X13default
    tcrate: float | X13default


_REGRESSION_USERTYPE_ALLOWED: Final[frozenset[str]] = frozenset(
    {
        "constant",
        "seasonal",
        "td",
        "lom",
        "loq",
        "lpyear",
        "ao",
        "ls",
        "so",
        "transitory",
        "user",
        "holiday",
        "holiday2",
        "holiday3",
        "holiday4",
        "holiday5",
    }
)
_REGRESSION_AICTEST_ALLOWED: Final[frozenset[str]] = frozenset(
    {
        "td",
        "tdnolpyear",
        "tdstock",
        "td1coef",
        "td1nolpyear",
        "tdstock1coef",
        "lom",
        "loq",
        "lpyear",
        "easter",
        "easterstock",
        "user",
    }
)
_REGRESSION_PRINT_ALL: Final[list[str]] = [
    "regressionmatrix",
    "aictest",
    "outlier",
    "aoutlier",
    "levelshift",
    "seasonaloutlier",
    "transitory",
    "temporarychange",
    "tradingday",
    "holiday",
    "regseasonal",
    "userdef",
    "chi2test",
    "dailyweights",
]
_REGRESSION_SAVE_ALL: Final[list[str]] = [
    "regressionmatrix",
    "outlier",
    "aoutlier",
    "levelshift",
    "seasonaloutlier",
    "transitory",
    "temporarychange",
    "tradingday",
    "holiday",
    "regseasonal",
    "userdef",
]


def _check_calendar_variable_bounds(v: X13var) -> None:
    """Enforce per-type ``n`` bounds for calendar-int X13var regressors.

    Mirrors ``x13spec.jl:2294-2311``. Bounds are X-13 documented limits:
    ``tdstock`` ∈ [1, 31], ``easter`` ∈ [0, 25], ``labor`` ∈ [1, 25],
    ``thank`` ∈ [-8, 17], ``sceaster`` ∈ [1, 24], ``easterstock`` ∈ [1, 25].
    """
    if isinstance(v, tdstock):
        if v.n < 1 or v.n > 31:
            msg = (
                f"tdstock variables must have a value between 1 and 31 "
                f"(inclusive). Received: {v.n}."
            )
            raise ValueError(msg)
    elif isinstance(v, easter):
        if v.n < 0 or v.n > 25:
            msg = (
                f"easter variables must have a value between 1 and 25 (inclusive). Received: {v.n}."
            )
            raise ValueError(msg)
    elif isinstance(v, labor):
        if v.n < 1 or v.n > 25:
            msg = (
                f"labor variables must have a value between 1 and 25 (inclusive). Received: {v.n}."
            )
            raise ValueError(msg)
    elif isinstance(v, thank):
        if v.n < -8 or v.n > 17:
            msg = (
                f"thank variables must have a value between -8 and 17 (inclusive). Received: {v.n}."
            )
            raise ValueError(msg)
    elif isinstance(v, sceaster):
        if v.n < 1 or v.n > 24:
            msg = (
                f"sceaster variables must have a value between 1 and 24 "
                f"(inclusive). Received: {v.n}."
            )
            raise ValueError(msg)
    elif isinstance(v, easterstock) and (v.n < 1 or v.n > 25):
        msg = (
            f"easterstock variables must have a value between 1 and 25 "
            f"(inclusive). Received: {v.n}."
        )
        raise ValueError(msg)


def _check_outlier_overlaps(
    variables: list[_VariableArg],
) -> None:
    """Warn (via :class:`UserWarning`) on overlapping ``aos`` / ``lss`` ranges.

    Mirrors ``x13spec.jl:2319-2332``. Range outliers (``aos`` / ``lss``)
    in the same regression cause coefficient identification issues; the
    upstream warns, we follow.
    """
    aos_ranges: list[MITRange] = [MITRange(v.mit1, v.mit2) for v in variables if isinstance(v, aos)]
    lss_ranges: list[MITRange] = [MITRange(v.mit1, v.mit2) for v in variables if isinstance(v, lss)]
    for label, ranges in (("aos", aos_ranges), ("lss", lss_ranges)):
        for i in range(len(ranges)):
            for j in range(i + 1, len(ranges)):
                a, b = ranges[i], ranges[j]
                if not (a.last() < b.first() or b.last() < a.first()):
                    warnings.warn(
                        f"The variables argument has overlapping {label} "
                        f"specifications: {a!r} and {b!r}.",
                        UserWarning,
                        stacklevel=3,
                    )


def regression(  # noqa: PLR0912
    *,
    aicdiff: list[_FloatOrNone] | list[float] | X13default = _X13DEFAULT,
    aictest: str | list[str] | X13default = _X13DEFAULT,
    chi2test: bool | X13default = _X13DEFAULT,
    chi2testcv: float | X13default = _X13DEFAULT,
    data: MVTSeries | X13default = _X13DEFAULT,
    file: str | X13default = _X13DEFAULT,
    format: str | X13default = _X13DEFAULT,
    print: str | list[str] | X13default = _X13DEFAULT,
    save: str | list[str] | X13default = _X13DEFAULT,
    savelog: str | list[str] | X13default | None = None,
    pvaictest: float | X13default = _X13DEFAULT,
    testalleaster: bool | X13default = _X13DEFAULT,
    tlimit: float | X13default = _X13DEFAULT,
    usertype: str | list[str] | X13default = _X13DEFAULT,
    variables: _VariablesField = _X13DEFAULT,
    b: list[float] | X13default = _X13DEFAULT,
    fixb: list[bool] | X13default = _X13DEFAULT,
    centeruser: str | X13default = _X13DEFAULT,
    eastermeans: bool | X13default = _X13DEFAULT,
    noapply: str | X13default = _X13DEFAULT,
    tcrate: float | X13default = _X13DEFAULT,
) -> X13regression:
    """Build the ``regression`` spec — regressors / outliers for regARIMA.

    Mirrors ``x13spec.jl:2219-2354``. The ``start`` and ``user`` fields
    are derived from ``data`` (mirrors the Julia upstream's derivation;
    no user-facing ``start=`` / ``user=`` kwargs are accepted because
    Julia overrides them).

    Validation:

    * ``aicdiff`` and ``pvaictest`` are mutually exclusive.
    * ``usertype`` (when set) must be one of the documented X-13 effect
      types; vector form's length must match the number of user series.
    * ``aictest`` (when set) must be in the documented set of testable
      effects.
    * Per-variable ``n`` bounds checked for ``tdstock`` / ``easter`` /
      ``labor`` / ``thank`` / ``sceaster`` / ``easterstock``.
    * Overlapping ``aos`` / ``lss`` ranges emit a :class:`UserWarning`.
    """
    if savelog is None:
        savelog = ["aictest", "chi2test"]

    start: MIT | X13default = _X13DEFAULT
    user: str | list[str] | X13default = _X13DEFAULT
    if not isinstance(data, X13default):
        start = data.range.first()
        names = list(data.column_names)
        user = names[0] if len(names) == 1 else names

    if not isinstance(aicdiff, X13default) and not isinstance(pvaictest, X13default):
        msg = (
            "The aicdiff argument cannot be used in the same regression spec "
            "as the pvaictest argument."
        )
        raise ValueError(msg)

    if not isinstance(usertype, X13default):
        if (
            isinstance(usertype, list)
            and isinstance(user, list)
            and len(usertype) > 1
            and len(usertype) != len(user)
        ):
            msg = (
                f"The usertype argument must have the same length as "
                f"the number of user series provided ({len(user)}) when "
                f"more than a single type is specified. "
                f"Received: {usertype}"
            )
            raise ValueError(msg)
        if isinstance(usertype, list):
            bad = [u for u in usertype if u not in _REGRESSION_USERTYPE_ALLOWED]
            if bad:
                msg = (
                    f"The usertype argument can only have the following "
                    f"values: {sorted(_REGRESSION_USERTYPE_ALLOWED)}. "
                    f"Received: {usertype}"
                )
                raise ValueError(msg)
        elif isinstance(usertype, str) and usertype not in _REGRESSION_USERTYPE_ALLOWED:
            msg = (
                f"The usertype argument can only have the following values: "
                f"{sorted(_REGRESSION_USERTYPE_ALLOWED)}. "
                f"Received: {usertype}"
            )
            raise ValueError(msg)

    if not isinstance(variables, X13default):
        vars_list: list[_VariableArg] = (
            list(variables) if isinstance(variables, list) else [variables]
        )
        for v in vars_list:
            if isinstance(v, X13var):
                _check_calendar_variable_bounds(v)
        _check_outlier_overlaps(vars_list)

    if isinstance(aictest, str):
        if aictest not in _REGRESSION_AICTEST_ALLOWED:
            msg = (
                f"aictest can only contain these entries: "
                f"{sorted(_REGRESSION_AICTEST_ALLOWED)}. Received: {aictest}."
            )
            raise ValueError(msg)
    elif isinstance(aictest, list):
        bad_aic = [a for a in aictest if a not in _REGRESSION_AICTEST_ALLOWED]
        if bad_aic:
            msg = (
                f"aictest can only contain these entries: "
                f"{sorted(_REGRESSION_AICTEST_ALLOWED)}. Received: {aictest}."
            )
            raise ValueError(msg)

    print = _expand_all(print, _REGRESSION_PRINT_ALL)
    save = _expand_all(save, _REGRESSION_SAVE_ALL)

    return X13regression(
        aicdiff=aicdiff,
        aictest=aictest,
        chi2test=chi2test,
        chi2testcv=chi2testcv,
        data=data,
        file=file,
        format=format,
        print=print,
        save=save,
        savelog=savelog,
        pvaictest=pvaictest,
        start=start,
        testalleaster=testalleaster,
        tlimit=tlimit,
        user=user,
        usertype=usertype,
        variables=variables,
        b=b,
        fixb=fixb,
        centeruser=centeruser,
        eastermeans=eastermeans,
        noapply=noapply,
        tcrate=tcrate,
    )


# ---------------------------------------------------------------------------
# X13forecast container + forecast() builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class X13forecast:
    """The ``forecast`` spec block — out-of-sample forecast generation.

    Mirrors ``x13spec.jl:310-318``. ``maxlead`` controls how far ahead
    point forecasts and their variances are produced; ``maxback`` does
    the same for backcasts.
    """

    exclude: int | X13default
    lognormal: bool | X13default
    maxback: int | X13default
    maxlead: int | X13default
    print: str | list[str] | X13default
    save: str | list[str] | X13default
    probability: float | X13default


_FORECAST_PRINT_ALL: Final[list[str]] = [
    "transformed",
    "variances",
    "forecasts",
    "transformedbcst",
    "backcasts",
]
_FORECAST_SAVE_ALL: Final[list[str]] = list(_FORECAST_PRINT_ALL)


def forecast(
    *,
    exclude: int | X13default = _X13DEFAULT,
    lognormal: bool | X13default = _X13DEFAULT,
    maxback: int | X13default = _X13DEFAULT,
    maxlead: int | X13default = _X13DEFAULT,
    print: str | list[str] | X13default = _X13DEFAULT,
    save: str | list[str] | X13default = _X13DEFAULT,
    probability: float | X13default = _X13DEFAULT,
) -> X13forecast:
    """Build the ``forecast`` spec — forecast / backcast options.

    Mirrors ``x13spec.jl:1409-1429``. Expands ``print="all"`` /
    ``save="all"`` to the upstream-defined "everything" lists; otherwise
    pass-through.
    """
    print = _expand_all(print, _FORECAST_PRINT_ALL)
    save = _expand_all(save, _FORECAST_SAVE_ALL)
    return X13forecast(
        exclude=exclude,
        lognormal=lognormal,
        maxback=maxback,
        maxlead=maxlead,
        print=print,
        save=save,
        probability=probability,
    )


# ---------------------------------------------------------------------------
# X13seats container + seats() builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class X13seats:
    """The ``seats`` spec block — model-based signal-extraction adjustment.

    Mirrors ``x13spec.jl:409-430``. Activates the SEATS module for
    seasonal decomposition based on the regARIMA model.
    """

    appendfcst: bool | X13default
    finite: bool | X13default
    hpcycle: bool | X13default
    noadmiss: bool | X13default
    out: int | X13default
    print: str | list[str] | X13default
    save: str | list[str] | X13default
    savelog: str | list[str] | X13default
    printphtrf: bool | X13default
    qmax: int | X13default
    statseas: bool | X13default
    tabtables: list[str] | X13default
    bias: int | X13default
    epsiv: float | X13default
    epsphi: int | X13default
    hplan: int | X13default
    imean: bool | X13default
    maxit: int | X13default
    rmod: float | X13default
    xl: float | X13default


_SEATS_SAVE_ALL: Final[list[str]] = [
    "trend",
    "seasonal",
    "irregular",
    "seasonaladj",
    "transitory",
    "adjustfac",
    "adjustmentratio",
    "trendfcstdecomp",
    "seasonalfcstdecomp",
    "ofd",
    "seasonaladjfcstdecomp",
    "transitoryfcstdecomp",
    "seasadjconst",
    "trendconst",
    "totaladjustment",
    "difforiginal",
    "diffseasonaladj",
    "difftrend",
    "seasonalsum",
    "cycle",
    "longtermtrend",
    "componentmodels",
    "filtersaconc",
    "filtersasym",
    "filtertrendconc",
    "filtertrendsym",
    "squaredgainsaconc",
    "squaredgainsasym",
    "squaredgaintrendconc",
    "squaredgaintrendsym",
    "timeshiftsaconc",
    "timeshifttrendconc",
    "wkendfilter",
    "seasonalpct",
    "irregularpct",
    "transitorypct",
    "adjustfacpct",
]


def seats(
    *,
    appendfcst: bool | X13default = _X13DEFAULT,
    finite: bool | X13default = _X13DEFAULT,
    hpcycle: bool | X13default = _X13DEFAULT,
    noadmiss: bool | X13default = _X13DEFAULT,
    out: int | X13default = 0,
    print: str | list[str] | X13default = _X13DEFAULT,
    save: str | list[str] | X13default = _X13DEFAULT,
    printphtrf: bool | X13default = _X13DEFAULT,
    qmax: int | X13default = _X13DEFAULT,
    statseas: bool | X13default = _X13DEFAULT,
    tabtables: list[str] | X13default = _X13DEFAULT,
    bias: int | X13default = _X13DEFAULT,
    epsiv: float | X13default = _X13DEFAULT,
    epsphi: int | X13default = _X13DEFAULT,
    hplan: int | X13default = _X13DEFAULT,
    imean: bool | X13default = _X13DEFAULT,
    maxit: int | X13default = _X13DEFAULT,
    rmod: float | X13default = _X13DEFAULT,
    xl: float | X13default = _X13DEFAULT,
) -> X13seats:
    """Build the ``seats`` spec — SEATS signal-extraction options.

    Mirrors ``x13spec.jl:2478-2527``. The upstream's ``savelog`` default
    is the empty list to keep the binary happy on writethrough (see the
    ``savelog = _X13default`` line at ``x13spec.jl:2513``); we honour
    that and surface ``savelog`` as :data:`_X13DEFAULT` internally rather
    than as a user-tunable kwarg.

    Validation:

    * ``epsiv`` must be > 0.
    * Setting ``hplan`` while ``hpcycle=False`` emits a warning (HP
      filters are applied regardless because ``hplan`` is set).
    * ``print="all"`` is rejected (upstream Julia warns + drops); we
      raise :exc:`ValueError` to surface the misuse explicitly under
      the project's ``error::UserWarning`` filter.
    * ``save="all"`` expands to the SEATS-specific everything list AND
      forces ``out=0`` (mirrors upstream ``out=0`` reset).
    """
    if not isinstance(epsiv, X13default) and epsiv <= 0.0:
        msg = f"epsiv should be a small positive number. Received: {epsiv}."
        raise ValueError(msg)

    if (
        not isinstance(hpcycle, X13default)
        and not isinstance(hplan, X13default)
        and hpcycle is False
    ):
        warnings.warn(
            f"Hodrick-Prescott filters will be used even though hpcycle is "
            f"{hpcycle} because an hplan value has been specified.",
            UserWarning,
            stacklevel=2,
        )

    if (isinstance(print, str) and print == "all") or (
        isinstance(print, list) and print == ["all"]
    ):
        msg = (
            "The print='all' option is not available for the seats spec. "
            "Pass an explicit list of table names instead."
        )
        raise ValueError(msg)

    if (isinstance(save, str) and save == "all") or (isinstance(save, list) and save == ["all"]):
        save = list(_SEATS_SAVE_ALL)
        out = 0

    return X13seats(
        appendfcst=appendfcst,
        finite=finite,
        hpcycle=hpcycle,
        noadmiss=noadmiss,
        out=out,
        print=print,
        save=save,
        savelog=_X13DEFAULT,
        printphtrf=printphtrf,
        qmax=qmax,
        statseas=statseas,
        tabtables=tabtables,
        bias=bias,
        epsiv=epsiv,
        epsphi=epsphi,
        hplan=hplan,
        imean=imean,
        maxit=maxit,
        rmod=rmod,
        xl=xl,
    )


# ---------------------------------------------------------------------------
# X13x11 container + x11() builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class X13x11:
    """The ``x11`` spec block — X-11 (enhanced) seasonal adjustment.

    Mirrors ``x13spec.jl:489-510``. The traditional Census-Bureau X-11
    method, with auto-selection or per-period seasonal-MA configuration
    and Henderson trend-MA selection.
    """

    appendbcst: bool | X13default
    appendfcst: bool | X13default
    final: str | list[str] | X13default
    mode: str | X13default
    print: str | list[str] | X13default
    save: str | list[str] | X13default
    savelog: str | list[str] | X13default
    seasonalma: str | list[str] | X13default
    sigmalim: list[_FloatOrNone] | list[float] | X13default
    title: str | list[str] | X13default
    trendma: int | X13default
    type: str | X13default
    calendarsigma: str | X13default
    centerseasonal: bool | X13default
    keepholiday: bool | X13default
    print1stpass: bool | X13default
    sfshort: bool | X13default
    sigmavec: list[str] | X13default
    trendic: float | X13default
    true7term: bool | X13default


_X11_PRINT_ALL: Final[list[str]] = [
    "adjustdiff",
    "adjustfac",
    "adjustmentratio",
    "calendar",
    "calendaradjchanges",
    "combholiday",
    "ftestd8",
    "irregular",
    "irrwt",
    "movseasrat",
    "origchanges",
    "qstat",
    "replacsi",
    "residualseasf",
    "sachanges",
    "seasadj",
    "seasonal",
    "seasonaldiff",
    "tdaytype",
    "trend",
    "trendchanges",
    "unmodsi",
    "unmodsiox",
    "x11diag",
    "yrtotals",
    "adjoriginalc",
    "adjoriginald",
    "autosf",
    "extreme",
    "extremeb",
    "ftestb1",
    "irregularadjao",
    "irregularb",
    "irregularc",
    "irrwtb",
    "mcdmovavg",
    "modirregular",
    "modoriginal",
    "modseasadj",
    "modsic4",
    "modsid4",
    "replacsib4",
    "replacsib9",
    "replacsic9",
    "robustsa",
    "seasadjb11",
    "seasadjb6",
    "seasadjc11",
    "seasadjc6",
    "seasadjconst",
    "seasadjd6",
    "seasonalb10",
    "seasonalb5",
    "seasonalc10",
    "seasonalc5",
    "seasonald5",
    "sib3",
    "sib8",
    "tdadjorig",
    "tdadjorigb",
    "trendadjls",
    "trendb2",
    "trendb7",
    "trendc2",
    "trendc7",
    "trendconst",
    "trendd2",
    "trendd7",
    "irregularplot",
    "origwsaplot",
    "ratioplotorig",
    "ratioplotsa",
    "seasadjplot",
    "seasonalplot",
    "trendplot",
]
_X11_SAVE_ALL: Final[list[str]] = [
    "adjustdiff",
    "adjustfac",
    "adjustmentratio",
    "calendar",
    "calendaradjchanges",
    "combholiday",
    "irregular",
    "irrwt",
    "origchanges",
    "replacsi",
    "sachanges",
    "seasadj",
    "seasonal",
    "seasonaldiff",
    "totaladjustment",
    "trend",
    "trendchanges",
    "unmodsi",
    "unmodsiox",
    "adjoriginalc",
    "adjoriginald",
    "extreme",
    "extremeb",
    "irregularadjao",
    "irregularb",
    "irregularc",
    "irrwtb",
    "mcdmovavg",
    "modirregular",
    "modoriginal",
    "modseasadj",
    "modsic4",
    "modsid4",
    "replacsic9",
    "robustsa",
    "seasadjb11",
    "seasadjb6",
    "seasadjc11",
    "seasadjc6",
    "seasadjconst",
    "seasadjd6",
    "seasonalb10",
    "seasonalb5",
    "seasonalc10",
    "seasonalc5",
    "seasonald5",
    "sib3",
    "sib8",
    "tdadjorig",
    "tdadjorigb",
    "trendadjls",
    "trendb2",
    "trendb7",
    "trendc2",
    "trendc7",
    "trendconst",
    "trendd2",
    "trendd7",
    "adjustfacpct",
    "calendaradjchangespct",
    "irregularpct",
    "origchangespct",
    "sachangespct",
    "seasonalpct",
    "trendchangespct",
]


def x11(
    *,
    appendbcst: bool | X13default = _X13DEFAULT,
    appendfcst: bool | X13default = _X13DEFAULT,
    final: str | list[str] | X13default = _X13DEFAULT,
    mode: str | X13default = _X13DEFAULT,
    print: str | list[str] | X13default = _X13DEFAULT,
    save: str | list[str] | X13default = _X13DEFAULT,
    savelog: str | list[str] | X13default | None = None,
    seasonalma: str | list[str] | X13default = _X13DEFAULT,
    sigmalim: list[_FloatOrNone] | list[float] | X13default = _X13DEFAULT,
    title: str | list[str] | X13default = _X13DEFAULT,
    trendma: int | X13default = _X13DEFAULT,
    type: str | X13default = _X13DEFAULT,
    calendarsigma: str | X13default = _X13DEFAULT,
    centerseasonal: bool | X13default = _X13DEFAULT,
    keepholiday: bool | X13default = _X13DEFAULT,
    print1stpass: bool | X13default = _X13DEFAULT,
    sfshort: bool | X13default = _X13DEFAULT,
    sigmavec: list[str] | X13default = _X13DEFAULT,
    trendic: float | X13default = _X13DEFAULT,
    true7term: bool | X13default = _X13DEFAULT,
) -> X13x11:
    """Build the ``x11`` spec — X-11 enhanced seasonal-adjustment options.

    Mirrors ``x13spec.jl:3180-3228``. The upstream's ``savelog`` default
    is ``"alldiagnostics"``; the Python port uses :data:`None` as the
    sentinel that triggers that default (per the established pattern in
    :func:`automdl` / :func:`regression` etc.).

    Validation:

    * ``trendma`` must be an odd integer in ``[3, 101]``.
    * ``sigmavec`` requires ``calendarsigma="select"`` (other values
      raise :exc:`ValueError`).
    """
    if savelog is None:
        savelog = "alldiagnostics"

    if not isinstance(trendma, X13default) and (trendma % 2 != 1 or trendma < 3 or trendma > 101):
        msg = f"trendma must be an odd number between 3 and 101. Received: {trendma}."
        raise ValueError(msg)

    if not isinstance(sigmavec, X13default) and (
        isinstance(calendarsigma, X13default)
        or (isinstance(calendarsigma, str) and calendarsigma != "select")
    ):
        msg = "The sigmavec argument can only be specified when calendarsigma='select'."
        raise ValueError(msg)

    print = _expand_all(print, _X11_PRINT_ALL)
    save = _expand_all(save, _X11_SAVE_ALL)

    return X13x11(
        appendbcst=appendbcst,
        appendfcst=appendfcst,
        final=final,
        mode=mode,
        print=print,
        save=save,
        savelog=savelog,
        seasonalma=seasonalma,
        sigmalim=sigmalim,
        title=title,
        trendma=trendma,
        type=type,
        calendarsigma=calendarsigma,
        centerseasonal=centerseasonal,
        keepholiday=keepholiday,
        print1stpass=print1stpass,
        sfshort=sfshort,
        sigmavec=sigmavec,
        trendic=trendic,
        true7term=true7term,
    )


# ===========================================================================
# M2.3 — Rare spec builders
# ===========================================================================
#
# Eleven builders that round out X-13 spec parity. Same shape pattern as
# M2.2: frozen-slotted container dataclass + builder function with kwarg
# validation. ``.spc`` text emission deferred to M2.4 alongside
# ``_write.py``. See ``x13spec.jl`` for the upstream signatures.
#
# Order matches the alphabetical Julia ordering: check, estimate, force,
# history, identify, metadata, outlier, pickmdl, slidingspans, spectrum,
# x11regression.


# ---------------------------------------------------------------------------
# X13check container + check() builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class X13check:
    """The ``check`` spec block — regARIMA residual diagnostics.

    Mirrors ``x13spec.jl:274-282``. Carries ACF / PACF lag limits and
    significance thresholds used for residual-adequacy reporting.
    """

    maxlag: int | X13default
    qtype: str | X13default
    print: str | list[str] | X13default
    save: str | list[str] | X13default
    savelog: str | list[str] | X13default
    acflimit: float | X13default
    qlimit: float | X13default


_CHECK_PRINT_ALL: Final[list[str]] = [
    "acf",
    "acfplot",
    "pacf",
    "pacfplot",
    "acfsquared",
    "acfsquaredplot",
    "normalitytest",
    "durbinwatson",
    "friedmantest",
    "histogram",
]
_CHECK_SAVE_ALL: Final[list[str]] = ["acf", "pacf", "acfsquared"]


def check(
    *,
    maxlag: int | X13default = _X13DEFAULT,
    qtype: str | X13default = _X13DEFAULT,
    print: str | list[str] | X13default = _X13DEFAULT,
    save: str | list[str] | X13default = _X13DEFAULT,
    savelog: str | list[str] | X13default | None = None,
    acflimit: float | X13default = _X13DEFAULT,
    qlimit: float | X13default = _X13DEFAULT,
) -> X13check:
    """Build the ``check`` spec — residual ACF / Ljung-Box diagnostics.

    Mirrors ``x13spec.jl:1149-1169``. Expands ``print="all"`` /
    ``save="all"`` to the upstream-defined "everything" lists; otherwise
    pass-through. No numeric range validation in the upstream — :exc:`ValueError`
    surfaces from the writer if X-13 rejects the values.
    """
    if savelog is None:
        savelog = "alldiagnostics"
    print = _expand_all(print, _CHECK_PRINT_ALL)
    save = _expand_all(save, _CHECK_SAVE_ALL)
    return X13check(
        maxlag=maxlag,
        qtype=qtype,
        print=print,
        save=save,
        savelog=savelog,
        acflimit=acflimit,
        qlimit=qlimit,
    )


# ---------------------------------------------------------------------------
# X13estimate container + estimate() builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class X13estimate:
    """The ``estimate`` spec block — regARIMA estimation controls.

    Mirrors ``x13spec.jl:284-294``. Selects exact-vs-conditional likelihood,
    iteration limits, and optional preloaded ``.mdl`` file with fix-policy.
    """

    exact: str | X13default
    maxiter: int | X13default
    outofsample: bool | X13default
    print: str | list[str] | X13default
    save: str | list[str] | X13default
    savelog: str | list[str] | X13default
    tol: float | X13default
    file: str | X13default
    fix: str | X13default


_ESTIMATE_PRINT_ALL: Final[list[str]] = [
    "options",
    "model",
    "estimates",
    "averagefcsterr",
    "lkstats",
    "iterations",
    "iterationerrors",
    "regcmatrix",
    "armacmatrix",
    "lformulas",
    "roots",
    "regressioneffects",
    "regressionresiduals",
    "residuals",
]
_ESTIMATE_SAVE_ALL: Final[list[str]] = [
    "model",
    "estimates",
    "lkstats",
    "iterations",
    "regcmatrix",
    "armacmatrix",
    "roots",
    "regressioneffects",
    "regressionresiduals",
    "residuals",
]


def estimate(
    *,
    exact: str | X13default = _X13DEFAULT,
    maxiter: int | X13default = _X13DEFAULT,
    outofsample: bool | X13default = _X13DEFAULT,
    print: str | list[str] | X13default = _X13DEFAULT,
    save: str | list[str] | X13default = _X13DEFAULT,
    savelog: str | list[str] | X13default | None = None,
    tol: float | X13default = _X13DEFAULT,
    file: str | X13default = _X13DEFAULT,
    fix: str | X13default = _X13DEFAULT,
) -> X13estimate:
    """Build the ``estimate`` spec — regARIMA estimation options.

    Mirrors ``x13spec.jl:1228-1251``. Expands ``print="all"`` /
    ``save="all"`` to the upstream-defined "everything" lists.
    """
    if savelog is None:
        savelog = "alldiagnostics"
    print = _expand_all(print, _ESTIMATE_PRINT_ALL)
    save = _expand_all(save, _ESTIMATE_SAVE_ALL)
    return X13estimate(
        exact=exact,
        maxiter=maxiter,
        outofsample=outofsample,
        print=print,
        save=save,
        savelog=savelog,
        tol=tol,
        file=file,
        fix=fix,
    )


# ---------------------------------------------------------------------------
# X13force container + force() builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class X13force:
    """The ``force`` spec block — yearly-total forcing on seasonally adjusted output.

    Mirrors ``x13spec.jl:296-308``. Activates Denton or regression-based
    benchmarking to make the SA series' yearly totals match the target.
    """

    lambda_: float | X13default
    mode: str | X13default
    print: str | list[str] | X13default
    save: str | list[str] | X13default
    rho: float | X13default
    round: bool | X13default
    start: str | X13default
    target: str | X13default
    type: str | X13default
    usefcst: bool | X13default
    indforce: bool | X13default


_FORCE_PRINT_ALL: Final[list[str]] = [
    "seasadjtot",
    "saround",
    "revsachanges",
    "rndsachanges",
]
_FORCE_SAVE_ALL: Final[list[str]] = [
    "seasadjtot",
    "saround",
    "revsachanges",
    "rndsachanges",
    "revsachangespct",
    "rndsachangespct",
]


def force(
    *,
    lambda_: float | X13default = _X13DEFAULT,
    mode: str | X13default = _X13DEFAULT,
    print: str | list[str] | X13default = _X13DEFAULT,
    save: str | list[str] | X13default = _X13DEFAULT,
    rho: float | X13default = _X13DEFAULT,
    round: bool | X13default = _X13DEFAULT,
    start: str | X13default = _X13DEFAULT,
    target: str | X13default = _X13DEFAULT,
    type: str | X13default = _X13DEFAULT,
    usefcst: bool | X13default = _X13DEFAULT,
    indforce: bool | X13default = _X13DEFAULT,
) -> X13force:
    """Build the ``force`` spec — yearly-total forcing options.

    Mirrors ``x13spec.jl:1343-1372``. The Julia ``lambda`` keyword renames
    to ``lambda_`` (Python's reserved word); ``round`` and ``type`` keep
    their X-13 names (not Python reserved words, only built-in shadows).

    Validation:

    * ``rho`` must be in ``[0.0, 1.0]``.
    """
    if not isinstance(rho, X13default) and (rho < 0.0 or rho > 1.0):
        msg = f"rho must be between 0 and 1. Received: {rho}."
        raise ValueError(msg)
    print = _expand_all(print, _FORCE_PRINT_ALL)
    save = _expand_all(save, _FORCE_SAVE_ALL)
    return X13force(
        lambda_=lambda_,
        mode=mode,
        print=print,
        save=save,
        rho=rho,
        round=round,
        start=start,
        target=target,
        type=type,
        usefcst=usefcst,
        indforce=indforce,
    )


# ---------------------------------------------------------------------------
# X13history container + history() builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class X13history:
    """The ``history`` spec block — revisions / forecast-error history analysis.

    Mirrors ``x13spec.jl:321-340``. Drives the truncated-series re-run
    pipeline used to characterise revisions and out-of-sample forecast
    errors at user-specified lags.
    """

    endtable: MIT | X13default
    estimates: str | list[str] | X13default
    fixmdl: bool | X13default
    fixreg: bool | X13default
    fstep: int | list[int] | X13default
    print: str | list[str] | X13default
    save: str | list[str] | X13default
    savelog: str | list[str] | X13default
    sadjlags: int | list[int] | X13default
    start: MIT | X13default
    target: str | X13default
    trendlags: int | list[int] | X13default
    fixx11reg: bool | X13default
    outlier: str | X13default
    outlierwin: int | X13default
    refresh: bool | X13default
    transformfcst: bool | X13default
    x11outlier: bool | X13default


_HISTORY_PRINT_ALL: Final[list[str]] = [
    "header",
    "outlierhistory",
    "sarevisions",
    "sasummary",
    "chngrevisions",
    "chngsummary",
    "indsarevisions",
    "indsasummary",
    "trendrevisions",
    "trendsummary",
    "trendchngrevisions",
    "trendchngsummary",
    "sfrevisions",
    "sfsummary",
    "lkhdhistory",
    "fcsterrors",
    "armahistory",
    "tdhistory",
    "sfilterhistory",
    "saestimates",
    "chngestimates",
    "indsaestimates",
    "trendestimates",
    "trendchngestimates",
    "sfestimates",
    "fcsthistory",
]
_HISTORY_SAVE_ALL: Final[list[str]] = [
    "outlierhistory",
    "sarevisions",
    "chngrevisions",
    "indsarevisions",
    "trendrevisions",
    "trendchngrevisions",
    "sfrevisions",
    "lkhdhistory",
    "fcsterrors",
    "armahistory",
    "tdhistory",
    "sfilterhistory",
    "saestimates",
    "chngestimates",
    "indsaestimates",
    "trendestimates",
    "trendchngestimates",
    "sfestimates",
    "fcsthistory",
]

_HISTORY_FSTEP_MAX_LENGTH: Final[int] = 4
_HISTORY_SADJLAGS_MAX_LENGTH: Final[int] = 5


def history(
    *,
    endtable: MIT | X13default = _X13DEFAULT,
    estimates: str | list[str] | X13default = _X13DEFAULT,
    fixmdl: bool | X13default = _X13DEFAULT,
    fixreg: bool | X13default = _X13DEFAULT,
    fstep: int | list[int] | X13default = _X13DEFAULT,
    print: str | list[str] | X13default = _X13DEFAULT,
    save: str | list[str] | X13default = _X13DEFAULT,
    savelog: str | list[str] | X13default | None = None,
    sadjlags: int | list[int] | X13default = _X13DEFAULT,
    start: MIT | X13default = _X13DEFAULT,
    target: str | X13default = _X13DEFAULT,
    trendlags: int | list[int] | X13default = _X13DEFAULT,
    fixx11reg: bool | X13default = _X13DEFAULT,
    outlier: str | X13default = _X13DEFAULT,
    outlierwin: int | X13default = _X13DEFAULT,
    refresh: bool | X13default = _X13DEFAULT,
    transformfcst: bool | X13default = _X13DEFAULT,
    x11outlier: bool | X13default = _X13DEFAULT,
) -> X13history:
    """Build the ``history`` spec — truncated-series revisions analysis.

    Mirrors ``x13spec.jl:1602-1656``.

    Validation:

    * ``fstep`` (when a list) must have length ≤ 4 with every entry ≥ 1.
      Scalar ``fstep`` must be ≥ 1.
    * ``sadjlags`` (when a list) must have length ≤ 5 with every entry
      ≥ 1. Scalar ``sadjlags`` must be ≥ 1.
    """
    if savelog is None:
        savelog = ["alldiagnostics"]

    if isinstance(fstep, list):
        if len(fstep) > _HISTORY_FSTEP_MAX_LENGTH:
            msg = f"fstep can contain up to four forecast leads. Received: {fstep}."
            raise ValueError(msg)
        if any(v < 1 for v in fstep):
            msg = f"fstep values cannot be less than one. Received: {fstep}."
            raise ValueError(msg)
    elif isinstance(fstep, int) and not isinstance(fstep, bool) and fstep < 1:
        msg = f"fstep cannot be less than one. Received: {fstep}."
        raise ValueError(msg)

    if isinstance(sadjlags, list):
        if len(sadjlags) > _HISTORY_SADJLAGS_MAX_LENGTH:
            msg = f"sadjlags can contain up to five revision lags. Received: {sadjlags}."
            raise ValueError(msg)
        if any(v < 1 for v in sadjlags):
            msg = f"sadjlags values cannot be less than one. Received: {sadjlags}."
            raise ValueError(msg)
    elif isinstance(sadjlags, int) and not isinstance(sadjlags, bool) and sadjlags < 1:
        msg = f"sadjlags cannot be less than one. Received: {sadjlags}."
        raise ValueError(msg)

    print = _expand_all(print, _HISTORY_PRINT_ALL)
    save = _expand_all(save, _HISTORY_SAVE_ALL)
    return X13history(
        endtable=endtable,
        estimates=estimates,
        fixmdl=fixmdl,
        fixreg=fixreg,
        fstep=fstep,
        print=print,
        save=save,
        savelog=savelog,
        sadjlags=sadjlags,
        start=start,
        target=target,
        trendlags=trendlags,
        fixx11reg=fixx11reg,
        outlier=outlier,
        outlierwin=outlierwin,
        refresh=refresh,
        transformfcst=transformfcst,
        x11outlier=x11outlier,
    )


# ---------------------------------------------------------------------------
# X13identify container + identify() builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class X13identify:
    """The ``identify`` spec block — ACF / PACF plots for ARIMA identification.

    Mirrors ``x13spec.jl:342-348``. Sample ACFs and PACFs are produced
    for every combination of nonseasonal-difference orders (``diff``) and
    seasonal-difference orders (``sdiff``).
    """

    diff: list[int] | X13default
    sdiff: list[int] | X13default
    maxlag: int | X13default
    print: str | list[str] | X13default
    save: str | list[str] | X13default


_IDENTIFY_PRINT_ALL: Final[list[str]] = [
    "acf",
    "acfplot",
    "pacf",
    "pacfplot",
    "regcoefficients",
]
_IDENTIFY_SAVE_ALL: Final[list[str]] = ["acf", "pacf"]


def identify(
    *,
    diff: list[int] | X13default = _X13DEFAULT,
    sdiff: list[int] | X13default = _X13DEFAULT,
    maxlag: int | X13default = _X13DEFAULT,
    print: str | list[str] | X13default = _X13DEFAULT,
    save: str | list[str] | X13default = _X13DEFAULT,
) -> X13identify:
    """Build the ``identify`` spec — ARIMA-identification ACFs/PACFs.

    Mirrors ``x13spec.jl:1686-1706``. No range validation upstream — the
    binary applies its own bounds.
    """
    print = _expand_all(print, _IDENTIFY_PRINT_ALL)
    save = _expand_all(save, _IDENTIFY_SAVE_ALL)
    return X13identify(
        diff=diff,
        sdiff=sdiff,
        maxlag=maxlag,
        print=print,
        save=save,
    )


# ---------------------------------------------------------------------------
# X13metadata container + metadata() builder
# ---------------------------------------------------------------------------


_METADATA_MAX_ENTRIES: Final[int] = 20
_METADATA_MAX_KEY_OR_VALUE_LENGTH: Final[int] = 132
_METADATA_MAX_TOTAL_LENGTH: Final[int] = 2000


@dataclass(frozen=True, slots=True)
class X13metadata:
    """The ``metadata`` spec block — diagnostic-summary key/value entries.

    Mirrors ``x13spec.jl:350-352``. Stores a tuple of ``(key, value)``
    pairs that X-13 emits into the ``.udg`` diagnostic-summary file.

    The Julia upstream's ``Pair{String,String}`` / ``Vector{Pair{...}}``
    ports as a ``tuple[tuple[str, str], ...]`` (frozen, order-preserving,
    pickleable) — see :func:`metadata` for the user-facing input shapes.
    """

    entries: tuple[tuple[str, str], ...]


def metadata(
    entries: tuple[str, str] | list[tuple[str, str]] | tuple[tuple[str, str], ...] | dict[str, str],
) -> X13metadata:
    """Build the ``metadata`` spec — diagnostic-summary key/value entries.

    Mirrors ``x13spec.jl:1721-1748``. Accepts a single ``(key, value)``
    pair, an iterable of pairs, or a ``dict`` (the dict form is the
    Python-idiomatic shape; iteration order preserved per PEP 468).

    Validation (mirrors upstream):

    * At most 20 entries.
    * No single key or value exceeds 132 characters.
    * Concatenated keys and concatenated values each ≤ 2000 characters.
    """
    if isinstance(entries, dict):
        items: tuple[tuple[str, str], ...] = tuple(entries.items())
    elif (
        isinstance(entries, tuple)
        and len(entries) == 2
        and isinstance(entries[0], str)
        and isinstance(entries[1], str)
    ):
        items = (entries,)
    else:
        items = tuple((k, v) for k, v in entries)

    if len(items) > _METADATA_MAX_ENTRIES:
        msg = (
            f"A maximum of {_METADATA_MAX_ENTRIES} metadata entries can be "
            f"specified. Received: {len(items)} entries."
        )
        raise ValueError(msg)

    keys = [k for k, _ in items]
    values = [v for _, v in items]
    if any(len(k) > _METADATA_MAX_KEY_OR_VALUE_LENGTH for k in keys):
        msg = (
            f"Keys in the metadata spec can have a maximum length of "
            f"{_METADATA_MAX_KEY_OR_VALUE_LENGTH} characters."
        )
        raise ValueError(msg)
    total_keys = sum(len(k) for k in keys)
    if total_keys > _METADATA_MAX_TOTAL_LENGTH:
        msg = (
            f"Keys in the metadata spec can have a maximum combined "
            f"length of {_METADATA_MAX_TOTAL_LENGTH} characters. "
            f"Received: {total_keys} characters."
        )
        raise ValueError(msg)
    if any(len(v) > _METADATA_MAX_KEY_OR_VALUE_LENGTH for v in values):
        msg = (
            f"Values in the metadata spec can have a maximum length of "
            f"{_METADATA_MAX_KEY_OR_VALUE_LENGTH} characters."
        )
        raise ValueError(msg)
    total_values = sum(len(v) for v in values)
    if total_values > _METADATA_MAX_TOTAL_LENGTH:
        msg = (
            f"Values in the metadata spec can have a maximum combined "
            f"length of {_METADATA_MAX_TOTAL_LENGTH} characters. "
            f"Received: {total_values} characters."
        )
        raise ValueError(msg)

    return X13metadata(entries=items)


# ---------------------------------------------------------------------------
# X13outlier container + outlier() builder
# ---------------------------------------------------------------------------


_OUTLIER_LSRUN_MAX: Final[int] = 5
_OUTLIER_CRITICAL_MAX_LENGTH: Final[int] = 3


@dataclass(frozen=True, slots=True)
class X13outlier:
    """The ``outlier`` spec block — automatic outlier identification.

    Mirrors ``x13spec.jl:354-365``. Triggers the iterative add-detection
    loop for additive outliers, level shifts, and temporary changes.
    """

    critical: float | list[float | None] | list[float] | X13default
    lsrun: int | X13default
    method: str | X13default
    print: str | list[str] | X13default
    save: str | list[str] | X13default
    savelog: str | list[str] | X13default
    span: MITRange | Span | X13default
    types: str | list[str] | X13default
    almost: float | X13default
    tcrate: float | X13default


_OUTLIER_PRINT_ALL: Final[list[str]] = [
    "header",
    "iterations",
    "tests",
    "temporaryls",
    "finaltests",
]
_OUTLIER_SAVE_ALL: Final[list[str]] = ["iterations", "finaltests"]


def outlier(
    *,
    critical: float | list[float | None] | list[float] | X13default = _X13DEFAULT,
    lsrun: int | X13default = _X13DEFAULT,
    method: str | X13default = _X13DEFAULT,
    print: str | list[str] | X13default = _X13DEFAULT,
    save: str | list[str] | X13default = _X13DEFAULT,
    savelog: str | list[str] | X13default | None = None,
    span: MITRange | Span | X13default = _X13DEFAULT,
    types: str | list[str] | X13default = _X13DEFAULT,
    almost: float | X13default = _X13DEFAULT,
    tcrate: float | X13default = _X13DEFAULT,
) -> X13outlier:
    """Build the ``outlier`` spec — automatic outlier detection.

    Mirrors ``x13spec.jl:1833-1881``.

    Validation:

    * ``critical`` (when a list) must have length ≤ 3.
    * ``lsrun`` must be in ``[0, 5]``.
    * ``almost`` must be > 0.
    * ``tcrate`` must be in ``(0.0, 1.0)``.
    * :class:`Span` ``span`` rejects fuzzy ``e`` (``M11`` / ``Q2``) —
      mirrored as "endpoints must be MIT or None"; the builder accepts
      only :class:`MIT` or :data:`None` already, so the upstream check
      collapses to a no-op here.
    """
    if savelog is None:
        savelog = "identified"

    if isinstance(critical, list) and len(critical) > _OUTLIER_CRITICAL_MAX_LENGTH:
        msg = (
            f"critical can contain up to {_OUTLIER_CRITICAL_MAX_LENGTH} "
            f"values. Received: {critical}."
        )
        raise ValueError(msg)

    if not isinstance(lsrun, X13default) and (lsrun < 0 or lsrun > _OUTLIER_LSRUN_MAX):
        msg = f"lsrun can take values from 0 to {_OUTLIER_LSRUN_MAX}. Received: {lsrun}."
        raise ValueError(msg)

    if not isinstance(almost, X13default) and almost < 0.0:
        msg = f"almost must have a value greater than zero. Received: {almost}."
        raise ValueError(msg)

    if not isinstance(tcrate, X13default) and (tcrate <= 0.0 or tcrate >= 1.0):
        msg = f"tcrate must be a number greater than zero and less than one. Received: {tcrate}."
        raise ValueError(msg)

    print = _expand_all(print, _OUTLIER_PRINT_ALL)
    save = _expand_all(save, _OUTLIER_SAVE_ALL)
    return X13outlier(
        critical=critical,
        lsrun=lsrun,
        method=method,
        print=print,
        save=save,
        savelog=savelog,
        span=span,
        types=types,
        almost=almost,
        tcrate=tcrate,
    )


# ---------------------------------------------------------------------------
# X13pickmdl container + pickmdl() builder
# ---------------------------------------------------------------------------


_PICKMDL_MIN_MODELS: Final[int] = 2


@dataclass(frozen=True, slots=True)
class X13pickmdl:
    """The ``pickmdl`` spec block — X-11-ARIMA model selection from candidates.

    Mirrors ``x13spec.jl:367-380``. Picks the ARIMA part from either a
    list of candidate :class:`ArimaModel` instances or a file containing
    such a list.
    """

    bcstlim: int | X13default
    fcstlim: int | X13default
    models: list[ArimaModel] | X13default
    identify: str | X13default
    method: str | X13default
    mode: str | X13default
    outofsample: bool | X13default
    overdiff: float | X13default
    print: str | list[str] | X13default
    savelog: str | list[str] | X13default
    qlim: int | X13default
    file: str | X13default


_PICKMDL_PRINT_ALL: Final[list[str]] = [
    "pickmdlchoice",
    "header",
    "usermodels",
]

_PICKMDL_PERCENT_MAX: Final[int] = 100
_PICKMDL_OVERDIFF_MIN: Final[float] = 0.9
_PICKMDL_OVERDIFF_MAX: Final[float] = 1.0


def pickmdl(
    *models: ArimaModel,
    bcstlim: int | X13default = _X13DEFAULT,
    fcstlim: int | X13default = _X13DEFAULT,
    identify: str | X13default = _X13DEFAULT,
    method: str | X13default = _X13DEFAULT,
    mode: str | X13default = _X13DEFAULT,
    outofsample: bool | X13default = _X13DEFAULT,
    overdiff: float | X13default = _X13DEFAULT,
    print: str | list[str] | X13default = _X13DEFAULT,
    savelog: str | list[str] | X13default | None = None,
    qlim: int | X13default = _X13DEFAULT,
    file: str | X13default = _X13DEFAULT,
) -> X13pickmdl:
    """Build the ``pickmdl`` spec — automatic ARIMA model selection.

    Mirrors ``x13spec.jl:1972-2030``. The Julia ``pickmdl(models::Vector{ArimaModel})``
    and ``pickmdl(models::ArimaModel...)`` overloads collapse to a single
    Python signature with a positional ``*models`` varargs — pass either
    ``pickmdl(m1, m2)`` or ``pickmdl(*[m1, m2])``.

    The ``identify`` kwarg shadows the :func:`identify` spec builder at
    module level; this is the X-13 grammar's name (``identify=:first`` /
    ``:all``) and the per-kwarg ``A002`` ignore preserves the surface.

    Validation:

    * Either ``models`` (≥ 2 candidates) OR ``file=`` must be supplied.
    * At most one candidate may have ``default=True``.
    * ``bcstlim`` / ``fcstlim`` / ``qlim`` ∈ ``[0, 100]``.
    * ``overdiff`` ∈ ``[0.9, 1.0]``.
    """
    if savelog is None:
        savelog = "automodel"

    if not isinstance(bcstlim, X13default) and (bcstlim < 0 or bcstlim > _PICKMDL_PERCENT_MAX):
        msg = (
            f"bcstlim must be a value between 0 and {_PICKMDL_PERCENT_MAX} "
            f"(inclusive). Received: {bcstlim}."
        )
        raise ValueError(msg)
    if not isinstance(fcstlim, X13default) and (fcstlim < 0 or fcstlim > _PICKMDL_PERCENT_MAX):
        msg = (
            f"fcstlim must be a value between 0 and {_PICKMDL_PERCENT_MAX} "
            f"(inclusive). Received: {fcstlim}."
        )
        raise ValueError(msg)
    if not isinstance(qlim, X13default) and (qlim < 0 or qlim > _PICKMDL_PERCENT_MAX):
        msg = (
            f"qlim must be a value between 0 and {_PICKMDL_PERCENT_MAX} "
            f"(inclusive). Received: {qlim}."
        )
        raise ValueError(msg)

    if not isinstance(overdiff, X13default):
        if overdiff > _PICKMDL_OVERDIFF_MAX:
            msg = f"overdiff must not be greater than 1. Received: {overdiff}."
            raise ValueError(msg)
        if overdiff < _PICKMDL_OVERDIFF_MIN:
            msg = f"overdiff should not be less than {_PICKMDL_OVERDIFF_MIN}. Received: {overdiff}."
            raise ValueError(msg)

    models_value: list[ArimaModel] | X13default
    if models:
        if len(models) < _PICKMDL_MIN_MODELS:
            msg = (
                f"pickmdl spec must be provided with at least "
                f"{_PICKMDL_MIN_MODELS} candidate models. "
                f"Received: {len(models)}. {list(models)}"
            )
            raise ValueError(msg)
        num_defaults = sum(1 for m in models if m.default)
        if num_defaults > 1:
            msg = (
                f"pickmdl can only have one model specified as a default, "
                f"but {num_defaults} of the provided models are flagged "
                f"as defaults."
            )
            raise ValueError(msg)
        models_value = list(models)
    else:
        models_value = _X13DEFAULT
        if isinstance(file, X13default):
            msg = (
                "pickmdl spec must either be constructed with one or more "
                "ArimaModels or with the file keyword argument specified."
            )
            raise ValueError(msg)

    print = _expand_all(print, _PICKMDL_PRINT_ALL)
    return X13pickmdl(
        bcstlim=bcstlim,
        fcstlim=fcstlim,
        models=models_value,
        identify=identify,
        method=method,
        mode=mode,
        outofsample=outofsample,
        overdiff=overdiff,
        print=print,
        savelog=savelog,
        qlim=qlim,
        file=file,
    )


# ---------------------------------------------------------------------------
# X13slidingspans container + slidingspans() builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class X13slidingspans:
    """The ``slidingspans`` spec block — stability analysis over moving spans.

    Mirrors ``x13spec.jl:432-448``. Compares seasonal-adjustment output
    across overlapping subspans of the series for revisions stability.
    """

    cutchng: float | X13default
    cutseas: float | X13default
    cuttd: float | X13default
    fixmdl: bool | str | X13default
    fixreg: list[str] | X13default
    length: int | X13default
    numspans: int | X13default
    outlier: str | X13default
    print: str | list[str] | X13default
    save: str | list[str] | X13default
    savelog: str | list[str] | X13default
    start: MIT | X13default
    additivesa: str | X13default
    fixx11reg: bool | X13default
    x11outlier: bool | X13default


_SLIDINGSPANS_PRINT_ALL: Final[list[str]] = [
    "header",
    "ssftest",
    "factormeans",
    "percent",
    "summary",
    "yysummary",
    "indfactormeans",
    "indpercent",
    "indsummary",
    "yypercent",
    "sfspans",
    "chngspans",
    "saspans",
    "ychngspans",
    "tdspans",
    "indyypercent",
    "indyysummary",
    "indsfspans",
    "indchngspans",
    "indsaspans",
    "indychngspans",
]
_SLIDINGSPANS_SAVE_ALL: Final[list[str]] = [
    "sfspans",
    "chngspans",
    "saspans",
    "ychngspans",
    "tdspans",
    "indsfspans",
    "indchngspans",
    "indsaspans",
    "indychngspans",
]


def slidingspans(
    *,
    cutchng: float | X13default = _X13DEFAULT,
    cutseas: float | X13default = _X13DEFAULT,
    cuttd: float | X13default = _X13DEFAULT,
    fixmdl: bool | str | X13default = _X13DEFAULT,
    fixreg: list[str] | X13default = _X13DEFAULT,
    length: int | X13default = _X13DEFAULT,
    numspans: int | X13default = _X13DEFAULT,
    outlier: str | X13default = _X13DEFAULT,
    print: str | list[str] | X13default = _X13DEFAULT,
    save: str | list[str] | X13default = _X13DEFAULT,
    savelog: str | list[str] | X13default | None = None,
    start: MIT | X13default = _X13DEFAULT,
    additivesa: str | X13default = _X13DEFAULT,
    fixx11reg: bool | X13default = _X13DEFAULT,
    x11outlier: bool | X13default = _X13DEFAULT,
) -> X13slidingspans:
    """Build the ``slidingspans`` spec — stability analysis options.

    Mirrors ``x13spec.jl:2666-2701``. The Julia ``length`` field collides
    with Python's :func:`len` built-in but is not a Python reserved word,
    so the kwarg keeps its X-13 name with a per-line ``A002`` ignore.

    Validation:

    * Setting both ``fixmdl=True`` and ``fixreg=[...]`` warns
      (``fixreg`` is ignored by X-13 in that combination, mirroring the
      upstream ``@warn``).
    """
    if savelog is None:
        savelog = "percents"

    if not isinstance(fixmdl, X13default) and not isinstance(fixreg, X13default) and fixmdl is True:
        warnings.warn(
            "fixreg will be ignored because fixmdl is set to true.",
            UserWarning,
            stacklevel=2,
        )

    print = _expand_all(print, _SLIDINGSPANS_PRINT_ALL)
    save = _expand_all(save, _SLIDINGSPANS_SAVE_ALL)
    return X13slidingspans(
        cutchng=cutchng,
        cutseas=cutseas,
        cuttd=cuttd,
        fixmdl=fixmdl,
        fixreg=fixreg,
        length=length,
        numspans=numspans,
        outlier=outlier,
        print=print,
        save=save,
        savelog=savelog,
        start=start,
        additivesa=additivesa,
        fixx11reg=fixx11reg,
        x11outlier=x11outlier,
    )


# ---------------------------------------------------------------------------
# X13spectrum container + spectrum() builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class X13spectrum:
    """The ``spectrum`` spec block — frequency-domain seasonality diagnostics.

    Mirrors ``x13spec.jl:450-465``. Computes AR-spectrum or periodogram
    estimates plus the QS statistic for both monthly and quarterly series.
    """

    logqs: bool | X13default
    print: str | list[str] | X13default
    save: str | list[str] | X13default
    savelog: str | list[str] | X13default
    qcheck: bool | X13default
    start: MIT | X13default
    tukey120: bool | X13default
    decibel: bool | X13default
    difference: bool | str | X13default
    maxar: int | X13default
    peakwidth: int | X13default
    series: str | X13default
    siglevel: int | X13default
    type: str | X13default


_SPECTRUM_PRINT_ALL: Final[list[str]] = [
    "qcheck",
    "qs",
    "specorig",
    "specsa",
    "specirr",
    "specseatssa",
    "specseatsirr",
    "specextresiduals",
    "specresidual",
    "speccomposite",
    "specindirr",
    "specindsa",
    "tukeypeaks",
]
_SPECTRUM_SAVE_ALL: Final[list[str]] = [
    "specorig",
    "specsa",
    "specirr",
    "specseatssa",
    "specseatsirr",
    "specextresiduals",
    "specresidual",
    "speccomposite",
    "specindirr",
    "specindsa",
]


def spectrum(
    *,
    logqs: bool | X13default = _X13DEFAULT,
    print: str | list[str] | X13default = _X13DEFAULT,
    save: str | list[str] | X13default = _X13DEFAULT,
    savelog: str | list[str] | X13default | None = None,
    qcheck: bool | X13default = _X13DEFAULT,
    start: MIT | X13default = _X13DEFAULT,
    tukey120: bool | X13default = _X13DEFAULT,
    decibel: bool | X13default = _X13DEFAULT,
    difference: bool | str | X13default = _X13DEFAULT,
    maxar: int | X13default = _X13DEFAULT,
    peakwidth: int | X13default = _X13DEFAULT,
    series: str | X13default = _X13DEFAULT,
    siglevel: int | X13default = _X13DEFAULT,
    type: str | X13default = _X13DEFAULT,
) -> X13spectrum:
    """Build the ``spectrum`` spec — spectral-diagnostic options.

    Mirrors ``x13spec.jl:2785-2816``. No numeric validation upstream;
    range checks happen in the X-13 binary.

    The ``series`` kwarg shadows the :func:`series` spec builder at
    module level; this is the X-13 grammar's name (``series=:original``
    selects which series feeds the spectrum) and the per-kwarg ``A002``
    ignore preserves the surface.
    """
    if savelog is None:
        savelog = "alldiagnostics"
    print = _expand_all(print, _SPECTRUM_PRINT_ALL)
    save = _expand_all(save, _SPECTRUM_SAVE_ALL)
    return X13spectrum(
        logqs=logqs,
        print=print,
        save=save,
        savelog=savelog,
        qcheck=qcheck,
        start=start,
        tukey120=tukey120,
        decibel=decibel,
        difference=difference,
        maxar=maxar,
        peakwidth=peakwidth,
        series=series,
        siglevel=siglevel,
        type=type,
    )


# ---------------------------------------------------------------------------
# X13x11regression container + x11regression() builder
# ---------------------------------------------------------------------------


_X11REGRESSION_AICTEST_ALLOWED: Final[frozenset[str]] = frozenset(
    {"td", "tdstock", "td1coef", "tdstock1coef", "easter", "user"}
)
_X11REGRESSION_USERTYPE_ALLOWED: Final[frozenset[str]] = frozenset({"td", "holiday", "user"})
_X11REGRESSION_TDPRIOR_LENGTH: Final[int] = 7


@dataclass(frozen=True, slots=True)
class X13x11regression:
    """The ``x11regression`` spec block — calendar / outlier regression on irregulars.

    Mirrors ``x13spec.jl:512-547``. The X-11 sibling of the regARIMA
    :class:`X13regression` block — applies calendar / trading-day
    regressions to the X-11 irregular component.
    """

    aicdiff: float | X13default
    aictest: str | list[str] | X13default
    critical: float | X13default
    data: MVTSeries | X13default
    file: str | X13default
    format: str | X13default
    outliermethod: str | X13default
    outlierspan: MITRange | Span | X13default
    print: str | list[str] | X13default
    save: str | list[str] | X13default
    savelog: str | list[str] | X13default
    prior: bool | X13default
    sigma: float | X13default
    span: MITRange | Span | X13default
    start: MIT | X13default
    tdprior: list[float] | X13default
    user: str | list[str] | X13default
    usertype: str | list[str] | X13default
    variables: _VariablesField
    almost: float | X13default
    b: list[float] | X13default
    fixb: list[bool] | X13default
    centeruser: str | X13default
    eastermeans: bool | X13default
    forcecal: bool | X13default
    noapply: list[str] | X13default
    reweight: bool | X13default
    umdata: MVTSeries | X13default
    umfile: str | X13default
    umformat: str | X13default
    umname: list[str] | str | X13default
    umprecision: int | X13default
    umstart: MIT | X13default
    umtrimzero: bool | str | X13default


_X11REGRESSION_PRINT_ALL: Final[list[str]] = [
    "priortd",
    "extremeval",
    "x11reg",
    "tradingday",
    "combtradingday",
    "holiday",
    "calendar",
    "combcalendar",
    "outlierhdr",
    "xaictest",
    "extremevalb",
    "x11regb",
    "tradingdayb",
    "combtradingdayb",
    "holidayb",
    "calendarb",
    "combcalendarb",
    "outlieriter",
    "outliertests",
    "xregressionmatrix",
    "xregressioncmatrix",
]
_X11REGRESSION_SAVE_ALL: Final[list[str]] = [
    "priortd",
    "extremeval",
    "tradingday",
    "combtradingday",
    "holiday",
    "calendar",
    "combcalendar",
    "extremevalb",
    "tradingdayb",
    "combtradingdayb",
    "holidayb",
    "calendarb",
    "combcalendarb",
    "outlieriter",
    "xregressionmatrix",
    "xregressioncmatrix",
]

_X11REGRESSION_TD_AIC_TYPES: Final[frozenset[str]] = frozenset(
    {"td", "tdstock", "td1coef", "tdstock1coef"}
)


def _x11regression_variable_type(v: _VariableArg) -> str:
    """Resolve a regressor argument to its X-13 type-token name."""
    if isinstance(v, str):
        return v
    return v.__class__.__name__


def x11regression(  # noqa: PLR0912, PLR0915
    *,
    aicdiff: float | X13default = _X13DEFAULT,
    aictest: str | list[str] | X13default = _X13DEFAULT,
    critical: float | X13default = _X13DEFAULT,
    data: MVTSeries | X13default = _X13DEFAULT,
    file: str | X13default = _X13DEFAULT,
    format: str | X13default = _X13DEFAULT,
    outliermethod: str | X13default = _X13DEFAULT,
    outlierspan: MITRange | Span | X13default = _X13DEFAULT,
    print: str | list[str] | X13default = _X13DEFAULT,
    save: str | list[str] | X13default = _X13DEFAULT,
    savelog: str | list[str] | X13default | None = None,
    prior: bool | X13default = _X13DEFAULT,
    sigma: float | X13default = _X13DEFAULT,
    span: MITRange | Span | X13default = _X13DEFAULT,
    tdprior: list[float] | X13default = _X13DEFAULT,
    usertype: str | list[str] | X13default = _X13DEFAULT,
    variables: _VariablesField = _X13DEFAULT,
    almost: float | X13default = _X13DEFAULT,
    b: list[float] | X13default = _X13DEFAULT,
    fixb: list[bool] | X13default = _X13DEFAULT,
    centeruser: str | X13default = _X13DEFAULT,
    eastermeans: bool | X13default = _X13DEFAULT,
    forcecal: bool | X13default = _X13DEFAULT,
    noapply: list[str] | X13default = _X13DEFAULT,
    reweight: bool | X13default = _X13DEFAULT,
    umdata: MVTSeries | X13default = _X13DEFAULT,
    umfile: str | X13default = _X13DEFAULT,
    umformat: str | X13default = _X13DEFAULT,
    umprecision: int | X13default = _X13DEFAULT,
    umtrimzero: bool | str | X13default = _X13DEFAULT,
) -> X13x11regression:
    """Build the ``x11regression`` spec — calendar / outlier irregular-component regression.

    Mirrors ``x13spec.jl:3420-3559``. The ``start`` / ``user`` / ``umstart`` /
    ``umname`` fields are derived from ``data`` / ``umdata`` (mirrors the
    Julia upstream's derivation; no user-facing kwargs accepted).

    Validation:

    * ``aictest`` (when set) restricted to ``{td, tdstock, td1coef,
      tdstock1coef, easter, user}``. If both a TD ``aictest`` entry and a
      TD ``variables`` entry are present, the AIC-test set must be a
      subset of the variables-used set.
    * ``sigma`` must be > 0.
    * ``tdprior`` must have length 7 and all entries ≥ 0.
    * ``usertype`` (when set) restricted to ``{td, holiday, user}``.
      Vector form's length must match the number of user series.
    * :class:`Span` ``outlierspan`` rejects fuzzy ``e`` (the builder
      accepts only :class:`MIT` or :data:`None` already; the upstream
      check collapses to a no-op).
    """
    if savelog is None:
        savelog = "aictest"

    start: MIT | X13default = _X13DEFAULT
    user: str | list[str] | X13default = _X13DEFAULT
    if not isinstance(data, X13default):
        start = data.range.first()
        names = list(data.column_names)
        user = names[0] if len(names) == 1 else names

    umstart: MIT | X13default = _X13DEFAULT
    umname: list[str] | str | X13default = _X13DEFAULT
    if not isinstance(umdata, X13default):
        umstart = umdata.range.first()
        umnames = list(umdata.column_names)
        umname = umnames[0] if len(umnames) == 1 else umnames

    if not isinstance(variables, X13default) and not isinstance(aictest, X13default):
        vars_list: list[_VariableArg] = (
            list(variables) if isinstance(variables, list) else [variables]
        )
        aics_list = [aictest] if isinstance(aictest, str) else list(aictest)
        types_used = {_x11regression_variable_type(v) for v in vars_list}
        has_td_in_aics = any(a in _X11REGRESSION_TD_AIC_TYPES for a in aics_list)
        has_td_in_vars = bool(types_used & _X11REGRESSION_TD_AIC_TYPES)
        if has_td_in_aics and has_td_in_vars:
            for aic in aics_list:
                if aic in _X11REGRESSION_TD_AIC_TYPES and aic not in types_used:
                    msg = (
                        f"Trading day regressors specified in the aictest "
                        f"must correspond with trading day regressors "
                        f"provided in the variables argument. {aic} was "
                        f"specified in the aictest argument, but the "
                        f"variables argument uses "
                        f"{sorted(types_used & _X11REGRESSION_TD_AIC_TYPES)}."
                    )
                    raise ValueError(msg)

    if isinstance(aictest, str):
        if aictest not in _X11REGRESSION_AICTEST_ALLOWED:
            msg = (
                f"aictest can only contain these entries: "
                f"{sorted(_X11REGRESSION_AICTEST_ALLOWED)}. "
                f"Received: {aictest}."
            )
            raise ValueError(msg)
    elif isinstance(aictest, list):
        bad_aic = [a for a in aictest if a not in _X11REGRESSION_AICTEST_ALLOWED]
        if bad_aic:
            msg = (
                f"aictest can only contain these entries: "
                f"{sorted(_X11REGRESSION_AICTEST_ALLOWED)}. "
                f"Received: {aictest}."
            )
            raise ValueError(msg)

    if not isinstance(sigma, X13default) and sigma <= 0.0:
        msg = f"sigma must be a number greater than 0. Received: {sigma}."
        raise ValueError(msg)

    if not isinstance(tdprior, X13default):
        if len(tdprior) != _X11REGRESSION_TDPRIOR_LENGTH:
            msg = (
                f"tdprior must have a length of exactly "
                f"{_X11REGRESSION_TDPRIOR_LENGTH}. Received: {tdprior}."
            )
            raise ValueError(msg)
        if any(x < 0.0 for x in tdprior):
            msg = f"tdprior values must all be greater than or equal to 0. Received: {tdprior}."
            raise ValueError(msg)

    if not isinstance(usertype, X13default):
        if (
            isinstance(usertype, list)
            and isinstance(user, list)
            and len(usertype) > 1
            and len(usertype) != len(user)
        ):
            msg = (
                f"The usertype argument must have the same length as "
                f"the number of user series provided ({len(user)}) when "
                f"more than a single type is specified. "
                f"Received: {usertype}"
            )
            raise ValueError(msg)
        if isinstance(usertype, list):
            bad = [u for u in usertype if u not in _X11REGRESSION_USERTYPE_ALLOWED]
            if bad:
                msg = (
                    f"The usertype argument can only have the following "
                    f"values: {sorted(_X11REGRESSION_USERTYPE_ALLOWED)}. "
                    f"\n\nReceived: {usertype}"
                )
                raise ValueError(msg)
        elif isinstance(usertype, str) and usertype not in _X11REGRESSION_USERTYPE_ALLOWED:
            msg = (
                f"The usertype argument can only have the following "
                f"values: {sorted(_X11REGRESSION_USERTYPE_ALLOWED)}. "
                f"\n\nReceived: {usertype}"
            )
            raise ValueError(msg)

    print = _expand_all(print, _X11REGRESSION_PRINT_ALL)
    save = _expand_all(save, _X11REGRESSION_SAVE_ALL)
    return X13x11regression(
        aicdiff=aicdiff,
        aictest=aictest,
        critical=critical,
        data=data,
        file=file,
        format=format,
        outliermethod=outliermethod,
        outlierspan=outlierspan,
        print=print,
        save=save,
        savelog=savelog,
        prior=prior,
        sigma=sigma,
        span=span,
        start=start,
        tdprior=tdprior,
        user=user,
        usertype=usertype,
        variables=variables,
        almost=almost,
        b=b,
        fixb=fixb,
        centeruser=centeruser,
        eastermeans=eastermeans,
        forcecal=forcecal,
        noapply=noapply,
        reweight=reweight,
        umdata=umdata,
        umfile=umfile,
        umformat=umformat,
        umname=umname,
        umprecision=umprecision,
        umstart=umstart,
        umtrimzero=umtrimzero,
    )
