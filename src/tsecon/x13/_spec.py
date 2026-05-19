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

  The Julia source has **26** ``X13var`` types; the session-48 inventory
  prose said 25 — see the session-49 SESSION_LOG entry for the correction.

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
Decision recorded in session 49 (see ``claude_files/SESSION_LOG.md``).

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

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Final

from tsecon.frequencies import Monthly
from tsecon.mit import MIT, mit2yp
from tsecon.mitrange import MITRange
from tsecon.x13._consts import _ORDERED_MONTH_NAMES

__all__ = [
    "ArimaModel",
    "ArimaSpec",
    "RegimeChange",
    "X13default",
    "X13var",
    "ao",
    "aos",
    "easter",
    "easterstock",
    "labor",
    "lom",
    "loq",
    "lpyear",
    "ls",
    "lss",
    "qd",
    "qi",
    "rp",
    "sceaster",
    "seasonal",
    "sincos",
    "so",
    "tc",
    "td",
    "td1coef",
    "td1nolpyear",
    "tdnolpyear",
    "tdstock",
    "tdstock1coef",
    "thank",
    "tl",
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
