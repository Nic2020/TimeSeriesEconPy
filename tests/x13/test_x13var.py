# SPDX-License-Identifier: MIT
"""Round-trip ``__str__`` and constructor semantics for the 26 X13var types.

Mirrors the .spc tokens TimeSeriesEcon.jl emits via ``x13write.jl:277-281``.
Each parametric test below pins one Julia → Python fixture line and locks
the .spc-grammar token the binary will receive when M2.4 wires up the
file writer.

The tests are pure-Python (no X13as binary required) — they exercise only
the dataclass + serialization surface. M2.5 adds end-to-end tests against
the actual binary.
"""

from __future__ import annotations

import pytest

from tsecon import MIT, MITRange, Monthly, Quarterly, Yearly
from tsecon.x13 import (
    ArimaModel,
    ArimaSpec,
    RegimeChange,
    X13default,
    X13var,
    ao,
    aos,
    easter,
    easterstock,
    labor,
    lom,
    loq,
    lpyear,
    ls,
    lss,
    qd,
    qi,
    rp,
    sceaster,
    seasonal,
    sincos,
    so,
    tc,
    td,
    td1coef,
    td1nolpyear,
    tdnolpyear,
    tdstock,
    tdstock1coef,
    thank,
    tl,
)
from tsecon.x13._spec import _mit_to_spc  # private helper, tested directly

# --- Fixture MITs ----------------------------------------------------------
# Three sample MITs covering the three frequency-formatter branches in
# ``_mit_to_spc``: Monthly (month-name suffix), Quarterly (period int),
# Yearly (period == 1 always).

_M = MIT.from_yp(Monthly(), 2020, 7)  # → "2020.jul"
_Q = MIT.from_yp(Quarterly(), 2020, 3)  # → "2020.3"
_Y = MIT.from_yp(Yearly(), 2020, 1)  # → "2020.1"


# ---------------------------------------------------------------------------
# Helper serialization
# ---------------------------------------------------------------------------


class TestMitToSpc:
    """``_mit_to_spc`` mirrors ``x13write.jl:286-293``."""

    def test_monthly_uses_month_abbr(self) -> None:
        assert _mit_to_spc(_M) == "2020.jul"

    def test_monthly_lowercase_january(self) -> None:
        assert _mit_to_spc(MIT.from_yp(Monthly(), 2024, 1)) == "2024.jan"

    def test_monthly_lowercase_december(self) -> None:
        assert _mit_to_spc(MIT.from_yp(Monthly(), 2024, 12)) == "2024.dec"

    def test_quarterly_uses_period_int(self) -> None:
        assert _mit_to_spc(_Q) == "2020.3"

    def test_yearly_period_is_one(self) -> None:
        assert _mit_to_spc(_Y) == "2020.1"


# ---------------------------------------------------------------------------
# X13default sentinel
# ---------------------------------------------------------------------------


class TestX13Default:
    """``X13default`` is a singleton, mirrors Julia's ``isa X13default`` dispatch."""

    def test_singleton_identity(self) -> None:
        assert X13default() is X13default()

    def test_repr(self) -> None:
        assert repr(X13default()) == "X13default()"

    def test_no_fields(self) -> None:
        # __slots__ = () means no instance dict, no attributes settable.
        with pytest.raises(AttributeError):
            X13default().foo = 1  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# X13var base
# ---------------------------------------------------------------------------


class TestX13VarBase:
    """The :class:`X13var` base class hierarchy invariants."""

    @pytest.mark.parametrize(
        "cls",
        [
            ao,
            ls,
            tc,
            so,
            aos,
            lss,
            rp,
            qd,
            qi,
            tl,
            tdstock,
            tdstock1coef,
            easter,
            labor,
            thank,
            sceaster,
            easterstock,
            sincos,
            td,
            tdnolpyear,
            td1coef,
            td1nolpyear,
            lpyear,
            lom,
            loq,
            seasonal,
        ],
    )
    def test_inherits_x13var(self, cls: type) -> None:
        """All 26 concrete types inherit from :class:`X13var`."""
        assert issubclass(cls, X13var)


# ---------------------------------------------------------------------------
# Point outliers (ao / ls / tc / so)
# ---------------------------------------------------------------------------


class TestPointOutliers:
    """4 point outliers: ``ao(mit)``, ``ls(mit)``, ``tc(mit)``, ``so(mit)``.

    Mirrors ``x13write.jl:277``:
    ``"$(nameof(typeof(val)))$(x13write(val.mit))"``.
    """

    @pytest.mark.parametrize(
        ("cls", "expected_prefix"),
        [(ao, "ao"), (ls, "ls"), (tc, "tc"), (so, "so")],
    )
    def test_monthly(self, cls: type, expected_prefix: str) -> None:
        assert str(cls(_M)) == f"{expected_prefix}2020.jul"

    @pytest.mark.parametrize(
        ("cls", "expected_prefix"),
        [(ao, "ao"), (ls, "ls"), (tc, "tc"), (so, "so")],
    )
    def test_quarterly(self, cls: type, expected_prefix: str) -> None:
        assert str(cls(_Q)) == f"{expected_prefix}2020.3"

    def test_equality(self) -> None:
        assert ao(_Q) == ao(_Q)
        assert ao(_Q) != ao(_M)
        assert ao(_Q) != ls(_Q)  # different types, even with same mit

    def test_frozen(self) -> None:
        with pytest.raises(Exception):  # noqa: B017 — FrozenInstanceError or AttributeError
            ao(_Q).mit = _M  # type: ignore[misc]

    def test_hashable(self) -> None:
        # Frozen dataclasses are hashable; covers Set / dict-key use.
        assert hash(ao(_Q)) == hash(ao(_Q))
        assert {ao(_Q), ao(_Q), ao(_M)} == {ao(_Q), ao(_M)}


# ---------------------------------------------------------------------------
# Range outliers (aos / lss / rp / qd / qi / tl)
# ---------------------------------------------------------------------------


class TestRangeOutliers:
    """6 range outliers — two MITs joined with ``-`` plus a classmethod alt-ctor.

    Mirrors ``x13write.jl:278``:
    ``"$(nameof(typeof(val)))$(x13write(val.mit1))-$(x13write(val.mit2))"``.
    """

    @pytest.mark.parametrize(
        ("cls", "name"),
        [(aos, "aos"), (lss, "lss"), (rp, "rp"), (qd, "qd"), (qi, "qi"), (tl, "tl")],
    )
    def test_two_mit_constructor(self, cls: type, name: str) -> None:
        mit1 = MIT.from_yp(Monthly(), 2020, 1)
        mit2 = MIT.from_yp(Monthly(), 2020, 6)
        assert str(cls(mit1, mit2)) == f"{name}2020.jan-2020.jun"

    @pytest.mark.parametrize(
        ("cls", "name"),
        [(aos, "aos"), (lss, "lss"), (rp, "rp"), (qd, "qd"), (qi, "qi"), (tl, "tl")],
    )
    def test_from_range_classmethod(self, cls: type, name: str) -> None:
        mr = MITRange(MIT.from_yp(Quarterly(), 2019, 4), MIT.from_yp(Quarterly(), 2020, 2))
        instance = cls.from_range(mr)
        assert str(instance) == f"{name}2019.4-2020.2"

    def test_from_range_equivalent_to_two_mit(self) -> None:
        a, b = MIT.from_yp(Monthly(), 2020, 1), MIT.from_yp(Monthly(), 2020, 6)
        assert aos(a, b) == aos.from_range(MITRange(a, b))


# ---------------------------------------------------------------------------
# Single-int calendar regressors
# ---------------------------------------------------------------------------


class TestCalendarIntRegressors:
    """7 calendar / TD-stock regressors with a single ``n: int``.

    Mirrors ``x13write.jl:279``:
    ``"$(nameof(typeof(val)))[$(val.n)]"``.
    """

    @pytest.mark.parametrize(
        ("cls", "name"),
        [
            (tdstock, "tdstock"),
            (tdstock1coef, "tdstock1coef"),
            (easter, "easter"),
            (labor, "labor"),
            (thank, "thank"),
            (sceaster, "sceaster"),
            (easterstock, "easterstock"),
        ],
    )
    def test_serializes_with_brackets(self, cls: type, name: str) -> None:
        assert str(cls(7)) == f"{name}[7]"

    def test_zero_n(self) -> None:
        assert str(easter(0)) == "easter[0]"


# ---------------------------------------------------------------------------
# sincos (tuple of ints)
# ---------------------------------------------------------------------------


class TestSincos:
    """``sincos(n: tuple[int, ...])`` mirrors ``x13write.jl:280``.

    Julia uses ``Vector{Int64}``; Python uses :class:`tuple` so the frozen
    dataclass stays hashable. .spc form: ``sincos[1 2]``.
    """

    def test_serializes_with_space_separator(self) -> None:
        assert str(sincos((1, 2))) == "sincos[1 2]"

    def test_single_element(self) -> None:
        assert str(sincos((6,))) == "sincos[6]"

    def test_three_elements(self) -> None:
        assert str(sincos((1, 2, 3))) == "sincos[1 2 3]"

    def test_rejects_list(self) -> None:
        # Frozen-dataclass equality + hashability requires tuple. The
        # __post_init__ enforces this with a self-documenting message.
        with pytest.raises(TypeError, match="tuple"):
            sincos([1, 2])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Regime-change types (td / tdnolpyear / td1coef / td1nolpyear /
#                     lpyear / lom / loq / seasonal)
# ---------------------------------------------------------------------------


_REGIME_TYPES = [
    (td, "td"),
    (tdnolpyear, "tdnolpyear"),
    (td1coef, "td1coef"),
    (td1nolpyear, "td1nolpyear"),
    (lpyear, "lpyear"),
    (lom, "lom"),
    (loq, "loq"),
    (seasonal, "seasonal"),
]


class TestRegimeChange:
    """:class:`RegimeChange` :class:`~enum.StrEnum` invariants."""

    def test_members(self) -> None:
        assert set(RegimeChange) == {
            RegimeChange.BOTH,
            RegimeChange.ZEROBEFORE,
            RegimeChange.ZEROAFTER,
            RegimeChange.NEITHER,
        }

    def test_string_equality(self) -> None:
        assert RegimeChange.BOTH == "both"
        assert RegimeChange.NEITHER == "neither"


class TestRegimeBearingTypes:
    """8 regime-bearing types: ``td*`` / ``lpyear`` / ``lom`` / ``loq`` / ``seasonal``.

    Mirrors ``x13write.jl:281`` — the three-form ternary on
    ``val.regimechange``.
    """

    @pytest.mark.parametrize(("cls", "name"), _REGIME_TYPES)
    def test_no_args_serializes_to_bare_name(self, cls: type, name: str) -> None:
        """``td()`` etc. — Julia's no-arg form, ``regimechange=:neither``."""
        assert str(cls()) == name

    @pytest.mark.parametrize(("cls", "name"), _REGIME_TYPES)
    def test_mit_only_defaults_to_both(self, cls: type, name: str) -> None:
        """``td(mit)`` — Julia's one-arg form, ``regimechange=:both`` default."""
        assert str(cls(_M)) == f"{name}/2020.jul/"

    @pytest.mark.parametrize(("cls", "name"), _REGIME_TYPES)
    def test_explicit_both(self, cls: type, name: str) -> None:
        assert str(cls(_M, RegimeChange.BOTH)) == f"{name}/2020.jul/"

    @pytest.mark.parametrize(("cls", "name"), _REGIME_TYPES)
    def test_zerobefore(self, cls: type, name: str) -> None:
        assert str(cls(_M, RegimeChange.ZEROBEFORE)) == f"{name}//2020.jul/"

    @pytest.mark.parametrize(("cls", "name"), _REGIME_TYPES)
    def test_zeroafter(self, cls: type, name: str) -> None:
        assert str(cls(_M, RegimeChange.ZEROAFTER)) == f"{name}/2020.jul//"

    @pytest.mark.parametrize(("cls", "name"), _REGIME_TYPES)
    def test_explicit_neither_keeps_bare_name(self, cls: type, name: str) -> None:
        """Explicit ``regimechange="neither"`` matches the no-arg form's output."""
        assert str(cls(_M, RegimeChange.NEITHER)) == name

    @pytest.mark.parametrize(("cls", "name"), _REGIME_TYPES)
    def test_string_regimechange_normalised(self, cls: type, name: str) -> None:
        """Plain string ``"both"`` normalises to :attr:`RegimeChange.BOTH`."""
        instance = cls(_M, "both")
        assert instance.regimechange is RegimeChange.BOTH
        assert str(instance) == f"{name}/2020.jul/"

    @pytest.mark.parametrize(("cls", "name"), _REGIME_TYPES)
    def test_rejects_regime_without_mit(self, cls: type, name: str) -> None:
        """``td(regimechange=BOTH)`` with no MIT raises (would emit an invalid token)."""
        with pytest.raises(ValueError, match="requires an MIT"):
            cls(None, RegimeChange.BOTH)

    @pytest.mark.parametrize(("cls", "name"), _REGIME_TYPES)
    def test_quarterly_serialization(self, cls: type, name: str) -> None:
        assert str(cls(_Q)) == f"{name}/2020.3/"

    @pytest.mark.parametrize(("cls", "name"), _REGIME_TYPES)
    def test_equality(self, cls: type, name: str) -> None:
        assert cls(_M) == cls(_M, RegimeChange.BOTH)
        assert cls() == cls(None, RegimeChange.NEITHER)


# ---------------------------------------------------------------------------
# ArimaSpec / ArimaModel
# ---------------------------------------------------------------------------


class TestArimaSpec:
    """``ArimaSpec`` mirrors ``x13spec.jl:216-228``."""

    def test_default_zeros(self) -> None:
        s = ArimaSpec()
        assert (s.p, s.d, s.q, s.period) == (0, 0, 0, 0)

    def test_pdq_positional(self) -> None:
        s = ArimaSpec(1, 1, 1)
        assert (s.p, s.d, s.q, s.period) == (1, 1, 1, 0)

    def test_pdq_with_period(self) -> None:
        s = ArimaSpec(0, 1, 1, 12)
        assert (s.p, s.d, s.q, s.period) == (0, 1, 1, 12)

    def test_missing_lags_via_tuple(self) -> None:
        """Julia's ``ArimaSpec([2, 3], 0, 0)`` ports as ``ArimaSpec((2, 3), 0, 0)``."""
        s = ArimaSpec((2, 3), 0, 0)
        assert s.p == (2, 3)

    def test_missing_lags_via_list_coerced_to_tuple(self) -> None:
        s = ArimaSpec([2, 3], 0, 0)
        assert s.p == (2, 3)
        assert isinstance(s.p, tuple)

    def test_two_seasonal_returns_pair(self) -> None:
        s1, s2 = ArimaSpec.two_seasonal(1, 1, 1, 0, 1, 1)
        assert (s1.p, s1.d, s1.q) == (1, 1, 1)
        assert (s2.p, s2.d, s2.q) == (0, 1, 1)
        assert s1.period == 0
        assert s2.period == 0

    def test_rejects_bool_order(self) -> None:
        with pytest.raises(TypeError, match="not bool"):
            ArimaSpec(True, 0, 0)  # type: ignore[arg-type]

    def test_rejects_non_int_in_tuple(self) -> None:
        with pytest.raises(TypeError, match="elements must be int"):
            ArimaSpec((1, "x"), 0, 0)  # type: ignore[list-item]

    def test_period_x13default(self) -> None:
        s = ArimaSpec(1, 1, 1, X13default())
        assert isinstance(s.period, X13default)

    def test_field_assignment_allowed(self) -> None:
        """ArimaSpec is mutable to mirror Julia's ``mutable struct``."""
        s = ArimaSpec()
        s.p = 2
        assert s.p == 2


class TestArimaModel:
    """``ArimaModel`` mirrors ``x13spec.jl:232-242``."""

    def test_from_pdq_single_operator(self) -> None:
        m = ArimaModel.from_pdq(1, 1, 1)
        assert len(m.specs) == 1
        assert (m.specs[0].p, m.specs[0].d, m.specs[0].q) == (1, 1, 1)
        assert m.default is False

    def test_from_pdq_default_flag(self) -> None:
        m = ArimaModel.from_pdq(0, 1, 1, default=True)
        assert m.default is True

    def test_from_pdq_seasonal_two_operators(self) -> None:
        m = ArimaModel.from_pdq_seasonal(1, 1, 1, 0, 1, 1)
        assert len(m.specs) == 2
        assert (m.specs[0].p, m.specs[0].q) == (1, 1)
        assert (m.specs[1].p, m.specs[1].q) == (0, 1)

    def test_from_specs_varargs(self) -> None:
        s1, s2 = ArimaSpec(1, 1, 1), ArimaSpec(0, 1, 1, 12)
        m = ArimaModel.from_specs(s1, s2)
        assert m.specs == [s1, s2]

    def test_from_specs_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            ArimaModel.from_specs()

    def test_with_period(self) -> None:
        m = ArimaModel.from_pdq(0, 1, 1, 12)
        assert m.specs[0].period == 12
