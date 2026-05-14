# SPDX-License-Identifier: MIT
"""Tests for ``tsecon.Workspace``.

Ports the cases from ``TimeSeriesEcon.jl/test/test_workspace.jl`` that don't
depend on still-unported modules (MVTSeries, various.jl helpers like
``overlay`` / ``compare`` / ``reindex``, and the legacy
``clean_old_frequencies`` shim).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from tsecon import (
    MIT,
    Duration,
    MITRange,
    TSeries,
    Workspace,
    mm,
    qq,
    yy,
)
from tsecon.frequencies import Monthly, Quarterly, Yearly

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty(self) -> None:
        w = Workspace()
        assert len(w) == 0
        assert w.is_empty()

    def test_kwargs(self) -> None:
        w = Workspace(a=1, b="hello")
        assert w.a == 1
        assert w.b == "hello"
        assert len(w) == 2

    def test_from_dict_positional(self) -> None:
        w = Workspace({"a": 1, "b": 2})
        assert w.a == 1
        assert w.b == 2

    def test_from_pairs(self) -> None:
        w = Workspace([("a", 1), ("b", 2)])
        assert w.a == 1
        assert w.b == 2

    def test_kwargs_override_positional(self) -> None:
        # Mirror dict-style: kwargs win over positional mapping.
        w = Workspace({"a": 1}, a=99)
        assert w.a == 99

    def test_keys_coerced_to_str(self) -> None:
        w = Workspace({"a": 1, 42: "answer"})
        assert "a" in w
        assert "42" in w

    def test_from_dict_recursive(self) -> None:
        nested = {"w": {"w": {"w": {"a": 1}}}}
        w = Workspace.from_dict(nested, recursive=True)
        assert isinstance(w, Workspace)
        assert isinstance(w.w, Workspace)
        assert isinstance(w.w.w, Workspace)
        assert isinstance(w.w.w.w, Workspace)
        assert w.w.w.w.a == 1

    def test_from_dict_non_recursive_keeps_inner_dicts(self) -> None:
        w = Workspace.from_dict({"w": {"a": 1}})
        assert isinstance(w.w, dict)
        assert w.w == {"a": 1}


# ---------------------------------------------------------------------------
# Access (attribute + bracket)
# ---------------------------------------------------------------------------


class TestAccess:
    def test_dot_get_set(self) -> None:
        w = Workspace()
        w.a = 1
        assert w.a == 1
        w.a = 2
        assert w.a == 2

    def test_bracket_get_set(self) -> None:
        w = Workspace()
        w["a"] = 1
        assert w["a"] == 1

    def test_dot_and_bracket_agree(self) -> None:
        w = Workspace(a=1)
        assert w.a == w["a"] == 1

    def test_missing_attribute_raises(self) -> None:
        w = Workspace()
        with pytest.raises(AttributeError, match="missing"):
            _ = w.missing

    def test_missing_key_raises(self) -> None:
        w = Workspace()
        with pytest.raises(KeyError):
            _ = w["missing"]

    def test_del_by_attr(self) -> None:
        w = Workspace(a=1, b=2)
        del w.a
        assert "a" not in w
        assert "b" in w

    def test_del_by_key(self) -> None:
        w = Workspace(a=1, b=2)
        del w["a"]
        assert "a" not in w

    def test_reserved_attribute_blocked(self) -> None:
        w = Workspace()
        with pytest.raises(AttributeError, match="reserved"):
            del w._c

    def test_dir_includes_members(self) -> None:
        w = Workspace(alpha=1, beta=2)
        names = dir(w)
        assert "alpha" in names
        assert "beta" in names


# ---------------------------------------------------------------------------
# Subset access
# ---------------------------------------------------------------------------


class TestSubset:
    def test_tuple_subset(self) -> None:
        w = Workspace(a=1, b=2, c=3)
        sub = w["a", "c"]
        assert isinstance(sub, Workspace)
        assert list(sub.keys()) == ["a", "c"]
        assert sub.a == 1
        assert sub.c == 3

    def test_list_subset(self) -> None:
        w = Workspace(a=1, b=2, c=3)
        sub = w[["a", "b"]]
        assert isinstance(sub, Workspace)
        assert len(sub) == 2

    def test_subset_unknown_key_raises(self) -> None:
        w = Workspace(a=1)
        with pytest.raises(KeyError):
            _ = w[("a", "z")]

    def test_invalid_key_type(self) -> None:
        w = Workspace(a=1)
        with pytest.raises(TypeError):
            _ = w[42]


# ---------------------------------------------------------------------------
# dict-like interface
# ---------------------------------------------------------------------------


class TestDictLike:
    def test_iter_yields_keys_in_order(self) -> None:
        w = Workspace(b=2, a=1, c=3)
        assert list(iter(w)) == ["b", "a", "c"]

    def test_keys_values_items(self) -> None:
        w = Workspace(a=1, b=2)
        assert list(w.keys()) == ["a", "b"]
        assert list(w.values()) == [1, 2]
        assert list(w.items()) == [("a", 1), ("b", 2)]

    def test_contains(self) -> None:
        w = Workspace(a=1)
        assert "a" in w
        assert "b" not in w
        # Non-string keys are never contained.
        assert 1 not in w  # type: ignore[operator]

    def test_get(self) -> None:
        w = Workspace(a=1)
        assert w.get("a") == 1
        assert w.get("missing") is None
        assert w.get("missing", "default") == "default"


# ---------------------------------------------------------------------------
# merge / empty / copy
# ---------------------------------------------------------------------------


class TestMutation:
    def test_merge_returns_new(self) -> None:
        a = Workspace(a=1, b=2)
        b = a.merge(Workspace(z=12))
        assert "z" not in a
        assert b.z == 12
        assert list(b.keys()) == ["a", "b", "z"]

    def test_merge_other_wins(self) -> None:
        a = Workspace(a=1)
        b = a.merge(Workspace(a=99))
        assert b.a == 99

    def test_merge_inplace(self) -> None:
        a = Workspace(a=1)
        result = a.merge_inplace(Workspace(z=2))
        assert result is a
        assert a.z == 2

    def test_merge_with_plain_dict(self) -> None:
        a = Workspace(a=1)
        b = a.merge({"z": 2})
        assert b.z == 2

    def test_empty_inplace(self) -> None:
        a = Workspace(a=1, b=2)
        a.empty_inplace()
        assert a.is_empty()

    def test_copy_is_shallow(self) -> None:
        v = [1, 2]
        a = Workspace(x=v)
        b = a.copy()
        assert b.x is v
        # but the storage is independent
        b.x = [9, 9]
        assert a.x == [1, 2]

    def test_copy_module_works(self) -> None:
        a = Workspace(x=1, y=2)
        b = copy.copy(a)
        assert b == a
        b.x = 99
        assert a.x == 1

    def test_deepcopy_independent_tseries_value(self) -> None:
        # B1 regression: copy.deepcopy on a Workspace with a TSeries member
        # must produce an independent TSeries (not aliased) and a fresh
        # container dict.
        ts = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
        a = Workspace(s=ts)
        b = copy.deepcopy(a)
        assert b.s is not a.s
        b.s.values[0] = 99.0
        assert a.s.values[0] == 1.0

    def test_deepcopy_honors_memo(self) -> None:
        # Shared identity inside a Workspace round-trips through deepcopy.
        shared = [1, 2, 3]
        a = Workspace(x=shared, y=shared)
        b = copy.deepcopy(a)
        assert b.x is b.y
        assert b.x is not shared

    def test_deepcopy_handles_self_reference(self) -> None:
        # The memo dict makes the deepcopy cycle-safe.
        a = Workspace(x=1)
        a.self_ref = a
        b = copy.deepcopy(a)
        assert b.self_ref is b
        assert b is not a

    def test_copy_deep_kwarg_equivalent_to_deepcopy(self) -> None:
        ts = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
        a = Workspace(s=ts)
        b = a.copy(deep=True)
        assert b.s is not a.s
        b.s.values[0] = 99.0
        assert a.s.values[0] == 1.0


# ---------------------------------------------------------------------------
# filter / map
# ---------------------------------------------------------------------------


class TestFilterMap:
    def test_filter_keeps_matching(self) -> None:
        w = Workspace(a=1, b="hello", c=2.5)
        sub = w.filter(lambda _k, v: isinstance(v, (int, float)) and not isinstance(v, bool))
        assert list(sub.keys()) == ["a", "c"]

    def test_filter_inplace_returns_self(self) -> None:
        w = Workspace(a=1, b=2, c=3)
        result = w.filter_inplace(lambda _k, v: v >= 2)
        assert result is w
        assert list(w.keys()) == ["b", "c"]

    def test_map_preserves_keys(self) -> None:
        w = Workspace(a=1, b=2, c=3)
        sq = w.map(lambda v: v * v)
        assert list(sq.keys()) == list(w.keys())
        assert sq.a == 1
        assert sq.b == 4
        assert sq.c == 9


# ---------------------------------------------------------------------------
# frequency_of
# ---------------------------------------------------------------------------


class TestFrequencyOf:
    def test_empty_returns_none(self) -> None:
        assert Workspace().frequency_of() is None

    def test_empty_check_raises(self) -> None:
        with pytest.raises(ValueError, match="doesn't have a frequency"):
            Workspace().frequency_of(check=True)

    def test_single_frequency(self) -> None:
        w = Workspace(a=5, b=qq(2020, 1), c=TSeries(qq(2020, 1), np.arange(5.0)))
        f = w.frequency_of()
        assert isinstance(f, Quarterly)

    def test_mixed_frequencies_returns_none(self) -> None:
        w = Workspace(a=qq(2020, 1), b=mm(2020, 1))
        assert w.frequency_of() is None

    def test_mixed_frequencies_check_raises(self) -> None:
        w = Workspace(a=qq(2020, 1), b=mm(2020, 1))
        with pytest.raises(ValueError, match="multiple frequencies"):
            w.frequency_of(check=True)

    def test_recursive_nested_workspace(self) -> None:
        inner = Workspace(a=TSeries(qq(2020, 1), np.arange(5.0)))
        outer = Workspace(inner=inner, mit=qq(2020, 1))
        assert isinstance(outer.frequency_of(), Quarterly)


# ---------------------------------------------------------------------------
# rangeof / rangeof_span
# ---------------------------------------------------------------------------


class TestRange:
    def test_rangeof_intersection(self) -> None:
        w = Workspace(
            a=TSeries(qq(2020, 1), np.arange(10.0)),
            b=TSeries(qq(2021, 1), np.arange(6.0)),
        )
        r = w.rangeof()
        assert r.start == qq(2021, 1)
        assert r.stop == qq(2022, 2)

    def test_rangeof_no_members_raises(self) -> None:
        with pytest.raises(ValueError, match="no rangeable members"):
            Workspace(scalar=1).rangeof()

    def test_rangeof_mixed_freq_raises(self) -> None:
        w = Workspace(
            a=TSeries(qq(2020, 1), np.arange(5.0)),
            b=TSeries(mm(2020, 1), np.arange(5.0)),
        )
        with pytest.raises(TypeError, match="Mixing frequencies"):
            w.rangeof()

    def test_rangeof_span(self) -> None:
        w = Workspace(
            a=MITRange(qq(2020, 1), qq(2025, 3)),
            b=MITRange(qq(2019, 1), qq(2023, 2)),
            c=TSeries(qq(1995, 3), np.random.default_rng(0).standard_normal(22)),
        )
        span = w.rangeof_span()
        assert span.start == qq(1995, 3)
        assert span.stop == qq(2025, 3)


# ---------------------------------------------------------------------------
# strip
# ---------------------------------------------------------------------------


class TestStrip:
    def test_strip_trims_leading_and_trailing_nans(self) -> None:
        rng = MITRange(qq(2020, 1), qq(2022, 2))
        ts = TSeries(rng, np.arange(10.0))
        ts[qq(2020, 1) : qq(2020, 3)] = np.nan
        w = Workspace(ts=ts)
        w.strip_inplace()
        assert w.ts.range == MITRange(qq(2020, 4), qq(2022, 2))

    def test_strip_recurses_into_nested_workspace(self) -> None:
        rng = MITRange(qq(2020, 1), qq(2022, 2))
        inner_ts = TSeries(rng, np.arange(10.0))
        inner_ts[qq(2021, 4) : qq(2022, 2)] = np.nan
        nested = Workspace(ts=inner_ts)
        outer = Workspace(inner=nested)
        outer.strip_inplace()
        assert outer.inner.ts.range == MITRange(qq(2020, 1), qq(2021, 3))

    def test_strip_recursive_false_skips_nested(self) -> None:
        rng = MITRange(qq(2020, 1), qq(2022, 2))
        inner_ts = TSeries(rng, np.arange(10.0))
        inner_ts[qq(2021, 4) : qq(2022, 2)] = np.nan
        nested = Workspace(ts=inner_ts)
        outer = Workspace(inner=nested)
        outer.strip_inplace(recursive=False)
        # nested.ts should be untouched
        assert outer.inner.ts.range == rng

    def test_strip_all_nans_yields_empty(self) -> None:
        rng = MITRange(qq(2020, 1), qq(2020, 4))
        ts = TSeries(rng, np.full(4, np.nan))
        w = Workspace(ts=ts)
        w.strip_inplace()
        assert w.ts.is_empty()


# ---------------------------------------------------------------------------
# equality / repr
# ---------------------------------------------------------------------------


class TestEqRepr:
    def test_eq_same_keys_and_values(self) -> None:
        a = Workspace(a=1, b=2)
        b = Workspace(a=1, b=2)
        assert a == b

    def test_eq_order_matters(self) -> None:
        # Match Julia where keys preserve insertion order — different order
        # is not equal.
        a = Workspace(a=1, b=2)
        b = Workspace(b=2, a=1)
        assert a != b

    def test_eq_tseries_uses_equals(self) -> None:
        t1 = TSeries(qq(2020, 1), np.arange(5.0))
        t2 = TSeries(qq(2020, 1), np.arange(5.0))
        a = Workspace(t=t1)
        b = Workspace(t=t2)
        assert a == b

    def test_eq_unhashable(self) -> None:
        w = Workspace(a=1)
        with pytest.raises(TypeError):
            hash(w)

    def test_repr_empty(self) -> None:
        assert repr(Workspace()) == "Empty Workspace"

    def test_repr_summary(self) -> None:
        w = Workspace(a=1, b="hello", c=Workspace(), d=Workspace(t=4.5))
        text = repr(w)
        assert "Workspace with 4 variables" in text
        assert "a ⇒ 1" in text
        assert "b ⇒ 'hello'" in text
        assert "c ⇒ Empty Workspace" in text
        assert "d ⇒ Workspace with 1 variable" in text

    def test_repr_singular_variable(self) -> None:
        w = Workspace(only=1)
        assert "Workspace with 1 variable" in repr(w)
        assert "Workspace with 1 variables" not in repr(w)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


class TestMisc:
    def test_workspace_can_hold_heterogeneous_values(self) -> None:
        w = Workspace()
        w.scalar = 1
        w.string = "hello"
        w.mit = qq(2020, 1)
        w.dur = qq(2020, 2) - qq(2020, 1)
        w.range = MITRange(qq(2020, 1), qq(2020, 4))
        w.ts = TSeries(qq(2020, 1), np.arange(4.0))
        w.nested = Workspace(inner=1)
        assert isinstance(w.dur, Duration)
        assert isinstance(w.range, MITRange)
        assert isinstance(w.ts, TSeries)
        assert isinstance(w.nested, Workspace)
        assert isinstance(w.mit, MIT)

    def test_yearly_value(self) -> None:
        w = Workspace(y=yy(2020))
        f = w.frequency_of()
        assert isinstance(f, Yearly)

    def test_monthly_value(self) -> None:
        w = Workspace(m=mm(2020, 3))
        f = w.frequency_of()
        assert isinstance(f, Monthly)
