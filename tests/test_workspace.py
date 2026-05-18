# SPDX-License-Identifier: MIT
"""Tests for ``tsecon.Workspace``.

Ports the cases from ``TimeSeriesEcon.jl/test/test_workspace.jl`` that don't
depend on still-unported modules (MVTSeries, various.jl helpers like
``overlay`` / ``compare`` / ``reindex``, and the legacy
``clean_old_frequencies`` shim).
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest

from tsecon import (
    MIT,
    Duration,
    MITRange,
    MVTSeries,
    TSeries,
    Workspace,
    copyto,
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


# ---------------------------------------------------------------------------
# copyto — in-place Workspace → MVTSeries materialiser
# (parity gap G13 / M1.6.3h)
# ---------------------------------------------------------------------------


def _make_dst(start: MIT | None = None, stop: MIT | None = None) -> MVTSeries:
    s = qq(2020, 1) if start is None else start
    e = qq(2024, 4) if stop is None else stop
    return MVTSeries(MITRange(s, e), ["a", "b"])


class TestCopyto:
    def test_all_columns_written_returns_self(self) -> None:
        dst = _make_dst()
        original_storage = dst._values
        w = Workspace(
            a=TSeries(qq(2020, 1), np.arange(20.0)),
            b=TSeries(qq(2020, 1), np.arange(20.0) * 2),
        )
        out = copyto(dst, w)
        assert out is dst
        assert dst._values is original_storage
        np.testing.assert_array_equal(dst.a.values, np.arange(20.0))
        np.testing.assert_array_equal(dst.b.values, np.arange(20.0) * 2)

    def test_storage_identity_preserved_across_repeated_calls(self) -> None:
        dst = _make_dst()
        ptr = dst._values.ctypes.data
        for i in range(5):
            w = Workspace(
                a=TSeries(qq(2020, 1), np.full(20, float(i))),
                b=TSeries(qq(2020, 1), np.full(20, float(i) + 0.5)),
            )
            copyto(dst, w)
            assert dst._values.ctypes.data == ptr

    def test_extra_workspace_keys_silently_ignored(self) -> None:
        dst = _make_dst()
        w = Workspace(
            a=TSeries(qq(2020, 1), np.arange(20.0)),
            b=TSeries(qq(2020, 1), np.arange(20.0) * 2),
            unused=TSeries(qq(2020, 1), np.arange(20.0) * 3),
            scalar_param=0.42,
        )
        copyto(dst, w)
        np.testing.assert_array_equal(dst.a.values, np.arange(20.0))
        np.testing.assert_array_equal(dst.b.values, np.arange(20.0) * 2)

    def test_missing_column_leaves_dst_untouched_silent_default(self) -> None:
        dst = _make_dst()
        dst._values[:] = -1.0  # sentinel
        w = Workspace(a=TSeries(qq(2020, 1), np.arange(20.0)))
        copyto(dst, w)
        np.testing.assert_array_equal(dst.a.values, np.arange(20.0))
        np.testing.assert_array_equal(dst.b.values, np.full(20, -1.0))

    def test_missing_column_verbose_warns_at_end(self) -> None:
        dst = _make_dst()
        w = Workspace(a=TSeries(qq(2020, 1), np.arange(20.0)))
        with pytest.warns(UserWarning, match=r"Variables not copied.*\bb\b") as wlog:
            copyto(dst, w, verbose=True)
        assert len(wlog) == 1
        np.testing.assert_array_equal(dst.a.values, np.arange(20.0))

    def test_verbose_no_warning_when_all_present(self) -> None:
        dst = _make_dst()
        w = Workspace(
            a=TSeries(qq(2020, 1), np.arange(20.0)),
            b=TSeries(qq(2020, 1), np.arange(20.0) * 2),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any UserWarning would fail
            copyto(dst, w, verbose=True)

    def test_verbose_warns_with_joined_missing_names(self) -> None:
        dst = MVTSeries(MITRange(qq(2020, 1), qq(2024, 4)), ["a", "b", "c"])
        w = Workspace(a=TSeries(qq(2020, 1), np.arange(20.0)))
        with pytest.warns(UserWarning, match="b, c") as wlog:
            copyto(dst, w, verbose=True)
        assert len(wlog) == 1

    def test_non_tseries_value_raises_type_error(self) -> None:
        dst = _make_dst()
        w = Workspace(a=42.0)
        with pytest.raises(TypeError, match="'a' is float, expected TSeries"):
            copyto(dst, w)

    def test_nested_workspace_value_raises_type_error(self) -> None:
        dst = _make_dst()
        w = Workspace(a=Workspace(inner=1))
        with pytest.raises(TypeError, match="'a' is Workspace, expected TSeries"):
            copyto(dst, w)

    def test_frequency_mismatch_raises_type_error(self) -> None:
        dst = _make_dst()
        w = Workspace(a=TSeries(yy(2020), np.arange(5.0)))
        with pytest.raises(TypeError, match="'a' has frequency"):
            copyto(dst, w)

    def test_trange_narrower_than_dst_writes_only_overlap(self) -> None:
        dst = _make_dst()
        dst._values[:] = -1.0
        w = Workspace(
            a=TSeries(qq(2020, 1), np.arange(20.0)),
            b=TSeries(qq(2020, 1), np.arange(20.0) * 2),
        )
        sub = MITRange(qq(2021, 1), qq(2022, 4))  # rows 4..11
        copyto(dst, w, trange=sub)
        # untouched outside trange
        np.testing.assert_array_equal(dst.a.values[:4], np.full(4, -1.0))
        np.testing.assert_array_equal(dst.a.values[12:], np.full(8, -1.0))
        # written inside trange (a[t] = t-offset where offset is 0 in src)
        np.testing.assert_array_equal(dst.a.values[4:12], np.arange(4.0, 12.0))

    def test_trange_wider_than_source_writes_only_overlap(self) -> None:
        dst = _make_dst()
        dst._values[:] = -1.0
        # source covers only rows 4..7 (qq(2021,1)..qq(2021,4))
        w = Workspace(a=TSeries(qq(2021, 1), np.array([10.0, 20.0, 30.0, 40.0])))
        copyto(dst, w)  # trange defaults to dst.range — 20 rows
        # only the source-overlap was written
        np.testing.assert_array_equal(dst.a.values[:4], np.full(4, -1.0))
        np.testing.assert_array_equal(dst.a.values[4:8], [10.0, 20.0, 30.0, 40.0])
        np.testing.assert_array_equal(dst.a.values[8:], np.full(12, -1.0))

    def test_trange_outside_dst_range_raises_index_error(self) -> None:
        dst = _make_dst()
        w = Workspace(a=TSeries(qq(2018, 1), np.arange(40.0)))
        bad = MITRange(qq(2018, 1), qq(2020, 4))  # starts before dst
        with pytest.raises(IndexError, match="not contained"):
            copyto(dst, w, trange=bad)

    def test_trange_frequency_mismatch_raises_type_error(self) -> None:
        dst = _make_dst()
        w = Workspace(a=TSeries(qq(2020, 1), np.arange(20.0)))
        bad = MITRange(yy(2020), yy(2022))
        with pytest.raises(TypeError, match="trange has frequency"):
            copyto(dst, w, trange=bad)

    def test_trange_negative_step_raises_value_error(self) -> None:
        dst = _make_dst()
        w = Workspace(a=TSeries(qq(2020, 1), np.arange(20.0)))
        rev = MITRange(qq(2022, 4), qq(2020, 1), -1)
        with pytest.raises(ValueError, match="step=1"):
            copyto(dst, w, trange=rev)

    def test_dst_must_be_mvtseries(self) -> None:
        w = Workspace(a=TSeries(qq(2020, 1), np.arange(5.0)))
        with pytest.raises(TypeError, match="dst must be MVTSeries"):
            copyto(w, w)  # type: ignore[arg-type]

    def test_src_must_be_workspace(self) -> None:
        dst = _make_dst()
        with pytest.raises(TypeError, match="src must be Workspace"):
            copyto(dst, {"a": TSeries(qq(2020, 1), np.arange(20.0))})  # type: ignore[arg-type]

    def test_empty_overlap_silently_skips(self) -> None:
        dst = _make_dst()
        dst._values[:] = -1.0
        # source disjoint from dst.range
        w = Workspace(a=TSeries(qq(2030, 1), np.arange(4.0)))
        copyto(dst, w)
        np.testing.assert_array_equal(dst.a.values, np.full(20, -1.0))

    def test_repeated_call_overwrites(self) -> None:
        dst = _make_dst()
        w1 = Workspace(
            a=TSeries(qq(2020, 1), np.full(20, 1.0)),
            b=TSeries(qq(2020, 1), np.full(20, 2.0)),
        )
        w2 = Workspace(
            a=TSeries(qq(2020, 1), np.full(20, 3.0)),
            b=TSeries(qq(2020, 1), np.full(20, 4.0)),
        )
        copyto(dst, w1)
        copyto(dst, w2)
        np.testing.assert_array_equal(dst.a.values, np.full(20, 3.0))
        np.testing.assert_array_equal(dst.b.values, np.full(20, 4.0))
