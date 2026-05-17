# SPDX-License-Identifier: MIT
"""Tests for :mod:`tsecon._various` — ``overlay`` / ``compare`` / ``reindex``.

Ports the tutorial-1 §14 / §15 / §"reindex" examples from
``TimeSeriesEcon.jl`` and locks the Python-side surface: kwarg
behaviour, frequency-mismatch raises, copy-vs-wrap semantics,
``CompareResult`` introspection, and the recursive container walks.
"""

from __future__ import annotations

import math

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings

from tsecon import (
    MIT,
    MITRange,
    MVTSeries,
    TSeries,
    Unit,
    Workspace,
    Yearly,
    compare,
    mm,
    overlay,
    qq,
    reindex,
    yy,
)
from tsecon._various import CompareDifference, CompareResult


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------


class TestOverlayTSeries:
    """Tutorial 1 §14 first mode: range-union TSeries with first-non-NaN wins."""

    def test_single_argument(self) -> None:
        # A single-TSeries overlay matches Julia's `overlay(x::TSeries, tseries::TSeries...)`
        # for the empty tail — same values + range, possibly a fresh TSeries object.
        t = TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
        out = overlay(t)
        assert out.range == t.range
        np.testing.assert_array_equal(out.values, t.values)

    def test_two_inputs_first_wins_on_overlap(self) -> None:
        # 2020Q1/Q2 are covered by both x1 (non-NaN) and x2; x1 wins position-by-position.
        x1 = TSeries(qq(2020, 1), [1.0, 2.0, np.nan, 4.0])
        x2 = TSeries(qq(2019, 3), [10.0, 20.0, 30.0, 40.0])
        out = overlay(x1, x2)
        assert out.firstdate == qq(2019, 3)
        assert out.lastdate == qq(2020, 4)
        expected = [10.0, 20.0, 1.0, 2.0, math.nan, 4.0]
        for got, want in zip(out.values, expected, strict=True):
            if math.isnan(want):
                assert math.isnan(got)
            else:
                assert got == want

    def test_three_inputs_first_non_nan_wins(self) -> None:
        # Tutorial-style: x1 NaN on Q2/Q3; x2 covers 2019Q3..2020Q2 NaN on Q4/Q1; x3 covers Q2 onward.
        x1 = TSeries(qq(2020, 1), [1.0, 1.0, 1.0, 1.0])
        x1[MITRange(qq(2020, 2), qq(2020, 3))] = np.nan
        x2 = TSeries(qq(2019, 3), [2.0, 2.0, 2.0, 2.0])
        x2[MITRange(qq(2019, 4), qq(2020, 1))] = np.nan
        x3 = TSeries(qq(2020, 2), [3.0, 3.0, 3.0, 3.0])
        out = overlay(x1, x2, x3)
        assert out.range == MITRange(qq(2019, 3), qq(2021, 1))
        # 2020Q1: x1=1.0 (valid) wins
        assert out[qq(2020, 1)] == 1.0
        # 2020Q2: x1=NaN → x2 covers Q2 with value 2.0 (valid) → wins
        assert out[qq(2020, 2)] == 2.0
        # 2020Q3: x1=NaN, x2 doesn't cover, x3=3.0
        assert out[qq(2020, 3)] == 3.0
        # 2019Q4: x1 doesn't cover, x2=NaN, x3 doesn't cover → stays NaN
        assert math.isnan(out[qq(2019, 4)])

    def test_rng_forces_output_range(self) -> None:
        x1 = TSeries(qq(2020, 1), [1.0, 2.0, 3.0, 4.0])
        x2 = TSeries(qq(2019, 3), [10.0, 20.0])
        forced = MITRange(qq(2020, 1), qq(2020, 4))
        out = overlay(x1, x2, rng=forced)
        assert out.range == forced
        assert list(out.values) == [1.0, 2.0, 3.0, 4.0]

    def test_dtype_promotion_int_to_float(self) -> None:
        # Mixed int + float input → float64 output.
        x1 = TSeries(qq(2020, 1), np.array([1, 2, 3], dtype=np.int64))
        x2 = TSeries(qq(2020, 1), np.array([1.5, 2.5, 3.5]))
        out = overlay(x1, x2)
        assert np.issubdtype(out.values.dtype, np.floating)

    def test_frequency_mismatch_raises(self) -> None:
        x1 = TSeries(qq(2020, 1), [1.0, 2.0])
        x2 = TSeries(yy(2020), [10.0, 20.0])
        with pytest.raises(TypeError, match="Mixing frequencies"):
            overlay(x1, x2)

    def test_empty_overlap_keeps_nan(self) -> None:
        x1 = TSeries(qq(2020, 1), [1.0, 2.0])
        x2 = TSeries(qq(2021, 1), [3.0, 4.0])
        out = overlay(x1, x2)
        # No overlap between x1 and x2 → output is 2020Q1..2021Q2; gap 2020Q3..2020Q4 stays NaN.
        assert out.range == MITRange(qq(2020, 1), qq(2021, 2))
        assert math.isnan(out[qq(2020, 3)])
        assert math.isnan(out[qq(2020, 4)])
        assert out[qq(2021, 1)] == 3.0

    def test_zero_args_raises(self) -> None:
        with pytest.raises(TypeError, match="at least one"):
            overlay()

    def test_rng_kwarg_rejected_for_workspaces(self) -> None:
        w = Workspace(a=1)
        with pytest.raises(TypeError, match="only valid when all arguments are TSeries"):
            overlay(w, w, rng=MITRange(qq(2020, 1), qq(2020, 4)))


class TestOverlayWorkspace:
    """Tutorial 1 §14 second mode: recursive Workspace / MVTSeries overlay."""

    def test_workspaces_recurse_tseries(self) -> None:
        x1 = TSeries(qq(2020, 1), [1.0, np.nan])
        x2 = TSeries(qq(2020, 1), [99.0, 22.0])
        w1 = Workspace(a=1, x=x1)
        w2 = Workspace(b=2, x=x2)
        out = overlay(w1, w2)
        assert isinstance(out, Workspace)
        # `a` from w1, `b` from w2, `x` recursively overlaid (x1 wins on Q1, x2 fills NaN at Q2)
        assert out.a == 1
        assert out.b == 2
        assert list(out.x.values) == [1.0, 22.0]

    def test_workspace_mvtseries_mixed(self) -> None:
        # Tutorial example: w1 has TSeries, w2 is an MVTSeries with x column.
        x1 = TSeries(qq(2020, 1), [1.0, 2.0])
        x2 = TSeries(qq(2020, 1), [10.0, 20.0])
        w1 = Workspace(x=x1, a=1)
        m2 = MVTSeries(qq(2020, 1), ["x", "b"], np.array([[100.0, 5.0], [200.0, 6.0]]))
        # m2.x is a TSeries view → x recursively overlays as TSeries; b is just the column.
        out = overlay(w1, m2)
        assert isinstance(out, Workspace)
        assert list(out.x.values) == [1.0, 2.0]

    def test_nested_workspaces(self) -> None:
        inner1 = Workspace(t=TSeries(qq(2020, 1), [1.0, np.nan]))
        inner2 = Workspace(t=TSeries(qq(2020, 1), [99.0, 5.0]))
        w1 = Workspace(level1=inner1)
        w2 = Workspace(level1=inner2)
        out = overlay(w1, w2)
        assert list(out.level1.t.values) == [1.0, 5.0]

    def test_workspace_scalar_first_wins(self) -> None:
        # When the same key holds a scalar in both, the leftmost non-typenan wins.
        w1 = Workspace(c=3)
        w2 = Workspace(c=5)
        assert overlay(w1, w2).c == 3

    def test_workspace_scalar_nan_falls_through(self) -> None:
        w1 = Workspace(c=float("nan"))
        w2 = Workspace(c=5.0)
        assert overlay(w1, w2).c == 5.0


class TestOverlayMVTSeries:
    """Column-set union, ordered by left-to-right appearance; range is span."""

    def test_column_union_preserves_order(self) -> None:
        m1 = MVTSeries(qq(2020, 1), ["a", "b"], np.array([[1.0, 2.0], [3.0, 4.0]]))
        m2 = MVTSeries(qq(2020, 1), ["a", "c"], np.array([[10.0, 20.0], [30.0, 40.0]]))
        out = overlay(m1, m2)
        # Column order matches left-to-right appearance: a (from m1), b (m1 only), c (m2 only).
        assert tuple(out.column_names) == ("a", "b", "c")

    def test_first_non_nan_wins_per_column(self) -> None:
        m1 = MVTSeries(qq(2020, 1), ["a"], np.array([[1.0], [np.nan], [3.0]]))
        m2 = MVTSeries(qq(2020, 1), ["a"], np.array([[99.0], [22.0], [99.0]]))
        out = overlay(m1, m2)
        assert list(out.a.values) == [1.0, 22.0, 3.0]

    def test_range_span(self) -> None:
        m1 = MVTSeries(qq(2020, 1), ["a"], np.array([[1.0], [2.0]]))
        m2 = MVTSeries(qq(2020, 3), ["a"], np.array([[3.0], [4.0]]))
        out = overlay(m1, m2)
        assert out.range == MITRange(qq(2020, 1), qq(2020, 4))


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


class TestCompareEqual:
    """Recursive comparison across TSeries / MVTSeries / Workspace / scalar pairs."""

    def test_identical_workspaces_are_equal(self) -> None:
        w1 = Workspace(a=1, t=TSeries(qq(2020, 1), [1.0, 2.0]))
        w2 = Workspace(a=1, t=TSeries(qq(2020, 1), [1.0, 2.0]))
        res = compare(w1, w2, quiet=True)
        assert res
        assert res.equal

    def test_differ_in_tseries_reports_path(self) -> None:
        w1 = Workspace(t=TSeries(qq(2020, 1), [1.0, 2.0]))
        w2 = Workspace(t=TSeries(qq(2020, 1), [1.0, 3.0]))
        res = compare(w1, w2, quiet=True)
        assert not res
        # We expect at least one CompareDifference identifying `_.t` as different.
        paths = [d.path for d in res.differences]
        assert ("_", "t") in paths

    def test_atol_allows_close_match(self) -> None:
        w1 = Workspace(t=TSeries(qq(2020, 1), [1.0, 2.0]))
        w2 = Workspace(t=TSeries(qq(2020, 1), [1.0, 2.0 + 1e-9]))
        # Default rtol ≈ sqrt(eps) ≈ 1.49e-8 covers the 1e-9 gap.
        assert compare(w1, w2, quiet=True)
        # An explicit small atol still passes.
        assert compare(w1, w2, atol=1e-8, quiet=True)

    def test_rtol_zero_atol_zero_is_strict(self) -> None:
        w1 = Workspace(t=TSeries(qq(2020, 1), [1.0, 2.0]))
        w2 = Workspace(t=TSeries(qq(2020, 1), [1.0, 2.0 + 1e-12]))
        # rtol=0 + atol=0 → strict equality (no slack)
        assert not compare(w1, w2, atol=0.0, rtol=0.0, quiet=True)

    def test_nans_kwarg_makes_nan_equal_nan(self) -> None:
        n1 = Workspace(t=TSeries(qq(2020, 1), [1.0, float("nan")]))
        n2 = Workspace(t=TSeries(qq(2020, 1), [1.0, float("nan")]))
        # Default: NaN != NaN
        assert not compare(n1, n2, quiet=True)
        # With nans=True: NaN == NaN
        assert compare(n1, n2, nans=True, quiet=True)

    def test_ignoremissing_skips_unmatched_keys(self) -> None:
        w1 = Workspace(a=1)
        w2 = Workspace(a=1, b=2)
        # Default: b is "missing in left" → different
        assert not compare(w1, w2, quiet=True)
        # ignoremissing: b is skipped → equal
        assert compare(w1, w2, ignoremissing=True, quiet=True)

    def test_showequal_emits_same_lines(self) -> None:
        w1 = Workspace(a=1, b=2)
        w2 = Workspace(a=1, b=2)
        res = compare(w1, w2, showequal=True, quiet=True)
        assert res
        # showequal: each key plus top-level should produce a "same" line.
        msgs = [d.message for d in res.differences]
        assert msgs.count("same") >= 3  # a, b, plus top-level

    def test_compare_scalar_numbers(self) -> None:
        # Out at top level: a plain number-vs-number comparison.
        assert compare(1.0, 1.0 + 1e-12, quiet=True)
        assert not compare(1.0, 2.0, quiet=True)

    def test_compare_tseries_freq_mismatch_is_different(self) -> None:
        a = TSeries(qq(2020, 1), [1.0, 2.0])
        b = TSeries(yy(2020), [1.0, 2.0])
        assert not compare(a, b, quiet=True)

    def test_compare_tseries_range_mismatch_is_different(self) -> None:
        a = TSeries(qq(2020, 1), [1.0, 2.0])
        b = TSeries(qq(2020, 2), [2.0, 3.0])
        assert not compare(a, b, quiet=True)
        # With ignoremissing, the overlap (Q2 only) is compared → equal there
        assert compare(a, b, ignoremissing=True, quiet=True)

    def test_trange_restricts_window(self) -> None:
        a = TSeries(qq(2020, 1), [1.0, 2.0, 999.0, 4.0])
        b = TSeries(qq(2020, 1), [1.0, 2.0, 3.0, 4.0])
        # Whole range: different.
        assert not compare(a, b, quiet=True)
        # Restricting to Q1..Q2 (where they match): equal.
        assert compare(a, b, trange=MITRange(qq(2020, 1), qq(2020, 2)), quiet=True)

    def test_compare_mvtseries_equal_and_differ(self) -> None:
        m1 = MVTSeries(qq(2020, 1), ["a", "b"], np.array([[1.0, 2.0], [3.0, 4.0]]))
        m2 = MVTSeries(qq(2020, 1), ["a", "b"], np.array([[1.0, 2.0], [3.0, 4.0]]))
        assert compare(m1, m2, quiet=True)
        m2.b[qq(2020, 2)] = 999.0
        assert not compare(m1, m2, quiet=True)

    def test_compare_left_right_labels(self) -> None:
        w1 = Workspace(a=1)
        w2 = Workspace(b=2)
        res = compare(w1, w2, left="ALPHA", right="BETA", quiet=True)
        msgs = [d.message for d in res.differences]
        assert any("missing in ALPHA" in m for m in msgs)
        assert any("missing in BETA" in m for m in msgs)


class TestCompareResult:
    """Surface contract of :class:`CompareResult` (truthy, str, repr)."""

    def test_bool_for_equal(self) -> None:
        res = compare(1, 1, quiet=True)
        assert bool(res) is True

    def test_bool_for_diff(self) -> None:
        res = compare(1, 2, quiet=True)
        assert bool(res) is False

    def test_str_renders_differences(self) -> None:
        w1 = Workspace(t=TSeries(qq(2020, 1), [1.0, 2.0]))
        w2 = Workspace(t=TSeries(qq(2020, 1), [1.0, 9.0]))
        res = compare(w1, w2, quiet=True)
        s = str(res)
        assert "_.t: different" in s
        assert "_: different" in s

    def test_str_for_equal(self) -> None:
        # Equal comparison still emits a top-level "same" line.
        res = compare(1, 1, quiet=True)
        assert "_: same" in str(res)

    def test_repr_form(self) -> None:
        res = compare(1, 2, quiet=True)
        # `1 difference` (singular form).
        assert "1 difference" in repr(res)
        assert "equal=False" in repr(res)

    def test_differences_are_named_tuples_of_path(self) -> None:
        w1 = Workspace(t=TSeries(qq(2020, 1), [1.0]))
        w2 = Workspace(t=TSeries(qq(2020, 1), [9.0]))
        res = compare(w1, w2, quiet=True)
        for d in res.differences:
            assert isinstance(d, CompareDifference)
            assert isinstance(d.path, tuple)
            assert isinstance(d.message, str)

    def test_quiet_does_not_print(self, capsys: pytest.CaptureFixture[str]) -> None:
        compare(1, 1, quiet=True)
        out = capsys.readouterr().out
        assert out == ""

    def test_default_prints_to_stdout(self, capsys: pytest.CaptureFixture[str]) -> None:
        compare(1, 1)
        out = capsys.readouterr().out
        assert "_: same" in out


# ---------------------------------------------------------------------------
# reindex
# ---------------------------------------------------------------------------


class TestReindex:
    """Five dispatch cases × copy=True/False; frequency-mismatch raises; round-trip."""

    def test_mit(self) -> None:
        # Julia tutorial: reindex(2022Q4, 2022Q1 => 1U) === 4U
        out = reindex(qq(2022, 4), (qq(2022, 1), MIT(Unit(), 1)))
        assert out == MIT(Unit(), 4)

    def test_mitrange(self) -> None:
        # Julia: reindex(2021Q1:2022Q4, 2022Q1 => 1U) === -3U:4U
        out = reindex(MITRange(qq(2021, 1), qq(2022, 4)), (qq(2022, 1), MIT(Unit(), 1)))
        assert out.start == MIT(Unit(), -3)
        assert out.stop == MIT(Unit(), 4)

    def test_mitrange_preserves_step(self) -> None:
        rng = MITRange(qq(2020, 1), qq(2022, 1), 2)  # 9 quarters, step 2
        out = reindex(rng, (qq(2020, 1), qq(2018, 1)))
        assert out.step == 2
        assert out.start == qq(2018, 1)
        assert out.stop == qq(2020, 1)

    def test_tseries_copy_false_aliases(self) -> None:
        t = TSeries(qq(2021, 1), np.array([1.0, 2.0, 3.0]))
        out = reindex(t, (qq(2021, 1), MIT(Unit(), 1)), copy=False)
        # Mutating the original's buffer should affect `out` (wrap semantics).
        t.values[0] = 999.0
        assert out.values[0] == 999.0
        assert out.firstdate == MIT(Unit(), 1)

    def test_tseries_copy_true_independent(self) -> None:
        t = TSeries(qq(2021, 1), np.array([1.0, 2.0, 3.0]))
        out = reindex(t, (qq(2021, 1), MIT(Unit(), 1)), copy=True)
        t.values[0] = 999.0
        assert out.values[0] == 1.0

    def test_mvtseries_copy_false_aliases(self) -> None:
        m = MVTSeries(qq(2021, 1), ["a", "b"], np.array([[1.0, 10.0], [2.0, 20.0]]))
        out = reindex(m, (qq(2021, 1), MIT(Unit(), 1)), copy=False)
        m.values[0, 0] = 999.0
        assert out.values[0, 0] == 999.0

    def test_mvtseries_copy_true_independent(self) -> None:
        m = MVTSeries(qq(2021, 1), ["a", "b"], np.array([[1.0, 10.0], [2.0, 20.0]]))
        out = reindex(m, (qq(2021, 1), MIT(Unit(), 1)), copy=True)
        m.values[0, 0] = 999.0
        assert out.values[0, 0] == 1.0

    def test_workspace_recursive(self) -> None:
        # Julia tutorial workspace case.
        w = Workspace(
            a=TSeries(qq(2020, 1), [1.0, 2.0]),
            b=TSeries(qq(2021, 1), [10.0, 20.0]),
            c=1,
            d="string",
        )
        w1 = reindex(w, (qq(2021, 1), MIT(Unit(), 1)))
        assert w1.a.firstdate == MIT(Unit(), -3)
        assert w1.b.firstdate == MIT(Unit(), 1)
        assert w1.c == 1
        assert w1.d == "string"

    def test_workspace_skips_other_frequency(self) -> None:
        w = Workspace(
            qtr=TSeries(qq(2020, 1), [1.0]),
            mth=TSeries(mm(2020, 1), [2.0]),
        )
        out = reindex(w, (qq(2020, 1), MIT(Unit(), 1)))
        # Only the quarterly member gets reindexed; the monthly one passes through.
        assert out.qtr.firstdate == MIT(Unit(), 1)
        assert out.mth.firstdate == mm(2020, 1)

    def test_round_trip(self) -> None:
        t = TSeries(qq(2021, 1), [1.0, 2.0, 3.0, 4.0])
        a = qq(2021, 1)
        b = MIT(Unit(), 1)
        round_trip = reindex(reindex(t, (a, b)), (b, a))
        assert round_trip.firstdate == t.firstdate
        np.testing.assert_array_equal(round_trip.values, t.values)

    def test_bad_pair_type_raises(self) -> None:
        t = TSeries(qq(2021, 1), [1.0])
        with pytest.raises(TypeError, match="2-tuple"):
            reindex(t, [qq(2021, 1), qq(2020, 1)])  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="two MIT instances"):
            reindex(t, (qq(2021, 1), 5))  # type: ignore[arg-type]

    def test_frequency_mismatch_raises(self) -> None:
        t = TSeries(qq(2021, 1), [1.0])
        with pytest.raises(TypeError, match="Mixing frequencies"):
            reindex(t, (yy(2021), MIT(Unit(), 1)))

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError, match="does not support"):
            reindex("not a tsecon type", (qq(2020, 1), qq(2020, 2)))

    def test_workspace_nested(self) -> None:
        inner = Workspace(t=TSeries(qq(2020, 1), [1.0, 2.0]))
        w = Workspace(outer=inner, val=42)
        out = reindex(w, (qq(2020, 1), MIT(Unit(), 1)))
        assert out.outer.t.firstdate == MIT(Unit(), 1)
        assert out.val == 42


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------


@st.composite
def _quarterly_tseries(
    draw: st.DrawFn,
    min_size: int = 1,
    max_size: int = 20,
) -> TSeries:
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    start_q = draw(st.integers(min_value=1, max_value=4))
    start_y = draw(st.integers(min_value=2000, max_value=2030))
    vals = draw(
        st.lists(
            st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n,
            max_size=n,
        )
    )
    return TSeries(qq(start_y, start_q), np.array(vals, dtype=np.float64))


class TestOverlayProperty:
    """Properties that must hold for any TSeries / Workspace input."""

    @settings(max_examples=50, suppress_health_check=(HealthCheck.too_slow,))
    @given(t=_quarterly_tseries())
    def test_overlay_single_recovers_values(self, t: TSeries) -> None:
        out = overlay(t)
        assert out.range == t.range
        np.testing.assert_array_equal(out.values, t.values)

    @settings(max_examples=30, suppress_health_check=(HealthCheck.too_slow,))
    @given(t=_quarterly_tseries(), forced_range=st.booleans())
    def test_overlay_of_itself_recovers_values(self, t: TSeries, forced_range: bool) -> None:
        # overlay(t, t) over the natural span equals t (NaN-free inputs).
        out = overlay(t, t, rng=t.range if forced_range else None)
        assert out.range == t.range
        np.testing.assert_array_equal(out.values, t.values)


class TestCompareProperty:
    """compare(x, x).equal == True for any constructible x; differs detect at least one diff."""

    @settings(max_examples=50, suppress_health_check=(HealthCheck.too_slow,))
    @given(t=_quarterly_tseries())
    def test_compare_self_is_equal(self, t: TSeries) -> None:
        res = compare(t, t, quiet=True)
        assert res.equal

    @settings(max_examples=30, suppress_health_check=(HealthCheck.too_slow,))
    @given(t=_quarterly_tseries(min_size=2))
    def test_perturbation_detects_diff(self, t: TSeries) -> None:
        t2 = TSeries(t.firstdate, t.values.copy())
        # Bump one element by a large value so floating slack cannot mask it.
        t2.values[0] += 1.0
        # Tolerate the corner case where the original value is overflowingly large.
        assume(abs(t.values[0]) < 1e5)
        res = compare(t, t2, quiet=True)
        assert not res.equal


class TestReindexProperty:
    """``reindex(reindex(x, (a, b)), (b, a)) == x`` for the canonical TSeries case."""

    @settings(max_examples=50, suppress_health_check=(HealthCheck.too_slow,))
    @given(t=_quarterly_tseries())
    def test_reindex_round_trip(self, t: TSeries) -> None:
        a = t.firstdate
        b = MIT(Unit(), 1)
        rt = reindex(reindex(t, (a, b)), (b, a))
        assert rt.firstdate == t.firstdate
        assert rt.frequency == t.frequency
        np.testing.assert_array_equal(rt.values, t.values)


# ---------------------------------------------------------------------------
# CompareResult dataclass surface
# ---------------------------------------------------------------------------


class TestCompareResultDataclass:
    """The :class:`CompareResult` API contract."""

    def test_default_equal_is_true(self) -> None:
        r = CompareResult(equal=True)
        assert r
        assert r.differences == []

    def test_default_str_when_no_differences(self) -> None:
        r = CompareResult(equal=True)
        assert str(r) == "(no differences)"

    def test_repr_singular_plural(self) -> None:
        r = CompareResult(equal=False, differences=[CompareDifference(("a",), "x")])
        assert "1 difference" in repr(r)
        r2 = CompareResult(
            equal=False,
            differences=[CompareDifference(("a",), "x"), CompareDifference(("b",), "y")],
        )
        assert "2 differences" in repr(r2)


# ---------------------------------------------------------------------------
# Other frequency dispatch
# ---------------------------------------------------------------------------


def test_reindex_yearly_to_unit() -> None:
    # Cross-frequency reindex: tag a yearly TSeries as Unit-indexed.
    t = TSeries(yy(2020), [1.0, 2.0, 3.0])
    out = reindex(t, (yy(2020), MIT(Unit(), 0)))
    assert out.firstdate == MIT(Unit(), 0)
    assert isinstance(out.frequency, Unit)
    assert list(out.values) == [1.0, 2.0, 3.0]


def test_overlay_works_with_yearly() -> None:
    a = TSeries(yy(2020), [1.0, np.nan, 3.0])
    b = TSeries(yy(2020), [99.0, 22.0, 99.0])
    out = overlay(a, b)
    assert list(out.values) == [1.0, 22.0, 3.0]


def test_compare_workspace_with_mvtseries_member() -> None:
    # Tutorial-style: Workspace containing an MVTSeries, compared to a near-equal copy.
    inner1 = MVTSeries(qq(2020, 1), ["a", "b"], np.array([[1.0, 2.0], [3.0, 4.0]]))
    inner2 = MVTSeries(qq(2020, 1), ["a", "b"], np.array([[1.0, 2.0], [3.0, 4.0 + 0.001]]))
    v1 = Workspace(z=inner1)
    v2 = Workspace(z=inner2)
    # Different by 0.001; default tolerance won't accept that.
    assert not compare(v1, v2, quiet=True)
    # atol=0.01 should accept.
    assert compare(v1, v2, atol=0.01, quiet=True)
