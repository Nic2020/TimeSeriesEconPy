# SPDX-License-Identifier: MIT
"""Tests for ``tsecon._math``: shift / lag / lead / diff / pct / apct / ytypct / moving / undiff.

Ports the relevant cases from ``TimeSeriesEcon.jl``'s ``test/test_tseries.jl``
(``@testset "Iris"`` shift section, ``@testset "TS.math"`` lag/lead section,
``@testset "pct"``) and the moving/undiff cases from
``test/test_mvtseries.jl`` (the TSeries-flavored ones — MVTSeries overloads
land when MVTSeries lands), plus Python-specific edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsecon import (
    apct,
    diff,
    lag,
    lag_inplace,
    lead,
    lead_inplace,
    mm,
    moving,
    moving_average,
    moving_sum,
    pct,
    qq,
    shift,
    shift_inplace,
    undiff,
    undiff_inplace,
    ytypct,
    yy,
)
from tsecon.frequencies import Monthly, Quarterly, Unit, Yearly
from tsecon.mit import MIT
from tsecon.mitrange import MITRange
from tsecon.tseries import TSeries

# ---------------------------------------------------------------------------
# shift / shift_inplace
# ---------------------------------------------------------------------------


class TestShift:
    def test_positive_k_moves_firstdate_earlier(self) -> None:
        # Mirrors `shift(TSeries(2020Q1, 1:4), 1)` from the Julia docstring.
        t = TSeries(qq(2020, 1), np.arange(1, 5, dtype=float))
        s = shift(t, 1)
        assert s.firstdate == qq(2019, 4)
        assert np.array_equal(s.values, [1.0, 2.0, 3.0, 4.0])

    def test_negative_k_moves_firstdate_later(self) -> None:
        # Mirrors `shift(TSeries(2020Q1, 1:4), -1)`.
        t = TSeries(qq(2020, 1), np.arange(1, 5, dtype=float))
        s = shift(t, -1)
        assert s.firstdate == qq(2020, 2)
        assert np.array_equal(s.values, [1.0, 2.0, 3.0, 4.0])

    def test_zero_shift_returns_equal(self) -> None:
        t = TSeries(yy(2000), np.arange(5, dtype=float))
        s = shift(t, 0)
        assert s.equals(t)

    def test_does_not_alias_source(self) -> None:
        t = TSeries(yy(2000), np.arange(4, dtype=float))
        s = shift(t, 1)
        s.values[0] = 999.0
        assert t.values[0] == 0.0

    def test_iris_shift_equals_julia(self) -> None:
        # @testset "Iris" line: shift(x, 1) == TSeries(qq(2019, 4), zeros(3))
        x = TSeries(qq(2020, 1), np.zeros(3))
        assert shift(x, 1).equals(TSeries(qq(2019, 4), np.zeros(3)))

    def test_inplace_mutates(self) -> None:
        # shift!(x, 1) followed by x == TSeries(qq(2019, 4), zeros(3))
        x = TSeries(qq(2020, 1), np.zeros(3))
        original_buffer = x.values
        result = shift_inplace(x, 1)
        assert result is x
        assert x.firstdate == qq(2019, 4)
        assert x.values is original_buffer

    def test_works_on_unit_frequency(self) -> None:
        rng = MITRange(MIT(Unit(), 1), MIT(Unit(), 4))
        t = TSeries(rng, np.arange(4.0))
        s = shift(t, 2)
        assert s.firstdate == MIT(Unit(), -1)
        assert s.values[0] == 0.0


# ---------------------------------------------------------------------------
# lag / lead
# ---------------------------------------------------------------------------


class TestLagLead:
    def test_default_lag_is_shift_minus_one(self) -> None:
        # Mirrors TS.math testset:
        # y = cumsum(x); y1 = lag(y); rangeof(y1) == 1 .+ rangeof(y)
        x = TSeries(yy(2000), np.ones(11))
        y = TSeries(yy(2000), np.cumsum(x.values))
        y1 = lag(y)
        assert y1.firstdate == y.firstdate + 1
        assert y1.range.last() == y.range.last() + 1
        assert np.array_equal(y1.values, y.values)

    def test_lag_inplace_mutates(self) -> None:
        # lag!(y); y1 == y && y1 !== y
        x = TSeries(yy(2000), np.ones(11))
        y = TSeries(yy(2000), np.cumsum(x.values))
        y1 = lag(y)
        z = y
        result = lag_inplace(y)
        assert result is y
        assert z is y
        assert y1.equals(y)
        assert y1 is not y

    def test_lead_with_k(self) -> None:
        # y2 = lead(y, 2); rangeof(y2) == -2 .+ rangeof(y)
        x = TSeries(yy(2000), np.ones(11))
        y = TSeries(yy(2000), np.cumsum(x.values))
        lag_inplace(y)
        y2 = lead(y, 2)
        assert y2.firstdate == y.firstdate - 2
        assert y2.range.last() == y.range.last() - 2
        assert np.array_equal(y2.values, y.values)

    def test_lead_inplace_with_k(self) -> None:
        # lead!(y, 3); lead(y2) == y
        x = TSeries(yy(2000), np.ones(11))
        y = TSeries(yy(2000), np.cumsum(x.values))
        lag_inplace(y)
        y2 = lead(y, 2)
        lead_inplace(y, 3)
        assert lead(y2).equals(y)
        assert lead(y2) is not y

    def test_lag_lead_inverse(self) -> None:
        t = TSeries(mm(2020, 6), np.arange(10.0))
        assert lag(lead(t)).equals(t)
        assert lead(lag(t, 4), 4).equals(t)


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


class TestDiff:
    def test_default_first_difference(self) -> None:
        # pct testset line: diff(t1).values == [1, 2, 4]; rangeof == 2001Y:2003Y
        t1 = TSeries(yy(2000), np.asarray([1.0, 2.0, 4.0, 8.0]))
        d = diff(t1)
        assert np.array_equal(d.values, [1.0, 2.0, 4.0])
        assert d.range == MITRange(yy(2001), yy(2003))

    def test_diff_k_negative_2(self) -> None:
        t1 = TSeries(yy(2000), np.asarray([1.0, 2.0, 4.0, 8.0]))
        d = diff(t1, -2)
        # diff(t,k) = t - shift(t,k); shift(t,-2) starts at 2002Y.
        # Intersection 2002Y:2003Y. t[2002..2003]=[4,8], shift[2002..2003]=[1,2].
        assert np.array_equal(d.values, [3.0, 6.0])
        assert d.range == MITRange(yy(2002), yy(2003))

    def test_diff_k_positive(self) -> None:
        t = TSeries(yy(2000), np.asarray([0.0, 1.0, 4.0, 9.0, 16.0]))
        d = diff(t, 1)
        # shift(t,1) starts at 1999Y; intersection 2000..2003 vs 1999..2003 = 2000..2003
        # t[2000..2003]=[0,1,4,9]; shift[2000..2003]=[1,4,9,16]; diff = -1,-3,-5,-7
        assert np.array_equal(d.values, [-1.0, -3.0, -5.0, -7.0])
        assert d.range == MITRange(yy(2000), yy(2003))


# ---------------------------------------------------------------------------
# pct
# ---------------------------------------------------------------------------


class TestPct:
    def test_pct_simple(self) -> None:
        t1 = TSeries(yy(2000), np.asarray([1.0, 2.0, 4.0, 8.0]))
        p = pct(t1)
        assert np.array_equal(p.values, [100.0, 100.0, 100.0])
        assert p.range == MITRange(yy(2001), yy(2003))

    def test_pct_k_minus_2(self) -> None:
        t1 = TSeries(yy(2000), np.asarray([1.0, 2.0, 4.0, 8.0]))
        p = pct(t1, -2)
        assert np.array_equal(p.values, [300.0, 300.0])
        assert p.range == MITRange(yy(2002), yy(2003))

    def test_pct_islog(self) -> None:
        t2 = TSeries(yy(2000), np.log(np.asarray([1.0, 2.0, 4.0, 8.0])))
        p = pct(t2, islog=True)
        np.testing.assert_allclose(p.values, [100.0, 100.0, 100.0])
        assert p.range == MITRange(yy(2001), yy(2003))

    def test_pct_quarterly(self) -> None:
        t3 = TSeries(qq(2000, 1), 2.0 ** np.arange(1, 21))
        p = pct(t3)
        # First three values: each doubles, so pct = 100.
        assert np.allclose(p.values[:3], [100.0, 100.0, 100.0])


# ---------------------------------------------------------------------------
# apct
# ---------------------------------------------------------------------------


class TestApct:
    def test_apct_quarterly(self) -> None:
        # apct(t3).values[1:3] == [1500, 1500, 1500]; rangeof == 2000Q2:2004Q4
        t3 = TSeries(qq(2000, 1), 2.0 ** np.arange(1, 21))
        a = apct(t3)
        assert np.allclose(a.values[:3], [1500.0, 1500.0, 1500.0])
        assert a.range == MITRange(qq(2000, 2), qq(2004, 4))

    def test_apct_monthly(self) -> None:
        # apct(t4).values[1:3] == [409500, 409500, 409500]; rangeof == 2000M2:2001M8
        t4 = TSeries(mm(2000, 1), 2.0 ** np.arange(1, 21))
        a = apct(t4)
        assert np.allclose(a.values[:3], [409500.0, 409500.0, 409500.0])
        assert a.range == MITRange(mm(2000, 2), mm(2001, 8))

    def test_apct_islog(self) -> None:
        # apct(t5, true).values[1:3] ≈ [1500, 1500, 1500]
        t5 = TSeries(qq(2000, 1), np.log(2.0 ** np.arange(1, 21)))
        a = apct(t5, True)
        assert np.allclose(a.values[:3], [1500.0, 1500.0, 1500.0])
        assert a.range == MITRange(qq(2000, 2), qq(2004, 4))

    def test_apct_rejects_unit_frequency(self) -> None:
        t = TSeries(MITRange(MIT(Unit(), 1), MIT(Unit(), 5)), np.arange(5.0))
        with pytest.raises(TypeError, match="YPFrequency"):
            apct(t)


# ---------------------------------------------------------------------------
# ytypct
# ---------------------------------------------------------------------------


class TestYtypct:
    def test_ytypct_yearly(self) -> None:
        # ytypct(t1).values[1:3] == [100, 100, 100]; rangeof == 2001Y:2003Y
        t1 = TSeries(yy(2000), np.asarray([1.0, 2.0, 4.0, 8.0]))
        y = ytypct(t1)
        assert np.allclose(y.values[:3], [100.0, 100.0, 100.0])
        assert y.range == MITRange(yy(2001), yy(2003))

    def test_ytypct_quarterly(self) -> None:
        # ytypct(t3).values[1:3] == [1500, 1500, 1500]; rangeof == 2001Q1:2004Q4
        t3 = TSeries(qq(2000, 1), 2.0 ** np.arange(1, 21))
        y = ytypct(t3)
        assert np.allclose(y.values[:3], [1500.0, 1500.0, 1500.0])
        assert y.range == MITRange(qq(2001, 1), qq(2004, 4))

    def test_ytypct_monthly(self) -> None:
        # ytypct(t4).values[1:3] == [409500, 409500, 409500]; rangeof == 2001M1:2001M8
        t4 = TSeries(mm(2000, 1), 2.0 ** np.arange(1, 21))
        y = ytypct(t4)
        assert np.allclose(y.values[:3], [409500.0, 409500.0, 409500.0])
        assert y.range == MITRange(mm(2001, 1), mm(2001, 8))

    def test_ytypct_rejects_unit_frequency(self) -> None:
        t = TSeries(MITRange(MIT(Unit(), 1), MIT(Unit(), 12)), np.arange(12.0))
        with pytest.raises(TypeError, match="YPFrequency"):
            ytypct(t)


# ---------------------------------------------------------------------------
# Cross-references — sanity checks between functions
# ---------------------------------------------------------------------------


class TestCrossReferences:
    def test_lag_equals_shift_minus(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(8.0))
        assert lag(t, 3).equals(shift(t, -3))

    def test_lead_equals_shift_plus(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(8.0))
        assert lead(t, 2).equals(shift(t, 2))

    def test_diff_pct_relationship(self) -> None:
        t = TSeries(yy(2000), np.asarray([10.0, 20.0, 40.0, 80.0]))
        d = diff(t)
        p = pct(t)
        # pct = diff / shift * 100; both have the same range.
        shifted = shift(t, -1)[d.range]
        np.testing.assert_allclose(p.values, (d.values / shifted.values) * 100)

    def test_apct_quarterly_uses_n_equals_4(self) -> None:
        # Quarterly => ppy=4 => apct = ((a/b)^4 - 1) * 100.
        # Verify on a doubling series so the exponent is observable.
        t = TSeries(qq(2010, 1), np.asarray([1.0, 2.0]))
        a = apct(t)
        assert np.allclose(a.values, [(2.0**4 - 1.0) * 100.0])

    def test_frequency_propagates(self) -> None:
        # Input starts at 1.0 (not 0.0) so pct's t / lag(t) division
        # never sees a zero divisor — this test asserts the frequency
        # *tag* propagates, not divide-by-zero semantics. The deliberate
        # divide-by-zero contract lives in
        # TestPctZeroSemantics::test_pct_emits_runtimewarning_on_zero.
        t = TSeries(qq(2020, 1), np.arange(1.0, 9.0))
        assert isinstance(shift(t, 1).frequency, Quarterly)
        assert isinstance(pct(t).frequency, Quarterly)
        assert isinstance(apct(t).frequency, Quarterly)
        assert isinstance(ytypct(t).frequency, Quarterly)

    def test_apct_monthly_uses_n_equals_12(self) -> None:
        t = TSeries(mm(2020, 1), np.asarray([1.0, 2.0]))
        a = apct(t)
        assert isinstance(a.frequency, Monthly)
        assert np.allclose(a.values, [(2.0**12 - 1.0) * 100.0])

    def test_apct_yearly_uses_n_equals_1(self) -> None:
        t = TSeries(yy(2020), np.asarray([1.0, 2.0]))
        a = apct(t)
        assert isinstance(a.frequency, Yearly)
        assert np.allclose(a.values, [100.0])


# ---------------------------------------------------------------------------
# pct divide-by-zero contract — locks in the library decision documented
# in claude_files/paper/NOTES.md "Performance / NumPy semantics".
# ---------------------------------------------------------------------------


class TestPctZeroSemantics:
    """Locks in the user-facing contract: ``pct()`` does NOT silence NumPy's
    ``RuntimeWarning: divide by zero`` when the input contains a zero. We
    preserve NumPy semantics rather than pandas-style "raise on zero" so
    users who already understand NumPy aren't re-trained.

    See ``claude_files/paper/NOTES.md`` "Performance / NumPy semantics"
    for the design rationale. This test is the executable counterpart of
    that paper note — if a future change accidentally silences the
    warning (e.g. by wrapping pct in ``np.errstate(divide='ignore')``),
    this test fires.
    """

    def test_pct_emits_runtimewarning_on_zero(self) -> None:
        # Input contains a zero divisor: pct = (t[i] / t[i-1] - 1) * 100,
        # so the second element divides 1.0 / 0.0 -> inf, and NumPy emits
        # RuntimeWarning. We assert the warning is observable to callers.
        t = TSeries(qq(2020, 1), np.asarray([0.0, 1.0, 2.0, 3.0]))
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            result = pct(t)
        # Sanity: the inf landed where we expected; subsequent values are
        # finite. (Range is 3 long because pct drops the first element.)
        assert np.isinf(result.values[0])
        assert np.isfinite(result.values[1:]).all()

    def test_pct_on_clean_input_emits_no_warning(self) -> None:
        # Non-zero divisors -> no warning. Complementary to the above:
        # silence is the contract for clean data.
        t = TSeries(qq(2020, 1), np.asarray([1.0, 2.0, 4.0, 8.0]))
        # pytest's filterwarnings=error::RuntimeWarning policy (pyproject)
        # would already fail this test if pct emitted a warning here;
        # the explicit assertion documents the intent.
        result = pct(t)
        np.testing.assert_allclose(result.values, [100.0, 100.0, 100.0])


# ---------------------------------------------------------------------------
# moving / moving_average / moving_sum
# ---------------------------------------------------------------------------


class TestMoving:
    """Ports the Julia ``test_mvtseries.jl`` "moving average" cases.

    Builds ``x = TSeries(1U, 1:10)`` and walks through the windows the Julia
    tests exercise: 1 / 4 / 2 forward, then 1 / -4 / -2 backward.
    """

    @pytest.fixture
    def x(self) -> TSeries:
        return TSeries(MIT(Unit(), 1), np.arange(1, 11, dtype=np.float64))

    def test_window_of_1_is_identity(self, x: TSeries) -> None:
        result = moving(x, 1)
        assert result.range == x.range
        assert np.allclose(result.values, x.values)

    def test_backward_window_of_4(self, x: TSeries) -> None:
        # Julia: x_m4.values == collect(4:10) .- 1.5
        result = moving(x, 4)
        assert result.range == MITRange(MIT(Unit(), 4), MIT(Unit(), 10))
        assert np.allclose(result.values, np.arange(4, 11) - 1.5)

    def test_forward_window_of_4(self, x: TSeries) -> None:
        # Julia: x_m4_forward.values == collect(1:7) .+ 1.5
        result = moving(x, -4)
        assert result.range == MITRange(MIT(Unit(), 1), MIT(Unit(), 7))
        assert np.allclose(result.values, np.arange(1, 8) + 1.5)

    def test_backward_window_of_2(self, x: TSeries) -> None:
        # Julia: x_m2.values == collect(2:10) .- 0.5
        result = moving(x, 2)
        assert result.range == MITRange(MIT(Unit(), 2), MIT(Unit(), 10))
        assert np.allclose(result.values, np.arange(2, 11) - 0.5)

    def test_forward_window_of_2(self, x: TSeries) -> None:
        # Julia: x_m2_forward.values == collect(1:9) .+ 0.5
        result = moving(x, -2)
        assert result.range == MITRange(MIT(Unit(), 1), MIT(Unit(), 9))
        assert np.allclose(result.values, np.arange(1, 10) + 0.5)

    def test_moving_equals_moving_average(self, x: TSeries) -> None:
        # Julia: moving(x, -4) == moving_average(x, -4)
        assert np.allclose(moving(x, -4).values, moving_average(x, -4).values)

    def test_moving_sum_is_n_times_moving_average(self, x: TSeries) -> None:
        # Julia: 4 * moving(x, -4) == moving_sum(x, -4)
        a = moving(x, -4)
        s = moving_sum(x, -4)
        assert np.allclose(4.0 * a.values, s.values)

    # -- edge cases -------------------------------------------------------

    def test_quarterly_window(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(8.0))
        ma = moving(t, 4)
        assert ma.firstdate == qq(2020, 4)
        assert ma.range == MITRange(qq(2020, 4), qq(2021, 4))
        # Mean of [0,1,2,3] = 1.5; mean of [1,2,3,4]=2.5; ...
        expected = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        assert np.allclose(ma.values, expected)

    def test_window_equals_series_length_is_single_point(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(4.0))
        ma = moving(t, 4)
        assert len(ma) == 1
        assert ma.firstdate == qq(2020, 4)
        assert np.allclose(ma.values, [1.5])

    def test_zero_window_raises(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(4.0))
        with pytest.raises(ValueError, match="nonzero"):
            moving(t, 0)

    def test_window_larger_than_series_raises(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(3.0))
        with pytest.raises(ValueError, match="exceeds series length"):
            moving(t, 5)
        with pytest.raises(ValueError, match="exceeds series length"):
            moving(t, -5)

    def test_integer_series_yields_float_result(self) -> None:
        # Matches Julia: zeros(len+1) accumulator promotes to Float64.
        t = TSeries(MIT(Unit(), 1), np.arange(1, 6, dtype=np.int64))
        ma = moving(t, 2)
        assert ma.values.dtype == np.float64
        assert np.allclose(ma.values, [1.5, 2.5, 3.5, 4.5])

    def test_does_not_alias_source(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(6.0))
        ma = moving(t, 3)
        ma.values[0] = 999.0
        assert t.values[0] == 0.0  # original untouched

    def test_moving_sum_window_of_1_is_identity(self) -> None:
        t = TSeries(qq(2020, 1), np.asarray([1.0, 2.0, 3.0, 4.0]))
        ms = moving_sum(t, 1)
        assert np.allclose(ms.values, t.values)
        assert ms.range == t.range


# ---------------------------------------------------------------------------
# undiff
# ---------------------------------------------------------------------------


class TestUndiff:
    """Ports the Julia ``test_mvtseries.jl`` undiff cases (TSeries flavor)."""

    @pytest.fixture
    def x_a(self) -> TSeries:
        # Mirrors `x.a = collect(1:10)` at `1U:10U`.
        return TSeries(MIT(Unit(), 1), np.arange(1, 11, dtype=np.int64))

    def test_undiff_diff_with_anchor_at_first_recovers_original(self, x_a: TSeries) -> None:
        # Julia: undiff(diff(x.a), 1U => x.a[1]) == x.a
        d = diff(x_a)
        u = undiff(d, (MIT(Unit(), 1), int(x_a.values[0])))
        assert u.frequency == x_a.frequency
        assert u.range == x_a.range
        assert np.allclose(u.values, x_a.values)

    def test_undiff_diff_with_float_anchor_at_first(self, x_a: TSeries) -> None:
        # Julia: undiff(diff(x.a), 1U => 1.0) == x.a
        d = diff(x_a)
        u = undiff(d, (MIT(Unit(), 1), 1.0))
        assert u.values.dtype == np.float64
        assert np.allclose(u.values, x_a.values)

    def test_undiff_with_mid_range_anchor_drops_first(self, x_a: TSeries) -> None:
        # Julia: undiff(diff(x.a), 2U => 2.0) == x.a[2U:10U]
        d = diff(x_a)
        u = undiff(d, (MIT(Unit(), 2), 2.0))
        assert u.range == MITRange(MIT(Unit(), 2), MIT(Unit(), 10))
        assert np.allclose(u.values, x_a.values[1:])

    def test_undiff_with_anchor_in_middle_offsets_uniformly(self, x_a: TSeries) -> None:
        # Julia: undiff(diff(x.a), 5U => 5.0) == x.a[2U:10U]
        d = diff(x_a)
        u = undiff(d, (MIT(Unit(), 5), 5.0))
        assert u.range == MITRange(MIT(Unit(), 2), MIT(Unit(), 10))
        assert np.allclose(u.values, x_a.values[1:])

    def test_undiff_default_anchor_is_zero(self, x_a: TSeries) -> None:
        # Julia: undiff(diff(x.a)) == collect(0:9)
        d = diff(x_a)
        u = undiff(d)
        assert u.range == MITRange(MIT(Unit(), 1), MIT(Unit(), 10))
        assert np.allclose(u.values, np.arange(0, 10))

    # -- Quarterly anchor scenarios (port from Julia, deterministic seed) --

    @pytest.fixture
    def tt(self) -> TSeries:
        rng = np.random.default_rng(seed=42)
        return TSeries(qq(2020, 1), rng.standard_normal(20))

    def test_undiff_inserts_anchor_zero_before_firstdate(self, tt: TSeries) -> None:
        # Julia: undiff(tt)[begin+1:end] ≈ cumsum(tt)
        u = undiff(tt)
        assert u.firstdate == qq(2019, 4)
        assert u.lastdate == qq(2024, 4)
        # Skip the inserted zero entry; the rest should equal cumsum(tt).
        assert np.allclose(u.values[1:], np.cumsum(tt.values))
        assert u.values[0] == 0.0

    def test_undiff_scalar_anchor_shifts_uniformly(self, tt: TSeries) -> None:
        # Julia: undiff(tt, 7) ≈ undiff(tt) .+ 7
        a = undiff(tt, 7)
        b = undiff(tt)
        assert np.allclose(a.values, b.values + 7.0)

    def test_undiff_pair_anchor_shifts_uniformly(self, tt: TSeries) -> None:
        # Julia: undiff(tt, 2020Q1 => 7) ≈ undiff(tt, 2020Q1 => 0.0) .+ 7
        a = undiff(tt, (qq(2020, 1), 7.0))
        b = undiff(tt, (qq(2020, 1), 0.0))
        assert np.allclose(a.values, b.values + 7.0)

    def test_undiff_anchor_at_first_matches_cumsum(self, tt: TSeries) -> None:
        # Julia: undiff(tt, 2020Q1 => tt[begin]) ≈ cumsum(tt)
        u = undiff(tt, (qq(2020, 1), float(tt.values[0])))
        assert u.range == tt.range
        assert np.allclose(u.values, np.cumsum(tt.values))

    def test_undiff_anchor_at_mid_with_tseries_value(self, tt: TSeries) -> None:
        # Julia: undiff(tt, 2021Q1 => tt) ≈ cumsum(tt) - cumsum(tt)[2021Q1] + tt[2021Q1]
        u = undiff(tt, (qq(2021, 1), tt))
        idx_2021q1 = qq(2021, 1).value - tt.firstdate.value
        cs = np.cumsum(tt.values)
        expected = cs - cs[idx_2021q1] + tt.values[idx_2021q1]
        assert np.allclose(u.values, expected)

    def test_undiff_anchor_at_mid_with_wider_tseries(self) -> None:
        # Julia: tt at 2020Q1:2024Q4; qq_ones at 2019Q1:2050Q4, ones.
        # undiff(tt, 2021Q1 => qq_ones) ≈ cumsum(tt) - cumsum(tt)[2021Q1] + 1
        rng = np.random.default_rng(seed=42)
        tt = TSeries(qq(2020, 1), rng.standard_normal(20))
        qq_ones = TSeries(MITRange(qq(2019, 1), qq(2050, 4)), 1.0)
        u = undiff(tt, (qq(2021, 1), qq_ones))
        idx = qq(2021, 1).value - tt.firstdate.value
        cs = np.cumsum(tt.values)
        expected = cs - cs[idx] + 1.0
        assert np.allclose(u.values, expected)

    def test_undiff_tseries_anchor_uses_default_date(self) -> None:
        # When anchor is a bare TSeries, date defaults to firstdate(dvar)-1
        # and value is anchor[that date].
        dvar = TSeries(qq(2020, 2), np.asarray([1.0, 1.0, 1.0]))
        anchor_ts = TSeries(qq(2020, 1), np.asarray([5.0]))
        u = undiff(dvar, anchor_ts)
        # Default date = 2020Q1, anchor_ts[2020Q1] = 5.0. cumsum + correction.
        assert u.firstdate == qq(2020, 1)
        # extended dvar at 2020Q1 = 0, then 1,1,1 -> cumsum [0,1,2,3], + 5 → [5,6,7,8].
        assert np.allclose(u.values, [5.0, 6.0, 7.0, 8.0])

    # -- error paths -----------------------------------------------------

    def test_undiff_mixed_freq_anchor_raises(self) -> None:
        d = TSeries(qq(2020, 1), np.asarray([1.0, 1.0]))
        bad = TSeries(yy(2020), np.asarray([1.0, 1.0]))
        with pytest.raises(TypeError, match="Mixing frequencies"):
            undiff(d, bad)

    def test_undiff_bad_anchor_tuple_raises(self) -> None:
        d = TSeries(qq(2020, 1), np.asarray([1.0, 1.0]))
        with pytest.raises(TypeError, match="tuple"):
            undiff(d, (1, 2, 3))  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="tuple"):
            undiff(d, ("not-an-MIT", 1.0))  # type: ignore[arg-type]

    def test_undiff_bare_mit_anchor_raises(self) -> None:
        d = TSeries(qq(2020, 1), np.asarray([1.0, 1.0]))
        with pytest.raises(TypeError, match="bare MIT"):
            undiff(d, qq(2020, 1))


# ---------------------------------------------------------------------------
# undiff: backward integration via anchor placement (M1.6.1)
#
# The math is direction-agnostic — ``undiff`` computes a cumulative sum
# and shifts it so ``result[anchor_date] == anchor_value``. Forward
# integration anchors before the data; backward integration anchors at
# or after the data. These tests lock the symmetric behaviour the M1.6.1
# session promised in the docstring.
# ---------------------------------------------------------------------------


class TestUndiffBackwardIntegration:
    def test_anchor_at_lastdate_recovers_backward_from_end(self) -> None:
        # dvar over [1U..5U] with values [1, 1, 1, 1, 1]; anchor at 5U=10.
        # result[5] = 10, result[4] = 10 - dvar[5] = 9, ..., result[1] = 6.
        d = TSeries(MIT(Unit(), 1), np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
        u = undiff(d, (MIT(Unit(), 5), 10.0))
        assert isinstance(u, TSeries)
        assert u.range == MITRange(MIT(Unit(), 1), MIT(Unit(), 5))
        np.testing.assert_array_equal(u.values, [6.0, 7.0, 8.0, 9.0, 10.0])

    def test_anchor_at_lastdate_plus_one_extends_with_zero(self) -> None:
        # Anchor strictly after data: dvar is extended with zeros so the
        # anchor falls inside the new range. The tail position equals the
        # anchor (flat tail because dvar[anchor_date] is filled with 0).
        d = TSeries(MIT(Unit(), 1), np.array([1.0, 1.0, 1.0]))
        u = undiff(d, (MIT(Unit(), 4), 10.0))
        assert isinstance(u, TSeries)
        assert u.range == MITRange(MIT(Unit(), 1), MIT(Unit(), 4))
        # cumsum([1, 1, 1, 0]) = [1, 2, 3, 3]; correction = 10 - 3 = 7;
        # final = [8, 9, 10, 10].
        np.testing.assert_array_equal(u.values, [8.0, 9.0, 10.0, 10.0])

    def test_undiff_inverts_diff_via_terminal_anchor(self) -> None:
        # Round-trip: t → diff(t) → undiff(diff, (t.lastdate, t.values[-1]))
        # reconstructs t on diff(t)'s range (one period shorter than t since
        # diff drops the leading period). Anchoring at the terminal value
        # locks the constant of integration to t's actual endpoint.
        rng_ = np.random.default_rng(seed=42)
        t = TSeries(qq(2020, 1), rng_.standard_normal(12))
        d = diff(t)
        u = undiff(d, (t.lastdate, float(t.values[-1])))
        assert u.range == d.range
        # Compare against t restricted to diff's range.
        np.testing.assert_allclose(u.values, t.values[1:], rtol=1e-12)

    def test_quarterly_terminal_anchor(self) -> None:
        # Quarterly: dvar = [2, -1, 0.5]; anchor at 2020Q4 = 100.
        # Working backward: result[Q4]=100, result[Q3]=100-0.5=99.5,
        # result[Q2]=99.5-(-1)=100.5, result[Q1]=100.5-2=98.5.
        d = TSeries(qq(2020, 2), np.array([2.0, -1.0, 0.5]))
        u = undiff(d, (qq(2020, 4), 100.0))
        assert isinstance(u, TSeries)
        assert u.range == MITRange(qq(2020, 2), qq(2020, 4))
        np.testing.assert_allclose(u.values, [100.5, 99.5, 100.0], rtol=1e-12)

    def test_anchor_at_default_then_at_end_differ_only_in_constant(self) -> None:
        # Both forward and backward anchoring produce series with identical
        # slope (= dvar). The only difference is the constant of integration.
        d = TSeries(MIT(Unit(), 1), np.array([1.0, 2.0, -1.0, 0.5, 3.0]))
        u_fwd = undiff(d, 0)  # default: anchor at firstdate-1=0U, value 0
        u_bwd = undiff(d, (MIT(Unit(), 5), 0.0))  # anchor at end, value 0
        # Trim u_fwd to the inner range so shapes match.
        common = u_fwd[MITRange(MIT(Unit(), 1), MIT(Unit(), 5))]
        assert isinstance(common, TSeries)
        # Their first differences are identical (== dvar).
        np.testing.assert_allclose(np.diff(common.values), np.diff(u_bwd.values))
        # And the constant offset between the two is uniform.
        offset = common.values - u_bwd.values
        np.testing.assert_allclose(offset, offset[0] * np.ones_like(offset))


# ---------------------------------------------------------------------------
# undiff_inplace
# ---------------------------------------------------------------------------


class TestUndiffInplace:
    """Ports the Julia ``undiff!`` cases on TSeries."""

    def test_no_effect_when_fromdate_in_tail(self) -> None:
        # Julia: undiff!(x2.a, diff(x.a * 3); fromdate=6U) → [1,2,3,4,5,6,9,12,15,18]
        x_a = TSeries(MIT(Unit(), 1), np.arange(1.0, 11.0))
        d = diff(TSeries(MIT(Unit(), 1), np.arange(1.0, 11.0) * 3.0))
        target = x_a.copy()
        undiff_inplace(target, d, fromdate=MIT(Unit(), 6))
        assert np.allclose(target.values, [1, 2, 3, 4, 5, 6, 9, 12, 15, 18])

    def test_fromdate_earlier_propagates_through_more_periods(self) -> None:
        # Julia: undiff!(ts1, ts2, fromdate=4U) → [1,2,3,4,7,10,13,16,19,22]
        ts1 = TSeries(MIT(Unit(), 1), np.arange(1.0, 11.0))
        ts2 = diff(TSeries(MIT(Unit(), 1), np.arange(1.0, 11.0) * 3.0))
        undiff_inplace(ts1, ts2, fromdate=MIT(Unit(), 4))
        assert np.allclose(ts1.values, [1, 2, 3, 4, 7, 10, 13, 16, 19, 22])

    def test_idempotent_when_already_applied(self) -> None:
        # Julia: after fromdate=4U, calling fromdate=8U leaves the series unchanged.
        ts1 = TSeries(MIT(Unit(), 1), np.arange(1.0, 11.0))
        ts2 = diff(TSeries(MIT(Unit(), 1), np.arange(1.0, 11.0) * 3.0))
        undiff_inplace(ts1, ts2, fromdate=MIT(Unit(), 4))
        snapshot = ts1.values.copy()
        undiff_inplace(ts1, ts2, fromdate=MIT(Unit(), 8))
        assert np.allclose(ts1.values, snapshot)

    def test_extends_var_to_cover_dvar(self) -> None:
        var = TSeries(qq(2020, 1), np.asarray([10.0]))
        dvar = TSeries(qq(2020, 2), np.asarray([1.0, 2.0, 3.0]))
        result = undiff_inplace(var, dvar)
        # Resized to cover dvar.lastdate = 2020Q4.
        assert result is var
        assert var.range == MITRange(qq(2020, 1), qq(2020, 4))
        # fromdate defaults to firstdate(dvar)-1 = 2020Q1; var[2020Q1] stays 10,
        # then cumsum: 10+1=11, 11+2=13, 13+3=16.
        assert np.allclose(var.values, [10.0, 11.0, 13.0, 16.0])

    def test_returns_same_object(self) -> None:
        var = TSeries(qq(2020, 1), np.asarray([1.0, 2.0, 3.0, 4.0]))
        dvar = diff(var)
        assert undiff_inplace(var.copy(), dvar) is not var  # different object
        # Same object on a single call:
        v = var.copy()
        assert undiff_inplace(v, dvar) is v

    def test_fromdate_before_var_firstdate_raises(self) -> None:
        var = TSeries(qq(2020, 2), np.asarray([1.0, 2.0]))
        dvar = TSeries(qq(2020, 2), np.asarray([1.0, 1.0]))
        with pytest.raises(ValueError, match="Range mismatch"):
            undiff_inplace(var, dvar, fromdate=qq(2020, 1))

    def test_mixed_freq_raises(self) -> None:
        var = TSeries(qq(2020, 1), np.asarray([1.0]))
        dvar = TSeries(yy(2020), np.asarray([1.0]))
        with pytest.raises(TypeError, match="Mixing frequencies"):
            undiff_inplace(var, dvar)


# ---------------------------------------------------------------------------
# Round-trip: diff / undiff inverse property
# ---------------------------------------------------------------------------


class TestDiffUndiffRoundTrip:
    def test_undiff_inverts_diff(self) -> None:
        rng = np.random.default_rng(seed=7)
        t = TSeries(qq(2020, 1), rng.standard_normal(12))
        d = diff(t)
        # Anchor at the first known value of t.
        u = undiff(d, (t.firstdate, float(t.values[0])))
        # Result starts at firstdate(d)-1 ... only if firstdate is outside.
        # Here anchor is at firstdate(t) = firstdate(d) is firstdate(t)+1,
        # so firstdate(t) is just before d and we extend by one period.
        assert u.range == t.range
        assert np.allclose(u.values, t.values)

    def test_undiff_inplace_inverts_diff(self) -> None:
        rng = np.random.default_rng(seed=11)
        t = TSeries(qq(2020, 1), rng.standard_normal(12))
        d = diff(t)
        # Start with the anchor value at firstdate(t), then let undiff_inplace
        # fill in the rest.
        v = TSeries(t.range, dtype=t.values.dtype)
        v.values[0] = t.values[0]
        undiff_inplace(v, d, fromdate=t.firstdate)
        assert np.allclose(v.values, t.values)
