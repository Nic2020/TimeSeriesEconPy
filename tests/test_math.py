# SPDX-License-Identifier: MIT
"""Tests for ``tsecon._math``: shift / lag / lead / diff / pct / apct / ytypct.

Ports the relevant cases from ``TimeSeriesEcon.jl``'s ``test/test_tseries.jl``
(``@testset "Iris"`` shift section, ``@testset "TS.math"`` lag/lead section,
and ``@testset "pct"``), plus Python-specific edge cases.
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
    pct,
    qq,
    shift,
    shift_inplace,
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
        t = TSeries(qq(2020, 1), np.arange(8.0))
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
