# SPDX-License-Identifier: MIT
"""Tests for the YP-only fconvert subsystem.

Mirrors the YP-relevant blocks from
``TimeSeriesEcon.jl/test/test_fconvert.jl``:

* ``@testset "fconvert, general"``
* ``@testset "fconvert, YPFrequencies, to higher"``
* ``@testset "fconvert, YPFrequencies, to lower"``
* ``@testset "fconvert, YPFrequencies, to similar"``
* ``@testset "fconvert, pass function"`` (YP combinations only)
* ``@testset "fconvert, pass custom function"``
* ``@testset "fconvert, all combinations"`` (YP combinations only)
* ``@testset "extend series"``

Calendar-frequency (Daily / BDaily / Weekly) blocks are deferred to the
follow-up session; their absence here is intentional, not a coverage gap
within the YP-only scope.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsecon import (
    MIT,
    HalfYearly,
    MITRange,
    Monthly,
    Quarterly,
    TSeries,
    Unit,
    Yearly,
    extend_series,
    fconvert,
    fconvert_parts,
    fconvert_range,
    fconvert_tseries,
    mitrange,
    mm,
    qq,
    strip_tseries,
    strip_tseries_inplace,
    trim_series,
    yy,
)
from tsecon.fconvert import divide_uneven, linear_uneven, repeat_uneven

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def hh(year: int, half: int) -> MIT:
    """HalfYearly literal helper (no upstream `hh` shorthand)."""
    return MIT.from_yp(HalfYearly(), year, half)


# ---------------------------------------------------------------------------
# General behaviour (mirrors Julia ``@testset "fconvert, general"``)
# ---------------------------------------------------------------------------


class TestFconvertGeneral:
    def test_unit_to_unit_is_identity(self):
        t = TSeries(MIT(Unit(), 5), np.arange(1, 11, dtype=np.int64))
        assert fconvert(Unit, t) is t

    def test_unit_target_from_yp_raises(self):
        q = TSeries(qq(5, 1), np.arange(1, 11, dtype=np.float64))
        with pytest.raises(ValueError, match="Unit"):
            fconvert(Unit, q)

    def test_yp_target_from_unit_raises(self):
        t = TSeries(MIT(Unit(), 5), np.arange(1, 11, dtype=np.int64))
        with pytest.raises(ValueError, match="Unit"):
            fconvert(Quarterly, t)

    def test_quarterly_to_monthly_const(self):
        q = TSeries(qq(5, 1), np.arange(1, 11, dtype=np.float64))
        mq = fconvert(Monthly, q, method="const")
        assert mq.frequency == Monthly()
        np.testing.assert_array_equal(mq.values, np.repeat(np.arange(1.0, 11.0), 3))

    def test_quarterly_to_yearly_default_is_mean(self):
        q = TSeries(qq(5, 1), np.arange(1, 11, dtype=np.float64))
        yq = fconvert(Yearly, q)
        assert yq.frequency == Yearly()
        np.testing.assert_array_equal(yq.values, np.array([2.5, 6.5]))

    def test_quarterly_to_yearly_method_variants(self):
        q = TSeries(qq(5, 1), np.arange(1, 11, dtype=np.float64))
        eq = np.testing.assert_array_equal
        eq(fconvert(Yearly, q, method="point", ref="end").values, [4.0, 8.0])
        eq(fconvert(Yearly, q, method="end").values, [4.0, 8.0])
        eq(fconvert(Yearly, q, method="point", ref="begin").values, [1.0, 5.0, 9.0])
        eq(fconvert(Yearly, q, method="begin").values, [1.0, 5.0, 9.0])
        eq(fconvert(Yearly, q, method="sum").values, [10.0, 26.0])

    def test_halfyearly_to_yearly(self):
        h = TSeries(hh(5, 1), np.arange(1, 11, dtype=np.float64))
        yh = fconvert(Yearly, h)
        eq = np.testing.assert_array_equal
        assert yh.frequency == Yearly()
        eq(yh.values, [1.5, 3.5, 5.5, 7.5, 9.5])
        end_vals = [2.0, 4.0, 6.0, 8.0, 10.0]
        begin_vals = [1.0, 3.0, 5.0, 7.0, 9.0]
        eq(fconvert(Yearly, h, method="point", ref="end").values, end_vals)
        eq(fconvert(Yearly, h, method="end").values, end_vals)
        eq(fconvert(Yearly, h, method="point", ref="begin").values, begin_vals)
        eq(fconvert(Yearly, h, method="begin").values, begin_vals)
        eq(fconvert(Yearly, h, method="sum").values, [3.0, 7.0, 11.0, 15.0, 19.0])

    def test_partial_starting_period_truncation_monthly_to_yearly(self):
        # Mirrors the Julia ``for i = 1:11`` loop on M-to-Y truncation.
        # ``1M1 .+ (i:50)`` is a range from ``1M1+i`` to ``1M1+50``; Python's
        # equivalent uses ``mitrange`` and a zero-filled TSeries.
        for i in range(1, 12):
            rng_a = mitrange(mm(1, 1) + i, mm(1, 1) + 50)
            t = TSeries(rng_a, np.zeros(len(rng_a)))
            assert fconvert(Yearly, t, method="mean").range == mitrange(yy(2), yy(4))
            rng_b = mitrange(mm(1, 1), mm(1, 1) + 47 + i)
            t2 = TSeries(rng_b, np.zeros(len(rng_b)))
            assert fconvert(Yearly, t2).range == mitrange(yy(1), yy(4))

    def test_partial_starting_period_truncation_quarterly_to_yearly(self):
        for i in range(1, 4):
            rng_a = mitrange(qq(1, 1) + i, qq(1, 1) + 50)
            t = TSeries(rng_a, np.zeros(len(rng_a)))
            assert fconvert(Yearly, t, method="mean").range == mitrange(yy(2), yy(12))
            rng_b = mitrange(qq(1, 1), qq(1, 1) + 47 + i)
            t2 = TSeries(rng_b, np.zeros(len(rng_b)))
            assert fconvert(Yearly, t2).range == mitrange(yy(1), yy(12))

    def test_partial_starting_period_truncation_quarterly_from_monthly(self):
        # Mirrors ``rangeof(fconvert(Quarterly, TSeries(1M1 .+ (i:50)),
        # method=:mean)) == 1Q2+div(i-1, 3):5Q1``.
        for i in range(1, 12):
            rng_a = mitrange(mm(1, 1) + i, mm(1, 1) + 50)
            t = TSeries(rng_a, np.zeros(len(rng_a)))
            start_offset = (i - 1) // 3
            expected = mitrange(qq(1, 2) + start_offset, qq(5, 1))
            assert fconvert(Quarterly, t, method="mean").range == expected

    def test_partial_starting_period_truncation_halfyearly_to_yearly(self):
        for i in range(1, 3):
            rng_a = mitrange(hh(1, 1) + i, hh(1, 1) + 50)
            t = TSeries(rng_a, np.zeros(len(rng_a)))
            assert fconvert(Yearly, t, method="mean").range == mitrange(yy(2), yy(25))

    def test_wrong_method_for_higher_conversion_raises(self):
        q = TSeries(qq(5, 1), np.arange(1, 11, dtype=np.float64))
        with pytest.raises(ValueError, match="higher frequency"):
            fconvert(Monthly, q, method="mean")

    def test_same_frequency_identity(self):
        # MIT, MITRange, TSeries — same-frequency call returns the input.
        assert fconvert(Monthly, mm(1, 1)) == mm(1, 1)
        rng = mitrange(mm(1, 1), mm(1, 5))
        assert fconvert(Monthly, rng) == rng
        ts = TSeries(mm(1, 1), np.arange(1, 5, dtype=np.float64))
        assert fconvert(Monthly, ts) is ts


# ---------------------------------------------------------------------------
# YP → higher (mirrors Julia ``@testset "fconvert, YPFrequencies, to higher"``)
# ---------------------------------------------------------------------------


class TestYPHigher:
    def test_yearly_to_quarterly_default_endperiods(self):
        y1 = TSeries(MIT(Yearly(), 22), np.array([1, 2]))
        q1 = fconvert(Quarterly, y1)
        assert q1.range == mitrange(qq(22, 1), qq(23, 4))
        np.testing.assert_array_equal(q1.values, [1, 1, 1, 1, 2, 2, 2, 2])

        q1_begin = fconvert(Quarterly, y1, ref="begin")
        assert q1_begin.range == mitrange(qq(22, 1), qq(23, 4))
        np.testing.assert_array_equal(q1_begin.values, [1, 1, 1, 1, 2, 2, 2, 2])

        r1 = fconvert(Quarterly, y1.range, trim="end")
        assert r1 == mitrange(qq(22, 1), qq(23, 4))

    def test_yearly_july_to_quarterly_default(self):
        y2 = TSeries(MIT(Yearly(end_month=7), 22), np.array([1, 2]))
        q2 = fconvert(Quarterly, y2)
        np.testing.assert_array_equal(q2.values, [1, 1, 1, 1, 2, 2, 2, 2])
        assert q2.range == mitrange(qq(21, 3), qq(23, 2))

        q2_begin = fconvert(Quarterly, y2, ref="begin")
        assert q2_begin.range == mitrange(qq(21, 4), qq(23, 3))
        np.testing.assert_array_equal(q2_begin.values, [1, 1, 1, 1, 2, 2, 2, 2])

    def test_yearly_july_to_quarterly_january(self):
        y3 = TSeries(MIT(Yearly(end_month=7), 22), np.array([1, 2]))
        q1 = Quarterly(end_month=1)
        q3 = fconvert(q1, y3)
        assert q3.range == MITRange(MIT(q1, 21 * 4 + 3), MIT(q1, 23 * 4 + 2))
        np.testing.assert_array_equal(q3.values, [1, 1, 1, 1, 2, 2, 2, 2])

        q3_begin = fconvert(q1, y3, ref="begin")
        assert q3_begin.range == MITRange(MIT(q1, 21 * 4 + 3), MIT(q1, 23 * 4 + 2))

    def test_yearly_to_monthly(self):
        y4 = TSeries(MIT(Yearly(), 22), np.array([1, 2]))
        m1 = fconvert(Monthly, y4)
        assert m1.range == mitrange(mm(22, 1), mm(23, 12))
        np.testing.assert_array_equal(m1.values, [1] * 12 + [2] * 12)

    def test_yearly_july_to_monthly(self):
        y5 = TSeries(MIT(Yearly(end_month=7), 22), np.array([1, 2]))
        m2 = fconvert(Monthly, y5)
        assert m2.range == mitrange(mm(21, 8), mm(23, 7))
        np.testing.assert_array_equal(m2.values, [1] * 12 + [2] * 12)

    def test_yearly_to_monthly_linear_begin(self):
        y6 = TSeries(MIT(Yearly(), 2022), np.arange(1, 5, dtype=np.float64))
        m6_begin = fconvert(Monthly, y6, method="linear", ref="begin")
        assert m6_begin.range == mitrange(mm(2022, 1), mm(2025, 12))
        # Spot-check a handful of cells from the Julia reference vector.
        np.testing.assert_allclose(
            m6_begin.values[:13],
            [
                1.0,
                1.0833333333333333,
                1.1666666666666667,
                1.25,
                1.3333333333333335,
                1.4166666666666665,
                1.5,
                1.5833333333333335,
                1.6666666666666665,
                1.75,
                1.8333333333333335,
                1.9166666666666665,
                2.0,
            ],
        )

    def test_yearly_to_monthly_linear_end(self):
        y6 = TSeries(MIT(Yearly(), 2022), np.arange(1, 5, dtype=np.float64))
        m6_end = fconvert(Monthly, y6, method="linear", ref="end")
        assert m6_end.range == mitrange(mm(2022, 1), mm(2025, 12))
        np.testing.assert_allclose(
            m6_end.values[:13],
            [
                1 / 12,
                2 / 12,
                3 / 12,
                4 / 12,
                5 / 12,
                6 / 12,
                7 / 12,
                8 / 12,
                9 / 12,
                10 / 12,
                11 / 12,
                1.0,
                13 / 12,
            ],
        )

    def test_quarterly_to_monthly_linear(self):
        q7 = TSeries(qq(2022, 1), np.arange(1, 13, dtype=np.float64))
        m7_begin = fconvert(Monthly, q7, method="linear", ref="begin")
        assert m7_begin.range == mitrange(mm(2022, 1), mm(2024, 12))
        np.testing.assert_allclose(m7_begin.values[:4], [1.0, 4 / 3, 5 / 3, 2.0])

        m7_end = fconvert(Monthly, q7, method="linear", ref="end")
        assert m7_end.range == mitrange(mm(2022, 1), mm(2024, 12))
        np.testing.assert_allclose(m7_end.values[:4], [1 / 3, 2 / 3, 1.0, 4 / 3])

    def test_halfyearly_to_monthly_const_even_linear(self):
        h8 = TSeries(hh(2022, 1), np.arange(1, 6, dtype=np.float64))
        m8 = fconvert(Monthly, h8)
        assert m8.range == mitrange(mm(2022, 1), mm(2024, 6))
        np.testing.assert_array_equal(
            m8.values,
            [1] * 6 + [2] * 6 + [3] * 6 + [4] * 6 + [5] * 6,
        )
        m8_even = fconvert(Monthly, h8, method="even")
        np.testing.assert_allclose(
            m8_even.values,
            np.array([1] * 6 + [2] * 6 + [3] * 6 + [4] * 6 + [5] * 6, dtype=np.float64) / 6,
        )
        m8_lin_begin = fconvert(Monthly, h8, method="linear", ref="begin")
        np.testing.assert_allclose(
            m8_lin_begin.values[:13],
            [
                1,
                1 + 1 / 6,
                1 + 2 / 6,
                1 + 3 / 6,
                1 + 4 / 6,
                1 + 5 / 6,
                2,
                2 + 1 / 6,
                2 + 2 / 6,
                2 + 3 / 6,
                2 + 4 / 6,
                2 + 5 / 6,
                3,
            ],
        )


# ---------------------------------------------------------------------------
# YP → lower (mirrors Julia ``@testset "fconvert, YPFrequencies, to lower"``)
# ---------------------------------------------------------------------------


class TestYPLower:
    def test_quarterly_to_yearly_mean(self):
        vals = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
        q1 = TSeries(qq(1, 2), np.array(vals, dtype=np.float64))
        y1 = fconvert(Yearly, q1, method="mean")
        assert y1.range == mitrange(yy(2), yy(4))
        np.testing.assert_array_equal(y1.values, [3, 5, 7])
        assert fconvert(Yearly, q1.range) == mitrange(yy(2), yy(4))

    def test_quarterly_february_to_yearly_with_intermediate_monthly(self):
        q_feb = Quarterly(end_month=2)
        vals = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]
        q2 = TSeries(MIT(q_feb, 9), np.array(vals, dtype=np.float64))
        y2 = fconvert(Yearly, q2, method="mean")
        assert y2.range == mitrange(yy(3), yy(5))
        np.testing.assert_array_equal(y2.values, [3, 5, 7])
        # Monthly intermediate matches Julia value
        y2m = fconvert(Yearly, fconvert(Monthly, q2), method="mean")
        np.testing.assert_allclose(y2m.values, [3 + 1 / 6, 5 + 1 / 6, 7 + 1 / 6])

    def test_partial_quarterly_to_yearly(self):
        vals = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8]
        q3 = TSeries(qq(1, 2), np.array(vals, dtype=np.float64))
        y3 = fconvert(Yearly, q3, method="mean")
        assert y3.range == mitrange(yy(2), yy(4))
        np.testing.assert_array_equal(y3.values, [3, 5, 7])

    def test_monthly_to_yearly_default(self):
        m5 = TSeries(mm(20, 1), np.arange(1, 37, dtype=np.float64))
        y5 = fconvert(Yearly, m5, method="mean")
        assert y5.range == mitrange(yy(20), yy(22))
        np.testing.assert_array_equal(y5.values, [6.5, 18.5, 30.5])

    def test_monthly_to_yearly_september(self):
        m6 = TSeries(mm(20, 1), np.arange(1, 37, dtype=np.float64))
        y_sep = Yearly(end_month=9)
        y6 = fconvert(y_sep, m6, method="mean")
        assert y6.range == MITRange(MIT(y_sep, 21), MIT(y_sep, 22))
        np.testing.assert_array_equal(y6.values, [15.5, 27.5])

    def test_monthly_to_quarterly_default(self):
        m7 = TSeries(mm(20, 1), np.arange(1, 37, dtype=np.float64))
        q7 = fconvert(Quarterly, m7, method="mean")
        assert q7.range == mitrange(qq(20, 1), qq(22, 4))
        np.testing.assert_array_equal(q7.values, np.arange(2, 36, 3))

    def test_monthly_to_quarterly_february(self):
        q_feb = Quarterly(end_month=2)
        m8 = TSeries(mm(20, 1), np.arange(1, 37, dtype=np.float64))
        q8 = fconvert(q_feb, m8, method="mean")
        assert q8.range == MITRange(MIT(q_feb, 20 * 4 + 1), MIT(q_feb, 22 * 4 + 3))
        np.testing.assert_array_equal(q8.values, np.arange(4, 35, 3))

    def test_quarterly_to_yearly_point_ref_begin_and_end(self):
        q9 = TSeries(qq(2022, 1), np.arange(1, 11, dtype=np.float64))
        y9_begin = fconvert(Yearly, q9, method="point", ref="begin")
        assert y9_begin.range == mitrange(yy(2022), yy(2024))
        np.testing.assert_array_equal(y9_begin.values, [1, 5, 9])
        assert fconvert(Yearly, q9, method="begin").equals(y9_begin)

        y9_end = fconvert(Yearly, q9, method="point", ref="end")
        assert y9_end.range == mitrange(yy(2022), yy(2023))
        np.testing.assert_array_equal(y9_end.values, [4, 8])
        assert fconvert(Yearly, q9, method="end").equals(y9_end)

    def test_quarterly_to_yearly_august_point(self):
        q9 = TSeries(qq(2022, 1), np.arange(1, 11, dtype=np.float64))
        y_aug = Yearly(end_month=8)
        y9_begin = fconvert(y_aug, q9, method="point", ref="begin")
        assert y9_begin.range == MITRange(MIT(y_aug, 2023), MIT(y_aug, 2024))
        np.testing.assert_array_equal(y9_begin.values, [3, 7])
        y9_end = fconvert(y_aug, q9, method="point", ref="end")
        assert y9_end.range == MITRange(MIT(y_aug, 2022), MIT(y_aug, 2024))
        np.testing.assert_array_equal(y9_end.values, [2, 6, 10])

    def test_quarterly_to_yearly_november_mean_ref_begin_vs_end(self):
        q10 = TSeries(qq(2022, 2), np.arange(2, 12, dtype=np.float64))
        y_nov = Yearly(end_month=11)
        y10_end = fconvert(y_nov, q10, method="mean", ref="end")
        assert y10_end.range == MITRange(MIT(y_nov, 2023), MIT(y_nov, 2024))
        np.testing.assert_array_equal(y10_end.values, [5.5, 9.5])
        y10_begin = fconvert(y_nov, q10, method="mean", ref="begin")
        assert y10_begin.range == MITRange(MIT(y_nov, 2023), MIT(y_nov, 2023))
        np.testing.assert_array_equal(y10_begin.values, [6.5])


# ---------------------------------------------------------------------------
# YP → similar (mirrors Julia ``@testset "fconvert, YPFrequencies, to similar"``)
# ---------------------------------------------------------------------------


class TestYPSimilar:
    def test_quarterly_february_to_default_quarterly_point_end_and_begin(self):
        q_feb = Quarterly(end_month=2)
        qs1 = TSeries(MIT(q_feb, 2022 * 4 + 1), np.arange(2.0, 5.0))
        qs2 = fconvert(Quarterly, qs1, method="point", ref="end")
        assert qs2.range == mitrange(qq(2022, 2), qq(2022, 4))
        np.testing.assert_array_equal(qs2.values, [2.0, 3.0, 4.0])
        assert fconvert(Quarterly, qs1, method="end").equals(qs2)

        qs3 = fconvert(Quarterly, qs1, method="point", ref="begin")
        assert qs3.range == mitrange(qq(2022, 2), qq(2022, 4))
        np.testing.assert_array_equal(qs3.values, [2.0, 3.0, 4.0])
        assert fconvert(Quarterly, qs1, method="begin").equals(qs3)


# ---------------------------------------------------------------------------
# MIT-only conversions
# ---------------------------------------------------------------------------


class TestFconvertMIT:
    def test_yearly_to_quarterly_ref_end(self):
        # 22Y end-of-year aligns with 22Q4 in default Quarterly{3}.
        assert fconvert(Quarterly, MIT(Yearly(), 22)) == qq(22, 4)

    def test_yearly_to_quarterly_ref_begin(self):
        assert fconvert(Quarterly, MIT(Yearly(), 22), ref="begin") == qq(22, 1)

    def test_yearly_to_monthly_default(self):
        assert fconvert(Monthly, MIT(Yearly(), 22)) == mm(22, 12)
        assert fconvert(Monthly, MIT(Yearly(), 22), ref="begin") == mm(22, 1)

    def test_quarterly_to_yearly_default(self):
        # 22Q3 ends in September; default Yearly end_month=12 covers Jan-Dec of
        # year 22, so the end-aligned conversion lands in 22Y.
        assert fconvert(Yearly, qq(22, 3)) == yy(22)

    def test_unit_target_raises(self):
        with pytest.raises(ValueError, match="Unit"):
            fconvert(Unit, mm(1, 1))


# ---------------------------------------------------------------------------
# MITRange-only conversions
# ---------------------------------------------------------------------------


class TestFconvertMITRange:
    def test_yearly_to_quarterly(self):
        rng = fconvert(Quarterly, mitrange(yy(2022), yy(2024)))
        assert rng == mitrange(qq(2022, 1), qq(2024, 4))

    def test_monthly_to_quarterly_trim_variants(self):
        rng = mitrange(mm(2022, 2), mm(2022, 7))
        assert fconvert(Quarterly, rng, trim="begin") == mitrange(qq(2022, 2), qq(2022, 3))
        assert fconvert(Quarterly, rng, trim="end") == mitrange(qq(2022, 1), qq(2022, 2))
        assert fconvert(Quarterly, rng, trim="both") == mitrange(qq(2022, 2), qq(2022, 2))

    def test_invalid_trim_raises(self):
        with pytest.raises(ValueError, match="trim argument"):
            fconvert(Quarterly, mitrange(mm(1, 1), mm(1, 12)), trim="bogus")

    def test_parts_six_tuple(self):
        # parts=True returns the six-tuple used internally by the TSeries path.
        result = fconvert_range(Yearly, mitrange(qq(2, 2), qq(4, 4)), trim="both", parts=True)
        assert isinstance(result, tuple)
        assert len(result) == 6
        assert all(isinstance(v, int) for v in result)


# ---------------------------------------------------------------------------
# Custom-function dispatch
# ---------------------------------------------------------------------------


class TestCustomFunction:
    def test_aggregator_lower_matches_builtin(self):
        # mean / sum / min / max via callable equal the named methods.
        for f, name in [(np.mean, "mean"), (np.sum, "sum"), (np.min, "min"), (np.max, "max")]:
            t = TSeries(qq(5, 1), np.arange(1, 17, dtype=np.float64))
            ref_choice: str
            for ref_choice in ("begin", "end"):
                a = fconvert_tseries(f, Yearly, t, ref=ref_choice)
                b = fconvert_tseries(Yearly, t, method=name, ref=ref_choice)
                assert a.range == b.range
                np.testing.assert_allclose(a.values, b.values)

    def test_higher_helpers_match_named_methods(self):
        t = TSeries(yy(2022), np.arange(1, 5, dtype=np.float64))
        pairs = [
            (repeat_uneven, "const"),
            (divide_uneven, "even"),
            (linear_uneven, "linear"),
        ]
        for f, name in pairs:
            for ref_choice in ("begin", "end"):
                a = fconvert_tseries(f, Monthly, t, ref=ref_choice)
                b = fconvert_tseries(Monthly, t, method=name, ref=ref_choice)
                assert a.range == b.range
                np.testing.assert_allclose(a.values, b.values)

    def test_lower_custom_second_highest(self):
        # Mirrors Julia's ``second_highest`` example.
        def second_highest(x: np.ndarray) -> float:
            return float(x[0]) if len(x) == 1 else float(np.sort(x)[-2])

        ts = TSeries(mm(2022, 1), np.arange(1, 13, dtype=np.float64))
        ts_q = fconvert_tseries(second_highest, Quarterly, ts)
        np.testing.assert_array_equal(ts_q.values, [2, 5, 8, 11])

    def test_higher_custom_half_value(self):
        def half_value(x: np.ndarray, inner_lengths: np.ndarray, **_kwargs: object) -> np.ndarray:
            out = np.empty(int(inner_lengths.sum()), dtype=np.float64)
            pos = 0
            for i, length in enumerate(inner_lengths):
                out[pos : pos + length] = x[i] / 2
                pos += length
            return out

        ts2 = TSeries(qq(2022, 1), np.arange(1, 5, dtype=np.float64))
        ts_m = fconvert_tseries(half_value, Monthly, ts2)
        np.testing.assert_array_equal(
            ts_m.values,
            [0.5, 0.5, 0.5, 1, 1, 1, 1.5, 1.5, 1.5, 2, 2, 2],
        )

    def test_higher_custom_uses_kwarg(self):
        def replace_with_kwarg(_x: np.ndarray, _inner: np.ndarray, **kwargs: object) -> np.ndarray:
            return np.asarray(kwargs["replacement"].values)

        ts1 = TSeries(qq(2022, 1), np.arange(1, 5, dtype=np.float64))
        ts2 = TSeries(mm(2022, 1), np.arange(13, 25, dtype=np.float64))
        ts_m = fconvert(replace_with_kwarg, Monthly, ts1, replacement=ts2)
        np.testing.assert_array_equal(ts_m.values, ts2.values)


# ---------------------------------------------------------------------------
# All YP combinations smoke (mirrors a YP-only slice of ``"all combinations"``)
# ---------------------------------------------------------------------------


_YP_FREQS = [
    Yearly(),
    Yearly(end_month=3),
    Yearly(end_month=7),
    HalfYearly(),
    HalfYearly(end_month=1),
    HalfYearly(end_month=4),
    Quarterly(),
    Quarterly(end_month=1),
    Monthly(),
]


class TestAllYPCombinations:
    @pytest.mark.parametrize("f_from", _YP_FREQS)
    @pytest.mark.parametrize("f_to", _YP_FREQS)
    def test_combinations_smoke(self, f_from, f_to):
        if f_from == f_to:
            return
        t_from = TSeries(MIT(f_from, 100), np.arange(1, 801, dtype=np.float64))
        t_to = fconvert(f_to, t_from)
        assert t_to.frequency == f_to
        assert len(t_to) > 0
        if f_to.periods_per_year < f_from.periods_per_year:
            for method in ("mean", "sum", "point", "min", "max", "begin", "end"):
                for ref_choice in ("begin", "end"):
                    sub = fconvert(f_to, t_from, method=method, ref=ref_choice)
                    assert sub.frequency == f_to
                    assert len(sub) > 0
                    assert -1e6 < float(np.min(sub.values)) < 1e6
                    assert -1e6 < float(np.max(sub.values)) < 1e6
        elif f_to.periods_per_year > f_from.periods_per_year:
            for method in ("const", "even", "linear"):
                for ref_choice in ("begin", "end"):
                    sub = fconvert(f_to, t_from, method=method, ref=ref_choice)
                    assert sub.frequency == f_to
                    assert len(sub) > 0
                    assert -1e6 < float(np.min(sub.values)) < 1e6
                    assert -1e6 < float(np.max(sub.values)) < 1e6

    @pytest.mark.parametrize("f_from", _YP_FREQS)
    @pytest.mark.parametrize("f_to", _YP_FREQS)
    def test_range_combinations_smoke(self, f_from, f_to):
        if f_from == f_to:
            return
        rng = MITRange(MIT(f_from, 100), MIT(f_from, 899))
        out = fconvert(f_to, rng)
        assert out.frequency == f_to
        assert len(out) > 0


# ---------------------------------------------------------------------------
# extend_series (YP only)
# ---------------------------------------------------------------------------


class TestExtendSeries:
    def test_monthly_extend_to_quarterly_default(self):
        tm = TSeries(mm(2022, 2), np.arange(1.0, 8.0))
        ex1 = extend_series(Quarterly, tm)
        assert ex1.range == mitrange(mm(2022, 1), mm(2022, 9))
        np.testing.assert_array_equal(ex1.values, [1.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 6.5])

    def test_monthly_extend_method_end(self):
        tm = TSeries(mm(2022, 2), np.arange(1.0, 8.0))
        ex2 = extend_series(Quarterly, tm, method="end")
        assert ex2.range == mitrange(mm(2022, 1), mm(2022, 9))
        np.testing.assert_array_equal(ex2.values, [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0])

    def test_monthly_extend_direction_end_only(self):
        tm = TSeries(mm(2022, 2), np.arange(1.0, 8.0))
        ex3 = extend_series(Quarterly, tm, direction="end")
        assert ex3.range == mitrange(mm(2022, 2), mm(2022, 9))
        np.testing.assert_array_equal(ex3.values, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 6.5])

    def test_monthly_extend_direction_begin_only(self):
        tm = TSeries(mm(2022, 2), np.arange(1.0, 8.0))
        ex4 = extend_series(Quarterly, tm, direction="begin")
        assert ex4.range == mitrange(mm(2022, 1), mm(2022, 8))
        np.testing.assert_array_equal(ex4.values, [1.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    def test_monthly_extend_to_yearly_both(self):
        tm = TSeries(mm(2022, 2), np.arange(1.0, 8.0))
        ex5 = extend_series(Yearly, tm)
        assert ex5.range == mitrange(mm(2022, 1), mm(2022, 12))
        np.testing.assert_array_equal(
            ex5.values, [4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 4.0, 4.0, 4.0, 4.0]
        )

    def test_monthly_extend_to_yearly_method_end(self):
        tm = TSeries(mm(2022, 2), np.arange(1.0, 8.0))
        ex5 = extend_series(Yearly, tm, method="end")
        assert ex5.range == mitrange(mm(2022, 1), mm(2022, 12))
        np.testing.assert_array_equal(
            ex5.values, [1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0, 7.0, 7.0]
        )

    def test_extend_does_not_mutate_input(self):
        tm = TSeries(mm(2022, 2), np.arange(1.0, 8.0))
        original = tm.copy()
        extend_series(Quarterly, tm)
        assert tm.equals(original)


# ---------------------------------------------------------------------------
# strip helpers
# ---------------------------------------------------------------------------


class TestStrip:
    def test_strip_tseries_trims_leading_and_trailing_nan(self):
        rng = mitrange(mm(1, 1), mm(1, 7))
        t = TSeries(rng, np.array([np.nan, np.nan, 3.0, 4.0, 5.0, np.nan, np.nan]))
        stripped = strip_tseries(t)
        assert stripped.range == mitrange(mm(1, 3), mm(1, 5))
        np.testing.assert_array_equal(stripped.values, [3.0, 4.0, 5.0])
        # original untouched
        assert len(t) == 7

    def test_strip_inplace_mutates(self):
        rng = mitrange(mm(1, 1), mm(1, 7))
        t = TSeries(rng, np.array([np.nan, np.nan, 3.0, 4.0, 5.0, np.nan, np.nan]))
        strip_tseries_inplace(t)
        assert t.range == mitrange(mm(1, 3), mm(1, 5))
        np.testing.assert_array_equal(t.values, [3.0, 4.0, 5.0])

    def test_strip_all_nan_results_in_empty_range(self):
        rng = mitrange(mm(1, 1), mm(1, 4))
        t = TSeries(rng, np.array([np.nan, np.nan, np.nan, np.nan]))
        stripped = strip_tseries(t)
        assert stripped.is_empty()

    def test_strip_does_not_touch_bool(self):
        rng = mitrange(mm(1, 1), mm(1, 4))
        t = TSeries(rng, np.array([False, True, False, True]))
        stripped = strip_tseries(t)
        assert stripped.range == rng


# ---------------------------------------------------------------------------
# trim_series
# ---------------------------------------------------------------------------


class TestTrimSeries:
    def test_trim_monthly_to_quarterly_both(self):
        tm = TSeries(mm(2022, 2), np.arange(1.0, 8.0))
        # rangeof(tm) = 2022M2:2022M8; fconvert(Quarterly, .., trim="both")
        # -> 2022Q2:2022Q2 (one full quarter Apr-Jun); back to monthly:
        # 2022M4:2022M6.
        out = trim_series(Quarterly, tm)
        assert out.range == mitrange(mm(2022, 4), mm(2022, 6))
        np.testing.assert_array_equal(out.values, [3.0, 4.0, 5.0])

    def test_trim_returns_new_tseries(self):
        tm = TSeries(mm(2022, 2), np.arange(1.0, 8.0))
        out = trim_series(Quarterly, tm)
        out.values[0] = -999
        # original untouched (different storage — TSeries getitem on a range
        # already returns a fresh TSeries).
        assert tm.values[2] == 3.0


# ---------------------------------------------------------------------------
# fconvert_parts smoke
# ---------------------------------------------------------------------------


class TestFconvertParts:
    def test_yearly_quarterly_ref_end(self):
        period, src_month, tgt_month = fconvert_parts(Quarterly, MIT(Yearly(), 22), ref="end")
        # 22Y ends in Dec of year 22 → calendar month 12*23 = 276; same for target.
        assert (src_month, tgt_month) == (276, 276)
        # Target Quarterly{3} period for end-of-year 22 = 22*4+3 = 91 (i.e. 22Q4).
        assert period == 91

    def test_quarterly_yearly_ref_begin(self):
        period, src_month, tgt_month = fconvert_parts(Yearly, qq(22, 2), ref="begin")
        # 22Q2 begins in calendar month 22*12+4 = 268
        assert src_month == 268
        # Target Yearly{12} period for begin-of-22Q2 → year 22, target month 22*12+1 = 265
        assert tgt_month == 265
        assert period == 22
