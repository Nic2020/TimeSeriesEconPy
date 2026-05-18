# SPDX-License-Identifier: MIT
"""Tests for the M1.5 fourth Cython port — fconvert lower-aggregate kernel.

Covers the kernel pair (``_fconvert_kernels.aggregate_groups_numpy`` and
the matching ``_fconvert_kernels_cy.aggregate_groups_cython`` when
compiled), the public ``fconvert(target, t, method=...)`` dispatcher path
through ``_tseries._fconvert_lower_aggregate_yp``, and the
``fconvert_is_cython`` introspection helper. Behaviour-level tests for
the full fconvert surface live in ``test_fconvert.py``; this file
validates the kernel-direct contract and the equivalence between the
Cython kernel (when present) and its NumPy reference sibling.

See ``claude_files/decisions/18_cython_port_plan.md`` § "Tier 1 —
fconvert_lower_aggregate_kernel".
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from tsecon import (
    Monthly,
    Quarterly,
    TSeries,
    Yearly,
    daily,
    fconvert,
    fconvert_is_cython,
    mm,
    qq,
)
from tsecon._fconvert_kernels import (
    METHOD_FIRST,
    METHOD_LAST,
    METHOD_MAX,
    METHOD_MEAN,
    METHOD_MIN,
    METHOD_SUM,
    aggregate_groups_numpy,
)
from tsecon.fconvert._tseries import _aggregate_groups_dispatch

try:
    from tsecon._fconvert_kernels_cy import (  # type: ignore[import-not-found]
        aggregate_groups_cython,
    )

    _CY = True
except ImportError:
    _CY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_groups(n_groups: int, group_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Build uniform-size group_starts / group_lengths arrays."""
    starts = np.arange(0, n_groups * group_size, group_size, dtype=np.int64)
    lengths = np.full(n_groups, group_size, dtype=np.int64)
    return starts, lengths


def _gen_inputs() -> list[tuple[np.ndarray, int, int]]:
    """Yield ``(values, n_groups, group_size)`` triples for parametric tests.

    Covers the Quarterly→Yearly (4-per-group), Monthly→Quarterly
    (3-per-group), and Monthly→Yearly (12-per-group) regimes plus a
    couple of synthetic shapes for edge coverage.
    """
    rng = np.random.default_rng(seed=20260516)
    return [
        (np.arange(100.0), 25, 4),  # QQ → YY
        (np.arange(120.0), 40, 3),  # MM → QQ
        (np.arange(120.0), 10, 12),  # MM → YY
        (rng.standard_normal(100), 25, 4),
        (rng.standard_normal(120), 40, 3),
        (rng.standard_normal(60), 5, 12),
        (np.full(8, 7.5), 2, 4),
        (np.linspace(-1.0, 1.0, 20), 5, 4),
    ]


# ---------------------------------------------------------------------------
# Kernel-direct equivalence: Cython ≡ NumPy reference
# ---------------------------------------------------------------------------


class TestFconvertKernelAgreesWithNumpy:
    """Cython kernel matches the NumPy reference output to within FP tolerance.

    Same precedent as the rec_linear / stats kernel-equivalence tests —
    the two implementations share the same naive-accumulator algorithm,
    so length-100 well-conditioned inputs agree at ``rtol=1e-12``.
    """

    @pytest.mark.parametrize(
        "method_code",
        [METHOD_MEAN, METHOD_SUM, METHOD_MIN, METHOD_MAX, METHOD_FIRST, METHOD_LAST],
    )
    @pytest.mark.parametrize(("values", "n_groups", "group_size"), _gen_inputs())
    def test_kernel_matches_numpy(
        self,
        method_code: int,
        values: np.ndarray,
        n_groups: int,
        group_size: int,
    ) -> None:
        if not _CY:
            pytest.skip("Cython fconvert kernel not compiled")
        v = np.ascontiguousarray(values, dtype=np.float64)
        starts, lengths = _uniform_groups(n_groups, group_size)
        np.testing.assert_allclose(
            aggregate_groups_cython(v, starts, lengths, method_code),
            aggregate_groups_numpy(v, starts, lengths, method_code),
            rtol=1e-12,
            atol=1e-15,
        )

    def test_kernel_handles_variable_group_lengths(self) -> None:
        """Variable-size groups (ragged) work as documented in the contract."""
        if not _CY:
            pytest.skip("Cython fconvert kernel not compiled")
        v = np.arange(20.0)
        starts = np.array([0, 3, 8, 15], dtype=np.int64)
        lengths = np.array([3, 5, 7, 5], dtype=np.int64)
        for code in (METHOD_MEAN, METHOD_SUM, METHOD_MIN, METHOD_MAX):
            a = aggregate_groups_cython(v, starts, lengths, code)
            b = aggregate_groups_numpy(v, starts, lengths, code)
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# Public API agreement — fconvert(method=...) still matches per-group reduction
# ---------------------------------------------------------------------------


class TestPublicFconvertAgreesWithGroupwise:
    """``fconvert(target, t, method=...)`` matches per-group NumPy reduction.

    Belt-and-braces: the existing ``test_fconvert.py`` covers behaviour
    (range alignment, ref/trim, edge cases), but explicit equivalence
    tests against the group-reshape reduction lock the Cython-dispatch
    branch's output to the per-group NumPy reference.
    """

    def test_qq_to_yy_mean_matches_groupwise(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(100, dtype=np.float64))
        y = fconvert(Yearly(), t, method="mean")
        # 25 years x 4 quarters each, starting at year 2020.
        expected = t.values.reshape(25, 4).mean(axis=1)
        np.testing.assert_allclose(y.values, expected, rtol=1e-12, atol=1e-15)

    def test_qq_to_yy_sum_matches_groupwise(self) -> None:
        t = TSeries(qq(2020, 1), np.arange(100, dtype=np.float64))
        y = fconvert(Yearly(), t, method="sum")
        expected = t.values.reshape(25, 4).sum(axis=1)
        np.testing.assert_allclose(y.values, expected, rtol=1e-12, atol=1e-15)

    def test_qq_to_yy_min_matches_groupwise(self) -> None:
        rng = np.random.default_rng(seed=20260516)
        t = TSeries(qq(2020, 1), rng.standard_normal(100))
        y = fconvert(Yearly(), t, method="min")
        expected = t.values.reshape(25, 4).min(axis=1)
        np.testing.assert_allclose(y.values, expected, rtol=1e-12, atol=1e-15)

    def test_qq_to_yy_max_matches_groupwise(self) -> None:
        rng = np.random.default_rng(seed=20260516)
        t = TSeries(qq(2020, 1), rng.standard_normal(100))
        y = fconvert(Yearly(), t, method="max")
        expected = t.values.reshape(25, 4).max(axis=1)
        np.testing.assert_allclose(y.values, expected, rtol=1e-12, atol=1e-15)

    def test_mm_to_qq_mean_matches_groupwise(self) -> None:
        t = TSeries(mm(2020, 1), np.arange(120, dtype=np.float64))
        q = fconvert(Quarterly(), t, method="mean")
        expected = t.values.reshape(40, 3).mean(axis=1)
        np.testing.assert_allclose(q.values, expected, rtol=1e-12, atol=1e-15)

    def test_mm_to_yy_mean_matches_groupwise(self) -> None:
        t = TSeries(mm(2020, 1), np.arange(120, dtype=np.float64))
        y = fconvert(Yearly(), t, method="mean")
        expected = t.values.reshape(10, 12).mean(axis=1)
        np.testing.assert_allclose(y.values, expected, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# Kernel-direct fallback (always-callable NumPy reference)
# ---------------------------------------------------------------------------


class TestFconvertKernelFallback:
    """The NumPy kernel is always callable, regardless of compile state."""

    def test_aggregate_mean_returns_float64_array(self) -> None:
        v = np.arange(20.0)
        starts, lengths = _uniform_groups(4, 5)
        out = aggregate_groups_numpy(v, starts, lengths, METHOD_MEAN)
        assert out.dtype == np.float64
        assert out.shape == (4,)
        np.testing.assert_allclose(out, [2.0, 7.0, 12.0, 17.0], rtol=1e-12)

    def test_aggregate_sum_returns_float64_array(self) -> None:
        v = np.arange(20.0)
        starts, lengths = _uniform_groups(4, 5)
        out = aggregate_groups_numpy(v, starts, lengths, METHOD_SUM)
        np.testing.assert_allclose(out, [10.0, 35.0, 60.0, 85.0], rtol=1e-12)

    def test_aggregate_first_returns_float64_array(self) -> None:
        v = np.arange(20.0)
        starts, lengths = _uniform_groups(4, 5)
        out = aggregate_groups_numpy(v, starts, lengths, METHOD_FIRST)
        np.testing.assert_allclose(out, [0.0, 5.0, 10.0, 15.0], rtol=1e-12)

    def test_aggregate_last_returns_float64_array(self) -> None:
        v = np.arange(20.0)
        starts, lengths = _uniform_groups(4, 5)
        out = aggregate_groups_numpy(v, starts, lengths, METHOD_LAST)
        np.testing.assert_allclose(out, [4.0, 9.0, 14.0, 19.0], rtol=1e-12)

    def test_unknown_method_code_raises(self) -> None:
        v = np.arange(10.0)
        starts = np.array([0, 5], dtype=np.int64)
        lengths = np.array([5, 5], dtype=np.int64)
        with pytest.raises(ValueError, match="unknown method_code"):
            aggregate_groups_numpy(v, starts, lengths, 99)

    def test_fconvert_is_cython_returns_bool(self) -> None:
        assert isinstance(fconvert_is_cython(), bool)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestFconvertKernelEdgeCases:
    """Empty / degenerate inputs follow the documented contract."""

    def test_empty_groups_returns_empty_array(self) -> None:
        v = np.arange(10.0)
        starts = np.array([], dtype=np.int64)
        lengths = np.array([], dtype=np.int64)
        out = aggregate_groups_numpy(v, starts, lengths, METHOD_MEAN)
        assert out.shape == (0,)
        assert out.dtype == np.float64

    def test_single_element_groups(self) -> None:
        v = np.array([1.0, 2.0, 3.0, 4.0])
        starts = np.arange(4, dtype=np.int64)
        lengths = np.ones(4, dtype=np.int64)
        for code in (METHOD_MEAN, METHOD_SUM, METHOD_MIN, METHOD_MAX, METHOD_FIRST, METHOD_LAST):
            out = aggregate_groups_numpy(v, starts, lengths, code)
            np.testing.assert_allclose(out, v, rtol=1e-12, atol=1e-15)

    def test_nan_propagates_through_mean(self) -> None:
        v = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])
        starts, lengths = _uniform_groups(2, 4)
        out = aggregate_groups_numpy(v, starts, lengths, METHOD_MEAN)
        assert np.isnan(out[0])
        np.testing.assert_allclose(out[1], 6.5, rtol=1e-12)

    def test_fconvert_falls_back_for_non_float64(self) -> None:
        """Non-float64 inputs route through the Python-loop aggregator path.

        The Cython kernel only handles float64; the dispatcher detects
        dtype and falls back to the per-row aggregator list-comp for
        any other dtype. We assert that fconvert still produces the
        correct output in that branch.
        """
        t = TSeries(qq(2020, 1), np.arange(100, dtype=np.int64))
        y = fconvert(Yearly(), t, method="sum")
        expected = np.arange(100).reshape(25, 4).sum(axis=1)
        np.testing.assert_allclose(np.asarray(y.values), expected, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# Scope cut — kernel is YP-only; calendar inputs must use the fallback path
# ---------------------------------------------------------------------------


class TestKernelScopeCut:
    """Lock the session-21 YP-only scope cut for the aggregate-groups kernel.

    The kernel signature is integer-offset group-based and was wired
    only for YPFrequency → YPFrequency aggregation
    (Quarterly→Yearly, Monthly→Quarterly, Monthly→Yearly). The calendar
    aggregate path (``_fconvert_lower_calendar_to_yp_or_weekly``) has
    irregular group spec and deliberately does not call the dispatcher
    — see ``claude_files/parity/PARITY.md:86`` and
    ``claude_files/decisions/18_cython_port_plan.md``.

    Without these tests, a future patch could accidentally route a
    calendar input through the kernel; the integer-offset group-start
    arithmetic would silently produce plausible-looking-but-wrong
    aggregates with no signal in CI. The two tests below lock both
    directions: (1) the runtime guard fires on a non-YP caller; (2) the
    calendar code path never invokes the dispatcher.
    """

    def test_yp_target_false_raises_assertion(self) -> None:
        """Direct call with ``is_yp_target=False`` raises ``AssertionError``."""
        v = np.arange(20.0)
        starts, lengths = _uniform_groups(4, 5)
        with pytest.raises(AssertionError, match="YP-only"):
            _aggregate_groups_dispatch(v, starts, lengths, METHOD_MEAN, is_yp_target=False)

    def test_daily_to_monthly_does_not_call_kernel_dispatcher(self) -> None:
        """A calendar→YP fconvert must aggregate per-target without the kernel.

        The calendar path's irregular group spec would silently produce
        wrong output if the kernel were called; this test mock-patches
        the dispatcher and asserts the calendar fconvert never reaches it.
        """
        # 91 days starting 2024-01-01 — covers a complete Jan / Feb / Mar
        # (2024 is a leap year, so 31 + 29 + 31 = 91). Non-uniform group
        # sizes make this structurally incompatible with the YP kernel's
        # uniform-stride group_starts arithmetic.
        t = TSeries(daily("2024-01-01"), np.arange(91, dtype=np.float64))
        with mock.patch("tsecon.fconvert._tseries._aggregate_groups_dispatch") as spy:
            result = fconvert(Monthly(), t, method="mean")
        spy.assert_not_called()
        # Sanity: the calendar path still produced the right answer.
        expected_jan = np.arange(31).mean()
        expected_feb = np.arange(31, 31 + 29).mean()
        expected_mar = np.arange(31 + 29, 91).mean()
        np.testing.assert_allclose(
            result.values, [expected_jan, expected_feb, expected_mar], rtol=1e-12
        )
