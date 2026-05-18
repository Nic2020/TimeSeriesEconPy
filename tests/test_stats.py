# SPDX-License-Identifier: MIT
"""Behavioural tests for the ``axis=`` kwarg on the five MVTSeries reductions.

Closes ``claude_files/parity/PARITY_GAPS.md`` G11: Julia's
``Statistics.{mean,std,var,median,quantile}(::MVTSeries; dims=)`` overloads
return per-column / per-row TSeries; the Python port now exposes the same
through a NumPy-conventional ``axis=`` kwarg.

Scope (M1.6.3f): five public functions (:func:`tsecon.mean`,
:func:`tsecon.std`, :func:`tsecon.var`, :func:`tsecon.median`,
:func:`tsecon.quantile`). ``cor`` / ``cov`` axis= semantics stage as
M1.6.3f.1 — the matrix return shape makes the per-row / per-column
contract non-trivial.

Return shapes
-------------
* ``axis=None``: scalar (or ndarray for ``quantile`` with array ``p``).
* ``axis=0`` on MVTSeries: single-row MVTSeries with the input column
  names anchored at the input ``firstdate``.
* ``axis=1`` on MVTSeries: 1-D TSeries indexed by the input range.
* ``axis=0`` on TSeries: equivalent to ``axis=None`` (1-D NumPy convention).
* ``axis=1`` on TSeries: ``ValueError`` naming ``axis``.
* ``axis ∉ {None, 0, 1}``: ``ValueError`` naming ``axis``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from tsecon import (
    MIT,
    MITRange,
    MVTSeries,
    TSeries,
    bdaily,
    mean,
    median,
    qq,
    quantile,
    std,
    var,
)
from tsecon._options import clear_holidays_map, getoption, setoption

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

# The five functions whose axis= contract we lock. Each is wrapped so a single
# parametric grid can call them through a common signature ``f(t, axis=...)``.
# quantile is fixed to ``p=0.5`` (median equivalent) here so the grid stays
# uniform; the array-``p`` × axis combination is tested separately.
_REDUCTION_FNS: dict[str, Callable[..., Any]] = {
    "mean": mean,
    "std": std,
    "var": var,
    "median": median,
    "quantile": lambda t, **kw: quantile(t, 0.5, **kw),
}

# Manual reference reducers — applied to ndarray inputs, return ndarray /
# scalar. The axis= path's correctness criterion is that its output matches
# these references applied to the same data.
_NUMPY_REFS: dict[str, Callable[..., Any]] = {
    "mean": lambda a, **kw: np.mean(a, **kw),
    "std": lambda a, **kw: np.std(a, ddof=1, **kw),
    "var": lambda a, **kw: np.var(a, ddof=1, **kw),
    "median": lambda a, **kw: np.median(a, **kw),
    "quantile": lambda a, **kw: np.quantile(a, 0.5, **kw),
}


def _mvts(values: np.ndarray, *, freq_start: MIT | None = None) -> MVTSeries:
    """Build an MVTSeries with the canonical 5-column-letter names."""
    if freq_start is None:
        freq_start = qq(2020, 1)
    ncols = values.shape[1]
    names = tuple("abcdefghij"[:ncols])
    return MVTSeries(freq_start, names, values)


@pytest.fixture
def reset_holidays_map() -> Any:
    """Save / clear / restore the global ``bdaily_holidays_map`` per test."""
    saved = getoption("bdaily_holidays_map")
    clear_holidays_map()
    yield
    if saved is None:
        clear_holidays_map()
    else:
        setoption("bdaily_holidays_map", saved)


# ---------------------------------------------------------------------------
# Behavioural grid — function × axis × shape
# ---------------------------------------------------------------------------


class TestMVTSeriesAxisReductions:
    """Lock the four axis= return shapes against manual NumPy references."""

    @pytest.mark.parametrize("fn_name", list(_REDUCTION_FNS))
    def test_axis_none_unchanged(self, fn_name: str) -> None:
        """``axis=None`` is the pre-G11 behaviour — flat scalar over the matrix."""
        f = _REDUCTION_FNS[fn_name]
        ref = _NUMPY_REFS[fn_name]
        rng_np = np.random.default_rng(seed=20260518)
        values = rng_np.standard_normal((10, 4))
        m = _mvts(values)
        # axis=None matches np.* on the raveled matrix.
        np.testing.assert_allclose(float(f(m)), float(ref(values)), rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("fn_name", list(_REDUCTION_FNS))
    def test_axis_0_returns_single_row_mvts(self, fn_name: str) -> None:
        """``axis=0`` returns a single-row MVTSeries with the input column names."""
        f = _REDUCTION_FNS[fn_name]
        ref = _NUMPY_REFS[fn_name]
        rng_np = np.random.default_rng(seed=20260518)
        values = rng_np.standard_normal((10, 4))
        m = _mvts(values)
        result = f(m, axis=0)
        assert isinstance(result, MVTSeries)
        assert result.shape == (1, 4)
        assert result.column_names == m.column_names
        assert result.firstdate == m.firstdate
        expected = ref(values, axis=0)
        np.testing.assert_allclose(result.values.ravel(), expected, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("fn_name", list(_REDUCTION_FNS))
    def test_axis_1_returns_1d_tseries(self, fn_name: str) -> None:
        """``axis=1`` returns a 1-D TSeries indexed by the input range."""
        f = _REDUCTION_FNS[fn_name]
        ref = _NUMPY_REFS[fn_name]
        rng_np = np.random.default_rng(seed=20260518)
        values = rng_np.standard_normal((10, 4))
        m = _mvts(values)
        result = f(m, axis=1)
        assert isinstance(result, TSeries)
        assert len(result) == 10
        assert result.firstdate == m.firstdate
        assert result.range == m.range
        expected = ref(values, axis=1)
        np.testing.assert_allclose(result.values, expected, rtol=1e-12, atol=1e-15)

    @pytest.mark.parametrize("fn_name", list(_REDUCTION_FNS))
    def test_axis_invalid_raises(self, fn_name: str) -> None:
        """``axis ∉ {None, 0, 1}`` raises ``ValueError`` naming the offending value."""
        f = _REDUCTION_FNS[fn_name]
        m = _mvts(np.ones((5, 3)))
        with pytest.raises(ValueError, match=r"axis must be None, 0, or 1; got axis=2"):
            f(m, axis=2)
        with pytest.raises(ValueError, match=r"axis must be None, 0, or 1; got axis=-1"):
            f(m, axis=-1)

    @pytest.mark.parametrize("fn_name", list(_REDUCTION_FNS))
    def test_tseries_axis_1_raises(self, fn_name: str) -> None:
        """A 1-D TSeries has no axis 1 — the error message points the user to axis=0."""
        f = _REDUCTION_FNS[fn_name]
        t = TSeries(qq(2020, 1), np.arange(10.0))
        with pytest.raises(ValueError, match="axis=1 is not valid for a 1-D TSeries"):
            f(t, axis=1)

    @pytest.mark.parametrize("fn_name", list(_REDUCTION_FNS))
    def test_tseries_axis_0_matches_axis_none(self, fn_name: str) -> None:
        """``axis=0`` on a 1-D TSeries is an alias of ``axis=None`` (NumPy convention)."""
        f = _REDUCTION_FNS[fn_name]
        t = TSeries(qq(2020, 1), np.arange(1.0, 11.0))
        # Both paths return the same scalar.
        np.testing.assert_allclose(float(f(t)), float(f(t, axis=0)), rtol=1e-12)

    def test_axis_0_column_names_preserved_under_non_default_names(self) -> None:
        """Custom column names propagate through the single-row MVTSeries."""
        values = np.arange(12.0, dtype=float).reshape(3, 4)
        m = MVTSeries(qq(2020, 1), ("alpha", "beta", "gamma", "delta"), values)
        result = mean(m, axis=0)
        assert result.column_names == ("alpha", "beta", "gamma", "delta")

    def test_axis_1_range_unchanged_under_arbitrary_firstdate(self) -> None:
        """The output TSeries's range equals the input MVTSeries's range exactly."""
        firstdate = qq(2018, 3)
        m = MVTSeries(firstdate, ("a", "b"), np.arange(20.0).reshape(10, 2))
        result = mean(m, axis=1)
        assert result.firstdate == firstdate
        assert result.lastdate == m.lastdate
        assert result.range == m.range


# ---------------------------------------------------------------------------
# BDaily kwargs × axis=
# ---------------------------------------------------------------------------


class TestMVTSeriesAxisBDaily:
    """``skip_all_nans`` / ``skip_holidays`` / ``holidays_map`` compose with ``axis=``.

    axis=0: the row mask drops rows from each column's reduction; the
    single-row output summarises the kept rows.

    axis=1: the per-row reduction runs over the full matrix; positions where
    the BDaily filter would have dropped the row are masked to NaN in the
    output, preserving the contiguous MIT range.
    """

    @pytest.mark.parametrize("fn_name", ["mean", "std", "var", "median", "quantile"])
    def test_axis_0_with_holidays_map_matches_per_column(self, fn_name: str) -> None:
        f = _REDUCTION_FNS[fn_name]
        rng = np.random.default_rng(seed=20260518)
        nrows = 20
        values = rng.standard_normal((nrows, 3))
        bd_start = bdaily("2020-01-01")
        m = MVTSeries(bd_start, ("a", "b", "c"), values)
        # Build a holidays_map that drops every third row.
        mask_vals = np.ones(nrows, dtype=bool)
        mask_vals[::3] = False
        h_map = TSeries(bd_start, mask_vals)
        result = f(m, axis=0, holidays_map=h_map)
        assert isinstance(result, MVTSeries)
        assert result.shape == (1, 3)
        # Each column's value matches f applied to that column (with the same mask).
        for j, col_name in enumerate(m.column_names):
            col_t = m[col_name]
            expected = f(col_t, holidays_map=h_map)
            np.testing.assert_allclose(
                result.values[0, j],
                expected,
                rtol=1e-12,
                atol=1e-15,
            )

    @pytest.mark.parametrize("fn_name", ["mean", "std", "var", "median", "quantile"])
    def test_axis_1_with_holidays_map_masks_output(self, fn_name: str) -> None:
        """``axis=1`` + holidays_map: dropped-row positions are NaN in the output TSeries."""
        f = _REDUCTION_FNS[fn_name]
        ref = _NUMPY_REFS[fn_name]
        rng = np.random.default_rng(seed=20260518)
        nrows = 20
        values = rng.standard_normal((nrows, 3))
        bd_start = bdaily("2020-01-01")
        m = MVTSeries(bd_start, ("a", "b", "c"), values)
        mask_vals = np.ones(nrows, dtype=bool)
        mask_vals[::3] = False
        h_map = TSeries(bd_start, mask_vals)
        result = f(m, axis=1, holidays_map=h_map)
        assert isinstance(result, TSeries)
        assert len(result) == nrows
        # On kept rows, the per-row reduction equals np.* applied row-wise.
        expected_per_row = ref(values, axis=1)
        np.testing.assert_allclose(
            result.values[mask_vals],
            expected_per_row[mask_vals],
            rtol=1e-12,
            atol=1e-15,
        )
        # On dropped rows, the value is NaN.
        assert np.all(np.isnan(result.values[~mask_vals]))

    def test_axis_0_skip_all_nans_drops_all_nan_rows(self) -> None:
        """``skip_all_nans=True`` drops rows that are NaN in every column."""
        nrows = 5
        values = np.tile(np.arange(1.0, 4.0), (nrows, 1))
        # Row 2 is all-NaN.
        values[2, :] = np.nan
        bd_start = bdaily("2020-01-01")
        m = MVTSeries(bd_start, ("a", "b", "c"), values)
        result = mean(m, axis=0, skip_all_nans=True)
        # The kept rows (0, 1, 3, 4) are all [1, 2, 3]; per-column mean is [1, 2, 3].
        np.testing.assert_allclose(result.values.ravel(), [1.0, 2.0, 3.0])

    def test_bdaily_kwargs_on_non_bdaily_with_axis_raises(self) -> None:
        """BDaily kwargs against a non-BDaily MVTSeries still raise (matches axis=None)."""
        m = _mvts(np.ones((5, 2)))  # Quarterly
        with pytest.raises(TypeError, match="only valid for BDaily series"):
            mean(m, axis=0, skip_all_nans=True)
        # axis=1 path raises via bdaily_row_keep_mask -> _require_bdaily.
        with pytest.raises(TypeError, match="only defined for BDaily series"):
            mean(m, axis=1, skip_all_nans=True)


# ---------------------------------------------------------------------------
# quantile array-p × axis= guard
# ---------------------------------------------------------------------------


class TestQuantileArrayPAxis:
    """Array-``p`` + axis=  fails with a guiding TypeError; scalar-``p`` still works."""

    def test_array_p_axis_0_raises(self) -> None:
        m = _mvts(np.arange(20.0).reshape(5, 4))
        with pytest.raises(TypeError, match="p must be a scalar probability"):
            quantile(m, np.array([0.25, 0.5, 0.75]), axis=0)

    def test_array_p_axis_1_raises(self) -> None:
        m = _mvts(np.arange(20.0).reshape(5, 4))
        with pytest.raises(TypeError, match="p must be a scalar probability"):
            quantile(m, np.array([0.25, 0.5, 0.75]), axis=1)

    def test_array_p_axis_none_still_works(self) -> None:
        """Array-``p`` × axis=None preserves the existing array return shape."""
        m = _mvts(np.arange(20.0).reshape(5, 4))
        result = quantile(m, np.array([0.25, 0.5, 0.75]))
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------


@st.composite
def _mvts_strategy(draw: st.DrawFn, min_rows: int = 2, max_rows: int = 30) -> MVTSeries:
    """Draw an MVTSeries with float64 values in a well-conditioned range."""
    nrows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    ncols = draw(st.integers(min_value=2, max_value=5))
    values = draw(
        st.lists(
            st.floats(
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_infinity=False,
                width=64,
            ),
            min_size=nrows * ncols,
            max_size=nrows * ncols,
        )
    )
    arr = np.array(values, dtype=np.float64).reshape(nrows, ncols)
    return _mvts(arr)


_PROPERTY_FNS = ["mean", "std", "var", "median"]


class TestMVTSeriesAxisProperties:
    """Hypothesis: per-column and per-row reductions agree with the manual paths.

    Property 1 (axis=None flat reduction):
        ``f(m) == f(m.values.ravel())``.

    Property 2 (axis=0 per-column reduction):
        For every column name ``col``, ``f(m, axis=0)[col][m.firstdate] == f(m[col])``.

    Property 3 (axis=1 per-row reduction):
        For every MIT in range, ``f(m, axis=1)[mit] == np.<f>(m.values[i, :])``.
    """

    @settings(
        max_examples=80,
        deadline=None,
        suppress_health_check=(HealthCheck.too_slow,),
    )
    @given(m=_mvts_strategy())
    def test_axis_none_matches_flat_numpy(self, m: MVTSeries) -> None:
        flat = m.values.ravel()
        np.testing.assert_allclose(float(mean(m)), float(np.mean(flat)), rtol=1e-10)
        np.testing.assert_allclose(float(std(m)), float(np.std(flat, ddof=1)), rtol=1e-10)
        np.testing.assert_allclose(float(var(m)), float(np.var(flat, ddof=1)), rtol=1e-10)
        np.testing.assert_allclose(float(median(m)), float(np.median(flat)), rtol=1e-10)

    @settings(
        max_examples=80,
        deadline=None,
        suppress_health_check=(HealthCheck.too_slow,),
    )
    @given(m=_mvts_strategy())
    def test_axis_0_matches_per_column(self, m: MVTSeries) -> None:
        for fn_name in _PROPERTY_FNS:
            f = _REDUCTION_FNS[fn_name]
            per_col = f(m, axis=0)
            for col_name in m.column_names:
                col_scalar = float(per_col[col_name][m.firstdate])
                ref_scalar = float(f(m[col_name]))
                np.testing.assert_allclose(col_scalar, ref_scalar, rtol=1e-10, atol=1e-12)

    @settings(
        max_examples=80,
        deadline=None,
        suppress_health_check=(HealthCheck.too_slow,),
    )
    @given(m=_mvts_strategy())
    def test_axis_1_matches_per_row(self, m: MVTSeries) -> None:
        for fn_name in _PROPERTY_FNS:
            f = _REDUCTION_FNS[fn_name]
            ref = _NUMPY_REFS[fn_name]
            per_row = f(m, axis=1)
            expected = ref(m.values, axis=1)
            np.testing.assert_allclose(per_row.values, expected, rtol=1e-10, atol=1e-12)
            # MIT-by-MIT spot-check: indexing into the result must match
            # np.<f>(row).
            for i, mit in enumerate(m.range):
                row_scalar = float(per_row[mit])
                row_ref = float(ref(m.values[i, :]))
                np.testing.assert_allclose(row_scalar, row_ref, rtol=1e-10, atol=1e-12)


# Used to silence ruff F401 — MIT is imported as a re-export sanity check.
_ = MIT
_ = MITRange
