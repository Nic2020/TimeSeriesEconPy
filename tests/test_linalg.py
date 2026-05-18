# SPDX-License-Identifier: MIT
"""Behavioural tests for :mod:`tsecon.linalg` — the ``@`` (matrix multiply)
overloads on :class:`~tsecon.tseries.TSeries` and
:class:`~tsecon.mvtseries.MVTSeries`.

Closes ``claude_files/parity/PARITY_GAPS.md`` **G12** (M1.6.3g). The Python
port mirrors ``TimeSeriesEcon.jl/src/linalg.jl`` exactly: every overload
forwards to the underlying ``Vector`` / ``Matrix`` and returns a plain
:class:`numpy.ndarray` (the Julia test row ``x * x3 == _vals(x) * _vals(x3)``
documents the same). ``@`` is the PEP 465 spelling of Julia's ``*`` matrix
product; element-wise ``*`` keeps the existing range-intersection /
frequency-checked semantics.

What's covered
--------------
* ``ndarray @ TSeries`` / ``TSeries @ ndarray`` — both directions (the
  left-operand path goes through ``__matmul__``; the right-operand path
  goes through NumPy's ufunc dispatch into ``__array_ufunc__``).
* ``TSeries @ TSeries`` — vector-vector ``@`` is NumPy's inner product
  (returns a scalar). Note: Julia's overload would also strip labels
  (``_vals(t1) * _vals(t2)``) but ``Vector * Vector`` raises a
  ``MethodError`` in Julia; in Python ``@`` defaults to the inner
  product. Documented divergence.
* ``ndarray @ MVTSeries`` / ``MVTSeries @ ndarray`` — the (k, n) @ (n, c)
  and (n, c) @ (c, m) shapes.
* ``MVTSeries @ MVTSeries`` — (n1, c1) @ (n2, c2); only the inner
  dimensions need to match (column count of LHS = row count of RHS).
  Frequency / range / column-name mismatches are *intentionally*
  permitted: matmul is a numerical operation on the underlying buffer.
* Element-wise ``*`` semantics are unchanged.
* Shape mismatches raise ``ValueError`` from NumPy's matmul gufunc.

Out of scope (intentional non-ports)
------------------------------------
* ``transpose`` / ``adjoint``. The Julia overloads also strip labels and
  return a row vector / transposed matrix with no wrapped type;
  ``.T`` on TSeries / MVTSeries would have no clean semantics (the row
  axis is time). Users who want the bare transpose write
  ``np.asarray(x).T``. See ``src/tsecon/linalg.py`` and the migration
  page.
* ``\\`` / ``/`` (linear-solve). ``np.linalg.solve(A, np.asarray(t))``
  covers it directly.
"""

from __future__ import annotations

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

from tsecon import MITRange, MVTSeries, TSeries, mm, qq

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tseries(n: int, *, start_year: int = 2020, start_q: int = 1) -> TSeries:
    """Length-``n`` quarterly TSeries with values ``0, 1, ..., n-1``."""
    start = qq(start_year, start_q)
    rng = MITRange(start, start + (n - 1))
    return TSeries(rng, np.arange(n, dtype=float))


def _mvts(nrows: int, names: list[str]) -> MVTSeries:
    """Quarterly MVTSeries of shape ``(nrows, len(names))``."""
    ncols = len(names)
    values = np.arange(nrows * ncols, dtype=float).reshape(nrows, ncols)
    return MVTSeries(qq(2020, 1), names, values)


# ---------------------------------------------------------------------------
# Matrix @ TSeries  /  TSeries @ Matrix
# ---------------------------------------------------------------------------


class TestMatrixTSeriesMatmul:
    """``ndarray @ TSeries`` returns a 1-D ndarray equal to ``A @ t.values``."""

    def test_square_matrix_left_returns_ndarray(self) -> None:
        t = _tseries(5)
        a = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 5.0],
            ]
        )
        out = a @ t
        assert isinstance(out, np.ndarray)
        assert not isinstance(out, TSeries)
        np.testing.assert_array_equal(out, a @ t.values)

    def test_identity_returns_values_copy(self) -> None:
        t = _tseries(7)
        identity = np.eye(7)
        out = identity @ t
        np.testing.assert_array_equal(out, t.values)

    def test_rectangular_left_returns_shape_n(self) -> None:
        t = _tseries(4)
        a = np.arange(12.0).reshape(3, 4)  # shape (3, 4)
        out = a @ t
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)
        np.testing.assert_array_equal(out, a @ t.values)

    def test_dtype_promotion_int_matrix_float_series(self) -> None:
        t = _tseries(3)
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
        out = a @ t
        assert out.dtype == np.float64  # int64 @ float64 → float64
        np.testing.assert_array_equal(out, a @ t.values)


class TestTSeriesMatrixMatmul:
    """``TSeries @ ndarray`` returns a 1-D ndarray equal to ``t.values @ A``."""

    def test_square_matrix_right_returns_ndarray(self) -> None:
        t = _tseries(5)
        a = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
        out = t @ a
        assert isinstance(out, np.ndarray)
        assert not isinstance(out, TSeries)
        np.testing.assert_array_equal(out, t.values @ a)

    def test_rectangular_right_returns_shape_m(self) -> None:
        t = _tseries(4)
        a = np.arange(12.0).reshape(4, 3)  # shape (4, 3)
        out = t @ a
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)
        np.testing.assert_array_equal(out, t.values @ a)


class TestTSeriesTSeriesMatmul:
    """``t1 @ t2`` is NumPy's 1-D inner product — a scalar.

    Documented divergence from Julia: Julia's ``Vector * Vector`` raises a
    ``MethodError``; Python ``@`` defaults to ``np.matmul`` which returns
    the inner product. The label-stripping behaviour matches.
    """

    def test_returns_scalar_inner_product(self) -> None:
        t1 = TSeries(MITRange(qq(2020, 1), qq(2020, 1) + 3), np.array([1.0, 2.0, 3.0, 4.0]))
        t2 = TSeries(MITRange(qq(2020, 1), qq(2020, 1) + 3), np.array([5.0, 6.0, 7.0, 8.0]))
        out = t1 @ t2
        assert np.isscalar(out) or (isinstance(out, np.ndarray) and out.ndim == 0)
        assert float(out) == 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8

    def test_different_frequencies_strips_labels_no_typeerror(self) -> None:
        # No frequency check on matmul (matches Julia's _vals-stripping).
        t1 = _tseries(3)  # quarterly
        t2 = TSeries(
            MITRange(qq(2025, 2), qq(2025, 2) + 2),  # different range
            np.array([1.0, 2.0, 3.0]),
        )
        out = t1 @ t2
        assert float(out) == 0 * 1 + 1 * 2 + 2 * 3

    def test_length_mismatch_raises_valueerror(self) -> None:
        t1 = _tseries(3)
        t2 = _tseries(4)
        with pytest.raises(ValueError, match="size"):
            _ = t1 @ t2


# ---------------------------------------------------------------------------
# Matrix @ MVTSeries  /  MVTSeries @ Matrix
# ---------------------------------------------------------------------------


class TestMatrixMVTSeriesMatmul:
    """``ndarray @ MVTSeries`` returns a 2-D ndarray equal to ``A @ mvts.values``."""

    def test_square_left(self) -> None:
        mvts = _mvts(4, ["a", "b", "c"])  # shape (4, 3)
        a = np.eye(4)
        out = a @ mvts
        assert isinstance(out, np.ndarray)
        assert not isinstance(out, MVTSeries)
        assert out.shape == (4, 3)
        np.testing.assert_array_equal(out, mvts.values)

    def test_rectangular_left(self) -> None:
        mvts = _mvts(4, ["a", "b", "c"])  # (4, 3)
        a = np.arange(8.0).reshape(2, 4)  # (2, 4)
        out = a @ mvts
        assert out.shape == (2, 3)
        np.testing.assert_array_equal(out, a @ mvts.values)


class TestMVTSeriesMatrixMatmul:
    """``MVTSeries @ ndarray`` returns a 2-D ndarray equal to ``mvts.values @ A``."""

    def test_square_right(self) -> None:
        mvts = _mvts(4, ["a", "b", "c"])  # (4, 3)
        a = np.eye(3)
        out = mvts @ a
        assert isinstance(out, np.ndarray)
        assert not isinstance(out, MVTSeries)
        assert out.shape == (4, 3)
        np.testing.assert_array_equal(out, mvts.values)

    def test_rectangular_right(self) -> None:
        mvts = _mvts(4, ["a", "b", "c"])  # (4, 3)
        a = np.arange(15.0).reshape(3, 5)  # (3, 5)
        out = mvts @ a
        assert out.shape == (4, 5)
        np.testing.assert_array_equal(out, mvts.values @ a)


class TestMVTSeriesMVTSeriesMatmul:
    """``mvts1 @ mvts2`` is ``mvts1.values @ mvts2.values`` (raw matmul).

    No range intersection, no column-name alignment, no frequency check —
    matches Julia's ``_vals(x) * _vals(y)``.
    """

    def test_compatible_shapes_returns_matrix(self) -> None:
        left = _mvts(4, ["a", "b", "c"])  # (4, 3)
        right = MVTSeries(qq(2020, 1), ["x", "y"], np.arange(6.0).reshape(3, 2))  # (3, 2)
        out = left @ right
        assert isinstance(out, np.ndarray)
        assert not isinstance(out, MVTSeries)
        assert out.shape == (4, 2)
        np.testing.assert_array_equal(out, left.values @ right.values)

    def test_different_frequencies_no_typeerror(self) -> None:
        # Element-wise * would raise on mixed frequency; matmul does not.
        left = _mvts(3, ["a", "b"])  # quarterly (3, 2)
        right = MVTSeries(mm(2024, 1), ["x", "y"], np.arange(4.0).reshape(2, 2))  # monthly (2, 2)
        out = left @ right
        assert out.shape == (3, 2)
        np.testing.assert_array_equal(out, left.values @ right.values)


# ---------------------------------------------------------------------------
# Shape mismatch / element-wise contract preservation
# ---------------------------------------------------------------------------


class TestMatmulShapeMismatch:
    """NumPy's matmul gufunc raises ``ValueError`` on incompatible shapes."""

    def test_matrix_tseries_inner_mismatch(self) -> None:
        t = _tseries(4)
        a = np.eye(5)  # (5, 5) — 5 != 4
        with pytest.raises(ValueError, match=r"(size|dimension)"):
            _ = a @ t

    def test_tseries_matrix_inner_mismatch(self) -> None:
        t = _tseries(4)
        a = np.eye(5)
        with pytest.raises(ValueError, match=r"(size|dimension)"):
            _ = t @ a

    def test_mvts_matrix_inner_mismatch(self) -> None:
        mvts = _mvts(4, ["a", "b", "c"])  # (4, 3)
        a = np.eye(5)  # (5, 5) — 5 != 3
        with pytest.raises(ValueError, match=r"(size|dimension)"):
            _ = mvts @ a


class TestElementwiseMultiplyUnaffected:
    """`*` and `@` are two different ops; this session must not touch ``*``."""

    def test_tseries_times_scalar(self) -> None:
        t = _tseries(4)
        out = t * 2.0
        assert isinstance(out, TSeries)
        np.testing.assert_array_equal(out.values, t.values * 2.0)

    def test_tseries_times_tseries_intersects_range(self) -> None:
        # Element-wise * still does range intersection (frequency-checked).
        t1 = TSeries(MITRange(qq(2020, 1), qq(2020, 1) + 3), np.array([1.0, 2.0, 3.0, 4.0]))
        t2 = TSeries(MITRange(qq(2020, 2), qq(2020, 2) + 3), np.array([10.0, 20.0, 30.0, 40.0]))
        out = t1 * t2
        assert isinstance(out, TSeries)
        # Intersection: t1[Q2..Q4] * t2[Q2..Q4] = [2*10, 3*20, 4*30]
        np.testing.assert_array_equal(out.values, np.array([20.0, 60.0, 120.0]))

    def test_mvts_times_scalar(self) -> None:
        mvts = _mvts(3, ["a", "b"])
        out = mvts * 3.0
        assert isinstance(out, MVTSeries)
        np.testing.assert_array_equal(out.values, mvts.values * 3.0)


# ---------------------------------------------------------------------------
# Hypothesis property — associativity of matmul on TSeries
# ---------------------------------------------------------------------------


@st.composite
def _square_matrix_and_tseries(
    draw: st.DrawFn,
    *,
    min_n: int = 2,
    max_n: int = 8,
) -> tuple[np.ndarray, TSeries]:
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    a = np.asarray(
        draw(
            st.lists(
                st.lists(
                    st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
                    min_size=n,
                    max_size=n,
                ),
                min_size=n,
                max_size=n,
            )
        ),
        dtype=float,
    )
    vals = np.asarray(
        draw(
            st.lists(
                st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            )
        ),
        dtype=float,
    )
    t = TSeries(MITRange(qq(2020, 1), qq(2020, 1) + (n - 1)), vals)
    return a, t


class TestMatmulProperties:
    """Hypothesis-checked invariants of the matmul overload."""

    @settings(max_examples=80, deadline=None)
    @given(_square_matrix_and_tseries())
    def test_associativity_matrix_matrix_tseries(self, payload: tuple[np.ndarray, TSeries]) -> None:
        """``A @ (A @ t) == (A @ A) @ t`` (within float tolerance)."""
        a, t = payload
        lhs = a @ (a @ t)
        rhs = (a @ a) @ t
        np.testing.assert_allclose(lhs, rhs, rtol=1e-9, atol=1e-9)

    @settings(max_examples=80, deadline=None)
    @given(_square_matrix_and_tseries())
    def test_identity_left(self, payload: tuple[np.ndarray, TSeries]) -> None:
        """``I @ t == t.values`` (no labels survive)."""
        _, t = payload
        identity = np.eye(len(t))
        out = identity @ t
        assert isinstance(out, np.ndarray)
        assert not isinstance(out, TSeries)
        np.testing.assert_allclose(out, t.values, rtol=1e-12, atol=1e-12)

    @settings(max_examples=80, deadline=None)
    @given(_square_matrix_and_tseries())
    def test_identity_right(self, payload: tuple[np.ndarray, TSeries]) -> None:
        """``t @ I == t.values``."""
        _, t = payload
        identity = np.eye(len(t))
        out = t @ identity
        assert isinstance(out, np.ndarray)
        np.testing.assert_allclose(out, t.values, rtol=1e-12, atol=1e-12)
