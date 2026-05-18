# SPDX-License-Identifier: MIT
"""Statistics reductions for :class:`TSeries` and :class:`MVTSeries`.

Mirrors the ``Statistics.*`` overloads in ``TimeSeriesEcon.jl/src/tsmath.jl``
(lines 238-298). Six TSeries reductions (``mean`` / ``std`` / ``var`` /
``median`` / ``quantile`` / ``stdm`` / ``varm``) and a ``cor`` / ``cov``
pair (both for one and two arguments). MVTSeries has the same six
single-argument reductions; ``cor(MVTSeries)`` / ``cov(MVTSeries)`` return
column-correlation / column-covariance matrices.

For non-BDaily series, the functions are thin wrappers around the
corresponding ``numpy`` calls. For :class:`~tsecon.frequencies.BDaily` series
they additionally accept ``skip_all_nans`` / ``skip_holidays`` /
``holidays_map`` kwargs that filter the underlying values through
:func:`tsecon._bdaily.cleanedvalues` before the reduction. Passing those
kwargs on a non-BDaily series raises :class:`TypeError`.

The Julia overloads return scalar values for the single-series form and
matrices for the MVTSeries ``cor`` / ``cov``. We follow the same shape; the
column orientation matches Julia (variables = columns) by passing
``rowvar=False`` to ``np.corrcoef`` / ``np.cov``.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np

from tsecon._bdaily import cleanedvalues
from tsecon._kernel_dispatch import _dispatch_kernel, _is_kernel_eligible
from tsecon._stats_kernels import cor_numpy, mean_numpy, std_numpy, var_numpy
from tsecon.frequencies import BDaily
from tsecon.mvtseries import MVTSeries
from tsecon.tseries import TSeries

# Try to load the optional Cython-compiled accelerators. When the wheel was
# built without the C toolchain (or for editable installs that skipped the
# build hook), the import fails silently and the public API falls back to
# the NumPy reference. The user-facing surface is otherwise unchanged — this
# is the "always-fast public API" arm of [decision 17](claude_files/decisions/
# 17_cython_dispatch_strategy.md).
try:
    from tsecon._stats_kernels_cy import (  # type: ignore[import-not-found, unused-ignore]
        cor_cython,
        mean_cython,
        std_cython,
        var_cython,
    )

    _CYTHON_AVAILABLE: Final[bool] = True
except ImportError:
    _CYTHON_AVAILABLE = False  # type: ignore[misc]
    # Stub the cython names to their NumPy siblings so call sites that pass
    # `<op>_cython` into `_dispatch_kernel` resolve cleanly when the
    # extension isn't built. `_CYTHON_AVAILABLE` remains the single truth
    # of which path actually runs; the stubs are never reached because the
    # dispatcher short-circuits on the flag.
    cor_cython = cor_numpy
    mean_cython = mean_numpy
    std_cython = std_numpy
    var_cython = var_numpy

__all__ = [
    "cor",
    "cov",
    "mean",
    "median",
    "quantile",
    "stats_is_cython",
    "std",
    "stdm",
    "var",
    "varm",
]


def stats_is_cython() -> bool:
    """Return True iff the Cython-compiled stats kernels were importable.

    Useful for tests, benchmarks, and diagnostic prints — the public
    :func:`mean` / :func:`var` / :func:`std` / :func:`cor` are
    implementation-agnostic. When this returns ``False`` the same calls
    go through the pure-NumPy kernels in ``_stats_kernels.py``;
    behaviour is identical, only speed differs.
    """
    return _CYTHON_AVAILABLE


def _ravel_for_kernel(values: np.ndarray) -> np.ndarray | None:
    """Return a 1-D contiguous float64 view of ``values`` or ``None``.

    Mirrors Julia's ``mean(::MVTSeries)`` which iterates the matrix
    flat: a contiguous 2-D float64 array can be raveled into a 1-D view
    without copying, so the kernels apply equally to TSeries and to the
    overall reduction over an MVTSeries.
    """
    if values.dtype != np.float64:
        return None
    if values.ndim == 1 and values.flags["C_CONTIGUOUS"]:
        return values
    if values.ndim == 2 and values.flags["C_CONTIGUOUS"]:
        return values.ravel()
    return None


def _resolve_values(
    t: TSeries | MVTSeries,
    *,
    skip_all_nans: bool,
    skip_holidays: bool,
    holidays_map: TSeries | None,
) -> np.ndarray:
    if skip_all_nans or skip_holidays or holidays_map is not None:
        if not isinstance(t.frequency, BDaily):
            msg = (
                "skip_all_nans / skip_holidays / holidays_map are only valid for "
                f"BDaily series; got frequency {type(t.frequency).__name__}."
            )
            raise TypeError(msg)
        return cleanedvalues(
            t,
            skip_all_nans=skip_all_nans,
            skip_holidays=skip_holidays,
            holidays_map=holidays_map,
        )
    return np.asarray(t.values)


def mean(
    t: TSeries | MVTSeries,
    *,
    skip_all_nans: bool = False,
    skip_holidays: bool = False,
    holidays_map: TSeries | None = None,
) -> Any:
    """Arithmetic mean.

    For a :class:`~tsecon.tseries.TSeries` returns a scalar; for an
    :class:`~tsecon.mvtseries.MVTSeries` returns the overall mean (matching
    Julia's ``mean(::MVTSeries)`` which iterates the matrix flat). Use
    ``np.mean(mvts.values, axis=0)`` for per-column means.

    Parameters
    ----------
    t : TSeries or MVTSeries
        Input series. Any non-Unit frequency. For an MVTSeries the
        2-D values matrix is reduced flat.
    skip_all_nans : bool, default False
        BDaily only. If ``True``, drop entries that are NaN in
        ``t.values`` before reducing. ``TypeError`` on non-BDaily.
    skip_holidays : bool, default False
        BDaily only. If ``True``, drop entries on the dates marked
        by the holiday map (the explicit ``holidays_map`` if given,
        otherwise ``getoption('bdaily_holidays_map')``).
    holidays_map : TSeries, optional
        BDaily only. Boolean TSeries flagging the business days to
        include (``True``) / drop (``False``); ``AND``-ed with the
        NaN mask when ``skip_all_nans=True``.

    Returns
    -------
    float
        The arithmetic mean over the resolved values.

    Examples
    --------
    >>> import numpy as np
    >>> import tsecon as tse
    >>> t = tse.TSeries(tse.qq(2020, 1), np.array([1.0, 2.0, 3.0, 4.0]))
    >>> tse.mean(t)
    2.5
    """
    values = _resolve_values(
        t, skip_all_nans=skip_all_nans, skip_holidays=skip_holidays, holidays_map=holidays_map
    )
    flat = _ravel_for_kernel(values)
    if flat is not None and flat.shape[0] > 0:
        return _dispatch_kernel(_CYTHON_AVAILABLE, mean_cython, mean_numpy, flat)
    return np.mean(values)


def std(
    t: TSeries | MVTSeries,
    *,
    skip_all_nans: bool = False,
    skip_holidays: bool = False,
    holidays_map: TSeries | None = None,
) -> Any:
    """Sample standard deviation.

    Uses ``ddof=1`` to match Julia's ``Statistics.std`` (``corrected=true``
    by default). Pass ``np.std(t.values, ddof=0)`` directly if you need the
    population deviation.

    Parameters
    ----------
    t : TSeries or MVTSeries
        Input series. Any non-Unit frequency. For an MVTSeries the
        2-D values matrix is reduced flat.
    skip_all_nans : bool, default False
        BDaily only. If ``True``, drop NaN entries before reducing.
    skip_holidays : bool, default False
        BDaily only. If ``True``, drop entries marked by the holiday map.
    holidays_map : TSeries, optional
        BDaily only. Boolean TSeries flagging which business days to keep.

    Returns
    -------
    float
        Sample standard deviation with ``ddof=1``. Returns ``nan`` for
        an empty or single-element input.

    Examples
    --------
    >>> import numpy as np
    >>> import tsecon as tse
    >>> t = tse.TSeries(tse.qq(2020, 1), np.array([1.0, 2.0, 3.0]))
    >>> tse.std(t)
    1.0
    """
    values = _resolve_values(
        t, skip_all_nans=skip_all_nans, skip_holidays=skip_holidays, holidays_map=holidays_map
    )
    flat = _ravel_for_kernel(values)
    if flat is not None and flat.shape[0] > 1:
        return _dispatch_kernel(_CYTHON_AVAILABLE, std_cython, std_numpy, flat, 1)
    return np.std(values, ddof=1)


def var(
    t: TSeries | MVTSeries,
    *,
    skip_all_nans: bool = False,
    skip_holidays: bool = False,
    holidays_map: TSeries | None = None,
) -> Any:
    """Sample variance (``ddof=1`` to match Julia's ``Statistics.var``).

    Parameters
    ----------
    t : TSeries or MVTSeries
        Input series. Any non-Unit frequency. For an MVTSeries the
        2-D values matrix is reduced flat.
    skip_all_nans : bool, default False
        BDaily only. If ``True``, drop NaN entries before reducing.
    skip_holidays : bool, default False
        BDaily only. If ``True``, drop entries marked by the holiday map.
    holidays_map : TSeries, optional
        BDaily only. Boolean TSeries flagging which business days to keep.

    Returns
    -------
    float
        Sample variance with ``ddof=1``. Returns ``nan`` for an empty
        or single-element input.

    Examples
    --------
    >>> import numpy as np
    >>> import tsecon as tse
    >>> t = tse.TSeries(tse.qq(2020, 1), np.array([1.0, 2.0, 3.0]))
    >>> tse.var(t)
    1.0
    """
    values = _resolve_values(
        t, skip_all_nans=skip_all_nans, skip_holidays=skip_holidays, holidays_map=holidays_map
    )
    flat = _ravel_for_kernel(values)
    if flat is not None and flat.shape[0] > 1:
        return _dispatch_kernel(_CYTHON_AVAILABLE, var_cython, var_numpy, flat, 1)
    return np.var(values, ddof=1)


def median(
    t: TSeries | MVTSeries,
    *,
    skip_all_nans: bool = False,
    skip_holidays: bool = False,
    holidays_map: TSeries | None = None,
) -> Any:
    """Median value.

    Mirrors Julia's ``Statistics.median``. For an even-length input the
    midpoint of the two middle values is returned.

    Parameters
    ----------
    t : TSeries or MVTSeries
        Input series. Any non-Unit frequency. For an MVTSeries the
        2-D values matrix is reduced flat.
    skip_all_nans : bool, default False
        BDaily only. If ``True``, drop NaN entries before reducing.
    skip_holidays : bool, default False
        BDaily only. If ``True``, drop entries marked by the holiday map.
    holidays_map : TSeries, optional
        BDaily only. Boolean TSeries flagging which business days to keep.

    Returns
    -------
    float
        The median of the resolved values.

    Examples
    --------
    >>> import numpy as np
    >>> import tsecon as tse
    >>> t = tse.TSeries(tse.qq(2020, 1), np.array([1.0, 2.0, 3.0, 4.0]))
    >>> float(tse.median(t))
    2.5
    """
    values = _resolve_values(
        t, skip_all_nans=skip_all_nans, skip_holidays=skip_holidays, holidays_map=holidays_map
    )
    return np.median(values)


def quantile(
    t: TSeries,
    p: float | np.ndarray,
    *,
    skip_all_nans: bool = False,
    skip_holidays: bool = False,
    holidays_map: TSeries | None = None,
) -> Any:
    """``p``-th quantile of the series' values.

    Mirrors Julia's ``Statistics.quantile``.

    Parameters
    ----------
    t : TSeries
        Input series. Any non-Unit frequency.
    p : float or array-like of float
        Probability (or probabilities) in ``[0, 1]``.
    skip_all_nans : bool, default False
        BDaily only. If ``True``, drop NaN entries before reducing.
    skip_holidays : bool, default False
        BDaily only. If ``True``, drop entries marked by the holiday map.
    holidays_map : TSeries, optional
        BDaily only. Boolean TSeries flagging which business days to keep.

    Returns
    -------
    float or ndarray
        Scalar when ``p`` is a scalar, ndarray (same shape as ``p``)
        otherwise.

    Examples
    --------
    >>> import numpy as np
    >>> import tsecon as tse
    >>> t = tse.TSeries(tse.qq(2020, 1), np.array([1.0, 2.0, 3.0, 4.0]))
    >>> float(tse.quantile(t, 0.5))
    2.5
    """
    values = _resolve_values(
        t, skip_all_nans=skip_all_nans, skip_holidays=skip_holidays, holidays_map=holidays_map
    )
    return np.quantile(values, p)


def stdm(
    t: TSeries,
    m: float,
    *,
    skip_all_nans: bool = False,
    skip_holidays: bool = False,
    holidays_map: TSeries | None = None,
) -> Any:
    """Sample standard deviation with the mean ``m`` supplied externally.

    Mirrors Julia's ``Statistics.stdm``. Use this when you already have
    the mean (perhaps a known population mean, or one shared across
    several reductions) and want to avoid a redundant pass over ``t``.

    Parameters
    ----------
    t : TSeries
        Input series. Any non-Unit frequency.
    m : float
        The mean to centre around. Not validated against ``mean(t)``.
    skip_all_nans : bool, default False
        BDaily only. If ``True``, drop NaN entries before reducing.
    skip_holidays : bool, default False
        BDaily only. If ``True``, drop entries marked by the holiday map.
    holidays_map : TSeries, optional
        BDaily only. Boolean TSeries flagging which business days to keep.

    Returns
    -------
    float
        ``sqrt(sum((t - m)**2) / (n - 1))``. Returns ``nan`` when
        ``n <= 1``.

    Examples
    --------
    >>> import numpy as np
    >>> import tsecon as tse
    >>> t = tse.TSeries(tse.qq(2020, 1), np.array([1.0, 2.0, 3.0]))
    >>> tse.stdm(t, 2.0)
    1.0
    """
    values = _resolve_values(
        t, skip_all_nans=skip_all_nans, skip_holidays=skip_holidays, holidays_map=holidays_map
    )
    diffs = np.asarray(values, dtype=float) - float(m)
    n = diffs.shape[0]
    if n <= 1:
        return float("nan")
    return float(np.sqrt(np.sum(diffs * diffs) / (n - 1)))


def varm(
    t: TSeries,
    m: float,
    *,
    skip_all_nans: bool = False,
    skip_holidays: bool = False,
    holidays_map: TSeries | None = None,
) -> Any:
    """Sample variance with the mean ``m`` supplied externally.

    Mirrors Julia's ``Statistics.varm``. The variance counterpart of
    :func:`stdm`.

    Parameters
    ----------
    t : TSeries
        Input series. Any non-Unit frequency.
    m : float
        The mean to centre around. Not validated against ``mean(t)``.
    skip_all_nans : bool, default False
        BDaily only. If ``True``, drop NaN entries before reducing.
    skip_holidays : bool, default False
        BDaily only. If ``True``, drop entries marked by the holiday map.
    holidays_map : TSeries, optional
        BDaily only. Boolean TSeries flagging which business days to keep.

    Returns
    -------
    float
        ``sum((t - m)**2) / (n - 1)``. Returns ``nan`` when ``n <= 1``.

    Examples
    --------
    >>> import numpy as np
    >>> import tsecon as tse
    >>> t = tse.TSeries(tse.qq(2020, 1), np.array([1.0, 2.0, 3.0]))
    >>> tse.varm(t, 2.0)
    1.0
    """
    values = _resolve_values(
        t, skip_all_nans=skip_all_nans, skip_holidays=skip_holidays, holidays_map=holidays_map
    )
    diffs = np.asarray(values, dtype=float) - float(m)
    n = diffs.shape[0]
    if n <= 1:
        return float("nan")
    return float(np.sum(diffs * diffs) / (n - 1))


def cor(
    x: TSeries | MVTSeries,
    y: TSeries | None = None,
    *,
    skip_all_nans: bool = False,
    skip_holidays: bool = False,
    holidays_map: TSeries | None = None,
) -> Any:
    """Pearson correlation.

    Mirrors Julia's ``Statistics.cor`` overloads in ``tsmath.jl``
    (lines 249-272 and 297). Two call shapes:

    * ``cor(t)`` — for a :class:`~tsecon.tseries.TSeries` this returns
      ``1.0`` (the trivial self-correlation). For an
      :class:`~tsecon.mvtseries.MVTSeries` it returns the column-
      correlation matrix (variables on rows = columns of the input).
    * ``cor(x, y)`` — two TSeries of the same frequency, firstdate and
      length; returns the scalar Pearson correlation between them.

    Parameters
    ----------
    x : TSeries or MVTSeries
        First operand (or sole operand in the ``cor(t)`` / ``cor(mvts)``
        forms).
    y : TSeries, optional
        Second operand. When given, both ``x`` and ``y`` must be
        TSeries with matching frequency, firstdate, and length.
    skip_all_nans : bool, default False
        BDaily only. If ``True``, drop NaN entries before reducing.
    skip_holidays : bool, default False
        BDaily only. If ``True``, drop entries marked by the holiday map.
    holidays_map : TSeries, optional
        BDaily only. Boolean TSeries flagging which business days to keep.

    Returns
    -------
    float or ndarray
        Scalar in the single-TSeries or two-TSeries forms; an ``(n, n)``
        correlation matrix in the MVTSeries form. Returns ``nan`` when
        fewer than two filtered values remain.

    Raises
    ------
    TypeError
        ``cor(x, y)`` called with non-TSeries arguments.
    ValueError
        ``cor(x, y)`` arguments differ in frequency, firstdate, or
        length; or the BDaily filters produce mismatched lengths.

    Notes
    -----
    Correlation of a constant series is mathematically undefined (zero
    variance in the denominator). ``cor(x, y)`` returns ``nan`` and emits
    a ``RuntimeWarning`` when either input is bit-exactly constant, matching
    NumPy's ``np.corrcoef`` semantics on FP-exact constant input. The
    explicit guard is uniform across the NumPy and Cython kernels, so the
    return is ``nan`` for both ``np.full(N, 1.0)`` (where ``np.corrcoef``
    itself would warn) and ``np.full(N, 1e-60)`` (where pairwise summation
    in ``np.corrcoef`` would otherwise silently return ``1.0``).

    Examples
    --------
    >>> import numpy as np
    >>> import tsecon as tse
    >>> x = tse.TSeries(tse.qq(2020, 1), np.array([1.0, 2.0, 3.0, 4.0]))
    >>> y = tse.TSeries(tse.qq(2020, 1), np.array([2.0, 4.0, 6.0, 8.0]))
    >>> float(tse.cor(x, y))
    1.0
    """
    if y is None:
        if isinstance(x, MVTSeries):
            values = _resolve_values(
                x,
                skip_all_nans=skip_all_nans,
                skip_holidays=skip_holidays,
                holidays_map=holidays_map,
            )
            return np.corrcoef(values, rowvar=False)
        # cor(t) on a single TSeries — Julia returns 1.0 (vector self-correlation).
        return 1.0
    if not isinstance(x, TSeries) or not isinstance(y, TSeries):
        msg = (
            "cor(x, y) requires two TSeries arguments; got "
            f"{type(x).__name__!r} and {type(y).__name__!r}."
        )
        raise TypeError(msg)
    if x.frequency != y.frequency or x.firstdate != y.firstdate or len(x) != len(y):
        msg = (
            "cor(x, y) requires same-frequency same-firstdate same-length TSeries. "
            "Call cor(cleanedvalues(x), cleanedvalues(y)) directly to compare misaligned data."
        )
        raise ValueError(msg)
    xv = _resolve_values(
        x, skip_all_nans=skip_all_nans, skip_holidays=skip_holidays, holidays_map=holidays_map
    )
    yv = _resolve_values(
        y, skip_all_nans=skip_all_nans, skip_holidays=skip_holidays, holidays_map=holidays_map
    )
    if xv.shape[0] != yv.shape[0]:
        msg = (
            "cor(x, y): filtered arrays have unequal lengths "
            f"({xv.shape[0]} vs {yv.shape[0]}); supply matched holidays / NaN filters."
        )
        raise ValueError(msg)
    if xv.shape[0] < 2:
        return float("nan")
    if _is_kernel_eligible(xv, yv):
        return _dispatch_kernel(_CYTHON_AVAILABLE, cor_cython, cor_numpy, xv, yv)
    return float(np.corrcoef(xv, yv)[0, 1])


def cov(
    x: TSeries | MVTSeries,
    y: TSeries | None = None,
    *,
    skip_all_nans: bool = False,
    skip_holidays: bool = False,
    holidays_map: TSeries | None = None,
) -> Any:
    """Sample covariance (``ddof=1`` to match Julia's ``Statistics.cov``).

    Same call shapes as :func:`cor`:

    * ``cov(t)`` for a TSeries collapses to :func:`var`.
    * ``cov(mvts)`` returns the column-covariance matrix.
    * ``cov(x, y)`` returns the scalar covariance between two TSeries
      of the same frequency, firstdate, and length.

    Parameters
    ----------
    x : TSeries or MVTSeries
        First operand (or sole operand in the ``cov(t)`` / ``cov(mvts)``
        forms).
    y : TSeries, optional
        Second operand. When given, both ``x`` and ``y`` must be
        TSeries with matching frequency, firstdate, and length.
    skip_all_nans : bool, default False
        BDaily only. If ``True``, drop NaN entries before reducing.
    skip_holidays : bool, default False
        BDaily only. If ``True``, drop entries marked by the holiday map.
    holidays_map : TSeries, optional
        BDaily only. Boolean TSeries flagging which business days to keep.

    Returns
    -------
    float or ndarray
        Scalar in the TSeries forms; an ``(n, n)`` covariance matrix
        in the MVTSeries form.

    Raises
    ------
    TypeError
        ``cov(x, y)`` called with non-TSeries arguments.
    ValueError
        ``cov(x, y)`` arguments differ in frequency, firstdate, or
        length; or the BDaily filters produce mismatched lengths.

    Examples
    --------
    >>> import numpy as np
    >>> import tsecon as tse
    >>> x = tse.TSeries(tse.qq(2020, 1), np.array([1.0, 2.0, 3.0]))
    >>> y = tse.TSeries(tse.qq(2020, 1), np.array([2.0, 4.0, 6.0]))
    >>> float(tse.cov(x, y))
    2.0
    """
    if y is None:
        if isinstance(x, MVTSeries):
            values = _resolve_values(
                x,
                skip_all_nans=skip_all_nans,
                skip_holidays=skip_holidays,
                holidays_map=holidays_map,
            )
            return np.cov(values, rowvar=False, ddof=1)
        # cov(t) → var(t) (scalar)
        return var(
            x,
            skip_all_nans=skip_all_nans,
            skip_holidays=skip_holidays,
            holidays_map=holidays_map,
        )
    if not isinstance(x, TSeries) or not isinstance(y, TSeries):
        msg = (
            "cov(x, y) requires two TSeries arguments; got "
            f"{type(x).__name__!r} and {type(y).__name__!r}."
        )
        raise TypeError(msg)
    if x.frequency != y.frequency or x.firstdate != y.firstdate or len(x) != len(y):
        msg = (
            "cov(x, y) requires same-frequency same-firstdate same-length TSeries. "
            "Call cov(cleanedvalues(x), cleanedvalues(y)) directly for misaligned data."
        )
        raise ValueError(msg)
    xv = _resolve_values(
        x, skip_all_nans=skip_all_nans, skip_holidays=skip_holidays, holidays_map=holidays_map
    )
    yv = _resolve_values(
        y, skip_all_nans=skip_all_nans, skip_holidays=skip_holidays, holidays_map=holidays_map
    )
    if xv.shape[0] != yv.shape[0]:
        msg = (
            "cov(x, y): filtered arrays have unequal lengths "
            f"({xv.shape[0]} vs {yv.shape[0]}); supply matched holidays / NaN filters."
        )
        raise ValueError(msg)
    if xv.shape[0] < 2:
        return float("nan")
    return float(np.cov(xv, yv, ddof=1)[0, 1])
