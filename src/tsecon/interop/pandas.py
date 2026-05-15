# SPDX-License-Identifier: MIT
"""pandas DataFrame interop (lazy / optional).

See :doc:`/decisions/05_dataframe_interop` for the design rationale. pandas
is *not* a hard dependency: this module raises :class:`ImportError` with an
install hint when called without pandas installed.

Public API
----------

* :func:`to_pandas` — ``TSeries`` / ``MVTSeries`` / ``Workspace`` →
  ``pd.Series`` / ``pd.DataFrame``.
* :func:`from_pandas` — ``pd.Series`` / ``pd.DataFrame`` → ``TSeries`` /
  ``MVTSeries`` / ``Workspace``.

Required DataFrame shape (for :func:`from_pandas`)
--------------------------------------------------

The time axis must live on **one** of:

* the row **index**, of one of the following types:

  * :class:`pd.PeriodIndex` — frequency inferred from ``.freqstr``; no
    ``freq=`` kwarg needed. Supports pandas 2.x (``Q-DEC``, ``M``, ...)
    *and* pandas 3.x (``QE-DEC``, ``ME``, ``YE-DEC``, ...) aliases.
  * :class:`pd.DatetimeIndex` — frequency cannot be auto-inferred; the
    ``freq=`` kwarg is **required** (one of :class:`~tsecon.Daily`,
    :class:`~tsecon.BDaily`, :class:`~tsecon.Weekly`, etc.).
  * :class:`pd.RangeIndex` or any integer index — interpreted as
    :class:`~tsecon.frequencies.Unit` by default; pass ``freq=Unit()``
    explicitly to be explicit.
  * Object dtype of :class:`~tsecon.MIT` instances — the frequency is
    read off the MITs themselves; all MITs must share a single
    frequency. This is the round-trip target of
    :func:`to_pandas(..., index="mit")`.

* a column named ``time_col`` (must be passed explicitly). The same
  type / dtype rules apply, with a Date / Datetime column matching
  :class:`pd.DatetimeIndex`.

The time axis must be **strictly ascending and contiguous** (no gaps, no
duplicates). Sparse or unsorted inputs raise :class:`ValueError`; callers
who need gap-filling should ``sort_index().asfreq(...)`` on the pandas
side first.

For **wide** DataFrames (default, ``wide=True``), every non-time column
becomes a variable. For **long** DataFrames (``wide=False``), ``time_col``,
``name_col``, and ``value_col`` are all required; the frame is pivoted
into wide form before conversion.

Return type
-----------

* :class:`~tsecon.TSeries` when the input is a Series.
* :class:`~tsecon.Workspace` when ``to_workspace=True`` (one named
  TSeries per non-time column, in DataFrame column order; per-column
  dtypes preserved).
* :class:`~tsecon.MVTSeries` otherwise (one column per variable, single
  shared dtype, numeric-only).

Output shape (for :func:`to_pandas`)
------------------------------------

The ``index`` argument controls how the time axis is materialised:

* ``"auto"`` — :class:`pd.PeriodIndex` for Yearly / Quarterly / Monthly /
  Weekly, :class:`pd.DatetimeIndex` (period-end dates) for Daily / BDaily,
  :class:`pd.RangeIndex` for Unit, and an object-dtype Index of MIT for
  HalfYearly (which pandas has no period analogue for).
* ``"mit"`` — always emit an object-dtype Index of MIT instances. The only
  fully lossless option (re-`from_pandas` round-trips exactly).
* ``"date"`` — always emit a :class:`pd.DatetimeIndex` of period-end dates
  (or period-begin dates if ``date_ref="begin"``). Lossy for non-Daily
  frequencies on round-trip unless ``freq=`` is passed back to
  :func:`from_pandas`.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from tsecon.frequencies import Frequency, HalfYearly, Unit
from tsecon.interop._common import (
    freq_from_pandas_freqstr,
    freq_to_pandas_freqstr,
    mit_from_date,
    mits_to_dates,
    supports_pandas_period,
)
from tsecon.mit import MIT
from tsecon.mitrange import MITRange
from tsecon.mvtseries import MVTSeries
from tsecon.tseries import TSeries
from tsecon.workspace import Workspace

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["from_pandas", "to_pandas"]


IndexKind = Literal["auto", "mit", "date"]

_INSTALL_HINT = "Install with: pip install 'TimeSeriesEconPy[pandas]'"


def _require_pandas() -> Any:
    """Lazy-import pandas with a helpful error if it isn't installed.

    Centralised so the install-hint string stays consistent everywhere.
    """
    if find_spec("pandas") is None:
        msg = f"tsecon.interop.pandas requires pandas. {_INSTALL_HINT}"
        raise ImportError(msg)
    return import_module("pandas")


# ---------------------------------------------------------------------------
# to_pandas
# ---------------------------------------------------------------------------


def to_pandas(
    obj: TSeries | MVTSeries | Workspace,
    *,
    index: IndexKind = "auto",
    date_ref: Literal["begin", "end"] = "end",
    name: str | None = None,
) -> pd.Series | pd.DataFrame:
    """Convert a tsecon container to a pandas Series or DataFrame.

    Parameters
    ----------
    obj
        A :class:`~tsecon.TSeries` (→ ``pd.Series``), a
        :class:`~tsecon.MVTSeries` (→ ``pd.DataFrame`` with one column per
        variable), or a :class:`~tsecon.Workspace` containing only
        TSeries / MVTSeries members of a common frequency
        (→ ``pd.DataFrame``).
    index
        Time-axis representation. See module docstring.
    date_ref
        Period-anchor when ``index="date"``: ``"end"`` (default) uses the
        last calendar day of each period; ``"begin"`` uses the first.
    name
        Optional ``Series.name`` override when converting a TSeries.

    Returns
    -------
    pd.Series or pd.DataFrame
        A Series for TSeries; a DataFrame otherwise.
    """
    pd = _require_pandas()
    if isinstance(obj, TSeries):
        return _tseries_to_series(pd, obj, index=index, date_ref=date_ref, name=name)
    if isinstance(obj, MVTSeries):
        return _mvtseries_to_frame(pd, obj, index=index, date_ref=date_ref)
    if isinstance(obj, Workspace):
        return _workspace_to_frame(pd, obj, index=index, date_ref=date_ref)
    msg = (  # type: ignore[unreachable]
        f"to_pandas accepts TSeries, MVTSeries, or Workspace; got {type(obj).__name__}."
    )
    raise TypeError(msg)


def _build_index(
    pd: Any,
    rng: MITRange,
    freq: Frequency,
    *,
    kind: IndexKind,
    date_ref: Literal["begin", "end"],
) -> pd.Index:
    if kind == "mit":
        return pd.Index(list(rng), dtype=object, name="mit")
    if kind == "date":
        return pd.DatetimeIndex(mits_to_dates(rng, ref=date_ref), name="date")
    if kind != "auto":
        msg = f"index must be 'auto', 'mit', or 'date'; got {kind!r}."  # type: ignore[unreachable]
        raise ValueError(msg)
    # auto branch
    if isinstance(freq, Unit):
        return pd.RangeIndex(
            start=rng.start.value,
            stop=rng.stop.value + 1,
            step=1,
            name="t",
        )
    if isinstance(freq, HalfYearly):
        # No pandas period analogue; fall back to MIT index (lossless).
        return pd.Index(list(rng), dtype=object, name="mit")
    if supports_pandas_period(freq):
        # PeriodIndex needs a list of period-end dates plus the freq alias.
        # pandas snaps the dates to the period that contains them.
        dates = mits_to_dates(rng, ref="end")
        return pd.PeriodIndex(dates, freq=freq_to_pandas_freqstr(freq), name="period")
    # Daily / BDaily → DatetimeIndex
    return pd.DatetimeIndex(mits_to_dates(rng, ref=date_ref), name="date")


def _tseries_to_series(
    pd: Any,
    t: TSeries,
    *,
    index: IndexKind,
    date_ref: Literal["begin", "end"],
    name: str | None,
) -> pd.Series:
    idx = _build_index(pd, t.range, t.frequency, kind=index, date_ref=date_ref)
    return pd.Series(np.asarray(t.values), index=idx, name=name)


def _mvtseries_to_frame(
    pd: Any,
    m: MVTSeries,
    *,
    index: IndexKind,
    date_ref: Literal["begin", "end"],
) -> pd.DataFrame:
    idx = _build_index(pd, m.range, m.frequency, kind=index, date_ref=date_ref)
    return pd.DataFrame(np.asarray(m.values), index=idx, columns=list(m.column_names))


def _workspace_to_frame(
    pd: Any,
    w: Workspace,
    *,
    index: IndexKind,
    date_ref: Literal["begin", "end"],
) -> pd.DataFrame:
    members: list[tuple[str, TSeries]] = []
    for k, v in w.items():
        if isinstance(v, TSeries):
            members.append((k, v))
        elif isinstance(v, MVTSeries):
            for cname in v.column_names:
                members.append((f"{k}.{cname}", v[cname]))
    if not members:
        msg = "Workspace.to_pandas requires at least one TSeries or MVTSeries member."
        raise ValueError(msg)
    # Validate a single frequency across members.
    head_freq = members[0][1].frequency
    for nm, t in members[1:]:
        if t.frequency != head_freq:
            msg = (
                f"Workspace.to_pandas requires a single frequency across members; "
                f"{members[0][0]!r} is {type(head_freq).__name__} but {nm!r} is "
                f"{type(t.frequency).__name__}."
            )
            raise TypeError(msg)
    # Span the union of ranges (NaN-pad outside each member's range).
    lo = min(t.firstdate.value for _, t in members)
    hi = max(t.lastdate.value for _, t in members)
    full_rng = MITRange(MIT(head_freq, lo), MIT(head_freq, hi))
    idx = _build_index(pd, full_rng, head_freq, kind=index, date_ref=date_ref)
    out = pd.DataFrame(index=idx)
    for nm, t in members:
        # Align via NumPy reindex: build a full-length nan-filled column,
        # write the in-range slice. Object-dtype TSeries values (e.g. bool)
        # are coerced to float for the NaN pad.
        column = np.full(len(full_rng), np.nan, dtype=np.float64)
        offset = t.firstdate.value - lo
        column[offset : offset + len(t.values)] = t.values.astype(np.float64, copy=False)
        out[nm] = column
    return out


# ---------------------------------------------------------------------------
# from_pandas
# ---------------------------------------------------------------------------


def from_pandas(
    obj: pd.Series | pd.DataFrame,
    *,
    freq: Frequency | None = None,
    wide: bool = True,
    time_col: str | None = None,
    name_col: str | None = None,
    value_col: str | None = None,
    to_workspace: bool = False,
) -> TSeries | MVTSeries | Workspace:
    """Convert a pandas Series / DataFrame to a tsecon container.

    Parameters
    ----------
    obj
        A ``pd.Series`` (→ TSeries) or ``pd.DataFrame``.
    freq
        Required when the time axis is a :class:`pd.DatetimeIndex` (or a
        date-typed column) that has no inferable period semantics — pandas
        DatetimeIndex without ``.freq`` cannot be safely auto-classified
        between Daily / Weekly / Monthly / Quarterly / Yearly. Optional
        when the index is a ``PeriodIndex`` (frequency inferred from
        ``.freqstr``) or an object index of MIT (frequency inferred from
        the MITs themselves).
    wide
        When ``True`` (the default) every non-time column becomes a
        variable of the resulting MVTSeries. When ``False`` the DataFrame
        is assumed to be in long format with ``time_col`` / ``name_col`` /
        ``value_col``, and is pivoted to wide before conversion.
    time_col
        Optional name of the column to use as the time axis. When unset,
        the DataFrame index is used.
    name_col
        Long-format only. Name of the column holding the per-row variable
        name. Required when ``wide=False``.
    value_col
        Long-format only. Name of the column holding the per-row value.
        Required when ``wide=False``.
    to_workspace
        When ``True`` and ``obj`` is a DataFrame, return a
        :class:`~tsecon.Workspace` with one named TSeries per non-time
        column rather than a single MVTSeries. Useful when round-tripping
        a heterogeneous Workspace through pandas or when downstream code
        expects a Workspace. The default (``False``) preserves the
        established single-container MVTSeries return for DataFrames.

    Returns
    -------
    TSeries or MVTSeries or Workspace
        A TSeries when the input is a Series; a Workspace when
        ``to_workspace=True``; an MVTSeries otherwise.
    """
    pd = _require_pandas()
    if isinstance(obj, pd.Series):
        if to_workspace:
            msg = "to_workspace=True requires a DataFrame, not a Series."
            raise TypeError(msg)
        return _series_to_tseries(pd, obj, freq=freq)
    if isinstance(obj, pd.DataFrame):
        if not wide:
            obj = _long_to_wide(pd, obj, time_col=time_col, name_col=name_col, value_col=value_col)
            # After pivot, the time axis lives on the index.
            time_col = None
        if to_workspace:
            return _frame_to_workspace(pd, obj, freq=freq, time_col=time_col)
        return _frame_to_mvtseries(pd, obj, freq=freq, time_col=time_col)
    msg = f"from_pandas accepts pd.Series or pd.DataFrame; got {type(obj).__name__}."
    raise TypeError(msg)


def _series_to_tseries(
    pd: Any,
    s: pd.Series,
    *,
    freq: Frequency | None,
) -> TSeries:
    mits = _index_to_mits(pd, s.index, freq=freq)
    return _build_tseries(mits, s.to_numpy(), name=str(s.name) if s.name is not None else None)


def _frame_to_mvtseries(
    pd: Any,
    df: pd.DataFrame,
    *,
    freq: Frequency | None,
    time_col: str | None,
) -> MVTSeries:
    mits, data_df = _split_frame_time_and_data(pd, df, freq=freq, time_col=time_col)
    return _build_mvtseries(mits, list(data_df.columns), data_df.to_numpy())


def _frame_to_workspace(
    pd: Any,
    df: pd.DataFrame,
    *,
    freq: Frequency | None,
    time_col: str | None,
) -> Workspace:
    """Build a Workspace with one TSeries member per non-time column.

    Members are added in the DataFrame's column order. Each TSeries spans
    the full DataFrame range; the per-column dtype is preserved
    (unlike :func:`_frame_to_mvtseries` which forces a single shared dtype
    across all columns).
    """
    mits, data_df = _split_frame_time_and_data(pd, df, freq=freq, time_col=time_col)
    _validate_contiguous(mits)
    if not mits:
        msg = "Cannot build a Workspace from an empty DataFrame."
        raise ValueError(msg)
    rng = MITRange(mits[0], mits[-1])
    out = Workspace()
    for col_name in data_df.columns:
        series = data_df[col_name]
        out[str(col_name)] = TSeries(rng, series.to_numpy())
    return out


def _split_frame_time_and_data(
    pd: Any,
    df: pd.DataFrame,
    *,
    freq: Frequency | None,
    time_col: str | None,
) -> tuple[list[MIT], pd.DataFrame]:
    """Resolve a DataFrame's time axis to a list of MITs and the data subframe.

    Encapsulates the "is the time axis on the index or in a column" branch
    so both :func:`_frame_to_mvtseries` and :func:`_frame_to_workspace`
    share the resolution logic.
    """
    if time_col is not None:
        if time_col not in df.columns:
            msg = f"time_col={time_col!r} is not a column of the DataFrame."
            raise KeyError(msg)
        time_values = df[time_col]
        data_df = df.drop(columns=[time_col])
        mits = _column_to_mits(pd, time_values, freq=freq)
    else:
        mits = _index_to_mits(pd, df.index, freq=freq)
        data_df = df
    return mits, data_df


def _long_to_wide(
    pd: Any,
    df: pd.DataFrame,
    *,
    time_col: str | None,
    name_col: str | None,
    value_col: str | None,
) -> pd.DataFrame:
    if time_col is None or name_col is None or value_col is None:
        msg = (
            "Long-format from_pandas (wide=False) requires explicit "
            "time_col, name_col, and value_col."
        )
        raise ValueError(msg)
    pivoted = df.pivot(index=time_col, columns=name_col, values=value_col)
    # Preserve column order of first appearance.
    seen: list[str] = []
    for nm in df[name_col]:
        snm = str(nm)
        if snm not in seen:
            seen.append(snm)
    return pivoted[seen]


# ---------------------------------------------------------------------------
# Index → MIT-list helpers
# ---------------------------------------------------------------------------


def _index_to_mits(pd: Any, idx: pd.Index, *, freq: Frequency | None) -> list[MIT]:
    if isinstance(idx, pd.PeriodIndex):
        return _period_index_to_mits(pd, idx)
    if isinstance(idx, pd.DatetimeIndex):
        if freq is None:
            msg = (
                "from_pandas cannot infer a frequency from a DatetimeIndex; pass freq= explicitly."
            )
            raise ValueError(msg)
        return [mit_from_date(d.date() if hasattr(d, "date") else d, freq) for d in idx]
    if isinstance(idx, pd.RangeIndex) or _is_integer_index(idx):
        if freq is None:
            freq = Unit()
        if not isinstance(freq, Unit):
            msg = (
                "Integer DataFrame index requires freq=Unit() (or be left unset to "
                f"default to Unit); got freq={type(freq).__name__}()."
            )
            raise ValueError(msg)
        return [MIT(freq, int(v)) for v in idx]
    # object dtype: look for MITs
    if idx.dtype == object:
        return _mit_object_index_to_mits(idx)
    msg = (
        f"from_pandas does not know how to interpret index of dtype {idx.dtype}; "
        "pass freq= and an index that is PeriodIndex / DatetimeIndex / int / MIT."
    )
    raise TypeError(msg)


def _column_to_mits(pd: Any, col: pd.Series, *, freq: Frequency | None) -> list[MIT]:
    if isinstance(col.dtype, pd.PeriodDtype):
        # Period column: a frequency is encoded in the dtype itself.
        inferred = freq_from_pandas_freqstr(col.dtype.freq.freqstr)
        return [mit_from_date(p.end_time.date(), inferred) for p in col]
    if col.dtype.kind == "M":  # datetime64
        if freq is None:
            msg = (
                "from_pandas cannot infer a frequency from a datetime column; "
                "pass freq= explicitly."
            )
            raise ValueError(msg)
        return [mit_from_date(pd.Timestamp(v).date(), freq) for v in col]
    if col.dtype.kind in ("i", "u"):
        if freq is None:
            freq = Unit()
        if not isinstance(freq, Unit):
            msg = "Integer time column requires freq=Unit()."
            raise ValueError(msg)
        return [MIT(freq, int(v)) for v in col]
    if col.dtype == object:
        sample = col.iloc[0] if len(col) else None
        if isinstance(sample, MIT):
            return _mit_object_index_to_mits(col)
    msg = (
        f"Cannot interpret time column of dtype {col.dtype}; "
        "use a PeriodIndex / DatetimeIndex / integer column / MIT-object column."
    )
    raise TypeError(msg)


def _period_index_to_mits(pd: Any, idx: pd.PeriodIndex) -> list[MIT]:
    freq = freq_from_pandas_freqstr(idx.freqstr)
    return [mit_from_date(p.end_time.date(), freq) for p in idx]


def _is_integer_index(idx: Any) -> bool:
    try:
        return bool(np.issubdtype(idx.dtype, np.integer))
    except TypeError:
        return False


def _mit_object_index_to_mits(idx: Any) -> list[MIT]:
    raw: list[Any] = list(idx)
    if not raw:
        msg = "Cannot infer frequency from empty MIT-object index."
        raise ValueError(msg)
    for v in raw:
        if not isinstance(v, MIT):
            msg = f"Object index contains non-MIT entry {v!r}."
            raise TypeError(msg)
    mits = [cast("MIT", v) for v in raw]
    head = mits[0].frequency
    for v in mits[1:]:
        if v.frequency != head:
            msg = (
                f"Mixed frequencies in MIT index: {type(head).__name__} and "
                f"{type(v.frequency).__name__}."
            )
            raise TypeError(msg)
    return mits


def _build_tseries(mits: list[MIT], values: np.ndarray, *, name: str | None) -> TSeries:
    _validate_contiguous(mits)
    _ = name  # Series name is not preserved on TSeries (frequency / firstdate is the identity).
    if not mits:
        msg = "Cannot build a TSeries from an empty index."
        raise ValueError(msg)
    rng = MITRange(mits[0], mits[-1])
    return TSeries(rng, values)


def _build_mvtseries(mits: list[MIT], colnames: list[Any], values: np.ndarray) -> MVTSeries:
    _validate_contiguous(mits)
    if not mits:
        msg = "Cannot build an MVTSeries from an empty DataFrame index."
        raise ValueError(msg)
    rng = MITRange(mits[0], mits[-1])
    str_names = [str(n) for n in colnames]
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return MVTSeries(rng, str_names, values)


def _validate_contiguous(mits: list[MIT]) -> None:
    """Ensure the MIT list is strictly ascending by 1 step (no gaps / dupes).

    pandas can produce sparse / unsorted indexes; we don't try to repair
    them silently. Callers wanting a gap-filled TSeries should ``sort_index``
    and ``asfreq`` (or ``resample``) on the pandas side first.
    """
    for prev, cur in pairwise(mits):
        if cur.value != prev.value + 1:
            msg = (
                "from_pandas requires a strictly ascending, contiguous time axis; "
                f"gap between {prev} and {cur}. Sort and fill before converting."
            )
            raise ValueError(msg)
