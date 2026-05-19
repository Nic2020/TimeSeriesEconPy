# SPDX-License-Identifier: MIT
"""polars DataFrame interop (lazy / optional).

polars is *not* a hard dependency: this module raises :class:`ImportError`
with an install hint when called without polars installed.

polars has no row-index concept, so the time axis is always a regular
column. The default name is ``"time"`` (override with ``time_col=``).

Public API
----------

* :func:`to_polars` — ``TSeries`` / ``MVTSeries`` / ``Workspace`` →
  :class:`polars.DataFrame`.
* :func:`from_polars` — :class:`polars.DataFrame` → ``TSeries`` /
  ``MVTSeries`` / ``Workspace``.

Required DataFrame shape (for :func:`from_polars`)
--------------------------------------------------

The DataFrame must contain a column named ``time_col`` (default
``"time"``) of one of the following dtypes:

* :class:`polars.Date` or :class:`polars.Datetime` — the ``freq=`` kwarg
  is **required**; polars carries no period semantics so the frequency
  cannot be auto-inferred. Pass a :class:`~tsecon.Daily` /
  :class:`~tsecon.BDaily` / :class:`~tsecon.Weekly` / :class:`~tsecon.Monthly` /
  :class:`~tsecon.Quarterly` / :class:`~tsecon.HalfYearly` /
  :class:`~tsecon.Yearly` instance.
* Any integer dtype — interpreted as :class:`~tsecon.frequencies.Unit`
  by default; pass ``freq=Unit()`` explicitly to be explicit. The
  column values are taken as the raw ``MIT.value`` integers.

The time column must be **strictly ascending and contiguous** (no gaps,
no duplicates). Sparse or unsorted input raises :class:`ValueError`;
callers who need gap-filling should pre-process with polars before
calling :func:`from_polars`.

For **wide** DataFrames (default, ``wide=True``), every non-time column
becomes a variable. For **long** DataFrames (``wide=False``), ``time_col``,
``name_col``, and ``value_col`` are all required; the frame is pivoted
into wide form before conversion.

Return type
-----------

* :class:`~tsecon.Workspace` when ``to_workspace=True`` (one named
  TSeries per non-time column, in DataFrame column order; per-column
  dtypes preserved).
* :class:`~tsecon.TSeries` when the wide form has a single non-time
  column (and ``to_workspace=False``).
* :class:`~tsecon.MVTSeries` otherwise.

Output shape (for :func:`to_polars`)
------------------------------------

The time column dtype is :class:`polars.Date` for Daily / BDaily / Weekly
and YP frequencies (using period-end dates by default; ``date_ref="begin"``
swaps to period-begin), and :class:`polars.Int64` for Unit (raw MIT
values).
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from tsecon.frequencies import Frequency, Unit
from tsecon.interop._common import mit_from_date, mits_to_dates
from tsecon.mit import MIT
from tsecon.mitrange import MITRange
from tsecon.mvtseries import MVTSeries
from tsecon.tseries import TSeries
from tsecon.workspace import Workspace

if TYPE_CHECKING:
    import polars as pl

__all__ = ["from_polars", "to_polars"]


_INSTALL_HINT = "Install with: pip install 'TimeSeriesEconPy[polars]'"


def _require_polars() -> Any:
    if find_spec("polars") is None:
        msg = f"tsecon.interop.polars requires polars. {_INSTALL_HINT}"
        raise ImportError(msg)
    return import_module("polars")


# ---------------------------------------------------------------------------
# to_polars
# ---------------------------------------------------------------------------


def to_polars(
    obj: TSeries | MVTSeries | Workspace,
    *,
    time_col: str = "time",
    date_ref: Literal["begin", "end"] = "end",
    value_col: str = "value",
) -> Any:
    """Convert a tsecon container to a polars DataFrame.

    Parameters
    ----------
    obj
        A :class:`~tsecon.TSeries` (→ DataFrame with ``time_col`` plus a
        single value column named ``value_col``), an
        :class:`~tsecon.MVTSeries` (→ DataFrame with ``time_col`` plus one
        column per variable), or a :class:`~tsecon.Workspace` of TSeries /
        MVTSeries members of a common frequency.
    time_col
        Name of the emitted time column. Default ``"time"``.
    date_ref
        Period-anchor for non-Unit frequencies. ``"end"`` (default) uses
        the last calendar day of each period; ``"begin"`` uses the first.
    value_col
        Name of the value column when converting a TSeries (ignored for
        MVTSeries / Workspace). Default ``"value"``.

    Returns
    -------
    polars.DataFrame
        Annotated as ``Any`` because polars's type stubs are not used in
        tsecon's mypy config; the runtime return is always ``pl.DataFrame``.
    """
    pl = _require_polars()
    if isinstance(obj, TSeries):
        return _tseries_to_frame(pl, obj, time_col=time_col, value_col=value_col, date_ref=date_ref)
    if isinstance(obj, MVTSeries):
        return _mvtseries_to_frame(pl, obj, time_col=time_col, date_ref=date_ref)
    if isinstance(obj, Workspace):
        return _workspace_to_frame(pl, obj, time_col=time_col, date_ref=date_ref)
    msg = (  # type: ignore[unreachable]
        f"to_polars accepts TSeries, MVTSeries, or Workspace; got {type(obj).__name__}."
    )
    raise TypeError(msg)


def _time_column(
    pl: Any,
    rng: MITRange,
    freq: Frequency,
    *,
    time_col: str,
    date_ref: Literal["begin", "end"],
) -> Any:
    # Returns ``pl.Series``; the return type is Any because polars is
    # ignore_missing_imports-typed by the project mypy config.
    if isinstance(freq, Unit):
        return pl.Series(time_col, [m.value for m in rng], dtype=pl.Int64)
    return pl.Series(time_col, mits_to_dates(rng, ref=date_ref), dtype=pl.Date)


def _tseries_to_frame(
    pl: Any,
    t: TSeries,
    *,
    time_col: str,
    value_col: str,
    date_ref: Literal["begin", "end"],
) -> Any:
    """Build a polars DataFrame for this branch.

    Return type is ``Any`` because polars's type stubs are ignored in the
    project mypy config; the runtime return is always ``pl.DataFrame``.
    """
    tcol = _time_column(pl, t.range, t.frequency, time_col=time_col, date_ref=date_ref)
    vcol = pl.Series(value_col, np.asarray(t.values))
    return pl.DataFrame([tcol, vcol])


def _mvtseries_to_frame(
    pl: Any,
    m: MVTSeries,
    *,
    time_col: str,
    date_ref: Literal["begin", "end"],
) -> Any:
    """Build a polars DataFrame for this branch.

    Return type is ``Any`` because polars's type stubs are ignored in the
    project mypy config; the runtime return is always ``pl.DataFrame``.
    """
    tcol = _time_column(pl, m.range, m.frequency, time_col=time_col, date_ref=date_ref)
    cols: list[Any] = [tcol]
    arr = np.asarray(m.values)
    for j, nm in enumerate(m.column_names):
        if nm == time_col:
            msg = (
                f"MVTSeries column name {nm!r} collides with time_col={time_col!r}; "
                "pass a different time_col= to to_polars."
            )
            raise ValueError(msg)
        cols.append(pl.Series(nm, arr[:, j]))
    return pl.DataFrame(cols)


def _workspace_to_frame(
    pl: Any,
    w: Workspace,
    *,
    time_col: str,
    date_ref: Literal["begin", "end"],
) -> Any:
    members: list[tuple[str, TSeries]] = []
    for k, v in w.items():
        if isinstance(v, TSeries):
            members.append((k, v))
        elif isinstance(v, MVTSeries):
            for cname in v.column_names:
                members.append((f"{k}.{cname}", v[cname]))
    if not members:
        msg = "Workspace.to_polars requires at least one TSeries or MVTSeries member."
        raise ValueError(msg)
    head_freq = members[0][1].frequency
    for nm, t in members[1:]:
        if t.frequency != head_freq:
            msg = (
                f"Workspace.to_polars requires a single frequency across members; "
                f"{members[0][0]!r} is {type(head_freq).__name__} but {nm!r} is "
                f"{type(t.frequency).__name__}."
            )
            raise TypeError(msg)
    lo = min(t.firstdate.value for _, t in members)
    hi = max(t.lastdate.value for _, t in members)
    full_rng = MITRange(MIT(head_freq, lo), MIT(head_freq, hi))
    tcol = _time_column(pl, full_rng, head_freq, time_col=time_col, date_ref=date_ref)
    out_cols: list[Any] = [tcol]
    for nm, t in members:
        if nm == time_col:
            msg = (
                f"Workspace member {nm!r} collides with time_col={time_col!r}; "
                "pass a different time_col= to to_polars."
            )
            raise ValueError(msg)
        column = np.full(len(full_rng), np.nan, dtype=np.float64)
        offset = t.firstdate.value - lo
        column[offset : offset + len(t.values)] = t.values.astype(np.float64, copy=False)
        out_cols.append(pl.Series(nm, column))
    return pl.DataFrame(out_cols)


# ---------------------------------------------------------------------------
# from_polars
# ---------------------------------------------------------------------------


def from_polars(
    obj: pl.DataFrame,
    *,
    freq: Frequency | None = None,
    wide: bool = True,
    time_col: str = "time",
    name_col: str | None = None,
    value_col: str | None = None,
    to_workspace: bool = False,
) -> TSeries | MVTSeries | Workspace:
    """Convert a polars DataFrame to a tsecon container.

    Parameters
    ----------
    obj
        A :class:`polars.DataFrame`. Must contain ``time_col``.
    freq
        Required when the time column dtype is :class:`polars.Date` /
        :class:`polars.Datetime` — polars carries no period semantics, so
        the frequency cannot be auto-inferred. For an integer time column
        the default is :class:`~tsecon.frequencies.Unit`.
    wide
        When ``True`` (the default) every non-time column becomes a
        variable. When ``False`` the DataFrame is in long format with
        ``time_col`` / ``name_col`` / ``value_col`` and is pivoted to wide.
    time_col
        Name of the time column. Default ``"time"``.
    name_col
        Long-format only.
    value_col
        Long-format only. Also used to decide the resulting type: a
        single-value-column wide DataFrame returns a TSeries; otherwise
        an MVTSeries is returned.
    to_workspace
        When ``True``, return a :class:`~tsecon.Workspace` with one named
        TSeries per non-time column. Preserves per-column dtypes (unlike
        the MVTSeries return, which forces a single shared dtype). Wins
        over the single-value-column TSeries shortcut.

    Returns
    -------
    TSeries or MVTSeries or Workspace
        A Workspace when ``to_workspace=True``; a TSeries when the wide
        form has a single non-time column; otherwise an MVTSeries.
    """
    pl = _require_polars()
    if not isinstance(obj, pl.DataFrame):
        msg = f"from_polars accepts pl.DataFrame; got {type(obj).__name__}."
        raise TypeError(msg)
    if not wide:
        obj = _long_to_wide(pl, obj, time_col=time_col, name_col=name_col, value_col=value_col)
    if time_col not in obj.columns:
        msg = f"time_col={time_col!r} not found in DataFrame columns ({list(obj.columns)})."
        raise KeyError(msg)
    mits = _column_to_mits(pl, obj[time_col], freq=freq)
    data_df = obj.drop(time_col)
    if not mits:
        msg = "Cannot build a tsecon container from an empty DataFrame."
        raise ValueError(msg)
    _validate_contiguous(mits)
    rng = MITRange(mits[0], mits[-1])
    if to_workspace:
        out = Workspace()
        for col_name in data_df.columns:
            out[str(col_name)] = TSeries(rng, data_df[col_name].to_numpy())
        return out
    if data_df.width == 1:
        col_name = data_df.columns[0]
        return TSeries(rng, data_df[col_name].to_numpy())
    arr = data_df.to_numpy()
    return MVTSeries(rng, list(data_df.columns), arr)


def _column_to_mits(pl: Any, col: pl.Series, *, freq: Frequency | None) -> list[MIT]:
    dtype = col.dtype
    if dtype in (pl.Date, pl.Datetime):
        if freq is None:
            msg = (
                "from_polars cannot infer a frequency from a Date / Datetime "
                "column; pass freq= explicitly."
            )
            raise ValueError(msg)
        dates = col.to_list()
        return [mit_from_date(_as_date(d), freq) for d in dates]
    if dtype.is_integer():
        if freq is None:
            freq = Unit()
        if not isinstance(freq, Unit):
            msg = "Integer time column requires freq=Unit()."
            raise ValueError(msg)
        return [MIT(freq, int(v)) for v in col.to_list()]
    msg = f"Cannot interpret time column of dtype {dtype}; use a Date / Datetime / integer column."
    raise TypeError(msg)


def _long_to_wide(
    pl: Any,
    df: pl.DataFrame,
    *,
    time_col: str,
    name_col: str | None,
    value_col: str | None,
) -> pl.DataFrame:
    if name_col is None or value_col is None:
        msg = (
            "Long-format from_polars (wide=False) requires explicit "
            "time_col, name_col, and value_col."
        )
        raise ValueError(msg)
    # Preserve column order of first appearance, then pivot.
    seen: list[str] = []
    for nm in df[name_col].to_list():
        snm = str(nm)
        if snm not in seen:
            seen.append(snm)
    pivoted = df.pivot(
        index=time_col,
        on=name_col,
        values=value_col,
        aggregate_function="first",
    )
    return pivoted.select([time_col, *seen])


def _as_date(d: Any) -> Any:
    """Coerce a polars Date / Datetime value to a stdlib ``datetime.date``."""
    if d is None:
        msg = "Time column contains a null entry."
        raise ValueError(msg)
    if hasattr(d, "date") and not hasattr(d, "isoweekday"):
        # datetime.datetime → date
        return d.date()
    return d


def _validate_contiguous(mits: list[MIT]) -> None:
    for prev, cur in pairwise(mits):
        if cur.value != prev.value + 1:
            msg = (
                "from_polars requires a strictly ascending, contiguous time axis; "
                f"gap between {prev} and {cur}."
            )
            raise ValueError(msg)
