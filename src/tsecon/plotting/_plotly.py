# SPDX-License-Identifier: MIT
"""Plotly backend for :func:`tsecon.plot`.

Renders TSeries (single or many) and MVTSeries (panel grid) onto a
:class:`plotly.graph_objects.Figure`. YP frequencies use a numeric x-axis
with explicit ``tickvals`` / ``ticktext`` (plotly cannot apply a Python
callable as a tick formatter to a numeric axis); calendar frequencies
(Daily / BDaily / Weekly) use a datetime axis; Unit frequencies use an
integer axis.

Each public function returns the constructed Figure so the caller can
``.show()`` / ``.write_html()`` / further customise it. Passing ``fig=``
reuses an existing Figure.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tsecon.mitrange import MITRange
from tsecon.mvtseries import MVTSeries
from tsecon.plotting._common import (
    MitLoc,
    build_xy,
    check_uniform_frequency,
    intersect_ranges,
    normalize_label,
    normalize_vars,
    panel_grid,
    yp_tick_positions,
)
from tsecon.tseries import TSeries

__all__ = [
    "plot_mvtseries_panel",
    "plot_tseries_many",
]


def _yp_axis_for(
    fig: go.Figure,
    rng: MITRange,
    mit_loc: MitLoc,
    *,
    axis_id: str = "xaxis",
) -> None:
    """Apply explicit YP tickvals/ticktext to ``fig.layout[axis_id]``.

    The two-argument plotly ``update_layout`` form keeps non-default
    axes (``xaxis2``, ``xaxis3``…) reachable for the panel backend.
    """
    tickvals, ticktext = yp_tick_positions(rng, mit_loc)
    fig.layout[axis_id].update(tickmode="array", tickvals=tickvals, ticktext=ticktext)


# ---------------------------------------------------------------------------
# Single / many TSeries
# ---------------------------------------------------------------------------


def plot_tseries_many(
    series: Sequence[TSeries],
    *,
    trange: MITRange | None = None,
    mit_loc: MitLoc = "left",
    label: str | Sequence[str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    figsize: tuple[float, float] | None = None,
    fig: go.Figure | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Plot one or more :class:`TSeries` onto a single Plotly figure.

    All series must share a frequency. ``kwargs`` are forwarded as-is to
    :class:`plotly.graph_objects.Scatter`. Returns a
    :class:`plotly.graph_objects.Figure`.
    """
    if not series:
        msg = "plot_tseries_many requires at least one TSeries"
        raise ValueError(msg)
    check_uniform_frequency(series)
    labels = normalize_label(label, len(series))

    target_fig = fig if fig is not None else go.Figure()
    plotted_kind: str | None = None
    plotted_rng: MITRange | None = None

    for t, lab in zip(series, labels, strict=True):
        x, y, kind = build_xy(t, trange=trange, mit_loc=mit_loc)
        if len(x) == 0:
            continue
        plotted_kind = kind
        plotted_rng = t.range if trange is None else _intersect_or_none(t.range, trange)
        showlegend = lab is not None
        target_fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=lab if lab is not None else "",
                showlegend=showlegend,
                **kwargs,
            )
        )

    layout_kwargs: dict[str, Any] = {}
    if title is not None:
        layout_kwargs["title"] = title
    if xlabel is not None:
        layout_kwargs["xaxis_title"] = xlabel
    if ylabel is not None:
        layout_kwargs["yaxis_title"] = ylabel
    layout_kwargs["showlegend"] = legend and any(lab is not None for lab in labels)
    if figsize is not None:
        # plotly figsize is in pixels; matplotlib uses inches. Scale at 100 DPI
        # so a (6, 4) inches figure becomes a (600, 400) pixel plotly figure.
        layout_kwargs["width"] = int(figsize[0] * 100)
        layout_kwargs["height"] = int(figsize[1] * 100)
    if layout_kwargs:
        target_fig.update_layout(**layout_kwargs)
    if plotted_kind == "yp" and plotted_rng is not None and len(plotted_rng) > 0:
        _yp_axis_for(target_fig, plotted_rng, mit_loc)
    return target_fig


def _intersect_or_none(a: MITRange, b: MITRange) -> MITRange | None:
    """Return ``intersect_ranges(a, b)`` or ``None`` if frequencies disagree.

    The recipe arm has already enforced uniform frequency by the time this
    is reached, so the ``None`` branch is defensive.
    """
    if a.frequency != b.frequency:
        return None
    return intersect_ranges(a, b)


# ---------------------------------------------------------------------------
# MVTSeries panel
# ---------------------------------------------------------------------------


def plot_mvtseries_panel(
    datasets: Sequence[MVTSeries],
    *,
    trange: MITRange | None = None,
    mit_loc: MitLoc = "left",
    label: str | Sequence[str] | None = None,
    vars: Sequence[str | tuple[str, str]] | None = None,  # matches Julia kwarg name
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    figsize: tuple[float, float] | None = None,
    fig: go.Figure | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Plot one or more :class:`MVTSeries` as a panel of subplots, one per variable.

    Returns a :class:`plotly.graph_objects.Figure` with the panel layout
    determined by :func:`tsecon.plotting._common.panel_grid`. Each dataset
    contributes one trace per variable subplot; missing variables are
    skipped per-dataset (the subplot still draws for the datasets that
    have it).
    """
    if not datasets:
        msg = "plot_mvtseries_panel requires at least one MVTSeries"
        raise ValueError(msg)
    check_uniform_frequency(datasets)
    raw_labels = normalize_label(label, len(datasets))
    labels: list[str] = [
        lab if lab is not None else f"data{i + 1}" for i, lab in enumerate(raw_labels)
    ]
    var_pairs = normalize_vars(vars, datasets)
    nvars = len(var_pairs)
    nrows, ncols = panel_grid(nvars)

    subplot_titles = [vtitle for _name, vtitle in var_pairs]
    target_fig: go.Figure
    if fig is None:
        target_fig = make_subplots(
            rows=nrows, cols=ncols, subplot_titles=subplot_titles, shared_xaxes=False
        )
    else:
        target_fig = fig

    # Track which dataset's legend entry has been emitted so the legend on
    # the assembled figure shows each dataset exactly once (plotly's
    # showlegend is per-trace; we suppress repeats across subplots).
    legend_seen: dict[str, bool] = dict.fromkeys(labels, False)

    for ind, (vname, _vtitle) in enumerate(var_pairs):
        row = ind // ncols + 1
        col = ind % ncols + 1
        axis_idx = ind + 1
        axis_id = "xaxis" if axis_idx == 1 else f"xaxis{axis_idx}"
        kind_seen: str | None = None
        rng_seen: MITRange | None = None
        for d, lab in zip(datasets, labels, strict=True):
            if vname not in d.column_names:
                continue
            t = d.columns[vname]
            x, y, kind = build_xy(t, trange=trange, mit_loc=mit_loc)
            if len(x) == 0:
                continue
            kind_seen = kind
            rng_seen = t.range if trange is None else _intersect_or_none(t.range, trange)
            showlegend = legend and not legend_seen[lab]
            target_fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=lab,
                    legendgroup=lab,
                    showlegend=showlegend,
                    **kwargs,
                ),
                row=row,
                col=col,
            )
            legend_seen[lab] = True
        if kind_seen == "yp" and rng_seen is not None and len(rng_seen) > 0:
            _yp_axis_for(target_fig, rng_seen, mit_loc, axis_id=axis_id)
        if xlabel is not None:
            target_fig.update_xaxes(title_text=xlabel, row=row, col=col)
        if ylabel is not None:
            target_fig.update_yaxes(title_text=ylabel, row=row, col=col)

    layout_kwargs: dict[str, Any] = {"showlegend": legend}
    if title is not None:
        layout_kwargs["title"] = title
    if figsize is not None:
        layout_kwargs["width"] = int(figsize[0] * 100)
        layout_kwargs["height"] = int(figsize[1] * 100)
    target_fig.update_layout(**layout_kwargs)
    return target_fig
