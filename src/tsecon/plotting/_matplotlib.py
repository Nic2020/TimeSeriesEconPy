# SPDX-License-Identifier: MIT
"""Matplotlib backend for :func:`tsecon.plot`.

Renders TSeries (single or many) and MVTSeries (panel grid) onto a
``matplotlib.figure.Figure``. YP frequencies use a numeric x-axis with a
:class:`matplotlib.ticker.FuncFormatter`; calendar frequencies
(Daily / BDaily / Weekly) use a datetime axis; Unit frequencies use an
integer axis.

Each public function returns the constructed :class:`matplotlib.figure.Figure`
so the caller can save / display / further customise it. Passing ``ax=`` (for
the single/many-TSeries entry point) or ``fig=`` (for the panel entry point)
reuses an existing axes / figure.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

from tsecon.frequencies import YPFrequency
from tsecon.mitrange import MITRange, rangeof_span
from tsecon.mvtseries import MVTSeries
from tsecon.plotting._common import (
    MitLoc,
    build_xy,
    check_uniform_frequency,
    intersect_ranges,
    mit_formatter,
    mit_offset,
    normalize_label,
    normalize_vars,
    panel_grid,
    pick_yp_stride,
)
from tsecon.tseries import TSeries

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "plot_mvtseries_panel",
    "plot_tseries_many",
]


def _apply_yp_axis(ax: Axes, rng: MITRange, mit_loc: MitLoc) -> None:
    """Configure a matplotlib Axes for a YP numeric axis.

    Pairs :class:`matplotlib.ticker.FuncFormatter` (relabels float ticks
    as MIT strings) with a :class:`matplotlib.ticker.MultipleLocator`
    whose step equals the YP grid step ``stride / N`` and whose offset
    equals :func:`mit_offset`. This makes matplotlib's auto-locator pick
    ticks that *land on* the YP grid, so the off-grid ``"+"`` suffix
    only fires for ticks the *user* explicitly sets — not for the
    default tick choice.

    The stride comes from :func:`pick_yp_stride`, so both matplotlib and
    plotly emit identical tick positions for the same range.
    """
    freq = rng.frequency
    if not isinstance(freq, YPFrequency):
        return
    n = freq.periods_per_year
    ax.xaxis.set_major_formatter(FuncFormatter(mit_formatter(mit_loc, freq)))
    if len(rng) > 0:
        stride = pick_yp_stride(len(rng), n)
        base = stride / n  # year-fraction units between consecutive ticks
        offset = mit_offset(mit_loc, freq)
        ax.xaxis.set_major_locator(MultipleLocator(base=base, offset=offset))


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
    ax: Any = None,  # matplotlib.axes.Axes
    **kwargs: Any,
) -> Any:
    """Plot one or more :class:`TSeries` onto a single Axes.

    All series must share a frequency. The function returns the parent
    :class:`matplotlib.figure.Figure`. When ``ax`` is passed, that Axes is
    reused (and its parent Figure is returned).

    ``kwargs`` are forwarded as-is to :meth:`matplotlib.axes.Axes.plot`.
    """
    if not series:
        msg = "plot_tseries_many requires at least one TSeries"
        raise ValueError(msg)
    check_uniform_frequency(series)
    labels = normalize_label(label, len(series))

    target_ax: Axes
    fig: Any
    if ax is None:
        fig, target_ax = plt.subplots(figsize=figsize)
    else:
        target_ax = ax
        fig = target_ax.figure

    kind: str | None = None
    plotted_ranges: list[MITRange] = []
    for t, lab in zip(series, labels, strict=True):
        x, y, k = build_xy(t, trange=trange, mit_loc=mit_loc)
        # Skip empty (all-trimmed) series so matplotlib's plot() doesn't
        # raise on an empty datetime axis with an inherited locator.
        if len(x) == 0:
            continue
        kind = k
        plotted_ranges.append(t.range if trange is None else intersect_ranges(t.range, trange))
        target_ax.plot(x, y, label=lab if lab is not None else None, **kwargs)

    if kind == "yp" and plotted_ranges:
        _apply_yp_axis(target_ax, rangeof_span(*plotted_ranges), mit_loc)

    if title is not None:
        target_ax.set_title(title)
    if xlabel is not None:
        target_ax.set_xlabel(xlabel)
    if ylabel is not None:
        target_ax.set_ylabel(ylabel)
    if legend and any(lab is not None for lab in labels):
        target_ax.legend()
    return fig


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
    fig: Any = None,  # matplotlib.figure.Figure
    **kwargs: Any,
) -> Any:
    """Plot one or more :class:`MVTSeries` as a panel of subplots, one per variable.

    All datasets must share a frequency. The variable list is the union of
    column names across datasets in first-seen order (or the user-supplied
    ``vars`` list). Variables missing from a dataset are skipped for that
    dataset's trace; the subplot is still drawn so other datasets show.

    Returns the :class:`matplotlib.figure.Figure`. Passing ``fig=`` reuses
    an existing figure (its axes are cleared first to avoid duplicate
    overlays).
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

    target_fig: Figure
    if fig is None:
        target_fig = plt.figure(figsize=figsize)
    else:
        target_fig = fig
        target_fig.clear()

    axes_list: list[Axes] = []
    for ind in range(nvars):
        axes_list.append(target_fig.add_subplot(nrows, ncols, ind + 1))

    for (vname, vtitle), ax in zip(var_pairs, axes_list, strict=True):
        kind: str | None = None
        subplot_ranges: list[MITRange] = []
        for d, lab in zip(datasets, labels, strict=True):
            if vname not in d.column_names:
                continue
            col = d.columns[vname]
            x, y, k = build_xy(col, trange=trange, mit_loc=mit_loc)
            if len(x) == 0:
                continue
            kind = k
            subplot_ranges.append(
                col.range if trange is None else intersect_ranges(col.range, trange)
            )
            ax.plot(x, y, label=lab, **kwargs)
        ax.set_title(vtitle)
        if kind == "yp" and subplot_ranges:
            _apply_yp_axis(ax, rangeof_span(*subplot_ranges), mit_loc)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if legend and len(datasets) > 1:
            ax.legend()
    if title is not None:
        target_fig.suptitle(title)
    target_fig.tight_layout()
    return target_fig
