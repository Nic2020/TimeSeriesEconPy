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
from matplotlib.ticker import FuncFormatter

from tsecon.mitrange import MITRange
from tsecon.mvtseries import MVTSeries
from tsecon.plotting._common import (
    MitLoc,
    build_xy,
    check_uniform_frequency,
    mit_formatter,
    normalize_label,
    normalize_vars,
    panel_grid,
)
from tsecon.tseries import TSeries

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "plot_mvtseries_panel",
    "plot_tseries_many",
]


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
    freq = check_uniform_frequency(series)
    labels = normalize_label(label, len(series))

    target_ax: Axes
    fig: Any
    if ax is None:
        fig, target_ax = plt.subplots(figsize=figsize)
    else:
        target_ax = ax
        fig = target_ax.figure

    kind: str | None = None
    for t, lab in zip(series, labels, strict=True):
        x, y, k = build_xy(t, trange=trange, mit_loc=mit_loc)
        # Skip empty (all-trimmed) series so matplotlib's plot() doesn't
        # raise on an empty datetime axis with an inherited locator.
        if len(x) == 0:
            continue
        kind = k
        target_ax.plot(x, y, label=lab if lab is not None else None, **kwargs)

    if kind == "yp":
        target_ax.xaxis.set_major_formatter(FuncFormatter(mit_formatter(mit_loc, freq)))

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
    freq = check_uniform_frequency(datasets)
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
        for d, lab in zip(datasets, labels, strict=True):
            if vname not in d.column_names:
                continue
            col = d.columns[vname]
            x, y, k = build_xy(col, trange=trange, mit_loc=mit_loc)
            if len(x) == 0:
                continue
            kind = k
            ax.plot(x, y, label=lab, **kwargs)
        ax.set_title(vtitle)
        if kind == "yp":
            ax.xaxis.set_major_formatter(FuncFormatter(mit_formatter(mit_loc, freq)))
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
