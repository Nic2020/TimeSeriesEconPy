# SPDX-License-Identifier: MIT
"""TimeSeriesEcon plotting adapter.

The public entry point is :func:`plot`, which routes a sequence of
:class:`~tsecon.tseries.TSeries` or :class:`~tsecon.mvtseries.MVTSeries`
arguments to one of the bundled backends (``matplotlib`` or ``plotly``)
and returns that backend's native figure object. Methods
:meth:`~tsecon.tseries.TSeries.plot` and
:meth:`~tsecon.mvtseries.MVTSeries.plot` delegate here.

The backends are *optional* dependencies (extras ``matplotlib`` and
``plotly`` in ``pyproject.toml``); neither is imported at package import
time. ``backend="auto"`` resolves to the first installed backend in the
preference order ``matplotlib -> plotly``; passing ``backend="matplotlib"``
or ``backend="plotly"`` raises :class:`ImportError` with an install hint
if that backend is missing.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Sequence
from typing import Any, Literal

from tsecon.mitrange import MITRange
from tsecon.mvtseries import MVTSeries
from tsecon.plotting._common import (
    MitLoc,
    classify_inputs,
)
from tsecon.tseries import TSeries

__all__ = [
    "BackendName",
    "BackendNotAvailableError",
    "available_backends",
    "plot",
    "resolve_backend",
]

BackendName = Literal["auto", "matplotlib", "plotly"]
"""Named backends supported by :func:`plot`."""

_PREFERENCE_ORDER: tuple[Literal["matplotlib", "plotly"], ...] = ("matplotlib", "plotly")


class BackendNotAvailableError(ImportError):
    """Raised when a requested plotting backend is not installed."""


def _is_installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def available_backends() -> tuple[Literal["matplotlib", "plotly"], ...]:
    """Return the names of installed plotting backends in preference order."""
    return tuple(name for name in _PREFERENCE_ORDER if _is_installed(name))


def resolve_backend(backend: BackendName) -> Literal["matplotlib", "plotly"]:
    """Resolve ``backend="auto"`` to the first available backend.

    Raises :class:`BackendNotAvailableError` if the requested backend (or
    ``auto`` with no backends installed) is not available, with a
    pip-extras install hint.
    """
    if backend == "auto":
        for name in _PREFERENCE_ORDER:
            if _is_installed(name):
                return name
        msg = (
            "No plotting backend is installed. Install one of:\n"
            "  pip install 'TimeSeriesEconPy[matplotlib]'   # static figures\n"
            "  pip install 'TimeSeriesEconPy[plotly]'       # interactive notebooks\n"
            "  pip install 'TimeSeriesEconPy[all]'          # both"
        )
        raise BackendNotAvailableError(msg)
    if backend not in _PREFERENCE_ORDER:
        msg = f"Unknown backend {backend!r}. Expected one of 'auto', 'matplotlib', 'plotly'."
        raise ValueError(msg)
    if not _is_installed(backend):
        msg = (
            f"Plotting backend {backend!r} is not installed. Install it with:\n"
            f"  pip install 'TimeSeriesEconPy[{backend}]'"
        )
        raise BackendNotAvailableError(msg)
    return backend


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------


def plot(
    *series: TSeries | MVTSeries,
    backend: BackendName = "auto",
    trange: MITRange | None = None,
    mit_loc: MitLoc = "left",
    label: str | Sequence[str] | None = None,
    vars: Sequence[str | tuple[str, str]] | None = None,  # matches Julia kwarg name
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
) -> Any:
    """Plot one or more TSeries (single-axes) or MVTSeries (panel-grid) datasets.

    Parameters
    ----------
    *series : TSeries or MVTSeries
        One or more series of a single kind — mixing TSeries and MVTSeries in
        one call raises :class:`TypeError`, matching the Julia recipe arity.
    backend : {"auto", "matplotlib", "plotly"}
        ``"auto"`` (default) picks the first installed backend in the order
        matplotlib → plotly.
    trange : MITRange, optional
        Limit the rendered window to this range. All series must share a
        frequency.
    mit_loc : {"left", "middle", "right"}
        Where in each period interval the value's x-coordinate sits. For
        Daily / BDaily / Weekly the position is fixed at the period date
        and ``mit_loc`` is ignored.
    label : str or sequence of str, optional
        One label per series (TSeries) or per dataset (MVTSeries).
    vars : sequence of str or (str, str), optional
        For MVTSeries: select / order the variables shown. Each item is
        either a column name or a ``(name, subplot_title)`` pair. Capped at
        10 entries.
    title, xlabel, ylabel : str, optional
        Standard chart annotations. For panel plots, ``title`` is the
        super-title; ``xlabel`` / ``ylabel`` apply to every subplot.
    legend : bool
        Whether to draw a legend.
    figsize : (float, float), optional
        Matplotlib figure size (inches). The plotly backend interprets
        this as pixels at 100 DPI.

    Returns
    -------
    matplotlib.figure.Figure or plotly.graph_objects.Figure
        The backend's native figure object.
    """
    if not series:
        msg = "tsecon.plot() requires at least one TSeries or MVTSeries argument"
        raise TypeError(msg)
    kind = classify_inputs(series)
    backend_name = resolve_backend(backend)
    backend_mod = importlib.import_module(f"tsecon.plotting._{backend_name}")
    if kind == "tseries":
        if vars is not None:
            msg = "vars= is only valid for MVTSeries plotting, not TSeries"
            raise TypeError(msg)
        return backend_mod.plot_tseries_many(
            list(series),
            trange=trange,
            mit_loc=mit_loc,
            label=label,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            legend=legend,
            figsize=figsize,
            **kwargs,
        )
    return backend_mod.plot_mvtseries_panel(
        list(series),
        trange=trange,
        mit_loc=mit_loc,
        label=label,
        vars=vars,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        legend=legend,
        figsize=figsize,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Method bindings on TSeries / MVTSeries
# ---------------------------------------------------------------------------


def _tseries_plot(self: TSeries, **kwargs: Any) -> Any:
    """Plot this TSeries via :func:`tsecon.plot`. See its docstring for kwargs."""
    return plot(self, **kwargs)


def _mvtseries_plot(self: MVTSeries, **kwargs: Any) -> Any:
    """Plot this MVTSeries via :func:`tsecon.plot`. See its docstring for kwargs."""
    return plot(self, **kwargs)


TSeries.plot = _tseries_plot  # type: ignore[attr-defined]
MVTSeries.plot = _mvtseries_plot  # type: ignore[attr-defined]
