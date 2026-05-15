# SPDX-License-Identifier: MIT
"""Backend-shared plotting helpers.

Ports the ``mit_offset`` / ``mit_formatter`` logic from
``TimeSeriesEcon.jl/src/plotrecipes.jl`` and the input-normalisation surface
shared by every backend (``matplotlib``, ``plotly``):

* :func:`mit_offset` — within-period x-coordinate offset for a frequency.
* :func:`mit_formatter` — formatter callable that turns YP floats back into
  MIT strings, appending ``"+"`` when a tick is off-grid (matches Julia's
  ``mit_formatter`` recipe).
* :func:`build_xy` — turn a (TSeries, trange, mit_loc) triple into a
  (x, y) pair suitable for either backend.
* :func:`normalize_label`, :func:`normalize_vars`, :func:`panel_grid` —
  shared normalisations across MVTSeries panel plotting.
* :func:`classify_inputs` — split positional series into the ``all-TSeries``
  vs ``all-MVTSeries`` arms (no mixing — matches Julia's recipe arity).
"""

from __future__ import annotations

import datetime as _dt
import math
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Literal

import numpy as np

from tsecon.frequencies import (
    BDaily,
    CalendarFrequency,
    Daily,
    Frequency,
    Weekly,
    YPFrequency,
    ppy,
    prettyprint_frequency,
)
from tsecon.mit import MIT, mit_to_date
from tsecon.mitrange import MITRange
from tsecon.mvtseries import MVTSeries
from tsecon.tseries import TSeries

__all__ = [
    "AxisKind",
    "MitLoc",
    "PlotKind",
    "build_xy",
    "classify_inputs",
    "intersect_ranges",
    "mit_formatter",
    "mit_offset",
    "normalize_label",
    "normalize_vars",
    "panel_grid",
    "pick_yp_stride",
    "xaxis_kind",
    "yp_tick_positions",
]

MitLoc = Literal["left", "middle", "right"]
"""Where in the period interval the value's x-coordinate sits."""

AxisKind = Literal["yp", "date", "numeric"]
"""How the x-axis should be rendered.

* ``"yp"`` — float-numeric coordinates with a YP-aware tick formatter.
* ``"date"`` — :class:`datetime.date` coordinates (Daily / BDaily / Weekly).
* ``"numeric"`` — plain integer coordinates (Unit frequency).
"""

PlotKind = Literal["tseries", "mvtseries"]
"""Which recipe arm the caller selected."""


# ---------------------------------------------------------------------------
# Within-period offsets and formatter
# ---------------------------------------------------------------------------


def mit_offset(mit_loc: MitLoc, frequency: Frequency) -> float:
    """Return the within-period offset in MIT-value units.

    Matches Julia's ``mit_offset(::Val{loc}, ::Type{<:Frequency})`` overloads:

    * ``left`` → ``0.0`` for every frequency.
    * ``middle`` → ``0.5`` for non-YP, ``0.5 / N`` for ``YPFrequency{N}``.
    * ``right`` → ``1.0`` for non-YP, ``1.0 / N`` for ``YPFrequency{N}``.

    The YP scaling keeps the offset in the same units as
    ``float(mit) = year + (period - 1) / N`` so adding the offset shifts the
    plotted point within the year by a fraction-of-year.
    """
    if mit_loc not in ("left", "middle", "right"):
        msg = f"mit_loc must be 'left', 'middle', or 'right', got {mit_loc!r}"
        raise ValueError(msg)
    if mit_loc == "left":
        return 0.0
    n = frequency.periods_per_year if isinstance(frequency, YPFrequency) else 1
    return (0.5 if mit_loc == "middle" else 1.0) / n


def mit_formatter(mit_loc: MitLoc, frequency: Frequency) -> Callable[..., str]:
    """Return a callable ``f(x: float) -> str`` for YP tick formatting.

    The callable rounds the float ``x`` back to the implied
    ``MIT{frequency}`` (year, period) pair and returns its ``str(mit)``
    formatting. When ``x`` is more than ``0.1`` periods away from any real
    MIT (i.e. matplotlib's tick locator chose a non-grid value), the result
    is suffixed with ``"+"`` and a one-shot ``UserWarning`` is emitted —
    matching Julia's ``"xticks marked with (+) are not aligned with"`` log.

    Calling code uses the result as the ``matplotlib.ticker.FuncFormatter``
    callback. The closure ignores its ``pos`` argument by accepting only
    ``x``; matplotlib passes ``(value, pos)`` positionally but
    ``FuncFormatter`` happily accepts a single-arg callable too.
    """
    if not isinstance(frequency, YPFrequency):
        msg = f"mit_formatter is YP-only; got {prettyprint_frequency(frequency)}"
        raise TypeError(msg)
    n = frequency.periods_per_year
    offset = mit_offset(mit_loc, frequency)
    state = {"warned": False}

    def _fmt(x: float, _pos: int | None = None) -> str:
        # Recover the (yr, per) pair the float implies, then check that
        # it is within 0.1 periods of a real MIT. If not, suffix with "+".
        yr = math.floor(x - offset)
        per = 1 + math.floor(n * (x - yr - offset))
        per = max(1, min(n, per))
        try:
            xmit = MIT.from_yp(frequency, yr, per)
        except (ValueError, TypeError):
            return f"{x:g}"
        if n * abs(x - float(xmit) - offset) > 0.1:
            if not state["warned"]:
                warnings.warn(
                    "xticks marked with (+) are not aligned with "
                    f"{prettyprint_frequency(frequency)} MITs.",
                    stacklevel=2,
                )
                state["warned"] = True
            return str(xmit) + "+"
        return str(xmit)

    return _fmt


def pick_yp_stride(rng_length: int, n: int, *, target_n: int = 8) -> int:
    """Return the period-stride that gives ~``target_n`` ticks for a YP range.

    Stride candidates are ``1`` period, ``1`` year (``n`` periods), then
    ``k`` years for ``k`` in 2/5/10/25/50/100. The first stride whose
    resulting tick count is ``<= target_n`` wins; fallback is one period
    per tick (stride 1).

    Shared by :func:`yp_tick_positions` (plotly) and the matplotlib
    backend's :class:`matplotlib.ticker.MultipleLocator` configuration so
    both backends pick identical tick positions.
    """
    if rng_length <= 0:
        return 1
    candidates = [1, n, *(k * n for k in (2, 5, 10, 25, 50, 100))]
    for c in candidates:
        if rng_length / c <= target_n:
            return c
    return candidates[-1]


def yp_tick_positions(
    rng: MITRange,
    mit_loc: MitLoc,
    *,
    target_n: int = 8,
) -> tuple[list[float], list[str]]:
    """Return ``(tickvals, ticktext)`` for a YP MITRange.

    Used by the plotly backend, which (unlike matplotlib's
    ``FuncFormatter``) cannot apply a Python callable to a numeric axis.
    The stride is chosen by :func:`pick_yp_stride` so the total tick
    count lands near ``target_n``.
    """
    if not isinstance(rng.frequency, YPFrequency):
        msg = "yp_tick_positions is YP-only"
        raise TypeError(msg)
    n = rng.frequency.periods_per_year
    offset = mit_offset(mit_loc, rng.frequency)
    if len(rng) == 0:
        return ([], [])
    stride = pick_yp_stride(len(rng), n, target_n=target_n)
    tickvals: list[float] = []
    ticktext: list[str] = []
    for i in range(0, len(rng), stride):
        m = MIT(rng.frequency, rng.start.value + i)
        tickvals.append(float(m) + offset)
        ticktext.append(str(m))
    return tickvals, ticktext


# ---------------------------------------------------------------------------
# Range intersection / classification
# ---------------------------------------------------------------------------


def intersect_ranges(*ranges: MITRange) -> MITRange:
    """Return the intersection of two or more MITRanges of the same frequency.

    Mirrors Julia's ``intersect(rng1, rng2)`` on ``UnitRange{MIT{F}}``.
    Mixing frequencies raises ``TypeError``. The result may be empty
    (``start > stop``).
    """
    if not ranges:
        msg = "intersect_ranges requires at least one range"
        raise ValueError(msg)
    freq = ranges[0].frequency
    for r in ranges[1:]:
        if r.frequency != freq:
            raise TypeError(
                "Cannot intersect ranges of different frequencies: "
                f"{prettyprint_frequency(freq)} and {prettyprint_frequency(r.frequency)}"
            )
    start_value = max(r.start.value for r in ranges)
    stop_value = min(r.stop.value for r in ranges)
    return MITRange(MIT(freq, start_value), MIT(freq, stop_value))


def xaxis_kind(frequency: Frequency) -> AxisKind:
    """Classify the frequency for x-axis rendering.

    * ``YPFrequency`` → ``"yp"`` (numeric float axis, custom formatter).
    * ``Daily`` / ``BDaily`` / ``Weekly`` → ``"date"`` (datetime axis).
    * Everything else (``Unit``) → ``"numeric"`` (integer axis).
    """
    if isinstance(frequency, YPFrequency):
        return "yp"
    if isinstance(frequency, (Daily, BDaily, Weekly)):
        return "date"
    return "numeric"


def classify_inputs(series: Sequence[TSeries | MVTSeries]) -> PlotKind:
    """Return the recipe arm matching the input sequence.

    Returns ``"tseries"`` if every input is a TSeries and ``"mvtseries"``
    if every input is an MVTSeries. Raises ``TypeError`` for a mixed call
    or an empty call. Matches Julia's recipe arity
    (``many_tseries(ts::TSeries...)`` vs
    ``many_mvtseries(datasets::MVTSeries...)``).
    """
    if not series:
        msg = "plot() requires at least one TSeries or MVTSeries argument"
        raise TypeError(msg)
    if all(isinstance(s, TSeries) for s in series):
        return "tseries"
    if all(isinstance(s, MVTSeries) for s in series):
        return "mvtseries"
    msg = (
        "plot() does not accept a mix of TSeries and MVTSeries — pass all of one kind. "
        f"Got: {[type(s).__name__ for s in series]}"
    )
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Per-series x/y builder
# ---------------------------------------------------------------------------


def build_xy(
    t: TSeries,
    *,
    trange: MITRange | None,
    mit_loc: MitLoc,
) -> tuple[np.ndarray, np.ndarray, AxisKind]:
    """Return ``(x, y, axis_kind)`` for a single TSeries.

    * ``trange`` is intersected with ``t.range``; out-of-window data is
      dropped.
    * For YP frequencies, ``x`` is ``float(mit) + mit_offset`` for each MIT
      in the trimmed range.
    * For Daily / BDaily / Weekly, ``x`` is a ``numpy.ndarray`` of
      :class:`datetime.date` (with ``ref="end"`` to match the Julia
      ``Date(MIT)`` constructor). ``mit_loc`` is ignored for these — the
      Julia recipe replaces ``x`` with the date vector unconditionally.
    * For Unit, ``x`` is the integer MIT values plus ``mit_offset``.
    """
    rng = t.range if trange is None else intersect_ranges(t.range, trange)
    y = np.asarray(t[rng].values) if len(rng) > 0 else np.array([], dtype=t.dtype)
    kind = xaxis_kind(t.frequency)
    if kind == "yp":
        offset = mit_offset(mit_loc, t.frequency)
        x = np.fromiter(
            (float(MIT(t.frequency, rng.start.value + i)) + offset for i in range(len(rng))),
            dtype=float,
            count=len(rng),
        )
    elif kind == "date":
        x = np.empty(len(rng), dtype=object)
        for i in range(len(rng)):
            x[i] = mit_to_date(MIT(t.frequency, rng.start.value + i))
    else:
        offset = mit_offset(mit_loc, t.frequency)
        x = np.fromiter(
            (rng.start.value + i + offset for i in range(len(rng))),
            dtype=float,
            count=len(rng),
        )
    return x, y, kind


# ---------------------------------------------------------------------------
# Label / vars / panel normalisation
# ---------------------------------------------------------------------------


def normalize_label(label: str | Sequence[str] | None, n: int) -> list[str | None]:
    """Normalise a ``label=`` kwarg into a length-``n`` list.

    * ``None`` → ``[None] * n`` (each backend supplies a default).
    * Single string → ``[label] * n`` (broadcasts).
    * Sequence of strings — must match ``n`` exactly.
    """
    if label is None:
        return [None] * n
    if isinstance(label, str):
        return [label] * n
    out: list[str | None] = list(label)
    if len(out) != n:
        msg = f"Number of labels ({len(out)}) does not match number of series ({n})"
        raise ValueError(msg)
    if any(not isinstance(s, str) for s in out):
        msg = f"Each label must be a string, got {out!r}"
        raise TypeError(msg)
    return out


_VarSpec = str | tuple[str, str]


def normalize_vars(
    vars_: Sequence[_VarSpec] | None,
    datasets: Sequence[MVTSeries],
) -> list[tuple[str, str]]:
    """Return a list of ``(column_name, subplot_title)`` pairs.

    * ``vars_=None`` — union of column names across ``datasets``
      (preserving first-seen order), titles defaulting to the column name.
    * ``vars_`` may contain bare strings (title = name) or ``(name, title)``
      pairs.
    * The 10-variable cap from the Julia ``many_mvtseries`` recipe is
      enforced here.
    """
    if vars_ is None:
        seen: dict[str, None] = {}
        for d in datasets:
            for name in d.column_names:
                seen.setdefault(name, None)
        items: list[tuple[str, str]] = [(name, name) for name in seen]
    else:
        items = []
        for spec in vars_:
            if isinstance(spec, str):
                items.append((spec, spec))
            else:
                name, title = spec
                items.append((str(name), str(title)))
    max_vars = 10
    if len(items) > max_vars:
        msg = (
            f"Too many variables ({len(items)}); the plotting backend caps panel layouts at "
            f"{max_vars}. Try splitting into pages or passing vars=[...] to select a subset."
        )
        raise ValueError(msg)
    return items


def panel_grid(nvars: int) -> tuple[int, int]:
    """Return ``(nrows, ncols)`` for a panel of ``nvars`` subplots.

    Mirrors Plots.jl's default ``layout --> nvars`` shapes:

    ============  =======
    nvars         shape
    ============  =======
    1             (1, 1)
    2             (1, 2)
    3             (1, 3)
    4             (2, 2)
    5-6           (2, 3)
    7-9           (3, 3)
    10            (5, 2)
    ============  =======
    """
    if nvars < 1:
        msg = f"nvars must be >= 1, got {nvars}"
        raise ValueError(msg)
    presets: dict[int, tuple[int, int]] = {
        1: (1, 1),
        2: (1, 2),
        3: (1, 3),
        4: (2, 2),
        5: (2, 3),
        6: (2, 3),
        7: (3, 3),
        8: (3, 3),
        9: (3, 3),
        10: (5, 2),
    }
    return presets[nvars]


# ---------------------------------------------------------------------------
# Frequency uniformity / trange validation
# ---------------------------------------------------------------------------


def check_uniform_frequency(series: Iterable[TSeries | MVTSeries]) -> Frequency:
    """Return the shared frequency of every input; raise ``TypeError`` on a mix.

    Mirrors Julia's recipe-side assertion that ``trange`` only makes sense
    when every series has the same frequency. Empty input raises ``ValueError``.
    """
    series = list(series)
    if not series:
        raise ValueError("check_uniform_frequency: empty series list")
    freq = series[0].frequency
    for s in series[1:]:
        if s.frequency != freq:
            raise TypeError(
                "All series must share a frequency. Got "
                f"{prettyprint_frequency(freq)} and {prettyprint_frequency(s.frequency)}."
            )
    return freq


def to_date_axis_range(rng: MITRange, *, frequency: Frequency) -> tuple[_dt.date, _dt.date] | None:
    """Return ``(begin_date, end_date)`` for a date axis range, or None for empty.

    Used by both backends to set explicit x-axis limits on calendar
    frequencies (Daily / BDaily / Weekly) when the user passed ``trange=``.
    """
    if not isinstance(frequency, CalendarFrequency) or isinstance(frequency, YPFrequency):
        return None
    if len(rng) == 0:
        return None
    return (mit_to_date(rng.first()), mit_to_date(rng.last()))


def ppy_safe(frequency: Frequency) -> int:
    """Return ``ppy(frequency)`` or ``1`` for ``Unit`` (which raises)."""
    try:
        return ppy(frequency)
    except ValueError:
        return 1
