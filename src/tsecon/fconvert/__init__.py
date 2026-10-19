# SPDX-License-Identifier: MIT
"""Frequency-conversion subsystem.

Mirrors ``TimeSeriesEcon.jl/src/fconvert/``. The single user-facing entry is
[`fconvert`][tsecon.fconvert.fconvert], which dispatches on the second positional argument:

* ``fconvert(target, mit, *, ref=..., round_to=...)`` — convert an
  [`MIT`][tsecon.mit.MIT]. ``round_to`` is only meaningful for BDaily targets.
* ``fconvert(target, mitrange, *, trim=...)`` — convert an
  [`MITRange`][tsecon.mitrange.MITRange].
* ``fconvert(target, tseries, *, method=..., ref=...)`` — convert a
  [`TSeries`][tsecon.tseries.TSeries] using a built-in aggregator / spreader.
* ``fconvert(f, target, tseries, *, ref=..., **kwargs)`` — convert a TSeries
  using the custom callable ``f``.

Lower-level entries (also exported) for callers that already know the input
type:

* [`fconvert_mit`][tsecon.fconvert.fconvert_mit] /
  [`fconvert_range`][tsecon.fconvert.fconvert_range] /
  [`fconvert_tseries`][tsecon.fconvert.fconvert_tseries] — the per-input-type entry points (no
  Julia-style multiple-dispatch overhead).
* [`fconvert_parts`][tsecon.fconvert.fconvert_parts] — internal triple used by the YP→YP TSeries
  conversion path (``(period, source_month, target_month)``); exposed for
  advanced use.

Helpers and series-shape operations:

* [`extend_series`][tsecon.fconvert.extend_series] — pad to align with target-period boundaries.
* [`trim_series`][tsecon.fconvert.trim_series] — restrict to the inner aligned range.
* [`strip_tseries`][tsecon.fconvert.strip_tseries] /
  [`strip_tseries_inplace`][tsecon.fconvert.strip_tseries_inplace] — drop leading and trailing
  typenan entries.
* [`repeat_uneven`][tsecon.fconvert.repeat_uneven] /
  [`divide_uneven`][tsecon.fconvert.divide_uneven] /
  [`linear_uneven`][tsecon.fconvert.linear_uneven] — vector helpers used by the higher-frequency
  conversion methods (and available as the canonical custom-function aliases that mirror
  ``method="const" / "even" / "linear"`` exactly).

Currently still ⬜: the BDaily kwarg variants (``skip_holidays`` /
``skip_all_nans`` / ``holidays_map``) block on the ``options.jl`` port; and
BDaily-source [`extend_series`][tsecon.fconvert.extend_series] needs the ``cleanedvalues`` plumbing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, overload

from tsecon.fconvert._helpers import (
    divide_uneven,
    linear_uneven,
    repeat_uneven,
    strip_tseries,
    strip_tseries_inplace,
)
from tsecon.fconvert._mit import (
    fconvert_mit,
    fconvert_parts,
    fconvert_range,
)
from tsecon.fconvert._tseries import (
    extend_series,
    fconvert_is_cython,
    fconvert_tseries,
    trim_series,
)
from tsecon.frequencies import Frequency, FrequencyLike
from tsecon.mit import MIT
from tsecon.mitrange import MITRange
from tsecon.tseries import TSeries

__all__ = [
    "divide_uneven",
    "extend_series",
    "fconvert",
    "fconvert_is_cython",
    "fconvert_mit",
    "fconvert_parts",
    "fconvert_range",
    "fconvert_tseries",
    "linear_uneven",
    "repeat_uneven",
    "strip_tseries",
    "strip_tseries_inplace",
    "trim_series",
]


@overload
def fconvert(
    target: FrequencyLike,
    what: MIT,
    *,
    ref: str = "end",
    round_to: str = "current",
) -> MIT: ...
@overload
def fconvert(target: FrequencyLike, what: MITRange, *, trim: str = "both") -> MITRange: ...
@overload
def fconvert(
    target: FrequencyLike,
    what: TSeries,
    *,
    method: str | None = None,
    ref: str | None = None,
) -> TSeries: ...
@overload
def fconvert(
    f: Callable[..., Any],
    target: FrequencyLike,
    what: TSeries,
    *,
    ref: str | None = None,
    **kwargs: Any,
) -> TSeries: ...


def fconvert(*args: Any, **kwargs: Any) -> Any:
    """Convert an `MIT`, `MITRange`, or `TSeries` to a target frequency.

    See the module-level docstring for the four call shapes. Dispatches on
    the second positional argument (or, when a callable is the first
    positional argument, the third).
    """
    if len(args) == 0:
        msg = "fconvert: missing positional arguments."
        raise TypeError(msg)
    first = args[0]
    # fconvert(f, target, t, ...)
    if callable(first) and not isinstance(first, (type, Frequency)):
        if len(args) != 3:
            msg = (
                "fconvert: when the first positional argument is a callable, exactly three "
                "positional arguments are required: (f, target, t)."
            )
            raise TypeError(msg)
        return fconvert_tseries(args[0], args[1], args[2], **kwargs)
    # fconvert(target, what, ...)
    if len(args) != 2:
        msg = (
            "fconvert: expected two positional arguments (target, what) or three "
            f"with a callable first arg. Got {len(args)}."
        )
        raise TypeError(msg)
    what = args[1]
    if isinstance(what, MIT):
        return fconvert_mit(args[0], what, **kwargs)
    if isinstance(what, MITRange):
        return fconvert_range(args[0], what, **kwargs)
    if isinstance(what, TSeries):
        return fconvert_tseries(args[0], what, **kwargs)
    msg = (
        f"fconvert: second positional argument must be MIT, MITRange, or TSeries; "
        f"got {type(what).__name__}."
    )
    raise TypeError(msg)
