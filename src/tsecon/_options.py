# SPDX-License-Identifier: MIT
"""Global package options.

Mirrors ``TimeSeriesEcon.jl/src/options.jl``. A small dictionary of
process-global settings consulted by other modules at call time. Public API:

* :func:`getoption` / :func:`setoption` — generic accessors.
* :func:`set_holidays_map` — accept a pre-built BDaily Boolean :class:`TSeries`
  (``True`` = business day, ``False`` = holiday) and store it under
  ``"bdaily_holidays_map"``. The Julia upstream additionally exposes
  ``set_holidays_map(country, subdivision=None)`` which reads a TOML + packed
  binary file shipped in ``data/holidays.toml`` / ``data/holidays.bin``. We
  defer that loader to M4 (the holidays subsystem, which will use the
  ``holidays`` PyPI package directly rather than the bundled binary — see
  ``parity/PARITY.md``). Users who already have a custom calendar can pass it
  in here.
* :func:`clear_holidays_map` — set ``bdaily_holidays_map`` back to ``None``.

Recognised option names and their types:

==========================  ===========================  =================================
Name                        Type                         Default
==========================  ===========================  =================================
``bdaily_holidays_map``     ``TSeries[BDaily, bool]``    ``None``
                            or ``None``
``bdaily_creation_bias``    ``"strict" / "previous" /``  ``"strict"``
                            ``"next" / "nearest"``
``x13path``                 ``str``                      ``""``
==========================  ===========================  =================================

Setting an unknown option raises :class:`KeyError`; passing an invalid value
for a known option raises :class:`ValueError` (for the enum-like bias) or
:class:`TypeError` (for the wrong Python type).

The options dictionary is process-global. Tests that mutate it must restore
the previous value in a teardown — there is a context-manager helper
:func:`option_scope` for that pattern.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final, Literal

__all__ = [
    "VALID_BDAILY_BIASES",
    "BDailyBias",
    "OptionName",
    "clear_holidays_map",
    "get_holidays_map",
    "getoption",
    "option_scope",
    "set_holidays_map",
    "setoption",
]

OptionName = Literal["bdaily_holidays_map", "bdaily_creation_bias", "x13path"]
BDailyBias = Literal["strict", "previous", "next", "nearest"]

VALID_BDAILY_BIASES: Final[frozenset[str]] = frozenset({"strict", "previous", "next", "nearest"})

_DEFAULTS: Final[dict[str, Any]] = {
    "bdaily_holidays_map": None,
    "bdaily_creation_bias": "strict",
    "x13path": "",
}

_options: dict[str, Any] = dict(_DEFAULTS)


def getoption(name: OptionName | str) -> Any:
    """Return the current value of a recognised option.

    Raises :class:`KeyError` for unknown names.
    """
    if name not in _DEFAULTS:
        msg = f"unknown option: {name!r}. Recognised names: {sorted(_DEFAULTS)}."
        raise KeyError(msg)
    return _options[name]


def setoption(name: OptionName | str, value: Any) -> None:
    """Set the value of a recognised option, validating its type.

    * ``bdaily_creation_bias`` — must be one of ``"strict"``, ``"previous"``,
      ``"next"``, ``"nearest"``.
    * ``bdaily_holidays_map`` — must be ``None`` or a Boolean BDaily
      :class:`~tsecon.tseries.TSeries`.
    * ``x13path`` — must be a string.
    """
    if name == "bdaily_creation_bias":
        if value not in VALID_BDAILY_BIASES:
            msg = (
                f"bdaily_creation_bias must be one of {sorted(VALID_BDAILY_BIASES)}; "
                f"received {value!r}."
            )
            raise ValueError(msg)
        _options[name] = value
        return
    if name == "bdaily_holidays_map":
        if value is not None:
            _validate_holidays_map(value)
        _options[name] = value
        return
    if name == "x13path":
        if not isinstance(value, str):
            msg = f"x13path must be a string; received {type(value).__name__}."
            raise TypeError(msg)
        _options[name] = value
        return
    msg = f"unknown option: {name!r}. Recognised names: {sorted(_DEFAULTS)}."
    raise KeyError(msg)


def set_holidays_map(holidays_map: Any) -> None:
    """Install a BDaily Boolean holidays map under ``bdaily_holidays_map``.

    Convenience wrapper around ``setoption("bdaily_holidays_map", t)``. The
    map is a BDaily :class:`~tsecon.tseries.TSeries` of booleans where
    ``True`` marks a business day and ``False`` marks a holiday.

    The Julia upstream also exposes ``set_holidays_map(country, subdivision)``
    which loads a TOML + packed binary calendar; that loader is deferred to
    the M4 holidays subsystem (which will use the ``holidays`` PyPI package
    rather than a bundled binary). For M1, callers supply a calendar
    directly.
    """
    setoption("bdaily_holidays_map", holidays_map)


def clear_holidays_map() -> None:
    """Reset ``bdaily_holidays_map`` to ``None``."""
    setoption("bdaily_holidays_map", None)


def get_holidays_map() -> Any:
    """Return the currently installed holidays map (or ``None``)."""
    return _options["bdaily_holidays_map"]


@contextmanager
def option_scope(**overrides: Any) -> Iterator[None]:
    """Context manager that temporarily overrides one or more options.

    On exit, every overridden option is restored to its prior value.
    Convenient for tests; the wrapper goes through :func:`setoption` so the
    same validation applies.
    """
    saved: dict[str, Any] = {name: _options[name] for name in overrides}
    try:
        for name, value in overrides.items():
            setoption(name, value)
        yield
    finally:
        for name, value in saved.items():
            _options[name] = value


def _validate_holidays_map(value: Any) -> None:
    """Reject non-Boolean / non-BDaily holidays maps.

    Imported lazily to avoid a circular dependency: :mod:`tsecon.mit` imports
    :mod:`tsecon._options` at the top to back ``bdaily_creation_bias``, so a
    top-level ``from tsecon.tseries import TSeries`` here would loop back
    through ``tsecon.tseries`` → ``tsecon.mit`` → ``tsecon._options``.
    """
    from tsecon.frequencies import BDaily  # noqa: PLC0415 — circular-import-break
    from tsecon.tseries import TSeries  # noqa: PLC0415 — circular-import-break

    if not isinstance(value, TSeries):
        msg = (
            "bdaily_holidays_map must be a BDaily Boolean TSeries or None; "
            f"received {type(value).__name__}."
        )
        raise TypeError(msg)
    if not isinstance(value.frequency, BDaily):
        msg = (
            "bdaily_holidays_map must be a BDaily TSeries; got frequency "
            f"{type(value.frequency).__name__}."
        )
        raise TypeError(msg)
    if value.values.dtype != bool:
        msg = f"bdaily_holidays_map must be a Boolean TSeries; got dtype {value.values.dtype}."
        raise TypeError(msg)
