# SPDX-License-Identifier: MIT
"""Global package options.

Mirrors ``TimeSeriesEcon.jl/src/options.jl``. A small dictionary of
process-global settings consulted by other modules at call time. Public API:

* :func:`getoption` / :func:`setoption` — generic accessors.
* :func:`set_holidays_map` — install a BDaily Boolean :class:`TSeries`
  (``True`` = business day, ``False`` = holiday) under
  ``"bdaily_holidays_map"``. Accepts either a pre-built TSeries
  (``set_holidays_map(t)``) or a country / subdivision code
  (``set_holidays_map("CA", "ON")``); the country form requires the
  optional ``holidays`` extra (``pip install 'TimeSeriesEconPy[holidays]'``)
  and delegates to the upstream ``python-holidays`` package for the
  calendar data — single source of truth, no vendored CSVs.
* :func:`get_holidays_options` — list supported country codes (no arg) or
  subdivisions of a country (with a country code arg). Mirrors Julia's
  ``get_holidays_options``.
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
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Final, Literal

__all__ = [
    "VALID_BDAILY_BIASES",
    "BDailyBias",
    "OptionName",
    "clear_holidays_map",
    "get_holidays_map",
    "get_holidays_options",
    "getoption",
    "option_scope",
    "set_holidays_map",
    "setoption",
]

_HOLIDAYS_INSTALL_HINT = "Install with: pip install 'TimeSeriesEconPy[holidays]'"
# The country/subdivision loader builds a Boolean BDaily map spanning this
# fixed range, matching Julia's ``bdaily("1970-01-01"):bdaily("2049-12-31")``
# default in ``TimeSeriesEcon.jl/src/options.jl``. Both endpoints are
# weekdays, so the ``bdaily()`` call succeeds regardless of the current
# ``bdaily_creation_bias``.
_COVERED_FIRST_YEAR: Final[int] = 1970
_COVERED_LAST_YEAR: Final[int] = 2049

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


def set_holidays_map(arg: Any, subdivision: str | None = None) -> None:
    """Install a BDaily Boolean holidays map under ``bdaily_holidays_map``.

    Two call forms:

    * ``set_holidays_map(t)`` — install a pre-built BDaily Boolean
      :class:`~tsecon.tseries.TSeries` (``True`` = business day,
      ``False`` = holiday).
    * ``set_holidays_map("CA", "ON")`` — fetch the calendar for the given
      country / subdivision from the ``holidays`` PyPI package, build a
      BDaily Boolean TSeries spanning ``bdaily("1970-01-01")`` to
      ``bdaily("2049-12-31")`` (matches the Julia upstream's default
      range), and install it.

    Parameters
    ----------
    arg
        Either a pre-built BDaily Boolean :class:`~tsecon.tseries.TSeries`
        or a country code string accepted by
        :func:`holidays.country_holidays` (ISO 3166 alpha-2 like ``"CA"``,
        ``"US"``, ``"DK"``; alpha-3 codes like ``"CAN"`` are also accepted
        — whatever the upstream package supports).
    subdivision
        Optional subdivision code (ISO 3166-2 short form, e.g. ``"ON"``,
        ``"QC"``, ``"CA"``). Only meaningful when ``arg`` is a country
        code; passing it with a TSeries raises :class:`TypeError`.

    Raises
    ------
    ImportError
        ``arg`` is a string and the ``holidays`` package is not installed.
        The error message includes the ``pip install`` hint.
    ValueError
        ``arg`` is an unsupported country code, or ``subdivision`` is not
        a recognised subdivision of ``arg``. The Julia upstream raises
        :class:`ArgumentError` here.
    TypeError
        ``subdivision`` is passed alongside a TSeries, or ``arg`` is
        neither a string nor a TSeries.

    Notes
    -----
    The country / subdivision codes follow the ``python-holidays``
    package's conventions (ISO 3166), which differ in minor cases from
    the Julia upstream's CSV-derived names (e.g. the Julia CSVs spell
    a few subdivisions with spaces that the package replaces with
    underscores or drops). The package is the single source of truth;
    discover the supported codes with :func:`get_holidays_options`.

    Examples
    --------
    >>> set_holidays_map("DK")  # Denmark, federal holidays  # doctest: +SKIP
    >>> set_holidays_map("CA", "ON")  # Ontario, Canada  # doctest: +SKIP
    >>> from tsecon import bdaily, TSeries, MITRange  # doctest: +SKIP
    >>> cal = TSeries.trues(MITRange(bdaily("2022-01-03"), bdaily("2022-12-30")))
    >>> set_holidays_map(cal)  # pre-built TSeries  # doctest: +SKIP
    """
    if isinstance(arg, str):
        _set_holidays_map_from_country(arg, subdivision)
        return
    if subdivision is not None:
        msg = (
            "set_holidays_map: `subdivision=` is only meaningful when the first "
            f"argument is a country-code string; received {type(arg).__name__}."
        )
        raise TypeError(msg)
    setoption("bdaily_holidays_map", arg)


def get_holidays_options(country: str | None = None) -> tuple[str, ...]:
    """List supported country codes (or subdivisions of a given country).

    Mirrors Julia's ``get_holidays_options(country=nothing)``. Both no-arg
    and country-arg forms return a sorted tuple. Subdivisions are reported
    using the country's native code (ISO 3166-2 short form), exactly as
    the ``holidays`` package exposes them.

    Parameters
    ----------
    country
        Optional country code accepted by
        :func:`holidays.country_holidays`. When ``None``, return all
        supported country codes.

    Returns
    -------
    tuple of str
        Sorted country codes (no arg) or sorted subdivision codes (one
        arg). A country with no subdivisions returns an empty tuple.

    Raises
    ------
    ImportError
        The ``holidays`` package is not installed.
    ValueError
        ``country`` is not a supported country code. The Julia upstream
        raises :class:`ArgumentError`.
    """
    holidays_mod = _require_holidays()
    countries = holidays_mod.list_supported_countries()
    if country is None:
        return tuple(sorted(countries))
    if country not in countries:
        msg = (
            f"Unsupported country: {country!r}. "
            "Call get_holidays_options() (no argument) to list supported codes."
        )
        raise ValueError(msg)
    return tuple(sorted(countries[country]))


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


def _require_holidays() -> Any:
    """Lazy-import the ``holidays`` package with a helpful error if missing.

    Mirrors the :mod:`tsecon.interop.pandas` precedent so the install hint
    stays consistent across the codebase.
    """
    if find_spec("holidays") is None:
        msg = (
            "tsecon.set_holidays_map(country, ...) and tsecon.get_holidays_options "
            f"require the `holidays` package. {_HOLIDAYS_INSTALL_HINT}"
        )
        raise ImportError(msg)
    return import_module("holidays")


def _set_holidays_map_from_country(country: str, subdivision: str | None) -> None:
    """Build a BDaily Boolean holidays TSeries from the ``holidays`` package.

    The map spans ``bdaily("1970-01-01")`` to ``bdaily("2049-12-31")`` to
    match the Julia upstream's default coverage. Holidays that land on
    weekends are skipped — the BDaily index has no slot for them, so they
    can't be marked as ``False``; the resulting map flags only the
    business-day positions where the calendar contains an entry.
    """
    holidays_mod = _require_holidays()
    # Map the upstream's NotImplementedError (unknown country/subdivision)
    # to ValueError so callers can rely on the same error class as Julia's
    # ArgumentError, and the message names the offending input.
    try:
        country_obj = holidays_mod.country_holidays(
            country,
            subdiv=subdivision,
            years=range(_COVERED_FIRST_YEAR, _COVERED_LAST_YEAR + 1),
        )
    except NotImplementedError as exc:
        if subdivision is not None:
            msg = (
                f"Unsupported subdivision {subdivision!r} for country {country!r}. "
                f"Call get_holidays_options({country!r}) to list supported subdivisions."
            )
        else:
            msg = (
                f"Unsupported country: {country!r}. "
                "Call get_holidays_options() to list supported codes."
            )
        raise ValueError(msg) from exc

    # Local imports break the foundational-module circular dependency:
    # tsecon.mit imports tsecon._options for bdaily_creation_bias.
    import numpy as np  # noqa: PLC0415

    from tsecon.mit import bdaily  # noqa: PLC0415
    from tsecon.mitrange import MITRange  # noqa: PLC0415
    from tsecon.tseries import TSeries  # noqa: PLC0415

    covered_start = bdaily(f"{_COVERED_FIRST_YEAR}-01-01")
    covered_end = bdaily(f"{_COVERED_LAST_YEAR}-12-31")
    rng = MITRange(covered_start, covered_end)
    mask = np.ones(len(rng), dtype=bool)
    # Iterate the holiday dates (sparse) rather than every business day
    # in the 80-year range (~20k entries): O(H) ~ a few hundred entries
    # per country across the coverage window.
    for hdate in country_obj:
        if hdate.weekday() >= 5:  # Saturday / Sunday — not in the BDaily index.
            continue
        h_mit = bdaily(hdate)
        if h_mit < covered_start or h_mit > covered_end:
            continue
        mask[h_mit.value - covered_start.value] = False
    ts = TSeries(covered_start, mask)
    setoption("bdaily_holidays_map", ts)


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
