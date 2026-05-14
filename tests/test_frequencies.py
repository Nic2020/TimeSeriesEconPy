# SPDX-License-Identifier: MIT
"""Tests for the Frequency hierarchy and helpers."""

from __future__ import annotations

import pytest

from tsecon.frequencies import (
    BDaily,
    CalendarFrequency,
    Daily,
    Frequency,
    HalfYearly,
    Monthly,
    Quarterly,
    Unit,
    Weekly,
    Yearly,
    YPFrequency,
    endperiod,
    is_bdaily,
    is_daily,
    is_halfyearly,
    is_monthly,
    is_quarterly,
    is_weekly,
    is_yearly,
    ppy,
    prettyprint_frequency,
    sanitize_frequency,
)

# ---------------------------------------------------------------------------
# Singleton / cache semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        Unit,
        Monthly,
        Daily,
        BDaily,
        lambda: Yearly(),
        lambda: Yearly(end_month=6),
        lambda: HalfYearly(),
        lambda: HalfYearly(end_month=3),
        lambda: Quarterly(),
        lambda: Quarterly(end_month=2),
        lambda: Weekly(),
        lambda: Weekly(end_day=3),
    ],
)
def test_singleton_identity(factory: object) -> None:
    """Two calls with the same arguments return the same object (``is``)."""
    a = factory()  # type: ignore[operator]
    b = factory()  # type: ignore[operator]
    assert a is b


def test_distinct_params_distinct_singletons() -> None:
    assert Yearly() is not Yearly(end_month=6)
    assert Quarterly() is not Quarterly(end_month=2)
    assert Weekly() is not Weekly(end_day=3)


def test_equality_matches_identity() -> None:
    assert Yearly() == Yearly()
    assert Yearly(end_month=6) == Yearly(end_month=6)
    assert Yearly() != Yearly(end_month=6)


def test_hashable() -> None:
    d = {Yearly(): 1, Quarterly(): 2, Monthly(): 3, Daily(): 4}
    assert d[Yearly()] == 1
    assert d[Quarterly()] == 2
    assert d[Monthly()] == 3
    assert d[Daily()] == 4


# ---------------------------------------------------------------------------
# Hierarchy / isinstance
# ---------------------------------------------------------------------------


def test_subclass_relationships() -> None:
    assert issubclass(Yearly, YPFrequency)
    assert issubclass(YPFrequency, CalendarFrequency)
    assert issubclass(CalendarFrequency, Frequency)
    assert issubclass(Daily, CalendarFrequency)
    assert issubclass(Unit, Frequency)
    assert not issubclass(Unit, CalendarFrequency)


def test_isinstance_of_instances() -> None:
    assert isinstance(Yearly(), YPFrequency)
    assert isinstance(Quarterly(), CalendarFrequency)
    assert isinstance(Daily(), CalendarFrequency)
    assert isinstance(Unit(), Frequency)
    assert not isinstance(Unit(), CalendarFrequency)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, 13, -1])
def test_yearly_rejects_out_of_range(bad: int) -> None:
    with pytest.raises(ValueError):
        Yearly(end_month=bad)


@pytest.mark.parametrize("bad", [0, 4, -1, 7])
def test_quarterly_rejects_out_of_range(bad: int) -> None:
    with pytest.raises(ValueError):
        Quarterly(end_month=bad)


@pytest.mark.parametrize("bad", [0, 7, -1])
def test_halfyearly_rejects_out_of_range(bad: int) -> None:
    with pytest.raises(ValueError):
        HalfYearly(end_month=bad)


@pytest.mark.parametrize("bad", [0, 8, -1])
def test_weekly_rejects_out_of_range(bad: int) -> None:
    with pytest.raises(ValueError):
        Weekly(end_day=bad)


def test_rejects_non_int() -> None:
    with pytest.raises(TypeError):
        Yearly(end_month="hmm")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Quarterly(end_month=2.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Yearly(end_month=True)  # bool is rejected even though it's an int subclass


# ---------------------------------------------------------------------------
# ppy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("freq", "expected"),
    [
        (Yearly(), 1),
        (Yearly(end_month=3), 1),
        (HalfYearly(), 2),
        (Quarterly(), 4),
        (Quarterly(end_month=2), 4),
        (Monthly(), 12),
        (Daily(), 365),
        (BDaily(), 260),
        (Weekly(), 52),
        (Weekly(end_day=3), 52),
    ],
)
def test_ppy_instance(freq: Frequency, expected: int) -> None:
    assert ppy(freq) == expected


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        (Yearly, 1),
        (Quarterly, 4),
        (Monthly, 12),
        (Daily, 365),
        (BDaily, 260),
        (Weekly, 52),
    ],
)
def test_ppy_class(cls: type[Frequency], expected: int) -> None:
    assert ppy(cls) == expected


def test_ppy_unit_raises() -> None:
    with pytest.raises(ValueError):
        ppy(Unit())
    with pytest.raises(ValueError):
        ppy(Unit)


# ---------------------------------------------------------------------------
# endperiod
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("freq", "expected"),
    [
        (Yearly(), 12),
        (Yearly(end_month=2), 2),
        (Quarterly(), 3),
        (Quarterly(end_month=2), 2),
        (HalfYearly(), 6),
        (HalfYearly(end_month=4), 4),
        (Monthly(), 1),
        (Weekly(), 7),
        (Weekly(end_day=6), 6),
        (Daily(), 1),
        (BDaily(), 1),
        (Unit(), 1),
    ],
)
def test_endperiod(freq: Frequency, expected: int) -> None:
    assert endperiod(freq) == expected


# ---------------------------------------------------------------------------
# sanitize_frequency
# ---------------------------------------------------------------------------


def test_sanitize_frequency_class() -> None:
    assert sanitize_frequency(Monthly) == Monthly()
    assert sanitize_frequency(Yearly) == Yearly()
    assert sanitize_frequency(Quarterly) == Quarterly()
    assert sanitize_frequency(HalfYearly) == HalfYearly()
    assert sanitize_frequency(Weekly) == Weekly()


def test_sanitize_frequency_instance_passthrough() -> None:
    assert sanitize_frequency(Quarterly(end_month=2)) is Quarterly(end_month=2)


def test_sanitize_frequency_rejects_non_frequency() -> None:
    with pytest.raises(TypeError):
        sanitize_frequency("Yearly")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# prettyprint_frequency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("freq", "expected"),
    [
        (Yearly(), "Yearly"),
        (Yearly(end_month=2), "Yearly{2}"),
        (Quarterly(), "Quarterly"),
        (Quarterly(end_month=2), "Quarterly{2}"),
        (HalfYearly(), "HalfYearly"),
        (HalfYearly(end_month=4), "HalfYearly{4}"),
        (Weekly(), "Weekly"),
        (Weekly(end_day=3), "Weekly{3}"),
        (Monthly(), "Monthly"),
        (Daily(), "Daily"),
        (BDaily(), "BDaily"),
        (Unit(), "Unit"),
    ],
)
def test_prettyprint_frequency(freq: Frequency, expected: str) -> None:
    assert prettyprint_frequency(freq) == expected


# ---------------------------------------------------------------------------
# is_* predicates
# ---------------------------------------------------------------------------


def test_is_yearly() -> None:
    assert is_yearly(Yearly())
    assert is_yearly(Yearly(end_month=2))
    assert is_yearly(Yearly)
    assert not is_yearly(Quarterly())
    assert not is_yearly(YPFrequency)


def test_is_quarterly() -> None:
    assert is_quarterly(Quarterly())
    assert is_quarterly(Quarterly(end_month=2))
    assert is_quarterly(Quarterly)
    assert not is_quarterly(Yearly())


def test_is_halfyearly() -> None:
    assert is_halfyearly(HalfYearly())
    assert is_halfyearly(HalfYearly(end_month=2))
    assert not is_halfyearly(Yearly())


def test_is_monthly() -> None:
    assert is_monthly(Monthly())
    assert is_monthly(Monthly)
    assert not is_monthly(Yearly())


def test_is_weekly() -> None:
    assert is_weekly(Weekly())
    assert is_weekly(Weekly(end_day=3))
    assert not is_weekly(Yearly())


def test_is_daily() -> None:
    assert is_daily(Daily())
    assert is_daily(Daily)
    assert not is_daily(BDaily())


def test_is_bdaily() -> None:
    assert is_bdaily(BDaily())
    assert not is_bdaily(Daily())
