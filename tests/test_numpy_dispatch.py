# SPDX-License-Identifier: MIT
"""Unsupported NumPy inputs terminate and foreign overrides keep control."""

from typing import Any

import numpy as np
import pytest

from tsecon import MVTSeries, TSeries, qq


@pytest.fixture(params=[False, True], ids=["TSeries", "MVTSeries"])
def series(request: pytest.FixtureRequest) -> TSeries | MVTSeries:
    return (
        MVTSeries(qq(2020, 1), a=[1.0, 2.0, 3.0], b=[4.0, 5.0, 6.0])
        if request.param
        else TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
    )


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("mixed", [False, True])
def test_nested_stack_rejects(series: Any, container: Any, mixed: bool) -> None:
    other = np.asarray(series) if mixed else series
    with pytest.raises(TypeError):
        np.stack(container([series, other]))
    np.testing.assert_array_equal(
        np.stack([np.asarray(series), np.asarray(other)]),
        np.stack([series.values, series.values]),
    )


class Foreign:
    def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> str:
        return "foreign result"

    def __array__(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Foreign object must not be coerced")


@pytest.mark.parametrize("function", [np.stack, np.concatenate, np.array_equal, np.allclose])
def test_foreign_override_gets_control(series: Any, function: Any) -> None:
    other = Foreign()
    result = (
        function([series, other])
        if function in (np.stack, np.concatenate)
        else function(series, other)
    )
    assert result == "foreign result"


def test_foreign_ndarray_subclass_gets_control(series: Any) -> None:
    class ForeignArray(np.ndarray):
        def __array_function__(self, func: Any, types: Any, args: Any, kwargs: Any) -> str:
            return "array override"

    other = np.asarray(series).view(ForeignArray)
    assert np.array_equal(series, other) == "array override"


@pytest.mark.parametrize("kind", ["list", "dict", "cycle", "iterator"])
def test_unsafe_keyword_containers_are_not_delegated(series: Any, kind: str) -> None:
    payload: Any = {"list": [series], "dict": {"nested": series}}.get(kind)
    if kind == "cycle":
        payload = []
        payload.append(payload)
    if kind == "iterator":

        def forbidden_iterator() -> Any:
            raise AssertionError("Iterator consumed")
            yield None

        payload = forbidden_iterator()

    def forbidden_function(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Unsafe fallback invoked")

    assert (
        series.__array_function__(
            forbidden_function, (type(series),), (series,), {"extra": payload}
        )
        is NotImplemented
    )


def test_direct_keyword_series_is_unwrapped(series: Any) -> None:
    np.testing.assert_array_equal(np.copy(a=series), np.asarray(series))
    np.testing.assert_array_equal(
        np.reshape(series, (len(series), -1)), np.reshape(np.asarray(series), (len(series), -1))
    )


def test_out_rejected_without_mutation(series: Any) -> None:
    out = np.full_like(np.asarray(series), -10.0)
    with pytest.raises(TypeError):
        np.cumsum(series, axis=0, out=out)
    np.testing.assert_array_equal(out, -10.0)


def test_registered_concatenate_still_accepts_series(series: Any) -> None:
    # Both containers' registered behavior is independently covered by the baseline.
    result = np.concatenate([series, series])
    assert type(result) is type(series)
    np.testing.assert_array_equal(result.values, np.concatenate([series.values] * 2))


def test_shared_plain_containers_are_allowed(series: Any) -> None:
    shared = [1, 2]

    def plain_function(a: Any, config: Any) -> Any:
        assert isinstance(a, np.ndarray)
        return config

    assert series.__array_function__(
        plain_function, (type(series),), (series, [shared, shared]), {}
    ) == [shared, shared]


def test_deep_nesting_declines_without_python_recursion(series: Any) -> None:
    nested = series
    for _ in range(2000):
        nested = [nested]
    assert series.__array_function__(np.stack, (type(series),), (nested,), {}) is NotImplemented
