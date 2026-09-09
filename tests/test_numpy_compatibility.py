# SPDX-License-Identifier: MIT
"""Working array operations must retain their values and result categories."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from tsecon import MVTSeries, TSeries, qq


@pytest.mark.parametrize("series_type", [TSeries, MVTSeries])
@pytest.mark.parametrize(
    ("operation", "kind"),
    [
        (np.mean, "scalar"),
        (np.sum, "scalar"),
        (np.std, "scalar"),
        (np.median, "scalar"),
        (lambda x: np.quantile(x, 0.5), "scalar"),
        (lambda x: np.diff(x, axis=0), "array"),
        (lambda x: np.cumsum(x, axis=0), "array"),
        (lambda x: np.reshape(x, (-1,)), "array"),
        (np.transpose, "array"),
        (np.copy, "array"),
        (lambda x: np.isclose(x, 2.0), "array"),
        (lambda x: np.concatenate([x, x]), "series"),
        (lambda x: np.array_equal(x, x), "bool"),
        (lambda x: np.allclose(x, x), "bool"),
        (np.add.reduce, "reduction"),
        (np.add.accumulate, "array"),
        (np.sqrt, "series"),
    ],
    ids=[
        "mean",
        "sum",
        "std",
        "median",
        "quantile",
        "diff",
        "cumsum",
        "reshape",
        "transpose",
        "copy",
        "isclose",
        "concatenate",
        "array_equal",
        "allclose",
        "add_reduce",
        "add_accumulate",
        "sqrt",
    ],
)
def test_numpy_compatibility(
    series_type: type[TSeries] | type[MVTSeries],
    operation: Callable[..., Any],
    kind: str,
) -> None:
    series = (
        TSeries(qq(2020, 1), [1.0, 2.0, 3.0])
        if series_type is TSeries
        else MVTSeries(qq(2020, 1), a=[1.0, 2.0, 3.0], b=[4.0, 5.0, 6.0])
    )
    result = operation(series)
    expected = operation(np.asarray(series))
    np.testing.assert_allclose(np.asarray(result), expected)
    assert np.asarray(result).dtype == np.asarray(expected).dtype
    if kind == "series":
        assert type(result) is series_type
        assert result.firstdate == series.firstdate
        assert len(result) == np.asarray(expected).shape[0]
        if isinstance(series, MVTSeries):
            assert result.column_names == series.column_names
    elif kind == "array" or (kind == "reduction" and series_type is MVTSeries):
        assert isinstance(result, np.ndarray)
    elif kind == "bool":
        assert isinstance(result, (bool, np.bool_))
    else:
        assert np.isscalar(result)
