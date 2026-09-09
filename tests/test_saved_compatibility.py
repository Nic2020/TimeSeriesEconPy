# SPDX-License-Identifier: MIT
"""Read trusted, committed fixtures produced before storage refactoring."""

import hashlib
import importlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from tsecon import MIT, Duration, MITRange, MVTSeries, Quarterly, TSeries, Workspace
from tsecon.io import loads

FIXTURES = Path(__file__).parent / "fixtures" / "compatibility"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
EXPORTS = json.loads((FIXTURES / "exports.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("module_name", EXPORTS)
def test_saved_public_exports(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert set(EXPORTS[module_name]) <= set(module.__all__)
    for name in EXPORTS[module_name]:
        assert hasattr(module, name), f"{module_name}.{name}"


@pytest.mark.parametrize("filename", MANIFEST["files"])
def test_saved_object(filename: str) -> None:
    data = (FIXTURES / filename).read_bytes()
    assert hashlib.sha256(data).hexdigest() == MANIFEST["files"][filename]["sha256"]
    # Only these locally generated, reviewed repository fixtures are unpickled.
    obj = pickle.loads(data) if filename.endswith(".pickle") else loads(data.decode("utf-8"))
    name = filename.split(".", maxsplit=1)[0]
    start = MIT.from_yp(Quarterly(end_month=2), 2020, 1)
    if name == "date":
        assert isinstance(obj, MIT)
        assert obj == start
    elif name == "duration":
        assert isinstance(obj, Duration)
        assert obj == start + 2 - start
    elif name == "range":
        assert isinstance(obj, MITRange)
        assert list(obj) == [start, start + 2, start + 4]
    elif name == "workspace":
        assert isinstance(obj, Workspace)
        assert list(obj.keys()) == ["note", "nested"]
        assert obj.note == "baseline"
        assert isinstance(obj.nested, Workspace)
        assert obj.nested.series.firstdate == start
        np.testing.assert_array_equal(obj.nested.series.values, [7.0, 8.0])
    elif name == "table":
        assert isinstance(obj, MVTSeries)
        assert obj.firstdate == start
        assert tuple(obj.column_names) == ("a", "b")
        np.testing.assert_array_equal(obj.values, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    else:
        expected, dtype = {
            "float_series": ([1.5, np.nan, -2.0], "float64"),
            "int_series": ([1, -2147483648, 3], "int32"),
            "bool_series": ([True, False, True], "bool"),
            "empty_series": ([], "float32"),
        }[name]
        assert isinstance(obj, TSeries)
        assert obj.firstdate == start
        assert obj.values.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(obj.values, expected)


@pytest.mark.parametrize("suffix", ["json", "pickle"])
def test_saved_table_columns_write_through(suffix: str) -> None:
    data = (FIXTURES / f"table.{suffix}").read_bytes()
    obj = pickle.loads(data) if suffix == "pickle" else loads(data.decode("utf-8"))
    start = MIT.from_yp(Quarterly(end_month=2), 2020, 1)
    obj.a[start] = 9.0
    assert obj.values[0, 0] == 9.0
    obj.values[1, 1] = 10.0
    assert obj.b[start + 1] == 10.0
