# SPDX-License-Identifier: MIT
"""Tests for ``tsecon.io.json`` — round-trip JSON serialization.

The Julia upstream's ``serialize.jl`` is a binary protocol bound to
``Serialization.jl``; we don't port that. Instead, this suite tests the
Python-side JSON schema (``_type``-tagged objects) for round-trip parity.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from tsecon import (
    MIT,
    Duration,
    MITRange,
    TSeries,
    Workspace,
    bdaily,
    daily,
    mm,
    qq,
    weekly,
    yy,
)
from tsecon.frequencies import (
    BDaily,
    Daily,
    HalfYearly,
    Monthly,
    Quarterly,
    Unit,
    Weekly,
    Yearly,
)
from tsecon.io import dump, dumps, load, loads
from tsecon.io.json import from_jsonable, to_jsonable

# ---------------------------------------------------------------------------
# MIT round-trip across frequencies
# ---------------------------------------------------------------------------


class TestMITRoundTrip:
    @pytest.mark.parametrize(
        "value",
        [
            qq(2020, 1),
            qq(1999, 4),
            mm(2020, 7),
            yy(2025),
            MIT(Quarterly(end_month=1), 8080),
            MIT(HalfYearly(), 4040),
            MIT(Monthly(), 24241),
            MIT(Unit(), 7),
            daily("2024-05-13"),
            bdaily("2024-05-13"),
            weekly("2024-05-13"),
            MIT(Weekly(end_day=3), 1234),
        ],
    )
    def test_roundtrip(self, value: MIT) -> None:
        text = dumps(value)
        back = loads(text)
        assert isinstance(back, MIT)
        assert back == value
        assert back.frequency == value.frequency

    def test_non_default_quarterly_endmonth(self) -> None:
        q = MIT(Quarterly(end_month=2), 8081)
        back = loads(dumps(q))
        assert isinstance(back, MIT)
        assert back == q
        assert isinstance(back.frequency, Quarterly)
        assert back.frequency.end_month == 2

    def test_singleton_identity_preserved(self) -> None:
        # Frequency singletons are cached: round-trip should return the SAME
        # frequency instance, not a fresh equal one.
        q = qq(2020, 1)
        back = loads(dumps(q))
        assert back.frequency is q.frequency


# ---------------------------------------------------------------------------
# Duration / MITRange
# ---------------------------------------------------------------------------


class TestDurationMITRange:
    def test_duration_roundtrip(self) -> None:
        d = qq(2020, 2) - qq(2020, 1)
        assert isinstance(d, Duration)
        back = loads(dumps(d))
        assert isinstance(back, Duration)
        assert back == d

    def test_mitrange_unit_step(self) -> None:
        r = MITRange(qq(2020, 1), qq(2022, 4))
        back = loads(dumps(r))
        assert isinstance(back, MITRange)
        assert back == r

    def test_mitrange_with_step(self) -> None:
        r = MITRange(qq(2020, 1), qq(2025, 4), step=2)
        back = loads(dumps(r))
        assert isinstance(back, MITRange)
        assert back.step == 2
        assert back == r

    def test_mitrange_empty(self) -> None:
        # start > stop means empty
        r = MITRange(qq(2020, 1), qq(2019, 4))
        assert r.is_empty()
        back = loads(dumps(r))
        assert back.is_empty()
        assert back.frequency == r.frequency


# ---------------------------------------------------------------------------
# TSeries
# ---------------------------------------------------------------------------


class TestTSeriesRoundTrip:
    def test_float_values(self) -> None:
        t = TSeries(qq(2020, 1), np.array([1.0, 2.0, 3.5, -4.25]))
        back = loads(dumps(t))
        assert isinstance(back, TSeries)
        assert back.equals(t)

    def test_int_values_preserves_dtype(self) -> None:
        t = TSeries(qq(2020, 1), np.array([1, 2, 3], dtype=np.int64))
        back = loads(dumps(t))
        assert isinstance(back, TSeries)
        assert back.values.dtype == np.int64
        assert back.equals(t)

    def test_bool_values(self) -> None:
        t = TSeries(qq(2020, 1), np.array([True, False, True], dtype=np.bool_))
        back = loads(dumps(t))
        assert back.values.dtype == np.bool_
        assert back.equals(t)

    def test_empty(self) -> None:
        t = TSeries(qq(2020, 1))
        back = loads(dumps(t))
        assert isinstance(back, TSeries)
        assert back.is_empty()
        assert back.firstdate == qq(2020, 1)

    def test_nan_preserved(self) -> None:
        t = TSeries(qq(2020, 1), np.array([1.0, np.nan, 3.0]))
        back = loads(dumps(t))
        assert isinstance(back, TSeries)
        # allclose with equal_nan=True
        assert back.allclose(t)
        assert np.isnan(back.values[1])

    def test_inf_preserved(self) -> None:
        t = TSeries(qq(2020, 1), np.array([1.0, np.inf, -np.inf]))
        back = loads(dumps(t))
        assert back.values[1] == np.inf
        assert back.values[2] == -np.inf


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


class TestWorkspaceRoundTrip:
    def test_simple_scalars(self) -> None:
        w = Workspace(a=1, b=2.5, c="hello", d=True)
        back = loads(dumps(w))
        assert isinstance(back, Workspace)
        assert list(back.keys()) == list(w.keys())
        for k in w:
            assert back[k] == w[k]

    def test_with_tsecon_objects(self) -> None:
        w = Workspace(
            scalar=42,
            mit=qq(2020, 1),
            rng=MITRange(qq(2020, 1), qq(2025, 4)),
            ts=TSeries(qq(2020, 1), np.arange(5.0)),
        )
        back = loads(dumps(w))
        assert isinstance(back, Workspace)
        assert back.scalar == 42
        assert back.mit == qq(2020, 1)
        assert back.rng == MITRange(qq(2020, 1), qq(2025, 4))
        assert back.ts.equals(w.ts)

    def test_nested_workspace(self) -> None:
        inner = Workspace(x=1, y=TSeries(mm(2020, 1), np.array([1.0, 2.0])))
        outer = Workspace(inner=inner, scalar=99)
        back = loads(dumps(outer))
        assert isinstance(back, Workspace)
        assert isinstance(back.inner, Workspace)
        assert back.scalar == 99
        assert back.inner.x == 1
        assert back.inner.y.equals(inner.y)

    def test_insertion_order_preserved(self) -> None:
        w = Workspace(z=1, a=2, m=3)
        back = loads(dumps(w))
        assert list(back.keys()) == ["z", "a", "m"]

    def test_empty(self) -> None:
        back = loads(dumps(Workspace()))
        assert isinstance(back, Workspace)
        assert back.is_empty()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


class TestStreamIO:
    def test_dump_load_stream(self) -> None:
        w = Workspace(a=1, ts=TSeries(qq(2020, 1), np.arange(3.0)))
        buf = io.StringIO()
        dump(w, buf)
        buf.seek(0)
        back = load(buf)
        assert isinstance(back, Workspace)
        assert back.ts.equals(w.ts)

    def test_indent_produces_pretty_json(self) -> None:
        w = Workspace(a=1)
        out = dumps(w, indent=2)
        assert "\n" in out
        # Ensure it still parses round-trip
        back = loads(out)
        assert isinstance(back, Workspace)
        assert back.a == 1


# ---------------------------------------------------------------------------
# Frequency stand-alone
# ---------------------------------------------------------------------------


class TestFrequencySerialization:
    @pytest.mark.parametrize(
        "freq",
        [
            Yearly(),
            Yearly(end_month=6),
            HalfYearly(),
            Quarterly(),
            Quarterly(end_month=2),
            Monthly(),
            Weekly(),
            Weekly(end_day=3),
            Daily(),
            BDaily(),
            Unit(),
        ],
    )
    def test_freq_roundtrip(self, freq: object) -> None:
        # Frequencies serialize via the standalone _type=Frequency wrapper.
        text = dumps(freq)
        back = loads(text)
        assert back is freq  # singleton identity


# ---------------------------------------------------------------------------
# Plain values pass through
# ---------------------------------------------------------------------------


class TestPlainValues:
    def test_int_str_float_bool_none(self) -> None:
        for v in (1, 2.5, "hello", True, None):
            assert loads(dumps(v)) == v if v is not None else loads(dumps(v)) is None

    def test_list(self) -> None:
        assert loads(dumps([1, 2, 3])) == [1, 2, 3]

    def test_nested_dict_without_type_tag(self) -> None:
        assert loads(dumps({"a": 1, "b": [1, 2]})) == {"a": 1, "b": [1, 2]}

    def test_numpy_scalar_unwrapped(self) -> None:
        out = to_jsonable(np.int64(5))
        assert out == 5
        assert isinstance(out, int)

    def test_unknown_type_raises(self) -> None:
        class Foo:
            pass

        with pytest.raises(TypeError, match="Cannot JSON-encode"):
            to_jsonable(Foo())

    def test_unknown_type_tag_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown _type tag"):
            from_jsonable({"_type": "NotARealType"})


# ---------------------------------------------------------------------------
# Wire format spot checks (so we notice if the schema accidentally drifts)
# ---------------------------------------------------------------------------


class TestWireFormat:
    def test_mit_shape(self) -> None:
        d = to_jsonable(qq(2020, 1))
        assert d["_type"] == "MIT"
        assert d["freq"] == {"name": "Quarterly", "end_month": 3}
        assert isinstance(d["value"], int)

    def test_tseries_shape(self) -> None:
        d = to_jsonable(TSeries(qq(2020, 1), np.array([1.0, 2.0])))
        assert d["_type"] == "TSeries"
        assert d["dtype"] == "float64"
        assert d["values"] == [1.0, 2.0]
        assert d["firstdate"]["_type"] == "MIT"

    def test_workspace_shape(self) -> None:
        d = to_jsonable(Workspace(a=1, b=2))
        assert d["_type"] == "Workspace"
        assert d["items"] == [["a", 1], ["b", 2]]

    def test_dumps_is_valid_json(self) -> None:
        # Round-trip through stdlib json (no special handling): we should
        # produce valid JSON output for all-finite inputs.
        text = dumps(Workspace(a=1, ts=TSeries(qq(2020, 1), np.array([1.0, 2.0]))))
        json.loads(text)  # raises on bad JSON
