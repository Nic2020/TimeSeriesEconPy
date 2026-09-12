"""Empty monthly Float64 interchange and safe handling of reconstruction metadata."""

import hashlib
import importlib.util
import shutil
import sqlite3
import tomllib
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from tsecon import TSeries, mm
from tsecon.dataecon import DataEconError, open_dataecon

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built; configure TSECON_DATAECON_ROOT",
)
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(("name", "anchor"), [("empty", mm(2024, 1)), ("empty_later", mm(2025, 7))])
def test_julia_empty_fixture_preserves_anchor(name, anchor):
    fixture = FIXTURES / "julia_empty_monthly.daec"
    provenance = tomllib.loads((FIXTURES / "julia_empty_monthly.toml").read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    with open_dataecon(fixture) as db:
        first = db.read_series(name)
        second = db.read_series(name)
    assert first.firstdate == second.firstdate == anchor
    assert first.lastdate == anchor - 1
    assert first.values.shape == (0,)
    assert first.values.dtype == np.float64
    assert first.values.flags.owndata
    assert second.values.flags.owndata
    assert first.values is not second.values
    first[anchor] = 7.0
    assert second.values.size == 0
    assert second.firstdate == anchor


@pytest.mark.parametrize("anchor", [mm(2024, 1), mm(2025, 7)])
def test_empty_write_uses_canonical_metadata_and_preserves_guards(tmp_path, anchor):
    path = tmp_path / "empty.daec"
    source = TSeries(anchor, np.empty(0, dtype=np.float64))
    with open_dataecon(path, "a") as db:
        db.write_series("empty", source)
        result = db.read_series("empty")
        with pytest.raises(DataEconError) as caught:
            db.write_series("empty", source)
        assert caught.value.code == -985
    assert result.firstdate == anchor
    assert result.lastdate == anchor - 1
    assert result.values.flags.owndata
    assert result.values.size == 0
    with pytest.raises(ValueError, match="closed"):
        db.write_series("closed", source)
    with open_dataecon(path) as db:
        with pytest.raises(ValueError, match="read-only"):
            db.write_series("readonly", source)
        assert db.read_series("empty").firstdate == anchor
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute(
            "SELECT o.class,o.type,t.eltype,t.elfreq,a.ax_type,a.length,a.frequency,a.data,"
            "t.value IS NULL FROM objects o JOIN tseries t USING(id) "
            "JOIN axes a ON t.axis_id=a.id WHERE o.name='empty'"
        ).fetchone() == (2, 12, 4, 0, 1, 0, 32, anchor.value, 1)
        assert conn.execute(
            "SELECT count(*) FROM attributes JOIN objects USING(id) WHERE objects.name='empty'"
        ).fetchone() == (0,)


@pytest.mark.parametrize(
    ("key", "marker"),
    [
        ("jeltype", "Float128"),
        ("jeltype", "Base.Float64"),
        ("jeltype", " Float64"),
        ("jeltype", "error(123)"),
        ("jeltype", ""),
        ("jeltype", None),
        ("jtype", "Float64"),
        ("jtype", None),
    ],
)
def test_empty_reconstruction_rejected_without_poisoning_file(tmp_path, key, marker):
    path = tmp_path / "marker.daec"
    shutil.copyfile(FIXTURES / "julia_empty_monthly.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO attributes SELECT id,?,? FROM objects WHERE name='empty'",
            (key, marker),
        )
    with open_dataecon(path) as db:
        with pytest.raises(TypeError, match="reconstruction"):
            db.read_series("empty")
        assert db.read_series("empty_later").firstdate == mm(2025, 7)


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE tseries SET value=zeroblob(8) WHERE id=(SELECT id FROM objects WHERE name='empty')",
        "UPDATE axes SET length=1 WHERE data=24288",
        "UPDATE axes SET data=2147483648 WHERE data=24288",
        "UPDATE axes SET data=-2147483649 WHERE data=24288",
    ],
)
def test_empty_malformed_storage_rejected_before_pointer_use(tmp_path, sql):
    path = tmp_path / "invalid.daec"
    shutil.copyfile(FIXTURES / "julia_empty_monthly.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(sql)
    with open_dataecon(path) as db:
        with pytest.raises(ValueError):
            db.read_series("empty")
        assert db.read_series("empty_later").firstdate == mm(2025, 7)


def test_empty_bad_date_rejected_before_storage(tmp_path):
    with open_dataecon(tmp_path / "date.daec", "a") as db:
        with pytest.raises(ValueError, match="round-trip"):
            db.write_series("bad", TSeries(mm(-170000000, 1), np.empty(0)))
        with pytest.raises(DataEconError) as caught:
            db.read_series("bad")
        assert caught.value.code == -989
        db.write_series("good", TSeries(mm(2024, 1), np.empty(0)))


def test_julia_empty_float32_is_not_silently_promoted():
    with open_dataecon(FIXTURES / "julia_empty_monthly.daec") as db:
        result = db.read_series("empty_float32")
    assert result.values.dtype == np.float32
    assert result.values.size == 0
    assert result.values.flags.owndata
