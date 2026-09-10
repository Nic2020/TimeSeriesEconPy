"""Native integration tests; an absent optional build is the only skip condition."""

import gc
import hashlib
import importlib.util
import shutil
import sqlite3
import subprocess
import sys
import tomllib
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from tsecon import MVTSeries, TSeries, mm
from tsecon.dataecon import DataEconError, open_dataecon

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built; configure TSECON_DATAECON_ROOT",
)
FIXTURES = Path(__file__).parent / "fixtures"
VALUES = np.array([1.25, -2.5, 0.0, 4.75], dtype=np.float64)


def test_julia_fixture_values_and_compiler_abi():
    provenance = tomllib.loads((FIXTURES / "julia_monthly.toml").read_text())
    fixture = FIXTURES / "julia_monthly.daec"
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    with open_dataecon(fixture) as db:
        result = db.read_series("sample")
    assert result.firstdate == mm(2024, 1)
    assert result.lastdate == mm(2024, 4)
    assert result.values.dtype == np.float64
    assert result.values.flags.owndata
    np.testing.assert_array_equal(result.values, VALUES)
    _native = importlib.import_module("tsecon.dataecon._native")

    assert _native.version() == ("0.4.0", "0.4.0")
    actual = _native.abi_layout()
    assert actual["enums"] == tuple(provenance["layout"]["enums"])
    for name in ("object_t", "axis_t", "tseries_t"):
        expected = provenance["layout"][name]
        assert actual[name] == (expected["size"], tuple(expected["offsets"]))


def test_native_roundtrip_snapshot_and_metadata(tmp_path):
    path = tmp_path / "monthly.daec"
    source = TSeries(mm(2024, 1), VALUES.copy())
    with open_dataecon(path, "a") as db:
        db.write_series("sample", source)
        source.values[:] = 99
        first = db.read_series("sample")
        second = db.read_series("sample")
        first.values[0] = 100
        np.testing.assert_array_equal(second.values, VALUES)
    np.testing.assert_array_equal(second.values, VALUES)
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute(
            "SELECT o.class,o.type,t.eltype,t.elfreq,a.ax_type,a.length,a.frequency,a.data,"
            "length(t.value) FROM objects o JOIN tseries t ON o.id=t.id "
            "JOIN axes a ON t.axis_id=a.id WHERE o.name='sample'"
        ).fetchone() == (2, 12, 4, 0, 1, 4, 32, 24288, 32)
    path.rename(tmp_path / "closed.daec")


def test_nan_and_strided_input(tmp_path):
    values = np.array([np.nan, 1.0, -2.5, 2.0, 0.0, 3.0, 4.75, 4.0])
    expected = values[::2].copy()
    with open_dataecon(tmp_path / "nan.daec", "a") as db:
        db.write_series("sample", TSeries(mm(2024, 1), values[::2]))
        result = db.read_series("sample")
    np.testing.assert_array_equal(result.values, expected)


def test_unicode_object_name(tmp_path):
    path = tmp_path / "unicode-name.daec"
    with open_dataecon(path, "a") as db:
        db.write_series("série_日本", TSeries(mm(2024, 1), VALUES.copy()))
    with open_dataecon(path) as db:
        np.testing.assert_array_equal(db.read_series("série_日本").values, VALUES)


def test_missing_duplicate_and_readonly(tmp_path):
    path = tmp_path / "missing.daec"
    with pytest.raises(DataEconError) as caught:
        open_dataecon(path)
    assert caught.value.code > 0
    assert not path.exists()
    with open_dataecon(path, "a") as db:
        with pytest.raises(DataEconError) as caught:
            db.read_series("unknown")
        assert caught.value.code == -989
        assert caught.value.name == "unknown"
        db.write_series("sample", TSeries(mm(2024, 1), VALUES.copy()))
        with pytest.raises(DataEconError) as caught:
            db.write_series("sample", TSeries(mm(2024, 1), np.ones(4)))
        assert caught.value.code == -985
        np.testing.assert_array_equal(db.read_series("sample").values, VALUES)
    with open_dataecon(path) as db, pytest.raises(ValueError, match="read-only"):
        db.write_series("other", TSeries(mm(2024, 1), VALUES.copy()))


@pytest.mark.parametrize(
    ("sql", "exception"),
    [
        ("UPDATE tseries SET value=x'00'", ValueError),
        ("UPDATE axes SET length=1000000000", ValueError),
        ("UPDATE axes SET frequency=64", TypeError),
        ("UPDATE tseries SET eltype=1", TypeError),
        ("UPDATE axes SET data=-2000000000", ValueError),
        ("INSERT INTO attributes SELECT id,'jtype','error(123)' FROM tseries", TypeError),
        ("INSERT INTO attributes SELECT id,'jeltype','Float64' FROM tseries", TypeError),
    ],
)
def test_malformed_or_unsupported_storage(tmp_path, sql, exception):
    path = tmp_path / "bad.daec"
    shutil.copyfile(FIXTURES / "julia_monthly.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(sql)
    with open_dataecon(path) as db, pytest.raises(exception):
        db.read_series("sample")


def test_native_copy_failure_and_version_gate(tmp_path, monkeypatch):
    # Load the backend through the normal loader, then inject a Python failure
    # before native-buffer copying. No invalid native pointers are constructed.
    with open_dataecon(FIXTURES / "julia_monthly.daec"):
        pass
    _native = importlib.import_module("tsecon.dataecon._native")

    def fail(_):
        raise MemoryError("injected pre-copy failure")

    with monkeypatch.context() as patch:
        patch.setattr(_native, "validate_metadata", fail)
        with (
            pytest.raises(MemoryError, match="pre-copy"),
            open_dataecon(FIXTURES / "julia_monthly.daec") as db,
        ):
            db.read_series("sample")
        assert db.closed
    monkeypatch.setattr(_native, "version", lambda: ("0.4.0", "0.5.0"))
    path = tmp_path / "version.daec"
    with pytest.raises(ImportError, match=r"matching 0\.4\.0"):
        open_dataecon(path, "a")
    assert not path.exists()


def test_multiple_file_error_state(tmp_path):
    def worker(index):
        path = tmp_path / f"thread-{index}.daec"
        with open_dataecon(path, "a") as db:
            for attempt in range(10):
                name = f"missing-{index}-{attempt}"
                with pytest.raises(DataEconError) as caught:
                    db.read_series(name)
                assert caught.value.code == -989
                assert name in caught.value.native_message
                db.write_series(f"v{attempt}", TSeries(mm(2024, 1), VALUES.copy()))
                np.testing.assert_array_equal(db.read_series(f"v{attempt}").values, VALUES)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(worker, range(3)))


def test_fallback_finalizer_releases_file(tmp_path):
    path = tmp_path / "finalizer.daec"
    db = open_dataecon(path, "a")
    db.write_series("sample", TSeries(mm(2024, 1), VALUES.copy()))
    del db
    gc.collect()
    with open_dataecon(path) as reopened:
        np.testing.assert_array_equal(reopened.read_series("sample").values, VALUES)
    path.unlink()


def test_column_view_write_does_not_change_parent(tmp_path):
    parent = MVTSeries(mm(2024, 1), a=VALUES.copy(), b=np.ones(4))
    before = parent.values.copy()
    with open_dataecon(tmp_path / "column.daec", "a") as db:
        db.write_series("sample", parent["a"])
        loaded = db.read_series("sample")
    loaded.values[:] = 100
    np.testing.assert_array_equal(parent.values, before)


def test_non_roundtrippable_dates_rejected_before_storage(tmp_path):
    with open_dataecon(tmp_path / "date.daec", "a") as db:
        with pytest.raises(ValueError, match="round-trip"):
            db.write_series("invalid_date", TSeries(mm(-170000000, 1), VALUES.copy()))
        with pytest.raises(DataEconError) as caught:
            db.read_series("invalid_date")
        assert caught.value.code == -989


def test_corrupt_file_reports_native_error(tmp_path):
    path = tmp_path / "corrupt.daec"
    path.write_bytes(b"This is not a SQLite database." * 8)
    with pytest.raises(DataEconError) as caught, open_dataecon(path) as db:
        db.read_series("sample")
    assert caught.value.code > 0


def test_native_close_failure_never_retried():
    # A failed native read-only INSERT leaves a SQLite statement whose finalize
    # reports the prior error. Exercise the real failure in a subprocess because
    # the exported ABI has no safe recovery. The public API prevents this write.
    program = """
import sys
from tsecon.dataecon import open_dataecon, DataEconError
db = open_dataecon(sys.argv[1])
try:
    db._handle.write('readonly_fault', 32, 2024, 1, bytes(32))
except DataEconError as error:
    assert error.code > 0
else:
    raise AssertionError('native read-only write unexpectedly succeeded')
try:
    db.close()
except DataEconError as error:
    assert error.code > 0
else:
    raise AssertionError('expected native finalization error')
assert db.closed
for close in (db.close, db._handle.close):
    try:
        close()
    except ValueError:
        pass
    else:
        raise AssertionError('failed close was retried')
"""
    subprocess.run(
        [sys.executable, "-c", program, str(FIXTURES / "julia_monthly.daec")],
        check=True,
        timeout=30,
    )
