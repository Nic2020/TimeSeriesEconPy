"""File operations: delete, overwrite, truncate, in-memory databases and emptiness.

Mirrors Julia's ``delete_object``, ``DEFile{overwrite}`` stores,
``truncatedaec``/``empty!``, ``opendaecmem`` and ``isempty``. Destructive
operations are exercised only on fresh temporary files, fresh in-memory
databases and copies of the Julia fixture.
"""

import hashlib
import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tomllib
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

import tsecon.dataecon as de
from tsecon import MIT, TSeries, mm
from tsecon.dataecon import DataEconError, open_dataecon, open_dataecon_memory

FIXTURES = Path(__file__).parent / "fixtures"
NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built",
)
SERIES = TSeries(mm(2024, 1), np.array([1.0, 2.0, 3.0]))


def objects(path):
    with closing(sqlite3.connect(path)) as conn:
        return {
            row[0]: row[1:]
            for row in conn.execute("SELECT name,id,pid,class,type FROM objects WHERE id!=0")
        }


def table_count(path, table, where="1=1"):
    with closing(sqlite3.connect(path)) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0]


def fixture_copy(tmp_path, name="copy.daec"):
    path = tmp_path / name
    shutil.copyfile(FIXTURES / "julia_file_operations.daec", path)
    return path


# ---------------------------------------------------------------- no native needed


class RecordingHandle:
    def __init__(self):
        self.calls = []

    def close(self):
        self.calls.append("close")

    def catalog_size(self):
        self.calls.append("catalog_size")
        return 0

    def truncate(self):
        self.calls.append("truncate")

    def delete(self, name, recursive):
        self.calls.append(("delete", name, recursive))


@pytest.fixture
def readonly_owner(monkeypatch):
    handle = RecordingHandle()
    monkeypatch.setattr(de, "_open_native", lambda *_, **__: handle)
    return de.open_dataecon("example.daec"), handle


def test_readonly_owner_rejects_destructive_operations_before_native(readonly_owner):
    db, handle = readonly_owner
    with pytest.raises(ValueError, match="read-only"):
        db.delete("sample")
    with pytest.raises(ValueError, match="read-only"):
        db.truncate()
    with pytest.raises(ValueError, match="read-only"):
        db.write_scalar("sample", 1.0, overwrite=True)
    with pytest.raises(ValueError, match="read-only"):
        db.write_series("sample", SERIES, overwrite=True)
    assert db.is_empty() is True
    db.close()
    assert handle.calls == ["catalog_size", "close"]


def test_write_mode_truncates_after_opening(monkeypatch):
    handle = RecordingHandle()
    monkeypatch.setattr(de, "_open_native", lambda *_, **__: handle)
    db = de.open_dataecon("example.daec", "w")
    assert not db.closed
    db.delete("sample", recursive=True)
    db.close()
    assert handle.calls == ["truncate", ("delete", "sample", True), "close"]


def test_operations_after_close(readonly_owner):
    db, _ = readonly_owner
    db.close()
    for operation in (lambda: db.delete("x"), db.truncate, db.is_empty):
        with pytest.raises(ValueError, match="closed"):
            operation()


def test_failed_truncate_quarantines_owner(monkeypatch):
    handle = RecordingHandle()

    def fail():
        handle.calls.append("truncate")
        raise de.DataEconError(1, "truncate", "example.daec", "injected")

    handle.truncate = fail
    monkeypatch.setattr(de, "_open_native", lambda *_, **__: handle)
    db = de.open_dataecon("example.daec", "a")
    with pytest.raises(de.DataEconError, match="injected"):
        db.truncate()
    assert db.closed
    with pytest.raises(ValueError, match="truncate-failed"):
        db.is_empty()
    with pytest.raises(ValueError, match="truncate-failed"):
        db.close()
    assert handle.calls == ["truncate"]


@pytest.mark.parametrize("mode", ["x", "rw", "w+", "", None])
def test_invalid_modes(monkeypatch, mode):
    monkeypatch.setattr(de, "_open_native", lambda *_, **__: pytest.fail("reached native"))
    with pytest.raises(ValueError, match="mode"):
        de.open_dataecon("example.daec", mode)


@pytest.mark.parametrize("mode", ["r", "a", "w"])
def test_memory_literal_never_reaches_path_normalization(monkeypatch, tmp_path, mode):
    monkeypatch.setattr(de, "_open_native", lambda *_, **__: pytest.fail("reached native"))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="open_dataecon_memory"):
        de.open_dataecon(":memory:", mode)
    assert not (tmp_path / ":memory:").exists()
    assert os.listdir(tmp_path) == []


def test_delete_validates_name_before_native(readonly_owner):
    db, handle = readonly_owner
    for name, error in (
        ("", ValueError),
        ("a/b", ValueError),
        ("a\0b", ValueError),
        (1, TypeError),
    ):
        with pytest.raises(error):
            db.delete(name)
    assert handle.calls == []


# ---------------------------------------------------------------- native


@NATIVE
def test_julia_file_operations_fixture():
    fixture = FIXTURES / "julia_file_operations.daec"
    provenance = tomllib.loads(fixture.with_suffix(".toml").read_text())
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    with open_dataecon(fixture) as db:
        assert db.is_empty() is False
        assert db.read_scalar("a") == "replaced"
        np.testing.assert_array_equal(db.read_series("s2").values, SERIES.values)
        restored = db.read_series("overwritten_series")
        assert restored.firstdate == mm(2024, 1)
        np.testing.assert_array_equal(restored.values, SERIES.values)
        for name in ("b", "s", "r", "deleted_catalog", "gone_before_truncate", "gonecat"):
            with pytest.raises(DataEconError) as caught:
                db.read_scalar(name)
            assert caught.value.code == -989
        with pytest.raises(DataEconError):
            db.read_scalar("keep")  # a catalog is not a scalar
        with pytest.raises(ValueError, match="read-only"):
            db.delete("keep", recursive=True)
    objs = objects(fixture)
    assert set(objs) == {"keep", "z", "inner", "w", "a", "s2", "overwritten_series"}
    assert objs["keep"][0] == 1  # ids restarted after Julia's truncation
    assert objs["keep"][2] == 0  # class_catalog
    assert objs["a"][2:] == (1, 6)  # scalar, string
    assert objs["overwritten_series"][2:] == (2, 12)
    assert table_count(fixture, "attributes", "id!=0") == 0  # the Rational's jtype went with it
    assert table_count(fixture, "axes") == 1  # one shared axis, kept after the delete of "s"
    with closing(sqlite3.connect(fixture)) as conn:
        orphans = conn.execute(
            "SELECT COUNT(*) FROM objects o LEFT JOIN objects p ON o.pid=p.id WHERE p.id IS NULL"
        ).fetchone()[0]
    assert orphans == 0


@NATIVE
def test_delete_scalar_series_and_missing(tmp_path):
    path = tmp_path / "delete.daec"
    with open_dataecon(path, "a") as db:
        assert db.is_empty() is True
        db.write_scalar("x", 1)
        db.write_scalar("y", "text")
        db.write_series("s", SERIES)
        db.write_series("s2", SERIES)
        assert db.is_empty() is False
        loaded = db.read_series("s")
        db.delete("s")
        db.delete("y")
        with pytest.raises(DataEconError) as caught:
            db.delete("y")
        assert caught.value.code == -989
        assert caught.value.name == "y"
        with pytest.raises(DataEconError) as caught:
            db.read_scalar("y")
        assert caught.value.code == -989
        # The owner stays usable and earlier results remain valid.
        np.testing.assert_array_equal(loaded.values, SERIES.values)
        assert db.read_scalar("x") == 1
        np.testing.assert_array_equal(db.read_series("s2").values, SERIES.values)
        db.write_series("s", TSeries(mm(2025, 1), np.array([9.0])))
        assert db.read_series("s").firstdate == mm(2025, 1)
        db.delete("x")
        db.delete("s")
        db.delete("s2")
        assert db.is_empty() is True
    assert objects(path) == {}
    assert table_count(path, "scalars") == 0
    assert table_count(path, "tseries") == 0
    assert table_count(path, "axes") == 2  # axes are not objects and survive deletion


@NATIVE
def test_delete_catalog_requires_recursive(tmp_path):
    path = fixture_copy(tmp_path)
    before = objects(path)
    with open_dataecon(path, "a") as db:
        with pytest.raises(ValueError, match="recursive=True"):
            db.delete("keep")
        assert db.read_scalar("a") == "replaced"
    assert objects(path) == before
    with open_dataecon(path, "a") as db:
        db.delete("keep", recursive=True)
        assert db.is_empty() is False
        with pytest.raises(DataEconError) as caught:
            db.delete("keep", recursive=True)
        assert caught.value.code == -989
    after = objects(path)
    assert set(after) == {"a", "s2", "overwritten_series"}
    assert table_count(path, "objects_info", "id NOT IN (SELECT id FROM objects)") == 0
    assert table_count(path, "scalars", "id NOT IN (SELECT id FROM objects)") == 0
    # recursive=True on a plain object is simply a delete.
    with open_dataecon(path, "a") as db:
        db.delete("a", recursive=True)
    assert set(objects(path)) == {"s2", "overwritten_series"}


@NATIVE
def test_overwrite_replaces_across_classes_and_keeps_defaults(tmp_path):
    path = tmp_path / "overwrite.daec"
    with open_dataecon(path, "a") as db:
        db.write_scalar("v", 1)
        with pytest.raises(DataEconError) as caught:
            db.write_scalar("v", 2)
        assert caught.value.code == -985
        with pytest.raises(DataEconError):
            db.write_series("v", SERIES)
        assert db.read_scalar("v") == 1
        first_id = objects(path)  # not yet visible: writes commit at close
        assert first_id == {}
        db.write_scalar("v", "two", overwrite=True)
        assert db.read_scalar("v") == "two"
        db.write_series("v", SERIES, overwrite=True)
        np.testing.assert_array_equal(db.read_series("v").values, SERIES.values)
        with pytest.raises(DataEconError):
            db.read_scalar("v")
        db.write_scalar("v", np.float16(0.5), overwrite=True)
        assert db.read_scalar("v") == np.float16(0.5)
        db.write_scalar("fresh", 1.5, overwrite=True)  # overwrite of a missing name is a write
        assert db.read_scalar("fresh") == 1.5
        db.write_series("fresh", SERIES, overwrite=True)
        db.write_series("fresh", TSeries(mm(2030, 6), np.array([7.0])), overwrite=True)
        assert db.read_series("fresh").firstdate == mm(2030, 6)
    objs = objects(path)
    assert set(objs) == {"v", "fresh"}
    assert objs["v"][2:] == (1, 4)
    assert objs["v"][0] > 3  # replaced objects get new ids; old ids are gone
    assert table_count(path, "scalars") == 1
    assert table_count(path, "tseries") == 1


@NATIVE
def test_overwrite_validates_before_deleting(tmp_path):
    path = tmp_path / "validate.daec"
    with open_dataecon(path, "a") as db:
        db.write_scalar("keep_me", 1)
        db.write_series("keep_series", SERIES)
        for bad in (np.array(True), b"x", np.array([1.0]), object()):
            with pytest.raises(TypeError):
                db.write_scalar("keep_me", bad, overwrite=True)
        with pytest.raises(ValueError):
            db.write_scalar("keep_me", 2**63, overwrite=True)
        with pytest.raises(ValueError):
            db.write_scalar("keep_me", "with\0nul", overwrite=True)
        with pytest.raises(ValueError, match="date range"):
            db.write_scalar("keep_me", MIT(mm(2024, 1).frequency, -(2**31)), overwrite=True)
        with pytest.raises(TypeError):
            db.write_series("keep_series", np.ones(3), overwrite=True)
        with pytest.raises(ValueError, match="round-trip"):
            db.write_series("keep_series", TSeries(mm(-170000000, 1), np.ones(3)), overwrite=True)
        assert db.read_scalar("keep_me") == 1
        np.testing.assert_array_equal(db.read_series("keep_series").values, SERIES.values)


@NATIVE
def test_overwrite_refuses_catalogs(tmp_path):
    path = fixture_copy(tmp_path)
    before = objects(path)
    with open_dataecon(path, "a") as db:
        with pytest.raises(ValueError, match="catalog"):
            db.write_scalar("keep", 1, overwrite=True)
        with pytest.raises(ValueError, match="catalog"):
            db.write_series("keep", SERIES, overwrite=True)
        with pytest.raises(DataEconError) as caught:
            db.write_scalar("keep", 1)
        assert caught.value.code == -985
    assert objects(path) == before


@NATIVE
def test_truncate_then_write_then_reopen(tmp_path):
    path = tmp_path / "truncate.daec"
    with open_dataecon(path, "a") as db:
        db.write_scalar("gone", 1)
        db.write_series("gone_series", SERIES)
        assert db.is_empty() is False
        db.truncate()
        assert db.is_empty() is True
        assert not db.closed
        with pytest.raises(DataEconError) as caught:
            db.read_scalar("gone")
        assert caught.value.code == -989
        db.truncate()  # empty root: native no-op
        db.write_scalar("after", 42)
        db.write_series("after_series", SERIES)
        assert db.read_scalar("after") == 42
    with open_dataecon(path) as db:
        assert db.is_empty() is False
        assert db.read_scalar("after") == 42
        np.testing.assert_array_equal(db.read_series("after_series").values, SERIES.values)
    objs = objects(path)
    assert set(objs) == {"after", "after_series"}
    assert objs["after"][0] == 1  # ids restart after truncation
    assert table_count(path, "axes") == 1
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute(
            "SELECT value FROM attributes WHERE id=0 AND name='DE_VERSION'"
        ).fetchone() == ("0.4.0",)


@NATIVE
def test_write_mode_truncates_existing_and_creates_new(tmp_path):
    path = tmp_path / "wmode.daec"
    with open_dataecon(path, "w") as db:
        assert db.is_empty() is True
        db.write_scalar("first", 1)
    with open_dataecon(path, "a") as db:
        assert db.read_scalar("first") == 1
        db.write_scalar("second", 2)
    with open_dataecon(path, "w") as db:
        assert db.is_empty() is True
        with pytest.raises(DataEconError):
            db.read_scalar("first")
        db.write_scalar("third", 3)
    with open_dataecon(path) as db:
        assert db.is_empty() is False
        assert db.read_scalar("third") == 3
        with pytest.raises(DataEconError):
            db.read_scalar("second")
    assert set(objects(path)) == {"third"}


@NATIVE
def test_readonly_file_bytes_are_never_touched(tmp_path):
    path = fixture_copy(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with open_dataecon(path) as db:
        assert db.is_empty() is False
        for operation in (
            lambda: db.delete("a"),
            lambda: db.delete("keep", recursive=True),
            db.truncate,
            lambda: db.write_scalar("a", 1, overwrite=True),
            lambda: db.write_series("a", SERIES, overwrite=True),
        ):
            with pytest.raises(ValueError, match="read-only"):
                operation()
        assert db.read_scalar("a") == "replaced"
    with pytest.raises(ValueError, match="read-only"):
        open_dataecon(path).delete("a")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


@NATIVE
def test_memory_database_lifecycle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = open_dataecon_memory()
    assert db.path == ":memory:"
    assert not db.closed
    assert db.is_empty() is True
    db.write_scalar("m", 1.5)
    db.write_series("ms", SERIES)
    assert db.is_empty() is False
    assert db.read_scalar("m") == 1.5
    with pytest.raises(DataEconError) as caught:
        db.write_scalar("m", 2.5)
    assert caught.value.code == -985
    db.write_scalar("m", 2.5, overwrite=True)
    assert db.read_scalar("m") == 2.5
    db.delete("ms")
    db.truncate()
    assert db.is_empty() is True
    db.write_scalar("m", np.uint8(7))
    assert db.read_scalar("m") == np.uint8(7)  # memory databases are always writable
    db.close()
    assert db.closed
    with pytest.raises(ValueError, match="closed"):
        db.read_scalar("m")
    assert os.listdir(tmp_path) == []
    with open_dataecon_memory() as fresh:
        assert fresh.is_empty() is True  # each memory database starts empty
        with pytest.raises(DataEconError):
            fresh.read_scalar("m")
    assert de.DataEconFile.in_memory().is_empty() is True
    assert os.listdir(tmp_path) == []


@NATIVE
def test_memory_database_context_and_body_exception(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def body():
        with open_dataecon_memory() as db:
            db.write_scalar("m", 1)
            raise RuntimeError("body")

    with pytest.raises(RuntimeError, match="body"):
        body()
    assert os.listdir(tmp_path) == []


@NATIVE
def test_delete_and_truncate_interleave_with_pending_reads(tmp_path):
    path = tmp_path / "interleave.daec"
    with open_dataecon(path, "a") as db:
        db.write_scalar("a", np.int32(5))
        first = db.read_scalar("a")
        db.write_scalar("a", np.int32(6), overwrite=True)
        second = db.read_scalar("a")
        db.delete("a")
        db.truncate()
        assert first == np.int32(5)
        assert second == np.int32(6)
        assert db.is_empty() is True
        db.write_scalar("a", np.int32(7))
        assert db.read_scalar("a") == np.int32(7)


@NATIVE
def test_native_truncate_failure_quarantines_without_second_native_call():
    # A read-only handle cannot be truncated natively (the public API refuses
    # it first). Exercise the backend on a fresh copy in a subprocess: the
    # native failure must quarantine the handle, and no further native call
    # (including close) may be attempted, because the native finalization path
    # is unsafe after a failed statement.
    program = """
import shutil, sys, tempfile, os
from pathlib import Path
from tsecon.dataecon import open_dataecon, DataEconError
folder = tempfile.mkdtemp()
path = Path(folder) / 'copy.daec'
shutil.copyfile(sys.argv[1], path)
db = open_dataecon(path)
assert not db.is_empty()
try:
    db._handle.truncate()
except DataEconError as error:
    assert error.code > 0, error
    assert error.operation == 'truncate'
else:
    raise AssertionError('native read-only truncate unexpectedly succeeded')
for call in (db._handle.close, db._handle.truncate, db._handle.catalog_size):
    try:
        call()
    except ValueError as error:
        assert 'close failed' in str(error) or 'closed' in str(error)
    else:
        raise AssertionError('quarantined handle accepted a native call')
try:
    db.close()
except ValueError:
    pass
else:
    raise AssertionError('owner close did not report the quarantine')
assert db.closed
print('quarantined')
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(FIXTURES / "julia_file_operations.daec")],
        check=True,
        timeout=60,
        capture_output=True,
        text=True,
    )
    assert "quarantined" in result.stdout


# Fault injection after the overwrite delete. The native store creates the
# object row (_new_object) before it inserts the payload row, so a failure in
# between leaves the name present without a value. A SQLite trigger on a fresh
# temporary database makes the payload insert fail with SQLITE_CONSTRAINT (19)
# and needs no native hook. The failed statement makes the later native close
# fail too (statement finalization reports it), which quarantines the owner and
# leaves the native connection open; the whole sequence therefore runs in a
# subprocess and the file is inspected only after that process has exited.
FAULT_PROGRAM = """
import json, sys
import numpy as np
from tsecon import TSeries, mm
from tsecon.dataecon import open_dataecon, DataEconError
path, table = sys.argv[1], sys.argv[2]
series = TSeries(mm(2024, 1), np.array([1.0, 2.0, 3.0]))
db = open_dataecon(path, "a")
report = {}
try:
    if table == "scalars":
        db.write_scalar("keep", 2, overwrite=True)
    else:
        db.write_series("keep", series, overwrite=True)
except DataEconError as error:
    report["write"] = [error.code, error.operation, error.name]
else:
    raise AssertionError("injected store failure did not surface")
# The owner is still open; the failed store is a write error, not a quarantine.
report["closed_after_write"] = db.closed
try:
    db.read_scalar("keep")
except DataEconError as error:
    report["read_after_failure"] = error.code
else:
    report["read_after_failure"] = "readable"
try:
    db.close()
except DataEconError as error:
    report["close"] = error.code
else:
    report["close"] = "ok"
report["closed"] = db.closed
try:
    db.is_empty()
except ValueError as error:
    report["after_close"] = str(error)
print(json.dumps(report))
"""


@NATIVE
@pytest.mark.parametrize("table", ["scalars", "tseries"])
def test_overwrite_store_failure_after_delete_leaves_partial_replacement(tmp_path, table):
    path = tmp_path / "owned-test.daec"
    with open_dataecon(path, "a") as db:
        db.write_scalar("keep", 1)
        db.write_scalar("other", 3)
    before = objects(path)
    assert before["keep"][2:] == (1, 1)  # scalar, Int64
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            f"CREATE TRIGGER fail_{table} BEFORE INSERT ON {table} "
            "BEGIN SELECT RAISE(ABORT, 'injected store failure'); END"
        )
    result = subprocess.run(
        [sys.executable, "-c", FAULT_PROGRAM, str(path), table],
        check=True,
        timeout=120,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    operation = "write_scalar" if table == "scalars" else "write"
    assert report["write"] == [
        19,
        f"{operation} (overwrite; original deleted, partial replacement may remain)",
        "keep",
    ]
    assert report["closed_after_write"] is False
    assert report["read_after_failure"] != "readable"
    # The failed statement makes native close fail as well; quarantine holds.
    assert report["close"] == 19
    assert report["closed"] is True
    assert "close-failed" in report["after_close"]
    # Committed state: the original is gone, the name exists with a new id and
    # no payload row. No rollback happened and none is promised.
    after = objects(path)
    assert set(after) == {"keep", "other"}
    assert after["other"] == before["other"]
    assert after["keep"][0] != before["keep"][0]
    assert table_count(path, "scalars", f"id={after['keep'][0]}") == 0
    assert table_count(path, "tseries", f"id={after['keep'][0]}") == 0
    assert table_count(path, "objects_info", f"id={after['keep'][0]}") == 1
    # A fresh read-only owner sees the partial replacement as unreadable, not absent.
    with open_dataecon(path) as db:
        assert db.read_scalar("other") == 3
        with pytest.raises(DataEconError) as caught:
            db.read_scalar("keep")
        assert caught.value.code != -989
        with pytest.raises(DataEconError) as caught:
            db.read_series("keep")
        assert caught.value.code != -989
        assert db.is_empty() is False


@NATIVE
def test_overwrite_store_failure_without_prior_object_leaves_partial_object(tmp_path):
    # Same injection on a name that did not exist: no delete happens, the
    # ordinary "partial object may remain" label applies.
    path = tmp_path / "fresh.daec"
    with open_dataecon(path, "a") as db:
        db.write_scalar("other", 3)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "CREATE TRIGGER fail_scalars BEFORE INSERT ON scalars "
            "BEGIN SELECT RAISE(ABORT, 'injected store failure'); END"
        )
    program = FAULT_PROGRAM.replace('"keep", 2, overwrite=True', '"keep", 2')
    result = subprocess.run(
        [sys.executable, "-c", program, str(path), "scalars"],
        check=True,
        timeout=120,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["write"] == [19, "write_scalar (partial object may remain)", "keep"]
    assert report["close"] == 19
    after = objects(path)
    assert set(after) == {"other", "keep"}
    assert table_count(path, "scalars", f"id={after['keep'][0]}") == 0
