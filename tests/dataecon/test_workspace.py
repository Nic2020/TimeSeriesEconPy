"""Workspace interchange: whole catalog trees to and from the core ``Workspace``.

Mirrors Julia's ``writedb``/``readdb`` (and the single-object ``write_data``/
``read_data``) through ``write_workspace``/``read_workspace``,
``write_object``/``read_object`` and the ``save_workspace``/``load_workspace``
file forms, with skip-and-report as the default and an explicit strict mode.
Destructive operations run on fresh temporary files, fresh in-memory
databases and copies of the Julia fixture.
"""

import hashlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
import tomllib
from contextlib import closing
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

import tsecon.dataecon as de
from tsecon import Duration, MITRange, MVTSeries, TSeries, Workspace, mm, qq, yy
from tsecon.dataecon import (
    DataEconError,
    LoadedWorkspace,
    SkippedMember,
    StoredArray,
    StoredElement,
    StoredSeries,
    StoredText,
    WorkspaceReport,
    load_workspace,
    open_dataecon,
    open_dataecon_memory,
    save_workspace,
)
from tsecon.dataecon._workspace import _key_problem, classify, reader_for
from tsecon.frequencies import Monthly

FIXTURES = Path(__file__).parent / "fixtures"
NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built",
)
DATES = np.array([24288, 24289], dtype="<i8")
TENSOR = np.arange(1, 25, dtype=np.int64).reshape((2, 3, 4), order="F")
ORDER = [("b", 1), ("a", 2), ("B", 3), ("ä", 4), ("_", 5), ("10", 6), ("9", 7)]


def byte_sorted(names):
    return sorted(names, key=lambda n: n.encode("utf-8"))


def mixed_workspace() -> Workspace:
    """Every family the adapter writes, nested and empty Workspaces, byte-order keys."""
    ws = Workspace()
    ws.f = 1.5
    ws.i = 7
    ws.i8 = np.int8(-3)
    ws.u = np.uint64(2**63)
    ws.c = 1.0 + 2.0j
    ws.s = "vintage ‖ 3"
    ws.b = True
    ws.d = qq(2020, 1)
    ws.dur = Duration(Monthly(), 2)
    ws.ts = TSeries(mm(2024, 1), np.array([1.0, 2.0, 3.0]))
    ws.tsi = TSeries(qq(2020, 1), np.array([1, 2], dtype=np.int64))
    ws.tsb = TSeries(yy(2020), np.array([True, False]))
    ws.tse = TSeries(mm(2024, 1), np.array([], dtype=np.float64))
    ws.tsd = StoredSeries(mm(2024, 1), DATES, StoredElement.date(Monthly()))
    ws.mv = MVTSeries(qq(2024, 1), ("p", "q"), np.array([[1.0, 2.0], [3.0, 4.0]]))
    ws.mat = np.array([[1.0, 2.0], [3.0, 4.0]])
    ws.v = np.array([1, 2, 3], dtype=np.int32)
    ws.vs = ["a", "b"]
    ws.vt = StoredText((b"x", b"\xff"), "String")
    ws.va = StoredArray(DATES.reshape((1, 2)), StoredElement.date(Monthly()))
    ws.ur = range(1, 6)
    ws.mr = MITRange(qq(2020, 1), qq(2020, 4))
    ws.t3 = TENSOR
    ws.nested = Workspace(x=1, deeper=Workspace(y=2.0))
    ws.nested.deeper["日本語"] = 9
    ws.empty = Workspace()
    ws.order = Workspace()
    for name, value in ORDER:
        ws.order[name] = value
    return ws


# Every path the mixed Workspace stores below its destination, with the stored
# (class, type): the complete inventory, asserted as a whole.
MIXED_INVENTORY = {
    "/f": (1, 4),
    "/i": (1, 1),
    "/i8": (1, 1),
    "/u": (1, 2),
    "/c": (1, 5),
    "/s": (1, 6),
    "/b": (1, 1),
    "/d": (1, 3),
    "/dur": (1, 1),
    "/ts": (2, 12),
    "/tsi": (2, 12),
    "/tsb": (2, 12),
    "/tse": (2, 12),
    "/tsd": (2, 12),
    "/mv": (3, 21),
    "/mat": (3, 20),
    "/v": (2, 10),
    "/vs": (2, 10),
    "/vt": (2, 10),
    "/va": (3, 20),
    "/ur": (2, 11),
    "/mr": (2, 11),
    "/t3": (4, 30),
    "/nested": (0, 0),
    "/nested/x": (1, 1),
    "/nested/deeper": (0, 0),
    "/nested/deeper/y": (1, 4),
    "/nested/deeper/日本語": (1, 1),
    "/empty": (0, 0),
    "/order": (0, 0),
    **{f"/order/{name}": (1, 1) for name, _ in ORDER},
}
MIXED_MARKERS = {
    "/tsb": {"jeltype": "Bool"},
    "/tsd": {},
    "/vt": {"jeltype": "String"},
    "/vs": {},
    "/ur": {},
    "/mr": {},
}


def assert_mixed_loaded(ws: Workspace) -> None:
    """The mixed Workspace read back: keys in byte order, every value exact."""
    expected = mixed_workspace()
    assert list(ws.keys()) == byte_sorted(expected.keys())
    for name in ("f", "i", "c", "s", "d", "dur"):
        assert type(ws[name]) is type(expected[name])
        assert ws[name] == expected[name]
    assert ws.i8 == np.int8(-3)
    assert type(ws.i8) is np.int8
    assert ws.u == np.uint64(2**63)
    assert type(ws.u) is np.uint64
    assert ws.b == np.int8(1)
    assert type(ws.b) is np.int8
    for name in ("ts", "tsi", "tsb", "tse"):
        got, want = ws[name], expected[name]
        assert isinstance(got, TSeries)
        assert got.firstdate == want.firstdate
        assert got.values.dtype == want.values.dtype
        assert got.values.tolist() == want.values.tolist()
    assert isinstance(ws.tsd, StoredSeries)
    assert ws.tsd.values.tolist() == DATES.tolist()
    assert ws.tsd.element == StoredElement.date(Monthly())
    assert isinstance(ws.mv, MVTSeries)
    assert list(ws.mv.columns) == ["p", "q"]
    np.testing.assert_array_equal(ws.mv.values, expected.mv.values)
    np.testing.assert_array_equal(ws.mat, expected.mat)
    np.testing.assert_array_equal(ws.v, expected.v)
    assert ws.v.dtype == np.int32
    np.testing.assert_array_equal(ws.t3, TENSOR)
    assert ws.t3.shape == (2, 3, 4)
    assert ws.t3.dtype == np.int64
    assert ws.vs == ["a", "b"]
    assert isinstance(ws.vt, StoredText)
    assert ws.vt.values == (b"x", b"\xff")
    assert isinstance(ws.va, StoredArray)
    assert ws.va.shape == (1, 2)
    assert ws.ur == range(1, 6)
    assert list(ws.mr) == list(expected.mr)
    assert list(ws.nested.keys()) == ["deeper", "x"]
    assert ws.nested.x == 1
    assert list(ws.nested.deeper.keys()) == ["y", "日本語"]
    assert ws.nested.deeper.y == 2.0
    assert ws.nested.deeper["日本語"] == 9
    assert isinstance(ws.empty, Workspace)
    assert len(ws.empty) == 0
    assert list(ws.order.keys()) == ["10", "9", "B", "_", "a", "b", "ä"]
    assert [ws.order[n] for n, _ in ORDER] == [v for _, v in ORDER]


def stored_inventory(db, prefix=""):
    """Every object below ``prefix`` as ``{path: (class, type)}`` (the root prefix is '')."""
    entries = db.list_objects(prefix or "/", recursive=True)
    return {e.path[len(prefix) :]: (e.object_class, e.object_type) for e in entries}


def raw_rows(path, exclude=()):
    with closing(sqlite3.connect(path)) as conn:
        info = conn.execute(
            "SELECT o.fullpath, o.depth, b.class, b.type, b.name FROM objects_info o "
            "JOIN objects b ON o.id = b.id WHERE o.id != 0 ORDER BY o.fullpath"
        ).fetchall()
        attrs = conn.execute(
            "SELECT o.fullpath, a.name, a.value FROM attributes a JOIN objects_info o "
            "ON a.id = o.id ORDER BY o.fullpath, a.name"
        ).fetchall()
    return (
        [row for row in info if row[0] not in exclude],
        [row for row in attrs if row[0] not in exclude],
    )


# ---------------------------------------------------------------- no native needed


def test_report_types():
    skipped = SkippedMember("/a/b", "unsupported", "TypeError: no", True)
    report = WorkspaceReport("/a", 3, (skipped,))
    assert not report.ok
    assert report.count == 3
    assert report.skipped[0].subtree
    assert WorkspaceReport("/", 0, ()).ok
    loaded = LoadedWorkspace(Workspace(), report)
    ws, rep = loaded
    assert isinstance(ws, Workspace)
    assert rep is report
    assert loaded.workspace is ws
    with pytest.raises(AttributeError):
        skipped.path = "/x"  # frozen


@pytest.mark.parametrize(
    ("value", "family"),
    [
        (Workspace(), "workspace"),
        (TSeries(mm(2024, 1), np.array([1.0])), "series"),
        (MVTSeries(qq(2024, 1), ("p",), np.array([[1.0]])), "series"),
        (StoredSeries(mm(2024, 1), DATES, StoredElement.date(Monthly())), "series"),
        (np.array([1.0]), "array"),
        (np.array([[1.0]]), "array"),
        (StoredArray(DATES.reshape((1, 2)), StoredElement.date(Monthly())), "array"),
        (StoredText((b"a",), None), "array"),
        (["a"], "array"),
        (("a",), "array"),
        (range(1, 3), "array"),
        (MITRange(qq(2020, 1), qq(2020, 2)), "array"),
        (True, "scalar"),
        (1, "scalar"),
        (1.5, "scalar"),
        (1j, "scalar"),
        ("s", "scalar"),
        (qq(2020, 1), "scalar"),
        (Duration(Monthly(), 1), "scalar"),
        (np.int8(1), "scalar"),
        (np.bool_(True), "scalar"),
        (np.float64(1.0), "scalar"),
        (None, None),
        ({"a": 1}, None),
        ({1, 2}, None),
        (Fraction(1, 2), None),
        (Monthly(), None),
        (object(), None),
        (b"bytes", None),
    ],
)
def test_write_dispatch_is_by_type(value, family):
    assert classify(value) == family


def test_read_dispatch_covers_exactly_the_typed_readers():
    assert {
        k: v for k, v in [((c, t), reader_for(c, t)) for c in range(5) for t in range(33)] if v
    } == {
        (1, 1): "read_scalar",
        (1, 2): "read_scalar",
        (1, 3): "read_scalar",
        (1, 4): "read_scalar",
        (1, 5): "read_scalar",
        (1, 6): "read_scalar",
        (2, 10): "read_array",
        (2, 11): "read_array",
        (2, 12): "read_series",
        (3, 20): "read_array",
        (3, 21): "read_series",
        (4, 30): "read_array",
    }
    # Native-only types Julia never writes: class 0 is a catalog, handled separately.
    for pair in ((1, 7), (2, 13), (3, 22), (4, 31), (4, 32), (0, 0)):
        assert reader_for(*pair) is None


@pytest.mark.parametrize(
    ("name", "problem"),
    [
        ("ok", None),
        ("a b", None),
        ("a\\b", None),
        (".", None),
        ("日本語", None),
        ("", "Empty"),
        ("a/b", "'/'"),
        ("a\0b", "NUL"),
        ("   ", "Blank"),
        ("\t", "Blank"),
        (1, "strings"),
    ],
)
def test_member_key_rule(name, problem):
    found = _key_problem(name)
    assert (found is None) == (problem is None)
    if problem:
        assert problem in found


def test_workspace_methods_validate_before_native(monkeypatch):
    calls = []

    class Handle:
        def close(self):
            calls.append("close")

        def describe_path(self, path):
            calls.append(("describe", path))
            return (1, 0, 1, 4, "x", "/x", 1, 0)

    monkeypatch.setattr(de, "_open_native", lambda *_, **__: Handle())
    db = open_dataecon("example.daec")
    with pytest.raises(TypeError, match="requires a Workspace"):
        db.write_workspace({"a": 1})
    with pytest.raises(ValueError, match="Invalid DataEcon path"):
        db.write_workspace(Workspace(), "/a//b")
    with pytest.raises(ValueError, match="read-only"):
        db.write_workspace(Workspace(a=1))
    with pytest.raises(ValueError, match="read-only"):
        db.write_workspace(Workspace(), "/x")
    with pytest.raises(ValueError, match="read-only"):
        db.write_object("/y", 1.0)
    with pytest.raises(TypeError, match="write_workspace"):
        db.write_object("/y", Workspace())
    with pytest.raises(TypeError, match="No DataEcon storage class"):
        db.write_object("/y", object())
    with pytest.raises(ValueError, match="Invalid DataEcon path"):
        db.read_workspace("/a//b")
    with pytest.raises(ValueError, match="not a catalog"):
        db.read_workspace("/x")
    with pytest.raises(ValueError, match="mode must be"):
        save_workspace("example.daec", Workspace(), mode="r")
    db.close()
    with pytest.raises(ValueError, match="closed"):
        db.read_workspace()
    with pytest.raises(ValueError, match="closed"):
        db.write_workspace(Workspace())
    with pytest.raises(ValueError, match="closed"):
        db.read_object("/x")
    assert calls == [("describe", "/x"), "close"]


# ---------------------------------------------------------------- native


@NATIVE
def test_mixed_workspace_round_trips_at_root_and_nested():
    with open_dataecon_memory() as db:
        report = db.write_workspace(mixed_workspace())
        assert report == WorkspaceReport("/", len(MIXED_INVENTORY), ())
        assert stored_inventory(db) == MIXED_INVENTORY
        for path, markers in MIXED_MARKERS.items():
            assert db.get_attributes(path) == markers
        loaded = db.read_workspace()
        assert isinstance(loaded, LoadedWorkspace)
        assert loaded.report == WorkspaceReport("/", len(MIXED_INVENTORY), ())
        assert_mixed_loaded(loaded.workspace)
        # Nested destination: the same tree below an existing catalog.
        db.new_catalog("/dest/inner", parents=True)
        report = db.write_workspace(mixed_workspace(), "dest/inner")
        assert report.path == "/dest/inner"
        assert report.ok
        assert stored_inventory(db, "/dest/inner") == MIXED_INVENTORY
        ws, rep = db.read_workspace("/dest/inner")
        assert rep.path == "/dest/inner"
        assert rep.count == len(MIXED_INVENTORY)
        assert_mixed_loaded(ws)
        assert list(db.read_workspace("/dest").workspace.keys()) == ["inner"]
        assert_mixed_loaded(db.read_workspace("/dest").workspace.inner)
        # Ids follow depth-first insertion order (Julia's recursion order).
        ids = [
            db.object_id(p)
            for p in ("/f", "/ts", "/nested", "/nested/x", "/nested/deeper", "/empty")
        ]
        assert ids == sorted(ids)
        assert db.object_id("/nested/deeper/y") < db.object_id("/empty")


@NATIVE
def test_empty_workspaces_and_catalogs():
    with open_dataecon_memory() as db:
        assert db.write_workspace(Workspace()) == WorkspaceReport("/", 0, ())
        assert db.is_empty()
        ws, report = db.read_workspace()
        assert isinstance(ws, Workspace)
        assert len(ws) == 0
        assert report == WorkspaceReport("/", 0, ())
        db.write_workspace(Workspace(empty=Workspace(), deeper=Workspace(also=Workspace())))
        assert db.object_info("/empty").kind == "catalog"
        assert db.catalog_size("/empty") == 0
        assert db.catalog_size("/deeper/also") == 0
        ws, report = db.read_workspace()
        assert report.count == 3
        assert list(ws.keys()) == ["deeper", "empty"]
        assert len(ws.empty) == 0
        assert len(ws.deeper.also) == 0
        assert db.read_workspace("/empty") == LoadedWorkspace(
            Workspace(), WorkspaceReport("/empty", 0, ())
        )


@NATIVE
def test_unsupported_members_are_reported_and_siblings_kept():
    with open_dataecon_memory() as db:
        ws = Workspace(before=1, bad=object(), after=2)
        ws.sub = Workspace(ok=1, none=None, fraction=Fraction(1, 2), later=[1, 2], ok2=2)
        report = db.write_workspace(ws)
        assert [(s.path, s.category, s.subtree) for s in report.skipped] == [
            ("/bad", "unsupported", False),
            ("/sub/none", "unsupported", False),
            ("/sub/fraction", "unsupported", False),
            ("/sub/later", "unsupported", False),  # a list of numbers is not text
        ]
        assert all(s.reason.startswith("TypeError: ") for s in report.skipped)
        assert "object" in report.skipped[0].reason
        assert "plain Python strings" in report.skipped[3].reason
        assert report.count == 5
        assert stored_inventory(db) == {
            "/after": (1, 1),
            "/before": (1, 1),
            "/sub": (0, 0),
            "/sub/ok": (1, 1),
            "/sub/ok2": (1, 1),
        }
        # Strict: the first unsupported member raises its own TypeError, named,
        # after the members before it were stored; nothing after it is attempted.
        db.new_catalog("/strict")
        with pytest.raises(TypeError, match="No DataEcon storage class") as info:
            db.write_workspace(ws, "/strict", strict=True)
        assert "Workspace member '/strict/bad'" in info.value.__notes__
        assert stored_inventory(db, "/strict") == {"/before": (1, 1)}


@NATIVE
def test_invalid_keys_are_reported_before_native():
    with open_dataecon_memory() as db:
        ws = Workspace()
        ws["a/b"] = 1
        ws[""] = 2
        ws["   "] = 3
        ws["nul\0tail"] = 4
        ws["sub/tree"] = Workspace(inner=1)
        ws["fine"] = 5
        report = db.write_workspace(ws)
        assert [(s.path, s.category, s.subtree) for s in report.skipped] == [
            ("/a/b", "invalid", False),
            ("/", "invalid", False),
            ("/   ", "invalid", False),
            ("/nul\0tail", "invalid", False),
            ("/sub/tree", "invalid", True),
        ]
        assert report.count == 1
        assert stored_inventory(db) == {"/fine": (1, 1)}
        with pytest.raises(ValueError, match="'/'") as info:
            db.write_workspace(ws, strict=True)
        assert info.value.__notes__ == ["Workspace member '/a/b'"]


@NATIVE
def test_conflicts_and_overwrite_replacement():
    with open_dataecon_memory() as db:
        db.write_workspace(mixed_workspace())
        db.set_attribute("/f", "note", "kept?")
        db.write_scalar("/nested/extra", 3)
        before = {p: db.object_id(p) for p in ("/f", "/nested", "/nested/x", "/empty", "/s")}
        again = Workspace(f=2.5, nested=Workspace(x=10, brand_new=1), empty=1, s=Workspace(inner=1))
        report = db.write_workspace(again)
        assert [(s.path, s.category, s.subtree) for s in report.skipped] == [
            ("/f", "exists", False),
            ("/nested", "exists", True),
            ("/empty", "exists", False),
            ("/s", "exists", True),
        ]
        assert all("-985" in s.reason or "already exists" in s.reason for s in report.skipped)
        assert report.count == 0
        assert db.read_scalar("/f") == 1.5
        assert not db.exists("/nested/brand_new")
        assert db.read_scalar("/nested/x") == 1
        assert db.exists("/nested/extra")
        assert {p: db.object_id(p) for p in before} == before
        with pytest.raises(DataEconError, match="already exists") as info:
            db.write_workspace(again, strict=True)
        assert info.value.code == -985
        assert info.value.__notes__ == ["Workspace member '/f'"]
        # overwrite=True: a value replaces a value, a Workspace replaces a
        # catalog entirely (no merge) or a value; a value never replaces a catalog.
        report = db.write_workspace(again, overwrite=True)
        assert [(s.path, s.category, s.subtree) for s in report.skipped] == [
            ("/empty", "exists", False)
        ]
        assert "never replaces a catalog" in report.skipped[0].reason
        assert report.count == 6  # f, nested, nested/x, nested/brand_new, s, s/inner
        assert db.read_scalar("/f") == 2.5
        assert db.object_id("/f") != before["/f"]
        assert db.get_attribute("/f", "note") is None  # attributes go with the replaced object
        assert stored_inventory(db, "/nested") == {"/brand_new": (1, 1), "/x": (1, 1)}
        assert db.read_scalar("/nested/x") == 10
        assert db.object_id("/nested") != before["/nested"]
        assert db.object_info("/empty").kind == "catalog"
        assert db.object_id("/empty") == before["/empty"]
        assert db.object_info("/s").kind == "catalog"
        assert db.read_scalar("/s/inner") == 1
        with pytest.raises(DataEconError, match="never replaces a catalog") as info:
            db.write_workspace(Workspace(empty=1), overwrite=True, strict=True)
        assert info.value.code == -985
        assert db.object_info("/empty").kind == "catalog"


@NATIVE
def test_overwrite_count_and_leaf_over_leaf_families():
    with open_dataecon_memory() as db:
        db.write_workspace(Workspace(a=1, s=TSeries(mm(2024, 1), np.array([1.0])), v=[1, 2]))
        report = db.write_workspace(
            Workspace(a="text", s=np.array([1, 2]), v=Workspace(k=1)), overwrite=True
        )
        assert report.ok
        assert report.count == 4
        assert db.read_scalar("/a") == "text"
        np.testing.assert_array_equal(db.read_array("/s"), [1, 2])
        assert db.read_scalar("/v/k") == 1
        assert stored_inventory(db) == {"/a": (1, 6), "/s": (2, 10), "/v": (0, 0), "/v/k": (1, 1)}


@NATIVE
def test_destination_rules():
    with open_dataecon_memory() as db:
        db.write_scalar("/x", 1.0)
        with pytest.raises(DataEconError) as info:
            db.write_workspace(Workspace(z=1), "/nope")
        assert info.value.code == -989
        assert not db.exists("/z")
        with pytest.raises(ValueError, match="not a catalog"):
            db.write_workspace(Workspace(z=1), "/x")
        assert not db.exists("/x/z")
        assert db.catalog_size() == 1
        with pytest.raises(DataEconError) as info:
            db.read_workspace("/nope")
        assert info.value.code == -989
        with pytest.raises(ValueError, match="not a catalog"):
            db.read_workspace("/x")
        with pytest.raises(ValueError, match="not a catalog"):
            db.read_workspace("x")
        # The destination itself is never created; a Workspace member is.
        db.write_workspace(Workspace(made=Workspace()))
        assert db.object_info("/made").kind == "catalog"
        assert db.write_workspace(Workspace(z=1), "made").ok
        assert db.read_scalar("/made/z") == 1


@NATIVE
def test_read_reports_unsupported_marked_and_malformed_members(tmp_path):
    path = tmp_path / "foreign.daec"
    with open_dataecon(path, "a") as db:
        db.write_workspace(
            Workspace(a=1, marked=2, native_only=3, short=4.0, sub=Workspace(ok=1, bad=5.0), z=6)
        )
        ids = {p: db.object_id(p) for p in ("/marked", "/native_only", "/short", "/sub/bad")}
    with closing(sqlite3.connect(path)) as conn, conn:
        # A marked scalar (Julia's Symbol/Rational/Date route), a native-only
        # object type (Julia's loader has no method) and a malformed payload.
        conn.execute("INSERT INTO attributes VALUES (?, 'jtype', 'Symbol')", (ids["/marked"],))
        conn.execute("UPDATE objects SET class = 2, type = 13 WHERE id = ?", (ids["/native_only"],))
        conn.execute("UPDATE scalars SET value = X'01' WHERE id = ?", (ids["/short"],))
        conn.execute("UPDATE objects SET class = 4, type = 31 WHERE id = ?", (ids["/sub/bad"],))
    with open_dataecon(path) as db:
        ws, report = db.read_workspace()
        assert [(s.path, s.category, s.subtree) for s in report.skipped] == [
            ("/marked", "unsupported", False),
            ("/native_only", "unsupported", False),
            ("/short", "invalid", False),
            ("/sub/bad", "unsupported", False),
        ]
        assert "reconstruction attributes" in report.skipped[0].reason
        assert "native-only" in report.skipped[1].reason
        assert report.skipped[2].reason.startswith("ValueError: ")
        assert report.count == 4
        assert list(ws.keys()) == ["a", "sub", "z"]
        assert list(ws.sub.keys()) == ["ok"]
        assert ws.a == 1
        assert ws.sub.ok == 1
        assert ws.z == 6
        with pytest.raises(TypeError, match="reconstruction attributes") as info:
            db.read_workspace(strict=True)
        assert info.value.__notes__ == ["Workspace member '/marked'"]
        # A failing member inside a catalog leaves the catalog and its siblings.
        sub, rep = db.read_workspace("/sub")
        assert [s.path for s in rep.skipped] == ["/sub/bad"]
        assert list(sub.keys()) == ["ok"]
        with pytest.raises(TypeError, match="native-only"):
            db.read_object("/native_only")
        with pytest.raises(ValueError, match="catalog"):
            db.read_object("/sub")


@NATIVE
def test_catalog_listing_failure_is_one_skipped_subtree(monkeypatch):
    with open_dataecon_memory() as db:
        db.write_workspace(
            Workspace(a=1, broken=Workspace(x=1, deeper=Workspace(y=2)), fine=Workspace(z=3))
        )
        broken = db.object_id("/broken")
        original = db._children

        def failing(parent_id):
            if parent_id == broken:
                raise DataEconError(-983, "list", db.path, "injected listing failure", "/broken")
            return original(parent_id)

        monkeypatch.setattr(db, "_children", failing)
        ws, report = db.read_workspace()
        assert [(s.path, s.category, s.subtree) for s in report.skipped] == [
            ("/broken", "native", True)
        ]
        assert list(ws.keys()) == ["a", "fine"]
        assert ws.fine.z == 3
        assert report.count == 3  # a, fine, fine/z; the broken catalog does not count
        with pytest.raises(DataEconError, match="injected") as info:
            db.read_workspace(strict=True)
        assert info.value.__notes__ == ["Workspace member '/broken'"]
        # The destination's own listing failure is a plain error in both modes.
        with pytest.raises(DataEconError, match="injected"):
            db.read_workspace("/broken")


@NATIVE
def test_deep_nesting_is_iterative_in_both_directions():
    depth = 1050
    leaf = Workspace(leaf=1)
    chain = leaf
    for _ in range(depth):
        chain = Workspace(d=chain)
    chain.sibling = 2
    limit = sys.getrecursionlimit()
    assert depth > limit
    with open_dataecon_memory() as db:
        report = db.write_workspace(chain)
        assert report.ok
        assert report.count == depth + 2
        assert db.object_info("/" + "/".join(["d"] * depth) + "/leaf").depth == depth + 1
        ws, report = db.read_workspace()
        assert report.ok
        assert report.count == depth + 2
        assert list(ws.keys()) == ["d", "sibling"]
        level = 0
        while "d" in ws:
            ws = ws.d
            level += 1
        assert level == depth
        assert ws.leaf == 1
        assert sys.getrecursionlimit() == limit


@NATIVE
def test_cycles_are_reported_and_shared_children_stored_twice():
    with open_dataecon_memory() as db:
        shared = Workspace(z=1)
        ws = Workspace(p=shared, q=shared, before=1)
        ws.loop = Workspace(inner=Workspace())
        ws.loop.inner.back = ws.loop  # a cycle two levels down
        ws.self = ws
        ws.after = 2
        report = db.write_workspace(ws)
        assert [(s.path, s.category, s.subtree) for s in report.skipped] == [
            ("/loop/inner/back", "cycle", True),
            ("/self", "cycle", True),
        ]
        assert report.count == 8
        assert stored_inventory(db) == {
            "/after": (1, 1),
            "/before": (1, 1),
            "/loop": (0, 0),
            "/loop/inner": (0, 0),
            "/p": (0, 0),
            "/p/z": (1, 1),
            "/q": (0, 0),
            "/q/z": (1, 1),
        }
        assert db.object_id("/p/z") != db.object_id("/q/z")
        loaded = db.read_workspace().workspace
        assert loaded.p is not loaded.q
        assert loaded.p.z == loaded.q.z == 1
        db.new_catalog("/strict")
        with pytest.raises(ValueError, match="cycle") as info:
            db.write_workspace(ws, "/strict", strict=True)
        assert info.value.__notes__ == ["Workspace member '/strict/loop/inner/back'"]
        assert stored_inventory(db, "/strict") == {
            "/before": (1, 1),
            "/loop": (0, 0),
            "/loop/inner": (0, 0),
            "/p": (0, 0),
            "/p/z": (1, 1),
            "/q": (0, 0),
            "/q/z": (1, 1),
        }


@NATIVE
def test_write_object_and_read_object_dispatch():
    with open_dataecon_memory() as db:
        db.new_catalog("/c")
        db.write_object("/c/scalar", np.float32(1.5))
        db.write_object("/c/series", TSeries(mm(2024, 1), np.array([1.0])))
        db.write_object("/c/mv", MVTSeries(qq(2024, 1), ("p",), np.array([[1.0]])))
        db.write_object("/c/vector", np.array([1, 2]))
        db.write_object("/c/text", ["a"])
        db.write_object("/c/range", range(1, 4))
        db.write_object(
            "/c/stored", StoredSeries(mm(2024, 1), DATES, StoredElement.date(Monthly()))
        )
        assert db.read_object("/c/scalar") == np.float32(1.5)
        assert isinstance(db.read_object("/c/series"), TSeries)
        assert isinstance(db.read_object("/c/mv"), MVTSeries)
        np.testing.assert_array_equal(db.read_object("/c/vector"), [1, 2])
        assert db.read_object("/c/text") == ["a"]
        assert db.read_object("/c/range") == range(1, 4)
        assert isinstance(db.read_object("/c/stored"), StoredSeries)
        with pytest.raises(DataEconError) as info:
            db.write_object("/c/scalar", 2.0)
        assert info.value.code == -985
        db.write_object("/c/scalar", 2.0, overwrite=True)
        assert db.read_object("c/scalar") == 2.0
        with pytest.raises(TypeError, match="write_workspace"):
            db.write_object("/c/ws", Workspace(a=1))
        with pytest.raises(TypeError, match="No DataEcon storage class"):
            db.write_object("/c/none", None)
        with pytest.raises(ValueError, match="read_workspace"):
            db.read_object("/c")
        with pytest.raises(ValueError, match="not an object"):
            db.read_object("/")
        with pytest.raises(DataEconError) as info:
            db.read_object("/c/missing")
        assert info.value.code == -989


@NATIVE
def test_save_and_load_workspace_own_the_file(tmp_path):
    path = tmp_path / "saved.daec"
    assert save_workspace(path, Workspace(a=1)) == WorkspaceReport("/", 1, ())
    assert save_workspace(path, Workspace(b=2)).ok
    loaded = load_workspace(path)
    assert isinstance(loaded, LoadedWorkspace)
    assert list(loaded.workspace.keys()) == ["a", "b"]
    assert save_workspace(path, Workspace(c=3), mode="w").ok
    assert list(load_workspace(path).workspace.keys()) == ["c"]
    with open_dataecon(path, "a") as db:
        db.new_catalog("/sub")
    assert save_workspace(path, Workspace(k=1), "/sub", strict=True).ok
    assert load_workspace(path, "/sub").workspace.k == 1
    report = save_workspace(path, Workspace(c=4, bad=object()))
    assert [(s.path, s.category) for s in report.skipped] == [
        ("/c", "exists"),
        ("/bad", "unsupported"),
    ]
    assert save_workspace(path, Workspace(c=4), overwrite=True).ok
    # A strict failure still closes the file (the next open succeeds and sees the residue).
    with pytest.raises(TypeError, match="No DataEcon storage class"):
        save_workspace(path, Workspace(d=5, bad=object(), e=6), strict=True)
    ws, report = load_workspace(path, strict=True)
    assert report.ok
    assert list(ws.keys()) == ["c", "d", "sub"]
    assert ws.c == 4
    assert ws.d == 5
    with pytest.raises(DataEconError) as info:
        load_workspace(path, "/nope")
    assert info.value.code == -989
    with pytest.raises(ValueError, match="mode must be"):
        save_workspace(path, Workspace(), mode="r")
    # Values loaded from a closed file are usable and writable elsewhere.
    ws = load_workspace(path).workspace
    with open_dataecon_memory() as db:
        assert db.write_workspace(ws).ok
        assert db.read_workspace().workspace.sub.k == 1


@NATIVE
def test_results_survive_closure_and_are_writable_after(tmp_path):
    path = tmp_path / "owned.daec"
    with open_dataecon(path, "a") as db:
        db.write_workspace(mixed_workspace())
        ws = db.read_workspace().workspace
    assert_mixed_loaded(ws)
    ws.t3[0, 0, 0] = -1  # writable, owning copies
    ws.ts.values[0] = 9.0
    with open_dataecon(path) as db:
        assert db.read_array("/t3")[0, 0, 0] == 1
        assert db.read_series("/ts").values[0] == 1.0
    other = tmp_path / "other.daec"
    with open_dataecon(other, "a") as db:
        assert db.write_workspace(ws).ok
        assert db.read_array("/t3")[0, 0, 0] == -1
    assert raw_rows(other) == raw_rows(path)


@NATIVE
def test_readonly_and_closed_handles(tmp_path):
    path = tmp_path / "ro.daec"
    with open_dataecon(path, "a") as db:
        db.write_workspace(Workspace(a=1, sub=Workspace(b=2)))
    before = path.read_bytes()
    with open_dataecon(path) as db:
        with pytest.raises(ValueError, match="read-only"):
            db.write_workspace(Workspace(c=3))
        with pytest.raises(ValueError, match="read-only"):
            db.write_workspace(Workspace(c=3), "/sub", overwrite=True, strict=True)
        with pytest.raises(ValueError, match="read-only"):
            db.write_object("/c", 3)
        ws, report = db.read_workspace()
        assert report.ok
        assert ws.a == 1
        assert ws.sub.b == 2
        assert db.read_object("/sub/b") == 2
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="closed"):
        db.read_workspace()
    with pytest.raises(ValueError, match="closed"):
        db.write_workspace(Workspace())
    assert ws.sub.b == 2


NATIVE_FAILURE_PROGRAM = """
import json, sys
from tsecon import Workspace
from tsecon.dataecon import open_dataecon, DataEconError
report = {}
db = open_dataecon(sys.argv[1], "a")
ws = Workspace(first=1, boom=2, sub=Workspace(boom=3, ok=4), last=5)
result = db.write_workspace(ws)
report["skipped"] = [[s.path, s.category, s.subtree, "19" in s.reason] for s in result.skipped]
report["count"] = result.count
report["closed"] = db.closed
report["stored"] = [e.path for e in db.list_objects("/", recursive=True)]
db.new_catalog("/strict")
try:
    db.write_workspace(ws, "/strict", strict=True)
except DataEconError as error:
    report["strict"] = [error.code, error.name, error.__notes__]
report["strict_stored"] = [e.path for e in db.list_objects("/strict", recursive=True)]
# The failed INSERT is the statement's last outcome: the next store echoes
# it once, the one after succeeds, and close then succeeds (design 17).
report["echo"] = []
for attempt in range(2):
    try:
        db.write_scalar("/strict/echo%d" % attempt, attempt)
        report["echo"].append("ok")
    except DataEconError as error:
        report["echo"].append(error.code)
try:
    db.close()
except DataEconError as error:
    report["close"] = error.code
else:
    report["close"] = "ok"
report["closed_after"] = db.closed
print(json.dumps(report))
"""


@NATIVE
def test_native_store_failure_is_reported_without_hiding_owner_state(tmp_path):
    # An injected native failure on one member is reported (default) or raised
    # (strict) with its native code; the owner stays open and usable, and the
    # stale-statement echo of the failed INSERT hits the next store through
    # the same statement, exactly as for a direct write (documented residue).
    path = tmp_path / "native.daec"
    with open_dataecon(path, "a") as db:
        db.write_scalar("keep", 1)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "CREATE TRIGGER fail_boom BEFORE INSERT ON objects WHEN NEW.name = 'boom' "
            "BEGIN SELECT RAISE(ABORT, 'injected store failure'); END"
        )
    result = subprocess.run(
        [sys.executable, "-c", NATIVE_FAILURE_PROGRAM, str(path)],
        check=True,
        timeout=120,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    # /boom fails (19), /sub's catalog creation echoes the stale failure once
    # through the shared insert statement, then /last succeeds.
    assert report["skipped"] == [["/boom", "native", False, True], ["/sub", "native", True, True]]
    assert report["count"] == 2
    assert report["closed"] is False
    assert report["stored"] == ["/first", "/keep", "/last"]
    assert report["strict"][0] == 19
    assert report["strict"][2] == ["Workspace member '/strict/boom'"]
    assert report["strict_stored"] == ["/strict/first"]
    assert report["echo"] == [19, "ok"]
    assert report["close"] == "ok"
    assert report["closed_after"] is True
    with closing(sqlite3.connect(path)) as conn:
        names = sorted(r[0] for r in conn.execute("SELECT name FROM objects WHERE id != 0"))
    assert names == ["echo1", "first", "first", "keep", "last", "strict"]


@NATIVE
def test_julia_workspace_fixture(tmp_path):
    fixture = FIXTURES / "julia_workspace.daec"
    provenance = tomllib.loads(fixture.with_suffix(".toml").read_text(encoding="utf-8"))
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    assert provenance["timeseriesecon_sha"] == "fc0a0d01aed4903ea6c64b12d78d0bdf68468df6"
    expected = mixed_workspace()
    for name in ("vt", "va"):  # Julia's writer cannot store these forms
        del expected[name]
    julia_only = ["date", "r", "sym"]
    with open_dataecon(fixture) as db:
        ws, report = db.read_workspace()
        # Julia's marked scalars (Date, Rational, Symbol) are reported, never dropped silently.
        assert [(s.path, s.category, s.subtree) for s in report.skipped] == [
            (f"/{name}", "unsupported", False) for name in julia_only
        ]
        assert all("reconstruction attributes" in s.reason for s in report.skipped)
        assert report.count == 35
        assert list(ws.keys()) == byte_sorted(expected.keys())
        for name in ("f", "i", "c", "s", "d", "dur"):
            assert ws[name] == expected[name]
            assert type(ws[name]) is type(expected[name])
        assert ws.i8 == np.int8(-3)
        assert ws.u == np.uint64(2**63)
        assert ws.b == np.int8(1)
        for name in ("ts", "tsi", "tsb", "tse"):
            assert isinstance(ws[name], TSeries)
            assert ws[name].firstdate == expected[name].firstdate
            assert ws[name].values.dtype == expected[name].values.dtype
            assert ws[name].values.tolist() == expected[name].values.tolist()
        assert isinstance(ws.tsd, StoredSeries)
        assert ws.tsd.values.tolist() == DATES.tolist()
        assert isinstance(ws.mv, MVTSeries)
        assert list(ws.mv.columns) == ["p", "q"]
        np.testing.assert_array_equal(ws.mat, expected.mat)
        np.testing.assert_array_equal(ws.v, expected.v)
        np.testing.assert_array_equal(ws.t3, TENSOR)
        assert ws.vs == ["a", "b"]
        assert ws.ur == range(1, 6)
        assert list(ws.mr) == list(expected.mr)
        assert list(ws.nested.keys()) == ["deeper", "x"]
        assert ws.nested.deeper["日本語"] == 9
        assert len(ws.empty) == 0
        assert list(ws.order.keys()) == ["10", "9", "B", "_", "a", "b", "ä"]
        assert db.get_attribute("/f", "note") == "user attribute"  # not part of the Workspace
        assert db.get_attributes("/tse") == {"jeltype": "Float64"}  # Julia's own marker
        assert db.get_attributes("/sym") == {"jtype": "Symbol"}
        with pytest.raises(TypeError, match="reconstruction attributes"):
            db.read_workspace(strict=True)
        # Complete stored inventory, catalogs included, straight from the file.
        assert stored_inventory(db) == {
            **{p: ct for p, ct in MIXED_INVENTORY.items() if p not in ("/vt", "/va")},
            "/sym": (1, 6),
            "/r": (1, 4),
            "/date": (1, 4),
        }
    # Python's rewrite of what it loaded stores the same rows (paths, depths,
    # classes, types, marker attributes) except the three marked scalars it cannot
    # write and the user attribute, and Julia's redundant empty-Float64 marker.
    rewritten = tmp_path / "rewritten.daec"
    with open_dataecon(rewritten, "w") as db:
        assert db.write_workspace(ws) == WorkspaceReport("/", 35, ())
    info, attrs = raw_rows(fixture, exclude={"/sym", "/r", "/date"})
    assert raw_rows(rewritten) == (
        info,
        [
            row
            for row in attrs
            if row not in (("/f", "note", "user attribute"), ("/tse", "jeltype", "Float64"))
        ],
    )


def rename_row(path, object_id, name, fullpath):
    # A foreign SQLite writer can store names the native rule never allows.
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("UPDATE objects SET name = ? WHERE id = ?", (name, object_id))
        conn.execute("UPDATE objects_info SET fullpath = ? WHERE id = ?", (fullpath, object_id))


@NATIVE
def test_foreign_aliased_names_are_reported_not_read_as_another_object(tmp_path):
    # The listing returns names as C strings: a foreign name holding NUL is cut
    # short and can alias a sibling; a name holding '/' rebuilds a path into
    # another catalog. Neither may be read under the wrong key or overwrite it.
    path = tmp_path / "alias.daec"
    with open_dataecon(path, "a") as db:
        db.write_workspace(
            Workspace(
                safe=1,
                other=2,
                slashed=3,
                sub=Workspace(x=10, y=11),
                subcat=Workspace(z=12),
                tail=4,
            )
        )
        ids = {p: db.object_id(p) for p in ("/other", "/slashed", "/subcat", "/sub/y")}
    rename_row(path, ids["/other"], "safe\0suffix", "/safe\0suffix")
    rename_row(path, ids["/slashed"], "sub/x", "/sub/x")
    rename_row(path, ids["/subcat"], "sub\0alias", "/sub\0alias")
    with open_dataecon(path) as db:
        # The raw rows really are there and really alias.
        rows = db._children(0)
        assert [r[4] for r in rows] == ["safe", "safe", "sub", "sub", "sub/x", "tail"]
        ws, report = db.read_workspace()
        # The foreign row's fullpath "/sub/x" makes that path ambiguous, so
        # the real nested x fails its identity check too and is reported
        # rather than read from whichever row the exact match happens to find.
        assert [(s.path, s.category, s.subtree) for s in report.skipped] == [
            ("/safe", "invalid", False),
            ("/sub/x", "invalid", False),
            ("/sub", "invalid", True),
            ("/sub/x", "invalid", False),
        ]
        assert "repeats a member" in report.skipped[0].reason
        assert "resolves to object" in report.skipped[1].reason
        assert "repeats a member" in report.skipped[2].reason
        assert "not a valid path component" in report.skipped[3].reason
        assert dict(ws.items()) == {"safe": 1, "sub": ws.sub, "tail": 4}
        assert dict(ws.sub.items()) == {"y": 11}
        assert report.count == 4  # safe, sub, sub/y, tail; nothing aliased is counted
        with pytest.raises(ValueError, match="repeats a member") as info:
            db.read_workspace(strict=True)
        assert info.value.__notes__ == ["Workspace member '/safe'"]
        # The typed readers answer for the path the user names; an ambiguous
        # foreign fullpath returns whichever row the exact match finds.
        assert db.read_scalar("/safe") == 1
        assert db.read_scalar("/sub/y") == 11
        assert db.read_scalar("/sub/x") in (3, 10)


@NATIVE
def test_listed_row_must_resolve_to_itself(tmp_path):
    # A foreign objects_info row can point a name at another object's path;
    # the reconstructed path is then verified against the listed id before
    # any bytes are read, so the member is reported instead of misread.
    path = tmp_path / "identity.daec"
    with open_dataecon(path, "a") as db:
        db.write_workspace(Workspace(a=1, b=2, c=Workspace(k=3)))
        ids = {p: db.object_id(p) for p in ("/a", "/b", "/c")}
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("UPDATE objects_info SET fullpath = '/zz' WHERE id = ?", (ids["/a"],))
        conn.execute("UPDATE objects_info SET fullpath = '/b' WHERE id = ?", (ids["/c"],))
        conn.execute("UPDATE objects_info SET fullpath = '/c' WHERE id = ?", (ids["/b"],))
    with open_dataecon(path) as db:
        ws, report = db.read_workspace()
        assert [(s.path, s.category, s.subtree) for s in report.skipped] == [
            ("/a", "native", False),  # its path no longer exists at all (-989)
            ("/b", "invalid", False),  # resolves to the catalog's id, not the scalar's
            ("/c", "invalid", True),  # resolves to the scalar's id, not the catalog's
        ]
        assert "resolves to object" in report.skipped[1].reason
        assert len(ws) == 0
        assert report.count == 0


@NATIVE
def test_invalid_save_arguments_leave_files_untouched(tmp_path):
    path = tmp_path / "keep.daec"
    with open_dataecon(path, "a") as db:
        db.write_scalar("keep", 42)
    before = path.read_bytes()
    with pytest.raises(TypeError, match="requires a Workspace"):
        save_workspace(path, None, mode="w")
    with pytest.raises(TypeError, match="requires a Workspace"):
        save_workspace(path, {"a": 1}, mode="w")
    with pytest.raises(ValueError, match="Invalid DataEcon path"):
        save_workspace(path, Workspace(a=1), "/bad//path", mode="w")
    with pytest.raises(ValueError, match="mode must be"):
        save_workspace(path, Workspace(a=1), mode="x")
    assert path.read_bytes() == before
    with open_dataecon(path) as db:
        assert db.read_scalar("keep") == 42
    # Nor is a new file created for a refused call.
    fresh = tmp_path / "never.daec"
    with pytest.raises(TypeError):
        save_workspace(fresh, None, mode="w")
    with pytest.raises(ValueError, match="Invalid DataEcon path"):
        save_workspace(fresh, Workspace(), "//", mode="a")
    with pytest.raises(ValueError, match="Invalid DataEcon path"):
        load_workspace(fresh, "//")
    assert not fresh.exists()


@NATIVE
def test_read_order_is_depth_first_pre_order(tmp_path):
    # A nested catalog's members are read before the catalog's later siblings,
    # as Julia recurses, so reports and the first strict error follow that order.
    path = tmp_path / "order.daec"
    with open_dataecon(path, "a") as db:
        db.write_workspace(
            Workspace(a=Workspace(bad=1.0, deep=Workspace(worse=2.0), ok=3), m=4, z=5.0)
        )
        ids = {p: db.object_id(p) for p in ("/a/bad", "/a/deep/worse", "/z")}
    with closing(sqlite3.connect(path)) as conn, conn:
        for oid in ids.values():
            conn.execute("UPDATE scalars SET value = X'01' WHERE id = ?", (oid,))
    with open_dataecon(path) as db:
        ws, report = db.read_workspace()
        assert [s.path for s in report.skipped] == ["/a/bad", "/a/deep/worse", "/z"]
        assert list(ws.keys()) == ["a", "m"]
        assert list(ws.a.keys()) == ["deep", "ok"]
        assert report.count == 4  # a, a/deep, a/ok, m
        with pytest.raises(ValueError) as info:
            db.read_workspace(strict=True)
        assert info.value.__notes__ == ["Workspace member '/a/bad'"]
