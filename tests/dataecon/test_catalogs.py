"""Catalogs, nested paths, object ids, listing and attributes.

Mirrors Julia's ``new_catalog``, ``find_fullpath``/``find_object``,
``get_fullpath``, ``catalog_size``, ``list_catalog``, ``delete_object`` and the
attribute functions through path-addressed Python methods plus the explicit
id conversions. Destructive operations run on fresh temporary files, fresh
in-memory databases and copies of the Julia fixture.
"""

import hashlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
import tomllib
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

import tsecon.dataecon as de
from tsecon import MIT, MVTSeries, TSeries, mm, qq
from tsecon.dataecon import (
    DataEconError,
    ObjectInfo,
    StoredArray,
    StoredElement,
    StoredSeries,
    open_dataecon,
    open_dataecon_memory,
)
from tsecon.frequencies import Monthly

FIXTURES = Path(__file__).parent / "fixtures"
NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built",
)
SERIES = TSeries(mm(2024, 1), np.array([1.0, 2.0, 3.0]))
MATRIX = MVTSeries(qq(2024, 1), ("p", "q"), np.array([[1.0, 2.0], [3.0, 4.0]]))
TENSOR = np.arange(1, 25, dtype=np.int64).reshape((2, 3, 4), order="F")
DELIMITERS = ("\x1f", "\x1e\x1f", "\x1e" + "\x1f" * 5, "\x1e" + "\x1f" * 300, "\x1d\x1c", "‖")


# ---------------------------------------------------------------- no native needed


@pytest.mark.parametrize(
    ("path", "components"),
    [
        ("/", []),
        ("", []),
        ("x", ["x"]),
        ("/x", ["x"]),
        ("a/b/c", ["a", "b", "c"]),
        ("/a/b/c", ["a", "b", "c"]),
        ("/a/./b", ["a", ".", "b"]),
        ("/a/../b", ["a", "..", "b"]),
        ("/a\\b", ["a\\b"]),
        ("/ a /b ", [" a ", "b "]),
        ("/日本語/été/🙂", ["日本語", "été", "🙂"]),
    ],
)
def test_path_grammar(path, components):
    assert de._split_path(path) == components
    assert de._normalize(path) == "/" + "/".join(components)


@pytest.mark.parametrize(
    "path", ["//", "/a//b", "a//b", "/a/", "a/", "//a", "/a/ /b", "/a/\t", "/a\0b", "/\0"]
)
def test_invalid_path_syntax(path):
    with pytest.raises(ValueError, match="Invalid DataEcon path"):
        de._split_path(path)


def test_root_is_not_an_object_path():
    for root in ("/", ""):
        with pytest.raises(ValueError, match="root catalog"):
            de._object_path(root)
        with pytest.raises(ValueError, match="root catalog"):
            de._parent_and_leaf(root)
    assert de._parent_and_leaf("x") == ("/", "x")
    assert de._parent_and_leaf("/a/b/c") == ("/a/b", "c")
    with pytest.raises(TypeError):
        de._split_path(b"/a")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("record", "kind"),
    [
        ((1, 0, 0, 0, "c", "/c", 1, 0), "catalog"),
        ((1, 0, 1, 4, "s", "/s", 1, 0), "scalar"),
        ((1, 0, 2, 12, "t", "/t", 1, 0), "series"),
        ((1, 0, 3, 21, "m", "/m", 1, 0), "series"),
        ((1, 0, 2, 10, "v", "/v", 1, 0), "array"),
        ((1, 0, 2, 11, "r", "/r", 1, 0), "array"),
        ((1, 0, 3, 20, "x", "/x", 1, 0), "array"),
        ((1, 0, 4, 30, "n", "/n", 1, 0), "array"),
        ((1, 0, 2, 13, "o", "/o", 1, 0), "unknown"),
        ((1, 0, 4, 31, "d", "/d", 1, 0), "unknown"),
    ],
)
def test_object_info_kinds(record, kind):
    info = de._info(record)
    assert isinstance(info, ObjectInfo)
    assert (info.id, info.parent_id, info.name, info.path, info.depth) == (
        1,
        0,
        record[4],
        record[5],
        1,
    )
    assert info.kind == kind
    assert (info.object_class, info.object_type) == record[2:4]
    with pytest.raises(AttributeError):
        info.kind = "x"  # type: ignore[misc]


def test_reserved_attribute_names_and_validation_precede_native(monkeypatch):
    calls = []

    class Handle:
        def close(self):
            calls.append("close")

    monkeypatch.setattr(de, "_open_native", lambda *_, **__: Handle())
    db = open_dataecon("example.daec", "a")
    for key in ("jtype", "jeltype", "DE_VERSION"):
        with pytest.raises(ValueError, match="reserved"):
            db.set_attribute("/x", key, "value")
    with pytest.raises(ValueError, match="NUL"):
        db.set_attribute("/x", "k\0", "value")
    with pytest.raises(ValueError, match="NUL"):
        db.set_attribute("/x", "k", "v\0")
    with pytest.raises(TypeError):
        db.set_attribute("/x", "k", 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        db.get_attribute("/x", 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Invalid DataEcon path"):
        db.set_attribute("/x//y", "k", "v")
    with pytest.raises(ValueError, match="root catalog always exists"):
        db.new_catalog("/")
    with pytest.raises(ValueError, match="max_depth"):
        db.list_objects("/", recursive=True, max_depth=-1)
    db.close()
    assert calls == ["close"]


def test_readonly_owner_refuses_catalog_writes_before_native(monkeypatch):
    calls = []

    class Handle:
        def close(self):
            calls.append("close")

        def describe_path(self, path):
            calls.append(("describe", path))
            return (1, 0, 1, 4, "x", "/x", 1, 0)

    monkeypatch.setattr(de, "_open_native", lambda *_, **__: Handle())
    db = open_dataecon("example.daec")
    with pytest.raises(ValueError, match="read-only"):
        db.new_catalog("/c")
    with pytest.raises(ValueError, match="read-only"):
        db.new_catalog("/c/d", parents=True)
    with pytest.raises(ValueError, match="read-only"):
        db.set_attribute("/x", "k", "v")
    with pytest.raises(ValueError, match="read-only"):
        db.delete("/c/x", recursive=True)
    with pytest.raises(ValueError, match="read-only"):
        db.write_scalar("/c/x", 1.0, overwrite=True)
    assert db.object_info("/x").kind == "scalar"
    db.close()
    assert calls == [("describe", "/x"), "close"]


# ---------------------------------------------------------------- native


@NATIVE
def test_nested_round_trips_every_object_family():
    with open_dataecon_memory() as db:
        db.new_catalog("/inputs")
        db.new_catalog("/inputs/raw")
        db.new_catalog("inputs/raw/日本語")
        db.write_scalar("/inputs/raw/scalar", 1.5)
        db.write_scalar("inputs/raw/text", "vintage ‖ 3")
        db.write_array("/inputs/raw/vector", np.array([1, 2, 3], dtype=np.int32))
        db.write_array("/inputs/raw/matrix", np.array([[1.0, 2.0], [3.0, 4.0]]))
        db.write_array("/inputs/raw/tensor", TENSOR)
        db.write_array("/inputs/raw/text_vector", ["a", "b‖c"])
        db.write_series("/inputs/raw/series", SERIES)
        db.write_series("/inputs/raw/mvtseries", MATRIX)
        db.write_series(
            "/inputs/raw/日本語/dates",
            StoredSeries(
                mm(2024, 1), np.array([24288, 24289], dtype="<i8"), StoredElement.date(Monthly())
            ),
        )
        db.write_array(
            "/inputs/raw/日本語/date_array",
            StoredArray(np.array([[1, 2], [3, 4]], dtype="<i8"), StoredElement.date(Monthly())),
        )
        assert db.read_scalar("inputs/raw/scalar") == 1.5
        assert db.read_scalar("/inputs/raw/text") == "vintage ‖ 3"
        np.testing.assert_array_equal(db.read_array("/inputs/raw/vector"), [1, 2, 3])
        assert db.read_array("/inputs/raw/vector").dtype == np.int32
        np.testing.assert_array_equal(db.read_array("/inputs/raw/matrix"), [[1.0, 2.0], [3.0, 4.0]])
        tensor = db.read_array("/inputs/raw/tensor")
        np.testing.assert_array_equal(tensor, TENSOR)
        assert tensor.flags.owndata
        assert tensor.flags.writeable
        assert db.read_array("/inputs/raw/text_vector") == ["a", "b‖c"]
        series = db.read_series("/inputs/raw/series")
        assert series.firstdate == mm(2024, 1)
        np.testing.assert_array_equal(series.values, SERIES.values)
        matrix = db.read_series("inputs/raw/mvtseries")
        assert isinstance(matrix, MVTSeries)
        assert list(matrix.columns) == ["p", "q"]
        dates = db.read_series("/inputs/raw/日本語/dates")
        assert isinstance(dates, StoredSeries)
        assert dates.tolist() == [MIT(Monthly(), 24288), MIT(Monthly(), 24289)]
        array = db.read_array("/inputs/raw/日本語/date_array")
        assert isinstance(array, StoredArray)
        assert array.shape == (2, 2)
        assert [e.name for e in db.list_objects("/inputs/raw")] == [
            "matrix",
            "mvtseries",
            "scalar",
            "series",
            "tensor",
            "text",
            "text_vector",
            "vector",
            "日本語",
        ]
        assert db.catalog_size("/inputs/raw") == 9
        assert db.catalog_size("/inputs/raw/日本語") == 2
        assert db.catalog_size() == 1


@NATIVE
def test_missing_parent_and_non_catalog_parent_are_refused_without_residue():
    with open_dataecon_memory() as db:
        db.write_scalar("s", 1)
        for writer, value in (
            (db.write_scalar, 1.0),
            (db.write_series, SERIES),
            (db.write_series, MATRIX),
            (db.write_array, TENSOR),
            (db.write_array, np.zeros(2)),
            (db.write_array, np.zeros((2, 2))),
        ):
            with pytest.raises(DataEconError) as caught:
                writer("/nope/deeper/x", value)
            assert caught.value.code == -989
            assert caught.value.name == "/nope/deeper"
            with pytest.raises(ValueError, match="not a catalog"):
                writer("/s/child", value)
            with pytest.raises(ValueError, match="root catalog"):
                writer("/", value)
        with pytest.raises(DataEconError) as caught:
            db.new_catalog("/nope/child")
        assert caught.value.code == -989
        with pytest.raises(ValueError, match="not a catalog"):
            db.new_catalog("/s/child")
        assert [e.path for e in db.list_objects("/", recursive=True)] == ["/s"]
        assert not db.exists("/nope")
        assert db.catalog_size() == 1


@NATIVE
def test_existing_names_and_nested_overwrite():
    with open_dataecon_memory() as db:
        db.new_catalog("/c")
        db.write_scalar("/c/x", 1)
        db.set_attribute("/c/x", "note", "old")
        first = db.object_id("/c/x")
        with pytest.raises(DataEconError) as caught:
            db.write_scalar("/c/x", 2)
        assert caught.value.code == -985
        assert caught.value.name == "x"
        with pytest.raises(DataEconError) as caught:
            db.new_catalog("/c/x")
        assert caught.value.code == -985
        with pytest.raises(DataEconError) as caught:
            db.new_catalog("/c")
        assert caught.value.code == -985
        db.new_catalog("/c", exist_ok=True)
        with pytest.raises(DataEconError) as caught:
            db.new_catalog("/c/x", exist_ok=True)  # exists but is not a catalog
        assert caught.value.code == -985
        db.write_series("/c/x", SERIES, overwrite=True)
        assert db.object_id("/c/x") > first
        assert db.get_attributes("/c/x") == {}  # attributes went with the old object
        assert db.object_info("/c/x").kind == "series"
        with pytest.raises(ValueError, match="catalog"):
            db.write_scalar("/c", 1, overwrite=True)
        assert db.object_info("/c").kind == "catalog"
        # A rejected value never deletes the existing nested object.
        with pytest.raises(TypeError):
            db.write_scalar("/c/x", object(), overwrite=True)  # type: ignore[arg-type]
        assert db.object_info("/c/x").kind == "series"


@NATIVE
def test_new_catalog_parents_and_exist_ok():
    with open_dataecon_memory() as db:
        with pytest.raises(DataEconError) as caught:
            db.new_catalog("/a/b/c")
        assert caught.value.code == -989
        assert not db.exists("/a")
        db.new_catalog("/a/b/c", parents=True)
        assert [e.path for e in db.list_objects("/", recursive=True)] == ["/a", "/a/b", "/a/b/c"]
        db.new_catalog("/a/b/c", parents=True, exist_ok=True)
        db.new_catalog("a/b/d", parents=True)
        assert db.catalog_size("/a/b") == 2
        db.write_scalar("/a/s", 1)
        with pytest.raises(ValueError, match="not a catalog"):
            db.new_catalog("/a/s/t/u", parents=True)
        assert not db.exists("/a/s/t")
        with pytest.raises(DataEconError) as caught:
            db.new_catalog("/a/s", parents=True, exist_ok=True)
        assert caught.value.code == -985
        assert db.exists("/")
        assert db.exists("")
        assert db.exists("a/b")
        assert not db.exists("/a/b/z")


@NATIVE
def test_listing_order_recursion_and_depth():
    with open_dataecon_memory() as db:
        db.new_catalog("/cat")
        for name in ("b", "B", "a", "1", " sp", "_u", "ä", "日本語", "🙂", "Z"):
            db.write_scalar(f"/cat/{name}", 1)
        db.new_catalog("/cat/sub")
        db.new_catalog("/cat/sub/deep")
        db.write_series("/cat/sub/deep/x", SERIES)
        db.write_array("/cat/sub/m", np.zeros((1, 2)))
        db.write_scalar("/top", 2)
        names = [e.name for e in db.list_objects("/cat")]
        assert names == sorted(names, key=lambda n: n.encode("utf-8"))
        assert names == [" sp", "1", "B", "Z", "_u", "a", "b", "sub", "ä", "日本語", "🙂"]
        assert [e.kind for e in db.list_objects("/")] == ["catalog", "scalar"]
        full = db.list_objects("/", recursive=True)
        paths = [e.path for e in full]
        assert paths == [
            "/cat",
            "/cat/ sp",
            "/cat/1",
            "/cat/B",
            "/cat/Z",
            "/cat/_u",
            "/cat/a",
            "/cat/b",
            "/cat/sub",
            "/cat/sub/deep",
            "/cat/sub/deep/x",
            "/cat/sub/m",
            "/cat/ä",
            "/cat/日本語",
            "/cat/🙂",
            "/top",
        ]
        assert [e.depth for e in full][:2] == [1, 2]
        assert {e.path: e.depth for e in full}["/cat/sub/deep/x"] == 4
        assert {e.path: e.kind for e in full}["/cat/sub/m"] == "array"
        assert all(e.parent_id == db.object_id(e.path.rpartition("/")[0] or "/") for e in full)
        assert [e.path for e in db.list_objects("/cat/sub", recursive=True, max_depth=1)] == [
            "/cat/sub/deep",
            "/cat/sub/m",
        ]
        assert [e.path for e in db.list_objects("/cat/sub", recursive=True, max_depth=2)] == [
            "/cat/sub/deep",
            "/cat/sub/deep/x",
            "/cat/sub/m",
        ]
        assert db.list_objects("/cat/sub", max_depth=0) == []
        assert db.list_objects("/cat/sub", recursive=True, max_depth=0) == []
        assert db.list_objects("/cat/sub") == db.list_objects("/cat/sub", max_depth=5)
        assert db.list_objects("/cat/sub/deep") == [
            ObjectInfo(
                db.object_id("/cat/sub/deep/x"),
                db.object_id("/cat/sub/deep"),
                "/cat/sub/deep/x",
                "x",
                "series",
                2,
                12,
                4,
            )
        ]
        with pytest.raises(ValueError, match="not a catalog"):
            db.list_objects("/top")
        with pytest.raises(ValueError, match="not a catalog"):
            db.catalog_size("/top")
        with pytest.raises(DataEconError) as caught:
            db.list_objects("/missing")
        assert caught.value.code == -989
        with pytest.raises(DataEconError) as caught:
            db.catalog_size("/missing")
        assert caught.value.code == -989
        assert db.catalog_size("/cat") == 11
        assert db.catalog_size("/") == 2


@NATIVE
def test_object_ids_and_paths():
    with open_dataecon_memory() as db:
        assert db.object_id("/") == 0
        assert db.object_id("") == 0
        assert db.object_path(0) == "/"
        root = db.object_info("/")
        assert (root.id, root.parent_id, root.path, root.name, root.kind, root.depth) == (
            0,
            0,
            "/",
            "/",
            "catalog",
            0,
        )
        db.new_catalog("/c")
        db.write_scalar("/c/x", 1)
        info = db.object_info("c/x")
        assert info == db.object_info_by_id(info.id)
        assert info.parent_id == db.object_id("/c")
        assert info.path == "/c/x"
        assert info.depth == 2
        assert db.object_path(info.id) == "/c/x"
        for bad in (-1, -(2**63)):
            with pytest.raises(ValueError, match="nonnegative"):
                db.object_info_by_id(bad)
        with pytest.raises(TypeError):
            db.object_info_by_id(True)
        with pytest.raises(TypeError):
            db.object_info_by_id("1")  # type: ignore[arg-type]
        with pytest.raises(DataEconError) as caught:
            db.object_info_by_id(987654)
        assert caught.value.code == -989
        with pytest.raises(DataEconError) as caught:
            db.object_id("/c/missing")
        assert caught.value.code == -989
        # Observed native facts, not promises: overwrite assigns a new id, a
        # deleted id stays dead, truncation restarts the sequence.
        old = info.id
        db.write_scalar("/c/x", 2, overwrite=True)
        assert db.object_id("/c/x") > old
        with pytest.raises(DataEconError) as caught:
            db.object_info_by_id(old)
        assert caught.value.code == -989
        db.delete("/c/x")
        with pytest.raises(DataEconError):
            db.object_path(old + 1)
        db.truncate()
        db.new_catalog("/again")
        assert db.object_id("/again") == 1


@NATIVE
def test_nested_delete_and_root_protection(tmp_path):
    path = tmp_path / "delete.daec"
    with open_dataecon(path, "a") as db:
        db.new_catalog("/a/b", parents=True)
        db.write_scalar("/a/b/x", 1)
        db.write_scalar("/a/y", 2)
        db.set_attribute("/a/b", "k", "v")
        with pytest.raises(ValueError, match="recursive=True"):
            db.delete("/a/b")
        with pytest.raises(ValueError, match="recursive=True"):
            db.delete("a")
        for root in ("/", ""):
            with pytest.raises(ValueError, match="root catalog"):
                db.delete(root, recursive=True)
        with pytest.raises(DataEconError) as caught:
            db.delete("/a/b/missing")
        assert caught.value.code == -989
        assert caught.value.name == "/a/b/missing"
        db.delete("/a/b/x")
        assert db.catalog_size("/a/b") == 0
        db.delete("/a/b", recursive=True)  # an empty catalog still needs the flag
        assert not db.exists("/a/b")
        assert db.exists("/a/y")
        db.write_scalar("/a/b", 3)  # the name is free again, as a scalar
        db.delete("/a", recursive=True)
        assert db.is_empty()
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM objects WHERE id != 0").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM attributes WHERE id != 0").fetchone()[0] == 0


@NATIVE
def test_attributes_round_trip_losslessly():
    with open_dataecon_memory() as db:
        db.new_catalog("/c")
        db.write_series("/c/s", SERIES)
        assert db.get_attributes("/c/s") == {}
        assert db.get_attribute("/c/s", "missing") is None
        db.set_attribute("/c/s", "note", "first")
        db.set_attribute("/c/s", "note", "second")
        db.set_attribute("/c/s", "empty", "")
        db.set_attribute("/c/s", "", "empty name")
        db.set_attribute("/c/s", "unicodé/名 ", "välue 🙂\nline\ttab")
        db.set_attribute("/c/s", "long", "x" * 100_000)
        for delimiter in DELIMITERS:
            db.set_attribute("/c/s", f"value{delimiter}with", f"a{delimiter}b‖c")
        expected = {
            "note": "second",
            "empty": "",
            "": "empty name",
            "unicodé/名 ": "välue 🙂\nline\ttab",
            "long": "x" * 100_000,
            **{f"value{d}with": f"a{d}b‖c" for d in DELIMITERS},
        }
        assert db.get_attributes("/c/s") == expected
        assert db.get_attribute("/c/s", "") == "empty name"
        assert db.get_attribute("/c/s", "note") == "second"
        # Catalogs and the root carry attributes too; DE_VERSION is readable.
        db.set_attribute("/c", "owner", "me")
        db.set_attribute("/", "root_note", "hello")
        assert db.get_attributes("/c") == {"owner": "me"}
        assert db.get_attributes("/") == {"DE_VERSION": "0.4.0", "root_note": "hello"}
        assert db.get_attribute("", "DE_VERSION") == "0.4.0"
        with pytest.raises(DataEconError) as caught:
            db.get_attribute("/c/missing", "k")
        assert caught.value.code == -989
        with pytest.raises(DataEconError) as caught:
            db.set_attribute("/c/missing", "k", "v")
        assert caught.value.code == -989
        with pytest.raises(DataEconError) as caught:
            db.get_attributes("/c/missing")
        assert caught.value.code == -989
        # Attributes survive further writes and are copied out of native memory.
        first = db.get_attributes("/c/s")
        db.set_attribute("/c/s", "note", "third")
        assert first["note"] == "second"


@NATIVE
def test_attribute_names_made_of_delimiters_are_still_enumerated():
    with open_dataecon_memory() as db:
        db.write_scalar("x", 1)
        names = {
            "\x1f",
            "\x1f" * 2,
            "\x1f" * 3,
            "\x1f" * 1000,
            "a\x1fb",
            "\x1fc",
            "d\x1f",
            "d\x1e\x1f",
            "\x1e\x1f\x1f\x1e",
            "\x1d\x1c",
            "\x1e",
            "\x1e" * 4,
            "",
        }
        for index, name in enumerate(sorted(names)):
            db.set_attribute("/x", name, str(index))
        assert db.get_attributes("/x") == {n: str(i) for i, n in enumerate(sorted(names))}
        assert db.get_attribute("/x", "\x1f" * 1000) == str(sorted(names).index("\x1f" * 1000))


@NATIVE
def test_damaged_attribute_name_is_refused_not_guessed(tmp_path):
    # A name holding NUL can only come from a foreign SQL writer (the C API
    # truncates it); the joined C string then loses pieces and no delimiter
    # can restore them, so enumeration fails instead of returning a wrong map.
    path = tmp_path / "damaged.daec"
    with open_dataecon(path, "a") as db:
        db.write_scalar("x", 1)
        db.set_attribute("/x", "fine", "1")
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("INSERT INTO attributes VALUES (1, 'b' || char(0) || 'c', '2')")
    with open_dataecon(path) as db:
        with pytest.raises(ValueError, match="cannot be enumerated"):
            db.get_attributes("/x")
        assert db.get_attribute("/x", "fine") == "1"


@NATIVE
def test_reserved_markers_are_readable_and_protected():
    with open_dataecon_memory() as db:
        db.write_series("b", TSeries(mm(2024, 1), np.array([True, False])))
        db.write_array(
            "bits",
            StoredArray(
                np.array([1, 0], dtype=np.int8),
                StoredElement.numeric(np.dtype("i1"), "Bool"),
                object_marker="BitVector",
            ),
        )
        assert db.get_attribute("/b", "jeltype") == "Bool"
        assert db.get_attributes("/bits") == {"jeltype": "Bool", "jtype": "BitVector"}
        for key in ("jtype", "jeltype", "DE_VERSION"):
            with pytest.raises(ValueError, match="reserved"):
                db.set_attribute("/b", key, "Float64")
            with pytest.raises(ValueError, match="reserved"):
                db.set_attribute("/", key, "9.9.9")
        assert db.get_attribute("/b", "jeltype") == "Bool"
        assert db.get_attributes("/") == {"DE_VERSION": "0.4.0"}
        db.set_attribute("/b", "source", "survey")
        assert db.get_attributes("/b") == {"jeltype": "Bool", "source": "survey"}
        np.testing.assert_array_equal(db.read_series("/b").values, [True, False])


@NATIVE
def test_null_attribute_value_is_refused_not_substituted(tmp_path):
    path = tmp_path / "null.daec"
    with open_dataecon(path, "a") as db:
        db.write_scalar("x", 1)
        db.set_attribute("/x", "k", "v")
        db.set_attribute("/x", "other", "o")
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("UPDATE attributes SET value = NULL WHERE name = 'k'")
    with open_dataecon(path) as db:
        with pytest.raises(ValueError, match="SQL NULL"):
            db.get_attribute("/x", "k")
        with pytest.raises(ValueError, match="SQL NULL"):
            db.get_attributes("/x")
        assert db.get_attribute("/x", "other") == "o"
        assert db.read_scalar("/x") == 1


@NATIVE
def test_readonly_file_reads_catalogs_and_refuses_writes(tmp_path):
    path = tmp_path / "ro.daec"
    with open_dataecon(path, "a") as db:
        db.new_catalog("/c")
        db.write_scalar("/c/x", 1)
        db.set_attribute("/c/x", "k", "v")
    before = path.read_bytes()
    with open_dataecon(path) as db:
        assert db.read_scalar("/c/x") == 1
        assert db.get_attributes("/c/x") == {"k": "v"}
        assert [e.path for e in db.list_objects("/", recursive=True)] == ["/c", "/c/x"]
        assert db.catalog_size("/c") == 1
        assert db.exists("/c/x")
        assert db.object_path(db.object_id("/c/x")) == "/c/x"
        for call in (
            lambda: db.new_catalog("/d"),
            lambda: db.new_catalog("/d/e", parents=True),
            lambda: db.set_attribute("/c/x", "k", "w"),
            lambda: db.delete("/c/x"),
            lambda: db.delete("/c", recursive=True),
            lambda: db.write_scalar("/c/x", 2, overwrite=True),
            lambda: db.write_scalar("/c/y", 2),
        ):
            with pytest.raises(ValueError, match="read-only"):
                call()
        assert db.get_attributes("/c/x") == {"k": "v"}
    assert path.read_bytes() == before


@NATIVE
def test_results_survive_closure_and_are_owned():
    db = open_dataecon_memory()
    db.new_catalog("/c")
    db.write_scalar("/c/x", 1)
    db.set_attribute("/c/x", "k", "v")
    listing = db.list_objects("/", recursive=True)
    attributes = db.get_attributes("/c/x")
    info = db.object_info("/c/x")
    db.close()
    assert [e.path for e in listing] == ["/c", "/c/x"]
    assert attributes == {"k": "v"}
    assert info.path == "/c/x"
    for call in (
        lambda: db.list_objects(),
        lambda: db.get_attributes("/c/x"),
        lambda: db.object_info("/c/x"),
        lambda: db.exists("/c"),
        lambda: db.catalog_size(),
    ):
        with pytest.raises(ValueError, match="closed"):
            call()


@NATIVE
def test_windows_style_separators_and_dots_are_ordinary_names():
    with open_dataecon_memory() as db:
        db.new_catalog("/cat")
        db.write_scalar("cat\\x", 1)  # one root object named 'cat\\x', unlike Julia on Windows
        db.write_scalar("/cat/.", 2)
        db.write_scalar("/cat/..", 3)
        db.write_scalar("/cat/ padded ", 4)
        assert db.read_scalar("cat\\x") == 1
        assert db.exists("/cat\\x")
        assert db.catalog_size("/cat") == 3
        assert db.read_scalar("/cat/.") == 2
        assert db.read_scalar("/cat/..") == 3
        assert db.read_scalar("/cat/ padded ") == 4
        with pytest.raises(DataEconError) as caught:
            db.read_scalar("/cat/x")
        assert caught.value.code == -989
        assert sorted(e.name for e in db.list_objects("/")) == ["cat", "cat\\x"]


@NATIVE
def test_reopen_keeps_paths_ids_and_attributes(tmp_path):
    path = tmp_path / "reopen.daec"
    with open_dataecon(path, "a") as db:
        db.new_catalog("/a/b", parents=True)
        db.write_series("/a/b/s", SERIES)
        db.set_attribute("/a/b/s", "k", "v")
        ids = {e.path: e.id for e in db.list_objects("/", recursive=True)}
    with open_dataecon(path) as db:
        assert {e.path: e.id for e in db.list_objects("/", recursive=True)} == ids
        assert db.get_attributes("/a/b/s") == {"k": "v"}
        assert db.object_path(ids["/a/b/s"]) == "/a/b/s"
    with open_dataecon(path, "a") as db:
        db.write_scalar("/a/new", 1)
        assert db.object_id("/a/new") > max(ids.values())
    with closing(sqlite3.connect(path)) as conn:
        rows = dict(conn.execute("SELECT fullpath, depth FROM objects_info").fetchall())
        assert rows == {"": 0, "/a": 1, "/a/b": 2, "/a/b/s": 3, "/a/new": 2}


RESIDUE_PROGRAM = """
import json, sys
from tsecon.dataecon import open_dataecon, DataEconError
report = {}
db = open_dataecon(sys.argv[1], "a")
try:
    db.new_catalog("/p/q/boom", parents=True)
except DataEconError as error:
    report["new_catalog"] = [error.code, error.operation, error.name]
report["chain"] = [e.path for e in db.list_objects("/", recursive=True)]
# The failed INSERT leaves the cached statement in an error state: the next
# call through the same statement reports the stale code without executing.
for attempt in range(3):
    try:
        db.new_catalog("/p/after%d" % attempt)
        report["after%d" % attempt] = "ok"
    except DataEconError as error:
        report["after%d" % attempt] = error.code
report["exists"] = [db.exists("/p/after%d" % i) for i in range(3)]
try:
    db.close()
except DataEconError as error:
    report["close"] = error.code
else:
    report["close"] = "ok"
report["closed"] = db.closed
print(json.dumps(report))
"""


@NATIVE
def test_parents_failure_leaves_partial_chain_and_echoes_once(tmp_path):
    # A native failure while creating the last catalog leaves the catalogs
    # created before it (no rollback is promised), and the native library then
    # reports the same error once more on the next use of that statement. Both
    # are documented residue; the close afterwards fails and quarantines.
    path = tmp_path / "residue.daec"
    with open_dataecon(path, "a") as db:
        db.write_scalar("keep", 1)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute(
            "CREATE TRIGGER fail_boom BEFORE INSERT ON objects WHEN NEW.name = 'boom' "
            "BEGIN SELECT RAISE(ABORT, 'injected catalog failure'); END"
        )
    result = subprocess.run(
        [sys.executable, "-c", RESIDUE_PROGRAM, str(path)],
        check=True,
        timeout=120,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["new_catalog"] == [19, "new_catalog", "boom"]
    assert report["chain"] == ["/keep", "/p", "/p/q"]
    assert report["after0"] == 19
    assert report["after1"] == "ok"
    assert report["after2"] == "ok"
    assert report["exists"] == [False, True, True]
    # The successful reuse reset the statement, so the close no longer reports
    # the failure (the failed-close rule needs the error to be the statement's
    # last outcome) and the writes are committed.
    assert report["close"] == "ok"
    assert report["closed"] is True
    with closing(sqlite3.connect(path)) as conn:
        names = sorted(r[0] for r in conn.execute("SELECT name FROM objects WHERE id != 0"))
    assert names == ["after1", "after2", "keep", "p", "q"]


@NATIVE
def test_random_names_over_the_delimiter_alphabet_are_enumerated_exactly():
    # Names built only from the delimiter bytes (plus one letter) attack the
    # split: trailing runs, embedded delimiters and empty names all appear.
    rng = np.random.default_rng(20260914)
    alphabet = ["\x1c", "\x1d", "\x1e", "\x1f", "a"]
    with open_dataecon_memory() as db:
        for trial in range(25):
            db.write_scalar(f"t{trial}", 1)
            names = {
                "".join(rng.choice(alphabet, size=int(rng.integers(0, 7))))
                for _ in range(int(rng.integers(1, 12)))
            }
            expected = {name: str(index) for index, name in enumerate(sorted(names))}
            for name, value in expected.items():
                db.set_attribute(f"/t{trial}", name, value)
            assert db.get_attributes(f"/t{trial}") == expected, trial


# ---------------------------------------------------------------- Julia fixture

CATALOG_ORDER_NAMES = ["b", "B", "a", "1", " sp", "_u", "ä", "Z"]
CATALOG_ATTRIBUTES = {
    "note": "second",
    "empty": "",
    "": "empty name",
    "unicodé/名 ": "välue \U0001f642\nline\ttab",
    "delim": "a‖b",
    "unit": "a\x1fb",
    "k‖3": "v3",
    "run\x1e\x1f\x1f": "r",
    "long": "x" * 5000,
}
CATALOG_DATES = [24288, 24289]


def rewrite_catalog_tree(db, prefix=""):
    """Write the Julia fixture's tree through Python (no child under a scalar)."""
    db.new_catalog(f"{prefix}/cat/sub/deep", parents=True)
    db.new_catalog(f"{prefix}/cat/日本語")
    db.write_scalar(f"{prefix}/cat/scalar", 1.5)
    db.write_scalar(f"{prefix}/cat/sub/text", "vintage ‖ 3")
    db.write_array(f"{prefix}/cat/sub/vector", np.array([1, 2, 3], dtype=np.int32))
    db.write_array(f"{prefix}/cat/sub/matrix", np.array([[1.0, 2.0], [3.0, 4.0]]))
    db.write_array(f"{prefix}/cat/sub/deep/tensor", TENSOR)
    db.write_series(f"{prefix}/cat/sub/deep/series", SERIES)
    db.write_series(f"{prefix}/cat/sub/mvtseries", MATRIX)
    db.write_series(f"{prefix}/cat/sub/bools", TSeries(mm(2024, 1), np.array([True, False])))
    db.write_array(f"{prefix}/cat/sub/text_vector", ["a", "b"])
    codes = np.array(CATALOG_DATES, dtype="<i8")
    db.write_series(
        f"{prefix}/cat/日本語/dates",
        StoredSeries(mm(2024, 1), codes, StoredElement.date(Monthly())),
    )
    db.write_array(
        f"{prefix}/cat/日本語/date_array",
        StoredArray(codes.reshape((1, 2)), StoredElement.date(Monthly())),
    )
    for name in CATALOG_ORDER_NAMES:
        db.write_scalar(f"{prefix}/cat/{name}", 1)
    db.write_scalar(f"{prefix}/cat/scalar_parent", 2)
    db.set_attribute(f"{prefix}/cat/scalar", "note", "first")
    for name, value in CATALOG_ATTRIBUTES.items():
        db.set_attribute(f"{prefix}/cat/scalar", name, value)
    db.set_attribute(f"{prefix}/cat", "owner", "julia")
    db.set_attribute("/", "root_note", "hello")


def raw_tree(path, exclude=()):
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


@NATIVE
def test_julia_catalogs_fixture(tmp_path):
    fixture = FIXTURES / "julia_catalogs.daec"
    provenance = tomllib.loads(fixture.with_suffix(".toml").read_text(encoding="utf-8"))
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    assert provenance["timeseriesecon_sha"] == "fc0a0d01aed4903ea6c64b12d78d0bdf68468df6"
    assert provenance["layout"]["object_t"] == {"size": 32, "offsets": [0, 8, 16, 20, 24]}
    with open_dataecon(fixture) as db:
        assert db.catalog_size("/cat") == 12
        assert db.catalog_size() == 1
        listed = db.list_objects("/cat")
        assert [e.name for e in listed] == [
            " sp",
            "1",
            "B",
            "Z",
            "_u",
            "a",
            "b",
            "scalar",
            "scalar_parent",
            "sub",
            "ä",
            "日本語",
        ]
        assert [e.kind for e in listed][7:10] == ["scalar", "scalar", "catalog"]
        paths = [e.path for e in db.list_objects("/", recursive=True)]
        assert "/cat/sub/deep/tensor" in paths
        assert "/cat/日本語/dates" in paths
        # Julia stored a child under a scalar: readable by path, never listed.
        assert "/cat/scalar_parent/child" not in paths
        assert db.read_scalar("/cat/scalar_parent/child") == 3
        assert db.exists("/cat/scalar_parent/child")
        assert db.object_info("/cat/scalar_parent/child").parent_id == db.object_id(
            "/cat/scalar_parent"
        )
        with pytest.raises(ValueError, match="not a catalog"):
            db.list_objects("/cat/scalar_parent")
        assert len(paths) == 24
        assert db.read_scalar("/cat/scalar") == 1.5
        assert db.read_scalar("cat/sub/text") == "vintage ‖ 3"
        np.testing.assert_array_equal(db.read_array("/cat/sub/vector"), [1, 2, 3])
        np.testing.assert_array_equal(db.read_array("/cat/sub/deep/tensor"), TENSOR)
        assert db.read_series("/cat/sub/deep/series").values.tolist() == [1.0, 2.0, 3.0]
        assert list(db.read_series("/cat/sub/mvtseries").columns) == ["p", "q"]
        np.testing.assert_array_equal(db.read_series("/cat/sub/bools").values, [True, False])
        assert db.read_array("/cat/sub/text_vector") == ["a", "b"]
        dates = db.read_series("/cat/日本語/dates")
        assert isinstance(dates, StoredSeries)
        assert dates.values.tolist() == CATALOG_DATES
        assert db.read_array("/cat/日本語/date_array").shape == (1, 2)
        assert db.get_attributes("/cat/scalar") == CATALOG_ATTRIBUTES
        assert db.get_attribute("/cat/scalar", "k‖3") == "v3"
        assert db.get_attributes("/cat") == {"owner": "julia"}
        assert db.get_attributes("/") == {"DE_VERSION": "0.4.0", "root_note": "hello"}
        assert db.get_attribute("/cat/sub/bools", "jeltype") == "Bool"
        info = db.object_info("/cat/sub/deep/tensor")
        assert (info.depth, info.kind, info.object_class, info.object_type) == (4, "array", 4, 30)
        with closing(sqlite3.connect(fixture)) as conn:
            rows = dict(conn.execute("SELECT fullpath, id FROM objects_info WHERE id != 0"))
        assert {e.path: e.id for e in db.list_objects("/", recursive=True)} == {
            p: i for p, i in rows.items() if p != "/cat/scalar_parent/child"
        }
    # Python's rewrite of the same tree stores identical paths, depths, classes,
    # types and attribute rows (the scalar's child excepted: Python refuses it).
    rewritten = tmp_path / "rewritten.daec"
    with open_dataecon(rewritten, "w") as db:
        rewrite_catalog_tree(db)
    assert raw_tree(rewritten) == raw_tree(fixture, exclude={"/cat/scalar_parent/child"})


@NATIVE
def test_deep_catalog_chains_list_without_recursion():
    # The writer imposes no depth limit, so unlimited listing must not depend
    # on the interpreter's recursion limit (an explicit stack, not recursion).
    depth = 1050
    with open_dataecon_memory() as db:
        chain = "/" + "/".join(["c"] * depth)
        db.new_catalog(chain, parents=True)
        db.write_scalar(chain + "/leaf", 1)
        db.new_catalog("/sibling")
        assert db.object_info(chain).depth == depth
        full = db.list_objects("/", recursive=True)
        assert len(full) == depth + 2
        assert [e.depth for e in full[:3]] == [1, 2, 3]
        assert full[depth].path == chain + "/leaf"
        assert full[depth].depth == depth + 1
        assert full[-1].path == "/sibling"  # emitted after the whole chain (pre-order)
        # Finite-depth controls and a nested starting catalog.
        assert [e.depth for e in db.list_objects("/", recursive=True, max_depth=3)] == [1, 2, 3, 1]
        middle = "/" + "/".join(["c"] * 500)
        nested = db.list_objects(middle, recursive=True)
        assert len(nested) == depth - 500 + 1
        assert nested[0].path == middle + "/c"
        assert nested[-1].path == chain + "/leaf"
        assert [e.depth for e in db.list_objects(middle, recursive=True, max_depth=2)] == [501, 502]
        assert len(db.list_objects(middle)) == 1


@NATIVE
def test_foreign_nul_attribute_value_is_cut_like_julia(tmp_path):
    # Python and Julia never write NUL into an attribute (Python refuses it,
    # the C API truncates it). A foreign SQLite writer can. The ABI returns a
    # C string without a length, so both getters return the prefix before the
    # NUL, exactly as the pinned Julia reader does; nothing can detect the cut.
    # This documents the boundary of the exactness guarantee.
    path = tmp_path / "foreign-nul.daec"
    with open_dataecon(path, "a") as db:
        db.write_scalar("x", 1)
        db.set_attribute("/x", "note", "placeholder")
        db.set_attribute("/x", "other", "kept")
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("UPDATE attributes SET value = ? WHERE name = 'note'", ("before\x00after",))
    with open_dataecon(path) as db:
        assert db.get_attribute("/x", "note") == "before"
        assert db.get_attributes("/x") == {"note": "before", "other": "kept"}
