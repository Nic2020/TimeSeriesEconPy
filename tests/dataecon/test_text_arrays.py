# SPDX-License-Identifier: MIT
"""Text matrices and tensors: packed NUL-terminated UTF-8 at ranks two to five."""

from __future__ import annotations

import importlib.util
import shutil
import sqlite3
import tomllib
from contextlib import closing
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from tsecon import Workspace
from tsecon.dataecon import (
    MAX_UNICODE_BYTES,
    StoredText,
    open_dataecon,
    open_dataecon_memory,
)
from tsecon.dataecon._arrays import text_count
from tsecon.dataecon._codec import (
    MAX_BYTES,
    ArrayPayload,
    MatrixPayload,
    TensorPayload,
    decode_array,
    encode_array,
    tensor_metadata,
    validate_matrix_metadata,
    validate_matrix_payload,
    validate_tensor_payload,
)

FIXTURES = Path(__file__).parent / "fixtures"
NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built; configure TSECON_DATAECON_ROOT",
)
M23 = np.array([["r1c1", "r1c2", "r1c3"], ["r2c1", "r2c2", "r2c3"]])
MULTI22 = np.array([["é", "aé"], ["\U0001f642", "z"]])
T223 = np.array([f"e{i}" for i in range(1, 13)]).reshape((2, 2, 3), order="F")
H24 = np.array([f"h{i}" for i in range(1, 25)])
SHAPES = [(2, 3), (3, 2), (1, 2, 3), (3, 1, 2, 1, 1), (2, 3, 1), (1, 1, 6, 1)]


def packed(values: np.ndarray) -> bytes:
    """The payload DataEcon stores: column-major UTF-8 elements, one NUL each."""
    return b"".join(item.encode("utf-8") + b"\0" for item in values.ravel(order="F").tolist())


def labels(shape: tuple[int, ...]) -> np.ndarray:
    """Distinct labels at every position, so any order mistake changes the bytes."""
    size = int(np.prod(shape))
    return np.array([f"p{i}" for i in range(size)], dtype=str).reshape(shape, order="F").copy()


def julia_fixture_inventory() -> dict[str, tuple[np.ndarray | StoredText, str | None]]:
    """Every object of ``julia_text_arrays.daec`` with its Python reading and stored marker."""
    symbols = np.array([["alpha", "b"], ["", "d"]])
    substrings = np.array([["ab", "bc"]])
    sym = lambda a: StoredText.from_numpy(a, "Symbol")  # noqa: E731
    return {
        "/text2_2x3": (M23, None),
        "/text2_3x2": (M23.T.copy(), None),
        "/text2_blank_2x2": (np.array([["", ""], ["", ""]]), None),
        "/text2_symbol_2x2": (sym(symbols), "Symbol"),
        "/text2_substring_1x2": (
            StoredText.from_numpy(substrings, "SubString{String}"),
            "SubString{String}",
        ),
        "/text2_empty_0x0": (np.empty((0, 0), dtype=str), "String"),
        "/text2_empty_0x3": (np.empty((0, 3), dtype=str), "String"),
        "/text2_empty_3x0": (np.empty((3, 0), dtype=str), "String"),
        "/text2_symbol_empty_0x2": (StoredText((), "Symbol", (0, 2)), "Symbol"),
        "/text3_2x2x3": (T223, None),
        "/text3_3x2x4": (H24.reshape((3, 2, 4), order="F"), None),
        "/text3_4x3x2": (H24.reshape((4, 3, 2), order="F"), None),
        "/text4_1x2x3x1": (
            np.array([f"f{i}" for i in range(1, 7)]).reshape((1, 2, 3, 1), order="F"),
            None,
        ),
        "/text5_2x1x2x1x1": (
            np.array([f"g{i}" for i in range(1, 5)]).reshape((2, 1, 2, 1, 1), order="F"),
            None,
        ),
        "/text3_symbol_1x2x3": (
            sym(np.array(["a", "bb", "ccc", "", "e", "f"]).reshape((1, 2, 3), order="F")),
            "Symbol",
        ),
        "/text5_symbol_1x1x2x1x1": (sym(np.array(["p", "q"]).reshape((1, 1, 2, 1, 1))), "Symbol"),
        "/text3_substring_1x1x2": (
            StoredText.from_numpy(substrings.reshape((1, 1, 2)), "SubString{String}"),
            "SubString{String}",
        ),
        "/text2_multibyte_2x2": (MULTI22, None),
        "/text2_multibyte_symbol_2x2": (sym(MULTI22), "Symbol"),
        "/text3_multibyte_1x2x2": (MULTI22.reshape((1, 2, 2), order="F"), None),
        "/text5_multibyte_symbol_2x1x1x1x2": (
            sym(MULTI22.reshape((2, 1, 1, 1, 2), order="F")),
            "Symbol",
        ),
        "/text3_empty_0x2x3": (np.empty((0, 2, 3), dtype=str), None),
        "/text3_empty_marked_0x2x3": (np.empty((0, 2, 3), dtype=str), "String"),
        "/text5_symbol_empty_0x0x0x0x0": (StoredText((), "Symbol", (0, 0, 0, 0, 0)), "Symbol"),
        "/text4_empty_1x0x1x0": (np.empty((1, 0, 1, 0), dtype=str), None),
        "/text2_string_marker_2x3": (StoredText.from_numpy(M23, "String"), "String"),
        "/text3_string_marker_2x2x3": (StoredText.from_numpy(T223, "String"), "String"),
        "/text2_abstract_2x3": (StoredText.from_numpy(M23, "AbstractString"), "AbstractString"),
        "/text_ws/tm": (M23, None),
        "/text_ws/ts": (sym(symbols), "Symbol"),
        "/text_ws/tt": (T223, None),
        "/text_ws/t5": (sym(np.array(["p", "q"]).reshape((1, 1, 2, 1, 1))), "Symbol"),
        "/text_ws/tv": (["a", "b"], None),
    }


def assert_text_equal(actual, expected) -> None:
    if isinstance(expected, StoredText):
        assert actual == expected
        assert actual.tolist() == expected.tolist()
    elif isinstance(expected, list):
        assert actual == expected
    else:
        assert isinstance(actual, np.ndarray)
        assert actual.dtype.kind == "U"
        assert actual.shape == expected.shape
        assert actual.tolist() == expected.tolist()
        assert actual.flags.owndata
        assert actual.flags.c_contiguous
        assert actual.flags.writeable


# ---- StoredText at every rank ---------------------------------------------


def test_rank_one_container_is_unchanged():
    stored = StoredText((b"a", b"\xff"), "String")
    assert stored.shape == (2,)
    assert stored.ndim == 1
    assert stored.size == 2
    assert len(stored) == 2
    assert stored == StoredText((b"a", b"\xff"), "String", (2,))
    assert StoredText.from_list(["x", "y"]) == StoredText((b"x", b"y"))
    assert StoredText.from_list(["x", "y"]).tolist() == ["x", "y"]
    assert not stored.is_text
    with pytest.raises(ValueError, match="element 1"):
        stored.tolist()


def test_values_are_column_major_and_tolist_is_row_major():
    stored = StoredText.from_list([["a", "b", "c"], ["d", "e", "f"]])
    assert stored.shape == (2, 3)
    assert stored.values == (b"a", b"d", b"b", b"e", b"c", b"f")
    assert stored.tolist() == [["a", "b", "c"], ["d", "e", "f"]]
    assert stored.to_numpy().tolist() == [["a", "b", "c"], ["d", "e", "f"]]
    assert stored.to_numpy().flags.c_contiguous
    assert stored.to_numpy().flags.owndata
    cube = StoredText.from_list([[["a", "b"], ["c", "d"]], [["e", "f"], ["g", "h"]]])
    assert cube.shape == (2, 2, 2)
    assert cube.values == (b"a", b"e", b"c", b"g", b"b", b"f", b"d", b"h")
    assert cube.tolist() == [[["a", "b"], ["c", "d"]], [["e", "f"], ["g", "h"]]]


@pytest.mark.parametrize("shape", SHAPES)
def test_from_numpy_matches_from_list_for_any_layout(shape):
    values = labels(shape)
    direct = StoredText.from_numpy(values)
    assert direct.shape == shape
    assert direct == StoredText.from_list(values.tolist())
    assert direct == StoredText.from_numpy(np.asfortranarray(values))
    assert direct == StoredText.from_numpy(np.ascontiguousarray(values))
    assert direct.to_numpy().tolist() == values.tolist()
    # The values tuple is exactly the stored column-major order.
    assert b"".join(v + b"\0" for v in direct.values) == packed(values)


def test_empty_shapes_are_preserved_by_the_container():
    for shape in ((0,), (0, 3), (3, 0), (0, 2, 3), (1, 0, 1, 0), (0, 0, 0, 0, 0)):
        stored = StoredText((), None, shape)
        assert stored.shape == shape
        assert stored.size == 0
        assert stored.to_numpy().shape == shape
        assert stored.tolist() == np.empty(shape, dtype=str).tolist()
        assert StoredText.from_numpy(np.empty(shape, dtype=str)) == stored
    assert StoredText.from_list([]).shape == (0,)
    assert StoredText.from_list([[], []]).shape == (2, 0)
    assert StoredText.from_list([[[]]]).shape == (1, 1, 0)


def test_container_refuses_inconsistent_construction():
    with pytest.raises(ValueError, match="describes 4 elements"):
        StoredText((b"a", b"b"), None, (2, 2))
    with pytest.raises(ValueError, match="non-negative"):
        StoredText((), None, (-1, 0))
    with pytest.raises(ValueError, match="non-negative"):
        StoredText((b"a",), None, (1, 1, 1, 1, 1, 1))
    with pytest.raises(ValueError, match="non-negative"):
        StoredText((b"a",), None, ())
    with pytest.raises(ValueError, match="non-negative"):
        StoredText((b"a",), None, (np.int64(1),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ragged"):
        StoredText.from_list([["a", "b"], ["c"]])
    with pytest.raises(TypeError, match="plain Python strings"):
        StoredText.from_list([["a"], [b"b"]])
    with pytest.raises(TypeError, match="plain Python strings"):
        StoredText.from_list("ab")
    # Other iterables are never coerced through iteration: a nested dict would
    # otherwise contribute its keys, a set an arbitrary order.
    with pytest.raises(TypeError, match="lists or tuples"):
        StoredText.from_list([["a", "b"], {"c": 1, "d": 2}])
    with pytest.raises(TypeError, match="lists or tuples"):
        StoredText.from_list({"a", "b"})
    with pytest.raises(TypeError, match="lists or tuples"):
        StoredText.from_list(iter(["a", "b"]))
    with pytest.raises(TypeError, match="lists or tuples"):
        encode_array([["a", "b"], {"c": 1, "d": 2}])
    assert StoredText.from_list((("a", "b"), ["c", "d"])).shape == (2, 2)
    with pytest.raises(TypeError, match="plain Python strings"):
        StoredText.from_list([["a", 1]])
    with pytest.raises(ValueError, match="at most 5"):
        StoredText.from_list([[[[[["a"]]]]]])
    with pytest.raises(TypeError, match="str_"):
        StoredText.from_numpy(np.array([[b"a"]]))
    with pytest.raises(TypeError, match="str_"):
        StoredText.from_numpy(np.array([["a"]], dtype=object))
    with pytest.raises(ValueError, match="one to 5"):
        StoredText.from_numpy(np.array("scalar"))
    with pytest.raises(ValueError, match="one to 5"):
        StoredText.from_numpy(np.empty((1,) * 6, dtype=str))
    with pytest.raises(ValueError, match="NUL"):
        StoredText.from_numpy(np.array([["a\0b"]]))
    with pytest.raises(ValueError, match="NUL"):
        StoredText((b"a\0",), None, (1, 1))


def test_decode_error_names_the_logical_index():
    stored = StoredText((b"a", b"b", b"\xff", b"d"), None, (2, 2))
    assert not stored.is_text
    with pytest.raises(ValueError, match=r"element 2 \(index \(0, 1\)\)"):
        stored.tolist()
    with pytest.raises(ValueError, match=r"element 2"):
        stored.to_numpy()
    assert stored.values[2] == b"\xff"


# ---- codec: layout, markers, limits ----------------------------------------


@pytest.mark.parametrize("shape", SHAPES)
def test_text_payload_is_column_major_at_every_rank(shape):
    values = labels(shape)
    encoded = encode_array(values)
    if len(shape) == 2:
        assert isinstance(encoded, MatrixPayload)
        assert (encoded.object_type, encoded.element, encoded.element_frequency) == (20, 6, 0)
        assert (encoded.rows, encoded.columns) == shape
        assert encoded.names is None
    else:
        assert isinstance(encoded, TensorPayload)
        assert (encoded.object_type, encoded.element, encoded.shape) == (30, 6, shape)
    assert encoded.payload == packed(values)
    assert encoded.marker is None
    assert encoded.object_marker is None
    # Sliced, transposed and Fortran inputs snapshot the same logical values.
    assert encode_array(np.asfortranarray(values)).payload == encoded.payload
    assert encode_array(values.tolist()).payload == encoded.payload
    assert encode_array(StoredText.from_numpy(values)).payload == encoded.payload
    padded = np.full(tuple(n + 1 for n in shape), "zz")
    padded[tuple(slice(0, n) for n in shape)] = values
    assert encode_array(padded[tuple(slice(0, n) for n in shape)]).payload == encoded.payload


def test_transposed_matrices_store_different_bytes_and_axes():
    a, b = encode_array(M23), encode_array(M23.T)
    assert (a.rows, a.columns, b.rows, b.columns) == (2, 3, 3, 2)
    assert a.payload != b.payload
    assert a.payload == b"r1c1\0r2c1\0r1c2\0r2c2\0r1c3\0r2c3\0"
    assert b.payload == b"r1c1\0r1c2\0r1c3\0r2c1\0r2c2\0r2c3\0"


def test_one_run_at_two_tensor_shapes_stores_identical_bytes():
    a = encode_array(H24.reshape((3, 2, 4), order="F"))
    b = encode_array(H24.reshape((4, 3, 2), order="F"))
    assert a.payload == b.payload == packed(H24)
    assert a.shape == (3, 2, 4)
    assert b.shape == (4, 3, 2)


def test_rank_one_str_array_writes_as_a_text_vector():
    encoded = encode_array(np.array(["x", "yy"]))
    assert isinstance(encoded, ArrayPayload)
    assert (encoded.object_type, encoded.element, encoded.length) == (10, 6, 2)
    assert encoded.payload == b"x\0yy\0"


def test_multibyte_text_is_sized_in_bytes_at_every_rank():
    for shape in ((2, 2), (1, 2, 2), (2, 1, 1, 1, 2)):
        values = MULTI22.reshape(shape, order="F")
        encoded = encode_array(values)
        assert len(encoded.payload) == 14  # 2 + 3 + 4 + 1 bytes plus four terminators
        assert encoded.payload == packed(values)


def test_empty_text_arrays_carry_the_string_token_and_keep_their_shape():
    for shape in ((0, 3), (3, 0), (0, 0), (0, 2, 3), (1, 0, 1, 0), (0, 0, 0, 0, 0)):
        encoded = encode_array(np.empty(shape, dtype=str))
        assert encoded.payload == b""
        assert encoded.marker == "String"
        if len(shape) == 2:
            assert (encoded.rows, encoded.columns) == shape
            metadata = (3, 20, 6, 0, 0, shape[0], 0, 0, 0, shape[1], 0, 0, 0)
        else:
            assert encoded.shape == shape
            metadata = tensor_metadata(6, 0, shape, 0)
        decoded = decode_array(metadata, b"", "String", None)
        assert isinstance(decoded, np.ndarray)
        assert decoded.shape == shape
        assert decoded.dtype.kind == "U"
        # An unmarked empty (what a raw writer or Python's tensor rule for
        # numeric kinds would store) reads the same way.
        assert decode_array(metadata, b"", None, None).shape == shape
        # A Symbol token on an empty is preserved with the shape.
        symbols = decode_array(metadata, b"", "Symbol", None)
        assert symbols == StoredText((), "Symbol", shape)


def test_markers_are_preserved_or_refused_by_the_finite_table():
    metadata = (3, 20, 6, 0, 0, 2, 0, 0, 0, 3, 0, 0, 30)
    payload = packed(M23)
    assert isinstance(decode_array(metadata, payload, None, None), np.ndarray)
    for token in ("Symbol", "SubString{String}", "AbstractString", "String"):
        stored = decode_array(metadata, payload, token, None)
        assert stored == StoredText.from_numpy(M23, token)
        assert encode_array(stored).marker == token
    with pytest.raises(TypeError, match="Char"):
        decode_array(metadata, payload, "Char", None)
    with pytest.raises(TypeError, match="Whole-object"):
        decode_array(metadata, payload, None, "Matrix{String}")
    with pytest.raises(TypeError, match="Whole-object"):
        decode_array(tensor_metadata(6, 0, (2, 2, 3), 39), packed(T223), None, "Array{String,3}")
    with pytest.raises(TypeError, match="Unsupported text reconstruction marker"):
        encode_array(StoredText((b"a",), "Vector{String}", (1, 1)))
    with pytest.raises(TypeError, match="never evaluated"):
        decode_array(metadata, payload, 'error("x")', None)


def test_contradictory_metadata_is_refused_before_any_allocation():
    good = (3, 20, 6, 0, 0, 2, 0, 0, 0, 3, 0, 0, 30)
    payload = packed(M23)
    with pytest.raises(TypeError, match="no element frequency"):
        validate_matrix_metadata((3, 20, 6, 32, 0, 2, 0, 0, 0, 3, 0, 0, 30))
    with pytest.raises(TypeError, match="cannot hold text"):
        validate_matrix_metadata((3, 21, 6, 0, 1, 2, 32, 0, 2, 3, 0, 0, 30))
    with pytest.raises(ValueError, match="signed 64-bit"):
        validate_matrix_metadata((3, 20, 6, 0, 0, 1 << 40, 0, 0, 0, 1 << 40, 0, 0, 0))
    with pytest.raises(ValueError, match="oversized"):
        validate_matrix_metadata((3, 20, 6, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1))
    with pytest.raises(TypeError, match="no element frequency"):
        validate_tensor_payload(tensor_metadata(6, 65, (2, 2, 3), 39), packed(T223), None, None)
    with pytest.raises(ValueError, match="signed 64-bit"):
        validate_tensor_payload(tensor_metadata(6, 0, (1 << 31, 1 << 32, 1), 0), b"", None, None)
    # Terminator count must equal the element count, and the payload must end
    # with a terminator; extra packed elements are refused rather than dropped.
    for bad in (payload[:-1], payload + b"x\0", b"abcdef", b"", payload[:-5]):
        with pytest.raises(ValueError, match="NUL terminators"):
            validate_matrix_payload(good, bad, None, None)
    with pytest.raises(ValueError, match="empty payload"):
        validate_matrix_payload((3, 20, 6, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0), b"\0", None, None)
    with pytest.raises(ValueError, match="NUL terminators"):
        validate_tensor_payload(tensor_metadata(6, 0, (2, 2, 2), 39), packed(T223), None, None)


def test_capacity_is_enforced_cumulatively_before_encoding(monkeypatch):
    monkeypatch.setattr("tsecon.dataecon._arrays.MAX_BYTES", 16)
    with pytest.raises(ValueError, match="above the"):
        encode_array(np.array([["aaaaaaaa", "bbbbbbbb"], ["cccccccc", "dddddddd"]]))
    with pytest.raises(ValueError, match="above the"):
        encode_array([["aaaaaaaa", "bbbbbbbb"], ["cccccccc", "dddddddd"]])
    with pytest.raises(ValueError, match="above the"):
        encode_array(StoredText((b"a" * 8, b"b" * 8, b"c" * 8), None, (3, 1)))
    assert len(encode_array(np.array([["aaaaaaa", "bbbbbbb"]])).payload) == 16


def test_text_input_forms_are_exact():
    with pytest.raises(TypeError, match="bytes and object"):
        encode_array(np.array([[b"a", b"b"]]))
    with pytest.raises(TypeError, match="bytes and object"):
        encode_array(np.array([["a", "b"]], dtype=object))
    with pytest.raises(TypeError, match="plain Python strings"):
        encode_array([["a", "b"], ["c", 1]])
    with pytest.raises(ValueError, match="ragged"):
        encode_array([["a", "b"], ["c"]])
    with pytest.raises(ValueError, match="NUL"):
        encode_array(np.array([["a", "b\0c"]]))
    with pytest.raises(ValueError, match="one to 5"):
        encode_array(np.empty((1,) * 6, dtype=str))
    with pytest.raises(ValueError, match="one to 5"):
        encode_array(np.array("scalar"))

    class Text(str):
        pass

    with pytest.raises(TypeError, match="plain Python strings"):
        encode_array([[Text("a")]])


# ---- file round trips --------------------------------------------------------


@NATIVE
@pytest.mark.parametrize("shape", SHAPES)
def test_text_arrays_round_trip_and_own_their_result(shape):
    values = labels(shape)
    with open_dataecon_memory() as db:
        db.write_array("plain", values)
        db.write_array("nested", values.tolist())
        db.write_array("symbols", StoredText.from_numpy(values, "Symbol"))
        db.write_array("empty", np.empty((*shape[:-1], 0), dtype=str))
        plain, nested, symbols, empty = (
            db.read_array(n) for n in ("plain", "nested", "symbols", "empty")
        )
        assert db.get_attributes("plain") == {}
        assert db.get_attributes("symbols") == {"jeltype": "Symbol"}
        assert db.get_attributes("empty") == {"jeltype": "String"}
    assert_text_equal(plain, values)
    assert_text_equal(nested, values)
    assert symbols == StoredText.from_numpy(values, "Symbol")
    assert symbols.to_numpy().tolist() == values.tolist()
    assert empty.shape == (*shape[:-1], 0)
    assert empty.size == 0
    plain[...] = "changed"
    assert plain.tolist() != values.tolist()


@NATIVE
def test_writes_snapshot_the_input_and_never_mutate_it():
    values = M23.copy()
    view = values[:, ::2]
    with open_dataecon_memory() as db:
        db.write_array("sliced", view)
        values[...] = "last"
        assert db.read_array("sliced").tolist() == [["r1c1", "r1c3"], ["r2c1", "r2c3"]]
        assert view.tolist() == [["last", "last"], ["last", "last"]]


@NATIVE
def test_raw_bytes_survive_a_read_modify_write_at_rank_two():
    stored = StoredText((b"\xff", b"ok", b"a\xc3", b""), None, (2, 2))
    with open_dataecon_memory() as db:
        db.write_array("raw", stored)
        back = db.read_array("raw")
        assert back == stored
        assert not back.is_text
        db.write_array("copy", back)
        assert db.read_array("copy") == stored
        assert db.get_attributes("copy") == {}


@NATIVE
def test_refused_overwrite_leaves_the_original_intact():
    with open_dataecon_memory() as db:
        db.write_array("keep", M23)
        for bad in (
            [["a", "b"], ["c"]],
            np.array([["a\0b"]]),
            np.array([[b"a"]]),
            StoredText((b"a",), "Char", (1, 1)),
        ):
            with pytest.raises((TypeError, ValueError)):
                db.write_array("keep", bad, overwrite=True)
            assert db.read_array("keep").tolist() == M23.tolist()
        db.write_array("keep", T223, overwrite=True)
        assert db.read_array("keep").shape == (2, 2, 3)
        with pytest.raises(Exception, match="already exists"):
            db.write_array("keep", M23)


@NATIVE
def test_workspace_round_trips_text_arrays_and_reports_unsupported_members():
    ws = Workspace(
        tm=M23,
        tt=T223,
        ts=StoredText.from_numpy(MULTI22, "Symbol"),
        nested=Workspace(
            t5=np.array(["p", "q"]).reshape((1, 1, 2, 1, 1)), e=np.empty((0, 2), dtype=str)
        ),
    )
    with open_dataecon_memory() as db:
        report = db.write_workspace(ws)
        assert report.ok
        assert report.count == 6
        loaded = db.read_workspace()
        assert loaded.report.ok
        assert loaded.report.count == 6
        assert_text_equal(loaded.workspace.tm, M23)
        assert_text_equal(loaded.workspace.tt, T223)
        assert loaded.workspace.ts == ws.ts
        assert loaded.workspace.nested.t5.shape == (1, 1, 2, 1, 1)
        assert loaded.workspace.nested.e.shape == (0, 2)
    ws.bad = np.array([["a", "b"]], dtype=object)
    ws.ragged = [["a", "b"], ["c"]]
    with open_dataecon_memory() as db:
        report = db.write_workspace(ws)
        assert not report.ok
        assert report.count == 6
        assert {(s.path, s.category) for s in report.skipped} == {
            ("/bad", "unsupported"),
            ("/ragged", "invalid"),
        }
        assert all("text" in s.reason for s in report.skipped)
        db.new_catalog("/strict")
        with pytest.raises(TypeError, match="bytes and object"):
            db.write_workspace(ws, "/strict", strict=True)


@NATIVE
@pytest.mark.parametrize(
    ("name", "statements", "error", "match"),
    [
        (
            "text2_2x3",
            ["UPDATE mvtseries SET value=substr(value, 1, 29) WHERE id=:id"],
            ValueError,
            "NUL terminators",
        ),
        (
            "text2_2x3",
            ["UPDATE mvtseries SET value=CAST(value || x'7800' AS BLOB) WHERE id=:id"],
            ValueError,
            "NUL terminators",
        ),
        (
            "text2_2x3",
            ["UPDATE mvtseries SET value=x'616263646566' WHERE id=:id"],
            ValueError,
            "NUL terminators",
        ),
        (
            "text2_2x3",
            ["UPDATE mvtseries SET value=NULL WHERE id=:id"],
            ValueError,
            "NUL terminators",
        ),
        (
            "text2_2x3",
            ["UPDATE mvtseries SET elfreq=32 WHERE id=:id"],
            TypeError,
            "no element frequency",
        ),
        ("text2_2x3", ["INSERT INTO attributes VALUES(:id,'jeltype','Char')"], TypeError, "Char"),
        (
            "text2_2x3",
            ["INSERT INTO attributes VALUES(:id,'jtype','Matrix{String}')"],
            TypeError,
            "Whole-object",
        ),
        (
            "text2_empty_0x3",
            ["UPDATE mvtseries SET value=x'00' WHERE id=:id"],
            ValueError,
            "oversized",
        ),
        (
            "text3_2x2x3",
            ["UPDATE ndtseries SET value=substr(value, 1, 38) WHERE id=:id"],
            ValueError,
            "NUL terminators",
        ),
        (
            "text3_2x2x3",
            ["UPDATE ndtseries SET value=NULL WHERE id=:id"],
            ValueError,
            "NUL terminators",
        ),
        (
            "text3_2x2x3",
            ["UPDATE ndtseries SET elfreq=65 WHERE id=:id"],
            TypeError,
            "no element frequency",
        ),
        (
            "text3_2x2x3",
            ["INSERT INTO attributes VALUES(:id,'jtype','Array{String,3}')"],
            TypeError,
            "Whole-object",
        ),
        (
            "text3_2x2x3",
            ["DELETE FROM ndaxes WHERE obj_id=:id AND axis_index=1"],
            TypeError,
            "axis 1 is missing",
        ),
        # Julia's MVTSeries never holds text: a text payload under type 21 is native-only.
        (
            "text2_2x3",
            [
                "UPDATE objects SET type=21 WHERE id=:id",
                "UPDATE axes SET ax_type=1, frequency=32 "
                "WHERE id=(SELECT axis1_id FROM mvtseries WHERE id=:id)",
                "UPDATE axes SET ax_type=2 WHERE id=(SELECT axis2_id FROM mvtseries WHERE id=:id)",
            ],
            TypeError,
            "cannot hold text",
        ),
    ],
)
def test_malformed_text_rows_are_refused(tmp_path, name, statements, error, match):
    path = tmp_path / "malformed.daec"
    shutil.copyfile(FIXTURES / "julia_text_arrays.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        (oid,) = conn.execute("SELECT id FROM objects WHERE name=?", (name,)).fetchone()
        for statement in statements:
            conn.execute(statement, {"id": oid})
    with open_dataecon(path) as db, pytest.raises(error, match=match):
        db.read_array(name)


# ---- bounded materialization, bounded snapshots, captured shape -------------


def _forbid_unicode_allocation(monkeypatch):
    def spy(decoded, shape):
        raise AssertionError("a fixed-width Unicode array was allocated")

    monkeypatch.setattr("tsecon.dataecon._arrays.unicode_array", spy)


def test_plain_reads_never_materialize_above_the_unicode_cap(monkeypatch):
    # 20 elements whose longest is 20 code points: 20 * 20 * 4 = 1,600 bytes of
    # fixed-width Unicode for a 40-byte payload. Under a 64-byte cap the read
    # returns the lossless container without allocating.
    monkeypatch.setattr("tsecon.dataecon._arrays.MAX_UNICODE_BYTES", 64)
    _forbid_unicode_allocation(monkeypatch)
    payload = b"x" * 20 + b"\0" * 20
    metadata = (3, 20, 6, 0, 0, 1, 0, 0, 0, 20, 0, 0, len(payload))
    stored = decode_array(metadata, payload, None, None)
    assert isinstance(stored, StoredText)
    assert stored.shape == (1, 20)
    assert stored.marker is None
    assert stored.is_text
    assert stored.unicode_nbytes == 1600
    assert stored.tolist() == [["x" * 20] + [""] * 19]
    with pytest.raises(ValueError, match="above the 64-byte limit"):
        stored.to_numpy()
    tensor = decode_array(tensor_metadata(6, 0, (1, 2, 10), len(payload)), payload, None, None)
    assert isinstance(tensor, StoredText)
    assert tensor.shape == (1, 2, 10)
    # Rank one never pads: the list form is returned as before.
    assert (
        decode_array((2, 10, 6, 0, 0, 20, 0, 0, len(payload)), payload, None, None)
        == ["x" * 20] + [""] * 19
    )
    # Rewriting the container stores the same bytes.
    assert encode_array(stored).payload == payload


def test_reads_under_the_cap_still_materialize(monkeypatch):
    monkeypatch.setattr("tsecon.dataecon._arrays.MAX_UNICODE_BYTES", 64)
    payload = b"ab\0c\0\0d\0"
    value = decode_array((3, 20, 6, 0, 0, 2, 0, 0, 0, 2, 0, 0, len(payload)), payload, None, None)
    assert isinstance(value, np.ndarray)
    assert value.tolist() == [["ab", ""], ["c", "d"]]
    assert value.dtype == np.dtype("<U2")
    assert value.flags.owndata
    assert value.flags.c_contiguous


def test_default_cap_admits_any_uniform_payload_and_refuses_padding_blowups():
    assert MAX_UNICODE_BYTES == 4 * MAX_BYTES
    uniform = StoredText((b"abcd",) * 4, None, (2, 2))
    assert uniform.unicode_nbytes == 4 * 4 * 4
    padded = StoredText((b"a" * 1000, *([b""] * 999)), None, (1, 1000))
    assert padded.unicode_nbytes == 1000 * 1000 * 4 > 40 * len(padded.values)


@pytest.mark.parametrize(
    "shape", [*SHAPES, (0, 3), (3, 0), (0, 2, 3), (1, 0, 1, 0), (2, 2, 2, 2, 2)]
)
def test_tolist_nests_without_a_padded_intermediate(monkeypatch, shape):
    _forbid_unicode_allocation(monkeypatch)
    values = labels(shape)
    stored = StoredText.from_numpy(values)
    assert stored.tolist() == values.tolist()
    # Nested lists cannot spell a leading zero dimension, so compare the lists.
    assert StoredText.from_list(values.tolist()).tolist() == values.tolist()


@NATIVE
def test_file_and_workspace_reads_honor_the_cap(monkeypatch):
    payload_shape = (1, 20)
    long = np.array([["x" * 20] + [""] * 19])
    with open_dataecon_memory() as db:
        db.write_array("wide", long)
        db.write_workspace(Workspace(member=long), "/")
        assert db.read_array("wide").tolist() == long.tolist()
        monkeypatch.setattr("tsecon.dataecon._arrays.MAX_UNICODE_BYTES", 64)
        _forbid_unicode_allocation(monkeypatch)
        stored = db.read_array("wide")
        assert isinstance(stored, StoredText)
        assert stored.shape == payload_shape
        loaded = db.read_workspace("/")
        assert loaded.report.ok
        assert isinstance(loaded.workspace.member, StoredText)
        assert loaded.workspace.member.tolist() == long.tolist()


class _Broadcast(np.ndarray):
    """A tiny-memory array whose logical size is refused before any copy."""

    def ravel(self, *args, **kwargs):
        raise RuntimeError("ravel reached: the logical array was copied")

    def tolist(self, *args, **kwargs):
        raise RuntimeError("tolist reached: the logical array was copied")


def test_oversized_inputs_are_refused_before_any_copy(monkeypatch):
    monkeypatch.setattr("tsecon.dataecon._arrays.MAX_BYTES", 8)
    broadcast = np.broadcast_to(np.array("a"), (2, 5)).view(_Broadcast)
    with pytest.raises(ValueError, match="10 bytes, above the 8-byte"):
        encode_array(broadcast)
    with pytest.raises(ValueError, match="10 bytes, above the 8-byte"):
        StoredText.from_numpy(broadcast)
    # NumPy itself cannot build a str_ array beyond the signed-64-bit element
    # count, so the largest constructible broadcast (2**60 elements) is
    # refused by the minimum packed size, and the count bound is checked on
    # the shape helper directly.
    huge = np.lib.stride_tricks.as_strided(
        np.array("a"), shape=(1 << 30, 1 << 30, 1), strides=(0, 0, 0)
    ).view(_Broadcast)
    with pytest.raises(ValueError, match="above the 8-byte"):
        encode_array(huge)
    with pytest.raises(ValueError, match="above the 8-byte"):
        StoredText.from_numpy(huge)
    with pytest.raises(ValueError, match="signed 64-bit"):
        text_count((1 << 31, 1 << 31, 2))
    # Nested traversal stops counting before the input is flattened or reordered.
    monkeypatch.setattr(
        "tsecon.dataecon._arrays.column_major",
        lambda *a: (_ for _ in ()).throw(AssertionError("reordered")),
    )
    with pytest.raises(ValueError, match="above the 8-byte"):
        encode_array([["a"] * 5] * 2)
    with pytest.raises(ValueError, match="above the 8-byte"):
        StoredText.from_list([[["a"] * 3] * 3])
    # Empty strings still cost their terminator.
    with pytest.raises(ValueError, match="above the 8-byte"):
        encode_array(np.full((3, 3), ""))
    assert len(encode_array(np.full((2, 4), "")).payload) == 8


class _ChunkSpy(np.ndarray):
    """Refuses whole-array copies and records the bytes of every chunk it hands out."""

    seen: ClassVar[list[int]] = []

    def ravel(self, *args, **kwargs):
        raise RuntimeError(f"ravel reached: {self.nbytes} bytes")

    def tolist(self, *args, **kwargs):
        # Only chunks (base-class results of __getitem__) may be listed.
        raise RuntimeError(f"tolist reached: {self.nbytes} bytes")

    def __getitem__(self, key):
        out = np.asarray(super().__getitem__(key))
        _ChunkSpy.seen.append(int(out.nbytes))
        return out


@pytest.mark.parametrize("entry", [encode_array, StoredText.from_numpy])
def test_long_string_broadcasts_are_refused_without_a_whole_copy(monkeypatch, entry):
    monkeypatch.setattr("tsecon.dataecon._arrays.TEXT_CHUNK_BYTES", 1 << 20)
    _ChunkSpy.seen = []
    array = np.broadcast_to(np.array("x" * 20000), (20000, 1)).view(_ChunkSpy)
    assert array.strides == (0, 0)
    assert array.size * array.dtype.itemsize == 1_600_000_000
    with pytest.raises(ValueError, match="above the 134217728-byte"):
        entry(array)
    assert _ChunkSpy.seen
    assert max(_ChunkSpy.seen) <= (1 << 20) + array.dtype.itemsize
    # Refused after the cumulative UTF-8 total crossed the cap, far short of the input.
    assert sum(_ChunkSpy.seen) < 1_600_000_000 // 2


@pytest.mark.parametrize("entry", [encode_array, StoredText.from_numpy])
def test_padded_inputs_whose_payload_fits_are_encoded_in_bounded_chunks(monkeypatch, entry):
    monkeypatch.setattr("tsecon.dataecon._arrays.TEXT_CHUNK_BYTES", 1 << 20)
    _ChunkSpy.seen = []
    # One long value pads 999 empty ones to 80 KB each: 80 MB of fixed width for
    # a 21,000-byte payload. Every layout goes through bounded chunks.
    base = np.array([["y" * 20000] + [""] * 999]).reshape(1000, 1, order="F")
    for array in (
        base.view(_ChunkSpy),
        np.asfortranarray(base.T).view(_ChunkSpy),
        np.broadcast_to(base[0, 0], (5, 200)).view(_ChunkSpy),
        base.reshape(-1).view(_ChunkSpy),
    ):
        _ChunkSpy.seen = []
        result = entry(array)
        payload = (
            result.payload
            if isinstance(result, MatrixPayload | ArrayPayload)
            else (b"".join(v + b"\0" for v in result.values))
        )
        expected = b"".join(v.encode() + b"\0" for v in np.asarray(array).ravel(order="F").tolist())
        assert payload == expected
        assert max(_ChunkSpy.seen) <= (1 << 20) + array.dtype.itemsize
        assert sum(_ChunkSpy.seen) >= np.asarray(array).size * array.dtype.itemsize


def test_chunked_traversal_preserves_column_major_order_and_count(monkeypatch):
    monkeypatch.setattr("tsecon.dataecon._arrays.TEXT_CHUNK_BYTES", 8)  # a few elements per chunk
    for shape in SHAPES:
        values = labels(shape)
        for layout in (
            values,
            np.asfortranarray(values),
            values[::-1] if values.ndim == 1 else values.T.T,
        ):
            assert encode_array(layout).payload == packed(np.asarray(layout))
            assert StoredText.from_numpy(layout) == StoredText.from_numpy(np.asarray(layout))
    strided = labels((12,))[::2]
    assert encode_array(strided).payload == packed(strided)


class _Shrink(np.ndarray):
    """Hands out one element fewer than asked, as a concurrently shrinking input would."""

    def __getitem__(self, key):
        out = np.asarray(super().__getitem__(key))
        return out[:-1] if out.ndim else out


class _Grow(np.ndarray):
    """Hands out one element more than asked, as a concurrently growing input would."""

    def __getitem__(self, key):
        out = np.asarray(super().__getitem__(key))
        return np.concatenate([out, np.array(["zz"])]) if out.ndim else out


@pytest.mark.parametrize("shape", [(2,), (2, 2), (1, 2, 2), (1, 1, 2, 1, 1)])
def test_snapshots_that_change_size_are_refused_at_every_rank(shape):
    for cls in (_Shrink, _Grow):
        array = np.array(labels(shape), dtype="<U2").view(cls)
        with pytest.raises(ValueError, match="changed size during the snapshot"):
            encode_array(array)


@NATIVE
def test_refused_snapshots_and_oversized_inputs_leave_the_original_intact(monkeypatch):
    with open_dataecon_memory() as db:
        db.write_array("keep", M23)
        shrink = np.array(["a", "b"]).view(_Shrink)
        with pytest.raises(ValueError, match="changed size"):
            db.write_array("keep", shrink, overwrite=True)
        assert db.read_array("keep").tolist() == M23.tolist()
        monkeypatch.setattr("tsecon.dataecon._arrays.MAX_BYTES", 8)
        broadcast = np.broadcast_to(np.array("a"), (2, 5)).view(_Broadcast)
        with pytest.raises(ValueError, match="above the 8-byte"):
            db.write_array("keep", broadcast, overwrite=True)
        with pytest.raises(ValueError, match="above the 8-byte"):
            db.write_array("keep", [["a"] * 5] * 2, overwrite=True)
        monkeypatch.setattr("tsecon.dataecon._arrays.MAX_BYTES", 128 * 1024 * 1024)
        assert db.read_array("keep").tolist() == M23.tolist()
        assert db.object_info("keep").object_type == 20


# ---- the Julia fixture -----------------------------------------------------


@NATIVE
def test_julia_fixture_inventory_is_complete_and_reads_exactly():
    expected = julia_fixture_inventory()
    provenance = tomllib.loads((FIXTURES / "julia_text_arrays.toml").read_text(encoding="utf-8"))
    assert provenance["timeseriesecon_sha"] == "fc0a0d01aed4903ea6c64b12d78d0bdf68468df6"
    assert provenance["native_version"] == "0.4.0"
    with open_dataecon(FIXTURES / "julia_text_arrays.daec") as db:
        entries = db.list_objects("/", recursive=True)
        assert {e.path for e in entries if e.kind != "catalog"} == set(expected)
        assert [e.path for e in entries if e.kind == "catalog"] == ["/text_ws"]
        for path, (value, marker) in expected.items():
            assert_text_equal(db.read_array(path), value)
            assert db.get_attributes(path) == ({} if marker is None else {"jeltype": marker})
            info = db.object_info(path)
            if isinstance(value, list):
                assert (info.object_class, info.object_type) == (2, 10)
            elif value.ndim == 2:
                assert (info.object_class, info.object_type) == (3, 20)
            else:
                assert (info.object_class, info.object_type) == (4, 30)
        loaded = db.read_workspace("/text_ws")
        assert loaded.report.ok
        assert loaded.report.count == 5
        assert list(loaded.workspace.keys()) == ["t5", "tm", "ts", "tt", "tv"]
        with pytest.raises(TypeError, match="read_array"):
            db.read_series("/text2_2x3")


@NATIVE
def test_python_rewrite_of_the_julia_fixture_is_byte_identical(tmp_path):
    """Every fixture object rewritten by Python stores the same payload, axes and marker."""
    rewritten = tmp_path / "rewrite.daec"
    with (
        open_dataecon(FIXTURES / "julia_text_arrays.daec") as source,
        open_dataecon(rewritten, "w") as target,
    ):
        for path in julia_fixture_inventory():
            if path.startswith("/text_ws/"):
                continue
            target.write_array(path.lstrip("/"), source.read_array(path))
    with (
        closing(sqlite3.connect(FIXTURES / "julia_text_arrays.daec")) as a,
        closing(sqlite3.connect(rewritten)) as b,
    ):
        query = (
            "SELECT o.name, o.class, o.type, m.value, x1.length, x2.length FROM objects o "
            "JOIN mvtseries m ON m.id=o.id JOIN axes x1 ON x1.id=m.axis1_id "
            "JOIN axes x2 ON x2.id=m.axis2_id WHERE o.pid=0 ORDER BY o.name"
        )
        assert a.execute(query).fetchall() == b.execute(query).fetchall()
        query = (
            "SELECT o.name, n.value, (SELECT group_concat(x.length, ',') FROM ndaxes d "
            "JOIN axes x ON x.id=d.axis_id WHERE d.obj_id=o.id ORDER BY d.axis_index) "
            "FROM objects o JOIN ndtseries n ON n.id=o.id WHERE o.pid=0 ORDER BY o.name"
        )
        assert a.execute(query).fetchall() == b.execute(query).fetchall()
        markers = (
            "SELECT o.name, a.name, a.value FROM attributes a JOIN objects o ON o.id=a.id "
            "WHERE o.pid=0 AND a.name IN ('jeltype','jtype') ORDER BY o.name, a.name"
        )
        source_markers = a.execute(markers).fetchall()
        # Python marks every empty text array; the fixture's raw unmarked empties gain the token.
        for name in ("text3_empty_0x2x3", "text4_empty_1x0x1x0"):
            source_markers.append((name, "jeltype", "String"))
        assert sorted(source_markers) == sorted(b.execute(markers).fetchall())
