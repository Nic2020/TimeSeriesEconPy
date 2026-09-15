# SPDX-License-Identifier: MIT
"""Plain matrices, MVTSeries, text vectors and represented plain arrays."""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sqlite3
import subprocess
import sys
import tomllib
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from tsecon import MVTSeries
from tsecon.dataecon import StoredArray, StoredText, _interpret, open_dataecon
from tsecon.dataecon._arrays import resolve_array_interpretation
from tsecon.dataecon._codec import (
    decode_matrix,
    encode_array,
    encode_mvtseries,
    join_names,
    split_names,
    split_text_payload,
    validate_array_payload,
    validate_matrix_metadata,
)
from tsecon.dataecon._represented import COMPLEXF16, INT128, UINT128, StoredElement
from tsecon.frequencies import BDaily, Daily, HalfYearly, Monthly, Quarterly, Unit, Weekly, Yearly
from tsecon.mit import MIT

FIXTURES = Path(__file__).parent / "fixtures"
NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built; configure TSECON_DATAECON_ROOT",
)
MATRIX_DTYPES = [
    "i1",
    "<i2",
    "<i4",
    "<i8",
    "u1",
    "<u2",
    "<u4",
    "<u8",
    "<f2",
    "<f4",
    "<f8",
    "<c8",
    "<c16",
    "?",
]
JULIA_NAMES = {
    "i1": "Int8",
    "<i2": "Int16",
    "<i4": "Int32",
    "<i8": "Int64",
    "u1": "UInt8",
    "<u2": "UInt16",
    "<u4": "UInt32",
    "<u8": "UInt64",
    "<f2": "Float16",
    "<f4": "Float32",
    "<f8": "Float64",
    "<c8": "ComplexF32",
    "<c16": "ComplexF64",
    "?": "Bool",
}


def sample_matrix(dtype: str) -> np.ndarray:
    """Six values shaped 2x3, including signed zeros, NaN payloads and extremes."""
    kind = np.dtype(dtype).kind
    if kind == "b":
        return np.array([[False, False, True], [True, True, False]])
    if kind == "i":
        info = np.iinfo(dtype)
        return np.array([[info.min, 0, 7], [-1, info.max, 9]], dtype=dtype)
    if kind == "u":
        info = np.iinfo(dtype)
        return np.array([[0, info.max, 9], [1, 7, 11]], dtype=dtype)
    if kind == "f":
        width = np.dtype(dtype).itemsize
        bits = {2: np.uint16, 4: np.uint32, 8: np.uint64}[width]
        raw = {
            2: [0x8000, 0x0001, 0x3D00, 0x7E55, 0x7C00, 0x0002],
            4: [0x80000000, 1, 0x3FA00000, 0x7FC00055, 0x7F800000, 2],
            8: [
                0x8000000000000000,
                1,
                0x3FF4000000000000,
                0x7FF8000000000055,
                0x7FF0000000000000,
                2,
            ],
        }[width]
        flat = np.array(raw, dtype=bits).view(dtype)
        return flat.reshape((2, 3), order="F")
    component = np.dtype(dtype).char.lower().replace("f", "f4").replace("d", "f8")
    component = "<f4" if np.dtype(dtype).itemsize == 8 else "<f8"
    real = np.array([-0.0, 2.5, 0.0, 1.0, 3.0, 5.0], dtype=component)
    imag = np.array([1.25, -3.0, -0.0, 2.0, 4.0, 6.0], dtype=component)
    flat = np.empty(6, dtype=dtype)
    flat.real, flat.imag = real, imag
    return flat.reshape((2, 3), order="F")


def wide_matrix(element: StoredElement) -> np.ndarray:
    """Four 128-bit values as their two-word carrier, shaped 2x2."""
    words = np.array([1, 0, 2, 0, 3, 0, 4, 0], dtype="<u8")
    return words.view(element.dtype).reshape((2, 2))


def same_values(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare exact stored bytes, so signed zeros and NaN payloads must match."""
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


# ---- pure codec and container behavior ------------------------------------


@pytest.mark.parametrize("dtype", MATRIX_DTYPES)
def test_matrix_payload_is_column_major_and_round_trips(dtype):
    values = sample_matrix(dtype)
    encoded = encode_array(values)
    assert (encoded.object_type, encoded.rows, encoded.columns) == (20, 2, 3)
    assert encoded.names is None
    stored = values.astype(np.int8) if values.dtype.kind == "b" else values
    assert encoded.payload == stored.tobytes(order="F")
    metadata = (3, 20, encoded.element, 0, 0, 2, 0, 0, 0, 3, 0, 0, len(encoded.payload))
    back = decode_matrix(metadata, encoded.payload, encoded.marker, None)
    assert same_values(back, values)
    assert back.flags["C_CONTIGUOUS"]
    assert back.flags["WRITEABLE"]
    assert back.flags["OWNDATA"]


def test_column_major_order_distinguishes_transposed_shapes():
    wide = np.array([[1, 3, 5], [2, 4, 6]], dtype=np.int64)
    tall = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.int64)
    assert encode_array(wide).payload == encode_array(tall).payload
    assert (encode_array(wide).rows, encode_array(wide).columns) == (2, 3)
    assert (encode_array(tall).rows, encode_array(tall).columns) == (3, 2)


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((0, 0), dtype=np.float64),
        np.zeros((0, 3), dtype=np.float64),
        np.zeros((3, 0), dtype=np.float64),
        np.zeros((1, 1), dtype=np.float64),
    ],
)
def test_degenerate_matrix_shapes_keep_their_stored_dimensions(values):
    encoded = encode_array(values)
    metadata = (
        3,
        20,
        4,
        0,
        0,
        values.shape[0],
        0,
        0,
        0,
        values.shape[1],
        0,
        0,
        len(encoded.payload),
    )
    assert decode_matrix(metadata, encoded.payload, encoded.marker, None).shape == values.shape


def test_noncontiguous_and_transposed_input_snapshots_logical_values():
    base = np.arange(12, dtype=np.int64).reshape((3, 4))
    view = base[::-1, ::2]
    encoded = encode_array(view)
    metadata = (3, 20, 1, 0, 0, 3, 0, 0, 0, 2, 0, 0, len(encoded.payload))
    decoded = decode_matrix(metadata, encoded.payload, None, None)
    assert same_values(decoded, np.ascontiguousarray(view))
    transposed = base.T
    encoded = encode_array(transposed)
    metadata = (3, 20, 1, 0, 0, 4, 0, 0, 0, 3, 0, 0, len(encoded.payload))
    assert same_values(
        decode_matrix(metadata, encoded.payload, None, None), np.ascontiguousarray(transposed)
    )
    assert base.tolist() == np.arange(12).reshape((3, 4)).tolist()


def test_matrix_element_count_and_payload_bounds_are_checked():
    with pytest.raises(ValueError, match="signed 64-bit"):
        validate_matrix_metadata((3, 20, 4, 0, 0, 2**62, 0, 0, 0, 2**62, 0, 0, 0))
    with pytest.raises(ValueError, match="element width or payload length"):
        validate_matrix_metadata((3, 20, 4, 0, 0, 2, 0, 0, 0, 2, 0, 0, 24))
    with pytest.raises(ValueError, match="Invalid or oversized"):
        validate_matrix_metadata((3, 20, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8))
    with pytest.raises(ValueError, match="negative"):
        validate_matrix_metadata((3, 20, 4, 0, 0, -1, 0, 0, 0, 2, 0, 0, 0))


@pytest.mark.parametrize(
    ("metadata", "match"),
    [
        ((2, 20, 4, 0, 0, 1, 0, 0, 0, 1, 0, 0, 8), "two-dimensional support"),
        ((3, 12, 4, 0, 0, 1, 0, 0, 0, 1, 0, 0, 8), "two-dimensional support"),
        ((3, 20, 4, 0, 1, 1, 32, 0, 0, 1, 0, 0, 8), "two plain axes"),
        ((3, 21, 4, 0, 0, 1, 0, 0, 2, 1, 0, 0, 8), "dated row axis"),
        ((3, 21, 4, 0, 1, 1, 32, 0, 0, 1, 0, 0, 8), "dated row axis"),
        ((3, 20, 6, 32, 0, 1, 0, 0, 0, 1, 0, 0, 2), "no element frequency"),
        ((3, 21, 6, 0, 1, 1, 32, 0, 2, 1, 0, 0, 2), "cannot hold text"),
        ((3, 20, 4, 0, 0, 1, 0, 0, 0, 1, 32, 0, 8), "no frequency or first date"),
    ],
)
def test_malformed_matrix_metadata_is_refused(metadata, match):
    with pytest.raises((TypeError, ValueError), match=match):
        validate_matrix_metadata(metadata)


# ---- names axis ------------------------------------------------------------


def test_names_axis_encoding_round_trips():
    assert join_names(["a", "b"]) == "a\nb"
    assert split_names("a\nb", 2) == ("a", "b")
    assert split_names("\na", 2) == ("", "a")


@pytest.mark.parametrize(
    ("names", "columns", "match"),
    [
        ("a\nb\nc", 2, "3 names for 2 columns"),
        ("solo", 2, "1 names for 2 columns"),
        ("a\na", 2, "duplicate column names"),
        ("", 0, "cannot encode zero columns"),
        (None, 2, "needs a named column axis"),
    ],
)
def test_unusable_names_axes_are_refused(names, columns, match):
    with pytest.raises((TypeError, ValueError), match=match):
        split_names(names, columns)


@pytest.mark.parametrize(
    ("names", "match"),
    [
        (["a\nb"], "cannot contain a newline"),
        (["a\0b"], "cannot contain NUL"),
        (["a", "a"], "must be distinct"),
        ([], "no columns"),
    ],
)
def test_unwritable_column_names_are_refused(names, match):
    with pytest.raises(ValueError, match=match):
        join_names(names)


def test_mvtseries_payload_matches_the_matrix_layout():
    series = MVTSeries(MIT(Monthly(), 24288), ("a", "b"), np.array([[1.0, 4.0], [2.0, 5.0]]))
    encoded = encode_mvtseries(series)
    assert (encoded.object_type, encoded.axis1_type, encoded.rows, encoded.columns) == (21, 1, 2, 2)
    assert (encoded.frequency, encoded.first, encoded.names) == (32, 24288, "a\nb")
    assert encoded.payload == np.array([1.0, 2.0, 4.0, 5.0]).tobytes()


# ---- text vectors ----------------------------------------------------------


def test_text_payload_splits_only_on_exact_terminator_counts():
    assert split_text_payload(b"alpha\0\0z\0", 3) == (b"alpha", b"", b"z")
    assert split_text_payload(b"", 0) == ()
    with pytest.raises(ValueError, match="terminators"):
        split_text_payload(b"a\0b\0c\0", 2)
    with pytest.raises(ValueError, match="terminators"):
        split_text_payload(b"a\0bc", 2)
    with pytest.raises(ValueError, match="empty payload"):
        split_text_payload(b"a\0", 0)


def test_text_vector_sizes_utf8_bytes_not_characters():
    values = ["é", "\U0001f642", "aé"]
    encoded = encode_array(values)
    # Julia's own writer allocates sum(length) + n = 7 bytes and fails; the
    # native packer needs the twelve UTF-8 bytes this writes.
    assert len(encoded.payload) == 12
    assert encoded.payload == "é\0\U0001f642\0aé\0".encode()


def test_text_elements_cannot_contain_nul():
    with pytest.raises(ValueError, match="cannot contain NUL"):
        encode_array(["a\0b"])
    with pytest.raises(ValueError, match="cannot contain NUL"):
        StoredText((b"a\0b",))


def test_stored_text_preserves_marker_and_raw_bytes():
    stored = StoredText((b"\xff", b"a"), None)
    assert not stored.is_text
    with pytest.raises(ValueError, match="not valid UTF-8"):
        stored.tolist()
    assert StoredText.from_list(["ab", "c"], "Symbol").tolist() == ["ab", "c"]
    assert len(StoredText.from_list(["ab", "c"], "Symbol")) == 2


def test_empty_text_vector_keeps_julias_element_token():
    assert encode_array([]).marker == "String"
    assert encode_array(StoredText((), "Symbol")).marker == "Symbol"


# ---- represented plain arrays ---------------------------------------------


def test_stored_array_carries_shape_element_and_marker():
    element = StoredElement.date(Monthly())
    values = np.array([[1, 2], [3, 4]], dtype="<i8")
    array = StoredArray(values, element)
    assert array.shape == (2, 2)
    assert array.ndim == 2
    assert array.tolist() == [
        [MIT(Monthly(), 1), MIT(Monthly(), 2)],
        [MIT(Monthly(), 3), MIT(Monthly(), 4)],
    ]
    assert array == StoredArray(values, element)
    assert "MIT{Monthly}" in repr(array)


def test_stored_array_copies_by_default_and_validates_mutation():
    values = np.array([1, 2], dtype="<i8")
    array = StoredArray(values, StoredElement.duration(Monthly()))
    values[0] = 99
    assert array.values[0] == 1
    array.validate()


def test_stored_array_rejects_bad_carriers():
    # Ranks three to five are tensors since the N-d contract; six is the limit.
    with pytest.raises(ValueError, match="one to 5 dimensions"):
        StoredArray(np.zeros((1, 1, 1, 1, 1, 1), dtype="<i8"), StoredElement.date(Monthly()))
    with pytest.raises(TypeError, match="storage dtype"):
        StoredArray(np.zeros(2, dtype="<i4"), StoredElement.date(Monthly()), copy=False)
    with pytest.raises(TypeError, match="belong in a plain NumPy array"):
        StoredArray(np.zeros(2, dtype="<i8"), StoredElement.numeric("<i8"))


def test_stored_array_interpretation_matches_the_element_table():
    element = StoredElement.numeric("<i8", "Float64")
    array = StoredArray(np.array([[1, 2], [3, 4]], dtype="<i8"), element)
    assert resolve_array_interpretation(array.values, element, None)[0] == "element"
    converted = array.to_interpreted()
    assert same_values(converted, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_stored_array_bool_marker_converts_only_zero_and_one():
    element = StoredElement.numeric("<i8", "Bool")
    with pytest.raises(TypeError, match="Boolean array"):
        StoredArray(np.array([1, 2], dtype="<i8"), element)
    words = np.array([0, 0, 1, 0], dtype="<i8").view(INT128.dtype).reshape((1, 2))
    wide = StoredArray(words, INT128.with_bool_marker())
    assert wide.to_bool().tolist() == [[False, True]]


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("Diagonal", [[1.0, 0.0], [0.0, 4.0]]),
        ("Symmetric", [[1.0, 3.0], [3.0, 4.0]]),
    ],
)
def test_structure_markers_reproduce_julias_reconstruction(token, expected):
    element = StoredElement.numeric("<f8")
    array = StoredArray(np.array([[1.0, 3.0], [2.0, 4.0]]), element, object_marker=token)
    assert array.to_interpreted().tolist() == expected


def test_hermitian_marker_conjugates_and_clears_the_diagonal_imaginary_part():
    element = StoredElement.numeric("<c16")
    values = np.array([[1 + 9j, 2 + 1j], [20 + 20j, 4 - 9j]], dtype="<c16")
    array = StoredArray(values, element, object_marker="Hermitian")
    result = array.to_interpreted()
    assert result.tolist() == [[1 + 0j, 2 + 1j], [2 - 1j, 4 + 0j]]


def test_structure_markers_need_a_square_matrix():
    array = StoredArray(np.zeros((2, 3)), StoredElement.numeric("<f8"), object_marker="Diagonal")
    with pytest.raises(ValueError, match="square matrices only"):
        array.to_interpreted()
    with pytest.raises(TypeError, match="1-dimensional DataEcon array"):
        StoredArray(np.zeros(3), StoredElement.numeric("<f8"), object_marker="Diagonal")


@pytest.mark.parametrize(
    ("token", "ndim", "kind"),
    [
        ("Vector", 1, "identity"),
        ("AbstractVector", 1, "identity"),
        ("Array", 1, "identity"),
        ("Any", 1, "identity"),
        ("Vector{Int64}", 1, "identity"),
        ("Array{Int64,1}", 1, "identity"),
        ("Array{Int64, 1}", 1, "identity"),
        ("Vector{Int16}", 1, "element"),
        ("Matrix", 2, "identity"),
        ("AbstractMatrix", 2, "identity"),
        ("Matrix{Float64}", 2, "element"),
        ("Array{Int64,2}", 2, "identity"),
    ],
)
def test_supported_array_object_markers(token, ndim, kind):
    element = StoredElement.numeric("<i8")
    shape = (2,) if ndim == 1 else (2, 2)
    values = np.ones(shape, dtype="<i8")
    assert StoredArray(values, element, object_marker=token).active_marker == token
    assert resolve_array_interpretation(values, element, token)[0] == kind


@pytest.mark.parametrize(
    "token", ["TSeries", "MVTSeries", "Symbol", "Ref{Int64}", "Vector{Any}", "NoSuchToken"]
)
def test_unsupported_array_object_markers_are_refused(token):
    with pytest.raises(TypeError, match="whole-object reconstruction marker"):
        StoredArray(np.ones(2, dtype="<i8"), StoredElement.numeric("<i8"), object_marker=token)


# ---- native round trips ----------------------------------------------------


@NATIVE
@pytest.mark.parametrize("dtype", MATRIX_DTYPES)
def test_matrix_round_trip_through_a_file(tmp_path, dtype):
    values = sample_matrix(dtype)
    path = tmp_path / "matrix.daec"
    with open_dataecon(path, "w") as db:
        db.write_array("m", values)
        db.write_array("empty", np.zeros((0, 2), dtype=dtype))
    with open_dataecon(path) as db:
        assert same_values(db.read_array("m"), values)
        empty = db.read_array("empty")
        assert empty.shape == (0, 2)
        assert empty.dtype == np.dtype(dtype)


@NATIVE
@pytest.mark.parametrize(
    "frequency",
    [Unit(), Daily(), BDaily(), Weekly(7), Monthly(), Quarterly(1), HalfYearly(1), Yearly(1)],
)
def test_mvtseries_round_trip_over_axis_families(tmp_path, frequency):
    series = MVTSeries(MIT(frequency, 0), ("a", "b"), np.array([[1.0, 2.0], [3.0, 4.0]]))
    path = tmp_path / "mvts.daec"
    with open_dataecon(path, "w") as db:
        db.write_series("m", series)
    with open_dataecon(path) as db:
        back = db.read_series("m")
        assert isinstance(back, MVTSeries)
        assert back.firstdate == series.firstdate
        assert tuple(back.columns) == ("a", "b")
        assert same_values(back.values, series.values)


@NATIVE
def test_mvtseries_and_matrix_use_their_own_reader(tmp_path):
    path = tmp_path / "split.daec"
    with open_dataecon(path, "w") as db:
        db.write_array("plain", np.ones((2, 2)))
        db.write_series("dated", MVTSeries(MIT(Monthly(), 0), ("a",), np.ones((2, 1))))
    with open_dataecon(path) as db:
        with pytest.raises(TypeError, match="read it with read_array"):
            db.read_series("plain")
        with pytest.raises(TypeError, match="read it with read_series"):
            db.read_array("dated")


@NATIVE
def test_text_vector_round_trip(tmp_path):
    path = tmp_path / "text.daec"
    with open_dataecon(path, "w") as db:
        db.write_array("ascii", ["alpha", "", "z"])
        db.write_array("multibyte", ["é", "\U0001f642"])
        db.write_array("symbol", StoredText.from_list(["alpha", "z"], "Symbol"))
        db.write_array("empty", [])
        db.write_array("raw", StoredText((b"\xff", b"a")))
    with open_dataecon(path) as db:
        assert db.read_array("ascii") == ["alpha", "", "z"]
        assert db.read_array("multibyte") == ["é", "\U0001f642"]
        symbol = db.read_array("symbol")
        assert isinstance(symbol, StoredText)
        assert symbol.marker == "Symbol"
        assert symbol.tolist() == ["alpha", "z"]
        assert db.read_array("empty") == []
        raw = db.read_array("raw")
        assert isinstance(raw, StoredText)
        assert raw.values == (b"\xff", b"a")


@NATIVE
@pytest.mark.parametrize(
    ("element", "values"),
    [
        (StoredElement.date(Monthly()), np.array([[-1, 0], [1, 2]], dtype="<i8")),
        (StoredElement.duration(Monthly()), np.array([[1, 2], [3, 4]], dtype="<i8")),
        (
            INT128,
            np.array([1, 0, 2, 0, 3, 0, 4, 0], dtype="<u8").view(INT128.dtype).reshape((2, 2)),
        ),
        (
            UINT128,
            np.array([1, 0, 2, 0, 3, 0, 4, 0], dtype="<u8").view(UINT128.dtype).reshape((2, 2)),
        ),
    ],
)
def test_represented_matrix_round_trip(tmp_path, element, values):
    array = StoredArray(values, element)
    path = tmp_path / "rep.daec"
    with open_dataecon(path, "w") as db:
        db.write_array("m", array)
    with open_dataecon(path) as db:
        back = db.read_array("m")
        assert isinstance(back, StoredArray)
        assert back == array


@NATIVE
def test_represented_vector_is_not_a_dated_series(tmp_path):
    array = StoredArray(np.array([-1, 0], dtype="<i8"), StoredElement.date(Monthly()))
    path = tmp_path / "vec.daec"
    with open_dataecon(path, "w") as db:
        db.write_array("v", array)
    with open_dataecon(path) as db:
        back = db.read_array("v")
        assert isinstance(back, StoredArray)
        assert back.ndim == 1
        assert back.tolist() == [MIT(Monthly(), -1), MIT(Monthly(), 0)]
        assert not hasattr(back, "firstdate")


@NATIVE
def test_structure_marker_survives_a_rewrite(tmp_path):
    array = StoredArray(
        np.array([[1.0, 3.0], [2.0, 4.0]]),
        StoredElement.numeric("<f8"),
        object_marker="Symmetric",
    )
    path = tmp_path / "sym.daec"
    with open_dataecon(path, "w") as db:
        db.write_array("s", array)
    with open_dataecon(path) as db:
        back = db.read_array("s")
    assert back == array
    assert back.object_marker == "Symmetric"
    assert back.to_interpreted().tolist() == [[1.0, 3.0], [3.0, 4.0]]


@NATIVE
def test_matrix_results_own_their_memory(tmp_path):
    path = tmp_path / "own.daec"
    with open_dataecon(path, "w") as db:
        db.write_array("m", np.array([[1.0, 2.0], [3.0, 4.0]]))
    with open_dataecon(path) as db:
        first = db.read_array("m")
        first[0, 0] = 99.0
        second = db.read_array("m")
    assert second[0, 0] == 1.0
    assert first[0, 0] == 99.0


@NATIVE
def test_matrix_overwrite_validates_before_deleting(tmp_path):
    path = tmp_path / "overwrite.daec"
    original = np.array([[1.0, 2.0], [3.0, 4.0]])
    with open_dataecon(path, "w") as db:
        db.write_array("m", original)
        with pytest.raises((TypeError, ValueError)):
            db.write_array("m", np.zeros((2, 2), dtype=np.longdouble), overwrite=True)
        assert same_values(db.read_array("m"), original)
        db.write_array("m", np.array([[5.0, 6.0], [7.0, 8.0]]), overwrite=True)
        assert db.read_array("m")[0, 0] == 5.0


@NATIVE
def test_zero_column_mvtseries_is_refused(tmp_path):
    series = MVTSeries(MIT(Monthly(), 0), (), np.zeros((2, 0)))
    with (
        open_dataecon(tmp_path / "z.daec", "w") as db,
        pytest.raises(ValueError, match="no columns"),
    ):
        db.write_series("m", series)


MATRIX_FAULT = r"""
import sqlite3, sys
import numpy as np
from tsecon.dataecon import open_dataecon

path, stage = sys.argv[1], sys.argv[2]
with open_dataecon(path, "w") as db:
    db.write_array("marked", np.zeros((2, 2), dtype=np.int16))
conn = sqlite3.connect(path)
conn.execute(
    "CREATE TRIGGER fail_attr BEFORE INSERT ON attributes WHEN NEW.name='jeltype' "
    "BEGIN SELECT RAISE(ABORT, 'injected'); END"
)
conn.commit()
conn.close()
try:
    with open_dataecon(path, "a") as db:
        db.write_array("marked", np.zeros((0, 2), dtype=np.int16), overwrite=True)
except Exception as error:
    print("ERROR", type(error).__name__)
conn = sqlite3.connect(path)
rows = conn.execute(
    "SELECT name, value FROM attributes WHERE name IN ('jeltype','jtype') ORDER BY name"
).fetchall()
oid = conn.execute("SELECT id FROM objects WHERE name='marked'").fetchone()
axes = conn.execute(
    "SELECT a1.length, a2.length FROM mvtseries m "
    "JOIN axes a1 ON a1.id=m.axis1_id JOIN axes a2 ON a2.id=m.axis2_id WHERE m.id=?",
    (oid[0],),
).fetchone()
print("ATTRS", rows)
print("SHAPE", axes)
"""


@NATIVE
def test_matrix_attribute_failure_leaves_the_unmarked_object(tmp_path):
    path = tmp_path / "fault.daec"
    run = subprocess.run(
        [sys.executable, "-c", MATRIX_FAULT, str(path), "element"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert "ERROR DataEconError" in run.stdout
    # The overwrite deleted the original and stored the replacement; only the
    # element marker failed, so an unmarked empty int16 matrix remains.
    assert "ATTRS []" in run.stdout
    assert "SHAPE (0, 2)" in run.stdout


# ---- Julia fixture ---------------------------------------------------------


@NATIVE
def test_julia_matrices_text_fixture(tmp_path):
    fixture = FIXTURES / "julia_matrices_text.daec"
    provenance = tomllib.loads(fixture.with_suffix(".toml").read_text(encoding="utf-8"))
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    with closing(sqlite3.connect(fixture)) as conn:
        stored_names = {name for (name,) in conn.execute("SELECT name FROM objects WHERE id != 0")}
    assert len(stored_names) == 68

    rewritten = tmp_path / "rewritten.daec"
    visited: set[str] = set()
    with open_dataecon(fixture) as db, open_dataecon(rewritten, "w") as out:
        for name in sorted(stored_names):
            if name.startswith("mvts_"):
                value = db.read_series(name)
                assert isinstance(value, MVTSeries)
                out.write_series(name, value)
            elif name.startswith("text_"):
                value = db.read_array(name)
                assert isinstance(value, (list, StoredText))
                out.write_array(name, value)
            else:
                value = db.read_array(name)
                assert isinstance(value, (np.ndarray, StoredArray))
                out.write_array(name, value)
            visited.add(name)
    assert visited == stored_names

    with closing(sqlite3.connect(rewritten)) as conn:
        assert {
            name for (name,) in conn.execute("SELECT name FROM objects WHERE id != 0")
        } == stored_names

    # Every rewritten object reads back identically to the fixture's value.
    with open_dataecon(fixture) as db, open_dataecon(rewritten) as out:
        for name in sorted(stored_names):
            reader = "read_series" if name.startswith("mvts_") else "read_array"
            original = getattr(db, reader)(name)
            copy = getattr(out, reader)(name)
            if isinstance(original, np.ndarray):
                assert same_values(copy, original)
            elif isinstance(original, MVTSeries):
                assert copy.firstdate == original.firstdate
                assert tuple(copy.columns) == tuple(original.columns)
                assert same_values(copy.values, original.values)
            else:
                assert copy == original


@NATIVE
def test_julia_fixture_special_matrices_keep_their_markers():
    fixture = FIXTURES / "julia_matrices_text.daec"
    with open_dataecon(fixture) as db:
        for name, token, expected in (
            ("matrix_diagonal", "Diagonal", [[3.0, 0.0], [0.0, 5.0]]),
            ("matrix_symmetric", "Symmetric", [[1.0, 3.0], [3.0, 4.0]]),
        ):
            value = db.read_array(name)
            assert isinstance(value, StoredArray)
            assert value.object_marker == token
            assert value.to_interpreted().tolist() == expected


@NATIVE
def test_julia_fixture_text_values():
    fixture = FIXTURES / "julia_matrices_text.daec"
    with open_dataecon(fixture) as db:
        assert db.read_array("text_ascii") == ["alpha", "", "z"]
        assert db.read_array("text_multibyte") == ["é", "\U0001f642", "aé"]
        assert db.read_array("text_one_empty_string") == [""]
        assert db.read_array("text_empty") == []
        symbol = db.read_array("text_symbol")
        assert isinstance(symbol, StoredText)
        assert symbol.marker == "Symbol"
        assert symbol.tolist() == ["alpha", "", "z"]


@NATIVE
@pytest.mark.parametrize(
    ("name", "statement", "error"),
    [
        ("matrix_Int16", "UPDATE mvtseries SET value=x'00' WHERE id=:id", ValueError),
        (
            "matrix_Int16",
            "UPDATE axes SET ax_type=1 WHERE id=(SELECT axis1_id FROM mvtseries WHERE id=:id)",
            TypeError,
        ),
        ("matrix_Int16", "INSERT INTO attributes VALUES(:id,'jtype','MVTSeries')", TypeError),
        ("matrix_Int16", "INSERT INTO attributes VALUES(:id,'jeltype','Bool')", TypeError),
        ("text_ascii", "UPDATE tseries SET value=x'6100' WHERE id=:id", ValueError),
        ("text_ascii", "INSERT INTO attributes VALUES(:id,'jeltype','Char')", TypeError),
        ("text_ascii", "INSERT INTO attributes VALUES(:id,'jtype','Vector')", TypeError),
    ],
)
def test_malformed_two_dimensional_and_text_objects_are_refused(tmp_path, name, statement, error):
    path = tmp_path / "malformed.daec"
    shutil.copyfile(FIXTURES / "julia_matrices_text.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        (oid,) = conn.execute("SELECT id FROM objects WHERE name=?", (name,)).fetchone()
        conn.execute(statement, {"id": oid})
    with open_dataecon(path) as db, pytest.raises(error):
        db.read_array(name)


@NATIVE
@pytest.mark.parametrize(
    ("statement", "match"),
    [
        (
            "UPDATE axes SET data='a' WHERE id=(SELECT axis2_id FROM mvtseries WHERE id=:id)",
            "1 names for 2 columns",
        ),
        (
            "UPDATE axes SET data='a\na' WHERE id=(SELECT axis2_id FROM mvtseries WHERE id=:id)",
            "duplicate column names",
        ),
    ],
)
def test_malformed_names_axes_are_refused(tmp_path, statement, match):
    path = tmp_path / "names.daec"
    shutil.copyfile(FIXTURES / "julia_matrices_text.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        (oid,) = conn.execute("SELECT id FROM objects WHERE name='mvts_float64'").fetchone()
        conn.execute(statement, {"id": oid})
    with open_dataecon(path) as db, pytest.raises(ValueError, match=match):
        db.read_series("mvts_float64")


# ---- corrections: ownership, carrier validation and exact conversions ------

# Recorded from the pinned Julia reference: the exact little-endian bytes
# `Matrix(Diagonal|Symmetric|Hermitian(A))` produces, including signed zeros,
# NaN payloads and complex conjugation. Stored and dense bytes are column-major.
STRUCTURE_CASES = (
    (
        "negzero_real_Diagonal",
        "<f8",
        2,
        2,
        "Diagonal",
        "0000000000000080000000000000204000000000000000800000000000000080",
        "0000000000000080000000000000000000000000000000000000000000000080",
    ),
    (
        "negzero_real_Symmetric",
        "<f8",
        2,
        2,
        "Symmetric",
        "0000000000000080000000000000204000000000000000800000000000000080",
        "0000000000000080000000000000008000000000000000800000000000000080",
    ),
    (
        "negzero_real_Hermitian",
        "<f8",
        2,
        2,
        "Hermitian",
        "0000000000000080000000000000204000000000000000800000000000000080",
        "0000000000000080000000000000008000000000000000800000000000000080",
    ),
    (
        "mixed_real_Diagonal",
        "<f8",
        2,
        2,
        "Diagonal",
        "000000000000f03f000000000000f87f00000000000004c00000000000001040",
        "000000000000f03f000000000000000000000000000000000000000000001040",
    ),
    (
        "mixed_real_Symmetric",
        "<f8",
        2,
        2,
        "Symmetric",
        "000000000000f03f000000000000f87f00000000000004c00000000000001040",
        "000000000000f03f00000000000004c000000000000004c00000000000001040",
    ),
    (
        "mixed_real_Hermitian",
        "<f8",
        2,
        2,
        "Hermitian",
        "000000000000f03f000000000000f87f00000000000004c00000000000001040",
        "000000000000f03f00000000000004c000000000000004c00000000000001040",
    ),
    (
        "nan_payload_Diagonal",
        "<f8",
        2,
        2,
        "Diagonal",
        "550000000000f87f0000000000000080000000000000f03f010000000000f8ff",
        "550000000000f87f00000000000000000000000000000000010000000000f8ff",
    ),
    (
        "nan_payload_Symmetric",
        "<f8",
        2,
        2,
        "Symmetric",
        "550000000000f87f0000000000000080000000000000f03f010000000000f8ff",
        "550000000000f87f000000000000f03f000000000000f03f010000000000f8ff",
    ),
    (
        "nan_payload_Hermitian",
        "<f8",
        2,
        2,
        "Hermitian",
        "550000000000f87f0000000000000080000000000000f03f010000000000f8ff",
        "550000000000f87f000000000000f03f000000000000f03f010000000000f8ff",
    ),
    (
        "negzero_complex_Diagonal",
        "<c16",
        2,
        2,
        "Diagonal",
        "00000000000000800000000000000080000000000000204000000000000020400000000000000080000000000000008000000000000000800000000000000080",
        "00000000000000800000000000000080000000000000000000000000000000000000000000000000000000000000000000000000000000800000000000000080",
    ),
    (
        "negzero_complex_Symmetric",
        "<c16",
        2,
        2,
        "Symmetric",
        "00000000000000800000000000000080000000000000204000000000000020400000000000000080000000000000008000000000000000800000000000000080",
        "00000000000000800000000000000080000000000000008000000000000000800000000000000080000000000000008000000000000000800000000000000080",
    ),
    (
        "negzero_complex_Hermitian",
        "<c16",
        2,
        2,
        "Hermitian",
        "00000000000000800000000000000080000000000000204000000000000020400000000000000080000000000000008000000000000000800000000000000080",
        "00000000000000800000000000000000000000000000008000000000000000000000000000000080000000000000008000000000000000800000000000000000",
    ),
    (
        "hermitian_complex_Diagonal",
        "<c16",
        2,
        2,
        "Diagonal",
        "000000000000f03f0000000000002240000000000000344000000000000034400000000000000040000000000000f0bf000000000000104000000000000022c0",
        "000000000000f03f00000000000022400000000000000000000000000000000000000000000000000000000000000000000000000000104000000000000022c0",
    ),
    (
        "hermitian_complex_Symmetric",
        "<c16",
        2,
        2,
        "Symmetric",
        "000000000000f03f0000000000002240000000000000344000000000000034400000000000000040000000000000f0bf000000000000104000000000000022c0",
        "000000000000f03f00000000000022400000000000000040000000000000f0bf0000000000000040000000000000f0bf000000000000104000000000000022c0",
    ),
    (
        "hermitian_complex_Hermitian",
        "<c16",
        2,
        2,
        "Hermitian",
        "000000000000f03f0000000000002240000000000000344000000000000034400000000000000040000000000000f0bf000000000000104000000000000022c0",
        "000000000000f03f00000000000000000000000000000040000000000000f03f0000000000000040000000000000f0bf00000000000010400000000000000000",
    ),
)


@pytest.mark.parametrize(
    ("name", "dtype", "rows", "columns", "token", "stored_hex", "dense_hex"), STRUCTURE_CASES
)
def test_structural_conversion_matches_julia_bit_for_bit(
    name, dtype, rows, columns, token, stored_hex, dense_hex
):
    stored = np.frombuffer(bytes.fromhex(stored_hex), dtype=dtype)
    values = stored.reshape((rows, columns), order="F").copy(order="C")
    array = StoredArray(values, StoredElement.numeric(dtype), object_marker=token)
    # Byte equality, not ==: a mirrored -0.0 must not become +0.0 and a NaN
    # must keep its payload.
    assert array.to_interpreted().tobytes(order="F").hex() == dense_hex


def test_symmetric_mirroring_preserves_negative_zero():
    values = np.array([[-0.0, -0.0], [8.0, -0.0]])
    for token in ("Symmetric", "Hermitian"):
        array = StoredArray(values, StoredElement.numeric("<f8"), object_marker=token)
        assert np.signbit(array.to_interpreted()).tolist() == [[True, True], [True, True]]


@pytest.mark.parametrize("token", ["Diagonal", "Symmetric", "Hermitian"])
def test_structural_conversion_works_on_wide_carriers(token):
    words = np.array([1, 0, 2, 0, 3, 0, 4, 0], dtype="<u8").view(INT128.dtype).reshape((2, 2))
    array = StoredArray(words, INT128, object_marker=token)
    expected = {
        "Diagonal": [[1, 0], [0, 4]],
        "Symmetric": [[1, 2], [2, 4]],
        "Hermitian": [[1, 2], [2, 4]],
    }[token]
    assert array.to_interpreted().tolist() == expected


def test_complexf16_hermitian_conjugates_and_clears_the_diagonal():
    parts = (
        np.array([0x3C00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4700, 0x4800], dtype="<u2")
        .view(COMPLEXF16.dtype)
        .reshape((2, 2))
    )
    dense = StoredArray(parts, COMPLEXF16, object_marker="Hermitian").to_interpreted()
    assert dense.values["imag"][0, 0] == 0
    assert dense.values["imag"][1, 1] == 0
    # The mirrored entry is the conjugate of the upper-triangle entry.
    assert dense.values["real"][1, 0] == parts["real"][0, 1]
    assert dense.values["imag"][1, 0] == -parts["imag"][0, 1]


@NATIVE
@pytest.mark.parametrize("shape", [(1, 1), (1, 3), (3, 1), (0, 3), (3, 0), (0, 0), (2, 3)])
def test_every_matrix_shape_reads_back_owning_and_writable(tmp_path, shape):
    values = np.ones(shape, dtype=np.float64)
    path = tmp_path / "own.daec"
    with open_dataecon(path, "w") as db:
        db.write_array("m", values)
    with open_dataecon(path) as db:
        result = db.read_array("m")
    # A Fortran-shaped view of the read-only payload is already C-contiguous
    # whenever a dimension is one or zero, so contiguity alone proves nothing.
    assert result.flags["WRITEABLE"]
    assert result.flags["OWNDATA"]
    assert result.shape == shape
    if result.size:
        result[0, 0] = 7.0
        assert result[0, 0] == 7.0


@NATIVE
@pytest.mark.parametrize("dtype", ["<f8", "?", "<i8"])
def test_singleton_matrix_results_survive_file_closure(tmp_path, dtype):
    values = np.ones((1, 2), dtype=dtype)
    path = tmp_path / "closed.daec"
    with open_dataecon(path, "w") as db:
        db.write_array("m", values)
    with open_dataecon(path) as db:
        result = db.read_array("m")
    result[0, 0] = values.dtype.type(0)
    assert result.flags["OWNDATA"]


@NATIVE
def test_represented_matrix_results_are_writable(tmp_path):
    array = StoredArray(np.array([[1, 2]], dtype="<i8"), StoredElement.date(Monthly()))
    path = tmp_path / "rep.daec"
    with open_dataecon(path, "w") as db:
        db.write_array("m", array)
    with open_dataecon(path) as db:
        back = db.read_array("m")
    assert back.values.flags["WRITEABLE"]
    back.values[0, 0] = 5
    assert back.values[0, 0] == 5


@NATIVE
def test_single_column_mvtseries_values_are_writable(tmp_path):
    series = MVTSeries(MIT(Monthly(), 0), ("a",), np.array([[1.0], [2.0]]))
    path = tmp_path / "mvts.daec"
    with open_dataecon(path, "w") as db:
        db.write_series("m", series)
    with open_dataecon(path) as db:
        back = db.read_series("m")
    assert back.values.flags["WRITEABLE"]
    # The column views share the matrix, so a write through one must land.
    back.a[0] = 9.0
    assert back.values[0, 0] == 9.0


def test_stored_array_never_casts_its_input():
    with pytest.raises(TypeError, match="never converted"):
        StoredArray(np.array([1.9]), StoredElement.numeric("<i8", "Int64"))
    with pytest.raises(TypeError, match="never converted"):
        StoredArray(np.array([1, 2], dtype="<i4"), StoredElement.numeric("<i8", "Int64"))
    with pytest.raises(TypeError, match="never converted"):
        StoredArray(np.array([1, 2], dtype=">i8"), StoredElement.numeric("<i8", "Int64"))


@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("extended", ["longdouble", "clongdouble"])
def test_stored_array_rejects_extended_precision_aliases(copy, extended):
    # Windows aliases these to float64/complex128 by dtype equality while the
    # carrier keeps type code "g"/"G".
    dtype, descriptor = {
        "longdouble": (np.longdouble, "<f8"),
        "clongdouble": (np.clongdouble, "<c16"),
    }[extended]
    values = np.zeros(2, dtype=dtype)
    with pytest.raises(TypeError, match="never converted"):
        StoredArray(values, StoredElement.numeric(descriptor, "Int64"), copy=copy)


def test_stored_array_copies_a_noncontiguous_input_without_casting():
    base = np.arange(8, dtype="<i8").reshape((2, 4))
    view = base[:, ::2]
    assert not view.flags["C_CONTIGUOUS"]
    array = StoredArray(view, StoredElement.numeric("<i8", "Int64"))
    assert array.values.tolist() == view.tolist()
    assert array.values.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError, match="C-contiguous"):
        StoredArray(view, StoredElement.numeric("<i8", "Int64"), copy=False)


def test_stored_array_capacity_is_checked_before_use(monkeypatch):
    monkeypatch.setattr("tsecon.dataecon._interpret.MAX_BYTES", 8)
    with pytest.raises(ValueError):
        StoredArray(np.zeros(4, dtype="<i8"), StoredElement.numeric("<i8", "Int64"))


@NATIVE
@pytest.mark.parametrize("shape", [(2,), (1, 2)])
def test_wide_bool_arrays_preserve_their_storage(tmp_path, shape):
    words = np.array([0, 0, 1, 0], dtype="<u8").view(INT128.dtype).reshape(shape)
    array = StoredArray(words, INT128.with_bool_marker())
    path = tmp_path / "wide.daec"
    with open_dataecon(path, "w") as db:
        db.write_array("wb", array)
    with open_dataecon(path) as db:
        back = db.read_array("wb")
    assert back == array
    assert back.element.julia_name == "Int128"
    assert back.element.marker == "Bool"
    assert back.to_bool().tolist() == np.array([False, True]).reshape(shape).tolist()


@NATIVE
def test_date_bool_marker_preserves_its_carrier(tmp_path):
    element = StoredElement.date(Monthly()).with_bool_marker()
    array = StoredArray(np.array([0, 1], dtype="<i8"), element)
    path = tmp_path / "datebool.daec"
    with open_dataecon(path, "w") as db:
        db.write_array("d", array)
    with open_dataecon(path) as db:
        back = db.read_array("d")
    assert back == array
    assert back.to_bool().tolist() == [False, True]


def test_ordinary_boolean_arrays_still_require_one_byte_storage():
    # An ordinary Bool marker yields Boolean values, which Julia stores as one
    # signed byte; a stored eight-byte payload claiming it is malformed.
    payload = np.array([0, 1], dtype="<i8").tobytes()
    with pytest.raises(TypeError, match="canonical one-byte"):
        validate_array_payload((2, 10, 1, 0, 0, 2, 0, 0, len(payload)), payload, "Bool", None)
    # The same rule does not touch a represented carrier, which preserves its
    # stored width under the wide-Bool policy.
    words = np.array([0, 0, 1, 0], dtype="<u8").tobytes()
    resolved = validate_array_payload((2, 10, 1, 0, 0, 2, 0, 0, len(words)), words, "Bool", None)
    assert isinstance(resolved, StoredElement)
    assert resolved.julia_name == "Int128"


@NATIVE
@pytest.mark.parametrize("shape", [(0,), (0, 2)])
def test_empty_foreign_markers_are_preserved_on_the_kind_default(tmp_path, shape):
    array = StoredArray(np.empty(shape, dtype="<i8"), StoredElement.numeric("<i8", "Float64"))
    path = tmp_path / "emptymark.daec"
    with open_dataecon(path, "w") as db:
        db.write_array("e", array)
    with open_dataecon(path) as db:
        back = db.read_array("e")
    assert back == array
    assert back.element.marker == "Float64"
    interpreted = back.to_interpreted()
    assert interpreted.dtype == np.dtype("<f8")
    assert interpreted.shape == shape


def test_empty_foreign_marker_on_a_nondefault_width_is_refused():
    with pytest.raises(TypeError, match="no stored width"):
        StoredArray(np.empty(0, dtype="<i2"), StoredElement.numeric("<i2", "Float64"))


def test_conversions_use_the_snapshot_shape_not_the_live_one():
    values = np.array([[1, 2], [3, 4]], dtype="<i8")
    array = StoredArray(values, StoredElement.numeric("<i8", "Float64"), copy=False)
    original = _interpret.convert_values

    def reshape_then_convert(snapshot, source, target):
        # Swap the live carrier for a differently shaped view of the same
        # buffer, the way a concurrent caller would. Assigning to ndarray.shape
        # would do the same but is deprecated from NumPy 2.5.
        array._values = array._values.reshape(-1)
        return original(snapshot, source, target)

    _interpret.convert_values = reshape_then_convert
    try:
        assert array.to_interpreted().shape == (2, 2)
    finally:
        _interpret.convert_values = original


class _ShrinkingArray(np.ndarray):
    """An ndarray whose bytes shrink mid-call, standing in for a racing writer."""

    def tobytes(self, order="C"):
        return super().tobytes(order)[:8]


def test_a_snapshot_size_change_is_refused():
    values = np.array([1, 2, 3, 4], dtype="<i8").view(_ShrinkingArray)
    array = StoredArray(values, StoredElement.numeric("<i8", "Float64"), copy=False)
    with pytest.raises(ValueError, match="changed size"):
        array.to_interpreted()


@NATIVE
def test_nonempty_string_marker_is_preserved(tmp_path):
    path = tmp_path / "marked.daec"
    with open_dataecon(path, "w") as db:
        db.write_array("marked", StoredText((b"a", b"bc"), "String"))
        db.write_array("plain", ["a", "bc"])
        db.write_array("empty", [])
    with open_dataecon(path) as db:
        marked = db.read_array("marked")
        plain = db.read_array("plain")
        empty = db.read_array("empty")
    # A nonempty "String" token is a marker no Julia writer produces, so it is
    # kept literally; the empty vector's token is Julia's own and is canonical.
    assert isinstance(marked, StoredText)
    assert marked.marker == "String"
    assert encode_array(marked).marker == "String"
    assert plain == ["a", "bc"]
    assert encode_array(plain).marker is None
    assert empty == []
    assert encode_array(empty).marker == "String"


def test_text_capacity_is_enforced_before_the_payload_is_built(monkeypatch):
    monkeypatch.setattr("tsecon.dataecon._arrays.MAX_BYTES", 16)
    with pytest.raises(ValueError, match="above the"):
        encode_array(["aaaaaaaa", "bbbbbbbb", "cccccccc"])
    with pytest.raises(ValueError, match="above the"):
        encode_array(StoredText((b"a" * 8, b"b" * 8, b"c" * 8)))
    assert len(encode_array(["abc", "de"]).payload) == 7
