# SPDX-License-Identifier: MIT
"""N-dimensional plain arrays: three to five plain axes through write_array/read_array."""

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

from tsecon import MVTSeries, mm
from tsecon.dataecon import (
    COMPLEXF16,
    INT128,
    DataEconError,
    StoredArray,
    StoredElement,
    _codec,
    open_dataecon,
    open_dataecon_memory,
)
from tsecon.dataecon._codec import (
    MAX_AXES,
    TensorPayload,
    decode_array,
    encode_array,
    tensor_metadata,
    validate_tensor_metadata,
    validate_tensor_payload,
)
from tsecon.dataecon._interpret import array_object_interpretation, numeric_target
from tsecon.frequencies import Monthly, Quarterly, Unit

FIXTURES = Path(__file__).parent / "fixtures"
NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built; configure TSECON_DATAECON_ROOT",
)
TENSOR_DTYPES = [
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
SHAPES = [(1, 2, 3), (3, 1, 2, 1, 1), (2, 3, 1), (1, 1, 6, 1)]


def sample_values(dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    """Six values, including signed zeros, NaN payloads and both extremes, in any shape."""
    kind = np.dtype(dtype).kind
    if kind == "b":
        flat = np.array([False, True, False, True, True, False])
    elif kind == "i":
        info = np.iinfo(dtype)
        flat = np.array([info.min, -1, 0, info.max, 7, 9], dtype=dtype)
    elif kind == "u":
        info = np.iinfo(dtype)
        flat = np.array([0, 1, info.max, 7, 9, 11], dtype=dtype)
    elif kind == "f":
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
    else:
        component = "<f4" if np.dtype(dtype).itemsize == 8 else "<f8"
        flat = np.empty(6, dtype=dtype)
        flat.real = np.array([-0.0, 2.5, 0.0, 1.0, 3.0, 5.0], dtype=component)
        flat.imag = np.array([1.25, -3.0, -0.0, 2.0, 4.0, 6.0], dtype=component)
    return np.asarray(flat).reshape(shape, order="F").copy()


def same_values(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare exact stored bytes, so signed zeros and NaN payloads must match."""
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def run(shape: tuple[int, ...]) -> np.ndarray:
    return np.arange(1, 25, dtype="<i8").reshape(shape, order="F").copy()


def metadata_with(shape: tuple[int, ...], nbytes: int, **header: int) -> tuple[int, ...]:
    values = list(tensor_metadata(header.pop("element", 1), header.pop("elfreq", 0), shape, nbytes))
    for key, value in header.items():
        values[{"cls": 0, "obj_type": 1, "naxes": 4}[key]] = value
    return tuple(values)


# ---- pure codec and container behavior ------------------------------------


@pytest.mark.parametrize("dtype", TENSOR_DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_tensor_payload_is_column_major_and_round_trips(dtype, shape):
    values = sample_values(dtype, shape)
    encoded = encode_array(values)
    assert isinstance(encoded, TensorPayload)
    assert encoded.object_type == 30
    assert encoded.shape == shape
    stored = values.astype(np.int8) if values.dtype.kind == "b" else values
    assert encoded.payload == stored.tobytes(order="F")
    assert encoded.marker == ("Bool" if dtype == "?" else None)
    back = decode_array(
        tensor_metadata(encoded.element, 0, shape, len(encoded.payload)),
        encoded.payload,
        encoded.marker,
        None,
    )
    assert same_values(back, values)
    assert back.flags["C_CONTIGUOUS"]
    assert back.flags["OWNDATA"]
    assert back.flags["WRITEABLE"]


def test_one_run_stores_identical_bytes_at_every_shape():
    # The reference's 2x3x4, 4x3x2, 2x2x2x3 and 2x2x2x3x1 tensors of 1..24
    # all store the run itself; only the axes distinguish them.
    reference = b"".join(i.to_bytes(8, "little") for i in range(1, 25))
    for shape in ((2, 3, 4), (4, 3, 2), (2, 2, 2, 3), (2, 2, 2, 3, 1)):
        assert encode_array(run(shape)).payload == reference
    cube = run((2, 3, 4))
    assert cube[1, 0, 0] == 2
    assert cube[0, 1, 0] == 3
    assert cube[0, 0, 1] == 7


def test_any_input_layout_snapshots_logical_values():
    base = np.arange(48, dtype="<i8").reshape((2, 4, 6))
    for view in (base[:, ::2, ::3], np.asfortranarray(base), base.transpose(2, 0, 1)):
        encoded = encode_array(view)
        back = decode_array(
            tensor_metadata(1, 0, encoded.shape, len(encoded.payload)),
            encoded.payload,
            None,
            None,
        )
        assert same_values(back, np.ascontiguousarray(view))


@pytest.mark.parametrize("shape", [(0, 2, 3), (2, 0, 3), (2, 3, 0), (0, 0, 0, 0, 0), (0, 1, 1, 1)])
def test_empty_tensors_keep_their_shape(shape):
    encoded = encode_array(np.empty(shape, dtype=np.float64))
    assert encoded.payload == b""
    assert encoded.marker is None
    back = decode_array(tensor_metadata(4, 0, shape, 0), b"", None, None)
    assert back.shape == shape
    assert back.dtype == np.dtype("<f8")
    assert back.flags["OWNDATA"]
    # Non-default empties carry Julia's element token, as vectors and matrices do.
    assert encode_array(np.empty(shape, dtype="<i2")).marker == "Int16"
    assert encode_array(np.empty(shape, dtype=bool)).marker == "Bool"


@pytest.mark.parametrize("ndim", [0, 6, 7])
def test_unsupported_ranks_are_refused_before_any_native_call(ndim):
    with pytest.raises(ValueError, match=f"{MAX_AXES}"):
        encode_array(np.zeros((1,) * ndim, dtype=np.float64))


def test_extended_precision_is_refused_at_every_rank():
    with pytest.raises(TypeError):
        encode_array(np.zeros((1, 1, 2), dtype=np.longdouble))
    with pytest.raises(TypeError):
        encode_array(np.zeros((1, 1, 2), dtype=np.clongdouble))


@pytest.mark.parametrize(
    ("metadata", "error", "match"),
    [
        (metadata_with((2, 2, 2), 64, cls=3), TypeError, "N-dimensional support"),
        (metadata_with((2, 2, 2), 64, obj_type=31), TypeError, "native capacity"),
        (metadata_with((2, 2, 2), 64, obj_type=32), TypeError, "native capacity"),
        (metadata_with((2, 2, 2), 64, element=6, elfreq=32), TypeError, "no element frequency"),
        (
            metadata_with((2, 2, 2), 64, element=4, elfreq=32),
            TypeError,
            "element frequency or kind",
        ),
        (metadata_with((2, 2, 2), 63), ValueError, "element width"),
        (metadata_with((2, 2, 2), 3), ValueError, "element width"),
        (metadata_with((0, 2, 2), 8), ValueError, "oversized"),
        (metadata_with((1 << 40, 1 << 40, 1), 8), ValueError, "signed 64-bit"),
        (metadata_with((1 << 31, 1 << 32, 1), 8), ValueError, "signed 64-bit"),
        (metadata_with((1 << 30, 1 << 30, 2), 8), ValueError, "element width"),
    ],
)
def test_malformed_tensor_metadata_is_refused(metadata, error, match):
    with pytest.raises(error, match=match):
        validate_tensor_metadata(metadata)


@pytest.mark.parametrize("naxes", [0, 1, 2])
def test_class_four_objects_with_fewer_than_three_axes_are_refused(naxes):
    # Native capacity: Julia's loader would return a 0-d array, vector or
    # matrix, but Julia's writer never produces these encodings.
    values = list(tensor_metadata(1, 0, (2,) * 3, 64))
    values[4] = naxes
    for index in range(naxes, 3):
        values[6 + 5 * index : 11 + 5 * index] = (-1, 0, 0, 0, 0)
    with pytest.raises(TypeError, match=f"{naxes} axes"):
        validate_tensor_metadata(tuple(values))


def test_gap_and_dangling_axis_slots_are_refused():
    # The native loader reports a missing ndaxes row as id -1 and a dangling
    # axis id as id 0; Julia's `if ax.id > 0` filter then silently returns a
    # lower-rank array. Both are refused by name.
    for bad_id in (-1, 0):
        values = list(tensor_metadata(1, 0, (2, 4, 1), 64))
        values[6 + 5 * 1 : 11 + 5 * 1] = (bad_id, 0, 0, 0, 0)
        with pytest.raises(TypeError, match="axis 1 is missing"):
            validate_tensor_metadata(tuple(values))
    values = list(tensor_metadata(1, 0, (2, 2, 2), 64))
    values[6 + 5 * 3] = 7
    with pytest.raises(TypeError, match="beyond its axis count"):
        validate_tensor_metadata(tuple(values))
    values = list(tensor_metadata(1, 0, (2, 2, 2), 64))
    values[6 + 5 * 1 + 1] = 1
    with pytest.raises(TypeError, match="plain axes only"):
        validate_tensor_metadata(tuple(values))


def test_product_overflow_is_checked_in_python_integers_before_the_payload():
    # 2^40 * 2^40 wraps to zero in 64-bit arithmetic; here the product is exact.
    metadata = tensor_metadata(1, 0, (1 << 40, 1 << 40, 1), 0)
    with pytest.raises(ValueError, match="signed 64-bit"):
        validate_tensor_payload(metadata, b"", None, None)


def test_stored_array_carries_ranks_three_to_five():
    codes = np.arange(6, dtype="<i8").reshape((1, 2, 3, 1, 1))
    array = StoredArray(codes, StoredElement.date(Monthly()))
    assert array.ndim == 5
    assert array.shape == (1, 2, 3, 1, 1)
    listed = array.tolist()
    assert listed[0][1][2][0][0].value == 5
    assert encode_array(array).shape == (1, 2, 3, 1, 1)
    with pytest.raises(ValueError, match=f"one to {MAX_AXES}"):
        StoredArray(np.zeros((1,) * 6, dtype="<i8"), StoredElement.date(Monthly()))


def test_represented_tensor_round_trips_through_the_codec():
    for element, values in (
        (StoredElement.date(Unit()), np.array([-(2**63), 2**63 - 1]).reshape((1, 2, 1))),
        (
            StoredElement.duration(Quarterly(1)),
            np.array([1, -2, 3, 0], dtype="<i8").reshape((1, 2, 2)),
        ),
        (
            INT128,
            np.array([1, 0, 2, 0, 3, 0, 4, 0], dtype="<u8").view(INT128.dtype).reshape((2, 1, 2)),
        ),
        (
            COMPLEXF16,
            np.array([0x8000, 0x3D00, 0x4000, 0x4200], dtype="<u2")
            .view(COMPLEXF16.dtype)
            .reshape((1, 1, 2)),
        ),
    ):
        array = StoredArray(values, element)
        encoded = encode_array(array)
        assert encoded.element == element.native_kind
        assert encoded.element_frequency == element.native_frequency
        back = decode_array(
            tensor_metadata(
                encoded.element, encoded.element_frequency, encoded.shape, len(encoded.payload)
            ),
            encoded.payload,
            encoded.marker,
            None,
        )
        assert back == array
        assert back.values.flags["OWNDATA"]
        assert back.values.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize(
    ("token", "ndim", "kind"),
    [
        ("Array", 3, "identity"),
        ("AbstractArray", 5, "identity"),
        ("Any", 4, "identity"),
        ("Array{Int64,3}", 3, "identity"),
        ("Array{Int64, 4}", 4, "identity"),
        ("Array{Float64,3}", 3, "element"),
        ("Array{MIT{Monthly},5}", 5, "element"),
        ("Array{Bool,3}", 3, "element"),
        ("BitArray{3}", 3, "element"),
        ("BitArray{5}", 5, "element"),
        ("BitVector", 1, "element"),
        ("BitMatrix", 2, "element"),
    ],
)
def test_supported_tensor_object_markers(token, ndim, kind):
    assert array_object_interpretation(token, numeric_target(np.dtype("<i8")), ndim)[0] == kind


@pytest.mark.parametrize(
    ("token", "ndim"),
    [
        ("Vector", 3),
        ("Matrix", 3),
        ("Array{Int64,2}", 3),
        ("Array{Int64,4}", 3),
        ("Array{Any,3}", 3),
        ("Diagonal", 3),
        ("Symmetric", 4),
        ("Hermitian", 5),
        ("Symbol", 3),
        ("TSeries", 3),
        ("BitArray", 3),
        ("BitArray{2}", 3),
        ("BitVector", 2),
        ("BitMatrix", 3),
        ("Array", 6),
    ],
)
def test_unsupported_tensor_object_markers_are_refused(token, ndim):
    with pytest.raises(TypeError):
        array_object_interpretation(token, numeric_target(np.dtype("<i8")), ndim)


def test_structure_markers_stay_two_dimensional():
    cube = np.arange(8, dtype="<i8").reshape((2, 2, 2))
    for token in ("Diagonal", "Symmetric", "Hermitian"):
        with pytest.raises(TypeError, match="3-dimensional"):
            StoredArray(cube, StoredElement.numeric(np.dtype("<i8"), None), object_marker=token)


def test_bit_array_tokens_require_exact_zero_and_one():
    carrier = StoredElement.numeric(np.dtype("i1"), "Bool")
    good = StoredArray(
        np.array([0, 1, 1, 0], dtype="i1").reshape((1, 2, 2)), carrier, object_marker="BitArray{3}"
    )
    assert good.to_interpreted().dtype == np.dtype(bool)
    assert good.to_interpreted().tolist() == [[[False, True], [True, False]]]
    with pytest.raises(ValueError, match="exactly zero or one"):
        StoredArray(
            np.array([0, 2, 1, 0], dtype="i1").reshape((1, 2, 2)),
            carrier,
            object_marker="BitArray{3}",
        )


def test_conversion_uses_the_snapshot_shape_at_rank_three():
    values = np.arange(8, dtype="<i8").reshape((2, 2, 2))
    array = StoredArray(values, StoredElement.numeric(np.dtype("<i8"), "Float64"))
    converted = array.to_interpreted()
    assert converted.shape == (2, 2, 2)
    assert converted.dtype == np.dtype("<f8")
    assert converted[1, 0, 1] == values[1, 0, 1]


# ---- native round trips ----------------------------------------------------


@NATIVE
@pytest.mark.parametrize("dtype", TENSOR_DTYPES)
def test_tensor_round_trip_through_a_file(tmp_path, dtype):
    path = tmp_path / "tensor.daec"
    with open_dataecon(path, "w") as db:
        for index, shape in enumerate(SHAPES):
            db.write_array(f"t{index}", sample_values(dtype, shape))
    with open_dataecon(path) as db:
        for index, shape in enumerate(SHAPES):
            back = db.read_array(f"t{index}")
            assert same_values(back, sample_values(dtype, shape))
            assert back.flags["OWNDATA"]
            assert back.flags["WRITEABLE"]


@NATIVE
@pytest.mark.parametrize(
    "shape", [(1, 1, 1), (1, 3, 1), (0, 2, 3), (2, 0, 3), (0, 0, 0, 0, 0), (1, 1, 1, 1, 1)]
)
def test_every_tensor_shape_reads_back_owning_and_writable(tmp_path, shape):
    path = tmp_path / "own.daec"
    with open_dataecon(path, "w") as db:
        db.write_array("m", np.ones(shape, dtype=np.float64))
    with open_dataecon(path) as db:
        result = db.read_array("m")
    assert result.shape == shape
    assert result.flags["WRITEABLE"]
    assert result.flags["OWNDATA"]
    if result.size:
        result[(0,) * len(shape)] = 7.0
        assert result[(0,) * len(shape)] == 7.0


@NATIVE
def test_tensor_and_series_readers_name_each_other(tmp_path):
    path = tmp_path / "which.daec"
    with open_dataecon(path, "w") as db:
        db.write_array("cube", run((2, 3, 4)))
    with open_dataecon(path) as db, pytest.raises(TypeError, match="read_array"):
        db.read_series("cube")


@NATIVE
def test_represented_tensors_round_trip_through_a_file(tmp_path):
    path = tmp_path / "rep.daec"
    arrays = {
        "mit": StoredArray(
            np.array([-1, 0, 1, 2], dtype="<i8").reshape((2, 1, 2)), StoredElement.date(Monthly())
        ),
        "unit5": StoredArray(
            np.array([-(2**63), 2**63 - 1], dtype="<i8").reshape((1, 2, 1, 1, 1)),
            StoredElement.date(Unit()),
        ),
        "words": StoredArray(
            np.array([1, 0, 2, 0, 3, 0, 4, 0], dtype="<u8").view(INT128.dtype).reshape((2, 1, 2)),
            INT128,
        ),
        "wide_bool": StoredArray(
            np.array([0, 0, 1, 0], dtype="<u8").view(INT128.dtype).reshape((1, 2, 1)),
            INT128.with_bool_marker(),
        ),
        "marked": StoredArray(run((2, 3, 4)), StoredElement.numeric(np.dtype("<i8"), "Float64")),
        "object": StoredArray(
            run((2, 3, 4)),
            StoredElement.numeric(np.dtype("<i8"), None),
            object_marker="Array{Float64,3}",
        ),
        "bits": StoredArray(
            np.array([0, 1, 1, 0], dtype="i1").reshape((1, 2, 2)),
            StoredElement.numeric(np.dtype("i1"), "Bool"),
            object_marker="BitArray{3}",
        ),
    }
    with open_dataecon(path, "w") as db:
        for name, array in arrays.items():
            db.write_array(name, array)
    with open_dataecon(path) as db:
        for name, array in arrays.items():
            back = db.read_array(name)
            assert back == array, name
            assert back.values.flags["OWNDATA"]
            assert back.values.flags["WRITEABLE"]
    assert arrays["wide_bool"].to_bool().tolist() == [[[False], [True]]]
    assert arrays["marked"].to_interpreted().dtype == np.dtype("<f8")
    assert arrays["bits"].to_interpreted().tolist() == [[[False, True], [True, False]]]


@NATIVE
def test_tensor_overwrite_validates_before_deleting(tmp_path):
    path = tmp_path / "overwrite.daec"
    with open_dataecon(path, "w") as db:
        db.write_array("t", run((2, 3, 4)))
        with pytest.raises(ValueError):
            db.write_array("t", np.zeros((1,) * 6), overwrite=True)
        with pytest.raises(TypeError):
            db.write_array("t", np.zeros((1, 1, 2), dtype=np.longdouble), overwrite=True)
        assert same_values(db.read_array("t"), run((2, 3, 4)))
        db.write_array("t", run((4, 3, 2)), overwrite=True)
        assert db.read_array("t").shape == (4, 3, 2)


class _ResizingArray(np.ndarray):
    """An ndarray that shrinks its last dimension while its bytes are taken.

    This reproduces, without threads, an input that changes size between the
    encoder's metadata capture and its snapshot: the bytes come back shorter,
    and a comparison against the *live* size would accept them and store the
    same native kind at a narrower width under the original shape.
    """

    def tobytes(self, order="C"):
        self.resize((*self.shape[:-1], 1), refcheck=False)
        return super().tobytes(order=order)


def _resizing(shape: tuple[int, ...]) -> np.ndarray:
    values = np.ndarray.__new__(_ResizingArray, shape, dtype=np.int64)
    values[...] = 7
    return values


@pytest.mark.parametrize("shape", [(2,), (1, 2), (1, 1, 2), (1, 1, 1, 2)])
def test_a_snapshot_that_shrinks_is_refused_at_every_rank(shape):
    # Every ordinary path fixes its expected byte count before the snapshot.
    with pytest.raises(ValueError, match="changed size during the snapshot"):
        encode_array(_resizing(shape))


def test_mvtseries_snapshot_size_is_fixed_before_the_bytes_are_taken(monkeypatch):
    series = MVTSeries(mm(2024, 1), ("a", "b"), np.array([[7, 9]], dtype=np.int64), copy=False)
    original = _codec._ordinary_payload

    def shrink_then_snapshot(values, order):
        # A concurrent writer shrinking the shared matrix mid-call.
        values.resize((1, 1), refcheck=False)
        return original(values, order)

    monkeypatch.setattr(_codec, "_ordinary_payload", shrink_then_snapshot)
    with pytest.raises(ValueError, match="changed size during the snapshot"):
        _codec.encode_mvtseries(series)


@NATIVE
@pytest.mark.parametrize("shape", [(2,), (1, 2), (1, 1, 2)])
def test_a_shrinking_overwrite_leaves_the_original_intact(tmp_path, shape):
    path = tmp_path / "shrink.daec"
    original = np.full(shape, 9, dtype=np.int64)
    with open_dataecon(path, "w") as db:
        db.write_array("x", original)
        with pytest.raises(ValueError, match="changed size"):
            db.write_array("x", _resizing(shape), overwrite=True)
        back = db.read_array("x")
    assert back.dtype == np.dtype("<i8")
    assert back.tolist() == original.tolist()


TENSOR_FAULT = """
import sqlite3, sys
import numpy as np
from tsecon.dataecon import open_dataecon

path = sys.argv[1]
with open_dataecon(path, "w") as db:
    db.write_array("marked", np.zeros((2, 2, 2), dtype=np.int16))
conn = sqlite3.connect(path)
conn.execute(
    "CREATE TRIGGER fail_attr BEFORE INSERT ON attributes WHEN NEW.name='jeltype' "
    "BEGIN SELECT RAISE(ABORT, 'injected'); END"
)
conn.commit()
conn.close()
try:
    with open_dataecon(path, "a") as db:
        db.write_array("marked", np.zeros((0, 2, 2), dtype=np.int16), overwrite=True)
except Exception as error:
    print("ERROR", type(error).__name__)
conn = sqlite3.connect(path)
rows = conn.execute(
    "SELECT name, value FROM attributes WHERE name IN ('jeltype','jtype') ORDER BY name"
).fetchall()
oid = conn.execute("SELECT id FROM objects WHERE name='marked'").fetchone()
dims = conn.execute(
    "SELECT a.length FROM ndaxes n JOIN axes a ON a.id=n.axis_id WHERE n.obj_id=? "
    "ORDER BY n.axis_index",
    (oid[0],),
).fetchall()
print("ATTRS", rows)
print("SHAPE", tuple(d for (d,) in dims))
"""


@NATIVE
def test_tensor_attribute_failure_leaves_the_unmarked_object(tmp_path):
    path = tmp_path / "fault.daec"
    result = subprocess.run(
        [sys.executable, "-c", TENSOR_FAULT, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "ERROR DataEconError" in result.stdout
    # The overwrite deleted the original and stored the replacement; only the
    # element marker failed, so an unmarked empty int16 tensor remains.
    assert "ATTRS []" in result.stdout
    assert "SHAPE (0, 2, 2)" in result.stdout


# ---- Julia fixture ---------------------------------------------------------


@NATIVE
def test_julia_tensors_fixture(tmp_path):
    fixture = FIXTURES / "julia_tensors.daec"
    provenance = tomllib.loads(fixture.with_suffix(".toml").read_text(encoding="utf-8"))
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    assert provenance["layout"]["ndtseries_t"] == {
        "size": 304,
        "offsets": [0, 32, 36, 40, 48, 288, 296],
    }
    with closing(sqlite3.connect(fixture)) as conn:
        stored_names = {name for (name,) in conn.execute("SELECT name FROM objects WHERE id != 0")}
    assert len(stored_names) == 70

    rewritten = tmp_path / "rewritten.daec"
    visited: set[str] = set()
    with open_dataecon(fixture) as db, open_dataecon(rewritten, "w") as out:
        for name in sorted(stored_names):
            value = db.read_array(name)
            assert isinstance(value, (np.ndarray, StoredArray))
            out.write_array(name, value)
            visited.add(name)
    assert visited == stored_names
    with closing(sqlite3.connect(rewritten)) as conn:
        assert {
            name for (name,) in conn.execute("SELECT name FROM objects WHERE id != 0")
        } == stored_names
    with open_dataecon(fixture) as db, open_dataecon(rewritten) as out:
        for name in sorted(stored_names):
            original = db.read_array(name)
            copy = out.read_array(name)
            if isinstance(original, np.ndarray):
                assert same_values(copy, original)
            else:
                assert copy == original


@NATIVE
def test_julia_fixture_tensor_values():
    with open_dataecon(FIXTURES / "julia_tensors.daec") as db:
        assert same_values(db.read_array("tensor_run_2x3x4"), run((2, 3, 4)))
        assert same_values(db.read_array("tensor_run_2x2x2x3x1"), run((2, 2, 2, 3, 1)))
        for dtype in TENSOR_DTYPES:
            token = {"?": "Bool"}.get(dtype)
            if token is None:
                token = np.dtype(dtype).name.title().replace("Uint", "UInt")
                token = {"Complex64": "ComplexF32", "Complex128": "ComplexF64"}.get(token, token)
            assert same_values(db.read_array(f"tensor_{token}"), sample_values(dtype, (1, 2, 3)))
            assert same_values(
                db.read_array(f"tensor5_{token}"), sample_values(dtype, (3, 1, 2, 1, 1))
            )
            empty = db.read_array(f"tensor_{token}_empty")
            assert empty.shape == (0, 2, 3)
            assert empty.dtype == np.dtype(dtype)
        assert db.read_array("tensor_empty_rank5").shape == (0, 0, 0, 0, 0)
        assert db.read_array("tensor_empty_rank4_bool").dtype == np.dtype(bool)
        mit = db.read_array("tensor_mit")
        assert isinstance(mit, StoredArray)
        assert mit.element.julia_name == "MIT{Monthly}"
        assert [m.value for m in mit.tolist()[0][0]] == [-1, 1]
        unit = db.read_array("tensor_mit_unit5")
        assert unit.shape == (1, 2, 1, 1, 1)
        assert unit.values.reshape(-1).tolist() == [
            -(2**63),
            2**63 - 1,
        ]
        bits = db.read_array("tensor_bitarray")
        assert isinstance(bits, StoredArray)
        assert bits.object_marker == "BitArray{3}"
        assert same_values(bits.to_interpreted(), sample_values("?", (1, 2, 3)))
        vector = db.read_array("bit_vector")
        assert vector.object_marker == "BitVector"
        assert vector.to_interpreted().tolist() == [
            True,
            False,
            True,
        ]
        matrix = db.read_array("bit_matrix")
        assert matrix.object_marker == "BitMatrix"
        assert matrix.to_interpreted().tolist() == [[True, False], [False, True]]
        for name in ("tensor_marked_float64", "tensor_object_float64"):
            value = db.read_array(name)
            assert value.to_interpreted().dtype == np.dtype("<f8")
            assert same_values(
                value.to_interpreted(), np.arange(1, 9, dtype="<f8").reshape((2, 2, 2), order="F")
            )
        assert db.read_array("tensor_object_identity").active_marker == "Array"
        assert db.read_array("tensor_marked_bool_wide").to_bool().tolist() == [[[False], [True]]]


@NATIVE
@pytest.mark.parametrize(
    ("name", "statements", "error", "match"),
    [
        (
            "tensor_Int16",
            ["UPDATE ndtseries SET value=x'00' WHERE id=:id"],
            ValueError,
            "element width",
        ),
        (
            "tensor_Int16",
            ["DELETE FROM ndaxes WHERE obj_id=:id AND axis_index=1"],
            TypeError,
            "axis 1 is missing",
        ),
        (
            "tensor_Int16",
            ["UPDATE ndaxes SET axis_id=999999 WHERE obj_id=:id AND axis_index=1"],
            TypeError,
            "axis 1 is missing",
        ),
        (
            "tensor_Int16",
            ["DELETE FROM ndaxes WHERE obj_id=:id AND axis_index>0"],
            TypeError,
            "1 axes",
        ),
        (
            "tensor_Int16",
            [
                "UPDATE ndaxes SET axis_id="
                "(SELECT id FROM axes WHERE ax_type=0 AND length=1099511627776) "
                "WHERE obj_id=:id AND axis_index<2"
            ],
            ValueError,
            "signed 64-bit",
        ),
        ("tensor_Int16", ["UPDATE objects SET type=31 WHERE id=:id"], TypeError, "native capacity"),
        (
            "tensor_Int16",
            ["INSERT INTO attributes VALUES(:id,'jtype','Matrix')"],
            TypeError,
            "Unsupported",
        ),
        (
            "tensor_Int16",
            ["INSERT INTO attributes VALUES(:id,'jtype','Diagonal')"],
            TypeError,
            "Unsupported",
        ),
        (
            "tensor_Int16",
            ["INSERT INTO attributes VALUES(:id,'jeltype','Bool')"],
            TypeError,
            "one-byte",
        ),
        # Numeric bytes relabelled as text carry the wrong number of terminators.
        (
            "tensor_Int64",
            ["UPDATE ndtseries SET eltype=6 WHERE id=:id"],
            ValueError,
            "NUL terminators",
        ),
        (
            "tensor_Float64",
            ["UPDATE ndtseries SET elfreq=32 WHERE id=:id"],
            TypeError,
            "element frequency or kind",
        ),
        # A sixth slot is refused by the native loader itself (DE_BAD_NUM_AXES).
        (
            "tensor_run_2x3x4",
            ["INSERT INTO ndaxes VALUES(:id, 5, 1)"],
            DataEconError,
            "number of axes",
        ),
    ],
)
def test_malformed_tensor_rows_are_refused(tmp_path, name, statements, error, match):
    path = tmp_path / "malformed.daec"
    shutil.copyfile(FIXTURES / "julia_tensors.daec", path)
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        (oid,) = conn.execute("SELECT id FROM objects WHERE name=?", (name,)).fetchone()
        conn.execute(
            "INSERT OR IGNORE INTO axes (ax_type, length, frequency, data) "
            "VALUES (0, 1099511627776, 0, NULL)"
        )
        for statement in statements:
            conn.execute(statement, {"id": oid})
    with open_dataecon(path) as db, pytest.raises(error, match=match):
        db.read_array(name)


@NATIVE
def test_native_ndtseries_layout_matches_the_julia_measurement():
    # Opening a database first registers the DLL directory on Windows; the
    # extension is then importable by name without touching private loaders.
    with open_dataecon_memory():
        pass
    layout = importlib.import_module("tsecon.dataecon._native").abi_layout()
    assert layout["max_axes"] == 5
    assert layout["ndtseries_t"] == (304, (0, 32, 36, 40, 48, 288, 296))
    assert layout["mvtseries_t"] == (152, (0, 32, 36, 40, 88, 136, 144))
