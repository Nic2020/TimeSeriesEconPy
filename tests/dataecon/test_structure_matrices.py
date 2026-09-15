# SPDX-License-Identifier: MIT
"""Constructing Diagonal/Symmetric/Hermitian-marked matrices from Python."""

from __future__ import annotations

import importlib.util
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from tsecon.dataecon import (
    COMPLEXF16,
    INT128,
    StoredArray,
    StoredElement,
    _interpret,
    open_dataecon,
)
from tsecon.dataecon._arrays import structure_dense
from tsecon.dataecon._codec import encode_array
from tsecon.frequencies import Monthly

FIXTURES = Path(__file__).parent / "fixtures"
NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built; configure TSECON_DATAECON_ROOT",
)
A = np.array([[1.0, 3.0], [20.0, 4.0]])
H = np.array([[1 + 9j, 2 - 1j], [20 + 20j, 4 - 9j]])
# [-0.0 NaN(payload 0x55); 8.0 -0.0], the D39 sign/payload matrix.
SIGNS = np.frombuffer(
    np.array(
        [0x8000000000000000, 0x7FF8000000000055, 0x4020000000000000, 0x8000000000000000],
        dtype="<u8",
    ).tobytes(),
    dtype="<f8",
).reshape((2, 2))
CSIGNS = np.array(
    [[complex(-0.0, -0.0), complex(-0.0, 7.0)], [complex(8.0, 8.0), complex(-0.0, -0.0)]]
)


def bits(array: np.ndarray) -> list[int]:
    return array.view(f"<u{array.dtype.itemsize}").reshape(-1).tolist()


def cf16(pairs: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.empty(len(pairs), dtype=COMPLEXF16.dtype)
    out["real"] = np.array([r for r, _ in pairs], dtype="<u2").view(np.float16)
    out["imag"] = np.array([i for _, i in pairs], dtype="<u2").view(np.float16)
    return out.reshape(shape)


# ---- materialisation rules -------------------------------------------------


def test_symmetric_uses_the_named_triangle_and_keeps_the_diagonal():
    upper = StoredArray.symmetric(A)
    lower = StoredArray.symmetric(A, "L")
    assert upper.object_marker == "Symmetric"
    assert upper.element == StoredElement.numeric("<f8")
    assert upper.values.tolist() == [[1.0, 3.0], [3.0, 4.0]]
    assert lower.values.tolist() == [[1.0, 20.0], [20.0, 4.0]]
    assert A.tolist() == [[1.0, 3.0], [20.0, 4.0]]  # the input is untouched
    assert upper.values.flags.c_contiguous
    assert upper.values.flags.owndata
    # Julia's loader mirrors the stored upper triangle: the dense form is a fixed point.
    for built in (upper, lower):
        assert built.to_interpreted().tolist() == built.values.tolist()
        assert encode_array(built).object_marker == "Symmetric"
        assert encode_array(built).marker is None


def test_hermitian_conjugates_the_mirror_and_clears_the_diagonal_imaginary_part():
    upper = StoredArray.hermitian(H)
    lower = StoredArray.hermitian(H, "L")
    assert upper.values.tolist() == [[1 + 0j, 2 - 1j], [2 + 1j, 4 + 0j]]
    assert lower.values.tolist() == [[1 + 0j, 20 - 20j], [20 + 20j, 4 + 0j]]
    real = StoredArray.hermitian(A, "L")
    assert real.values.tolist() == [[1.0, 20.0], [20.0, 4.0]]
    assert real.object_marker == "Hermitian"
    for built in (upper, lower, real):
        assert built.to_interpreted().tolist() == built.values.tolist()


def test_diagonal_from_a_vector_or_a_square_matrix():
    from_vector = StoredArray.diagonal(np.array([3.0, -0.0, 5.0]))
    assert from_vector.shape == (3, 3)
    assert from_vector.object_marker == "Diagonal"
    assert bits(from_vector.values) == [
        0x4008000000000000, 0, 0,
        0, 0x8000000000000000, 0,
        0, 0, 0x4014000000000000,
    ]  # fmt: skip
    from_matrix = StoredArray.diagonal(A)
    assert from_matrix.values.tolist() == [[1.0, 0.0], [0.0, 4.0]]
    complex_zero = StoredArray.diagonal(np.array([complex(1.0, -2.0), complex(-0.0, 0.0)]))
    assert bits(complex_zero.values.view("<f8")) == [
        0x3FF0000000000000, 0xC000000000000000, 0, 0,
        0, 0, 0x8000000000000000, 0,
    ]  # fmt: skip
    assert from_vector.to_interpreted().tolist() == from_vector.values.tolist()


def test_signed_zeros_and_nan_payloads_are_copied_bit_for_bit():
    lower = StoredArray.symmetric(SIGNS, "L")
    assert bits(lower.values) == [
        0x8000000000000000, 0x4020000000000000,
        0x4020000000000000, 0x8000000000000000,
    ]  # fmt: skip
    upper = StoredArray.symmetric(SIGNS, "U")
    assert bits(upper.values) == [
        0x8000000000000000, 0x7FF8000000000055,
        0x7FF8000000000055, 0x8000000000000000,
    ]  # fmt: skip
    diagonal = StoredArray.diagonal(SIGNS)
    assert bits(diagonal.values) == [0x8000000000000000, 0, 0, 0x8000000000000000]
    # Hermitian keeps the real part's -0.0 on the diagonal and clears only the
    # imaginary part to +0.0; the mirrored entry is conjugated.
    herm = StoredArray.hermitian(CSIGNS, "U")
    assert bits(herm.values.view("<f8")) == [
        0x8000000000000000, 0, 0x8000000000000000, 0x401C000000000000,
        0x8000000000000000, 0xC01C000000000000, 0x8000000000000000, 0,
    ]  # fmt: skip


def test_represented_elements_materialise_on_their_carriers():
    words = np.array([[-(2**127), 3], [20, 2**127 - 1]], dtype=object)
    carrier = (
        np.array(
            [[w & ((1 << 64) - 1), (w & ((1 << 128) - 1)) >> 64] for w in words.reshape(-1)],
            dtype="<u8",
        )
        .view(INT128.dtype)
        .reshape((2, 2))
    )
    i128 = StoredArray(carrier, INT128)
    upper = StoredArray.symmetric(i128)
    assert upper.element == INT128
    assert upper.tolist() == [[-(2**127), 3], [3, 2**127 - 1]]
    assert StoredArray.symmetric(i128, "L").tolist() == [[-(2**127), 20], [20, 2**127 - 1]]
    assert StoredArray.hermitian(i128, "L").tolist() == [[-(2**127), 20], [20, 2**127 - 1]]
    assert StoredArray.diagonal(i128).tolist() == [[-(2**127), 0], [0, 2**127 - 1]]
    c16 = StoredArray(
        cf16([(0x3C00, 0x4880), (0x4000, 0xBC00), (0x4D00, 0x4D00), (0x4400, 0x0000)], (2, 2)),
        COMPLEXF16,
    )
    lower = StoredArray.hermitian(c16, "L")
    assert lower.tolist() == [[1 + 0j, 20 - 20j], [20 + 20j, 4 + 0j]]
    # Conjugation flips the imaginary sign bit exactly, -0.0 and NaN payloads included.
    odd = StoredArray(
        cf16([(0x3C00, 0x8000), (0x4000, 0x7E55), (0x4D00, 0x0000), (0x4400, 0x0000)], (2, 2)),
        COMPLEXF16,
    )
    mirrored = StoredArray.hermitian(odd, "U")
    assert mirrored.values["imag"].view("<u2").tolist() == [[0x0000, 0x7E55], [0xFE55, 0x0000]]
    dates = StoredArray(np.array([[1, 3], [20, 4]], dtype="<i8"), StoredElement.date(Monthly()))
    diagonal = StoredArray.diagonal(dates)
    assert diagonal.element == StoredElement.date(Monthly())
    assert [[v.value for v in row] for row in diagonal.tolist()] == [[1, 0], [0, 4]]
    assert StoredArray.symmetric(dates, "L").values.tolist() == [[1, 20], [20, 4]]


def test_boolean_input_becomes_the_int8_carrier_with_the_inactive_bool_marker():
    built = StoredArray.symmetric(np.array([[True, True], [False, False]]), "L")
    assert built.element == StoredElement.numeric("<i1", "Bool")
    assert built.object_marker == "Symmetric"
    assert built.active_marker == "Symmetric"
    assert built.values.tolist() == [[1, 0], [0, 0]]
    encoded = encode_array(built)
    assert (encoded.marker, encoded.object_marker) == ("Bool", "Symmetric")
    # Julia's loader applies jtype first, so the Bool marker is inactive there too.
    assert built.to_interpreted().dtype == np.int8


@pytest.mark.parametrize("dtype", ["i1", "<u2", "<f2", "<f4", "<c8", "<i4"])
def test_every_ordinary_family_is_accepted(dtype):
    values = np.array([[1, 3], [20, 4]]).astype(dtype)
    built = StoredArray.symmetric(values, "L")
    assert built.element == StoredElement.numeric(np.dtype(dtype))
    assert built.values.dtype == np.dtype(dtype)
    assert built.values.tolist() == np.array([[1, 20], [20, 4]]).astype(dtype).tolist()
    assert (
        StoredArray.diagonal(values).values.tolist()
        == np.array([[1, 0], [0, 4]]).astype(dtype).tolist()
    )


def test_empty_and_singleton_wrappers():
    empty = StoredArray.symmetric(np.empty((0, 0)))
    assert empty.shape == (0, 0)
    assert encode_array(empty).object_marker == "Symmetric"
    assert encode_array(empty).marker is None
    assert StoredArray.diagonal(np.empty(0)).shape == (0, 0)
    assert StoredArray.hermitian(np.empty((0, 0), dtype=np.complex128)).shape == (0, 0)
    one = StoredArray.hermitian(np.array([[complex(2.0, 5.0)]]))
    assert one.values.tolist() == [[2 + 0j]]
    assert StoredArray.symmetric(np.array([[-0.0]])).values.tolist() == [[-0.0]]
    assert StoredArray.diagonal(np.array([7.5])).values.tolist() == [[7.5]]


def test_fortran_and_strided_inputs_give_c_contiguous_carriers():
    fortran = np.asfortranarray(A)
    for built in (
        StoredArray.symmetric(fortran, "L"),
        StoredArray.hermitian(fortran),
        StoredArray.diagonal(fortran),
        StoredArray.diagonal(np.arange(6.0)[::2]),
        StoredArray.symmetric(np.arange(16.0).reshape(4, 4)[::2, ::2]),
    ):
        assert built.values.flags.c_contiguous
        assert built.values.flags.owndata
    assert StoredArray.symmetric(fortran, "L").values.tolist() == [[1.0, 20.0], [20.0, 4.0]]


def test_refusals_are_explicit():
    with pytest.raises(ValueError, match="square"):
        StoredArray.symmetric(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="square"):
        StoredArray.hermitian(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="square"):
        StoredArray.diagonal(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="two-dimensional"):
        StoredArray.symmetric(np.zeros(4))
    with pytest.raises(ValueError, match="two-dimensional"):
        StoredArray.diagonal(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="uplo"):
        StoredArray.symmetric(A, "X")
    with pytest.raises(ValueError, match="uplo"):
        StoredArray.hermitian(A, "u")
    with pytest.raises(TypeError, match="NumPy array or a StoredArray"):
        StoredArray.symmetric([[1.0, 3.0], [20.0, 4.0]])
    with pytest.raises(TypeError, match="ordinary native-endian"):
        StoredArray.symmetric(np.array([["a", "b"], ["c", "d"]]))
    with pytest.raises(TypeError, match="ordinary native-endian"):
        StoredArray.symmetric(np.array([[1, 2], [3, 4]], dtype=object))
    with pytest.raises(TypeError, match="ordinary native-endian"):
        StoredArray.symmetric(np.array([[1, 2], [3, 4]], dtype=">i8"))
    with pytest.raises(TypeError, match="ordinary native-endian"):
        StoredArray.symmetric(np.array([[1, 2], [3, 4]], dtype=np.longdouble))
    marked = StoredArray.symmetric(A)
    with pytest.raises(ValueError, match="already"):
        StoredArray.hermitian(marked)
    # An empty wrapper cannot keep a non-default width: the element token is
    # inactive under the wrapper marker, so only the kind defaults are storable.
    with pytest.raises(ValueError, match="kind default"):
        StoredArray.diagonal(np.empty(0, dtype=np.float32))
    with pytest.raises(ValueError, match="kind default"):
        StoredArray.symmetric(np.empty((0, 0), dtype=bool))
    with pytest.raises(ValueError, match="kind default"):
        StoredArray.symmetric(StoredArray(np.empty((0, 0), dtype=INT128.dtype), INT128))
    assert StoredArray.symmetric(np.empty((0, 0), dtype=np.uint64)).shape == (0, 0)
    with pytest.raises(ValueError, match="already"):
        StoredArray.diagonal(marked)


# ---- capacity is checked before anything is converted or expanded ---------


class _Unconvertible(np.ndarray):
    """A view whose conversions and copies fail: a refusal must come first."""

    def astype(self, *args, **kwargs):
        raise AssertionError("astype was reached before validation")

    def copy(self, *args, **kwargs):
        raise AssertionError("copy was reached before validation")


@pytest.fixture
def no_large_allocation(monkeypatch):
    """Fail any ``np.zeros`` request beyond the patched limit's element count."""
    real_zeros = np.zeros

    def spy(shape, dtype=float, *args, **kwargs):
        size = int(np.prod(shape)) if not isinstance(shape, int) else shape
        if size * np.dtype(dtype).itemsize > _interpret.MAX_BYTES:
            raise AssertionError(f"np.zeros({shape!r}) was reached before validation")
        return real_zeros(shape, dtype, *args, **kwargs)

    monkeypatch.setattr(np, "zeros", spy)


def _i128_vector(count: int) -> StoredArray:
    return StoredArray(np.zeros(count, dtype=INT128.dtype), INT128)


@pytest.mark.usefixtures("no_large_allocation")
@pytest.mark.parametrize(
    ("make", "itemsize", "julia_name"),
    [
        (lambda n: np.arange(float(n)), 8, "Float64"),
        (lambda n: np.ones(n, dtype=bool), 1, "Int8"),
        (_i128_vector, 16, "Int128"),
    ],
    ids=["ordinary", "boolean", "represented"],
)
def test_diagonal_vectors_are_sized_before_the_square_is_allocated(
    monkeypatch, make, itemsize, julia_name
):
    # A three-element diagonal fits exactly at the patched limit; a fourth
    # element would need the 4-by-4 square, which is refused before np.zeros
    # (or a Boolean astype) is ever reached.
    monkeypatch.setattr(_interpret, "MAX_BYTES", 9 * itemsize)
    built = StoredArray.diagonal(make(3))
    assert built.shape == (3, 3)
    assert built.values.nbytes == 9 * itemsize
    refused = make(4)
    if isinstance(refused, np.ndarray):
        refused = refused.view(_Unconvertible)
    with pytest.raises(ValueError, match=f"4-by-4 Diagonal matrix of {julia_name}.*nothing"):
        StoredArray.diagonal(refused)


@pytest.mark.usefixtures("no_large_allocation")
@pytest.mark.parametrize("build", ["diagonal", "symmetric", "hermitian"])
def test_boolean_broadcasts_are_sized_before_the_int8_copy(monkeypatch, build):
    monkeypatch.setattr(_interpret, "MAX_BYTES", 9)
    construct = getattr(StoredArray, build)
    fits = np.broadcast_to(np.bool_(True), (3, 3)).view(_Unconvertible)
    # The refusal-first order lets a broadcast view through only after the
    # checks; its Int8 copy is then taken from the parent buffer, so the
    # spy view is replaced for the accepted case.
    accepted = construct(np.broadcast_to(np.bool_(True), (3, 3)))
    assert accepted.values.tolist() == (
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]] if build == "diagonal" else [[1, 1, 1]] * 3
    )
    assert accepted.element.marker == "Bool"
    too_large = np.broadcast_to(np.bool_(True), (4, 4)).view(_Unconvertible)
    with pytest.raises(ValueError, match=r"4-by-4 .* Int8 .*nothing was allocated"):
        construct(too_large)
    # Shape and uplo problems are refused before conversion as well.
    with pytest.raises(ValueError, match="square matrices only"):
        construct(np.broadcast_to(np.bool_(True), (4, 3)).view(_Unconvertible))
    with pytest.raises(ValueError, match="two-dimensional"):
        construct(np.broadcast_to(np.bool_(True), (2, 2, 2)).view(_Unconvertible))
    if build != "diagonal":
        with pytest.raises(ValueError, match="uplo"):
            construct(fits, "X")
        with pytest.raises(ValueError, match="two-dimensional"):
            construct(np.ones(3, dtype=bool).view(_Unconvertible))


@pytest.mark.usefixtures("no_large_allocation")
@pytest.mark.parametrize("build", ["diagonal", "symmetric", "hermitian"])
def test_square_inputs_at_the_boundary_are_accepted_and_one_beyond_refused(monkeypatch, build):
    # A represented input is built under the real limit, then the limit is
    # lowered so that its own square no longer fits.
    stored = StoredArray(np.zeros((4, 4), dtype=INT128.dtype), INT128)
    monkeypatch.setattr(_interpret, "MAX_BYTES", 9 * 16)
    construct = getattr(StoredArray, build)
    assert construct(np.arange(9.0).reshape(3, 3)).shape == (3, 3)
    assert construct(StoredArray(np.zeros((3, 3), dtype=INT128.dtype), INT128)).shape == (3, 3)
    assert construct(np.arange(16.0).reshape(4, 4)).values.nbytes == 128
    with pytest.raises(ValueError, match=r"5-by-5 .* Float64 .*nothing was allocated"):
        construct(np.arange(25.0).reshape(5, 5))
    # A square represented input is as large as its output, so the carrier's
    # own validation refuses it first; either way nothing is allocated.
    with pytest.raises(ValueError, match=r"Int128 .*nothing was allocated"):
        construct(stored)


def test_structure_capacity_uses_the_real_payload_limit():
    # 4096 * 4096 * 8 is exactly the 128 MiB payload limit; one more row is
    # refused before the square is requested (a spy makes sure of that).
    real_zeros = np.zeros
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(np, "zeros", lambda *a, **k: (_ for _ in ()).throw(AssertionError(a)))
        with pytest.raises(ValueError, match="4097-by-4097 Diagonal matrix of Float64"):
            StoredArray.diagonal(real_zeros(4097))
        with pytest.raises(ValueError, match="20000-by-20000 Diagonal matrix of Float64"):
            StoredArray.diagonal(real_zeros(20000))
        with pytest.raises(ValueError, match="20000-by-20000 Symmetric matrix of Int8"):
            StoredArray.symmetric(
                np.broadcast_to(np.bool_(True), (20000, 20000)).view(_Unconvertible)
            )
    assert _interpret.MAX_BYTES == 4096 * 4096 * 8


def test_structure_dense_helper_matches_the_loader_side_reconstruction():
    stored = np.array([[1.0, 3.0], [20.0, 4.0]])
    assert structure_dense(stored, "Symmetric").tolist() == [[1.0, 3.0], [3.0, 4.0]]
    assert structure_dense(stored, "Symmetric", "L").tolist() == [[1.0, 20.0], [20.0, 4.0]]
    assert structure_dense(stored, "Diagonal").tolist() == [[1.0, 0.0], [0.0, 4.0]]
    preserved = StoredArray(stored, StoredElement.numeric("<f8"), object_marker="Symmetric")
    assert preserved.to_interpreted().tolist() == [[1.0, 3.0], [3.0, 4.0]]


# ---- files -----------------------------------------------------------------


@NATIVE
def test_constructed_matrices_round_trip_with_their_markers(tmp_path):
    path = tmp_path / "structure.daec"
    values = {
        "sym_l": StoredArray.symmetric(A, "L"),
        "herm_u": StoredArray.hermitian(H),
        "diag": StoredArray.diagonal(np.array([3.0, -0.0, 5.0])),
        "bool": StoredArray.symmetric(np.array([[True, True], [False, False]])),
        "empty": StoredArray.diagonal(np.empty(0)),
        "c16": StoredArray.hermitian(
            StoredArray(
                cf16(
                    [(0x3C00, 0x4880), (0x4000, 0xBC00), (0x4D00, 0x4D00), (0x4400, 0x0000)], (2, 2)
                ),
                COMPLEXF16,
            ),
            "L",
        ),
    }
    with open_dataecon(path, "w") as db:
        for name, value in values.items():
            db.write_array(name, value)
        with pytest.raises(TypeError, match="read_array"):
            db.read_series("sym_l")
    with open_dataecon(path) as db:
        for name, value in values.items():
            back = db.read_array(name)
            assert back == value, name
            assert db.get_attribute(name, "jtype") == value.object_marker
            assert db.object_info(name).object_type == 20
        assert db.get_attributes("bool") == {"jeltype": "Bool", "jtype": "Symmetric"}
        assert db.get_attributes("empty") == {"jtype": "Diagonal"}
        assert db.read_array("sym_l").to_interpreted().tolist() == [[1.0, 20.0], [20.0, 4.0]]


def fixture_structures() -> dict[str, StoredArray]:
    """The Julia fixture's structure rows rebuilt with the Python constructors."""
    words = [-(2**127), 3, 20, 2**127 - 1]
    i128 = (
        np.array([[w & ((1 << 64) - 1), (w & ((1 << 128) - 1)) >> 64] for w in words], dtype="<u8")
        .view(INT128.dtype)
        .reshape((2, 2))
    )
    c16 = cf16([(0x3C00, 0x4880), (0x4000, 0xBC00), (0x4D00, 0x4D00), (0x4400, 0x0000)], (2, 2))
    mitm = np.array([[1, 3], [20, 4]], dtype="<i8")
    signs = SIGNS  # the fixture's column-major literal is [-0.0 NaN; 8.0 -0.0]
    return {
        "str_sym_u_f64": StoredArray.symmetric(A, "U"),
        "str_sym_l_f64": StoredArray.symmetric(A, "L"),
        "str_herm_u_c64": StoredArray.hermitian(H, "U"),
        "str_herm_l_c64": StoredArray.hermitian(H, "L"),
        "str_diag_vec_f64": StoredArray.diagonal(np.array([3.0, -0.0, 5.0])),
        "str_sym_l_signs": StoredArray.symmetric(signs, "L"),
        "str_herm_u_signs": StoredArray.hermitian(CSIGNS, "U"),
        "str_sym_u_i128": StoredArray.symmetric(StoredArray(i128, INT128), "U"),
        "str_herm_l_c16": StoredArray.hermitian(StoredArray(c16, COMPLEXF16), "L"),
        "str_diag_mit_m": StoredArray.diagonal(StoredArray(mitm, StoredElement.date(Monthly()))),
        "str_sym_l_bool": StoredArray.symmetric(np.array([[True, True], [False, False]]), "L"),
        "str_diag_f16": StoredArray.diagonal(np.array([[1, 3], [20, 4]], dtype=np.float16)),
        "str_sym_empty_f64": StoredArray.symmetric(np.empty((0, 0))),
        "str_diag_empty_f64": StoredArray.diagonal(np.empty(0)),
        "str_herm_1x1_c64": StoredArray.hermitian(np.array([[complex(2.0, 5.0)]])),
    }


@NATIVE
def test_julia_fixture_structures_equal_the_python_constructions_and_rewrite_identically(tmp_path):
    expected = fixture_structures()
    fixture = FIXTURES / "julia_represented_mvtseries.daec"
    rewritten = tmp_path / "rewrite.daec"
    with open_dataecon(fixture) as source, open_dataecon(rewritten, "w") as target:
        names = {e.path.lstrip("/") for e in source.list_objects("/") if e.path.startswith("/str_")}
        assert names == set(expected)
        for name, value in expected.items():
            back = source.read_array(name)
            # Julia writes a redundant element token on its empty wrappers; it
            # is inactive under the wrapper marker and kept verbatim on read.
            assert back.object_marker == value.object_marker, name
            assert back.element.with_marker(None) == value.element.with_marker(None), name
            assert back.values.tobytes() == value.values.tobytes(), name
            assert back.values.shape == value.values.shape, name
            redundant = name in ("str_sym_empty_f64", "str_diag_empty_f64")
            assert (back.element == value.element) is not redundant
            if value.values.size:
                assert back.to_interpreted().tolist() == value.to_interpreted().tolist()
            target.write_array(name, back)
    query = (
        "SELECT o.name, o.class, o.type, m.eltype, m.elfreq, m.value, x1.length, x2.length "
        "FROM objects o JOIN mvtseries m ON m.id=o.id JOIN axes x1 ON x1.id=m.axis1_id "
        "JOIN axes x2 ON x2.id=m.axis2_id WHERE o.pid=0 AND o.name LIKE 'str_%' ORDER BY o.name"
    )
    markers = (
        "SELECT o.name, a.name, a.value FROM attributes a JOIN objects o ON o.id=a.id "
        "WHERE o.pid=0 AND o.name LIKE 'str_%' AND a.name IN ('jeltype','jtype') "
        "ORDER BY o.name, a.name"
    )
    with closing(sqlite3.connect(fixture)) as a, closing(sqlite3.connect(rewritten)) as b:
        assert a.execute(query).fetchall() == b.execute(query).fetchall()
        # The redundant element token Julia writes on its empty wrappers is
        # inactive under the wrapper marker and rewritten verbatim.
        assert a.execute(markers).fetchall() == b.execute(markers).fetchall()
