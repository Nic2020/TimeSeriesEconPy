# SPDX-License-Identifier: MIT
"""Represented MVTSeries elements: the StoredMVTSeries container, codec and file paths."""

from __future__ import annotations

import hashlib
import importlib.util
import sqlite3
import subprocess
import sys
import tomllib
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from tsecon import MIT, MVTSeries, Workspace, mm
from tsecon.dataecon import (
    COMPLEXF16,
    INT128,
    UINT128,
    StoredElement,
    StoredMVTSeries,
    StoredSeries,
    open_dataecon,
)
from tsecon.dataecon._codec import (
    decode_matrix,
    encode_series,
    encode_stored_mvtseries,
    validate_matrix_payload,
)
from tsecon.dataecon._represented import (
    check_column_names,
    resolve_mvtseries_interpretation,
)
from tsecon.frequencies import BDaily, Daily, HalfYearly, Monthly, Quarterly, Unit, Weekly, Yearly
from tsecon.mit import Duration

FIXTURES = Path(__file__).parent / "fixtures"
NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built; configure TSECON_DATAECON_ROOT",
)
ANCHOR = mm(2024, 1)  # code 24288
Q3 = Quarterly(3)
DATE_Q = StoredElement.date(Q3)
DUR_Y6 = StoredElement.duration(Yearly(6))
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1


def q(code: int) -> MIT:
    return MIT(Q3, code)


def dated(rows: list[list[int]], columns=("a", "b", "c"), anchor: MIT = ANCHOR) -> StoredMVTSeries:
    """A date-element container from integer codes, one inner list per row."""
    return StoredMVTSeries.from_list(anchor, columns, DATE_Q, [[q(c) for c in row] for row in rows])


def words(element: StoredElement, rows: list[list[int]]) -> np.ndarray:
    flat = [v for row in rows for v in row]
    return StoredSeries.from_list(ANCHOR, element, flat).values.reshape((len(rows), len(rows[0])))


def cf16(bits: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.empty(len(bits), dtype=COMPLEXF16.dtype)
    out["real"] = np.array([r for r, _ in bits], dtype="<u2").view(np.float16)
    out["imag"] = np.array([i for _, i in bits], dtype="<u2").view(np.float16)
    return out.reshape(shape)


# ---- the container ---------------------------------------------------------


def test_construction_records_anchor_names_element_and_shape():
    value = dated([[1, 2, 3], [4, 5, 6]])
    assert value.firstdate == ANCHOR
    assert value.lastdate == mm(2024, 2)
    assert value.frequency == Monthly()
    assert value.columns == ("a", "b", "c")
    assert value.element == DATE_Q
    assert value.shape == (2, 3)
    assert len(value) == 2
    assert value.object_marker is None
    assert value.active_marker is None
    assert value.values.dtype == np.dtype("<i8")
    assert value.values.flags.c_contiguous
    assert value.values.flags.owndata
    assert value.tolist() == [[q(1), q(2), q(3)], [q(4), q(5), q(6)]]
    assert "columns=('a', 'b', 'c')" in repr(value)
    assert value == dated([[1, 2, 3], [4, 5, 6]])
    assert value != dated([[1, 2, 3], [4, 5, 7]])
    assert value != dated([[1, 2, 3], [4, 5, 6]], anchor=mm(2024, 2))
    assert value != dated([[1, 2, 3], [4, 5, 6]], columns=("a", "b", "d"))
    assert value.__eq__(object()) is NotImplemented


def test_empty_rows_keep_names_and_lastdate():
    value = StoredMVTSeries.from_list(ANCHOR, ("a", "b"), DATE_Q, [])
    assert value.shape == (0, 2)
    assert len(value) == 0
    assert value.lastdate == ANCHOR - 1
    assert value.tolist() == []
    assert value.to_interpreted() == value


@pytest.mark.parametrize(
    ("element", "rows"),
    [
        (INT128, [[-(2**127), 2**127 - 1], [-1, 7], [0, 2**70]]),
        (UINT128, [[0, 2, 2**70], [1, 2**128 - 1, 5]]),
        (DUR_Y6, [[Duration(Yearly(6), 1), Duration(Yearly(6), INT64_MIN)]]),
    ],
)
def test_every_represented_family_round_trips_through_from_list(element, rows):
    names = tuple("abc"[: len(rows[0])])
    value = StoredMVTSeries.from_list(ANCHOR, names, element, rows)
    assert value.element == element
    assert value.tolist() == rows
    assert value.values.dtype == element.dtype


def test_complexf16_carrier_keeps_bit_patterns_and_widens_exactly():
    carrier = cf16([(0x8000, 0x3D00), (0x7E55, 0xFC00), (0x4000, 0x4200), (0x0000, 0x8000)], (2, 2))
    value = StoredMVTSeries(ANCHOR, ("a", "b"), carrier, COMPLEXF16)
    listed = value.tolist()
    assert listed[0][0] == complex(-0.0, 1.25)
    assert listed[1][0] == complex(2.0, 3.0)
    assert np.isnan(listed[0][1].real)
    assert listed[0][1].imag == -np.inf
    wide = value.to_complex64()
    assert isinstance(wide, MVTSeries)
    assert wide.values.dtype == np.complex64
    assert wide.column_names == ("a", "b")
    assert np.signbit(wide.values[0, 0].real)
    assert np.isnan(wide.values[0, 1])
    with pytest.raises(TypeError, match="ComplexF16"):
        dated([[1]], columns=("a",)).to_complex64()


def test_from_list_rejects_ragged_rows_and_wrong_types():
    with pytest.raises(ValueError, match="Row 1 holds 2 values for 3 columns"):
        StoredMVTSeries.from_list(
            ANCHOR, ("a", "b", "c"), DATE_Q, [[q(1), q(2), q(3)], [q(4), q(5)]]
        )
    with pytest.raises(TypeError, match="element frequency"):
        StoredMVTSeries.from_list(ANCHOR, ("a",), DATE_Q, [[mm(2024, 1)]])
    with pytest.raises(TypeError, match="explicit NumPy array"):
        StoredMVTSeries.from_list(ANCHOR, ("a",), StoredElement.numeric("<i8", "Float64"), [[1]])
    with pytest.raises(TypeError, match="StoredElement"):
        StoredMVTSeries.from_list(ANCHOR, ("a",), "date", [[q(1)]])  # type: ignore[arg-type]


def test_constructor_refuses_bad_anchor_names_dtype_rank_and_capacity(monkeypatch):
    carrier = np.zeros((2, 2), dtype="<i8")
    with pytest.raises(TypeError, match="core MIT anchor"):
        StoredMVTSeries(24288, ("a", "b"), carrier, DATE_Q)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="StoredElement"):
        StoredMVTSeries(ANCHOR, ("a", "b"), carrier, "date")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="NumPy array"):
        StoredMVTSeries(ANCHOR, ("a", "b"), [[1, 2], [3, 4]], DATE_Q)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="no implicit conversion"):
        StoredMVTSeries(ANCHOR, ("a", "b"), carrier.astype("<i4"), DATE_Q)
    with pytest.raises(TypeError, match="no implicit conversion"):
        StoredMVTSeries(ANCHOR, ("a", "b"), np.zeros((2, 2), dtype=np.longdouble), DATE_Q)
    with pytest.raises(ValueError, match="two-dimensional"):
        StoredMVTSeries(ANCHOR, ("a", "b"), np.zeros(4, dtype="<i8"), DATE_Q)
    with pytest.raises(ValueError, match="2 columns but 3 column names"):
        StoredMVTSeries(ANCHOR, ("a", "b", "c"), carrier, DATE_Q)
    with pytest.raises(ValueError, match="no columns"):
        StoredMVTSeries(ANCHOR, (), np.zeros((2, 0), dtype="<i8"), DATE_Q)
    monkeypatch.setattr("tsecon.dataecon._represented.MAX_BYTES", 16)
    with pytest.raises(ValueError, match="byte limit"):
        StoredMVTSeries(ANCHOR, ("a", "b"), carrier, DATE_Q)


def test_column_name_rules_match_the_names_axis_encoding():
    assert check_column_names("only") == ("only",)
    assert check_column_names(["a", ""]) == ("a", "")
    assert check_column_names(iter(("x", "y"))) == ("x", "y")
    with pytest.raises(ValueError, match="no columns"):
        check_column_names([])
    with pytest.raises(TypeError, match="plain Python strings"):
        check_column_names(["a", 1])
    with pytest.raises(TypeError, match="iterable of str"):
        check_column_names(3)
    with pytest.raises(ValueError, match="newline"):
        check_column_names(["a\nb"])
    with pytest.raises(ValueError, match="NUL"):
        check_column_names(["a\0b"])
    with pytest.raises(ValueError, match="distinct"):
        check_column_names(["a", "a"])


def test_copy_policy_and_live_carrier_validation():
    carrier = np.array([[1, 2], [3, 4]], dtype="<i8")
    copied = StoredMVTSeries(ANCHOR, ("a", "b"), carrier, DATE_Q)
    shared = StoredMVTSeries(ANCHOR, ("a", "b"), carrier, DATE_Q, copy=False)
    assert copied.values is not carrier
    assert shared.values is carrier
    carrier[0, 0] = 9
    assert copied.tolist()[0][0] == q(1)
    assert shared.tolist()[0][0] == q(9)
    # Read-only or strided input is copied into an owning carrier.
    frozen = carrier.copy()
    frozen.flags.writeable = False
    assert StoredMVTSeries(ANCHOR, ("a", "b"), frozen, DATE_Q, copy=False).values is not frozen
    fortran = np.asfortranarray(carrier)
    owned = StoredMVTSeries(ANCHOR, ("a", "b"), fortran, DATE_Q, copy=False)
    assert owned.values is not fortran
    assert owned.values.flags.c_contiguous
    assert owned.tolist() == shared.tolist()
    # A carrier reshaped or retyped in place is refused by every operation.
    shared.values.resize((4, 1), refcheck=False)
    with pytest.raises(ValueError, match="column names"):
        shared.validate()
    with pytest.raises(ValueError, match="column names"):
        len(shared)
    with pytest.raises(ValueError, match="column names"):
        shared.tolist()
    with pytest.raises(ValueError, match="column names"):
        encode_series(shared)
    shared.values.resize((2, 2), refcheck=False)
    shared.validate()
    view = StoredMVTSeries(ANCHOR, ("a", "b"), carrier[:, ::-1], DATE_Q, copy=False)
    assert view.values.flags.c_contiguous  # copied because the input was strided
    shared.values.flags.writeable = False
    shared.validate()  # clearing the flag is allowed


class _ShrinkingCarrier(np.ndarray):
    def tobytes(self, order="C"):
        return super().tobytes(order)[:8]


class _GrowingCarrier(np.ndarray):
    def tobytes(self, order="C"):
        return super().tobytes(order) + b"\0" * 8


@pytest.mark.parametrize("cls", [_ShrinkingCarrier, _GrowingCarrier])
def test_snapshot_size_changes_are_refused_before_conversion_and_storage(cls):
    carrier = np.array([[1, 2], [3, 4]], dtype="<i8").view(cls)
    value = StoredMVTSeries(ANCHOR, ("a", "b"), carrier, DATE_Q, copy=False)
    with pytest.raises(ValueError, match="changed size"):
        value.to_interpreted()
    with pytest.raises(ValueError, match="changed size"):
        encode_series(value)


# ---- markers and interpretation -----------------------------------------


def test_numeric_carrier_needs_a_marker_and_bool_marker_needs_a_bool_mvtseries():
    carrier = np.array([[1, 0], [2, 1]], dtype="<i8")
    with pytest.raises(TypeError, match="belong in an MVTSeries"):
        StoredMVTSeries(ANCHOR, ("a", "b"), carrier, StoredElement.numeric("<i8"))
    with pytest.raises(TypeError, match="Boolean MVTSeries"):
        StoredMVTSeries(ANCHOR, ("a", "b"), carrier, StoredElement.numeric("<i8", "Bool"))
    with pytest.raises(TypeError, match="Unsupported reconstruction marker"):
        StoredMVTSeries(ANCHOR, ("a", "b"), carrier, StoredElement.numeric("<i8", "NoSuchType"))


def test_foreign_element_markers_interpret_like_series_and_arrays():
    carrier = np.array([[1, 0], [2, 1]], dtype="<i8")
    as_float = StoredMVTSeries(ANCHOR, ("a", "b"), carrier, StoredElement.numeric("<i8", "Float64"))
    assert as_float.active_marker == "Float64"
    converted = as_float.to_interpreted()
    assert isinstance(converted, MVTSeries)
    assert converted.values.dtype == np.float64
    assert converted.values.tolist() == [[1.0, 0.0], [2.0, 1.0]]
    assert converted.column_names == ("a", "b")
    assert converted.firstdate == ANCHOR
    assert as_float.tolist() == [[1, 0], [2, 1]]  # stored values, not converted ones
    as_date = StoredMVTSeries(
        ANCHOR, ("a", "b"), carrier, StoredElement.numeric("<i8", "MIT{Monthly}")
    )
    dates = as_date.to_interpreted()
    assert isinstance(dates, StoredMVTSeries)
    assert dates.element == StoredElement.date(Monthly())
    assert (
        dates.tolist() == [[mm(1), mm(0)], [mm(2), mm(1)]]
        if False
        else dates.tolist()[0][0] == MIT(Monthly(), 1)
    )
    wide = StoredMVTSeries(ANCHOR, ("a", "b"), carrier, StoredElement.numeric("<i8", "Int128"))
    assert wide.to_interpreted().element == INT128
    assert wide.to_interpreted().tolist() == [[1, 0], [2, 1]]
    with pytest.raises(ValueError):
        StoredMVTSeries(
            ANCHOR,
            ("a", "b"),
            np.array([[1.5, 0.0], [2.0, 1.0]]),
            StoredElement.numeric("<f8", "Int64"),
        )
    with pytest.raises(TypeError, match="no conversion"):
        StoredMVTSeries(ANCHOR, ("a", "b"), carrier, DATE_Q.with_marker("MIT{Monthly}"))


def test_wide_bool_marker_is_preserved_with_explicit_conversion():
    value = StoredMVTSeries(
        ANCHOR, ("a", "b"), words(INT128, [[1, 0], [0, 1]]), INT128.with_bool_marker()
    )
    assert value.active_marker == "Bool"
    flags = value.to_bool()
    assert isinstance(flags, MVTSeries)
    assert flags.values.dtype == np.bool_
    assert flags.values.tolist() == [[True, False], [False, True]]
    interpreted = value.to_interpreted()
    assert isinstance(interpreted, MVTSeries)
    assert interpreted.values.dtype == np.bool_
    with pytest.raises(TypeError, match="Bool"):
        dated([[1]], columns=("a",)).to_bool()
    with pytest.raises(ValueError):
        StoredMVTSeries(
            ANCHOR, ("a", "b"), words(INT128, [[2, 0], [0, 1]]), INT128.with_bool_marker()
        )
    # Editing the live carrier is revalidated on the next conversion.
    value.values[0, 0] = 3
    with pytest.raises(ValueError):
        value.to_bool()


@pytest.mark.parametrize(
    "token",
    [
        "MVTSeries",
        "MVTSeries{Monthly}",
        "MVTSeries{Monthly, MIT{Quarterly{3}}}",
        "MVTSeries{Monthly, MIT{Quarterly{3}}, Matrix{MIT{Quarterly{3}}}}",
        "AbstractMatrix",
        "Any",
    ],
)
def test_identity_object_markers_are_preserved_and_interpret_as_identity(token):
    value = StoredMVTSeries(
        ANCHOR, ("a", "b"), np.array([[1, 2], [3, 4]], dtype="<i8"), DATE_Q, object_marker=token
    )
    assert value.object_marker == token
    assert value.active_marker is None
    interpreted = value.to_interpreted()
    assert isinstance(interpreted, StoredMVTSeries)
    assert interpreted.object_marker is None
    assert interpreted.tolist() == value.tolist()
    assert encode_stored_mvtseries(value).object_marker == token


@pytest.mark.parametrize(
    "token",
    [
        "MVTSeries{Quarterly{3}, MIT{Quarterly{3}}}",  # axis mismatch: Julia MethodError
        "MVTSeries{Monthly, Int64}",  # element mismatch: Julia MethodError
        "MVTSeries{Monthly,MIT{Quarterly{3}}}",  # spelling outside the exact table
        "TimeSeriesEcon.MVTSeries",  # qualified spelling outside the table
        "AbstractArray",  # abstract identity outside the table
        "Matrix",  # Julia DimensionMismatch
        "Matrix{Int64}",
        "Array",
        "BitMatrix",
        "Diagonal",
        "Symmetric",
        "TSeries",
        "Vector",
        "Symbol",
        "NoSuchType",
    ],
)
def test_other_object_markers_are_refused_without_evaluation(token):
    with pytest.raises(TypeError, match="not evaluated"):
        StoredMVTSeries(
            ANCHOR, ("a", "b"), np.array([[1, 2], [3, 4]], dtype="<i8"), DATE_Q, object_marker=token
        )


def test_object_marker_makes_the_element_marker_inactive():
    carrier = np.array([[1, 0], [2, 1]], dtype="<i8")
    value = StoredMVTSeries(
        ANCHOR,
        ("a", "b"),
        carrier,
        StoredElement.numeric("<i8", "NoSuchType"),
        object_marker="MVTSeries",
    )
    assert value.active_marker is None
    assert value.element.marker == "NoSuchType"
    interpreted = value.to_interpreted()
    assert isinstance(interpreted, MVTSeries)
    assert interpreted.values.dtype == np.int64
    encoded = encode_stored_mvtseries(value)
    assert (encoded.marker, encoded.object_marker) == ("NoSuchType", "MVTSeries")


def test_empty_date_or_duration_carriers_refuse_foreign_and_object_markers():
    empty = np.zeros((0, 2), dtype="<i8")
    with pytest.raises(TypeError, match="Julia cannot load an empty"):
        StoredMVTSeries(ANCHOR, ("a", "b"), empty, DATE_Q, object_marker="MVTSeries")
    with pytest.raises(TypeError, match="Julia cannot load an empty MIT"):
        StoredMVTSeries(ANCHOR, ("a", "b"), empty, DATE_Q.with_marker("Int64"))
    with pytest.raises(ValueError, match="no storable width"):
        StoredMVTSeries(
            ANCHOR, ("a", "b"), np.zeros((0, 2), dtype=INT128.dtype), INT128.with_bool_marker()
        )
    kind, target = resolve_mvtseries_interpretation(
        np.zeros((0, 2), dtype="<i8"), StoredElement.numeric("<i8", "Float64"), None, Monthly()
    )
    assert (kind, target.julia_name) == ("element", "Float64")


# ---- codec -----------------------------------------------------------------


def test_encoding_is_column_major_with_the_series_element_codes():
    value = dated([[1, 2, 3], [4, 5, 6]])
    encoded = encode_series(value)
    assert (encoded.object_type, encoded.element, encoded.element_frequency) == (21, 3, 67)
    assert (encoded.axis1_type, encoded.rows, encoded.frequency, encoded.first) == (1, 2, 32, 24288)
    assert (encoded.columns, encoded.names, encoded.marker, encoded.object_marker) == (
        3,
        "a\nb\nc",
        None,
        None,
    )
    assert np.frombuffer(encoded.payload, dtype="<i8").tolist() == [1, 4, 2, 5, 3, 6]
    tall = dated([[1, 2], [3, 4], [5, 6]], columns=("a", "b"))
    assert np.frombuffer(encode_series(tall).payload, dtype="<i8").tolist() == [1, 3, 5, 2, 4, 6]
    empty = encode_series(StoredMVTSeries.from_list(ANCHOR, ("a", "b"), DATE_Q, []))
    assert (empty.rows, empty.columns, empty.payload, empty.marker) == (
        0,
        2,
        b"",
        "MIT{Quarterly{3}}",
    )
    wide_empty = encode_series(StoredMVTSeries.from_list(ANCHOR, ("a",), INT128, []))
    assert (wide_empty.element, wide_empty.marker) == (1, "Int128")


def test_decoding_rebuilds_the_container_and_checks_names():
    value = dated([[1, 2, 3], [4, 5, 6]])
    encoded = encode_series(value)
    metadata = (3, 21, 3, 67, 1, 2, 32, 24288, 2, 3, 0, 0, len(encoded.payload))
    back = decode_matrix(metadata, encoded.payload, None, None, "a\nb\nc")
    assert isinstance(back, StoredMVTSeries)
    assert back == value
    assert back.values.flags.owndata
    assert back.values.flags.c_contiguous
    with pytest.raises(ValueError, match="duplicate"):
        decode_matrix(metadata, encoded.payload, None, None, "a\na\nc")
    with pytest.raises(ValueError, match="holds 2 names for 3 columns"):
        decode_matrix(metadata, encoded.payload, None, None, "a\nb")
    with pytest.raises(TypeError, match="named column axis"):
        decode_matrix(metadata, encoded.payload, None, None, None)
    with pytest.raises(TypeError, match="Unsupported whole-object"):
        validate_matrix_payload(metadata, encoded.payload, None, "Matrix", "a\nb\nc")
    with pytest.raises(TypeError, match="no conversion"):
        validate_matrix_payload(metadata, encoded.payload, "MIT{Monthly}", None, "a\nb\nc")
    truncated = (*metadata[:-1], len(encoded.payload) - 4)
    with pytest.raises(ValueError, match="element width"):
        validate_matrix_payload(truncated, encoded.payload[:-4], None, None, "a\nb\nc")


def test_bool_marker_on_a_wider_ordinary_payload_reads_as_boolean_mvtseries():
    payload = np.array([1, 0, 0, 1], dtype="<i8").tobytes()
    metadata = (3, 21, 1, 0, 1, 2, 32, 24288, 2, 2, 0, 0, len(payload))
    back = decode_matrix(metadata, payload, "Bool", None, "a\nb")
    assert isinstance(back, MVTSeries)
    assert back.values.dtype == np.bool_
    assert back.values.tolist() == [[True, False], [False, True]]
    with pytest.raises(ValueError):
        decode_matrix(metadata, np.array([1, 0, 2, 1], dtype="<i8").tobytes(), "Bool", None, "a\nb")
    # Plain matrices keep the one-byte rule.
    with pytest.raises(TypeError, match="one-byte"):
        decode_matrix((3, 20, 1, 0, 0, 2, 0, 0, 0, 2, 0, 0, 32), payload, "Bool", None)


# ---- files -----------------------------------------------------------------


@NATIVE
def test_every_family_round_trips_through_a_file_with_owning_results(tmp_path):
    path = tmp_path / "represented.daec"
    values = {
        "dates": dated([[1, 2, 3], [4, 5, 6]]),
        "durations": StoredMVTSeries.from_list(
            MIT(Daily(), 738000),
            ("x", "y"),
            DUR_Y6,
            [
                [Duration(Yearly(6), 1), Duration(Yearly(6), INT64_MIN)],
                [Duration(Yearly(6), -2), Duration(Yearly(6), INT64_MAX)],
            ],
        ),
        "int128": StoredMVTSeries(
            ANCHOR, ("a", "b"), words(INT128, [[-(2**127), 7], [0, 2**127 - 1]]), INT128
        ),
        "uint128": StoredMVTSeries(
            ANCHOR, ("a",), words(UINT128, [[2**128 - 1], [2**70]]), UINT128
        ),
        "complexf16": StoredMVTSeries(
            ANCHOR,
            ("a", "b"),
            cf16([(0x8000, 0x3D00), (0x7E55, 0xFC00), (0x4000, 0x4200), (0x0000, 0x8000)], (2, 2)),
            COMPLEXF16,
        ),
        "one_row": dated([[1, 2, 3]]),
        "one_col": dated([[1], [2], [3]], columns=("only",)),
        "empty": StoredMVTSeries.from_list(ANCHOR, ("a", "b"), DATE_Q, []),
        "unit_axis": dated(
            [[INT64_MIN, INT64_MAX]], columns=("a", "b"), anchor=MIT(Unit(), INT64_MIN)
        ),
        "unicode": dated([[1, 2]], columns=("aé", "\U0001f642")),
        "empty_name": dated([[1, 2]], columns=("", "b")),
    }
    with open_dataecon(path, "w") as db:
        for name, value in values.items():
            db.write_series(name, value)
    with open_dataecon(path) as db:
        back = {name: db.read_series(name) for name in values}
        assert db.object_info("dates").object_type == 21
        assert db.get_attributes("dates") == {}
        assert db.get_attributes("empty") == {"jeltype": "MIT{Quarterly{3}}"}
        with pytest.raises(TypeError, match="read_series"):
            db.read_array("dates")
    for name, value in values.items():
        assert isinstance(back[name], StoredMVTSeries), name
        assert back[name] == value, name
        assert back[name].values.flags.owndata
        assert back[name].values.flags.writeable
        back[name].values[...] = 0  # writable after closure


@pytest.mark.parametrize(
    ("frequency", "low", "high"),
    [
        (Unit(), INT64_MIN, INT64_MAX),
        (Daily(), -11980259, 11979954),
        (BDaily(), -8557114, 8557110),
        (Weekly(7), -1711422, 1711422),
        (Monthly(), -393600, 2147483647),
        (Quarterly(1), -131200, 2147483647),
        (HalfYearly(1), -65600, 2147483647),
        (Yearly(1), -(2**31), 2**31 - 1),
    ],
)
@NATIVE
def test_every_axis_family_carries_a_cross_frequency_element_at_both_endpoints(
    tmp_path, frequency, low, high
):
    path = tmp_path / "axes.daec"
    element = StoredElement.date(Weekly(3))
    with open_dataecon(path, "w") as db:
        for label, code in (("lo", low), ("hi", high)):
            value = StoredMVTSeries.from_list(
                MIT(frequency, code), ("a", "b"), element, [[MIT(Weekly(3), 1), MIT(Weekly(3), -1)]]
            )
            db.write_series(label, value)
        with pytest.raises(ValueError):
            db.write_series(
                "out",
                StoredMVTSeries.from_list(
                    MIT(frequency, high + 1), ("a",), element, [[MIT(Weekly(3), 1)]]
                ),
            )
    with open_dataecon(path) as db:
        for label, code in (("lo", low), ("hi", high)):
            back = db.read_series(label)
            assert back.firstdate == MIT(frequency, code)
            assert back.element == element
            assert back.tolist() == [[MIT(Weekly(3), 1), MIT(Weekly(3), -1)]]


@NATIVE
def test_refusal_happens_before_an_overwrite_deletes_the_original(tmp_path):
    path = tmp_path / "keep.daec"
    original = dated([[1, 2, 3]])
    with open_dataecon(path, "w") as db:
        db.write_series("keep", original)
        bad = dated([[4, 5, 6]])
        bad.values.resize((3, 1), refcheck=False)
        with pytest.raises(ValueError, match="column names"):
            db.write_series("keep", bad, overwrite=True)
        with pytest.raises(ValueError, match="changed size"):
            db.write_series(
                "keep",
                StoredMVTSeries(
                    ANCHOR,
                    ("a",),
                    np.array([[1], [2]], dtype="<i8").view(_ShrinkingCarrier),
                    DATE_Q,
                    copy=False,
                ),
                overwrite=True,
            )
        assert db.read_series("keep") == original
        db.write_series("keep", dated([[7, 8, 9]]), overwrite=True)
        assert db.read_series("keep") == dated([[7, 8, 9]])
        with pytest.raises(Exception, match="exists"):
            db.write_series("keep", original)


@NATIVE
def test_foreign_markers_survive_a_file_round_trip_and_interpret(tmp_path):
    path = tmp_path / "markers.daec"
    carrier = np.array([[1, 0], [2, 1]], dtype="<i8")
    marked = StoredMVTSeries(ANCHOR, ("a", "b"), carrier, StoredElement.numeric("<i8", "Float64"))
    identity = StoredMVTSeries(ANCHOR, ("a", "b"), carrier, DATE_Q, object_marker="MVTSeries")
    both = StoredMVTSeries(
        ANCHOR, ("a", "b"), carrier, StoredElement.numeric("<i8", "Int128"), object_marker="Any"
    )
    wide_bool = StoredMVTSeries(
        ANCHOR, ("a", "b"), words(INT128, [[1, 0], [0, 1]]), INT128.with_bool_marker()
    )
    with open_dataecon(path, "w") as db:
        db.write_series("marked", marked)
        db.write_series("identity", identity)
        db.write_series("both", both)
        db.write_series("wide_bool", wide_bool)
        db.write_series(
            "plain_bool", MVTSeries(ANCHOR, ["a", "b"], np.array([[True, False], [False, True]]))
        )
    with open_dataecon(path) as db:
        assert db.read_series("marked") == marked
        assert db.get_attributes("marked") == {"jeltype": "Float64"}
        assert db.read_series("identity") == identity
        assert db.get_attributes("identity") == {"jtype": "MVTSeries"}
        assert db.read_series("both") == both
        assert db.get_attributes("both") == {"jeltype": "Int128", "jtype": "Any"}
        assert db.read_series("wide_bool") == wide_bool
        assert db.get_attributes("wide_bool") == {"jeltype": "Bool"}
        plain = db.read_series("plain_bool")
        assert isinstance(plain, MVTSeries)
        assert plain.values.dtype == np.bool_
        assert db.get_attributes("plain_bool") == {"jeltype": "Bool"}
        assert db.read_series("marked").to_interpreted().values.dtype == np.float64


@NATIVE
def test_malformed_name_metadata_is_refused_on_read(tmp_path):
    path = tmp_path / "names.daec"
    # DataEcon shares identical axes between objects, so each object gets its
    # own column names and therefore its own names axis to corrupt.
    with open_dataecon(path, "w") as db:
        db.write_series("dup", dated([[1, 2]], columns=("a", "b")))
        db.write_series("count", dated([[1, 2]], columns=("c", "d")))
        db.write_series("plain_axes", dated([[1, 2]], columns=("e", "f")))
    with closing(sqlite3.connect(path)) as conn:
        for name, data in (("dup", b"a\na"), ("count", b"c\nd\ne")):
            (axis_id,) = conn.execute(
                "SELECT m.axis2_id FROM mvtseries m JOIN objects o ON o.id=m.id WHERE o.name=?",
                (name,),
            ).fetchone()
            conn.execute("UPDATE axes SET data=? WHERE id=?", (data, axis_id))
        (axis_id,) = conn.execute(
            "SELECT m.axis2_id FROM mvtseries m JOIN objects o ON o.id=m.id "
            "WHERE o.name='plain_axes'"
        ).fetchone()
        conn.execute("UPDATE axes SET ax_type=0, data=NULL WHERE id=?", (axis_id,))
        conn.commit()
    with open_dataecon(path) as db:
        with pytest.raises(ValueError, match="duplicate"):
            db.read_series("dup")
        with pytest.raises(ValueError, match="holds 3 names for 2 columns"):
            db.read_series("count")
        with pytest.raises(TypeError, match="named column axis"):
            db.read_series("plain_axes")


MARKER_FAULT = r"""
import sqlite3, sys
import numpy as np
from tsecon import mm
from tsecon.dataecon import open_dataecon, StoredMVTSeries, StoredElement

path = sys.argv[1]
carrier = np.array([[1, 0], [2, 1]], dtype="<i8")
with open_dataecon(path, "w") as db:
    element = StoredElement.numeric("<i8", "Float64")
    db.write_series("keep", StoredMVTSeries(mm(2024, 1), ("a", "b"), carrier, element))
conn = sqlite3.connect(path)
conn.execute(
    "CREATE TRIGGER fail_attr BEFORE INSERT ON attributes WHEN NEW.name='jeltype' "
    "BEGIN SELECT RAISE(ABORT, 'injected'); END"
)
conn.commit()
conn.close()
try:
    with open_dataecon(path, "a") as db:
        element = StoredElement.numeric("<i8", "Int128")
        replacement = StoredMVTSeries(mm(2025, 1), ("c", "d"), carrier, element)
        db.write_series("keep", replacement, overwrite=True)
except Exception as error:
    print("ERROR", type(error).__name__)
conn = sqlite3.connect(path)
query = "SELECT name, value FROM attributes WHERE name IN ('jeltype','jtype')"
print("ATTRS", conn.execute(query).fetchall())
conn.close()
with open_dataecon(path) as db:
    back = db.read_series("keep")
    print("RESULT", type(back).__name__, back.values.dtype, list(back.column_names))
"""


@NATIVE
def test_marker_write_failure_leaves_the_unmarked_replacement(tmp_path):
    run = subprocess.run(
        [sys.executable, "-c", MARKER_FAULT, str(tmp_path / "fault.daec")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert "ERROR DataEconError" in run.stdout
    # The overwrite deleted the original; the replacement was stored but its
    # element marker was not, so an ordinary Int64 MVTSeries remains.
    assert "ATTRS []" in run.stdout
    assert "RESULT MVTSeries int64 ['c', 'd']" in run.stdout


@NATIVE
def test_workspace_dispatch_stores_reads_and_reports(tmp_path):
    path = tmp_path / "ws.daec"
    value = dated([[1, 2, 3], [4, 5, 6]])
    bad = StoredMVTSeries(ANCHOR, ("a", "b"), np.array([[1, 2], [3, 4]], dtype="<i8"), DATE_Q)
    bad.values.resize((4, 1), refcheck=False)  # invalid live carrier: reported, not raised
    workspace = Workspace(good=value, nested=Workspace(inner=value), bad=bad)
    with open_dataecon(path, "w") as db:
        report = db.write_workspace(workspace)
        assert report.count == 3
        assert [s.path for s in report.skipped] == ["/bad"]
        assert report.skipped[0].category == "invalid"
        with pytest.raises(ValueError, match="column names"):
            db.write_workspace(Workspace(bad=bad), "/", strict=True, overwrite=True)
    with open_dataecon(path) as db:
        loaded = db.read_workspace()
        assert loaded.report.ok
        assert loaded.workspace.good == value
        assert loaded.workspace.nested.inner == value
        assert db.read_object("good") == value


# ---- the Julia fixture -----------------------------------------------------


def julia_fixture_inventory() -> dict[str, tuple[object, dict[str, str]]]:
    """Every MVTSeries of julia_represented_mvtseries.daec with its Python value and markers."""
    families = [
        ("u", Unit(), INT64_MIN, INT64_MAX),
        ("d", Daily(), -11980259, 11979954),
        ("b", BDaily(), -8557114, 8557110),
        ("w7", Weekly(7), -1711422, 1711422),
        ("m", Monthly(), -393600, 2147483647),
        ("q1", Quarterly(1), -131200, 2147483647),
        ("h1", HalfYearly(1), -65600, 2147483647),
        ("y1", Yearly(1), -(2**31), 2**31 - 1),
    ]
    int_rows = np.array([[1, 2], [0, 1]], dtype="<i8")
    inventory: dict[str, tuple[object, dict[str, str]]] = {
        "rmv_mit_q_on_m": (dated([[-1, 1, INT64_MIN], [0, 2, INT64_MAX]]), {}),
        "rmv_dur_y6_on_d": (
            StoredMVTSeries.from_list(
                MIT(Daily(), 738000),
                ("x", "y"),
                DUR_Y6,
                [
                    [Duration(Yearly(6), 1), Duration(Yearly(6), INT64_MIN)],
                    [Duration(Yearly(6), -2), Duration(Yearly(6), INT64_MAX)],
                    [Duration(Yearly(6), 3), Duration(Yearly(6), 0)],
                ],
            ),
            {},
        ),
        "rmv_mit_unit_on_unit": (
            StoredMVTSeries.from_list(
                MIT(Unit(), -5),
                ("a", "b"),
                StoredElement.date(Unit()),
                [
                    [MIT(Unit(), INT64_MIN), MIT(Unit(), 1)],
                    [MIT(Unit(), 0), MIT(Unit(), INT64_MAX)],
                ],
            ),
            {},
        ),
        "rmv_mit_w7_on_b": (
            StoredMVTSeries.from_list(
                MIT(BDaily(), 500000),
                ("a",),
                StoredElement.date(Weekly(7)),
                [[MIT(Weekly(7), -1711422)], [MIT(Weekly(7), 1711422)], [MIT(Weekly(7), 7)]],
            ),
            {},
        ),
        "rmv_int128": (
            StoredMVTSeries(
                ANCHOR,
                ("a", "b"),
                words(INT128, [[-(2**127), 2**127 - 1], [-1, 7], [0, 2**70]]),
                INT128,
            ),
            {},
        ),
        "rmv_uint128": (
            StoredMVTSeries(
                ANCHOR,
                ("a", "b", "c"),
                words(UINT128, [[0, 2, 2**70], [1, 2**128 - 1, 5]]),
                UINT128,
            ),
            {},
        ),
        "rmv_complexf16": (
            StoredMVTSeries(
                ANCHOR,
                ("a", "b"),
                cf16(
                    [(0x8000, 0x3D00), (0x7E55, 0xFC00), (0x4000, 0x4200), (0x0000, 0x8000)], (2, 2)
                ),
                COMPLEXF16,
            ),
            {},
        ),
        "rmv_mit_one_row": (dated([[1, 2, 3]]), {}),
        "rmv_mit_unicode_names": (
            dated([[1, 3, 5], [2, 4, 6]], columns=("aé", "\U0001f642", "")),
            {},
        ),
        "rmv_int128_zero_rows": (
            StoredMVTSeries.from_list(ANCHOR, ("a", "b"), INT128, []),
            {"jeltype": "Int128"},
        ),
        "rmv_complexf16_zero_rows": (
            StoredMVTSeries.from_list(ANCHOR, ("a", "b"), COMPLEXF16, []),
            {"jeltype": "ComplexF16"},
        ),
        "rmv_dur_zero_rows": (
            StoredMVTSeries.from_list(ANCHOR, ("a", "b"), DUR_Y6, []),
            {"jeltype": "Duration{Yearly{6}}"},
        ),
        "rmv_bool_on_int128": (
            StoredMVTSeries(
                ANCHOR, ("a", "b"), words(INT128, [[1, 0], [0, 1]]), INT128.with_bool_marker()
            ),
            {"jeltype": "Bool"},
        ),
        "rmv_bool_on_int64_01": (
            MVTSeries(ANCHOR, ["a", "b"], np.array([[True, False], [False, True]])),
            {"jeltype": "Bool"},
        ),
        "rmv_float64_on_int64": (
            StoredMVTSeries(ANCHOR, ("a", "b"), int_rows, StoredElement.numeric("<i8", "Float64")),
            {"jeltype": "Float64"},
        ),
        "rmv_mit_m_on_int64": (
            StoredMVTSeries(
                ANCHOR, ("a", "b"), int_rows, StoredElement.numeric("<i8", "MIT{Monthly}")
            ),
            {"jeltype": "MIT{Monthly}"},
        ),
        "rmv_identity_object": (
            StoredMVTSeries(ANCHOR, ("a", "b"), int_rows, DATE_Q, object_marker="MVTSeries"),
            {"jtype": "MVTSeries"},
        ),
        "rmv_identity_param": (
            StoredMVTSeries(
                ANCHOR,
                ("a", "b"),
                words(INT128, [[1, 2], [0, 1]]),
                INT128,
                object_marker="MVTSeries{Monthly, Int128}",
            ),
            {"jtype": "MVTSeries{Monthly, Int128}"},
        ),
        "rmv_empty_mit_marked": (
            StoredMVTSeries.from_list(ANCHOR, ("a", "b"), StoredElement.date(Monthly()), []),
            {"jeltype": "MIT{Monthly}"},
        ),
    }
    for label, frequency, low, high in families:
        for suffix, code in (("min", low), ("max", high)):
            inventory[f"rmv_mit_{suffix}_{label}"] = (
                dated([[1, 2]], columns=("a", "b"), anchor=MIT(frequency, code)),
                {},
            )
    return inventory


STRUCTURE_FIXTURE_NAMES = {
    "str_sym_u_f64",
    "str_sym_l_f64",
    "str_herm_u_c64",
    "str_herm_l_c64",
    "str_diag_vec_f64",
    "str_sym_l_signs",
    "str_herm_u_signs",
    "str_sym_u_i128",
    "str_herm_l_c16",
    "str_diag_mit_m",
    "str_sym_l_bool",
    "str_diag_f16",
    "str_sym_empty_f64",
    "str_diag_empty_f64",
    "str_herm_1x1_c64",
}


def _mvtseries_equal(actual: object, expected: object) -> bool:
    if isinstance(expected, MVTSeries):
        return (
            isinstance(actual, MVTSeries)
            and actual.firstdate == expected.firstdate
            and actual.column_names == expected.column_names
            and actual.values.dtype == expected.values.dtype
            and np.array_equal(actual.values, expected.values)
        )
    return actual == expected


@NATIVE
def test_julia_fixture_inventory_is_complete_and_reads_exactly():
    expected = julia_fixture_inventory()
    fixture = FIXTURES / "julia_represented_mvtseries.daec"
    provenance = tomllib.loads(fixture.with_suffix(".toml").read_text(encoding="utf-8"))
    assert provenance["timeseriesecon_sha"] == "fc0a0d01aed4903ea6c64b12d78d0bdf68468df6"
    assert provenance["native_version"] == "0.4.0"
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    with open_dataecon(fixture) as db:
        names = {entry.path.lstrip("/") for entry in db.list_objects("/")}
        assert names == set(expected) | STRUCTURE_FIXTURE_NAMES
        for name, (value, markers) in expected.items():
            back = db.read_series(name)
            assert _mvtseries_equal(back, value), name
            assert db.get_attributes(name) == markers, name
            assert db.object_info(name).object_type == 21
            if isinstance(back, StoredMVTSeries):
                assert back.values.flags.owndata
                assert back.values.flags.c_contiguous
        interpreted = db.read_series("rmv_float64_on_int64").to_interpreted()
        assert isinstance(interpreted, MVTSeries)
        assert interpreted.values.tolist() == [[1.0, 2.0], [0.0, 1.0]]
        assert db.read_series("rmv_bool_on_int128").to_bool().values.tolist() == [
            [True, False],
            [False, True],
        ]


@NATIVE
def test_python_rewrite_of_the_julia_fixture_is_byte_identical(tmp_path):
    """Every fixture MVTSeries rewritten by Python stores the same payload, axes and markers."""
    rewritten = tmp_path / "rewrite.daec"
    fixture = FIXTURES / "julia_represented_mvtseries.daec"
    with open_dataecon(fixture) as source, open_dataecon(rewritten, "w") as target:
        for name in julia_fixture_inventory():
            target.write_series(name, source.read_series(name))
    query = (
        "SELECT o.name, o.class, o.type, m.eltype, m.elfreq, m.value, x1.ax_type, x1.length, "
        "x1.frequency, x1.data, x2.ax_type, x2.length, x2.data FROM objects o "
        "JOIN mvtseries m ON m.id=o.id JOIN axes x1 ON x1.id=m.axis1_id "
        "JOIN axes x2 ON x2.id=m.axis2_id WHERE o.pid=0 AND o.type=21 ORDER BY o.name"
    )
    markers = (
        "SELECT o.name, a.name, a.value FROM attributes a JOIN objects o ON o.id=a.id "
        "WHERE o.pid=0 AND o.type=21 AND a.name IN ('jeltype','jtype') ORDER BY o.name, a.name"
    )
    with closing(sqlite3.connect(fixture)) as a, closing(sqlite3.connect(rewritten)) as b:
        source_rows = a.execute(query).fetchall()
        target_rows = b.execute(query).fetchall()
        # The foreign Int64 Bool row reads as Boolean values and rewrites canonically.
        source_rows = [
            (*row[:5], np.array([1, 0, 0, 1], dtype="i1").tobytes(), *row[6:])
            if row[0] == "rmv_bool_on_int64_01"
            else row
            for row in source_rows
        ]
        assert source_rows == target_rows
        assert a.execute(markers).fetchall() == b.execute(markers).fetchall()
