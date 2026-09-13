"""Represented series through the real codec, owner and native ABI."""

import importlib.util
import json
import sqlite3
import struct
import subprocess
import sys
import warnings
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

import tsecon.dataecon._codec as codec
from tsecon import MIT, TSeries, mm
from tsecon.dataecon import (
    COMPLEXF16,
    INT128,
    UINT128,
    DataEconError,
    StoredElement,
    StoredSeries,
    open_dataecon,
)
from tsecon.dataecon._codec import MAX_BYTES, decode_series, encode_series, validate_metadata
from tsecon.frequencies import BDaily, Daily, HalfYearly, Monthly, Quarterly, Unit, Weekly, Yearly

NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built",
)
FREQUENCIES = [
    Unit(),
    Daily(),
    BDaily(),
    Monthly(),
    *[Weekly(d) for d in range(1, 8)],
    *[Quarterly(m) for m in range(1, 4)],
    *[HalfYearly(m) for m in range(1, 7)],
    *[Yearly(m) for m in range(1, 13)],
]
WIDE = [INT128, UINT128, COMPLEXF16]


@NATIVE
@pytest.mark.parametrize("element", WIDE)
@pytest.mark.parametrize("empty", [False, True])
def test_existing_julia_fixture_wide_controls_are_now_supported(element, empty):
    fixture = Path(__file__).parent / "fixtures/julia_series_elements.daec"
    name = "unsupported_" + element.julia_name + ("_empty" if empty else "")
    with open_dataecon(fixture) as db:
        result = db.read_series(name)
    assert isinstance(result, StoredSeries)
    assert result.element == element
    assert result.tolist() == ([] if empty else [1, 2])
    with closing(sqlite3.connect(fixture)) as sql:
        payload = sql.execute(
            "SELECT t.value FROM tseries t JOIN objects o ON o.id=t.id WHERE o.name=?", (name,)
        ).fetchone()[0]
    assert result.values.tobytes() == (payload or b"")


@NATIVE
def test_existing_julia_foreign_bool_control_is_now_supported():
    fixture = Path(__file__).parent / "fixtures/julia_series_elements.daec"
    with open_dataecon(fixture) as db:
        result = db.read_series("foreign_bool")
    assert isinstance(result, TSeries)
    assert result.values.dtype == np.dtype(bool)
    assert result.values.tolist() == [True]


def wide_values(element):
    if element == INT128:
        return [-(1 << 127), -1, 0, (1 << 127) - 1]
    if element == UINT128:
        return [0, 1 << 64, (1 << 128) - 1]
    return [complex(-0.0, 1.25), complex(2.0**-24, -65504)]


@pytest.mark.parametrize("frequency", FREQUENCIES)
@pytest.mark.parametrize("kind", ["date", "duration"])
@NATIVE
def test_element_frequency_and_full_codes_independent_of_axis(tmp_path, frequency, kind):
    element = StoredElement(kind, frequency)
    values = np.array([-(1 << 63), -1, 0, (1 << 63) - 1], dtype="<i8")
    anchor = MIT(Daily(), 739191)
    path = tmp_path / "codes.daec"
    with open_dataecon(path, "a") as db:
        for name, carrier in [("full", values), ("empty", values[:0])]:
            db.write_series(name, StoredSeries(anchor, carrier, element))
            result = db.read_series(name)
            assert isinstance(result, StoredSeries)
            assert result.element == element
            assert result.firstdate == anchor
            assert result.values.tobytes() == carrier.tobytes()
            assert result.values.flags.owndata
            assert result.values.flags.writeable
    assert result.firstdate == anchor  # no borrowed storage after close
    with closing(sqlite3.connect(path)) as sql:
        rows = sql.execute(
            "SELECT t.eltype,t.elfreq,a.frequency,a.data,LENGTH(t.value) "
            "FROM tseries t JOIN axes a ON a.id=t.axis_id ORDER BY t.id"
        ).fetchall()
        assert rows == [
            (element.native_kind, element.native_frequency, 12, 739191, 32),
            (element.native_kind, element.native_frequency, 12, 739191, None),
        ]
        assert sql.execute("SELECT value FROM attributes WHERE name='jeltype'").fetchall() == [
            (element.julia_name,)
        ]


@pytest.mark.parametrize("element", WIDE)
@pytest.mark.parametrize("empty", [False, True])
@NATIVE
def test_wide_round_trip_exact_bytes_and_markers(tmp_path, element, empty):
    original = StoredSeries.from_list(mm(2024, 11), element, [] if empty else wide_values(element))
    encoded = encode_series(original)
    assert encoded.element_frequency == 0
    assert decode_series(*encoded) == original
    with open_dataecon(tmp_path / "wide.daec", "a") as db:
        db.write_series("value", original)
        result = db.read_series("value")
        assert result == original
        db.write_series("copy", result)
        assert db.read_series("copy") == original
    assert result.values.flags.owndata
    assert result.values.flags.writeable


@pytest.mark.parametrize("element", WIDE)
@NATIVE
def test_wide_bool_preserves_carrier_and_marker_on_rewrite(tmp_path, element):
    items = [0j, 1 + 0j] if element == COMPLEXF16 else [0, 1]
    source = StoredSeries.from_list(mm(2024, 11), element.with_bool_marker(), items)
    path = tmp_path / "bool.daec"
    with open_dataecon(path, "a") as db:
        db.write_series("original", source)
        result = db.read_series("original")
        assert result == source
        db.write_series("rewrite", result)
        db.write_series("converted", result.to_bool())
    with closing(sqlite3.connect(path)) as sql:
        rows = sql.execute(
            "SELECT o.name,t.eltype,t.elfreq,t.value FROM tseries t "
            "JOIN objects o ON o.id=t.id ORDER BY o.id"
        ).fetchall()
        assert rows == [
            ("original", element.native_kind, 0, source.values.tobytes()),
            ("rewrite", element.native_kind, 0, source.values.tobytes()),
            ("converted", 1, 0, b"\0\1"),
        ]
        assert (
            sql.execute("SELECT value FROM attributes WHERE name='jeltype'").fetchall()
            == [("Bool",)] * 3
        )


@pytest.mark.parametrize(
    ("element", "frequency", "width"),
    [(1, 32, 4), (3, 32, 16), (2, 32, 8), (4, 12, 8), (5, 32, 8), (3, 0, 8)],
)
def test_bad_element_metadata_before_decode(element, frequency, width):
    with pytest.raises((TypeError, ValueError)):
        decode_series(32, 24298, bytes(width), element, frequency, 1, None)


@pytest.mark.parametrize("frequency", [14, 16, *range(24, 32), 64, 128, 256])
@pytest.mark.parametrize("kind", [1, 3])
def test_noncanonical_element_codes_refused(frequency, kind):
    with pytest.raises(TypeError):
        validate_metadata((2, 12, kind, frequency, 1, 1, 32, 24298, 8))


@pytest.mark.parametrize(("kind", "dtype"), [(1, "<i2"), (2, "<u8"), (4, "<f2"), (5, "<c16")])
def test_ordinary_foreign_bool_validation_and_empty(kind, dtype):
    for values in ([0, 1, 0], []):
        payload = np.array(values, dtype=dtype).tobytes()
        result = decode_series(32, 24298, payload, kind, 0, len(values), "Bool")
        assert isinstance(result, TSeries)
        assert result.values.dtype == np.dtype(bool)
        assert result.values.tolist() == [bool(v) for v in values]
    with pytest.raises(ValueError):
        decode_series(32, 24298, np.array([2], dtype=dtype).tobytes(), kind, 0, 1, "Bool")


@pytest.mark.parametrize(
    ("kind", "frequency"), [(1, 32), (3, 32), (3, 0), (6, 0), (2, 32), (4, 12)]
)
def test_empty_bool_does_not_bypass_element_validation(kind, frequency):
    with pytest.raises(TypeError):
        decode_series(32, 24298, b"", kind, frequency, 0, "Bool")


@NATIVE
def test_invalid_represented_overwrite_keeps_original(tmp_path):
    with open_dataecon(tmp_path / "overwrite.daec", "a") as db:
        db.write_scalar("keep", 7)
        source = StoredSeries.from_list(mm(2024, 11), INT128.with_bool_marker(), [0, 1])
        source.values.resize((0,), refcheck=False)
        with pytest.raises(ValueError):
            db.write_series("keep", source, overwrite=True)
        assert db.read_scalar("keep") == 7
        source = StoredSeries.from_list(MIT(Monthly(), 2147483640), INT128, [1])
        with pytest.raises(ValueError, match="monthly"):
            db.write_series("keep", source, overwrite=True)
        assert db.read_scalar("keep") == 7


@NATIVE
def test_native_explicit_fields_and_element_type_checks(tmp_path):
    with open_dataecon(tmp_path / "fields.daec", "a") as db:
        with pytest.raises(TypeError):
            db._handle.write("s", 32, 24298, bytes(8))
        for bad in [True, 32.0, "32"]:
            with pytest.raises(TypeError):
                db._handle.write("s", 32, 24298, bytes(8), False, 3, bad, 1, None)
        # Element code uses all Int64 bits, independent of native date packing.
        db._handle.write("s", 32, 24298, struct.pack("<q", -(1 << 63)), False, 3, 12, 1, None)
        assert db.read_series("s").values.tolist() == [-(1 << 63)]


FAULT = r"""
import json,sys,numpy as np
from tsecon import mm
from tsecon.dataecon import open_dataecon,DataEconError,StoredSeries,INT128,COMPLEXF16
path,kind=sys.argv[1:]
element=INT128 if kind=='integer' else COMPLEXF16
db=open_dataecon(path,'a')
try:
    db.write_series('target',StoredSeries.from_list(mm(2024,11),element,[]))
except DataEconError as exc:
    error=exc.code
else:
    raise AssertionError('injection did not fail')
result=db.read_series('target')
try:
    db.close()
except DataEconError as exc:
    close_error=exc.code
else:
    raise AssertionError('expected finalization failure')
print(json.dumps([error,str(result.values.dtype),len(result),close_error,db.closed]))
"""


@NATIVE
@pytest.mark.parametrize(("kind", "dtype"), [("integer", "int64"), ("complex", "complex128")])
def test_empty_wide_marker_failure_residue_and_close(tmp_path, kind, dtype):
    path = tmp_path / "marker.daec"
    with open_dataecon(path, "a") as db:
        db.write_scalar("other", 9)
    with closing(sqlite3.connect(path)) as sql, sql:
        sql.execute(
            "CREATE TRIGGER fail_marker BEFORE INSERT ON attributes "
            "WHEN NEW.name='jeltype' BEGIN SELECT RAISE(ABORT,'injected'); END"
        )
    run = subprocess.run(
        [sys.executable, "-c", FAULT, str(path), kind],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert json.loads(run.stdout) == [19, dtype, 0, 19, True]


# ---- Ownership, mutation, the snapshot guard, capacity and axis families ----

AXES = [f for f in FREQUENCIES if f != Unit()]


def absent(db, name):
    with pytest.raises(DataEconError) as info:
        db.read_series(name)
    return info.value.code == -989


@NATIVE
@pytest.mark.parametrize("copy", [True, False])
def test_read_results_own_their_storage_after_close_regardless_of_copy(tmp_path, copy):
    source = np.array([-(1 << 63), 7], dtype="<i8")
    series = StoredSeries(mm(2024, 11), source, StoredElement.duration(Weekly(7)), copy=copy)
    assert np.shares_memory(series.values, source) is not copy
    with open_dataecon(tmp_path / "own.daec", "a") as db:
        db.write_series("s", series)
        result = db.read_series("s")
    source[...] = 0
    series.values[...] = 1
    assert result.values.tolist() == [-(1 << 63), 7]
    assert result.values.flags.owndata
    assert result.values.flags.writeable
    assert result.values.flags.c_contiguous
    assert not np.shares_memory(result.values, series.values)
    result.values[0] = 3  # writable after the file is closed


@NATIVE
def test_edit_in_place_rewrite_and_reread(tmp_path):
    path = tmp_path / "edit.daec"
    with open_dataecon(path, "a") as db:
        db.write_series("s", StoredSeries.from_list(mm(2024, 11), INT128, [1, 2, 3]))
        edited = db.read_series("s")
        edited.values["hi"][1] = 1  # 2 + 2**64, without any conversion
        edited.values["lo"][2] = (1 << 64) - 1
        db.write_series("s", edited, overwrite=True)
    with open_dataecon(path) as db:
        again = db.read_series("s")
    assert again == edited
    assert again.tolist() == [1, 2 + (1 << 64), (1 << 64) - 1]


@NATIVE
@pytest.mark.parametrize("mutation", ["shape", "dtype", "strides"])
def test_in_place_shape_dtype_or_layout_change_is_refused_before_any_native_call(
    tmp_path, mutation
):
    series = StoredSeries.from_list(mm(2024, 11), INT128, [1, 2, 3, 4])
    if mutation == "shape":
        series.values.resize((2, 2), refcheck=False)
        expected = ValueError
    elif mutation == "strides":
        expected = ValueError
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                series.values.strides = (0,)
            except AttributeError:
                pytest.skip("NumPy removed in-place strides assignment.")
        assert all(issubclass(w.category, DeprecationWarning) for w in caught)
    else:
        expected = TypeError
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                series.values.dtype = np.dtype([("a", "<u8"), ("b", "<u8")])
            except AttributeError:
                pytest.skip("NumPy removed in-place dtype assignment.")
        assert all(issubclass(w.category, DeprecationWarning) for w in caught)
    with open_dataecon(tmp_path / "mutated.daec", "a") as db:
        db.write_scalar("keep", 1)
        with pytest.raises(expected):
            db.write_series("s", series)
        with pytest.raises(expected):
            db.write_series("keep", series, overwrite=True)
        assert absent(db, "s")
        assert db.read_scalar("keep") == 1


@NATIVE
def test_wrong_dtype_containers_are_refused_before_any_native_call(tmp_path):
    with open_dataecon(tmp_path / "dtype.daec", "a") as db:
        with pytest.raises(TypeError):
            db.write_series("s", TSeries(mm(2024, 11), np.zeros(2, dtype=INT128.dtype)))
        with pytest.raises(TypeError):
            db.write_series("s", TSeries(mm(2024, 11), np.zeros(2, dtype=COMPLEXF16.dtype)))
        with pytest.raises(TypeError):
            db.write_series("s", TSeries(mm(2024, 11), np.zeros(2, dtype=object)))
        with pytest.raises(TypeError):
            StoredSeries(mm(2024, 11), np.zeros(2, dtype="<i8"), INT128)
        assert absent(db, "s")


class EmptiedAfterValidation(StoredSeries):
    """Deterministic interleaving: the carrier is emptied right after validate()."""

    __slots__ = ()

    def validate(self):
        super().validate()
        self.values.resize((0,), refcheck=False)


class CorruptedAfterValidation(StoredSeries):
    """Deterministic interleaving: a marked carrier turns invalid after validate()."""

    __slots__ = ()

    def validate(self):
        super().validate()
        if self.element.kind == "complexf16":
            self.values["imag"][0] = np.float16(1)
        else:
            self.values["hi"][0] = 1


def marked_items(element):
    return [0j, 1 + 0j] if element == COMPLEXF16 else [0, 1]


@NATIVE
@pytest.mark.parametrize("element", WIDE)
def test_carrier_emptied_between_validation_and_snapshot_is_refused(tmp_path, element):
    # Without a descriptor-aware recheck an emptied "Bool"-marked wide carrier
    # would resolve to the ambiguous empty Boolean encoding and lose its width.
    series = EmptiedAfterValidation.from_list(
        mm(2024, 11), element.with_bool_marker(), marked_items(element)
    )
    with pytest.raises(ValueError, match="nothing was written"):
        encode_series(series)
    assert len(series) == 0
    assert series.element.marker == "Bool"
    with open_dataecon(tmp_path / "race.daec", "a") as db:
        db.write_scalar("keep", 1)
        series = EmptiedAfterValidation.from_list(
            mm(2024, 11), element.with_bool_marker(), marked_items(element)
        )
        with pytest.raises(ValueError, match="nothing was written"):
            db.write_series("keep", series, overwrite=True)
        assert db.read_scalar("keep") == 1
    # An unmarked carrier emptied at the same point is a valid empty series of
    # its own width: the descriptor still names the element and its marker.
    plain = EmptiedAfterValidation.from_list(mm(2024, 11), element, marked_items(element))
    encoded = encode_series(plain)
    assert (encoded.length, encoded.marker, encoded.payload) == (0, element.julia_name, b"")


@NATIVE
@pytest.mark.parametrize("element", WIDE)
def test_carrier_corrupted_between_validation_and_snapshot_is_refused(element):
    series = CorruptedAfterValidation.from_list(
        mm(2024, 11), element.with_bool_marker(), marked_items(element)
    )
    with pytest.raises(ValueError, match="exact zeros and ones"):
        encode_series(series)


@pytest.mark.parametrize("element", WIDE)
def test_carrier_resized_between_length_capture_and_snapshot_is_refused(monkeypatch, element):
    series = StoredSeries.from_list(mm(2024, 11), element.with_bool_marker(), marked_items(element))
    original = codec.validate_metadata

    def resize_then_validate(metadata):
        series.values.resize((0,), refcheck=False)
        return original(metadata)

    monkeypatch.setattr(codec, "validate_metadata", resize_then_validate)
    with pytest.raises(ValueError, match="changed size"):
        encode_series(series)
    monkeypatch.setattr(codec, "validate_metadata", original)
    grown = StoredSeries.from_list(mm(2024, 11), element.with_bool_marker(), marked_items(element))

    def grow_then_validate(metadata):
        grown.values.resize((3,), refcheck=False)
        return original(metadata)

    monkeypatch.setattr(codec, "validate_metadata", grow_then_validate)
    with pytest.raises(ValueError, match="changed size"):
        encode_series(grown)


@NATIVE
def test_invalid_marked_mutation_keeps_the_original_on_overwrite(tmp_path):
    with open_dataecon(tmp_path / "keep.daec", "a") as db:
        db.write_series("keep", StoredSeries.from_list(mm(2024, 11), UINT128, [5]))
        source = StoredSeries.from_list(mm(2024, 11), UINT128.with_bool_marker(), [0, 1])
        source.values["lo"][1] = 2
        with pytest.raises(ValueError, match="observation 1"):
            db.write_series("keep", source, overwrite=True)
        assert db.read_series("keep").tolist() == [5]
        assert source.element.marker == "Bool"
        assert source.tolist() == [0, 2]


@pytest.mark.parametrize(
    ("element", "frequency", "width"),
    [(1, 0, 16), (2, 0, 16), (5, 0, 4), (3, 32, 8), (1, 12, 8)],
)
def test_capacity_per_represented_width_without_allocation(element, frequency, width):
    length = MAX_BYTES // width
    validate_metadata((2, 12, element, frequency, 1, length, 12, 11979954, MAX_BYTES))
    with pytest.raises(ValueError):
        validate_metadata(
            (2, 12, element, frequency, 1, length + 1, 12, 11979954, MAX_BYTES + width)
        )
    with pytest.raises(ValueError):
        validate_metadata((2, 12, element, frequency, 1, length, 12, 11979954, MAX_BYTES - 1))
    with pytest.raises(ValueError):
        validate_metadata((2, 12, element, frequency, 1, 2, 268, 2**31 - 1, 2 * width))


@NATIVE
@pytest.mark.parametrize("frequency", AXES)
@pytest.mark.parametrize("element", [StoredElement.date(Quarterly(2)), INT128])
def test_represented_elements_over_every_axis_family(tmp_path, frequency, element):
    first = MIT(frequency, 100)
    if element.is_wide:
        values = np.frombuffer(bytes.fromhex("ff" * 16 + "00" * 16), dtype=element.dtype).copy()
    else:
        values = np.array([-(1 << 63), 8099], dtype="<i8")
    with open_dataecon(tmp_path / "axis.daec", "a") as db:
        for empty in (False, True):
            source = StoredSeries(first, values[:0] if empty else values, element)
            db.write_series("s", source, overwrite=True)
            actual = db.read_series("s")
            assert actual == source
            assert actual.firstdate == first
            assert actual.frequency == frequency
            assert actual.lastdate == MIT(frequency, 100 + len(source) - 1)
