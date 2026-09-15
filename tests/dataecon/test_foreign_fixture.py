"""Julia-written foreign reconstruction markers: preservation and interpretation.

Every ``fx_*`` object in ``julia_foreign_markers.daec`` was stored through the C
ABI with a ``jeltype``/``jtype`` attribute Julia's own writer never emits for it.
The generator loaded each object with the pinned Julia loader and materialized
the outcome in the fixture: a ``<name>_julia`` sibling written by Julia's own
writer from the loaded value, or a ``<name>_error`` string scalar naming the
exception. Python must preserve every loadable object (stored kind, bytes and
exact marker text) and its explicit interpretation, written through Python's
own writer, must reproduce Julia's sibling; every object Julia cannot load
must be refused. Nothing is evaluated from marker text.
"""

import hashlib
import importlib.util
import sqlite3
import tomllib
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from tsecon import TSeries, mm
from tsecon.dataecon import DataEconError, StoredSeries, open_dataecon
from tsecon.dataecon._metadata import JULIA_KIND_DEFAULTS, JULIA_NUMERIC_TYPES

NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built",
)
FIXTURE = Path(__file__).parent / "fixtures/julia_foreign_markers.daec"
ANCHOR = mm(2024, 11)  # code 24298
OBJECT_COUNT = 1394  # 697 cases, 477 Julia-written siblings, 220 error scalars
DTYPE_NAMES = {dtype: name for name, (_, dtype) in JULIA_NUMERIC_TYPES.items()}
# Objects Python reads as an ordinary TSeries rather than a preserved container:
# a Bool marker on an ordinary payload and the pre-existing canonical empty
# numeric marker behavior.
CANONICAL = {
    "fx_m_Int8_Bool": np.dtype(bool),
    "fx_m_Float64_Bool": np.dtype(bool),
    "fx_e_Int64_Bool": np.dtype(bool),
    "fx_e_Int64_Int64": np.dtype("<i8"),
    "fx_e_Int64_Int16": np.dtype("<i2"),
    "fx_v_float64_zero_Bool": np.dtype(bool),
}
PRESERVED_SAMPLES = {
    "fx_m_Int8_Int8": ("Int8", None),
    "fx_m_Int64_Int64": ("Int64", None),
    "fx_int64_alias_as_int": ("Int", None),
    "fx_int16_as_int64": ("Int64", None),
    "fx_int16_alias_as_int": ("Int", None),
    "fx_outer_tseries_inactive_eltype": ("NoSuchElement", "TSeries"),
    "fx_outer_vector_empty": ("Int16", "Vector"),
    "fx_d_mit_monthly_Bool": ("Bool", None),
    "fx_m_Int128_Float64": ("Float64", None),
    "fx_e_ComplexF64_Int128": ("Int128", None),
}
# Every loadable row now has a verified spelling in Python's finite table (the
# no-space and leading-space TSeries identities were added with the shared
# spelling work); the set stays as the hook for future Julia-only rows.
JULIA_ONLY: set[str] = set()
# Identity tokens on represented sources read as the unmarked stored family.
CANONICAL_STORED = {
    "fx_m_Int128_Int128",
    "fx_d_mit_monthly_MIT_Monthly",
    "fx_d_dur_daily_Duration_Daily",
}
REFUSED_SAMPLES = [
    "fx_m_Int8_Duration_Monthly",
    "fx_m_Float64_MIT_Monthly",
    "fx_d_mit_monthly_two_as_bool",
    "fx_d_mit_monthly_empty_as_bool",
    "fx_d_mit_monthly_Int8",
    "fx_d_mit_monthly_Float16",
    "fx_v_float64_fraction_Int8",
    "fx_v_float64_special_Int64",
    "fx_v_int64_bounds_Int8",
    "fx_o_tseries_mismatch_type",
    "fx_o_vector_nonempty",
    "fx_o_unknown",
    "fx_o_empty_string",
    "fx_o_eltype_empty_string",
]


def connect():
    return closing(sqlite3.connect(f"{FIXTURE.resolve().as_uri()}?mode=ro", uri=True))


def raw_rows(path=FIXTURE):
    """Per name: (eltype, elfreq, axis, first, length, payload, jeltype, jtype)."""
    with closing(sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)) as sql:
        rows = sql.execute(
            "SELECT o.name,t.eltype,t.elfreq,a.frequency,a.data,a.length,t.value,"
            "(SELECT value FROM attributes x WHERE x.id=o.id AND x.name='jeltype'),"
            "(SELECT value FROM attributes x WHERE x.id=o.id AND x.name='jtype') "
            "FROM objects o JOIN tseries t ON t.id=o.id JOIN axes a ON a.id=t.axis_id"
        ).fetchall()
    return {row[0]: (*row[1:6], row[6] or b"", row[7], row[8]) for row in rows}


def error_scalars():
    with connect() as sql:
        rows = sql.execute(
            "SELECT o.name,s.value FROM objects o JOIN scalars s ON s.id=o.id "
            "WHERE o.name LIKE 'fx_%_error'"
        ).fetchall()
    return {name: value.rstrip(b"\0").decode() for name, value in rows}


def classify(rows, errors):
    cases = sorted(n for n in rows if n.startswith("fx_") and not n.endswith("_julia"))
    loadable = [n for n in cases if f"{n}_julia" in rows]
    refused = [n for n in cases if f"{n}_error" in errors]
    return cases, loadable, refused


def same_payload(kind, width, actual, expected):
    """Byte equality, NaN-aware for floating kinds (payload bits are not promised)."""
    if actual == expected:
        return True
    if kind not in (4, 5):
        return False
    # Compare complex values as separate real components. NaN payload/sign
    # differences are allowed only when both corresponding components are NaN;
    # all finite bits, signed zeros and infinities must match exactly.
    component_width = width if kind == 4 else width // 2
    dtype = np.dtype(f"<f{component_width}")
    x, y = np.frombuffer(actual, dtype=dtype), np.frombuffer(expected, dtype=dtype)
    if x.shape != y.shape:
        return False
    both_nan = np.isnan(x) & np.isnan(y)
    if not np.array_equal(np.isnan(x), np.isnan(y)):
        return False
    bits = np.dtype(f"<u{component_width}")
    return bool(np.array_equal(x.view(bits)[~both_nan], y.view(bits)[~both_nan]))


def test_nan_aware_payload_comparison_still_checks_other_bits():
    assert same_payload(4, 8, np.float64(np.nan).tobytes(), np.float64(-np.nan).tobytes())
    assert not same_payload(4, 8, np.float64(-0.0).tobytes(), np.float64(0.0).tobytes())
    assert not same_payload(
        5,
        16,
        np.complex128(complex(np.nan, 1.0)).tobytes(),
        np.complex128(complex(np.nan, 2.0)).tobytes(),
    )
    assert not same_payload(
        5,
        16,
        np.complex128(complex(1.0, -0.0)).tobytes(),
        np.complex128(complex(1.0, 0.0)).tobytes(),
    )


def test_fixture_provenance_and_coverage():
    provenance = tomllib.loads(FIXTURE.with_suffix(".toml").read_text(encoding="utf-8"))
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    assert provenance["timeseriesecon_sha"] == "fc0a0d01aed4903ea6c64b12d78d0bdf68468df6"
    assert provenance["native_source_sha"] == "1a108688a044380f808bebf64079e32dbb9cd1a4"
    assert provenance["native_version"] == "0.4.0"
    rows = raw_rows()
    errors = error_scalars()
    cases, loadable, refused = classify(rows, errors)
    assert len(rows) + len(errors) == OBJECT_COUNT
    assert set(loadable).isdisjoint(refused)
    assert set(loadable) | set(refused) == set(cases)
    assert len(rows) == len(cases) + len(loadable)
    assert set(CANONICAL) <= set(loadable)
    assert set(loadable) >= CANONICAL_STORED
    assert set(loadable) >= JULIA_ONLY
    assert set(PRESERVED_SAMPLES) <= set(loadable)
    assert set(REFUSED_SAMPLES) <= set(refused)
    assert len(loadable) == 477
    assert len(refused) == 220
    assert {errors[f"{n}_error"] for n in refused} == {
        "MethodError",
        "InexactError",
        "DimensionMismatch",
        "UndefVarError",
    }


@NATIVE
def test_refused_objects_match_julia_failures():
    rows = raw_rows()
    errors = error_scalars()
    _, _, refused = classify(rows, errors)
    with open_dataecon(FIXTURE) as db:
        for name in refused:
            with pytest.raises((TypeError, ValueError)):
                db.read_series(name)
            with pytest.raises(DataEconError):
                db.read_series(f"{name}_julia")


def read_and_interpret(db, name, row):
    """Python's read of one loadable object and the value it interprets to."""
    kind, elfreq, _axis, _first, _length, payload, jeltype, jtype = row
    result = db.read_series(name)
    if isinstance(result, TSeries):
        # A Bool marker on an ordinary payload, or a token naming the stored
        # element: canonical values, no marker retained.
        assert jeltype == "Bool" or jtype is None, name
        if name in CANONICAL:
            assert result.values.dtype == CANONICAL[name], name
        return result, result
    if result.element.marker is None and result.object_marker is None:
        assert name in CANONICAL_STORED or jeltype == result.element.julia_name, name
        assert result.values.tobytes() == payload, name
        return result, result
    assert name not in CANONICAL, name
    assert result.element.native_kind == kind, name
    assert result.element.native_frequency == elfreq, name
    assert result.values.tobytes() == payload, name
    assert result.element.marker == jeltype, name
    assert result.object_marker == jtype, name
    assert result.values.flags.owndata
    interpreted = result.to_interpreted()
    assert result.values.tobytes() == payload, name  # interpretation changed nothing
    return result, interpreted


def check_vector_interpretation(name, interpreted, sibling):
    """A Vector object marker interprets to an empty array of the sibling's element type."""
    assert interpreted.shape == (0,), name
    assert DTYPE_NAMES[interpreted.dtype] == (sibling[6] or JULIA_KIND_DEFAULTS[sibling[0]]), name


def check_written_like_julia(name, actual, sibling):
    """Python's written interpretation must match Julia's sibling row."""
    kind, elfreq, axis, first, length, payload, jeltype, jtype = sibling
    assert actual[:5] == (kind, elfreq, axis, first, length), name
    width = len(payload) // length if length else 0
    assert same_payload(kind, width, actual[5], payload), name
    # Julia writes the element token on every empty series; Python omits the
    # token that names the native default (an unmarked empty reads the same).
    expected_marker = jeltype
    if length == 0 and jeltype == JULIA_KIND_DEFAULTS.get(kind):
        expected_marker = None
    assert actual[6:] == (expected_marker, jtype), name


@NATIVE
def test_loadable_objects_are_preserved_and_interpreted_like_julia(tmp_path):
    rows = raw_rows()
    errors = error_scalars()
    _, loadable, _ = classify(rows, errors)
    path = tmp_path / "python.daec"
    results = {}
    with open_dataecon(FIXTURE) as db, open_dataecon(path, "a") as out:
        for name in loadable:
            if name in JULIA_ONLY:
                with pytest.raises(TypeError):
                    db.read_series(name)
                continue
            result, interpreted = read_and_interpret(db, name, rows[name])
            results[name] = result
            if isinstance(result, StoredSeries):
                out.write_series(f"{name}_again", result)
            if isinstance(interpreted, np.ndarray):
                check_vector_interpretation(name, interpreted, rows[f"{name}_julia"])
                continue
            out.write_series(f"{name}_interpreted", interpreted)
    for name, (marker, outer) in PRESERVED_SAMPLES.items():
        assert results[name].element.marker == marker
        assert results[name].object_marker == outer
    written = raw_rows(path)
    for name in loadable:
        if f"{name}_again" in written:
            expected = rows[name]
            if results[name].element.marker is None:
                # A redundant token naming the stored family is canonical: the
                # rewrite writes what the unmarked container writes (nothing for
                # a nonempty series, the exact type token for an empty one).
                token = results[name].element.written_marker(expected[4])
                expected = (*expected[:6], token, expected[7])
            assert written[f"{name}_again"] == expected, name
        if f"{name}_interpreted" in written:
            check_written_like_julia(name, written[f"{name}_interpreted"], rows[f"{name}_julia"])
