"""Julia-written marker-mapped, wide and exceptional scalars: the reference fixture.

Every ``sm_*`` object in ``julia_scalar_markers.daec`` is either a scalar Julia's
own writer stored (Int128/UInt128/ComplexF16, Symbol and other string forms,
Date/DateTime, Rational, integer complex, Irrational) or an ordinary payload
stored through the C ABI with an injected ``jtype``. The generator loaded each
one with the pinned Julia loader and materialized the outcome as siblings:
``<name>_type`` and ``<name>_value`` (the loaded type and a canonical text of
the value), ``<name>_julia`` with ``<name>_rewrite`` (Julia's own writer
re-storing the loaded value, and ``"<type>:<text>"`` or the exception name for
what that re-stored object loads as) and ``<name>_error`` (the exception name)
where the load fails.

Python does not read these forms yet: ``read_scalar`` refuses every one of them
without evaluating anything, the exact attributes stay readable, and a Workspace
read reports them as skipped members while loading the plain siblings. The
exact reconstruction helpers must reproduce each materialized outcome from the
stored bytes alone, and Julia's own rewrite losses are recorded facts, not
policies.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sqlite3
import struct
import tomllib
from contextlib import closing
from pathlib import Path

import pytest

from tsecon.dataecon import _exact as ex
from tsecon.dataecon import open_dataecon

NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built",
)
FIXTURE = Path(__file__).parent / "fixtures/julia_scalar_markers.daec"
SIBLINGS = ("_type", "_value", "_julia", "_rewrite", "_error")
CASE_COUNT = 151
OBJECT_COUNT = 649
# Julia's own writer re-stores these loaded values through Float64 (integer
# complex, Bool as Int8) or through a rational constructor stricter than the
# integer route that loaded them, so its rewrite does not reproduce the load.
LOSSY_REWRITES = {
    "sm_inj_cint_on_int64_2p53p1": "Complex{Int64}:9007199254740992,0",
    "sm_inj_rat64_on_int64_max": "InexactError",
    "sm_inj_rational_int8_on_int8_min": "InexactError",
    "sm_inj_rational_int8_on_int64_min": "InexactError",
    "sm_inj_bool_on_int8_one": "Int8:1",
}
# Loaded values Julia's writer cannot store at all (its recursion overflows).
UNWRITABLE = {"sm_inj_bigint_on_int64", "sm_inj_bigfloat_on_float"}
FAILURES = ("OverflowError", "InexactError")


def raw_scalars(path=FIXTURE):
    """Per name: (type, frequency, payload, jtype)."""
    with closing(sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)) as sql:
        rows = sql.execute(
            "SELECT o.name, o.type, s.frequency, s.value,"
            "(SELECT value FROM attributes x WHERE x.id=o.id AND x.name='jtype') "
            "FROM objects o JOIN scalars s ON s.id=o.id WHERE o.class=1"
        ).fetchall()
    return {
        name: (kind, frequency, payload or b"", jtype)
        for name, kind, frequency, payload, jtype in rows
    }


def text_of(rows, name):
    """The string scalar ``name`` (invalid UTF-8 bytes escaped), or None when absent."""
    row = rows.get(name)
    if row is None:
        return None
    assert row[0] == 6
    assert row[2].endswith(b"\0")
    return row[2][:-1].decode("utf-8", "surrogateescape")


def cases(rows):
    return sorted(n for n in rows if n.startswith("sm_") and not n.endswith(SIBLINGS))


def outcome(rows, name):
    """Julia's materialized outcome: ``("error", type)`` or ``(type, canonical text)``."""
    error = text_of(rows, f"{name}_error")
    if error is not None:
        return "error", error
    return text_of(rows, f"{name}_type"), text_of(rows, f"{name}_value")


ROWS = raw_scalars()


def test_fixture_inventory_and_sibling_protocol():
    provenance = tomllib.loads(FIXTURE.with_suffix(".toml").read_text(encoding="utf-8"))
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == provenance["fixture_sha256"]
    assert provenance["timeseriesecon_sha"] == "fc0a0d01aed4903ea6c64b12d78d0bdf68468df6"
    names = cases(ROWS)
    assert len(names) == CASE_COUNT
    assert len(ROWS) == OBJECT_COUNT
    for name in names:
        kind, outcome_text = outcome(ROWS, name)
        has = {suffix: f"{name}{suffix}" in ROWS for suffix in SIBLINGS}
        if kind == "error":
            assert not any(has[s] for s in ("_type", "_value", "_julia", "_rewrite")), name
            assert outcome_text.endswith("Error") or outcome_text == "ErrorException", name
            continue
        assert has["_type"], name
        assert has["_value"], name
        assert not has["_error"], name
        assert has["_julia"] == has["_rewrite"] == (name not in UNWRITABLE), name
        if has["_rewrite"]:
            expected = LOSSY_REWRITES.get(name, f"{kind}:{outcome_text}")
            assert text_of(ROWS, f"{name}_rewrite") == expected, name
    assert set(LOSSY_REWRITES) <= set(names)


# ---- exact helpers against Julia's materialized outcomes -------------------


def _integer(payload):
    return int.from_bytes(payload, "little", signed=True)


def _rational_account(kind, payload, parameter):
    if kind == 4 and len(payload) == 8:
        num, den = ex.rationalize(ex.float64_from_bits(payload), parameter)
    elif kind == 1:
        num, den = ex.rational_from_integer(_integer(payload), parameter)
    else:
        return None
    return f"Rational{{{parameter}}}", f"{num}//{den}"


def _complex_account(kind, payload, parameter):
    if kind == 4 and len(payload) == 8:
        pair = ex.integer_complex_from_floats(ex.float64_from_bits(payload), 0.0, parameter)
    elif kind == 5 and len(payload) == 16:
        pair = ex.integer_complex_from_floats(*struct.unpack("<dd", payload), parameter)
    elif kind == 1:
        pair = ex.integer_complex_from_integer(_integer(payload), parameter)
    else:
        return None
    return f"Complex{{{parameter}}}", f"{pair[0]},{pair[1]}"


def _date_account(kind, payload, marker):
    if kind == 4 and len(payload) == 8:
        rata = ex.rata_die_ms_from_unix_seconds(ex.float64_from_bits(payload))
    elif kind == 1 and len(payload) == 8:
        rata = ex.rata_die_ms_from_unix_integer(_integer(payload))
    elif kind == 5 and len(payload) == 16 and struct.unpack("<dd", payload)[1] == 0.0:
        rata = ex.rata_die_ms_from_unix_seconds(struct.unpack("<dd", payload)[0])
    else:
        return None
    if marker == "Date":
        year, month, day = ex.date_from_rata_die_ms(rata)
        return "Date", f"{year}-{month}-{day}"
    y, mo, d, h, mi, s, ms = ex.civil_from_rata_die_ms(rata)
    return "DateTime", f"{y}-{mo}-{d}T{h}:{mi}:{s}.{ms}"


def _unmarked_account(kind, payload):
    if kind in (1, 2) and len(payload) == 16:
        signed = kind == 1
        return ("Int128" if signed else "UInt128"), str(
            ex.int128_from_bytes(payload, signed=signed)
        )
    if kind == 5 and len(payload) == 4:
        real, imag = ex.complexf16_from_bytes(payload)
        return "ComplexF16", (real.tobytes() + imag.tobytes()).hex()
    return None


def _account(kind, payload, marker):
    """The helpers' account of Julia's load, or None where no helper applies."""
    if marker is None:
        return _unmarked_account(kind, payload)
    if (parameter := ex.rational_parameter(marker)) is not None:
        return _rational_account(kind, payload, parameter)
    if (parameter := ex.integer_complex_parameter(marker)) is not None:
        return _complex_account(kind, payload, parameter)
    if marker in ("Date", "DateTime"):
        return _date_account(kind, payload, marker)
    return None


def _covered(rows):
    covered = []
    for name in cases(rows):
        kind, frequency, payload, marker = rows[name]
        if frequency != 0:
            continue  # a token on an MIT scalar: no helper here
        try:
            account = _account(kind, payload, marker)
        except ValueError as error:
            account = ("error", next(f for f in FAILURES if f in str(error)))
        if account is not None:
            covered.append((name, account))
    return covered


COVERED = _covered(ROWS)


def _numeric_route(kind, payload, marker):
    """Whether Julia's own convert route applies (string payloads and a nonzero
    imaginary part are Julia's MethodError/InexactError outside these routes)."""
    if kind not in (1, 4, 5):
        return False
    if kind == 5 and len(payload) == 16 and marker == "Date":
        return struct.unpack("<dd", payload)[1] == 0.0
    return True


def test_helpers_cover_every_rational_complex_date_and_wide_row():
    covered = {name for name, _ in COVERED}
    for name in cases(ROWS):
        kind, _, payload, marker = ROWS[name]
        if marker and (
            ex.rational_parameter(marker)
            or ex.integer_complex_parameter(marker)
            or marker in ("Date", "DateTime")
        ):
            assert (name in covered) == _numeric_route(kind, payload, marker), name
    assert len(COVERED) >= 90


@pytest.mark.parametrize(("name", "account"), COVERED, ids=[name for name, _ in COVERED])
def test_helpers_reproduce_the_materialized_outcome(name, account):
    assert account == outcome(ROWS, name), name


def _components(text):
    sign, body = (-1, text[1:]) if text.startswith("-") else (1, text)
    year, month, day = (int(part) for part in body.split("-"))
    return sign * year, month, day


def test_julia_written_dates_are_reproduced_bit_for_bit_from_the_components():
    # Julia's writer produced these Float64 bytes from a Date; the encoder
    # rebuilds the same bytes from the materialized calendar components.
    checked = 0
    for name in cases(ROWS):
        kind, _, payload, marker = ROWS[name]
        if marker == "Date" and kind == 4 and name.startswith("sm_date"):
            rata = ex.rata_die_ms_from_civil(*_components(outcome(ROWS, name)[1]))
            assert ex.float64_bits(ex.unix_seconds_from_rata_die_ms(rata)) == payload, name
            checked += 1
    assert checked == 9


def test_recorded_julia_facts():
    # float(1 // 3) reloads as the continued-fraction convergent for the
    # parameter, not the exact dyadic ratio; partial quotients above 2**53
    # round in Julia's Float64 loop; a millisecond beyond 2**53 ms is lost by
    # Julia's own writer (written .005, reloaded .004); an embedded NUL is
    # truncated by Julia's own loader; the marker text is never run here.
    assert outcome(ROWS, "sm_rational_third") == (
        "Rational{Int64}",
        "6004799503160661//18014398509481984",
    )
    assert outcome(ROWS, "sm_rational_int32") == ("Rational{Int32}", "1//3")
    assert outcome(ROWS, "sm_rational_3_2p62") == ("Rational{Int64}", "3//4611686018427387649")
    assert outcome(ROWS, "sm_inj_rat128_tiny") == (
        "Rational{Int128}",
        "408//150324684338411231602781414279658602777",
    )
    assert outcome(ROWS, "sm_datetime_year_300k") == ("DateTime", "300000-6-1T12:0:0.4")
    assert outcome(ROWS, "sm_string_embedded_nul") == ("String", "a")
    assert outcome(ROWS, "sm_inj_evaluated") == ("error", "ErrorException")
    assert outcome(ROWS, "sm_inj_unknown") == ("error", "UndefVarError")
    assert outcome(ROWS, "sm_inj_rational_int8_on_int8_min") == ("Rational{Int8}", "-128//1")
    assert outcome(ROWS, "sm_inj_rat8_neg128") == ("error", "InexactError")
    assert outcome(ROWS, "sm_inj_bigfloat_on_float") == (
        "BigFloat",
        "0.1000000000000000055511151231257827021181583404541015625",
    )


# ---- today's boundary through the public reader -----------------------------


def _plain(row):
    """Whether a sibling stored by Julia's writer is one Python reads today."""
    kind, _, payload, marker = row
    if marker is not None:
        return False
    if (kind in (1, 2) and len(payload) == 16) or (kind == 5 and len(payload) == 4):
        return False
    if kind == 6:
        body = payload[:-1]
        try:
            body.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return b"\0" not in body and payload.endswith(b"\0")
    return True


@NATIVE
def test_every_case_is_refused_unevaluated_and_its_attributes_stay_readable():
    with open_dataecon(FIXTURE) as db:
        for name in cases(ROWS):
            marker = ROWS[name][3]
            kind, _, payload, marker = ROWS[name]
            with pytest.raises((TypeError, ValueError)) as info:
                db.read_scalar(name)
            wide = (kind in (1, 2) and len(payload) == 16) or (kind == 5 and len(payload) == 4)
            if wide or marker is None:
                # Wide widths, ComplexF16, invalid UTF-8, embedded NUL and the
                # unterminated string are ValueError before any decoding and
                # before the marker is even looked at.
                assert isinstance(info.value, ValueError), name
            else:
                assert isinstance(info.value, TypeError), name
            assert db.get_attributes(name) == ({} if marker is None else {"jtype": marker}), name
            assert "error(" not in str(info.value)  # the marker text is never echoed or run


@NATIVE
def test_workspace_read_skips_every_case_and_loads_the_materialized_siblings():
    with open_dataecon(FIXTURE) as db:
        loaded = db.read_workspace()
    skipped = {member.path: member for member in loaded.report.skipped}
    names = cases(ROWS)
    # Every case is skipped; so are Julia's re-stored siblings (same markers
    # or widths, unless its writer canonicalised the value to a plain form)
    # and the two text siblings holding invalid UTF-8.
    expected = {f"/{name}" for name, row in ROWS.items() if name in names or not _plain(row)}
    assert set(skipped) == expected
    for member in skipped.values():
        assert member.category in ("unsupported", "invalid"), member
        assert not member.subtree
    for name in names:
        kind, text = outcome(ROWS, name)
        if kind == "error":
            assert loaded.workspace[f"{name}_error"] == text
        elif f"/{name}_value" in skipped:
            assert not _plain(ROWS[f"{name}_value"])
        else:
            assert loaded.workspace[f"{name}_type"] == kind
            assert loaded.workspace[f"{name}_value"] == text
    assert loaded.report.count == OBJECT_COUNT - len(skipped)
    assert len(loaded.workspace) == loaded.report.count
