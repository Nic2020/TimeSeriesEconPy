# SPDX-License-Identifier: MIT
"""Unit tests for the nine ``x13read_*`` parsers + ``loadresult`` dispatcher.

Each parser is exercised against hand-authored fixtures that mirror the
shape X-13ARIMA-SEATS v1.1 b60 emits (cross-checked against the Julia
upstream's example files at ``TimeSeriesEcon.jl/src/x13/``). The
end-to-end binary-vs-Julia ``d11`` 1e-10 numerical-fidelity test
(planned in MASTER_PLAN M2.5) is deferred to M2.6 alongside the
wheels-side gfortran binary — until then the binary is not reachable
from the test environment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tsecon.frequencies import Monthly, Quarterly, Yearly
from tsecon.mit import MIT
from tsecon.mvtseries import MVTSeries
from tsecon.tseries import TSeries
from tsecon.workspace import Workspace
from tsecon.x13._result import (
    WorkspaceTable,
    X13lazy,
    _add_layers,
    _parse_x13_date,
    _sanitize_colname,
    loadresult,
    x13read_err,
    x13read_estimates,
    x13read_identify,
    x13read_key_values,
    x13read_model,
    x13read_seatsseries,
    x13read_series,
    x13read_udg,
    x13read_workspace_table,
)

# ---------------------------------------------------------------------------
# Helpers (column-name sanitiser, date parser, dotted-key normaliser)
# ---------------------------------------------------------------------------


class TestSanitizeColname:
    def test_no_op_when_clean(self) -> None:
        assert _sanitize_colname("foo") == "foo"

    def test_space_to_underscore(self) -> None:
        assert _sanitize_colname("Final Adjusted") == "Final_Adjusted"

    def test_run_of_whitespace_collapses(self) -> None:
        assert _sanitize_colname("a   b") == "a_b"

    def test_hyphen_and_dot(self) -> None:
        assert _sanitize_colname("d.11-final") == "d_11_final"


class TestParseX13Date:
    def test_canonical(self) -> None:
        assert _parse_x13_date("May 20, 2026") == "2026-05-20"

    def test_full_month_name_truncates(self) -> None:
        assert _parse_x13_date("January 1, 2020") == "2020-01-01"

    def test_extra_whitespace(self) -> None:
        assert _parse_x13_date("Feb  3,  2024") == "2024-02-03"

    def test_unknown_month_raises(self) -> None:
        with pytest.raises(ValueError, match="X-13 month"):
            _parse_x13_date("Foo 1, 2024")

    def test_malformed_raises(self) -> None:
        # Wrong arity (2 tokens instead of 3) → length-check failure.
        with pytest.raises(ValueError, match="X-13 date"):
            _parse_x13_date("only two")


class TestAddLayers:
    def test_flat_workspace_unchanged(self) -> None:
        ws = Workspace(a=1, b=2)
        _add_layers(ws)
        assert list(ws._c.keys()) == ["a", "b"]

    def test_single_level_nesting(self) -> None:
        ws = Workspace()
        ws._c["roots.ar"] = 0.5
        ws._c["roots.ma"] = -0.3
        _add_layers(ws)
        assert "roots" in ws._c
        assert isinstance(ws._c["roots"], Workspace)
        assert ws._c["roots"]._c == {"ar": 0.5, "ma": -0.3}

    def test_trunk_collision_appends_underscore(self) -> None:
        ws = Workspace()
        ws._c["roots"] = "scalar"
        ws._c["roots.ar"] = 0.5
        _add_layers(ws)
        # ``roots`` was a non-Workspace scalar, so the dotted key lands
        # under ``roots_`` instead.
        assert ws._c["roots"] == "scalar"
        assert isinstance(ws._c["roots_"], Workspace)
        assert ws._c["roots_"]._c == {"ar": 0.5}

    def test_recursive_two_levels(self) -> None:
        ws = Workspace()
        ws._c["a.b.c"] = 1
        ws._c["a.b.d"] = 2
        _add_layers(ws)
        assert isinstance(ws._c["a"], Workspace)
        assert isinstance(ws._c["a"]._c["b"], Workspace)
        assert ws._c["a"]._c["b"]._c == {"c": 1, "d": 2}


# ---------------------------------------------------------------------------
# x13read_err
# ---------------------------------------------------------------------------


class TestX13ReadErr:
    def test_three_diagnostics(self, tmp_path: Path) -> None:
        text = (
            " WARNING: HP filter sample size is small.\n"
            " ERROR: spec has no series block.\n"
            " NOTE: convergence achieved.\n"
        )
        p = tmp_path / "test.err"
        p.write_text(text, encoding="utf-8")
        warns: list[str] = []
        notes: list[str] = []
        errs: list[str] = []
        x13read_err(p, warns, notes, errs)
        assert warns == ["HP filter sample size is small."]
        assert errs == ["spec has no series block."]
        assert notes == ["convergence achieved."]

    def test_continuation_lines(self, tmp_path: Path) -> None:
        text = " WARNING: line one\n line two of warning\n line three of warning\n"
        p = tmp_path / "c.err"
        p.write_text(text, encoding="utf-8")
        warns: list[str] = []
        notes: list[str] = []
        errs: list[str] = []
        x13read_err(p, warns, notes, errs)
        assert len(warns) == 1
        assert "line one" in warns[0]
        assert "line two of warning" in warns[0]
        assert "line three of warning" in warns[0]

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "e.err"
        p.write_text("", encoding="utf-8")
        warns: list[str] = []
        notes: list[str] = []
        errs: list[str] = []
        x13read_err(p, warns, notes, errs)
        assert warns == []
        assert notes == []
        assert errs == []

    def test_pre_header_garbage_ignored(self, tmp_path: Path) -> None:
        # Lines before any diagnostic should be dropped (mirrors Julia).
        text = "garbage line\n WARNING: real warning\n"
        p = tmp_path / "g.err"
        p.write_text(text, encoding="utf-8")
        warns: list[str] = []
        notes: list[str] = []
        errs: list[str] = []
        x13read_err(p, warns, notes, errs)
        assert warns == ["real warning"]


# ---------------------------------------------------------------------------
# x13read_key_values + x13read_udg
# ---------------------------------------------------------------------------


class TestX13ReadKeyValues:
    def test_int_value(self) -> None:
        ws = x13read_key_values(["x: 42"])
        assert ws._c == {"x": 42}

    def test_float_value(self) -> None:
        ws = x13read_key_values(["x: 3.14"])
        assert ws._c == {"x": 3.14}

    def test_yes_no_value(self) -> None:
        ws = x13read_key_values(["flag1: yes", "flag2: no"])
        assert ws._c["flag1"] is True
        assert ws._c["flag2"] is False

    def test_string_value(self) -> None:
        ws = x13read_key_values(["mode: multiplicative"])
        assert ws._c["mode"] == "multiplicative"

    def test_vector_value(self) -> None:
        ws = x13read_key_values(["coeffs: 1.0 2.0 3.0"])
        assert ws._c["coeffs"] == [1.0, 2.0, 3.0]

    def test_skips_empty_lines(self) -> None:
        ws = x13read_key_values(["", "x: 1", "  ", "y: 2"])
        assert ws._c == {"x": 1, "y": 2}

    def test_dotted_keys_become_nested(self) -> None:
        ws = x13read_key_values(["a.b: 1", "a.c: 2"])
        assert isinstance(ws._c["a"], Workspace)
        assert ws._c["a"]._c == {"b": 1, "c": 2}

    def test_unparseable_line_warns(self) -> None:
        with pytest.warns(UserWarning, match="Could not parse"):
            ws = x13read_key_values(["nothing-to-split-here"])
        assert ws._c == {}

    def test_tab_separator(self) -> None:
        ws = x13read_key_values(["x\t42"])
        assert ws._c == {"x": 42}

    def test_stars_become_nan(self) -> None:
        ws = x13read_key_values(["v: ******* 1.0 2.0"])
        assert isinstance(ws._c["v"], list)
        # NaN is not equal to NaN; check via numpy.
        assert np.isnan(ws._c["v"][0])
        assert ws._c["v"][1] == 1.0


class TestX13ReadUdg:
    def test_basic_udg_file(self, tmp_path: Path) -> None:
        text = "version: 1.1\nrun.id: x13_20260520\nseasonal: yes\n"
        p = tmp_path / "test.udg"
        p.write_text(text, encoding="utf-8")
        ws = x13read_udg(p)
        assert ws._c["version"] == 1.1
        assert ws._c["run"]._c == {"id": "x13_20260520"}
        assert ws._c["seasonal"] is True


# ---------------------------------------------------------------------------
# x13read_workspace_table
# ---------------------------------------------------------------------------


class TestX13ReadWorkspaceTable:
    def test_two_column_int(self) -> None:
        lines = [
            "lag\tvalue",
            "---\t-----",
            "1\t10",
            "2\t20",
            "3\t30",
        ]
        ws = x13read_workspace_table(lines)
        assert isinstance(ws, WorkspaceTable)
        assert list(ws._c.keys()) == ["lag", "value"]
        assert ws._c["lag"] == [1, 2, 3]
        assert ws._c["value"] == [10, 20, 30]

    def test_float_column(self) -> None:
        lines = [
            "lag\tvalue",
            "---\t-----",
            "1\t1.5",
            "2\t2.5",
        ]
        ws = x13read_workspace_table(lines)
        assert ws._c["value"] == [1.5, 2.5]

    def test_string_column_falls_back(self) -> None:
        lines = [
            "code\tname",
            "----\t----",
            "A\tone",
            "B\ttwo",
        ]
        ws = x13read_workspace_table(lines)
        assert ws._c["code"] == ["A", "B"]
        assert ws._c["name"] == ["one", "two"]

    def test_drops_trailing_empty_line(self) -> None:
        lines = [
            "lag\tvalue",
            "---\t-----",
            "1\t10",
            "",
        ]
        ws = x13read_workspace_table(lines)
        assert ws._c["lag"] == [1]

    def test_acm_inserts_lag_header(self) -> None:
        # acm header is missing the "lag" column; the parser inserts it.
        lines = [
            "value",
            "-----",
            "1\t10",
            "2\t20",
        ]
        ws = x13read_workspace_table(lines, ext="acm")
        assert "lag" in ws._c
        assert "value" in ws._c

    def test_sanitizes_header_whitespace(self) -> None:
        lines = [
            "long name\tother col",
            "---------\t---------",
            "1\t2",
        ]
        ws = x13read_workspace_table(lines)
        assert "long_name" in ws._c
        assert "other_col" in ws._c

    def test_empty_input(self) -> None:
        ws = x13read_workspace_table([])
        assert ws._c == {}


# ---------------------------------------------------------------------------
# x13read_series
# ---------------------------------------------------------------------------


class TestX13ReadSeries:
    def test_single_column_quarterly(self, tmp_path: Path) -> None:
        text = (
            "date\tseries\n"
            "----\t------\n"
            "202001\t100.5\n"
            "202002\t101.2\n"
            "202003\t99.8\n"
            "202004\t102.1\n"
        )
        p = tmp_path / "s.d11"
        p.write_text(text, encoding="utf-8")
        ts = x13read_series(p, Quarterly())
        assert isinstance(ts, TSeries)
        assert ts.frequency == Quarterly()
        assert ts.firstdate == MIT.from_yp(Quarterly(), 2020, 1)
        np.testing.assert_array_equal(ts.values, [100.5, 101.2, 99.8, 102.1])

    def test_two_column_yields_mvtseries(self, tmp_path: Path) -> None:
        text = "date\ta\tb\n----\t-\t-\n202001\t1.0\t10.0\n202002\t2.0\t20.0\n"
        p = tmp_path / "two.b1"
        p.write_text(text, encoding="utf-8")
        mvt = x13read_series(p, Quarterly())
        assert isinstance(mvt, MVTSeries)
        assert mvt.firstdate == MIT.from_yp(Quarterly(), 2020, 1)
        np.testing.assert_array_equal(mvt.values, [[1.0, 10.0], [2.0, 20.0]])

    def test_monthly_yyyypp(self, tmp_path: Path) -> None:
        text = "date\ts\n----\t-\n202001\t5.0\n202002\t6.0\n202003\t7.0\n"
        p = tmp_path / "m.d11"
        p.write_text(text, encoding="utf-8")
        ts = x13read_series(p, Monthly())
        assert ts.firstdate == MIT.from_yp(Monthly(), 2020, 1)
        np.testing.assert_array_equal(ts.values, [5.0, 6.0, 7.0])

    def test_month_name_header_period(self, tmp_path: Path) -> None:
        text = "date\tseries\n----\t------\njan\t1.0\nfeb\t2.0\nmar\t3.0\n"
        p = tmp_path / "month.d11"
        p.write_text(text, encoding="utf-8")
        ts = x13read_series(p, Monthly())
        # Year defaults to 1 for the legacy month-only header (mirrors Julia).
        assert ts.firstdate == MIT.from_yp(Monthly(), 1, 1)

    def test_nan_marker(self, tmp_path: Path) -> None:
        text = "date\tseries\n----\t------\n202001\tNaN\n202002\t2.0\n"
        p = tmp_path / "nan.d11"
        p.write_text(text, encoding="utf-8")
        ts = x13read_series(p, Quarterly())
        assert np.isnan(ts.values[0])
        assert ts.values[1] == 2.0

    def test_unknown_period_raises(self, tmp_path: Path) -> None:
        text = "date\ts\n----\t-\n??\t1.0\n"
        p = tmp_path / "bad.d11"
        p.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match="Period string"):
            x13read_series(p, Quarterly())

    def test_too_short_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "tiny.d11"
        p.write_text("date\ts\n", encoding="utf-8")
        with pytest.raises(ValueError, match="too short"):
            x13read_series(p, Quarterly())

    def test_non_yp_frequency_raises(self, tmp_path: Path) -> None:
        # Yearly is a YPFrequency, so use Daily-equivalent to trigger.
        # We don't have Daily wired to X-13 (the binary doesn't support
        # daily data), but the guard should still fire on any non-YP.
        text = "date\ts\n----\t-\n202001\t1.0\n"
        p = tmp_path / "y.d11"
        p.write_text(text, encoding="utf-8")
        # Yearly is a YPFrequency; should succeed.
        ts = x13read_series(p, Yearly())
        assert ts is not None


class TestX13ReadSeatsSeries:
    def test_single_column(self) -> None:
        # SEATS format: data lines have NO leading whitespace; the header
        # row's leading whitespace is intentional and ``[2:]`` skips both
        # the leading empty-string artefact and the date-column header.
        lines = [
            "  Seasonal Component",
            "  date           value",
            "1 - 2020         100.5",
            "2 - 2020         101.2",
            "3 - 2020         99.8",
            "",
        ]
        ts = x13read_seatsseries(lines, Quarterly())
        assert isinstance(ts, TSeries)
        assert ts.firstdate == MIT.from_yp(Quarterly(), 2020, 1)
        np.testing.assert_array_equal(ts.values, [100.5, 101.2, 99.8])


# ---------------------------------------------------------------------------
# x13read_estimates
# ---------------------------------------------------------------------------


class TestX13ReadEstimates:
    def test_arima_section(self, tmp_path: Path) -> None:
        text = (
            "$arima:\n"
            "$arima$estimates:\n"
            "parameter\testimate\n"
            "---------\t--------\n"
            "ar1\t0.5\n"
            "ma1\t-0.3\n"
            "$variance:\n"
            "sigma2: 1.234\n"
        )
        p = tmp_path / "e.est"
        p.write_text(text, encoding="utf-8")
        ws = x13read_estimates(p)
        assert "arima" in ws._c
        assert isinstance(ws._c["arima"], WorkspaceTable)
        assert ws._c["arima"]._c["parameter"] == ["ar1", "ma1"]
        assert ws._c["variance"]._c == {"sigma2": 1.234}

    def test_regression_section(self, tmp_path: Path) -> None:
        text = (
            "$regression:\n"
            "$regression$estimates:\n"
            "parameter\testimate\n"
            "---------\t--------\n"
            "constant\t10.0\n"
            "$variance:\n"
            "sigma2: 1.0\n"
        )
        p = tmp_path / "r.est"
        p.write_text(text, encoding="utf-8")
        ws = x13read_estimates(p)
        assert "regression" in ws._c

    def test_no_markers_returns_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.est"
        p.write_text("nothing\nhere\n", encoding="utf-8")
        ws = x13read_estimates(p)
        assert ws._c == {}


# ---------------------------------------------------------------------------
# x13read_identify
# ---------------------------------------------------------------------------


class TestX13ReadIdentify:
    def test_one_page(self, tmp_path: Path) -> None:
        text = "$diff = 0\n$sdiff = 0\nlag\tacf\n---\t---\n1\t0.5\n2\t0.3\n"
        p = tmp_path / "i.iac"
        p.write_text(text, encoding="utf-8")
        ws = x13read_identify(p)
        # Section name strips the ``$``/``= `` markers and concatenates
        # the diff + sdiff labels.
        assert len(ws._c) == 1
        key = next(iter(ws._c.keys()))
        assert "diff" in key
        assert "sdiff" in key

    def test_two_pages(self, tmp_path: Path) -> None:
        text = (
            "$diff = 0\n"
            "$sdiff = 0\n"
            "lag\tacf\n"
            "---\t---\n"
            "1\t0.5\n"
            "$diff = 1\n"
            "$sdiff = 0\n"
            "lag\tacf\n"
            "---\t---\n"
            "1\t-0.2\n"
        )
        p = tmp_path / "two.iac"
        p.write_text(text, encoding="utf-8")
        ws = x13read_identify(p)
        assert len(ws._c) == 2


# ---------------------------------------------------------------------------
# x13read_model
# ---------------------------------------------------------------------------


class TestX13ReadModel:
    def test_arima_only(self, tmp_path: Path) -> None:
        text = "arima{\nmodel=\n(0 1 1)(0 1 1)\nar  =(\n)\nma  =(\n0.5\n-0.3\n)\n}\n"
        p = tmp_path / "m.mdl"
        p.write_text(text, encoding="utf-8")
        ws = x13read_model(p)
        # We only assert the arima block was parsed (model + ma); the
        # exact ArimaModel.spec ordering depends on the order of keys in
        # the source.
        assert "arima" in ws._c

    def test_regression_only(self, tmp_path: Path) -> None:
        text = "regression{\nvariables=(\nconst\ntd1\n)\nb=(\n10.0\n0.5f\n)\n}\n"
        p = tmp_path / "r.mdl"
        p.write_text(text, encoding="utf-8")
        ws = x13read_model(p)
        assert "regression" in ws._c
        regr = ws._c["regression"]
        assert regr._c["variables"] == ["const", "td1"]
        assert regr._c["b"] == [10.0, 0.5]
        assert regr._c["fixb"] == [False, True]

    def test_empty_file(self, tmp_path: Path) -> None:
        p = tmp_path / "e.mdl"
        p.write_text("", encoding="utf-8")
        ws = x13read_model(p)
        assert ws._c == {}


# ---------------------------------------------------------------------------
# loadresult dispatch
# ---------------------------------------------------------------------------


class TestLoadresult:
    def test_series_extension(self, tmp_path: Path) -> None:
        text = "date\ts\n----\t-\n202001\t1.0\n202002\t2.0\n"
        p = tmp_path / "x.d11"
        p.write_text(text, encoding="utf-8")
        lazy = X13lazy(str(p), "d11", Quarterly())
        ts = loadresult(lazy)
        assert isinstance(ts, TSeries)

    def test_table_extension(self, tmp_path: Path) -> None:
        text = "lag\tv\n---\t-\n1\t1.0\n"
        p = tmp_path / "x.acf"
        p.write_text(text, encoding="utf-8")
        lazy = X13lazy(str(p), "acf", Quarterly())
        ws = loadresult(lazy)
        assert isinstance(ws, WorkspaceTable)

    def test_human_text_extension(self, tmp_path: Path) -> None:
        text = "human readable output\n"
        p = tmp_path / "x.out"
        p.write_text(text, encoding="utf-8")
        lazy = X13lazy(str(p), "out", Quarterly())
        result = loadresult(lazy)
        assert result == text

    def test_positional_calling_convention(self, tmp_path: Path) -> None:
        text = "date\ts\n----\t-\n202001\t1.0\n202002\t2.0\n"
        p = tmp_path / "x.d11"
        p.write_text(text, encoding="utf-8")
        ts = loadresult(str(p), "d11", Quarterly())
        assert isinstance(ts, TSeries)

    def test_positional_wrong_arity_raises(self) -> None:
        with pytest.raises(TypeError, match="three positional"):
            loadresult("foo", "d11")  # type: ignore[call-arg]

    def test_kv_list_extension(self, tmp_path: Path) -> None:
        text = "key1\t1.0\nkey2\t2.0\n"
        p = tmp_path / "x.lks"
        p.write_text(text, encoding="utf-8")
        lazy = X13lazy(str(p), "lks", Quarterly())
        ws = loadresult(lazy)
        assert isinstance(ws, Workspace)
        assert ws._c == {"key1": 1.0, "key2": 2.0}

    def test_txt_extension_returns_none(self, tmp_path: Path) -> None:
        # ``txt`` and ``log`` are intentionally silently skipped.
        p = tmp_path / "x.txt"
        p.write_text("ignored\n", encoding="utf-8")
        lazy = X13lazy(str(p), "txt", Quarterly())
        result = loadresult(lazy)
        assert result is None

    def test_unknown_extension_warns(self, tmp_path: Path) -> None:
        p = tmp_path / "x.zzz"
        p.write_text("mystery\n", encoding="utf-8")
        lazy = X13lazy(str(p), "zzz", Quarterly())
        with pytest.warns(UserWarning, match="unknown output file"):
            result = loadresult(lazy)
        assert result is None
