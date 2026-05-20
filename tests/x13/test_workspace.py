# SPDX-License-Identifier: MIT
"""Tests for :class:`WorkspaceTable`, :class:`X13ResultWorkspace`,
:class:`X13lazy`, :class:`X13result`, and the cleanup machinery.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tsecon import x13 as x13_pkg
from tsecon.frequencies import Quarterly
from tsecon.tseries import TSeries
from tsecon.workspace import Workspace
from tsecon.x13 import _result as _result_mod
from tsecon.x13 import _x13 as _x13_mod
from tsecon.x13._result import (
    WorkspaceTable,
    X13lazy,
    X13result,
    X13ResultWorkspace,
    _cleanup_outfolder,
    _format_workspace_table,
)
from tsecon.x13._spec import newspec

# ---------------------------------------------------------------------------
# WorkspaceTable
# ---------------------------------------------------------------------------


class TestWorkspaceTable:
    def test_empty_repr(self) -> None:
        assert repr(WorkspaceTable()) == "Empty WorkspaceTable"

    def test_two_columns_repr(self) -> None:
        wt = WorkspaceTable(lag=[1, 2, 3], value=[1.5, 2.5, 3.5])
        s = repr(wt)
        # Header row, separator row, three data rows.
        assert s.count("\n") == 4
        assert "lag" in s
        assert "value" in s
        assert "1.5" in s

    def test_is_a_workspace(self) -> None:
        wt = WorkspaceTable(a=[1])
        # Substitutability: a WorkspaceTable IS a Workspace (mirrors Julia's
        # AbstractWorkspace subtype relation).
        assert isinstance(wt, Workspace)

    def test_format_uses_g_for_floats(self) -> None:
        wt = WorkspaceTable(v=[1.0, 2.0])
        s = _format_workspace_table(wt)
        # ``{:g}`` strips trailing zeros, so ``1.0`` should render as ``1``.
        assert "1\n" in s or s.endswith("1") or " 1 " in s

    def test_format_nan(self) -> None:
        wt = WorkspaceTable(v=[float("nan"), 1.0])
        s = _format_workspace_table(wt)
        assert "NaN" in s

    def test_construction_via_dict(self) -> None:
        wt = WorkspaceTable({"a": [1, 2]})
        assert wt._c == {"a": [1, 2]}


# ---------------------------------------------------------------------------
# X13ResultWorkspace
# ---------------------------------------------------------------------------


class TestX13ResultWorkspace:
    def test_plain_value_attribute_access(self) -> None:
        ws = X13ResultWorkspace(x=42)
        assert ws.x == 42

    def test_plain_value_key_access(self) -> None:
        ws = X13ResultWorkspace(x=42)
        assert ws["x"] == 42

    def test_lazy_materialises_on_attribute_access(self, tmp_path: Path) -> None:
        text = "date\ts\n----\t-\n202001\t1.0\n202002\t2.0\n"
        p = tmp_path / "x.d11"
        p.write_text(text, encoding="utf-8")
        ws = X13ResultWorkspace()
        ws._c["d11"] = X13lazy(str(p), "d11", Quarterly())
        # First access — should trigger loadresult.
        val = ws.d11
        assert isinstance(val, TSeries)
        # The materialised value is written back; subsequent access is fast.
        assert ws._c["d11"] is val
        # Second access returns the cached object.
        assert ws.d11 is val

    def test_lazy_materialises_on_key_access(self, tmp_path: Path) -> None:
        text = "date\ts\n----\t-\n202001\t1.0\n"
        p = tmp_path / "x.d11"
        p.write_text(text, encoding="utf-8")
        ws = X13ResultWorkspace()
        ws._c["d11"] = X13lazy(str(p), "d11", Quarterly())
        val = ws["d11"]
        assert isinstance(val, TSeries)
        assert ws._c["d11"] is val

    def test_subset_returns_plain_workspace_unmaterialised(self, tmp_path: Path) -> None:
        text = "date\ts\n----\t-\n202001\t1.0\n"
        p = tmp_path / "x.d11"
        p.write_text(text, encoding="utf-8")
        ws = X13ResultWorkspace()
        ws._c["d11"] = X13lazy(str(p), "d11", Quarterly())
        ws._c["plain"] = 42
        sub = ws[["d11", "plain"]]
        # Subset is a plain Workspace; the X13lazy entry survives.
        assert isinstance(sub, Workspace)
        assert isinstance(sub._c["d11"], X13lazy)

    def test_missing_attribute_raises(self) -> None:
        ws = X13ResultWorkspace(a=1)
        with pytest.raises(AttributeError, match="has no member"):
            _ = ws.nonexistent

    def test_is_a_workspace(self) -> None:
        ws = X13ResultWorkspace()
        assert isinstance(ws, Workspace)


class TestX13Lazy:
    def test_repr_is_file_path(self) -> None:
        lazy = X13lazy("/tmp/x13_abc/foo.d11", "d11", Quarterly())
        assert repr(lazy) == "/tmp/x13_abc/foo.d11"
        assert str(lazy) == "/tmp/x13_abc/foo.d11"

    def test_frozen(self) -> None:
        lazy = X13lazy("/tmp/x.d11", "d11", Quarterly())
        with pytest.raises((AttributeError, Exception)):
            lazy.file = "/other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# X13result
# ---------------------------------------------------------------------------


class TestX13Result:
    def test_construction_populates_empty_workspaces(self, tmp_path: Path) -> None:
        spec = newspec(rng_for_test())
        r = X13result(spec, str(tmp_path), "stdout text")
        assert r.spec is spec
        assert r.outfolder == str(tmp_path)
        assert r.stdout == "stdout text"
        assert isinstance(r.series, X13ResultWorkspace)
        assert isinstance(r.tables, X13ResultWorkspace)
        assert isinstance(r.text, X13ResultWorkspace)
        assert isinstance(r.other, X13ResultWorkspace)
        assert r.series._c == {}
        assert r.errors == []
        assert r.warnings == []
        assert r.notes == []

    def test_repr_contains_field_summaries(self, tmp_path: Path) -> None:
        spec = newspec(rng_for_test())
        r = X13result(spec, str(tmp_path), "stdout")
        s = repr(r)
        assert "X13 results" in s
        assert "outfolder" in s
        assert "stdout" in s
        assert "errors" in s

    def test_finalize_removes_temp_folder(self, tmp_path: Path) -> None:
        # The finalizer fires when the X13result is GC'd. We can't reliably
        # trigger GC mid-test, so instead we test the cleanup callback
        # directly and the finalize wiring via _cleanup_handle.
        folder = tmp_path / "x13_test"
        folder.mkdir()
        (folder / "foo.txt").write_text("hi", encoding="utf-8")
        spec = newspec(rng_for_test())
        r = X13result(spec, str(folder), "")
        # The handle is registered.
        assert r._cleanup_handle.alive
        # Manually invoke the finalizer.
        r._cleanup_handle()
        assert not folder.exists()


# ---------------------------------------------------------------------------
# _cleanup_outfolder retry loop
# ---------------------------------------------------------------------------


class TestCleanupOutfolder:
    def test_removes_existing_folder(self, tmp_path: Path) -> None:
        folder = tmp_path / "x13_x"
        folder.mkdir()
        (folder / "f.txt").write_text("hi", encoding="utf-8")
        _cleanup_outfolder(str(folder))
        assert not folder.exists()

    def test_missing_folder_is_no_op(self, tmp_path: Path) -> None:
        _cleanup_outfolder(str(tmp_path / "does_not_exist"))
        # Should not raise.

    def test_empty_path_is_no_op(self) -> None:
        _cleanup_outfolder("")

    def test_handles_permission_error_then_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        folder = tmp_path / "x13_p"
        folder.mkdir()

        attempts = {"count": 0}
        real_rmtree = _result_mod.shutil.rmtree

        def flaky_rmtree(
            path: str, *args: object, ignore_errors: bool = False, **kw: object
        ) -> None:
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise PermissionError("simulated")
            real_rmtree(path, ignore_errors=ignore_errors)

        monkeypatch.setattr(_result_mod.shutil, "rmtree", flaky_rmtree)
        _result_mod._cleanup_outfolder(str(folder))
        assert attempts["count"] >= 2
        assert not folder.exists()


# ---------------------------------------------------------------------------
# get_cleanup_folders + cleanup sweep
# ---------------------------------------------------------------------------


class TestGetCleanupFolders:
    def test_finds_x13_prefixed_folders(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_x13_mod.tempfile, "gettempdir", lambda: str(tmp_path))
        (tmp_path / "x13_abc").mkdir()
        (tmp_path / "x13_def").mkdir()
        (tmp_path / "other_xyz").mkdir()
        (tmp_path / "x13_a_file").write_text("hi", encoding="utf-8")
        out = _x13_mod.get_cleanup_folders()
        names = sorted(os.path.basename(p) for p in out)
        assert names == ["x13_abc", "x13_def"]

    def test_empty_when_no_matches(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_x13_mod.tempfile, "gettempdir", lambda: str(tmp_path))
        assert _x13_mod.get_cleanup_folders() == []


class TestCleanup:
    def test_sweep_removes_x13_folders(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_x13_mod.tempfile, "gettempdir", lambda: str(tmp_path))
        (tmp_path / "x13_one").mkdir()
        (tmp_path / "x13_two").mkdir()
        (tmp_path / "keep_me").mkdir()
        with pytest.warns(UserWarning, match="Removed 2"):
            x13_pkg.cleanup()
        assert not (tmp_path / "x13_one").exists()
        assert not (tmp_path / "x13_two").exists()
        assert (tmp_path / "keep_me").exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rng_for_test() -> TSeries:
    """Return a small Quarterly TSeries useful for X13result construction tests."""
    from tsecon.mit import MIT  # noqa: PLC0415
    from tsecon.mitrange import MITRange  # noqa: PLC0415

    start = MIT.from_yp(Quarterly(), 2020, 1)
    end = MIT.from_yp(Quarterly(), 2020, 4)
    rng = MITRange(start, end)
    return TSeries(rng, np.ones(len(rng), dtype=np.float64))
