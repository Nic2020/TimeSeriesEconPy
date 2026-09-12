"""Installed-wheel checker output layout versus the workflow's Julia discovery contract.

Both Julia verification steps in ``.github/workflows/wheels.yml`` glob
``build/dataecon-interchange/*.daec`` and require exactly one match. The
checker therefore writes exactly one primary ``cpXY.daec`` into the discovery
directory and every auxiliary file below it (``fileops/``). These tests run
the checker's writer on a temporary workspace and apply the workflow's own
glob to the result; they do not need the workflow's shells.
"""

import importlib
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from tsecon import TSeries, mm
from tsecon.dataecon import open_dataecon

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).parent / "fixtures"
WORKFLOW = ROOT / ".github/workflows/wheels.yml"
NATIVE = pytest.mark.skipif(
    importlib.util.find_spec("tsecon.dataecon._native") is None,
    reason="DataEcon extension was not built",
)


@pytest.fixture(scope="module")
def checker():
    scripts = ROOT / "scripts"
    if not (scripts / "check_dataecon_wheel.py").is_file():
        pytest.skip("checker script is not part of this installation")
    sys.path.insert(0, str(scripts))
    try:
        return importlib.import_module("check_dataecon_wheel")
    finally:
        sys.path.remove(str(scripts))


def workflow_discovery_globs():
    """The glob each Julia verification step applies, read from the workflow itself."""
    if not WORKFLOW.is_file():
        pytest.skip("workflow file is not part of this installation")
    text = WORKFLOW.read_text(encoding="utf-8")
    windows = re.findall(r"Get-ChildItem (\S+\.daec)\)", text)
    unix = re.findall(r"outputs=\((\S+\.daec)\)", text)
    assert len(windows) == 1, "expected one discovery glob in the Windows step"
    assert len(unix) == 1, "expected one discovery glob in the Unix step"
    assert "$output.Count -ne 1" in text
    assert '"${#outputs[@]}" -eq 1' in text
    return windows[0], unix[0]


def test_workflow_discovers_one_file_in_the_interchange_directory():
    windows, unix = workflow_discovery_globs()
    assert windows == unix == "build/dataecon-interchange/*.daec"


def test_checker_places_auxiliary_output_below_the_discovery_directory(checker, tmp_path):
    primary = tmp_path / "build/dataecon-interchange/cp311.daec"
    auxiliary = checker.file_operations_output(primary)
    assert auxiliary == tmp_path / "build/dataecon-interchange/fileops/cp311-fileops.daec"
    assert auxiliary.parent != primary.parent
    assert checker.PRIMARY_OUTPUT.fullmatch(primary.name)
    assert not checker.PRIMARY_OUTPUT.fullmatch(auxiliary.name)


def test_checker_refuses_non_primary_files_in_the_discovery_directory(checker, tmp_path):
    (tmp_path / "cp311.daec").write_bytes(b"")
    checker.check_discovery_contract(tmp_path)
    (tmp_path / "cp311-fileops.daec").write_bytes(b"")
    with pytest.raises(ValueError, match="discovery"):
        checker.check_discovery_contract(tmp_path)


@NATIVE
def test_checker_output_satisfies_the_workflow_glob(checker, tmp_path, monkeypatch):
    windows, unix = workflow_discovery_globs()
    workspace = tmp_path / "workspace"
    output_dir = workspace / "build/dataecon-interchange"
    with open_dataecon(FIXTURES / "julia_monthly.daec") as db:
        series = db.read_series("sample")
    monkeypatch.chdir(workspace.parent)
    workspace.mkdir()
    primary = checker.write_interchange(output_dir, series)
    expected_name = f"cp{sys.version_info.major}{sys.version_info.minor}.daec"
    assert primary == output_dir / expected_name
    # The workflow globs are relative to the workspace; apply each one there.
    for pattern in (windows, unix):
        matches = sorted(workspace.glob(pattern))
        assert matches == [primary], pattern
    auxiliary = checker.file_operations_output(primary)
    assert auxiliary.is_file()
    assert sorted(p.name for p in output_dir.iterdir()) == [expected_name, "fileops"]
    # The auxiliary name sorts before the primary; a flat layout would have
    # made a "first match" selection pick the wrong file.
    assert auxiliary.name < primary.name
    with open_dataecon(auxiliary) as db:
        assert db.read_scalar("after_truncate") == 42
        np.testing.assert_array_equal(
            db.read_series("after_truncate_series").values, [1.0, 2.0, 3.0]
        )
    with open_dataecon(primary) as db:
        assert db.read_scalar("fo_overwritten") == "two"
        assert isinstance(db.read_series("fo_series_overwritten"), TSeries)
        assert db.read_series("sample").firstdate == mm(2024, 1)
    # A second run into the same directory is refused rather than duplicated.
    with pytest.raises(FileExistsError):
        checker.write_interchange(output_dir, series)
