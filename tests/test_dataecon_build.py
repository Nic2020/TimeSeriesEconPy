"""Build-input guards without a compiler or downloaded native dependency."""

import hashlib
import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/build_dataecon_windows.py"
spec = importlib.util.spec_from_file_location("dataecon_build", SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def test_unverified_archive_never_creates_output(tmp_path):
    archive = tmp_path / "unverified.zip"
    archive.write_bytes(b"unverified build input")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="SHA-256"):
        builder.extract_source(archive, output)
    assert not output.exists()


def test_existing_output_is_preserved(tmp_path, monkeypatch):
    archive = tmp_path / "input.zip"
    archive.write_bytes(b"fixture")
    monkeypatch.setattr(builder, "SOURCE_SHA256", hashlib.sha256(b"fixture").hexdigest())
    output = tmp_path / "output"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("existing work")
    with pytest.raises(FileExistsError, match="new output directory"):
        builder.extract_source(archive, output)
    assert sentinel.read_text() == "existing work"


def test_archive_path_escape_is_rejected_before_extraction(tmp_path, monkeypatch):
    archive = tmp_path / "input.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr(f"DataEcon-{builder.SOURCE_COMMIT}/../../escaped.txt", "bad")
    monkeypatch.setattr(builder, "SOURCE_SHA256", builder.file_hash(archive))
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="member path"):
        builder.extract_source(archive, output)
    assert not output.exists()
    assert not (tmp_path / "escaped.txt").exists()


@pytest.mark.parametrize("count", [0, 2])
def test_portability_patch_refuses_unexpected_source(tmp_path, count):
    path = tmp_path / "src/libdaec/dates.c"
    path.parent.mkdir(parents=True)
    original = builder.EPOCH_BEFORE * count + b"\n/* unexpected source */\n"
    path.write_bytes(original)
    with pytest.raises(ValueError, match="exactly once"):
        builder.apply_msvc_patch(tmp_path)
    assert path.read_bytes() == original
