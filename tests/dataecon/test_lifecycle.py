"""Python file lifecycle and feature isolation, including injected failures."""

import subprocess
import sys

import numpy as np
import pytest

import tsecon.dataecon as de
from tsecon import TSeries, mm


class FakeHandle:
    def __init__(self):
        self.closes = 0
        self.failure = None

    def close(self):
        self.closes += 1
        if self.failure:
            raise self.failure

    def read(self, name):
        return 2024, 1, np.ones(4).tobytes(), (2, 12, 4, 0, 1, 4, 32, 24288, 32), name


@pytest.fixture
def owner(monkeypatch):
    handle = FakeHandle()
    monkeypatch.setattr(de, "_open_native", lambda *_: handle)
    return de.open_dataecon("example.daec"), handle


def test_idempotent_close_and_use_after_close(owner):
    db, handle = owner
    db.close()
    db.close()
    assert handle.closes == 1
    assert db.closed
    with pytest.raises(ValueError, match="closed"):
        db.read_series("sample")
    with pytest.raises(ValueError, match="closed"):
        db.write_series("sample", TSeries(mm(2024, 1), np.ones(4)))
    with pytest.raises(ValueError, match="closed"):
        db.__enter__()


def test_body_exception_closes(owner):
    db, handle = owner
    with pytest.raises(RuntimeError, match="body"), db:
        raise RuntimeError("body")
    assert handle.closes == 1
    assert db.closed


def test_failed_close_is_terminal(owner):
    db, handle = owner
    handle.failure = de.DataEconError(-978, "close", db.path, "injected")
    with pytest.raises(de.DataEconError, match="injected"):
        db.close()
    assert db.closed
    with pytest.raises(ValueError, match="close-failed"):
        db.close()
    with pytest.raises(ValueError, match="close-failed"):
        db.read_series("sample")
    assert handle.closes == 1


def test_cleanup_does_not_mask_primary_exception(owner):
    db, handle = owner
    handle.failure = de.DataEconError(5, "close", db.path, "injected busy")
    primary = RuntimeError("body")
    with pytest.raises(RuntimeError, match="body") as caught, db:
        raise primary
    assert caught.value is primary
    assert "injected busy" in primary.__notes__[0]
    assert handle.closes == 1


def test_copy_failure_still_closes(owner, monkeypatch):
    db, handle = owner

    def fail(*_):
        raise MemoryError("copy failed")

    monkeypatch.setattr(de, "decode_series", fail)
    with pytest.raises(MemoryError, match="copy failed"), db:
        db.read_series("sample")
    assert handle.closes == 1


@pytest.mark.parametrize(
    ("path", "mode", "exception"),
    [
        ("file", "w", ValueError),
        ("", "r", ValueError),
        ("bad\0file", "r", ValueError),
        (b"file", "r", TypeError),
    ],
)
def test_invalid_open_never_reaches_native(path, mode, exception, monkeypatch):
    def fail(*_):
        pytest.fail("invalid input reached native open")

    monkeypatch.setattr(de, "_open_native", fail)
    with pytest.raises(exception):
        de.open_dataecon(path, mode)


@pytest.mark.parametrize(
    ("name", "exception"),
    [("", ValueError), ("a/b", ValueError), ("a\0b", ValueError), (1, TypeError)],
)
def test_invalid_name(owner, name, exception):
    db, _ = owner
    with db, pytest.raises(exception):
        db.read_series(name)


def test_missing_extension_has_actionable_error(monkeypatch):
    def fail(*_):
        raise ImportError("missing extension")

    monkeypatch.setattr(de.importlib, "import_module", fail)
    with pytest.raises(ImportError, match="TSECON_DATAECON_ROOT") as caught:
        de.open_dataecon("example.daec")
    assert "missing extension" in str(caught.value.__cause__)


def test_core_import_does_not_load_native():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tsecon; "
            "assert 'tsecon.dataecon._native' not in sys.modules; "
            "import tsecon.dataecon; assert 'tsecon.dataecon._native' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Native Windows path limitation")
def test_unicode_windows_path_rejected_before_native(tmp_path, monkeypatch):
    def fail(*_):
        pytest.fail("Unicode Windows path reached native open")

    monkeypatch.setattr(de, "_open_native", fail)
    path = tmp_path / "données_日本.daec"
    with pytest.raises(ValueError, match="ASCII file path"):
        de.open_dataecon(path, "a")
    assert not path.exists()
