"""Exercise floating-point compiler policy through the wheel build hook."""

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

cython_build = pytest.importorskip("Cython.Build", reason="Build-hook test requires Cython")
pytest.importorskip("setuptools", reason="Build-hook test requires setuptools")
setuptools_build_ext = importlib.import_module("setuptools.command.build_ext")


@pytest.mark.parametrize("platform", ["win32", "linux", "darwin"])
def test_statistics_rounding_flag_reaches_build_ext_only_for_statistics(monkeypatch, platform):
    root = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("rounding_build_hook", root / "hatch_build.py")
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    modules = ["tsecon._stats_kernels_cy", "tsecon._rolling_kernels_cy"]
    monkeypatch.setattr(hook, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(
        hook,
        "_discover_pyx_modules",
        lambda: [(name, root / "src" / (name.replace(".", "/") + ".pyx")) for name in modules],
    )
    monkeypatch.setattr(cython_build, "cythonize", lambda extensions, **kwargs: extensions)
    observed = {}

    class Build:
        def __init__(self, distribution):
            self.extensions = distribution.ext_modules

        def ensure_finalized(self):
            pass

        def run(self):
            for extension in self.extensions:
                observed[extension.name] = extension.extra_compile_args

        def get_ext_fullpath(self, module):
            return root / "build" / (module + ".extension")

    monkeypatch.setattr(setuptools_build_ext, "build_ext", Build)
    paths = hook.build_extensions_inplace()
    assert len(paths) == 2
    assert observed[modules[0]] == ([] if platform == "win32" else ["-ffp-contract=off"])
    assert observed[modules[1]] == []
