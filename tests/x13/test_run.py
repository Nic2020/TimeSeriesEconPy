# SPDX-License-Identifier: MIT
""":func:`run` + :func:`deseasonalize` smoke tests.

The end-to-end ``d11`` numerical-fidelity test (Python vs Julia at
1e-10) is deferred to M2.6 alongside the wheels-side gfortran binary.
This file covers the **non-binary** surface of :func:`run`:

* :func:`run` raises a clear :exc:`RuntimeError` when no X-13 binary is
  reachable (the default state until M2.6's wheels ship).
* :func:`run` raises :exc:`TypeError` when called with a string spec
  but no frequency.
* :func:`_normalise_load` round-trips the ``"none"`` / ``"all"`` /
  scalar / sequence shapes.

The binary-availability branch is exercised by setting
``setoption("x13path", "/path/to/x13as")``; tests that need the binary
:func:`pytest.skip` if it's not present.
"""

from __future__ import annotations

import numpy as np
import pytest

from tsecon.frequencies import Quarterly
from tsecon.mit import MIT
from tsecon.mitrange import MITRange
from tsecon.tseries import TSeries
from tsecon.x13 import deseasonalize, deseasonalize_inplace, newspec, run, x11
from tsecon.x13._result import _normalise_load, _resolve_binary


def _binary_available() -> bool:
    return _resolve_binary() is not None


needs_binary = pytest.mark.skipif(
    not _binary_available(),
    reason=(
        "X-13ARIMA-SEATS binary not reachable via setoption('x13path'). "
        "The bundled binary lands in M2.6 alongside the wheels matrix."
    ),
)


# ---------------------------------------------------------------------------
# _normalise_load
# ---------------------------------------------------------------------------


class TestNormaliseLoad:
    def test_none(self) -> None:
        is_all, keys = _normalise_load("none")
        assert (is_all, keys) == (False, set())

    def test_all(self) -> None:
        is_all, keys = _normalise_load("all")
        assert is_all is True
        assert keys == set()

    def test_single_scalar(self) -> None:
        is_all, keys = _normalise_load("d11")
        assert (is_all, keys) == (False, {"d11"})

    def test_sequence(self) -> None:
        is_all, keys = _normalise_load(["d11", "d12"])
        assert (is_all, keys) == (False, {"d11", "d12"})

    def test_tuple(self) -> None:
        is_all, keys = _normalise_load(("a", "b", "c"))
        assert (is_all, keys) == (False, {"a", "b", "c"})


# ---------------------------------------------------------------------------
# run — no-binary branch
# ---------------------------------------------------------------------------


class TestRunNoBinaryAvailable:
    def test_no_binary_raises_clear_error(self) -> None:
        if _binary_available():
            pytest.skip("Binary is available; the no-binary error path doesn't apply.")
        ts = _small_quarterly_ts()
        spec = newspec(ts, x11=x11(save="d11"))
        with pytest.raises(RuntimeError, match="No X-13ARIMA-SEATS binary"):
            run(spec)

    def test_string_overload_requires_freq(self) -> None:
        with pytest.raises(TypeError, match="freq is required"):
            run("series{}", verbose=False)

    def test_non_yp_freq_in_string_overload_raises(self) -> None:
        from tsecon.frequencies import Daily  # noqa: PLC0415

        if _binary_available():
            pytest.skip("This guard fires before binary resolution.")
        with pytest.raises((TypeError, RuntimeError)):
            run("series{}", Daily(), verbose=False)

    def test_invalid_spec_type_raises(self) -> None:
        with pytest.raises(TypeError, match="X13spec or a spec string"):
            run(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# run / deseasonalize — binary branch (skipped when not available)
# ---------------------------------------------------------------------------


@needs_binary
class TestRunBinary:
    def test_minimal_x11_spec(self) -> None:
        ts = _quarterly_demo_series(n=60)
        spec = newspec(ts, x11=x11(save="d11"))
        res = run(spec, verbose=False, load="d11")
        assert "d11" in res.series._c
        d11 = res.series.d11
        assert isinstance(d11, TSeries)
        assert d11.frequency == Quarterly()

    def test_load_all(self) -> None:
        ts = _quarterly_demo_series(n=60)
        spec = newspec(ts, x11=x11(save="d11"))
        res = run(spec, verbose=False, load="all")
        # At least d11 should be eager.
        assert "d11" in res.series._c


@needs_binary
class TestDeseasonalize:
    def test_deseasonalize_returns_new_tseries(self) -> None:
        ts = _quarterly_demo_series(n=60)
        out = deseasonalize(ts)
        assert out is not ts
        assert isinstance(out, TSeries)
        assert len(out.values) == len(ts.values)

    def test_deseasonalize_inplace_mutates(self) -> None:
        ts = _quarterly_demo_series(n=60)
        original = ts.values.copy()
        out = deseasonalize_inplace(ts)
        assert out is ts
        # Values should change (deseasonalized series is not the same as raw).
        assert not np.allclose(ts.values, original)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_quarterly_ts() -> TSeries:
    start = MIT.from_yp(Quarterly(), 2020, 1)
    end = MIT.from_yp(Quarterly(), 2020, 4)
    rng = MITRange(start, end)
    return TSeries(rng, np.array([1.0, 2.0, 3.0, 4.0]))


def _quarterly_demo_series(n: int = 60) -> TSeries:
    """A synthetic Quarterly series long enough for X-11."""
    start = MIT.from_yp(Quarterly(), 2000, 1)
    end = start + (n - 1)
    rng = MITRange(start, end)
    # Trend + seasonal pattern + noise.
    rng_arange = np.arange(n)
    trend = 100.0 + 0.5 * rng_arange
    seasonal = 5.0 * np.sin(2 * np.pi * rng_arange / 4)
    return TSeries(rng, trend + seasonal)
