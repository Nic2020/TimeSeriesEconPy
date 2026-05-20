# SPDX-License-Identifier: MIT
"""M2.6 fidelity test: Python ``d11`` matches Julia ``d11`` at 1e-10.

The Python and Julia X-13 wrappers are intended to produce identical
seasonal-adjustment output when run against the **same** X-13ARIMA-SEATS
binary. This test enforces that contract:

* The committed fixture ``tests/x13/fixtures/d11_julia_reference.csv``
  holds Julia's ``d11`` output for a fixed deterministic Quarterly
  series, captured by the sibling script ``capture_julia_d11.jl``
  with ``setoption(:x13path, ...)`` pointed at the same binary the
  Python wrapper resolves to.
* Re-running the Python wrapper against the same fixture series and
  the same binary should produce a ``d11`` whose values match the
  reference vector at ``atol = 1e-10`` (``rtol = 0``).

The test is binary-gated: if no x13as binary is reachable
(:func:`tsecon.x13._result._resolve_binary` returns ``None``), it skips.
M2.6's wheels matrix is the production source of the binary; locally,
``scripts/fetch_x13as_local.py`` populates it.

Why 1e-10
---------

The X-13ARIMA-SEATS binary writes ``d11`` values to its ``.d11`` output
file with eight significant digits in the default format (e.g.
``100.00157``). Both wrappers parse the same text into the same numpy
float64s, so the only sources of drift are (i) round-off in the binary
itself between identical invocations (none — X-13 is deterministic on
identical inputs) and (ii) any difference in the ``.spc`` text the two
wrappers serialise. The 1e-10 atol covers floating-point parsing
ambiguity (a hand-rounded eighth-digit value re-printed in full float64
representation can drift by ~5e-9 — see Julia's ``Printf`` source) with
two orders of magnitude of margin.

Regeneration
------------

Re-run the Julia capture if any of the following change:

* The bundled binary version (``X13AS_VERSION`` in :mod:`tsecon._mirror`).
* The fixture series construction below.
* The X-13 spec the wrapper emits (e.g. M2.x adds new default kwargs to
  ``x11`` / ``newspec``).

::

    julia --project=PATH/TO/TimeSeriesEcon.jl tests/x13/fixtures/capture_julia_d11.jl
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tsecon.frequencies import Quarterly
from tsecon.mit import MIT
from tsecon.mitrange import MITRange
from tsecon.tseries import TSeries
from tsecon.x13 import newspec, run, x11
from tsecon.x13._result import _resolve_binary

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "d11_julia_reference.csv"


needs_binary = pytest.mark.skipif(
    _resolve_binary() is None,
    reason=(
        "X-13ARIMA-SEATS binary not reachable. Run `python scripts/fetch_x13as_local.py` "
        "to populate src/tsecon/x13/_binary/, or wait for an M2.6 wheel install."
    ),
)


def _build_fixture_series() -> TSeries:
    """100-quarter Quarterly TSeries: trend + sinusoidal seasonal.

    Matches the Julia capture script ``capture_julia_d11.jl`` line-for-line.
    """
    n = 100
    start = MIT.from_yp(Quarterly(), 2000, 1)
    end = start + (n - 1)
    rng = MITRange(start, end)
    i = np.arange(n)
    trend = 100.0 + 0.5 * i
    seasonal = 5.0 * np.sin(2 * np.pi * i / 4)
    return TSeries(rng, trend + seasonal)


def _load_julia_d11() -> np.ndarray:
    """Read the committed Julia ``d11`` reference fixture."""
    if not FIXTURE_PATH.is_file():
        msg = (
            f"Julia reference fixture missing at {FIXTURE_PATH}. "
            "Regenerate via tests/x13/fixtures/capture_julia_d11.jl."
        )
        raise FileNotFoundError(msg)
    values: list[float] = []
    for raw in FIXTURE_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        values.append(float(line))
    return np.asarray(values, dtype=np.float64)


@needs_binary
class TestX13FidelityVsJulia:
    """Lock Python vs Julia ``d11`` parity at 1e-10 atol."""

    def test_d11_matches_julia_reference(self) -> None:
        ts = _build_fixture_series()
        spec = newspec(ts, x11=x11(save="d11"))
        result = run(spec, verbose=False, load="d11")
        py_d11 = result.series.d11.values
        jl_d11 = _load_julia_d11()
        assert py_d11.shape == jl_d11.shape, (
            f"d11 length mismatch: Python={py_d11.shape}, Julia={jl_d11.shape}. "
            "Possible spec drift; regenerate the Julia fixture."
        )
        np.testing.assert_allclose(py_d11, jl_d11, rtol=0.0, atol=1e-10)
