# SPDX-License-Identifier: MIT
"""TimeSeriesEconPy: a time-series language for macroeconomics.

Ported from `TimeSeriesEcon.jl`_ (Bank of Canada).

.. _TimeSeriesEcon.jl: https://github.com/bankofcanada/TimeSeriesEcon.jl
"""

from tsecon._mirror import MIRRORS_JULIA_SHA

__version__ = "0.0.1.dev0"

__all__ = [
    "MIRRORS_JULIA_SHA",
    "__version__",
]
