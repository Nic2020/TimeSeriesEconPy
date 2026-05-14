# SPDX-License-Identifier: MIT
"""I/O subsystem: JSON serialization (and future formats).

The Julia upstream's ``serialize.jl`` is a binary protocol bound to Julia's
``Serialization`` stdlib (Distributed.jl process communication). For Python the
analogous public-facing format is portable JSON, with ``pickle`` available as
the language-native fallback for in-process work.
"""

from __future__ import annotations

from tsecon.io.json import dump, dumps, load, loads

__all__ = ["dump", "dumps", "load", "loads"]
