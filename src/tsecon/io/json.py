# SPDX-License-Identifier: MIT
"""JSON round-trip for ``MIT``, ``Duration``, ``MITRange``, ``TSeries``, and ``Workspace``.

The Julia ``serialize.jl`` is a binary protocol coupled to ``Serialization.jl``;
it has no JSON. We invent the schema here: every tsecon object is encoded as a
JSON object with a ``_type`` discriminator, so a decoder can reconstruct the
right Python type without ambiguity.

Schema (stable starting from this version — a versioned ``_schema`` field will
be added if the schema ever needs to change):

* ``MIT``       → ``{"_type": "MIT", "freq": <Freq>, "value": <int>}``
* ``Duration``  → ``{"_type": "Duration", "freq": <Freq>, "value": <int>}``
* ``MITRange``  → ``{"_type": "MITRange", "start": <MIT>, "stop": <MIT>, "step": <int>}``
* ``TSeries``   → ``{"_type": "TSeries", "firstdate": <MIT>, "dtype": <str>,
  "values": [<scalar>...]}``
* ``Workspace`` → ``{"_type": "Workspace", "items": [[<key>, <value>], ...]}``

Frequencies are encoded as ``{"name": <class>, ...}``, with the extra parameter
where applicable (``end_month`` for Yearly/HalfYearly/Quarterly, ``end_day``
for Weekly).

NaN / +Inf / -Inf are emitted as JavaScript-style literals (``NaN`` /
``Infinity`` / ``-Infinity``), which is the Python ``json`` module's default
when ``allow_nan=True``. Strictly-valid-JSON consumers can pass
``allow_nan=False`` and supply their own sentinel handling.

MVTSeries support will be added when ``MVTSeries`` lands.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import IO, Any

import numpy as np

from tsecon.frequencies import (
    BDaily,
    Daily,
    Frequency,
    HalfYearly,
    Monthly,
    Quarterly,
    Unit,
    Weekly,
    Yearly,
)
from tsecon.mit import MIT, Duration
from tsecon.mitrange import MITRange
from tsecon.tseries import TSeries
from tsecon.workspace import Workspace

__all__ = ["dump", "dumps", "from_jsonable", "load", "loads", "to_jsonable"]


# ---------------------------------------------------------------------------
# Frequency <-> JSON-able dict
# ---------------------------------------------------------------------------


def _freq_to_dict(f: Frequency) -> dict[str, Any]:
    if isinstance(f, Yearly):
        return {"name": "Yearly", "end_month": f.end_month}
    if isinstance(f, HalfYearly):
        return {"name": "HalfYearly", "end_month": f.end_month}
    if isinstance(f, Quarterly):
        return {"name": "Quarterly", "end_month": f.end_month}
    if isinstance(f, Monthly):
        return {"name": "Monthly"}
    if isinstance(f, Weekly):
        return {"name": "Weekly", "end_day": f.end_day}
    if isinstance(f, Daily):
        return {"name": "Daily"}
    if isinstance(f, BDaily):
        return {"name": "BDaily"}
    if isinstance(f, Unit):
        return {"name": "Unit"}
    msg = f"Cannot serialize frequency of type {type(f).__name__}."
    raise TypeError(msg)


_FREQ_FACTORIES: dict[str, Callable[[dict[str, Any]], Frequency]] = {
    "Yearly": lambda d: Yearly(end_month=int(d.get("end_month", 12))),
    "HalfYearly": lambda d: HalfYearly(end_month=int(d.get("end_month", 6))),
    "Quarterly": lambda d: Quarterly(end_month=int(d.get("end_month", 3))),
    "Monthly": lambda _d: Monthly(),
    "Weekly": lambda d: Weekly(end_day=int(d.get("end_day", 7))),
    "Daily": lambda _d: Daily(),
    "BDaily": lambda _d: BDaily(),
    "Unit": lambda _d: Unit(),
}


def _freq_from_dict(d: dict[str, Any]) -> Frequency:
    name = d.get("name")
    factory = _FREQ_FACTORIES.get(name) if isinstance(name, str) else None
    if factory is None:
        msg = f"Unknown frequency name in JSON: {name!r}"
        raise ValueError(msg)
    return factory(d)


# ---------------------------------------------------------------------------
# Object <-> JSON-able tree
# ---------------------------------------------------------------------------


def to_jsonable(obj: Any) -> Any:
    """Convert a tsecon object (or nested structure) to a plain JSON-friendly tree.

    Pure Python ``int``, ``float``, ``bool``, ``str``, ``None``, ``list``, and
    ``dict`` values pass through unchanged. NumPy scalars are unwrapped via
    ``.item()``. NumPy arrays become Python lists.
    """
    if isinstance(obj, MIT):
        return {
            "_type": "MIT",
            "freq": _freq_to_dict(obj.frequency),
            "value": int(obj.value),
        }
    if isinstance(obj, Duration):
        return {
            "_type": "Duration",
            "freq": _freq_to_dict(obj.frequency),
            "value": int(obj.value),
        }
    if isinstance(obj, MITRange):
        return {
            "_type": "MITRange",
            "start": to_jsonable(obj.start),
            "stop": to_jsonable(obj.stop),
            "step": int(obj.step),
        }
    if isinstance(obj, TSeries):
        return {
            "_type": "TSeries",
            "firstdate": to_jsonable(obj.firstdate),
            "dtype": str(obj.values.dtype),
            "values": obj.values.tolist(),
        }
    if isinstance(obj, Workspace):
        return {
            "_type": "Workspace",
            "items": [[k, to_jsonable(v)] for k, v in obj.items()],
        }
    if isinstance(obj, Frequency):
        return {"_type": "Frequency", "freq": _freq_to_dict(obj)}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    msg = f"Cannot JSON-encode value of type {type(obj).__name__}."
    raise TypeError(msg)


def from_jsonable(obj: Any) -> Any:
    """Inverse of :func:`to_jsonable`: rebuild tsecon objects from a JSON tree."""
    if isinstance(obj, dict):
        tag = obj.get("_type")
        if tag is not None:
            return _decode_tagged(tag, obj)
        return {k: from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [from_jsonable(x) for x in obj]
    return obj


def _decode_tagged(tag: Any, obj: dict[str, Any]) -> Any:
    if tag == "MIT":
        return MIT(_freq_from_dict(obj["freq"]), int(obj["value"]))
    if tag == "Duration":
        return Duration(_freq_from_dict(obj["freq"]), int(obj["value"]))
    if tag == "MITRange":
        start = from_jsonable(obj["start"])
        stop = from_jsonable(obj["stop"])
        step = int(obj.get("step", 1))
        if not isinstance(start, MIT) or not isinstance(stop, MIT):
            msg = "Invalid MITRange in JSON: start/stop did not decode to MIT."
            raise ValueError(msg)
        return MITRange(start, stop, step)
    if tag == "TSeries":
        firstdate = from_jsonable(obj["firstdate"])
        if not isinstance(firstdate, MIT):
            msg = "Invalid TSeries in JSON: firstdate did not decode to MIT."
            raise ValueError(msg)
        dtype = np.dtype(obj.get("dtype", "float64"))
        values = np.asarray(obj["values"], dtype=dtype)
        return TSeries(firstdate, values)
    if tag == "Workspace":
        items = obj.get("items", [])
        out = Workspace()
        for pair in items:
            k, v = pair
            out[str(k)] = from_jsonable(v)
        return out
    if tag == "Frequency":
        return _freq_from_dict(obj["freq"])
    msg = f"Unknown _type tag in JSON: {tag!r}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dumps(obj: Any, *, indent: int | None = None, allow_nan: bool = True) -> str:
    """Serialize a tsecon (or plain) value to a JSON string."""
    return json.dumps(to_jsonable(obj), indent=indent, allow_nan=allow_nan)


def loads(s: str | bytes | bytearray) -> Any:
    """Parse a JSON string and rebuild tsecon objects from any ``_type``-tagged nodes."""
    return from_jsonable(json.loads(s))


def dump(obj: Any, fp: IO[str], *, indent: int | None = None, allow_nan: bool = True) -> None:
    """Write a tsecon (or plain) value as JSON to a writable text stream."""
    json.dump(to_jsonable(obj), fp, indent=indent, allow_nan=allow_nan)


def load(fp: IO[str]) -> Any:
    """Read a JSON document from a text stream and rebuild tsecon objects."""
    return from_jsonable(json.load(fp))
