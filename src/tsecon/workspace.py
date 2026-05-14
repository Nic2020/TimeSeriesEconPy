# SPDX-License-Identifier: MIT
"""Workspace — insertion-ordered name-indexed collection.

Mirrors ``TimeSeriesEcon.jl``'s ``Workspace`` (``workspaces.jl``): a container
that holds named values of any kind — scalars, MITs, ranges, TSeries,
MVTSeries (when ported), even nested Workspaces.

Julia uses ``OrderedDict{Symbol,Any}`` as the storage; Python's built-in
``dict`` preserves insertion order natively (since Python 3.7), so we use it
directly without an external dependency.

Both attribute access (``w.a``) and bracket access (``w["a"]``) are supported,
matching the Julia idiom of dual access via ``getproperty`` and ``getindex``.
Subset access (``w["a", "b"]`` or ``w[["a", "b"]]``) returns a new Workspace.

Deferred until later milestones (currently blocked on other ported modules):

* ``overlay`` / ``compare`` / ``compare_equal`` / ``reindex`` — live in
  ``various.jl`` (M4 🔵).
* ``@weval`` — Julia macro; no direct Python equivalent. Use a normal
  closure or ``eval`` if needed.
* ``copyto!`` into an ``MVTSeries`` destination — blocked on ``MVTSeries``
  (M1 ⬜).
* ``clean_old_frequencies`` — Julia compatibility shim for legacy quarterly
  numbering; no analogue needed in Python.

The standalone ``strip(t::TSeries)`` / ``strip!(t::TSeries)`` helpers live in
``fconvert_helpers.jl`` upstream and will land in the ``fconvert`` port; for
the workspace's ``strip_inplace`` we inline the ``_valid_range`` logic so the
test port doesn't have to wait.
"""

from __future__ import annotations

from collections.abc import Callable, ItemsView, Iterable, Iterator, KeysView, Mapping, ValuesView
from typing import Any

import numpy as np

from tsecon.frequencies import Frequency, prettyprint_frequency
from tsecon.mit import MIT, Duration
from tsecon.mitrange import MITRange, rangeof_span
from tsecon.tseries import TSeries, typenan

__all__ = ["Workspace"]


# Names that are reserved for the implementation and must not be confused with
# user keys. Anything stored under one of these names on the instance dict is
# treated as part of the Workspace plumbing.
_RESERVED: frozenset[str] = frozenset({"_c"})


def _has_frequency(value: object) -> bool:
    """Return True if ``frequency_of`` would succeed on ``value``."""
    return isinstance(value, (MIT, Duration, Frequency, MITRange, TSeries, Workspace))


def _frequency_of_value(value: object) -> Frequency | None:
    """Return the frequency of a value, or None if it has none.

    Mirrors the Julia ``_has_frequencyof``/``frequencyof`` dispatch: MITs,
    Durations, MITRanges, TSeries, and nested Workspaces contribute a
    frequency; everything else returns None.
    """
    if isinstance(value, (MIT, Duration)):
        return value.frequency
    if isinstance(value, Frequency):
        return value
    if isinstance(value, MITRange):
        return value.frequency
    if isinstance(value, TSeries):
        return value.frequency
    if isinstance(value, Workspace):
        return value.frequency_of()
    return None


class Workspace:
    """Insertion-ordered, name-indexed collection of arbitrary values.

    Construct with one of:

    * ``Workspace()`` — empty.
    * ``Workspace(a=1, b=2)`` — from keyword arguments.
    * ``Workspace({"a": 1, "b": 2})`` — from a mapping.
    * ``Workspace([("a", 1), ("b", 2)])`` — from an iterable of pairs.

    Access by attribute (``w.a``) or by key (``w["a"]``); subset by tuple or
    list of keys (``w["a", "b"]`` or ``w[["a", "b"]]``) to get a new Workspace.

    Keys must be strings. (Julia uses ``Symbol``; we standardize on ``str``
    since Python has no analogue and attribute access naturally produces
    strings.)
    """

    __slots__ = ("_c",)

    _c: dict[str, Any]

    # -- construction ------------------------------------------------------

    def __init__(
        self,
        items: Mapping[str, Any] | Workspace | Iterable[tuple[str, Any]] | None = None,
        /,
        **kwargs: Any,
    ) -> None:
        # bypass __setattr__ which routes to _c
        object.__setattr__(self, "_c", {})
        if items is not None:
            if isinstance(items, (Mapping, Workspace)):
                for k, v in items.items():
                    self._c[str(k)] = v
            else:
                for k, v in items:
                    self._c[str(k)] = v
        for k, v in kwargs.items():
            self._c[k] = v

    @classmethod
    def from_dict(cls, d: Mapping[Any, Any], *, recursive: bool = False) -> Workspace:
        """Build a Workspace from a mapping.

        With ``recursive=True``, any nested ``Mapping`` value is itself
        converted to a Workspace, recursively. Mirrors the Julia
        ``Workspace(d; recursive=true)`` constructor.
        """
        out = cls()
        for k, v in d.items():
            value = cls.from_dict(v, recursive=True) if recursive and isinstance(v, Mapping) else v
            out._c[str(k)] = value
        return out

    # -- attribute access (dot notation) -----------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only called when normal attribute lookup fails.
        try:
            return object.__getattribute__(self, "_c")[name]
        except KeyError as e:
            msg = f"Workspace has no member {name!r}"
            raise AttributeError(msg) from e

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _RESERVED:
            object.__setattr__(self, name, value)
            return
        self._c[name] = value

    def __delattr__(self, name: str) -> None:
        if name in _RESERVED:
            msg = f"{name!r} is reserved on Workspace."
            raise AttributeError(msg)
        try:
            del self._c[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __dir__(self) -> list[str]:
        # Surface member names for tab completion / IDE introspection.
        return [*object.__dir__(self), *self._c.keys()]

    # -- key access --------------------------------------------------------

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            try:
                return self._c[key]
            except KeyError as e:
                raise KeyError(key) from e
        if isinstance(key, (tuple, list)):
            return Workspace({str(k): self._c[str(k)] for k in key})
        msg = f"Workspace keys must be str or tuple/list of str, got {type(key).__name__}."
        raise TypeError(msg)

    def __setitem__(self, key: str, value: Any) -> None:
        if not isinstance(key, str):
            msg = f"Workspace keys must be str, got {type(key).__name__}."  # type: ignore[unreachable]
            raise TypeError(msg)
        self._c[key] = value

    def __delitem__(self, key: str) -> None:
        del self._c[key]

    # -- dict-like introspection -------------------------------------------

    def __len__(self) -> int:
        return len(self._c)

    def __iter__(self) -> Iterator[str]:
        return iter(self._c)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._c

    def keys(self) -> KeysView[str]:
        """Return a view of the member names."""
        return self._c.keys()

    def values(self) -> ValuesView[Any]:
        """Return a view of the member values."""
        return self._c.values()

    def items(self) -> ItemsView[str, Any]:
        """Return a view of the (name, value) pairs."""
        return self._c.items()

    def get(self, key: str, default: Any = None) -> Any:
        """Return ``self[key]`` if present, otherwise ``default``."""
        return self._c.get(key, default)

    def is_empty(self) -> bool:
        """Return True iff the Workspace has no members."""
        return len(self._c) == 0

    # -- mutation helpers --------------------------------------------------

    def merge(self, other: Workspace | Mapping[str, Any]) -> Workspace:
        """Return a new Workspace with ``other``'s entries overlaid onto a copy of ``self``.

        Mirrors Julia's ``merge(a, b)``. Keys in ``other`` win over keys in
        ``self``.
        """
        out = self.copy()
        out.merge_inplace(other)
        return out

    def merge_inplace(self, other: Workspace | Mapping[str, Any]) -> Workspace:
        """Update ``self`` in place with entries from ``other`` (other wins).

        Returns ``self`` for chaining. Mirrors Julia's ``merge!(a, b)``.
        """
        src = other._c if isinstance(other, Workspace) else other
        for k, v in src.items():
            self._c[str(k)] = v
        return self

    def empty_inplace(self) -> Workspace:
        """Remove all entries. Mirrors Julia's ``empty!(w)``."""
        self._c.clear()
        return self

    def copy(self) -> Workspace:
        """Return a shallow copy: same value references, fresh storage."""
        out = Workspace()
        out._c.update(self._c)
        return out

    def __copy__(self) -> Workspace:
        return self.copy()

    # -- filter / map ------------------------------------------------------

    def filter(self, predicate: Callable[[str, Any], bool]) -> Workspace:
        """Return a new Workspace containing only the (k, v) pairs that pass ``predicate``.

        Predicate is called as ``predicate(key, value)``. Mirrors Julia's
        ``filter(tuple -> ..., w)`` (where ``tuple`` is a ``(key, value)``
        pair).
        """
        return Workspace({k: v for k, v in self._c.items() if predicate(k, v)})

    def filter_inplace(self, predicate: Callable[[str, Any], bool]) -> Workspace:
        """Drop entries that fail ``predicate``; mutates and returns ``self``."""
        for k in [k for k, v in self._c.items() if not predicate(k, v)]:
            del self._c[k]
        return self

    def map(self, f: Callable[[Any], Any]) -> Workspace:
        """Apply ``f`` to each member's value; return a new Workspace.

        Keys are preserved. Mirrors Julia's ``map(f, w)``.
        """
        return Workspace({k: f(v) for k, v in self._c.items()})

    # -- strip -------------------------------------------------------------

    def strip_inplace(self, *, recursive: bool = True) -> Workspace:
        """Trim leading/trailing NaN values from every TSeries member.

        With ``recursive=True`` (the default) also strips TSeries inside
        nested Workspaces. Mirrors Julia's ``strip!(w; recursive)``.

        The per-TSeries trim mirrors ``strip!(t::TSeries)`` from
        ``fconvert_helpers.jl``: it resizes the TSeries to its inner range
        where ``values != typenan(dtype)``. For boolean dtypes the typenan
        is ``False``, which would strip everything — those are left alone.
        """
        for value in self._c.values():
            if isinstance(value, TSeries):
                _strip_tseries_inplace(value)
            elif recursive and isinstance(value, Workspace):
                value.strip_inplace(recursive=recursive)
        return self

    # -- range / frequency -------------------------------------------------

    def frequency_of(self, *, check: bool = False) -> Frequency | None:
        """Return the common frequency of all frequency-bearing members, or None.

        Recurses through nested Workspaces. If members carry distinct
        frequencies, returns None (or raises ``ValueError`` if
        ``check=True``). With ``check=True`` and no frequency-bearing
        members, also raises.
        """
        freqs: list[Frequency] = []
        for v in self._c.values():
            if not _has_frequency(v):
                continue
            f = _frequency_of_value(v)
            if f is not None:
                freqs.append(f)
        unique = {id(f): f for f in freqs}
        if len(unique) == 1:
            return next(iter(unique.values()))
        if check:
            if not freqs:
                msg = "The given workspace doesn't have a frequency."
                raise ValueError(msg)
            names = sorted({type(f).__name__ for f in freqs})
            msg = f"The given workspace has multiple frequencies: {', '.join(names)}."
            raise ValueError(msg)
        return None

    def rangeof(self) -> MITRange:
        """Return the intersection of the ranges of all rangeable members.

        Frequency-bearing members must share a single frequency, else
        ``TypeError`` is raised. Members without a range (scalars, strings,
        etc.) are skipped. If no member has a range, raises ``ValueError``.

        Mirrors Julia's ``rangeof(w)`` (which defaults to ``method=intersect``).
        """
        ranges: list[MITRange] = []
        for v in self._c.values():
            r = _rangeof_member(v)
            if r is not None:
                ranges.append(r)
        if not ranges:
            msg = "Workspace has no rangeable members."
            raise ValueError(msg)
        # All must share a frequency, then intersect.
        head_freq = ranges[0].frequency
        for r in ranges[1:]:
            if r.frequency != head_freq:
                msg = (
                    f"Mixing frequencies not allowed: {prettyprint_frequency(head_freq)} "
                    f"and {prettyprint_frequency(r.frequency)}."
                )
                raise TypeError(msg)
        lo = max(r.start.value for r in ranges)
        hi = min(r.stop.value for r in ranges)
        return MITRange(MIT(head_freq, lo), MIT(head_freq, hi))

    def rangeof_span(self) -> MITRange:
        """Return the smallest range covering all rangeable members.

        Frequency must be consistent across rangeable members. Mirrors
        Julia's ``rangeof_span(values(w)...)`` idiom.
        """
        pieces: list[object] = []
        for v in self._c.values():
            if isinstance(v, Workspace):
                pieces.append(v.rangeof_span())
                continue
            sub = _rangeof_member(v)
            if sub is not None:
                pieces.append(sub)
        return rangeof_span(*pieces)

    # -- equality ----------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Workspace):
            return NotImplemented
        if list(self._c.keys()) != list(other._c.keys()):
            return False
        return all(_values_equal(v, other._c[k]) for k, v in self._c.items())

    __hash__ = None  # type: ignore[assignment]

    # -- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        return _format_workspace(self)

    def __str__(self) -> str:
        return _format_workspace(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_tseries_inplace(t: TSeries) -> None:
    """Trim leading/trailing typenan entries from ``t`` in place.

    Mirrors ``strip!(t::TSeries)`` from ``fconvert_helpers.jl``. Float dtypes
    use NaN as the sentinel; integer dtypes use ``iinfo(dtype).max``. Bool
    dtypes are left alone (their typenan is ``False``, which would erase
    the whole array).
    """
    if t.is_empty():
        return
    arr = t.values
    dt = arr.dtype
    if dt == np.bool_:
        return
    sentinel = typenan(dt)
    valid = ~np.isnan(arr) if np.issubdtype(dt, np.floating) else arr != sentinel
    if not valid.any():
        # All entries are NaN: shrink to empty starting at original firstdate.
        empty = MITRange(t.firstdate, MIT(t.firstdate.frequency, t.firstdate.value - 1))
        t.resize(empty)
        return
    first_valid = int(np.argmax(valid))
    last_valid = len(arr) - 1 - int(np.argmax(valid[::-1]))
    new_start = MIT(t.firstdate.frequency, t.firstdate.value + first_valid)
    new_stop = MIT(t.firstdate.frequency, t.firstdate.value + last_valid)
    t.resize(MITRange(new_start, new_stop))


def _rangeof_member(value: object) -> MITRange | None:
    """Return the range of a Workspace member, or None if it has none."""
    if isinstance(value, TSeries):
        return value.range
    if isinstance(value, MITRange):
        return value
    if isinstance(value, Workspace):
        try:
            return value.rangeof()
        except ValueError:
            return None
    return None


def _values_equal(a: Any, b: Any) -> bool:
    """Equality that copes with TSeries / NumPy / nested Workspace values."""
    if isinstance(a, TSeries) and isinstance(b, TSeries):
        return a.equals(b)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    return bool(a == b)


def _format_workspace(w: Workspace) -> str:
    n = len(w)
    if n == 0:
        return "Empty Workspace"
    header = f"Workspace with {n} variable" + ("" if n == 1 else "s")
    rows: list[str] = []
    max_key = max(len(k) for k in w._c)
    for k, v in w._c.items():
        rows.append(f"  {k.rjust(max_key)} ⇒ {_format_value(v)}")
    return header + "\n" + "\n".join(rows)


def _format_value(v: object) -> str:
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, Workspace):
        n = len(v)
        if n == 0:
            return "Empty Workspace"
        return f"Workspace with {n} variable" + ("" if n == 1 else "s")
    if isinstance(v, TSeries):
        return f"{len(v)}-element TSeries{{{prettyprint_frequency(v.frequency)}}}"
    if isinstance(v, (MIT, Duration, MITRange, Frequency)):
        return str(v)
    if isinstance(v, (int, float, bool, complex)):
        return str(v)
    return type(v).__name__
