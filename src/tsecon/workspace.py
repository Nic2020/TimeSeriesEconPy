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

The :func:`copyto` free function mirrors Julia's
``Base.copyto!(::MVTSeries, ::Workspace; verbose=, trange=)``: an in-place
materialiser that writes Workspace members into the matching columns of a
pre-allocated MVTSeries without re-allocating the matrix buffer.

Deferred until later milestones (currently blocked on other ported modules):

* ``overlay`` / ``compare`` / ``compare_equal`` / ``reindex`` — live in
  ``various.jl`` (M4 🔵).
* ``@weval`` — Julia macro; no direct Python equivalent. Use a normal
  closure or ``eval`` if needed.
* ``clean_old_frequencies`` — Julia compatibility shim for legacy quarterly
  numbering; no analogue needed in Python.

The standalone ``strip!(t::TSeries)`` helper now lives in the ported fconvert
subpackage as :func:`tsecon.fconvert.strip_tseries_inplace`; the workspace
delegates to it.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, ItemsView, Iterable, Iterator, KeysView, Mapping, ValuesView
from copy import deepcopy
from typing import Any

import numpy as np

from tsecon.fconvert import strip_tseries_inplace
from tsecon.frequencies import Frequency, prettyprint_frequency
from tsecon.mit import MIT, Duration
from tsecon.mitrange import MITRange, rangeof_span
from tsecon.mvtseries import MVTSeries
from tsecon.tseries import TSeries

__all__ = ["Workspace", "copyto"]


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

    def copy(self, *, deep: bool = False) -> Workspace:
        """Return a copy of the Workspace.

        With ``deep=False`` (the default) the storage dict is fresh but
        values are shared by reference — Python-dict semantics. With
        ``deep=True``, every value is recursively :func:`copy.deepcopy`-ed,
        equivalent to ``copy.deepcopy(self)``. The kwarg matches the
        :meth:`TSeries.copy` signature for API uniformity; see
        ``claude_files/decisions/16_constructor_copy_semantics.md``.
        """
        if deep:
            return deepcopy(self)
        out = Workspace()
        out._c.update(self._c)
        return out

    def __copy__(self) -> Workspace:
        return self.copy()

    def __deepcopy__(self, memo: dict[int, Any]) -> Workspace:
        out = Workspace()
        memo[id(self)] = out
        out._c.update({k: deepcopy(v, memo) for k, v in self._c.items()})
        return out

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

        Delegates the per-TSeries trim to
        :func:`tsecon.fconvert.strip_tseries_inplace`. Bool-dtype TSeries are
        left alone (their typenan is ``False``, which would erase the array).
        """
        for value in self._c.values():
            if isinstance(value, TSeries):
                strip_tseries_inplace(value)
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

    def rangeof(self, *, method: str = "intersect") -> MITRange:
        """Return the combined range of all rangeable members.

        Frequency-bearing members must share a single frequency, else
        ``TypeError`` is raised. Members without a range (scalars, strings,
        etc.) are skipped. If no member has a range, raises ``ValueError``.

        Parameters
        ----------
        method
            ``"intersect"`` (default) returns the intersection of all member
            ranges. ``"union"`` returns the spanning union, equivalent to
            :meth:`rangeof_span`. Mirrors Julia's ``rangeof(w; method=intersect)``
            policy switch.
        """
        if method == "union":
            return self.rangeof_span()
        if method != "intersect":
            msg = f"method must be 'intersect' or 'union', got {method!r}."
            raise ValueError(msg)
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


# ---------------------------------------------------------------------------
# copyto — in-place materialiser (Julia: Base.copyto!(::MVTSeries, ::Workspace))
# ---------------------------------------------------------------------------


def copyto(
    dst: MVTSeries,
    src: Workspace,
    *,
    verbose: bool = False,
    trange: MITRange | None = None,
) -> MVTSeries:
    """Copy Workspace members into a pre-allocated MVTSeries in place.

    Mirrors Julia's ``Base.copyto!(x::MVTSeries, w::AbstractWorkspace;
    verbose=false, trange=rangeof(x))``. For each column name in
    ``dst.column_names``, the matching ``src[name]`` :class:`TSeries` is
    written into the destination column over the overlap of ``trange`` and
    the source TSeries's range; ``dst._values`` is mutated, never replaced.

    Parameters
    ----------
    dst
        Destination MVTSeries. Its matrix buffer is written through; storage
        identity is preserved (``id(dst._values)`` is unchanged on return).
    src
        Source Workspace. Only members whose names appear in
        ``dst.column_names`` are visited; extra Workspace entries are ignored.
    verbose
        When True, a single :class:`UserWarning` is emitted at the end of
        the loop listing every column that was not written because its name
        is absent from ``src``. Matches Julia's end-of-loop ``@warn`` shape.
    trange
        Optional :class:`MITRange` restricting the rows to write. Defaults
        to ``dst.range``. Must share ``dst``'s frequency, have ``step == 1``,
        and be contained in ``dst.range``. The actual rows written for any
        given column are the intersection of ``trange`` with that column's
        source :class:`TSeries` range (so a source that doesn't cover all
        of ``trange`` writes only the overlap, never raises).

    Returns
    -------
    dst : MVTSeries
        The same object passed in (enables chaining; storage identity
        preserved).

    Raises
    ------
    TypeError
        If ``trange`` (or, by default, ``dst``'s frequency) does not match a
        matching source member's frequency, or if a Workspace member at a
        matching key is not a :class:`TSeries`.
    ValueError
        If ``trange`` has a non-unit step.
    IndexError
        If ``trange`` is not contained in ``dst.range``.

    Examples
    --------
    Tight model-simulation loop reusing the same buffer:

    >>> import tsecon as ts
    >>> buf = ts.MVTSeries(ts.MITRange(ts.qq(2020, 1), ts.qq(2024, 4)), ["a", "b"])
    >>> for _ in range(100):
    ...     w = _step(...)               # produces a Workspace with TSeries 'a', 'b'
    ...     ts.copyto(buf, w)            # in-place; no MVTSeries reallocation

    See Also
    --------
    MVTSeries : the destination type.
    Workspace : the source type.
    """
    if not isinstance(dst, MVTSeries):
        msg = f"copyto dst must be MVTSeries, got {type(dst).__name__}."  # type: ignore[unreachable]
        raise TypeError(msg)
    if not isinstance(src, Workspace):
        msg = f"copyto src must be Workspace, got {type(src).__name__}."  # type: ignore[unreachable]
        raise TypeError(msg)

    target_range = trange if trange is not None else dst.range
    if target_range.frequency != dst.frequency:
        msg = (
            f"copyto: trange has frequency {prettyprint_frequency(target_range.frequency)}, "
            f"expected {prettyprint_frequency(dst.frequency)}."
        )
        raise TypeError(msg)
    if target_range.step != 1:
        msg = f"copyto: trange must have step=1, got step={target_range.step}."
        raise ValueError(msg)
    if not dst.is_empty() and (
        target_range.start.value < dst.firstdate.value
        or target_range.stop.value > dst.lastdate.value
    ):
        msg = f"copyto: trange {target_range!s} is not contained in dst range {dst.range!s}."
        raise IndexError(msg)

    missing: list[str] = []
    for name in dst.column_names:
        if name not in src:
            if verbose:
                missing.append(name)
            continue
        src_val = src[name]
        if not isinstance(src_val, TSeries):
            msg = (
                f"copyto: workspace member {name!r} is {type(src_val).__name__}, expected TSeries."
            )
            raise TypeError(msg)
        if src_val.frequency != dst.frequency:
            msg = (
                f"copyto: workspace member {name!r} has frequency "
                f"{prettyprint_frequency(src_val.frequency)}, expected "
                f"{prettyprint_frequency(dst.frequency)}."
            )
            raise TypeError(msg)
        lo = max(target_range.start.value, src_val.firstdate.value)
        hi = min(target_range.stop.value, src_val.lastdate.value)
        if lo > hi:
            continue
        overlap = MITRange(MIT(dst.frequency, lo), MIT(dst.frequency, hi))
        dst[overlap, name] = src_val[overlap]

    if verbose and missing:
        warnings.warn(
            f"Variables not copied (missing from Workspace): {', '.join(missing)}",
            UserWarning,
            stacklevel=2,
        )
    return dst
