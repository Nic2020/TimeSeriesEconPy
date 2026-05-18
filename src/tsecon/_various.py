# SPDX-License-Identifier: MIT
"""Misc helpers from ``TimeSeriesEcon.jl``'s ``various.jl``.

Three sibling functions sharing the ``LikeWorkspace`` dispatch (Workspace /
MVTSeries / :class:`collections.abc.Mapping`):

* :func:`overlay` — first-valid-wins composition over TSeries / Workspace /
  MVTSeries values. NaN (and the dtype-appropriate ``typenan``) marks
  "missing"; the leftmost non-missing value wins position-by-position.
* :func:`compare` — recursive structural / numeric comparison. Returns a
  :class:`CompareResult` (truthy on equality; ``.differences`` carries the
  per-leaf diff). Tolerance kwargs (``atol`` / ``rtol`` / ``nans``) and
  walk kwargs (``ignoremissing`` / ``showequal`` / ``quiet`` / ``trange``)
  match the Julia surface name-for-name.
* :func:`reindex` — shift every MIT-keyed position inside a container so
  that ``from`` maps to ``to``; the underlying values are preserved.
  Dispatches over MIT / MITRange / TSeries / MVTSeries / Workspace.

Julia's ``@compare`` macro collapses to a plain function alias here: the
macro existed only to capture the variable names for printing, and
:class:`CompareResult.differences` already exposes that information without
needing AST rewriting. Code that wrote ``@compare(v1, v2, atol=1e-5)``
ports to ``compare(v1, v2, atol=1e-5)`` — the Python idiom of returning a
structured diff matches the Julia behaviour for both interactive (``print(result)``)
and programmatic (``result.differences``) use.

Remaining pieces of ``various.jl`` are deferred:

* ``@showall`` — Julia macro for non-truncating display; Python uses
  ``print(repr(x))`` directly (the truncation lives in our ``__repr__``,
  not in a separate IO context).
* ``clean_old_frequencies`` — Julia v0.4→v0.5 frequency-sanitiser; not
  applicable to tsecon's cached-singleton frequency model. Tracked as
  parity gap G14.
* ``TOML.print`` overloads — left to v1.0 if the JSON round-trip
  (``io/json.py``) doesn't cover the use case.
* ``is_yearly`` / ``is_quarterly`` / etc. — already ported in
  :mod:`tsecon.frequencies` with PEP-8 underscore naming.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy as _shallow_copy
from dataclasses import dataclass, field
from math import sqrt
from typing import Any

import numpy as np

from tsecon.frequencies import prettyprint_frequency
from tsecon.mit import MIT
from tsecon.mitrange import MITRange, rangeof_span
from tsecon.mvtseries import MVTSeries
from tsecon.tseries import TSeries, typenan
from tsecon.workspace import Workspace

__all__ = [
    "CompareDifference",
    "CompareResult",
    "compare",
    "overlay",
    "reindex",
]


# ---------------------------------------------------------------------------
# LikeWorkspace dispatch helpers
# ---------------------------------------------------------------------------


# Julia's LikeWorkspace = Union{Workspace, MVTSeries, AbstractDict{Symbol,Any}}.
# Python's Mapping covers the dict case; we accept any Mapping subclass.
_LIKE_WORKSPACE_TYPES: tuple[type, ...] = (Workspace, MVTSeries, Mapping)


def _is_like_workspace(x: object) -> bool:
    return isinstance(x, _LIKE_WORKSPACE_TYPES)


def _ws_keys(x: object) -> list[str]:
    if isinstance(x, Workspace):
        return list(x.keys())
    if isinstance(x, MVTSeries):
        return list(x.column_names)
    if isinstance(x, Mapping):
        return [str(k) for k in x]
    msg = f"Not a LikeWorkspace container: {type(x).__name__}."
    raise TypeError(msg)


def _ws_get(x: object, key: str) -> Any:
    if isinstance(x, Workspace):
        return x[key]
    if isinstance(x, MVTSeries):
        return x[key]
    if isinstance(x, Mapping):
        if key in x:
            return x[key]
        # Allow non-str keys to round-trip via str equivalence.
        for k in x:
            if str(k) == key:
                return x[k]
        raise KeyError(key)
    msg = f"Not a LikeWorkspace container: {type(x).__name__}."
    raise TypeError(msg)


def _ws_has(x: object, key: str) -> bool:
    if isinstance(x, Workspace):
        return key in x
    if isinstance(x, MVTSeries):
        return key in x.column_names
    if isinstance(x, Mapping):
        if key in x:
            return True
        return any(str(k) == key for k in x)
    return False


def _check_same_freq(things: tuple[Any, ...], op: str) -> None:
    first_freq = things[0].frequency
    for t in things[1:]:
        if t.frequency != first_freq:
            msg = (
                f"Mixing frequencies not allowed in {op}: "
                f"{prettyprint_frequency(first_freq)} and {prettyprint_frequency(t.frequency)}."
            )
            raise TypeError(msg)


# ---------------------------------------------------------------------------
# typenan / istypenan
# ---------------------------------------------------------------------------


def _istypenan_scalar(x: Any) -> bool:
    """Return True iff ``x`` is the dtype-appropriate not-a-number sentinel.

    Mirrors Julia's ``istypenan``: NaN for floats; ``iinfo(dtype).max`` for
    NumPy integers; True for ``None``. Plain Python bools / ints / strings
    are never considered "typenan" (Julia treats them the same way through
    the catch-all ``istypenan(x) = false``).
    """
    if x is None:
        return True
    if isinstance(x, bool):
        return False
    if isinstance(x, float):
        return bool(np.isnan(x))
    if isinstance(x, complex):
        return bool(np.isnan(x.real)) or bool(np.isnan(x.imag))
    if isinstance(x, np.generic):
        if isinstance(x, np.bool_):
            return False
        if np.issubdtype(type(x), np.floating):
            return bool(np.isnan(x))
        if np.issubdtype(type(x), np.integer):
            # The issubdtype check above guarantees type(x) is an np.integer
            # subclass at runtime; mypy can't narrow type[np.generic] to the
            # specific integer type variable that np.iinfo wants.
            return int(x) == int(np.iinfo(type(x)).max)  # type: ignore[type-var]
        if np.issubdtype(type(x), np.complexfloating):
            return bool(np.isnan(x))
    return False


def _typenan_mask(arr: np.ndarray) -> np.ndarray:
    """Boolean mask where True marks positions equal to the dtype's typenan."""
    if np.issubdtype(arr.dtype, np.floating):
        return np.isnan(arr)  # type: ignore[no-any-return]
    if np.issubdtype(arr.dtype, np.integer):
        return arr == np.iinfo(arr.dtype).max  # type: ignore[no-any-return]
    if arr.dtype == np.bool_:
        # tsecon convention: bool typenan is False (see tseries.typenan).
        return ~arr
    if np.issubdtype(arr.dtype, np.complexfloating):
        return np.isnan(arr)  # type: ignore[no-any-return]
    return np.zeros(arr.shape, dtype=bool)


# ---------------------------------------------------------------------------
# overlay
# ---------------------------------------------------------------------------


def overlay(*args: Any, rng: MITRange | None = None) -> Any:
    """Return the first non-missing value, position-by-position, from left to right.

    "Missing" is defined by the dtype-appropriate ``typenan`` (NaN for
    floats, ``iinfo(dtype).max`` for integers, ``False`` for booleans).
    The dispatch is on the argument types — all arguments must agree on a
    container family for the recursive form to apply:

    * **All :class:`~tsecon.tseries.TSeries`** — build a new TSeries over
      either ``rng=`` (when given) or the union of all input ranges
      (``rangeof_span(*args)``). For each position, the leftmost input
      that has a non-typenan value at that position wins. The output
      dtype is the NumPy promotion of all input dtypes.
    * **All :class:`~tsecon.mvtseries.MVTSeries`** — return an MVTSeries
      over the union range and the ordered union of column names; each
      column is the overlay of the corresponding TSeries columns.
    * **All Workspace / MVTSeries / Mapping** — return a
      :class:`~tsecon.workspace.Workspace`. For each key present in any
      input, collect the values across inputs that have it and recurse;
      the recursive call may dispatch back into the TSeries / Workspace
      / MVTSeries / scalar paths.
    * **Mixed types** (e.g. a TSeries and a scalar) — fall through to
      the scalar walk: return the leftmost non-typenan value.

    Parameters
    ----------
    *args
        One or more values to overlay. At least one is required.
    rng
        Optional :class:`~tsecon.mitrange.MITRange` to force the output
        range. Only valid when all arguments are TSeries; raises
        :class:`TypeError` otherwise.

    Returns
    -------
    Any
        A TSeries / MVTSeries / Workspace / scalar matching the input
        family.

    Raises
    ------
    TypeError
        On empty argument list, on mixed-frequency inputs, or on
        ``rng=`` passed with non-TSeries arguments.
    """
    if not args:
        msg = "overlay() requires at least one argument."
        raise TypeError(msg)

    if all(isinstance(a, TSeries) for a in args):
        if rng is None:
            rng = rangeof_span(*(t.range for t in args))
        elif not isinstance(rng, MITRange):
            # Defensive runtime check: the signature types ``rng`` as
            # ``MITRange | None`` so mypy considers this branch unreachable,
            # but callers may pass Any-typed values (e.g. from JSON
            # deserialization) and we'd rather raise at the call site than
            # produce a confusing AttributeError deep inside _overlay_tseries.
            msg = f"overlay rng= must be an MITRange; got {type(rng).__name__}."  # type: ignore[unreachable]
            raise TypeError(msg)
        return _overlay_tseries(rng, *args)

    if rng is not None:
        msg = "overlay(..., rng=) is only valid when all arguments are TSeries."
        raise TypeError(msg)

    if all(isinstance(a, MVTSeries) for a in args):
        return _overlay_mvtseries(*args)

    if all(_is_like_workspace(a) for a in args):
        return _overlay_workspaces(*args)

    # Scalar / mixed fallback: first non-typenan wins (Julia's
    # `overlay(head, tail...) = istypenan(head) ? overlay(tail...) : head`).
    return _overlay_scalar(args)


def _overlay_scalar(args: tuple[Any, ...]) -> Any:
    if len(args) == 1:
        return args[0]
    for a in args:
        if not _istypenan_scalar(a):
            return a
    return args[-1]


def _overlay_tseries(rng: MITRange, *tseries: TSeries) -> TSeries:
    _check_same_freq(tseries, op="overlay")
    if rng.frequency != tseries[0].frequency:
        msg = (
            f"Mixing frequencies not allowed in overlay: "
            f"{prettyprint_frequency(rng.frequency)} (rng) and "
            f"{prettyprint_frequency(tseries[0].frequency)} (TSeries)."
        )
        raise TypeError(msg)
    promoted = np.result_type(*(t.values.dtype for t in tseries))
    n = len(rng)
    nan_val = typenan(promoted)
    out_vals = np.full(n, nan_val, dtype=promoted)
    rng_lo = rng.start.value
    rng_hi = rng.last().value if not rng.is_empty() else rng_lo - 1
    for t in tseries:
        # Early-exit when nothing left to fill (Julia's `any(istypenan, ret) || break`).
        if not _typenan_mask(out_vals).any():
            break
        if len(t) == 0:
            continue
        t_lo = t.firstdate.value
        t_hi = t.firstdate.value + len(t) - 1
        lo = max(t_lo, rng_lo)
        hi = min(t_hi, rng_hi)
        if hi < lo:
            continue
        src_a = lo - t_lo
        src_b = hi - t_lo + 1
        dst_a = lo - rng_lo
        dst_b = hi - rng_lo + 1
        src = np.asarray(t.values[src_a:src_b], dtype=promoted)
        dst_view = out_vals[dst_a:dst_b]
        dst_missing = _typenan_mask(dst_view)
        src_valid = ~_typenan_mask(src)
        mask = dst_missing & src_valid
        dst_view[mask] = src[mask]
    return TSeries(rng.start, out_vals)


def _overlay_workspaces(*workspaces: Any) -> Workspace:
    # Preserve left-to-right key ordering for determinism. (Julia uses
    # `mapreduce(keys, union, …)` which is unordered; walking inputs in
    # order gives a stable, tutorial-friendly result.)
    seen: dict[str, None] = {}
    for w in workspaces:
        for k in _ws_keys(w):
            seen.setdefault(k, None)
    out = Workspace()
    for name in seen:
        things = [_ws_get(w, name) for w in workspaces if _ws_has(w, name)]
        out[name] = overlay(*things)
    return out


def _overlay_mvtseries(*mvtss: MVTSeries) -> MVTSeries:
    _check_same_freq(mvtss, op="overlay")
    rng = rangeof_span(*(m.range for m in mvtss))
    seen: dict[str, None] = {}
    for m in mvtss:
        for n in m.column_names:
            seen.setdefault(n, None)
    names = list(seen)
    promoted = np.result_type(*(m.values.dtype for m in mvtss))
    nrows = len(rng)
    arr = np.empty((nrows, len(names)), dtype=promoted)
    for i, name in enumerate(names):
        cols = [m[name] for m in mvtss if name in m.column_names]
        overlaid = _overlay_tseries(rng, *cols)
        arr[:, i] = overlaid.values
    return MVTSeries(rng, names, arr)


# ---------------------------------------------------------------------------
# compare / @compare
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompareDifference:
    """One line of the recursive diff produced by :func:`compare`.

    Attributes
    ----------
    path
        Hierarchical tuple of names, e.g. ``('_', 'y', '2020Q3')``.
    message
        Short human-readable description: ``'different'``,
        ``'missing in <name>'``, or ``'same'``.
    """

    path: tuple[str, ...]
    message: str

    def __str__(self) -> str:
        return f"{'.'.join(self.path)}: {self.message}"


@dataclass
class CompareResult:
    """The structured return value of :func:`compare`.

    Truthy iff the two inputs compared as equal — use
    ``if compare(a, b): ...`` for the Julia-style one-line check.
    ``str(result)`` reproduces the printed diff (one line per
    :class:`CompareDifference`). ``.differences`` exposes the diff
    programmatically.
    """

    equal: bool
    differences: list[CompareDifference] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.equal

    def __str__(self) -> str:
        if not self.differences:
            return "(no differences)"
        return "\n".join(str(d) for d in self.differences)

    def __repr__(self) -> str:
        n = len(self.differences)
        suffix = "" if n == 1 else "s"
        return f"CompareResult(equal={self.equal}, {n} difference{suffix})"


class _Missing:
    """Sentinel for keys absent from one side of a recursive comparison."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<MISSING>"


_MISSING = _Missing()


_DEFAULT_RTOL = sqrt(float(np.finfo(np.float64).eps))


def compare(
    x: Any,
    y: Any,
    *,
    name: str = "_",
    showequal: bool = False,
    ignoremissing: bool = False,
    quiet: bool = False,
    left: str = "left",
    right: str = "right",
    atol: float = 0.0,
    rtol: float | None = None,
    nans: bool = False,
    trange: MITRange | None = None,
) -> CompareResult:
    r"""Recursively compare ``x`` and ``y``, returning a :class:`CompareResult`.

    The walk dispatches over:

    * :class:`~tsecon.workspace.Workspace`, :class:`~tsecon.mvtseries.MVTSeries`,
      and :class:`collections.abc.Mapping` — treated as name-keyed
      containers and recursed key-by-key. Missing keys produce
      ``"missing in <left|right>"`` lines (unless ``ignoremissing=True``).
    * :class:`~tsecon.tseries.TSeries` — compared range-by-range using
      :func:`numpy.allclose` with the given ``atol`` / ``rtol`` / ``nans``.
      Mismatched ranges count as different (unless ``ignoremissing=True``,
      in which case the intersection is compared).
    * NumPy arrays of numbers — compared by shape + element-wise
      :func:`numpy.allclose`.
    * NumPy arrays of other dtypes — walked element-by-element.
    * Scalars — compared via :func:`numpy.isclose`.
    * Everything else — compared with ``==``.

    Parameters
    ----------
    x, y
        The two values to compare.
    name
        Top-level name in the printed/structured diff (``"_"`` by default —
        matches Julia's ``Symbol("_")``).
    showequal
        When True, also emit ``"name: same"`` lines for matching leaves.
        Default False (only differences are reported, plus a final
        top-level summary line).
    ignoremissing
        When True, ignore keys present in one side but not the other
        (they don't print and don't affect ``equal``). Default False.
    quiet
        When True, suppress printing to stdout. The :class:`CompareResult`
        is still populated either way.
    left, right
        Names used in ``"missing in <name>"`` messages. Default
        ``"left"`` / ``"right"``.
    atol, rtol, nans
        Tolerances forwarded to :func:`numpy.isclose` /
        :func:`numpy.allclose`. ``rtol=None`` (the default) resolves to
        :math:`\\sqrt{\\varepsilon}` when ``atol == 0`` and to ``0`` when
        ``atol > 0`` (matching Julia's ``isapprox`` precedent).
    trange
        Optional :class:`~tsecon.mitrange.MITRange` restricting the
        compared window for TSeries-vs-TSeries leaves only.

    Returns
    -------
    CompareResult
        Truthy iff ``x`` and ``y`` compared as equal; ``.differences``
        carries the recursive diff regardless of ``quiet``.

    Notes
    -----
    Julia's accompanying ``@compare`` macro folds into this same
    function. The macro existed only to capture the input variable names
    for printing; Python callers can pass ``left=`` and ``right=`` if
    they want the same labelling.
    """
    if rtol is None:
        rtol = _DEFAULT_RTOL if atol == 0.0 else 0.0

    result = CompareResult(equal=True)
    kw: dict[str, Any] = {
        "showequal": showequal,
        "ignoremissing": ignoremissing,
        "quiet": quiet,
        "left": left,
        "right": right,
        "atol": atol,
        "rtol": rtol,
        "nans": nans,
        "trange": trange,
    }
    _compare_recurse(x, y, [name], result, kw)
    return result


def _compare_recurse(
    x: Any,
    y: Any,
    path: list[str],
    result: CompareResult,
    kw: dict[str, Any],
) -> bool:
    """Apply the Julia top-level ``compare()`` logic for one (x, y) pair.

    Returns True iff the pair is equal. Side-effects: appends 1+
    :class:`CompareDifference` to ``result.differences``; may set
    ``result.equal = False``; may print to stdout when ``quiet=False``.
    """
    is_top = len(path) == 1

    if x is _MISSING:
        if kw["ignoremissing"]:
            return True
        _record(path, f"missing in {kw['left']}", result, kw["quiet"])
        result.equal = False
        return False
    if y is _MISSING:
        if kw["ignoremissing"]:
            return True
        _record(path, f"missing in {kw['right']}", result, kw["quiet"])
        result.equal = False
        return False

    eq = _equal_dispatch(x, y, path, result, kw)
    if eq:
        if kw["showequal"] or is_top:
            _record(path, "same", result, kw["quiet"])
        return True
    _record(path, "different", result, kw["quiet"])
    result.equal = False
    return False


def _record(path: list[str], message: str, result: CompareResult, quiet: bool) -> None:
    diff = CompareDifference(path=tuple(path), message=message)
    result.differences.append(diff)
    if not quiet:
        print(str(diff))


def _equal_dispatch(
    x: Any,
    y: Any,
    path: list[str],
    result: CompareResult,
    kw: dict[str, Any],
) -> bool:
    """Dispatch ``compare_equal`` on the (type(x), type(y)) pair."""
    # TSeries-vs-TSeries: numeric isapprox over aligned range
    if isinstance(x, TSeries) and isinstance(y, TSeries):
        return _equal_tseries(x, y, kw)

    # MVTSeries / Workspace / Mapping — walk by key (LikeWorkspace)
    if _is_like_workspace(x) and _is_like_workspace(y):
        return _equal_like_workspace(x, y, path, result, kw)

    # ndarray-of-number vs ndarray-of-number — isapprox
    if (
        isinstance(x, np.ndarray)
        and isinstance(y, np.ndarray)
        and np.issubdtype(x.dtype, np.number)
        and np.issubdtype(y.dtype, np.number)
    ):
        return _equal_number_array(x, y, kw)

    # ndarray of object / mixed dtype — element-by-element walk (rare;
    # mirrors Julia's `compare_equal(::AbstractArray, ::AbstractArray)`).
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.shape == y.shape:
        return _equal_object_array(x, y, path, result, kw)

    # MIT / Duration / MITRange / scalars — try numeric equality, else ==
    if _is_number_scalar(x) and _is_number_scalar(y):
        return _equal_number_scalar(x, y, kw)

    # Fallback: structural equality.
    try:
        return bool(x == y)
    except (TypeError, ValueError):
        return False


def _is_number_scalar(x: Any) -> bool:
    if isinstance(x, bool):
        return True
    if isinstance(x, (int, float, complex)):
        return True
    if isinstance(x, np.generic):
        return bool(np.issubdtype(type(x), np.number)) or isinstance(x, np.bool_)
    return False


def _equal_number_scalar(x: Any, y: Any, kw: dict[str, Any]) -> bool:
    return bool(
        np.isclose(
            np.asarray(x, dtype=np.float64),
            np.asarray(y, dtype=np.float64),
            atol=kw["atol"],
            rtol=kw["rtol"],
            equal_nan=kw["nans"],
        )
    )


def _equal_number_array(x: np.ndarray, y: np.ndarray, kw: dict[str, Any]) -> bool:
    if x.shape != y.shape:
        return False
    return bool(np.allclose(x, y, atol=kw["atol"], rtol=kw["rtol"], equal_nan=kw["nans"]))


def _equal_tseries(x: TSeries, y: TSeries, kw: dict[str, Any]) -> bool:
    if x.frequency != y.frequency:
        return False
    trange = kw["trange"]
    if trange is not None and trange.frequency == x.frequency:
        rngx = _intersect_range(trange, x.range)
        rngy = _intersect_range(trange, y.range)
    else:
        rngx = x.range
        rngy = y.range

    if kw["ignoremissing"]:
        trng = _intersect_range(rngx, rngy)
    else:
        if rngx != rngy:
            return False
        trng = rngx

    if trng.is_empty():
        return True

    xs = x[trng]
    ys = y[trng]
    xv = xs.values if isinstance(xs, TSeries) else np.asarray(xs)
    yv = ys.values if isinstance(ys, TSeries) else np.asarray(ys)
    return bool(np.allclose(xv, yv, atol=kw["atol"], rtol=kw["rtol"], equal_nan=kw["nans"]))


def _intersect_range(a: MITRange, b: MITRange) -> MITRange:
    if a.is_empty() or b.is_empty():
        # Pick a canonical empty (matches MITRange constructor invariant).
        return MITRange(a.start, MIT(a.frequency, a.start.value - 1))
    lo = max(a.start.value, b.start.value)
    hi = min(a.last().value, b.last().value)
    if hi < lo:
        return MITRange(a.start, MIT(a.frequency, a.start.value - 1))
    return MITRange(MIT(a.frequency, lo), MIT(a.frequency, hi))


def _equal_like_workspace(
    x: Any,
    y: Any,
    path: list[str],
    result: CompareResult,
    kw: dict[str, Any],
) -> bool:
    seen: dict[str, None] = {}
    for k in _ws_keys(x):
        seen.setdefault(k, None)
    for k in _ws_keys(y):
        seen.setdefault(k, None)

    equal = True
    for key in seen:
        xv = _ws_get(x, key) if _ws_has(x, key) else _MISSING
        yv = _ws_get(y, key) if _ws_has(y, key) else _MISSING
        if not _compare_recurse(xv, yv, [*path, key], result, kw):
            equal = False
    return equal


def _equal_object_array(
    x: np.ndarray,
    y: np.ndarray,
    path: list[str],
    result: CompareResult,
    kw: dict[str, Any],
) -> bool:
    equal = True
    for idx in np.ndindex(x.shape):
        idx_str = ",".join(str(i) for i in idx) if len(idx) > 1 else str(idx[0])
        if not _compare_recurse(x[idx], y[idx], [*path, idx_str], result, kw):
            equal = False
    return equal


# ---------------------------------------------------------------------------
# reindex
# ---------------------------------------------------------------------------


def reindex(
    x: Any,
    old_to_new: tuple[MIT, MIT],
    *,
    copy: bool = False,
) -> Any:
    """Shift every MIT-keyed position so that ``old`` maps to ``new``.

    ``old_to_new`` is a 2-tuple ``(old_mit, new_mit)``. In MIT-value
    space, every output position is the input position plus
    ``new_mit.value - old_mit.value``, and the result carries
    ``new_mit.frequency`` as its frequency label.

    Parameters
    ----------
    x
        The value to reindex. Supported types:

        * :class:`~tsecon.mit.MIT` — single instant.
        * :class:`~tsecon.mitrange.MITRange` — both endpoints shifted;
          ``step`` is preserved.
        * :class:`~tsecon.tseries.TSeries` — anchor shifted; values
          unchanged.
        * :class:`~tsecon.mvtseries.MVTSeries` — same, on the matrix.
        * :class:`~tsecon.workspace.Workspace` — members whose
          frequency matches ``old_mit.frequency`` are reindexed
          (recursively for nested Workspaces); other members are
          carried through.

    old_to_new
        Pair ``(old_mit, new_mit)``: the input MIT (or one matching
        the input's frequency) to remap, and the target MIT (with
        the desired output frequency).
    copy
        When True, force an independent values buffer for the returned
        :class:`~tsecon.tseries.TSeries` / :class:`~tsecon.mvtseries.MVTSeries`
        and apply :func:`copy.copy` to non-reindexable Workspace members.
        Default False matches the wrap-by-default contract of
        :mod:`tsecon` constructors (see the project's design notes on
        constructor copy semantics).

    Returns
    -------
    The same kind of object as ``x``, with the MIT label shifted.

    Raises
    ------
    TypeError
        On a malformed ``old_to_new`` argument, an unsupported ``x``
        type, or a frequency mismatch between ``x`` and ``old_mit``.

    Examples
    --------
    Re-anchor a quarterly TSeries from a 2021Q1 origin to a Unit
    1-based origin::

        >>> from tsecon import TSeries, qq, period, Unit, reindex
        >>> ts = TSeries(qq(2021, 1), [1.0, 2.0, 3.0])
        >>> reindex(ts, (qq(2021, 1), period(Unit(), 1))).firstdate
        1U
    """
    if not isinstance(old_to_new, tuple) or len(old_to_new) != 2:
        # Defensive: signature types `old_to_new` as `tuple[MIT, MIT]` so
        # mypy considers this branch unreachable, but Any-typed callers
        # (e.g. JSON deserialization) need the friendly error.
        msg = (  # type: ignore[unreachable]
            "reindex pair must be a 2-tuple (old_mit, new_mit); "
            f"got {type(old_to_new).__name__}."
        )
        raise TypeError(msg)
    old_mit, new_mit = old_to_new
    if not isinstance(old_mit, MIT) or not isinstance(new_mit, MIT):
        msg = (  # type: ignore[unreachable]
            "reindex pair must contain two MIT instances; got "
            f"({type(old_mit).__name__}, {type(new_mit).__name__})."
        )
        raise TypeError(msg)
    return _reindex_dispatch(x, old_mit, new_mit, copy=copy)


def _reindex_dispatch(x: Any, old_mit: MIT, new_mit: MIT, *, copy: bool) -> Any:
    if isinstance(x, MIT):
        _check_reindex_freq(x.frequency, old_mit.frequency, "MIT")
        delta = x.value - old_mit.value
        return MIT(new_mit.frequency, new_mit.value + delta)
    if isinstance(x, MITRange):
        _check_reindex_freq(x.frequency, old_mit.frequency, "MITRange")
        start_delta = x.start.value - old_mit.value
        new_start = MIT(new_mit.frequency, new_mit.value + start_delta)
        if x.is_empty():
            return MITRange(new_start, MIT(new_mit.frequency, new_start.value - 1), x.step)
        new_stop_value = new_start.value + (len(x) - 1) * x.step
        new_stop = MIT(new_mit.frequency, new_stop_value)
        return MITRange(new_start, new_stop, x.step)
    if isinstance(x, TSeries):
        _check_reindex_freq(x.frequency, old_mit.frequency, "TSeries")
        ts_lag = x.firstdate.value - old_mit.value
        new_first = MIT(new_mit.frequency, new_mit.value + ts_lag)
        return TSeries(new_first, x.values, copy=copy)
    if isinstance(x, MVTSeries):
        _check_reindex_freq(x.frequency, old_mit.frequency, "MVTSeries")
        ts_lag = x.firstdate.value - old_mit.value
        new_first = MIT(new_mit.frequency, new_mit.value + ts_lag)
        return MVTSeries(new_first, list(x.column_names), x.values, copy=copy)
    if isinstance(x, Workspace):
        return _reindex_workspace(x, old_mit, new_mit, copy=copy)
    msg = f"reindex does not support type {type(x).__name__}."
    raise TypeError(msg)


def _reindex_workspace(w: Workspace, old_mit: MIT, new_mit: MIT, *, copy: bool) -> Workspace:
    out = Workspace()
    freq_from = old_mit.frequency
    for k, v in w.items():
        if isinstance(v, Workspace):
            out[k] = _reindex_workspace(v, old_mit, new_mit, copy=copy)
            continue
        if (
            isinstance(v, (MIT, MITRange, TSeries, MVTSeries))
            and v.frequency == freq_from
        ):
            out[k] = _reindex_dispatch(v, old_mit, new_mit, copy=copy)
            continue
        # Non-reindexable member: passthrough, copying when asked.
        if copy:
            try:
                out[k] = _shallow_copy(v)
            except TypeError:
                out[k] = v
        else:
            out[k] = v
    return out


def _check_reindex_freq(target_freq: Any, pair_freq: Any, kind: str) -> None:
    if target_freq != pair_freq:
        msg = (
            f"Mixing frequencies not allowed in reindex({kind}): "
            f"{prettyprint_frequency(target_freq)} (target) and "
            f"{prettyprint_frequency(pair_freq)} (pair[0])."
        )
        raise TypeError(msg)
