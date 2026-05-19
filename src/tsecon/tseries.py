# SPDX-License-Identifier: MIT
"""TSeries — frequency-tagged 1-D time series.

Mirrors ``TimeSeriesEcon.jl``'s ``TSeries``: a value array paired with a
frequency-tagged ``firstdate`` (an :class:`~tsecon.mit.MIT`).

Design: composition over an :class:`numpy.ndarray` plus the
``__array_ufunc__`` / ``__array_function__`` / ``__array__`` protocols. No
subclassing of ``ndarray`` (subclassing is a well-known source of
return-type surprises around the views returned by indexing).

Indexing:

* ``ts[int]`` / ``ts[slice-of-int]`` / ``ts[array-of-int]`` / ``ts[bool-array]``
  — pass through to the underlying NumPy array, returning a scalar or a plain
  ``ndarray`` (matching ``TimeSeriesEcon.jl``).
* ``ts[MIT]`` — frequency-checked scalar lookup.
* ``ts[MITRange]`` (unit step) — returns a new ``TSeries`` over the requested
  range. Step ranges return a plain ``ndarray``.
* ``ts[a:b]`` where ``a`` and ``b`` are MITs — sugar for ``ts[MITRange(a, b)]``.

Assignment with an out-of-range MIT or MITRange auto-extends the underlying
storage, filling the gap with the dtype's NaN sentinel (NaN for floats,
``iinfo(dtype).max`` for ints — see :func:`_typenan`).

Arithmetic and ufuncs flow through ``__array_ufunc__``: TSeries-vs-TSeries
operations align by MIT (intersection range, same-frequency required);
TSeries-vs-scalar and TSeries-vs-array broadcast over the existing range.

Whole-object comparison goes via :meth:`TSeries.equals` /
:meth:`TSeries.allclose`. The Python ``==`` / ``<`` / ``>`` operators are
elementwise (NumPy semantics), so ``TSeries`` is intentionally unhashable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any, ClassVar, Final, Union

import numpy as np
import numpy.typing as npt

from tsecon.frequencies import Frequency, Unit, prettyprint_frequency
from tsecon.linalg import _matmul_strip
from tsecon.mit import MIT, Duration
from tsecon.mitrange import MITRange, rangeof_span

__all__ = ["TSeries", "typenan"]


_ArrayLike = Union[np.ndarray, Sequence[Any]]  # noqa: UP007  (Union for mypy readability)


# ---------------------------------------------------------------------------
# typenan — dtype-appropriate "not a number" sentinel
# ---------------------------------------------------------------------------


def typenan(dtype: npt.DTypeLike) -> Any:
    """Return the not-a-number sentinel for ``dtype``.

    For floats this is ``NaN``; for signed and unsigned integers it is
    ``iinfo(dtype).max``; for booleans it is ``False``. Mirrors
    ``TimeSeriesEcon.jl``'s ``typenan`` (see ``tseries.jl``).
    """
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.floating):
        return np.array(np.nan, dtype=dt)[()]
    if np.issubdtype(dt, np.integer):
        return np.array(np.iinfo(dt).max, dtype=dt)[()]
    if dt == np.bool_:
        return np.bool_(False)
    msg = f"No typenan defined for dtype {dt}."
    raise TypeError(msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mixed_freq_error(left: object, right: object) -> TypeError:
    return TypeError(
        f"Mixing frequencies not allowed: {_freq_label(left)} and {_freq_label(right)}."
    )


def _freq_label(x: object) -> str:
    if isinstance(x, Frequency):
        return prettyprint_frequency(x)
    if isinstance(x, (MIT, Duration)):
        return prettyprint_frequency(x.frequency)
    if isinstance(x, MITRange):
        return prettyprint_frequency(x.frequency)
    if isinstance(x, TSeries):
        return prettyprint_frequency(x._firstdate.frequency)
    return type(x).__name__


def _is_scalar(x: object) -> bool:
    if isinstance(x, (bool, int, float, complex, np.generic)):
        return True
    return isinstance(x, np.ndarray) and x.ndim == 0


def _coerce_values(
    values: Any,
    *,
    dtype: npt.DTypeLike | None,
    copy: bool,
) -> np.ndarray:
    """Coerce ``values`` to a 1-D ndarray, wrapping by default.

    Without ``copy``, returns a view onto ``values`` whenever
    ``np.asarray(values, dtype=dtype)`` returns it unchanged (matching dtype,
    1-D, contiguous). With ``copy=True``, always allocates a fresh,
    non-aliased buffer in a single allocation.
    """
    arr = np.array(values, dtype=dtype, copy=True) if copy else np.asarray(values, dtype=dtype)
    if arr.ndim != 1:
        msg = f"values must be 1-D, got ndim={arr.ndim}."
        raise ValueError(msg)
    return arr


# ---------------------------------------------------------------------------
# TSeries
# ---------------------------------------------------------------------------


class TSeries:
    """A 1-D NumPy array paired with a frequency-tagged ``firstdate``.

    Construct with ``TSeries(firstdate, values)`` where ``firstdate`` is an
    :class:`~tsecon.mit.MIT` or :class:`~tsecon.mitrange.MITRange`. When the
    first argument is an MITRange, the second may be omitted (uninitialized,
    NaN-filled storage), a scalar (fill), or a length-matching array. See
    the class-level constructors :meth:`empty`, :meth:`zeros`, :meth:`ones`,
    :meth:`fill` for keyword-named convenience constructors.
    """

    __slots__ = ("_firstdate", "_values")

    # Bump priority above ndarray so that ``5 + ts`` dispatches through
    # ``TSeries.__array_ufunc__`` rather than the bare ndarray's __radd__.
    __array_priority__: ClassVar[float] = 1000.0

    _firstdate: MIT
    _values: np.ndarray

    # -- construction ------------------------------------------------------

    def __init__(
        self,
        firstdate_or_range: MIT | MITRange,
        values: _ArrayLike | float | int | bool | Callable[[int], np.ndarray] | None = None,
        *,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
    ) -> None:
        """Construct a TSeries.

        Parameters
        ----------
        firstdate_or_range : MIT or MITRange
            Either the MIT of the first stored entry (with ``values`` supplying
            the data) or an MITRange covering the whole series.
        values : array-like, scalar, callable, or None
            * ``None`` (only with an MITRange) — uninitialized storage filled
              with the dtype's NaN sentinel.
            * Scalar — fill the range with the scalar.
            * Array-like — wrap (or copy, with ``copy=True``) as the underlying
              buffer. Length must match the range.
            * Callable (only with an MITRange) — invoked as ``values(len(rng))``;
              the result must be a 1-D ``ndarray`` of matching length. Mirrors
              Julia's ``TSeries(rng, ini::Function)`` (e.g.,
              ``TSeries(rng, np.zeros)`` / ``TSeries(rng, np.ones)``).

        Notes
        -----
        Passing an already-compatible ``ndarray`` as ``values`` *wraps* the
        buffer rather than copying it (matching xarray's ``DataArray``).
        Set ``copy=True`` to force an independent allocation, or call
        :meth:`copy` / :func:`copy.deepcopy` post-construction. The
        wrap-vs-copy contract extends to the callable form: the array returned
        by the callable is the user's, so the default is to wrap.

        Examples
        --------
        >>> rng = MITRange(qq(2020, 1), qq(2020, 4))
        >>> TSeries(rng, np.zeros).values
        array([0., 0., 0., 0.])
        >>> TSeries(rng, np.ones).values
        array([1., 1., 1., 1.])
        """
        if isinstance(firstdate_or_range, MITRange):
            rng = firstdate_or_range
            length = len(rng)
            if callable(values) and not isinstance(values, (np.ndarray, Sequence)):
                result = values(length)
                if not isinstance(result, np.ndarray) or result.ndim != 1:
                    msg = (
                        f"TSeries(rng, callable): callable returned "
                        f"{type(result).__name__} of "
                        f"ndim={getattr(result, 'ndim', '?')}; "
                        f"expected 1-D ndarray of length {length}."
                    )
                    raise ValueError(msg)
                if len(result) != length:
                    msg = (
                        f"TSeries(rng, callable): callable returned length "
                        f"{len(result)}; expected {length} (matching range)."
                    )
                    raise ValueError(msg)
                values = result
            if values is None:
                target_dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float64)
                arr = np.full(length, typenan(target_dtype), dtype=target_dtype)
            elif _is_scalar(values):
                target_dtype = np.dtype(dtype) if dtype is not None else np.asarray(values).dtype
                arr = np.full(length, values, dtype=target_dtype)
            else:
                arr = _coerce_values(values, dtype=dtype, copy=copy)
                if arr.shape[0] != length:
                    msg = (
                        f"Range and data lengths mismatch: range has {length} entries, "
                        f"got {arr.shape[0]}."
                    )
                    raise ValueError(msg)
            self._firstdate = rng.start
            self._values = arr
            return

        if isinstance(firstdate_or_range, MIT):
            fd = firstdate_or_range
            if values is None:
                target_dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float64)
                arr = np.empty(0, dtype=target_dtype)
            elif _is_scalar(values):
                msg = (
                    "TSeries(MIT, scalar) is ambiguous; pass a length-matching array, "
                    "or use TSeries(MITRange, scalar) to fill a range."
                )
                raise TypeError(msg)
            else:
                arr = _coerce_values(values, dtype=dtype, copy=copy)
            self._firstdate = fd
            self._values = arr
            return

        msg = (  # type: ignore[unreachable]
            f"TSeries first argument must be MIT or MITRange, "
            f"got {type(firstdate_or_range).__name__}."
        )
        raise TypeError(msg)

    # -- alternate constructors -------------------------------------------

    @classmethod
    def empty(cls, rng: MITRange, *, dtype: npt.DTypeLike = np.float64) -> TSeries:
        """Construct an uninitialized TSeries over ``rng`` filled with the dtype's NaN."""
        return cls(rng, dtype=dtype)

    @classmethod
    def fill(
        cls,
        rng: MITRange,
        value: float | int | bool,
        *,
        dtype: npt.DTypeLike | None = None,
    ) -> TSeries:
        """Construct a TSeries over ``rng`` filled with ``value``."""
        return cls(rng, value, dtype=dtype)

    @classmethod
    def zeros(cls, rng: MITRange, *, dtype: npt.DTypeLike = np.float64) -> TSeries:
        """Construct a TSeries of zeros over ``rng``."""
        return cls(rng, 0, dtype=dtype)

    @classmethod
    def ones(cls, rng: MITRange, *, dtype: npt.DTypeLike = np.float64) -> TSeries:
        """Construct a TSeries of ones over ``rng``."""
        return cls(rng, 1, dtype=dtype)

    @classmethod
    def trues(cls, rng: MITRange) -> TSeries:
        """Construct a boolean TSeries of ``True`` values over ``rng``."""
        return cls(rng, True, dtype=np.bool_)

    @classmethod
    def falses(cls, rng: MITRange) -> TSeries:
        """Construct a boolean TSeries of ``False`` values over ``rng``."""
        return cls(rng, False, dtype=np.bool_)

    # -- accessors ---------------------------------------------------------

    @property
    def values(self) -> np.ndarray:
        """The underlying 1-D NumPy array. Mutating it mutates the TSeries."""
        return self._values

    @property
    def firstdate(self) -> MIT:
        """The MIT of the first stored entry."""
        return self._firstdate

    @property
    def lastdate(self) -> MIT:
        """The MIT of the last stored entry. Undefined for empty TSeries."""
        return MIT(self._firstdate.frequency, self._firstdate.value + len(self._values) - 1)

    @property
    def frequency(self) -> Frequency:
        """The shared frequency of all entries."""
        return self._firstdate.frequency

    @property
    def range(self) -> MITRange:
        """The MITRange covering ``firstdate..lastdate``.

        Empty TSeries return an empty MITRange (``start > stop``).
        """
        if len(self._values) == 0:
            empty_stop = MIT(self._firstdate.frequency, self._firstdate.value - 1)
            return MITRange(self._firstdate, empty_stop)
        return MITRange(self._firstdate, self.lastdate)

    @property
    def dtype(self) -> np.dtype[Any]:
        """The dtype of the underlying array."""
        return self._values.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying array (always 1-D)."""
        return self._values.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions (always 1)."""
        return 1

    # -- size / iteration --------------------------------------------------

    def __len__(self) -> int:
        return int(self._values.shape[0])

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __bool__(self) -> bool:
        msg = (
            "The truth value of a TSeries is ambiguous. Use .equals(), .allclose(), "
            ".any(), .all(), or len()."
        )
        raise ValueError(msg)

    def is_empty(self) -> bool:
        """Return True iff the underlying array has length zero."""
        return len(self._values) == 0

    def any(self) -> bool:
        """Return ``bool(self.values.any())``."""
        return bool(self._values.any())

    def all(self) -> bool:
        """Return ``bool(self.values.all())``."""
        return bool(self._values.all())

    # -- copy / similar ----------------------------------------------------

    def copy(self, *, deep: bool = False) -> TSeries:
        """Return an independent copy with its own storage.

        The ``deep`` kwarg is accepted for API uniformity with container
        types (:class:`~tsecon.workspace.Workspace`, MVTSeries) where a
        shallow copy would share value references. TSeries has no nested
        containers — the underlying ndarray is always copied — so
        ``deep=True`` is a semantic no-op here.
        """
        del deep  # accepted for uniformity; see docstring
        return TSeries(self._firstdate, self._values.copy())

    def __copy__(self) -> TSeries:
        return self.copy()

    def __deepcopy__(self, memo: dict[int, Any]) -> TSeries:
        new = self.copy()
        memo[id(self)] = new
        return new

    def similar(
        self,
        rng: MITRange | None = None,
        *,
        dtype: npt.DTypeLike | None = None,
    ) -> TSeries:
        """Return an uninitialized TSeries with matching shape (or the given range)."""
        if rng is None:
            rng = self.range
        target_dtype = dtype if dtype is not None else self._values.dtype
        return TSeries.empty(rng, dtype=target_dtype)

    # -- frequency / range helpers -----------------------------------------

    def _check_freq(self, other_freq: Frequency, *, op: str) -> None:
        if self._firstdate.frequency != other_freq:
            msg = (
                f"Mixing frequencies not allowed in {op}: "
                f"{_freq_label(self)} and {_freq_label(other_freq)}."
            )
            raise TypeError(msg)

    def _mit_to_index(self, m: MIT) -> int:
        return int(m.value - self._firstdate.value)

    # -- resize ------------------------------------------------------------

    def resize(self, rng: MITRange) -> TSeries:
        """Extend or shrink storage so the new range equals ``rng``.

        New entries are filled with the dtype's NaN sentinel. The TSeries is
        modified in place; the same object is returned for chaining.
        """
        self._check_freq(rng.frequency, op="resize")
        if rng.step != 1:
            msg = "resize requires a unit-step MITRange."
            raise ValueError(msg)
        new_len = len(rng)
        nan_val = typenan(self._values.dtype)
        if rng.start == self._firstdate:
            old_len = len(self._values)
            if new_len == old_len:
                return self
            new_arr = np.full(new_len, nan_val, dtype=self._values.dtype)
            keep = min(new_len, old_len)
            if keep > 0:
                new_arr[:keep] = self._values[:keep]
            self._values = new_arr
            return self
        # General case: align by MIT.
        new_arr = np.full(new_len, nan_val, dtype=self._values.dtype)
        old_range = self.range
        overlap_start_value = max(old_range.start.value, rng.start.value)
        overlap_stop_value = min(old_range.stop.value, rng.stop.value)
        if overlap_start_value <= overlap_stop_value:
            old_offset = overlap_start_value - self._firstdate.value
            new_offset = overlap_start_value - rng.start.value
            n = overlap_stop_value - overlap_start_value + 1
            new_arr[new_offset : new_offset + n] = self._values[old_offset : old_offset + n]
        self._values = new_arr
        self._firstdate = rng.start
        return self

    def _ensure_covers(self, rng: MITRange) -> None:
        """Extend storage in place so the range covers union(self.range, rng)."""
        if self._firstdate.frequency != rng.frequency:
            raise _mixed_freq_error(self, rng)
        span = rangeof_span(self.range, rng)
        if span != self.range:
            self.resize(span)

    # -- indexing: getitem -------------------------------------------------

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, MIT):
            self._check_freq(key.frequency, op="indexing")
            i = self._mit_to_index(key)
            if not 0 <= i < len(self._values):
                msg = f"MIT {key!s} is outside the stored range {self.range!s}."
                raise IndexError(msg)
            return self._values[i]
        if isinstance(key, MITRange):
            self._check_freq(key.frequency, op="indexing")
            i0 = self._mit_to_index(key.start)
            i1 = self._mit_to_index(key.last())
            if i0 < 0 or i1 >= len(self._values):
                msg = f"MITRange {key!s} is not contained in stored range {self.range!s}."
                raise IndexError(msg)
            if key.step == 1:
                # ts[range] returns a TSeries with its own buffer (per decision 16
                # scope note): pass copy=True so the slice view doesn't escape.
                return TSeries(key.start, self._values[i0 : i1 + 1], copy=True)
            return self._values[i0 : i1 + 1 : key.step].copy()
        if isinstance(key, slice):
            if isinstance(key.start, MIT) or isinstance(key.stop, MIT):
                if key.step is not None and not isinstance(key.step, int):
                    msg = "MIT slice step must be an integer."
                    raise TypeError(msg)
                if not (isinstance(key.start, MIT) and isinstance(key.stop, MIT)):
                    msg = "MIT slice must have MIT endpoints on both sides (no implicit begin/end)."
                    raise TypeError(msg)
                step = key.step if key.step is not None else 1
                return self[MITRange(key.start, key.stop, step)]
            return self._values[key].copy()
        if isinstance(key, bool):
            msg = "TSeries indexing with a bare bool is ambiguous."
            raise TypeError(msg)
        if isinstance(key, (int, np.integer)):
            return self._values[int(key)]
        if isinstance(key, TSeries):
            self._check_freq(key.frequency, op="boolean indexing")
            if key.frequency != self.frequency:
                raise _mixed_freq_error(self, key)
            if key.range != self.range:
                msg = "Boolean TSeries mask must have the same range as the indexed TSeries."
                raise IndexError(msg)
            if key._values.dtype != np.bool_:
                msg = "Boolean TSeries indexing requires a TSeries with dtype=bool."
                raise TypeError(msg)
            return self._values[key._values]
        arr = np.asarray(key)
        if arr.dtype == np.bool_:
            return self._values[arr]
        if np.issubdtype(arr.dtype, np.integer):
            return self._values[arr]
        msg = f"TSeries does not support indexing with {type(key).__name__}."
        raise TypeError(msg)

    # -- indexing: setitem -------------------------------------------------

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, MIT):
            self._check_freq(key.frequency, op="setting")
            if not (
                self._firstdate.value <= key.value <= self._firstdate.value + len(self._values) - 1
            ):
                self._ensure_covers(MITRange(key, key))
            i = self._mit_to_index(key)
            self._values[i] = value
            return
        if isinstance(key, MITRange):
            self._check_freq(key.frequency, op="setting")
            self._ensure_covers(key)
            i0 = self._mit_to_index(key.start)
            i1 = self._mit_to_index(key.last())
            slot = slice(i0, i1 + 1, key.step if key.step != 1 else None)
            if isinstance(value, TSeries):
                self._check_freq(value.frequency, op="setting")
                # Align the source by MIT.
                aligned = value[key]
                self._values[slot] = aligned.values if isinstance(aligned, TSeries) else aligned
                return
            self._values[slot] = value
            return
        if isinstance(key, slice):
            if isinstance(key.start, MIT) or isinstance(key.stop, MIT):
                if not (isinstance(key.start, MIT) and isinstance(key.stop, MIT)):
                    msg = "MIT slice must have MIT endpoints on both sides (no implicit begin/end)."
                    raise TypeError(msg)
                step = key.step if key.step is not None else 1
                self[MITRange(key.start, key.stop, step)] = value
                return
            self._values[key] = value
            return
        if isinstance(key, bool):
            msg = "TSeries indexing with a bare bool is ambiguous."
            raise TypeError(msg)
        if isinstance(key, (int, np.integer)):
            self._values[int(key)] = value
            return
        if isinstance(key, TSeries):
            self._check_freq(key.frequency, op="boolean indexing")
            if key.range != self.range:
                msg = "Boolean TSeries mask must have the same range as the indexed TSeries."
                raise IndexError(msg)
            if key._values.dtype != np.bool_:
                msg = "Boolean TSeries indexing requires a TSeries with dtype=bool."
                raise TypeError(msg)
            self._values[key._values] = value
            return
        arr = np.asarray(key)
        if arr.dtype == np.bool_:
            self._values[arr] = value
            return
        if np.issubdtype(arr.dtype, np.integer):
            self._values[arr] = value
            return
        msg = f"TSeries does not support indexing with {type(key).__name__}."
        raise TypeError(msg)

    # -- NumPy protocols ---------------------------------------------------

    def __array__(self, dtype: npt.DTypeLike | None = None, copy: bool | None = None) -> np.ndarray:
        """Return the underlying values array (frequency information is dropped).

        NumPy ``2.x`` passes ``copy`` to honor ``np.array(ts, copy=...)``. We
        respect it: ``copy=True`` returns an independent array, ``copy=False``
        attempts a zero-copy view and only succeeds when no dtype conversion is
        required.
        """
        if dtype is None or np.dtype(dtype) == self._values.dtype:
            if copy:
                return self._values.copy()
            return self._values
        if copy is False:
            msg = "Cannot honor copy=False when a dtype conversion is required."
            raise ValueError(msg)
        return self._values.astype(dtype)

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        out: Any = None,
        **kwargs: Any,
    ) -> Any:
        # Out-parameter handling is deferred until a real use case emerges.
        if out is not None:
            return NotImplemented

        # If any input is an MVTSeries, defer to its dispatcher (mirrors the
        # Julia precedence: MVTSeriesStyle > TSeriesStyle when mixed).
        for x in inputs:
            if x is not self and type(x).__name__ == "MVTSeries":
                return NotImplemented

        # np.matmul has shape semantics (n?,k),(k,m?)->(n?,m?) that don't fit
        # the element-wise range-intersection path below. Route to linalg.py
        # to match Julia's linalg.jl behavior (strip labels, return ndarray).
        if ufunc is np.matmul and method == "__call__" and len(inputs) == 2:
            return _matmul_strip(inputs[0], inputs[1])

        # Reduce / accumulate / outer / etc. — apply to the bare ndarray.
        if method != "__call__":
            arrays = tuple(np.asarray(x) if isinstance(x, TSeries) else x for x in inputs)
            return getattr(ufunc, method)(*arrays, **kwargs)

        # Compute the common range across TSeries inputs (intersection).
        tseries_inputs = [x for x in inputs if isinstance(x, TSeries)]
        if not tseries_inputs:
            return NotImplemented  # pragma: no cover  (numpy wouldn't call us)

        common_freq = tseries_inputs[0].frequency
        for t in tseries_inputs[1:]:
            if t.frequency != common_freq:
                raise _mixed_freq_error(tseries_inputs[0], t)

        start_value = max(t._firstdate.value for t in tseries_inputs)
        stop_value = min(t.lastdate.value for t in tseries_inputs)

        if start_value > stop_value:
            # Empty intersection → empty TSeries.
            empty_rng = MITRange(
                MIT(common_freq, start_value),
                MIT(common_freq, start_value - 1),
            )
            return TSeries(empty_rng)
        common_range = MITRange(MIT(common_freq, start_value), MIT(common_freq, stop_value))
        n = len(common_range)

        materialized: list[Any] = []
        for x in inputs:
            if isinstance(x, TSeries):
                off = start_value - x._firstdate.value
                materialized.append(x._values[off : off + n])
                continue
            if isinstance(x, np.ndarray):
                if x.ndim == 0 or (x.ndim == 1 and x.shape[0] == n):
                    materialized.append(x)
                    continue
                msg = f"Cannot broadcast array of shape {x.shape} against a TSeries of length {n}."
                raise ValueError(msg)
            materialized.append(x)

        result = ufunc(*materialized, **kwargs)
        if isinstance(result, tuple):
            return tuple(_maybe_wrap(r, common_range) for r in result)
        return _maybe_wrap(result, common_range)

    def __array_function__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        handler = _ARRAY_FUNCTION_HANDLERS.get(func)
        if handler is not None:
            return handler(*args, **kwargs)
        # Fallback: unwrap TSeries → ndarray, call the function, return raw.
        unwrapped_args = tuple(np.asarray(a) if isinstance(a, TSeries) else a for a in args)
        unwrapped_kwargs = {
            k: (np.asarray(v) if isinstance(v, TSeries) else v) for k, v in kwargs.items()
        }
        return func(*unwrapped_args, **unwrapped_kwargs)

    # -- arithmetic dunders ------------------------------------------------
    # NumPy dispatches arithmetic for ndarray operands through __array_ufunc__
    # via ndarray's own __add__/etc. For non-ndarray scalars, Python calls our
    # __add__ first; we route to np.add to keep one code path. Similarly for
    # __r*__.

    def __add__(self, other: Any) -> Any:
        return np.add(self, other)

    def __radd__(self, other: Any) -> Any:
        return np.add(other, self)

    def __sub__(self, other: Any) -> Any:
        return np.subtract(self, other)

    def __rsub__(self, other: Any) -> Any:
        return np.subtract(other, self)

    def __mul__(self, other: Any) -> Any:
        return np.multiply(self, other)

    def __rmul__(self, other: Any) -> Any:
        return np.multiply(other, self)

    def __matmul__(self, other: Any) -> Any:
        # Match Julia's linalg.jl: strip labels, return a plain ndarray.
        # `@` is the PEP 465 spelling of Julia's `*` matrix-product overload.
        return _matmul_strip(self, other)

    def __rmatmul__(self, other: Any) -> Any:
        return _matmul_strip(other, self)

    def __truediv__(self, other: Any) -> Any:
        return np.true_divide(self, other)

    def __rtruediv__(self, other: Any) -> Any:
        return np.true_divide(other, self)

    def __floordiv__(self, other: Any) -> Any:
        return np.floor_divide(self, other)

    def __rfloordiv__(self, other: Any) -> Any:
        return np.floor_divide(other, self)

    def __mod__(self, other: Any) -> Any:
        return np.remainder(self, other)

    def __pow__(self, other: Any) -> Any:
        return np.power(self, other)

    def __rpow__(self, other: Any) -> Any:
        return np.power(other, self)

    def __neg__(self) -> TSeries:
        return TSeries(self._firstdate, np.negative(self._values))

    def __pos__(self) -> TSeries:
        return self.copy()

    def __abs__(self) -> TSeries:
        return TSeries(self._firstdate, np.absolute(self._values))

    # -- elementwise comparisons (NumPy semantics) -------------------------
    # The np.* ufuncs accept TSeries because __array_ufunc__ takes over the
    # dispatch, but mypy's stubs don't know that; per-call ignores are the
    # least invasive option (see decision 02).

    def __eq__(self, other: object) -> Any:
        if not _is_compatible_operand(other):
            return NotImplemented
        return np.equal(self, other)  # type: ignore[call-overload]

    def __ne__(self, other: object) -> Any:
        if not _is_compatible_operand(other):
            return NotImplemented
        return np.not_equal(self, other)  # type: ignore[call-overload]

    def __lt__(self, other: object) -> Any:
        if not _is_compatible_operand(other):
            return NotImplemented
        return np.less(self, other)  # type: ignore[call-overload]

    def __le__(self, other: object) -> Any:
        if not _is_compatible_operand(other):
            return NotImplemented
        return np.less_equal(self, other)  # type: ignore[call-overload]

    def __gt__(self, other: object) -> Any:
        if not _is_compatible_operand(other):
            return NotImplemented
        return np.greater(self, other)  # type: ignore[call-overload]

    def __ge__(self, other: object) -> Any:
        if not _is_compatible_operand(other):
            return NotImplemented
        return np.greater_equal(self, other)  # type: ignore[call-overload]

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    # -- whole-object comparison ------------------------------------------

    def equals(self, other: object) -> bool:
        """Return True iff ``other`` is a TSeries with identical freq, range, and values."""
        if not isinstance(other, TSeries):
            return False
        if self.frequency != other.frequency:
            return False
        if self._firstdate != other._firstdate:
            return False
        return np.array_equal(self._values, other._values, equal_nan=False)

    def allclose(self, other: object, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Return True iff ``other`` is a same-shape TSeries with close values."""
        if not isinstance(other, TSeries):
            return False
        if self.frequency != other.frequency:
            return False
        if self._firstdate != other._firstdate or len(self) != len(other):
            return False
        return bool(np.allclose(self._values, other._values, rtol=rtol, atol=atol, equal_nan=True))

    # -- repr / str --------------------------------------------------------

    def __repr__(self) -> str:
        return _format_tseries(self)

    def __str__(self) -> str:
        return _format_tseries(self)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _maybe_wrap(result: Any, common_range: MITRange) -> Any:
    if isinstance(result, np.ndarray) and result.ndim == 1 and result.shape[0] == len(common_range):
        return TSeries(common_range.start, result)
    return result


def _is_compatible_operand(other: object) -> bool:
    return isinstance(
        other,
        (TSeries, np.ndarray, np.generic, int, float, complex, bool, list, tuple),
    )


# ---------------------------------------------------------------------------
# __array_function__ dispatch table
# ---------------------------------------------------------------------------


def _np_concatenate(*args: Any, **kwargs: Any) -> Any:
    # numpy.concatenate(arrays, axis=0, out=None, dtype=None, casting='same_kind')
    if "out" in kwargs and kwargs["out"] is not None:
        msg = "np.concatenate with out= is not supported for TSeries."
        raise TypeError(msg)
    arrays = args[0]
    if not isinstance(arrays, (list, tuple)):
        msg = "np.concatenate requires a sequence of arrays."
        raise TypeError(msg)
    # If the first element is a TSeries, use its frequency and firstdate;
    # subsequent TSeries arguments must share frequency (their firstdate is
    # ignored — mirrors Julia's `vcat`, which keeps the first arg's firstdate).
    head = arrays[0]
    if isinstance(head, TSeries):
        freq = head.frequency
        firstdate = head._firstdate
        pieces = []
        for a in arrays:
            if isinstance(a, TSeries):
                if a.frequency != freq:
                    raise _mixed_freq_error(head, a)
                pieces.append(a._values)
            else:
                pieces.append(np.asarray(a))
        return TSeries(firstdate, np.concatenate(pieces, **kwargs))
    return np.concatenate([np.asarray(a) for a in arrays], **kwargs)


def _np_array_equal(*args: Any, **kwargs: Any) -> bool:
    a, b = args[0], args[1]
    if isinstance(a, TSeries):
        a = a._values
    if isinstance(b, TSeries):
        b = b._values
    return bool(np.array_equal(a, b, **kwargs))


def _np_allclose(*args: Any, **kwargs: Any) -> bool:
    a, b = args[0], args[1]
    if isinstance(a, TSeries):
        a = a._values
    if isinstance(b, TSeries):
        b = b._values
    return bool(np.allclose(a, b, **kwargs))


_ARRAY_FUNCTION_HANDLERS: Final[dict[Any, Any]] = {
    np.concatenate: _np_concatenate,
    np.array_equal: _np_array_equal,
    np.allclose: _np_allclose,
}


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def _format_tseries(t: TSeries) -> str:
    typestr = f"TSeries{{{prettyprint_frequency(t.frequency)}"
    if t._values.dtype != np.float64:
        typestr += f",{t._values.dtype}"
    typestr += "}"
    n = len(t._values)
    if n == 0:
        return f"Empty {typestr} starting {t._firstdate}"
    header = f"{n}-element {typestr} with range {t.range}"
    rows: list[str] = []
    mit_strs = [str(t._firstdate + i) for i in range(n)]
    pad = max(len(s) for s in mit_strs) + 2
    # Truncate long series in the middle for readability.
    threshold = 20
    if n > threshold:
        head = 8
        tail = 8
        for i in range(head):
            rows.append(f"{mit_strs[i].rjust(pad)} : {t._values[i]}")
        rows.append("    ⋮")
        for i in range(n - tail, n):
            rows.append(f"{mit_strs[i].rjust(pad)} : {t._values[i]}")
    else:
        for i in range(n):
            rows.append(f"{mit_strs[i].rjust(pad)} : {t._values[i]}")
    return header + ":\n" + "\n".join(rows)


# Keep import alive (used in type annotations) — avoids ruff F401 for Unit.
_ = (Unit, Duration)
