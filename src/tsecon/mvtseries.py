# SPDX-License-Identifier: MIT
"""MVTSeries — multivariate, frequency-tagged time series.

Mirrors ``TimeSeriesEcon.jl``'s ``MVTSeries`` (``mvtseries.jl``): a 2-D
NumPy array whose rows correspond to moments in time and whose columns
correspond to named variables of a single, uniform ``dtype``.

Storage follows the Julia design: a contiguous ``ndarray`` of shape
``(nrows, ncols)`` plus a per-column ``TSeries`` "anchor" whose
``.values`` is a view onto the matrix column. Mutating ``mvts.a[date] = v``
writes back into the parent matrix.

Construction follows
``claude_files/decisions/16_constructor_copy_semantics.md``: a compatible
2-D ``ndarray`` passed as ``values`` is **wrapped** by default;
``copy=True`` forces an independent allocation. The constructor's
fast-path mirrors xarray's ``DataArray``.

This session ports the foundational ~60% of the Julia API: storage,
construction, single- and two-argument indexing, dot access, copy /
similar / equals, integer / boolean indexing, and ``__array__`` /
``__len__`` / ``__iter__``. The custom ``__array_ufunc__`` /
``__array_function__`` MVTSeries-aware dispatch, ``rename_columns_inplace``,
the Julia-flavoured aligned pretty-print, and the
``mvts_broadcast.jl`` machinery are deferred to a follow-up session.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from copy import deepcopy
from typing import Any, ClassVar, Union

import numpy as np
import numpy.typing as npt

from tsecon.frequencies import Frequency, prettyprint_frequency
from tsecon.mit import MIT
from tsecon.mitrange import MITRange, rangeof_span
from tsecon.tseries import TSeries, typenan

__all__ = ["MVTSeries"]


_ArrayLike = Union[np.ndarray, Sequence[Any]]  # noqa: UP007  (Union for mypy readability)
_NamesLike = Union[str, Iterable[Any]]  # noqa: UP007


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _names_as_list(names: _NamesLike) -> list[str]:
    """Coerce a name spec into a list[str] preserving order.

    Mirrors the Julia ``_names_as_vec`` helper. A bare ``str`` is treated
    as a single column name (not iterated character-by-character).
    """
    if isinstance(names, str):
        return [names]
    return [str(n) for n in names]


def _mixed_freq_error(left: object, right: object) -> TypeError:
    def _label(x: object) -> str:
        if isinstance(x, Frequency):
            return prettyprint_frequency(x)
        if isinstance(x, MIT):
            return prettyprint_frequency(x.frequency)
        if isinstance(x, MITRange):
            return prettyprint_frequency(x.frequency)
        if isinstance(x, (TSeries, MVTSeries)):
            return prettyprint_frequency(x.frequency)
        return type(x).__name__

    return TypeError(f"Mixing frequencies not allowed: {_label(left)} and {_label(right)}.")


def _is_scalar_number(x: object) -> bool:
    return isinstance(x, (bool, int, float, complex, np.generic))


def _is_int_like(x: object) -> bool:
    return isinstance(x, (int, np.integer)) and not isinstance(x, bool)


def _coerce_values_2d(
    values: Any,
    *,
    dtype: npt.DTypeLike | None,
    copy: bool,
) -> np.ndarray:
    """Coerce ``values`` to a 2-D ndarray, wrapping by default (decision 16).

    A 1-D input is reshaped to ``(n, 1)`` (column vector); a 2-D input is
    used as-is. ``copy=True`` forces a single fresh allocation.
    """
    arr = np.array(values, dtype=dtype, copy=True) if copy else np.asarray(values, dtype=dtype)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        msg = f"MVTSeries values must be 1-D or 2-D, got ndim={arr.ndim}."
        raise ValueError(msg)
    return arr


def _build_column_views(
    firstdate: MIT,
    names: list[str],
    matrix: np.ndarray,
) -> dict[str, TSeries]:
    """Build the column-name → TSeries-view dict for the given matrix.

    Each entry's ``.values`` is the matching column slice of ``matrix``,
    so mutations through the TSeries write back into ``matrix``.
    """
    cols: dict[str, TSeries] = {}
    for i, nm in enumerate(names):
        # TSeries(firstdate, ndarray, copy=False) wraps the column view;
        # writes through the TSeries propagate back to ``matrix``.
        cols[nm] = TSeries(firstdate, matrix[:, i])
    return cols


# ---------------------------------------------------------------------------
# MVTSeries
# ---------------------------------------------------------------------------


class MVTSeries:
    """A 2-D NumPy array paired with a frequency-tagged firstdate and named columns.

    See module docstring for the storage model; see
    ``claude_files/decisions/16_constructor_copy_semantics.md`` for the
    wrap-vs-copy contract on the constructor.
    """

    __slots__ = ("_columns", "_firstdate", "_values")

    __array_priority__: ClassVar[float] = 1000.0

    _firstdate: MIT
    _values: np.ndarray
    _columns: dict[str, TSeries]

    # -- construction ------------------------------------------------------

    def __init__(
        self,
        firstdate_or_range: MIT | MITRange | None = None,
        names: _NamesLike | None = None,
        values: _ArrayLike | TSeries | Callable[..., Any] | float | int | bool | None = None,
        *,
        dtype: npt.DTypeLike | None = None,
        copy: bool = False,
        **columns: Any,
    ) -> None:
        """Construct an MVTSeries.

        Parameters
        ----------
        firstdate_or_range
            An :class:`~tsecon.mit.MIT` (giving the first row date) or a
            :class:`~tsecon.mitrange.MITRange` (giving both first date and
            number of rows). If omitted and column kwargs are provided, the
            range is taken to be ``rangeof_span(*columns.values())``.
        names
            A column name (``str``), an iterable of names, or omitted to
            use the kwargs ``**columns`` to supply both names and values.
        values
            One of:

            * a 2-D ``ndarray`` (or 1-D for a single-column MVTSeries),
            * a scalar (fill),
            * a callable initializer ``init(nrows, ncols) -> ndarray``
              such as :func:`numpy.zeros` / :func:`numpy.ones`,
            * a :class:`~tsecon.tseries.TSeries` (init each column from it),
            * ``None`` (uninitialized → NaN-filled / typenan-filled).
        dtype
            Optional dtype override. Defaults to ``float64`` for empty /
            scalar / function initializers; for an array input, the dtype
            of that array.
        copy
            ``True`` forces an independent allocation when ``values`` is an
            ndarray. Default ``False`` (wrap-by-default per decision 16).
        **columns
            Alternative spelling: ``MVTSeries(rng, a=..., b=...)`` builds
            an MVTSeries with one column per kwarg. The value can be a
            :class:`~tsecon.tseries.TSeries`, a 1-D array, or a scalar
            (filled). If ``firstdate_or_range`` is omitted, the range is
            ``rangeof_span`` of all kwarg values that have a range.

        Notes
        -----
        Wrap-by-default: passing an already-compatible 2-D ``ndarray`` as
        ``values`` aliases that buffer (matches xarray's ``DataArray``).
        Use ``copy=True`` for an independent allocation. See
        ``claude_files/decisions/16_constructor_copy_semantics.md``.
        """
        # -- kwargs-only form: MVTSeries(rng?, **columns)
        if columns and (values is None and names is None):
            self._init_from_kwargs(firstdate_or_range, columns, dtype=dtype)
            return

        # -- bare default: MVTSeries() == MVTSeries(1U, 0x0)
        if firstdate_or_range is None:
            msg = "MVTSeries() requires either a firstdate/range or column kwargs."
            raise TypeError(msg)

        names_list: list[str] = [] if names is None else _names_as_list(names)
        ncols = len(names_list)

        # -- MITRange form
        if isinstance(firstdate_or_range, MITRange):
            rng = firstdate_or_range
            nrows = len(rng)

            target_dtype: np.dtype[Any]
            arr: np.ndarray

            if values is None:
                target_dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float64)
                arr = np.full((nrows, ncols), typenan(target_dtype), dtype=target_dtype)
            elif callable(values) and not isinstance(values, np.ndarray):
                # Initializer function. NumPy convention is one shape tuple
                # (``np.zeros((nrows, ncols))``); we also accept the two-
                # positional-arg form (``np.random.rand(nrows, ncols)``) by
                # trying the tuple call first and falling back on TypeError.
                init_fn = values
                try:
                    arr = np.asarray(init_fn((nrows, ncols)))
                except TypeError:
                    arr = np.asarray(init_fn(nrows, ncols))
                if dtype is not None:
                    arr = arr.astype(dtype, copy=False)
            elif _is_scalar_number(values):
                target_dtype = np.dtype(dtype) if dtype is not None else np.asarray(values).dtype
                arr = np.full((nrows, ncols), values, dtype=target_dtype)
            elif isinstance(values, TSeries):
                # Initialize all columns from the same TSeries (truncated/aligned).
                src = values
                if src.frequency != rng.frequency:
                    raise _mixed_freq_error(src.frequency, rng.frequency)
                target_dtype = np.dtype(dtype) if dtype is not None else np.dtype(src.values.dtype)
                arr = np.full((nrows, ncols), typenan(target_dtype), dtype=target_dtype)
                # Place src values where rng and src.range overlap.
                lo = max(rng.start.value, src.firstdate.value)
                hi = min(rng.stop.value, src.lastdate.value)
                if lo <= hi:
                    rng_off = lo - rng.start.value
                    src_off = lo - src.firstdate.value
                    n = hi - lo + 1
                    for j in range(ncols):
                        arr[rng_off : rng_off + n, j] = src.values[src_off : src_off + n]
            else:
                arr = _coerce_values_2d(values, dtype=dtype, copy=copy)
                if arr.shape != (nrows, ncols):
                    msg = (
                        "Number of periods and variables do not match size of data: "
                        f"({nrows}, {ncols}) != {arr.shape}."
                    )
                    raise ValueError(msg)
            self._firstdate = rng.start
            self._values = arr
            self._columns = _build_column_views(rng.start, names_list, arr)
            return

        # -- MIT form (firstdate only; nrows inferred from values or 0)
        if isinstance(firstdate_or_range, MIT):
            fd = firstdate_or_range
            if values is None:
                target_dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float64)
                arr = np.zeros((0, ncols), dtype=target_dtype)
            elif _is_scalar_number(values):
                msg = (
                    "MVTSeries(MIT, names, scalar) is ambiguous (unknown row count); "
                    "pass an MITRange to fill, or a 2-D array to size."
                )
                raise TypeError(msg)
            else:
                arr = _coerce_values_2d(values, dtype=dtype, copy=copy)
                if arr.shape[1] != ncols:
                    msg = f"Number of names and columns don't match: {ncols} != {arr.shape[1]}."
                    raise ValueError(msg)
            self._firstdate = fd
            self._values = arr
            self._columns = _build_column_views(fd, names_list, arr)
            return

        msg = (  # type: ignore[unreachable]
            f"MVTSeries first argument must be MIT, MITRange, or None; "
            f"got {type(firstdate_or_range).__name__}."
        )
        raise TypeError(msg)

    def _init_from_kwargs(
        self,
        firstdate_or_range: MIT | MITRange | None,
        columns: dict[str, Any],
        *,
        dtype: npt.DTypeLike | None,
    ) -> None:
        """Build from ``MVTSeries(rng?, name=value, ...)`` kwargs form."""
        # Figure out the range.
        rng: MITRange
        if isinstance(firstdate_or_range, MITRange):
            rng = firstdate_or_range
        elif isinstance(firstdate_or_range, MIT):
            # Compute end from the longest-tailed TSeries kwarg, else 0 rows.
            last_value = firstdate_or_range.value - 1
            for v in columns.values():
                if isinstance(v, TSeries):
                    last_value = max(last_value, v.lastdate.value)
                elif isinstance(v, np.ndarray) or (
                    isinstance(v, list) and not _is_scalar_number(v)
                ):
                    n = len(v)
                    last_value = max(last_value, firstdate_or_range.value + n - 1)
            stop = MIT(firstdate_or_range.frequency, last_value)
            rng = MITRange(firstdate_or_range, stop)
        else:
            # Span across all TSeries / MITRange kwargs.
            spans = [
                v.range if isinstance(v, TSeries) else v
                for v in columns.values()
                if isinstance(v, (TSeries, MITRange))
            ]
            if not spans:
                msg = (
                    "MVTSeries() with kwargs only requires at least one TSeries or MITRange "
                    "value to determine the range."
                )
                raise TypeError(msg)
            rng = rangeof_span(*spans)

        # Resolve element type.
        if dtype is not None:
            et: np.dtype[Any] = np.dtype(dtype)
        else:
            promoted: list[np.dtype[Any]] = [np.dtype(np.float64)]
            for v in columns.values():
                if isinstance(v, TSeries):
                    promoted.append(v.values.dtype)
                elif isinstance(v, np.ndarray):
                    promoted.append(v.dtype)
                elif isinstance(v, (bool, np.bool_)):
                    promoted.append(np.dtype(np.bool_))
                elif _is_scalar_number(v):
                    promoted.append(np.asarray(v).dtype)
            et = np.result_type(*promoted) if promoted else np.dtype(np.float64)

        nrows = len(rng)
        names_list = list(columns.keys())
        ncols = len(names_list)
        arr = np.full((nrows, ncols), typenan(et), dtype=et)
        self._firstdate = rng.start
        self._values = arr
        self._columns = _build_column_views(rng.start, names_list, arr)

        # Now copy each kwarg's data into its column.
        for j, (name, val) in enumerate(columns.items()):
            col_anchor = self._columns[name]
            if isinstance(val, TSeries):
                if val.frequency != rng.frequency:
                    raise _mixed_freq_error(val.frequency, rng.frequency)
                lo = max(rng.start.value, val.firstdate.value)
                hi = min(rng.stop.value, val.lastdate.value)
                if lo <= hi:
                    rng_off = lo - rng.start.value
                    src_off = lo - val.firstdate.value
                    n = hi - lo + 1
                    arr[rng_off : rng_off + n, j] = val.values[src_off : src_off + n]
            elif _is_scalar_number(val):
                arr[:, j] = val
            else:
                vec = np.asarray(val, dtype=et)
                if vec.ndim != 1 or vec.shape[0] != nrows:
                    msg = (
                        f"Column {name!r}: vector length {vec.shape[0]} does not match "
                        f"MVTSeries range length {nrows}."
                    )
                    raise ValueError(msg)
                arr[:, j] = vec
            # Re-anchor the column TSeries to the (re-)written column (the
            # buffer view is unchanged, but we keep the same TSeries instance).
            del col_anchor  # explicitly unused — we wrote through arr above

    # -- alternate constructors -------------------------------------------

    @classmethod
    def empty(
        cls,
        rng: MITRange,
        names: _NamesLike = (),
        *,
        dtype: npt.DTypeLike = np.float64,
    ) -> MVTSeries:
        """Construct an uninitialized MVTSeries (typenan-filled)."""
        return cls(rng, names, dtype=dtype)

    @classmethod
    def zeros(
        cls,
        rng: MITRange,
        names: _NamesLike,
        *,
        dtype: npt.DTypeLike = np.float64,
    ) -> MVTSeries:
        """Construct an MVTSeries of zeros."""
        return cls(rng, names, 0, dtype=dtype)

    @classmethod
    def ones(
        cls,
        rng: MITRange,
        names: _NamesLike,
        *,
        dtype: npt.DTypeLike = np.float64,
    ) -> MVTSeries:
        """Construct an MVTSeries of ones."""
        return cls(rng, names, 1, dtype=dtype)

    @classmethod
    def fill(
        cls,
        rng: MITRange,
        names: _NamesLike,
        value: float | int | bool,
        *,
        dtype: npt.DTypeLike | None = None,
    ) -> MVTSeries:
        """Construct an MVTSeries filled with ``value``."""
        return cls(rng, names, value, dtype=dtype)

    # -- accessors ---------------------------------------------------------

    @property
    def values(self) -> np.ndarray:
        """The underlying 2-D NumPy array. Mutating it mutates the MVTSeries."""
        return self._values

    @property
    def firstdate(self) -> MIT:
        """The MIT of the first stored row."""
        return self._firstdate

    @property
    def lastdate(self) -> MIT:
        """The MIT of the last stored row. Undefined when the MVTSeries is empty."""
        return MIT(self._firstdate.frequency, self._firstdate.value + self._values.shape[0] - 1)

    @property
    def frequency(self) -> Frequency:
        """The shared frequency of all rows."""
        return self._firstdate.frequency

    @property
    def range(self) -> MITRange:
        """The MITRange covering ``firstdate..lastdate``.

        For an empty MVTSeries (no rows) returns an empty MITRange.
        """
        if self._values.shape[0] == 0:
            empty_stop = MIT(self._firstdate.frequency, self._firstdate.value - 1)
            return MITRange(self._firstdate, empty_stop)
        return MITRange(self._firstdate, self.lastdate)

    @property
    def column_names(self) -> tuple[str, ...]:
        """The column names in insertion order."""
        return tuple(self._columns.keys())

    @property
    def columns(self) -> dict[str, TSeries]:
        """The column-name → TSeries-view mapping (live; do not mutate keys)."""
        return self._columns

    @property
    def dtype(self) -> np.dtype[Any]:
        """The dtype of the underlying matrix."""
        return self._values.dtype

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the underlying matrix ``(nrows, ncols)``."""
        return (int(self._values.shape[0]), int(self._values.shape[1]))

    @property
    def ndim(self) -> int:
        """Number of dimensions (always 2)."""
        return 2

    # -- size / iteration --------------------------------------------------

    def __len__(self) -> int:
        return int(self._values.shape[0])

    def __iter__(self) -> Iterator[np.ndarray]:
        # Row iteration, mirroring NumPy matrix iteration.
        return iter(self._values)

    def __contains__(self, key: object) -> bool:
        # Membership = column-name membership (matches Julia ``haskey``).
        return isinstance(key, str) and key in self._columns

    def keys(self) -> tuple[str, ...]:
        """Return the column names (insertion order)."""
        return tuple(self._columns.keys())

    def is_empty(self) -> bool:
        """Return True iff the MVTSeries has no rows or no columns."""
        return bool(self._values.shape[0] == 0 or self._values.shape[1] == 0)

    # -- copy / similar ----------------------------------------------------

    def copy(self, *, deep: bool = False) -> MVTSeries:
        """Return an independent copy with its own matrix buffer.

        ``deep`` is accepted for API uniformity with
        :meth:`~tsecon.tseries.TSeries.copy` and
        :meth:`~tsecon.workspace.Workspace.copy`; for MVTSeries it is a
        semantic no-op because the only mutable referents below the
        wrapper are the matrix buffer (always copied) and the per-column
        TSeries views (which are rebuilt to point at the fresh buffer).
        See ``claude_files/decisions/16_constructor_copy_semantics.md``.
        """
        del deep  # accepted for uniformity; see docstring
        return MVTSeries(
            self._firstdate,
            list(self._columns.keys()),
            self._values.copy(),
        )

    def __copy__(self) -> MVTSeries:
        return self.copy()

    def __deepcopy__(self, memo: dict[int, Any]) -> MVTSeries:
        new = self.copy()
        memo[id(self)] = new
        return new

    def similar(
        self,
        *,
        dtype: npt.DTypeLike | None = None,
        rng: MITRange | None = None,
        names: _NamesLike | None = None,
    ) -> MVTSeries:
        """Return an uninitialized MVTSeries with matching shape (or overrides)."""
        out_rng = rng if rng is not None else self.range
        out_names = list(self._columns.keys()) if names is None else _names_as_list(names)
        out_dtype = dtype if dtype is not None else self._values.dtype
        return MVTSeries.empty(out_rng, out_names, dtype=out_dtype)

    # -- frequency / range helpers -----------------------------------------

    def _check_freq_for(self, other_freq: Frequency, *, op: str) -> None:
        if self._firstdate.frequency != other_freq:
            msg = (
                f"Mixing frequencies not allowed in {op}: "
                f"{prettyprint_frequency(self.frequency)} and "
                f"{prettyprint_frequency(other_freq)}."
            )
            raise TypeError(msg)

    def _row_index(self, m: MIT) -> int:
        return int(m.value - self._firstdate.value)

    def _col_index(self, name: str) -> int:
        try:
            return list(self._columns.keys()).index(name)
        except ValueError as e:
            msg = f"MVTSeries has no column {name!r}."
            raise KeyError(msg) from e

    # -- dot access --------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only invoked when normal attribute lookup fails. Look up ``name``
        # in the column dict — if found, return the TSeries anchor (which
        # the user can mutate to write back into the parent matrix).
        try:
            return object.__getattribute__(self, "_columns")[name]
        except (AttributeError, KeyError) as e:
            msg = f"MVTSeries has no column {name!r}."
            raise AttributeError(msg) from e

    def __setattr__(self, name: str, value: Any) -> None:
        if name in MVTSeries.__slots__:
            object.__setattr__(self, name, value)
            return
        # Reroute to column setitem (matches Julia ``setproperty!``).
        if "_columns" in object.__dir__(self) and name in self._columns:
            self._set_column(name, value)
            return
        msg = (
            f"Cannot create new column {name!r} via attribute assignment. "
            f"Build a new MVTSeries with the additional column."
        )
        raise AttributeError(msg)

    def __dir__(self) -> list[str]:
        return [*object.__dir__(self), *self._columns.keys()]

    # -- indexing: getitem -------------------------------------------------

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple) and len(key) == 2 and not _is_name_tuple(key):
            return self._getitem_two(key[0], key[1])
        return self._getitem_one(key)

    def _getitem_one(self, key: Any) -> Any:
        # String → column TSeries.
        if isinstance(key, str):
            try:
                return self._columns[key]
            except KeyError as e:
                msg = f"MVTSeries has no column {key!r}."
                raise KeyError(msg) from e

        # MIT → row vector (1-D ndarray).
        if isinstance(key, MIT):
            self._check_freq_for(key.frequency, op="indexing")
            i = self._row_index(key)
            if not 0 <= i < self._values.shape[0]:
                msg = f"MIT {key!s} is outside the stored range {self.range!s}."
                raise IndexError(msg)
            return self._values[i, :]

        # MITRange → MVTSeries (rows, same columns).
        if isinstance(key, MITRange):
            self._check_freq_for(key.frequency, op="indexing")
            i0 = self._row_index(key.start)
            i1 = self._row_index(key.last())
            if i0 < 0 or i1 >= self._values.shape[0]:
                msg = f"MITRange {key!s} is not contained in stored range {self.range!s}."
                raise IndexError(msg)
            sub = self._values[i0 : i1 + 1, :].copy()
            return MVTSeries(key.start, list(self._columns.keys()), sub)

        # Tuple/list of column names → subset MVTSeries (copy).
        if isinstance(key, list) or (isinstance(key, tuple) and self._is_name_collection(key)):
            names = [str(n) for n in key]
            self._check_columns_exist(names)
            inds = [self._col_index(n) for n in names]
            sub = self._values[:, inds].copy()
            return MVTSeries(self._firstdate, names, sub)

        # Bool array on rows (1-D length nrows): return submatrix.
        if isinstance(key, np.ndarray) and key.dtype == np.bool_:
            if key.ndim == 1 and key.shape[0] == self._values.shape[0]:
                return self._values[key, :]
            # 2-D boolean: treat as flat NumPy fancy indexing.
            return self._values[key]

        # Bool list: convert and recurse.
        if isinstance(key, list) and all(isinstance(b, (bool, np.bool_)) for b in key):
            return self._getitem_one(np.asarray(key, dtype=bool))

        # Slice without MIT endpoints → fall through to ndarray.
        if isinstance(key, slice):
            if isinstance(key.start, MIT) or isinstance(key.stop, MIT):
                return self._slice_with_mit(key)
            return self._values[key]

        # Integer or integer-array → fall through to ndarray.
        if _is_int_like(key):
            return self._values[int(key)]
        if isinstance(key, np.ndarray) and np.issubdtype(key.dtype, np.integer):
            return self._values[key]

        msg = f"MVTSeries does not support indexing with {type(key).__name__}."
        raise TypeError(msg)

    def _getitem_two(self, r: Any, c: Any) -> Any:
        # Colon shortcuts.
        if isinstance(r, slice) and r == slice(None) and isinstance(c, slice) and c == slice(None):
            return self
        if isinstance(r, slice) and r == slice(None):
            return self._getitem_one(c)
        if isinstance(c, slice) and c == slice(None):
            return self._getitem_one(r)

        # Bool row-mask + (Symbol or list).
        if isinstance(r, np.ndarray) and r.dtype == np.bool_ and r.ndim == 1:
            if isinstance(c, str):
                col_idx = self._col_index(c)
                return self._values[r, col_idx]
            if isinstance(c, (list, tuple)):
                col_inds = np.asarray([self._col_index(str(n)) for n in c], dtype=np.intp)
                return self._values[np.ix_(r, col_inds)]
            if isinstance(c, np.ndarray) and c.dtype == np.bool_:
                return self._values[np.ix_(r, c)]

        # MIT + str → scalar.
        if isinstance(r, MIT) and isinstance(c, str):
            self._check_freq_for(r.frequency, op="indexing")
            i = self._row_index(r)
            j = self._col_index(c)
            if not 0 <= i < self._values.shape[0]:
                msg = f"MIT {r!s} is outside the stored range {self.range!s}."
                raise IndexError(msg)
            return self._values[i, j]

        # MIT + collection → row slice across multiple columns.
        if isinstance(r, MIT) and isinstance(c, (list, tuple)):
            self._check_freq_for(r.frequency, op="indexing")
            i = self._row_index(r)
            if not 0 <= i < self._values.shape[0]:
                msg = f"MIT {r!s} is outside the stored range {self.range!s}."
                raise IndexError(msg)
            row_inds = np.asarray([self._col_index(str(n)) for n in c], dtype=np.intp)
            return self._values[i, row_inds]

        # MITRange + str → TSeries (column over range).
        if isinstance(r, MITRange) and isinstance(c, str):
            self._check_freq_for(r.frequency, op="indexing")
            i0 = self._row_index(r.start)
            i1 = self._row_index(r.last())
            if i0 < 0 or i1 >= self._values.shape[0]:
                msg = f"MITRange {r!s} is not contained in stored range {self.range!s}."
                raise IndexError(msg)
            j = self._col_index(c)
            return TSeries(r.start, self._values[i0 : i1 + 1, j], copy=True)

        # MITRange + collection → MVTSeries.
        if isinstance(r, MITRange) and isinstance(c, (list, tuple)):
            self._check_freq_for(r.frequency, op="indexing")
            i0 = self._row_index(r.start)
            i1 = self._row_index(r.last())
            if i0 < 0 or i1 >= self._values.shape[0]:
                msg = f"MITRange {r!s} is not contained in stored range {self.range!s}."
                raise IndexError(msg)
            block_inds = np.asarray([self._col_index(str(n)) for n in c], dtype=np.intp)
            names = [str(n) for n in c]
            sub = self._values[i0 : i1 + 1][:, block_inds].copy()
            return MVTSeries(r.start, names, sub)

        # MIT-slice form: ``mvts[a:b, c]``.
        if isinstance(r, slice) and (isinstance(r.start, MIT) or isinstance(r.stop, MIT)):
            return self._getitem_two(self._slice_to_range(r), c)

        # Integer pass-throughs.
        if _is_int_like(r) and _is_int_like(c):
            return self._values[int(r), int(c)]
        if _is_int_like(r):
            return self._values[int(r), c]
        if _is_int_like(c):
            return self._values[r, int(c)]
        return self._values[r, c]

    def _slice_with_mit(self, key: slice) -> Any:
        return self._getitem_one(self._slice_to_range(key))

    def _slice_to_range(self, key: slice) -> MITRange:
        if not (isinstance(key.start, MIT) and isinstance(key.stop, MIT)):
            msg = "MIT slice must have MIT endpoints on both sides."
            raise TypeError(msg)
        step = key.step if key.step is not None else 1
        return MITRange(key.start, key.stop, step)

    def _is_name_collection(self, t: tuple[Any, ...]) -> bool:
        if len(t) == 0:
            return False
        # Heuristic: a name collection contains only string-like elements.
        return all(isinstance(x, str) for x in t)

    def _check_columns_exist(self, names: list[str]) -> None:
        unknown = [n for n in names if n not in self._columns]
        if unknown:
            msg = f"MVTSeries has no column(s) {unknown!r}."
            raise KeyError(msg)

    # -- indexing: setitem -------------------------------------------------

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, tuple) and len(key) == 2 and not _is_name_tuple(key):
            self._setitem_two(key[0], key[1], value)
            return
        self._setitem_one(key, value)

    def _setitem_one(self, key: Any, value: Any) -> None:
        if isinstance(key, str):
            self._set_column(key, value)
            return
        if isinstance(key, MIT):
            self._check_freq_for(key.frequency, op="setting")
            i = self._row_index(key)
            if not 0 <= i < self._values.shape[0]:
                msg = f"MIT {key!s} is outside the stored range {self.range!s}."
                raise IndexError(msg)
            self._values[i, :] = value
            return
        if isinstance(key, MITRange):
            self._check_freq_for(key.frequency, op="setting")
            i0 = self._row_index(key.start)
            i1 = self._row_index(key.last())
            if i0 < 0 or i1 >= self._values.shape[0]:
                msg = f"MITRange {key!s} is not contained in stored range {self.range!s}."
                raise IndexError(msg)
            self._assign_row_block(i0, i1, slice(None), value, src_range=key)
            return
        if isinstance(key, (list, tuple)) and self._is_name_collection(tuple(key)):
            names = [str(n) for n in key]
            self._check_columns_exist(names)
            inds = [self._col_index(n) for n in names]
            self._assign_col_block(inds, value, names=names)
            return
        if isinstance(key, np.ndarray) and key.dtype == np.bool_:
            if key.ndim == 1 and key.shape[0] == self._values.shape[0]:
                self._values[key, :] = np.asarray(value)
                return
            self._values[key] = value
            return
        if isinstance(key, list) and all(isinstance(b, (bool, np.bool_)) for b in key):
            self._setitem_one(np.asarray(key, dtype=bool), value)
            return
        if isinstance(key, slice):
            if isinstance(key.start, MIT) or isinstance(key.stop, MIT):
                self._setitem_one(self._slice_to_range(key), value)
                return
            self._values[key] = value
            return
        if _is_int_like(key):
            self._values[int(key)] = value
            return
        if isinstance(key, np.ndarray) and np.issubdtype(key.dtype, np.integer):
            self._values[key] = value
            return
        msg = f"MVTSeries does not support indexing with {type(key).__name__}."
        raise TypeError(msg)

    def _setitem_two(self, r: Any, c: Any, value: Any) -> None:
        # Colon shortcuts.
        if isinstance(r, slice) and r == slice(None) and isinstance(c, slice) and c == slice(None):
            self._values[:, :] = value
            return
        if isinstance(r, slice) and r == slice(None):
            self._setitem_one(c, value)
            return
        if isinstance(c, slice) and c == slice(None):
            self._setitem_one(r, value)
            return

        # Bool row-mask cases.
        if isinstance(r, np.ndarray) and r.dtype == np.bool_ and r.ndim == 1:
            if isinstance(c, str):
                col_idx = self._col_index(c)
                self._values[r, col_idx] = value
                return
            if isinstance(c, (list, tuple)):
                col_inds_set = np.asarray([self._col_index(str(n)) for n in c], dtype=np.intp)
                self._values[np.ix_(r, col_inds_set)] = value
                return
            if isinstance(c, np.ndarray) and c.dtype == np.bool_:
                self._values[np.ix_(r, c)] = value
                return

        # MIT + str.
        if isinstance(r, MIT) and isinstance(c, str):
            self._check_freq_for(r.frequency, op="setting")
            i = self._row_index(r)
            j = self._col_index(c)
            self._values[i, j] = value
            return

        # MIT + collection.
        if isinstance(r, MIT) and isinstance(c, (list, tuple)):
            self._check_freq_for(r.frequency, op="setting")
            i = self._row_index(r)
            row_col_inds = np.asarray([self._col_index(str(n)) for n in c], dtype=np.intp)
            arr = np.asarray(value)
            if arr.ndim == 0:
                msg = (
                    "Cannot assign scalar across multiple columns by tuple; use a list "
                    "with explicit broadcast or `mvts[MIT, name] = scalar`."
                )
                raise ValueError(msg)
            self._values[i, row_col_inds] = arr
            return

        # MITRange + str → assign a TSeries-aligned column slice.
        if isinstance(r, MITRange) and isinstance(c, str):
            self._check_freq_for(r.frequency, op="setting")
            i0 = self._row_index(r.start)
            i1 = self._row_index(r.last())
            if i0 < 0 or i1 >= self._values.shape[0]:
                msg = f"MITRange {r!s} is not contained in stored range {self.range!s}."
                raise IndexError(msg)
            j = self._col_index(c)
            if isinstance(value, TSeries):
                self._check_freq_for(value.frequency, op="setting")
                self._values[i0 : i1 + 1, j] = value[r].values
                return
            self._values[i0 : i1 + 1, j] = np.asarray(value)
            return

        # MITRange + collection.
        if isinstance(r, MITRange) and isinstance(c, (list, tuple)):
            self._check_freq_for(r.frequency, op="setting")
            i0 = self._row_index(r.start)
            i1 = self._row_index(r.last())
            if i0 < 0 or i1 >= self._values.shape[0]:
                msg = f"MITRange {r!s} is not contained in stored range {self.range!s}."
                raise IndexError(msg)
            names = [str(n) for n in c]
            block_col_inds = np.asarray([self._col_index(n) for n in names], dtype=np.intp)
            row_idx = np.arange(i0, i1 + 1, dtype=np.intp)
            if isinstance(value, MVTSeries):
                self._check_freq_for(value.frequency, op="setting")
                arr = np.column_stack(
                    [value[name][r].values for name in names if name in value._columns]
                )
                self._values[np.ix_(row_idx, block_col_inds)] = arr
                return
            arr2 = np.asarray(value)
            if arr2.ndim == 1:
                arr2 = arr2.reshape(-1, 1)
            self._values[np.ix_(row_idx, block_col_inds)] = arr2
            return

        # MIT-slice → range.
        if isinstance(r, slice) and (isinstance(r.start, MIT) or isinstance(r.stop, MIT)):
            self._setitem_two(self._slice_to_range(r), c, value)
            return

        # Integer pass-through.
        if _is_int_like(r) and _is_int_like(c):
            self._values[int(r), int(c)] = value
            return
        if _is_int_like(r):
            self._values[int(r), c] = value
            return
        if _is_int_like(c):
            self._values[r, int(c)] = value
            return
        self._values[r, c] = value

    def _set_column(self, name: str, value: Any) -> None:
        """Implement Julia's ``setproperty!(x, name, val)`` for an existing column."""
        if name not in self._columns:
            msg = (
                f"Cannot create new column {name!r}. Build a new MVTSeries with the "
                f"additional column."
            )
            raise KeyError(msg)
        col_idx = self._col_index(name)
        if isinstance(value, TSeries):
            self._check_freq_for(value.frequency, op="setting")
            # Align by MIT — only the overlap is written; the rest is left alone.
            lo = max(self._firstdate.value, value.firstdate.value)
            hi = min(self.lastdate.value, value.lastdate.value)
            if lo > hi:
                return
            i0 = lo - self._firstdate.value
            i1 = hi - self._firstdate.value
            src_off = lo - value.firstdate.value
            n = hi - lo + 1
            self._values[i0 : i1 + 1, col_idx] = value.values[src_off : src_off + n]
            return
        if isinstance(value, MVTSeries):
            self._check_freq_for(value.frequency, op="setting")
            if name not in value._columns:
                msg = f"Right-hand MVTSeries has no column {name!r}."
                raise KeyError(msg)
            self._set_column(name, value._columns[name])
            return
        if _is_scalar_number(value):
            self._values[:, col_idx] = value
            return
        arr = np.asarray(value)
        if arr.ndim != 1 or arr.shape[0] != self._values.shape[0]:
            msg = (
                f"Column assignment: expected a length-{self._values.shape[0]} vector, "
                f"got shape {arr.shape}."
            )
            raise ValueError(msg)
        self._values[:, col_idx] = arr

    def _assign_row_block(
        self,
        i0: int,
        i1: int,
        col_sel: Any,
        value: Any,
        *,
        src_range: MITRange | None = None,
    ) -> None:
        if isinstance(value, MVTSeries):
            self._check_freq_for(value.frequency, op="setting")
            # Only common columns are written (mirrors Julia
            # ``setindex!(x::MVTSeries, val::MVTSeries, rng)``).
            for name, src_col in value._columns.items():
                if name in self._columns:
                    j = self._col_index(name)
                    rng = src_range if src_range is not None else value.range
                    self._values[i0 : i1 + 1, j] = src_col[rng].values
            return
        if isinstance(value, TSeries):
            self._check_freq_for(value.frequency, op="setting")
            rng = src_range if src_range is not None else value.range
            aligned = value[rng].values
            self._values[i0 : i1 + 1, col_sel] = aligned
            return
        self._values[i0 : i1 + 1, col_sel] = value

    def _assign_col_block(
        self,
        col_inds: list[int],
        value: Any,
        *,
        names: list[str],
    ) -> None:
        del names  # currently only used for diagnostics; left for future use
        if isinstance(value, MVTSeries):
            self._check_freq_for(value.frequency, op="setting")
            rng = self.range
            arr = np.column_stack(
                [value[name][rng].values for name in self.column_names if name in value._columns]
            )
            self._values[:, col_inds[: arr.shape[1]]] = arr
            return
        arr2 = np.asarray(value)
        if arr2.ndim == 1:
            arr2 = arr2.reshape(-1, 1)
        self._values[:, col_inds] = arr2

    # -- NumPy interop -----------------------------------------------------

    def __array__(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool | None = None,
    ) -> np.ndarray:
        """Return the underlying matrix (frequency / column names dropped)."""
        if dtype is None or np.dtype(dtype) == self._values.dtype:
            if copy:
                return self._values.copy()
            return self._values
        if copy is False:
            msg = "Cannot honor copy=False when a dtype conversion is required."
            raise ValueError(msg)
        return self._values.astype(dtype)

    # -- whole-object comparison ------------------------------------------

    def equals(self, other: object) -> bool:
        """Return True iff ``other`` is an MVTSeries with same freq/range/cols/values."""
        if not isinstance(other, MVTSeries):
            return False
        if self.frequency != other.frequency:
            return False
        if self._firstdate != other._firstdate:
            return False
        if tuple(self._columns.keys()) != tuple(other._columns.keys()):
            return False
        return np.array_equal(self._values, other._values, equal_nan=False)

    def allclose(self, other: object, *, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Return True iff ``other`` is a same-shape MVTSeries with close values."""
        if not isinstance(other, MVTSeries):
            return False
        if self.frequency != other.frequency:
            return False
        if self._firstdate != other._firstdate or self.shape != other.shape:
            return False
        if tuple(self._columns.keys()) != tuple(other._columns.keys()):
            return False
        return bool(np.allclose(self._values, other._values, rtol=rtol, atol=atol, equal_nan=True))

    # -- equality / hash --------------------------------------------------

    def __eq__(self, other: object) -> Any:
        # Elementwise comparison (NumPy semantics), matching the Julia
        # AbstractMatrix convention. Use ``.equals()`` for whole-object
        # structural equality. The NumPy ufunc stubs reject our scalar /
        # ndarray operand union — the runtime call is fine, the type
        # ignore narrows for mypy only.
        if isinstance(other, MVTSeries):
            return np.equal(self._values, other._values)
        if isinstance(other, np.ndarray):
            return np.equal(self._values, other)
        if _is_scalar_number(other):
            return np.equal(self._values, other)  # type: ignore[call-overload]
        return NotImplemented

    __hash__: ClassVar[None] = None  # type: ignore[assignment]

    # -- repr --------------------------------------------------------------

    def __repr__(self) -> str:
        return _format_mvtseries(self)

    def __str__(self) -> str:
        return _format_mvtseries(self)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _is_name_tuple(t: tuple[Any, ...]) -> bool:
    """Return True iff every element of ``t`` is a non-empty string.

    Disambiguates ``mvts[("a", "b")]`` (column subset, one-arg) from
    ``mvts[a, b]`` (two-arg indexing) in Python's ``__getitem__``, which
    cannot distinguish them at the syntax level the way Julia can.
    """
    return len(t) >= 1 and all(isinstance(x, str) for x in t)


def _format_mvtseries(m: MVTSeries) -> str:
    nrows, ncols = m.shape
    dtype_part = "" if m._values.dtype == np.float64 else f",{m._values.dtype}"
    type_str = f"MVTSeries{{{prettyprint_frequency(m.frequency)}{dtype_part}}}"
    if ncols == 0:
        vars_str = "no variables"
    else:
        names = list(m._columns.keys())
        vars_str = "variables (" + ",".join(names[:3])
        if ncols > 3:
            vars_str += ",…"
        vars_str += ")"
    # The multiplication sign and ellipsis are intentional UI text matching
    # Julia's repr — RUF001 noqa silences the homoglyph-warning.
    head = f"{nrows}×{ncols} {type_str} with range {m.range} and {vars_str}"  # noqa: RUF001
    if nrows == 0 or ncols == 0:
        return head
    # Light tabular body for non-empty matrices: at most 12 rows, columns
    # truncated to 6. The Julia-style aligned print is a follow-up.
    body_lines: list[str] = []
    names = list(m._columns.keys())
    shown_cols = names[:6]
    pad_name = max((len(n) for n in shown_cols), default=0)
    mit_strs = [str(m._firstdate + i) for i in range(nrows)]
    pad_mit = max((len(s) for s in mit_strs), default=0)
    header_line = " " * (pad_mit + 3) + "  ".join(n.rjust(pad_name) for n in shown_cols)
    body_lines.append(header_line)
    rows_to_show = min(nrows, 12)
    for i in range(rows_to_show):
        cells = "  ".join(f"{m._values[i, m._col_index(n)]!s:>{pad_name}}" for n in shown_cols)
        body_lines.append(f"{mit_strs[i].rjust(pad_mit)} : {cells}")
    if nrows > rows_to_show:
        body_lines.append(" " * (pad_mit + 3) + "  ⋮")
    return head + ":\n" + "\n".join(body_lines)


# Keep imports alive for documentation cross-refs.
_ = deepcopy
