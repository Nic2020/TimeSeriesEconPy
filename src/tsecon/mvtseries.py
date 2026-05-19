# SPDX-License-Identifier: MIT
"""MVTSeries — multivariate, frequency-tagged time series.

Mirrors ``TimeSeriesEcon.jl``'s ``MVTSeries`` (``mvtseries.jl``): a 2-D
NumPy array whose rows correspond to moments in time and whose columns
correspond to named variables of a single, uniform ``dtype``.

Storage follows the Julia design: a contiguous ``ndarray`` of shape
``(nrows, ncols)`` plus a per-column ``TSeries`` "anchor" whose
``.values`` is a view onto the matrix column. Mutating ``mvts.a[date] = v``
writes back into the parent matrix.

Construction follows the wrap-by-default contract: a compatible
2-D ``ndarray`` passed as ``values`` is **wrapped** (aliased) by default;
``copy=True`` forces an independent allocation. The constructor's
fast-path mirrors xarray's ``DataArray``.

Broadcasting (``__array_ufunc__`` / arithmetic dunders) follows the
Julia rules from ``mvts_broadcast.jl``:

* MVTSeries vs MVTSeries: range intersection and column-name
  intersection (in left-arg column order). Mixed-frequency raises.
* MVTSeries vs TSeries: range intersection, all columns broadcast.
* MVTSeries vs scalar or matching-shape ndarray: unchanged shape.
  Mismatched-shape ndarrays raise.

``rename_columns_inplace`` mirrors ``rename_columns!`` (list / mapping /
function / ``prefix`` / ``suffix`` / ``replace`` keyword forms).
``hcat`` / ``vcat`` concatenate along columns / rows.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from typing import Any, ClassVar, Final, Union

import numpy as np
import numpy.typing as npt

from tsecon.frequencies import Frequency, prettyprint_frequency
from tsecon.linalg import _matmul_strip
from tsecon.mit import MIT
from tsecon.mitrange import MITRange, rangeof_span
from tsecon.tseries import TSeries, typenan

__all__ = ["MVTSeries", "hcat", "rename_columns_inplace", "vcat"]


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

    See the module docstring for the storage model and the wrap-vs-copy
    contract on the constructor.
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
        Use ``copy=True`` for an independent allocation.
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

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: Any,
        out: Any = None,
        **kwargs: Any,
    ) -> Any:
        # Defer out-parameter handling until a real caller needs it.
        if out is not None:
            return NotImplemented

        # np.matmul has shape semantics (n?,k),(k,m?)->(n?,m?) that don't fit
        # the element-wise range-intersection path below. Route to linalg.py
        # to match Julia's linalg.jl behavior (strip labels, return ndarray).
        if ufunc is np.matmul and method == "__call__" and len(inputs) == 2:
            return _matmul_strip(inputs[0], inputs[1])

        if method != "__call__":
            # Reductions / accumulations / outer / etc. unwrap to bare ndarray.
            arrays = tuple(np.asarray(x) if isinstance(x, MVTSeries) else x for x in inputs)
            return getattr(ufunc, method)(*arrays, **kwargs)

        return _mvts_dispatch_ufunc(ufunc, inputs, kwargs)

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
        # Fallback: unwrap MVTSeries → ndarray, call the function, return raw.
        unwrapped_args = tuple(np.asarray(a) if isinstance(a, MVTSeries) else a for a in args)
        unwrapped_kwargs = {
            k: (np.asarray(v) if isinstance(v, MVTSeries) else v) for k, v in kwargs.items()
        }
        return func(*unwrapped_args, **unwrapped_kwargs)

    # -- arithmetic dunders ------------------------------------------------
    # ndarray operands hit __array_ufunc__ via ndarray's __add__/etc. For
    # bare scalars and TSeries, route to the same np.* ufunc to keep one path.

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

    def __neg__(self) -> MVTSeries:
        return _build_like(self, np.negative(self._values))

    def __pos__(self) -> MVTSeries:
        return self.copy()

    def __abs__(self) -> MVTSeries:
        return _build_like(self, np.absolute(self._values))

    # -- in-place compound assignment -------------------------------------
    # ``mvts += rhs`` rewrites the matrix buffer in place when shapes line up;
    # column views (which share that buffer) continue to track the update.

    def __iadd__(self, other: Any) -> MVTSeries:
        return self._inplace(np.add, other)

    def __isub__(self, other: Any) -> MVTSeries:
        return self._inplace(np.subtract, other)

    def __imul__(self, other: Any) -> MVTSeries:
        return self._inplace(np.multiply, other)

    def __itruediv__(self, other: Any) -> MVTSeries:
        return self._inplace(np.true_divide, other)

    def __ipow__(self, other: Any) -> MVTSeries:
        return self._inplace(np.power, other)

    def _inplace(self, ufunc: np.ufunc, other: Any) -> MVTSeries:
        if isinstance(other, MVTSeries):
            self._check_freq_for(other.frequency, op=ufunc.__name__)
            rng, cols = _intersect_axes(self, other)
            n = len(rng)
            if n == 0 or not cols:
                # No rows in common or no shared columns — nothing to update.
                return self
            self_off = rng.start.value - self._firstdate.value
            other_off = rng.start.value - other._firstdate.value
            # Write each shared column directly so we hit the matrix buffer
            # (fancy indexing on a 2-D ndarray returns a copy, which would
            # silently break the in-place semantics).
            for name in cols:
                sj = self._col_index(name)
                oj = other._col_index(name)
                sub_self = self._values[self_off : self_off + n, sj]
                sub_other = other._values[other_off : other_off + n, oj]
                ufunc(sub_self, sub_other, out=sub_self)
            return self
        if isinstance(other, TSeries):
            self._check_freq_for(other.frequency, op=ufunc.__name__)
            lo = max(self._firstdate.value, other.firstdate.value)
            hi = min(self.lastdate.value, other.lastdate.value)
            if lo > hi:
                return self
            self_off = lo - self._firstdate.value
            other_off = lo - other.firstdate.value
            n = hi - lo + 1
            sub_self = self._values[self_off : self_off + n]
            sub_other = other.values[other_off : other_off + n][:, np.newaxis]
            ufunc(sub_self, sub_other, out=sub_self)
            return self
        rhs = np.asarray(other) if not isinstance(other, np.ndarray) else other
        if rhs.ndim == 0 or rhs.shape == self._values.shape:
            ufunc(self._values, rhs, out=self._values)
            return self
        # Row-broadcast (shape (ncols,) or (1, ncols)): NumPy handles this natively.
        if rhs.ndim == 1 and rhs.shape[0] == self._values.shape[1]:
            ufunc(self._values, rhs[np.newaxis, :], out=self._values)
            return self
        if rhs.shape == (1, self._values.shape[1]) or rhs.shape == (self._values.shape[0], 1):
            ufunc(self._values, rhs, out=self._values)
            return self
        msg = (
            f"In-place {ufunc.__name__}: right-hand shape {rhs.shape} is not compatible "
            f"with MVTSeries shape {self._values.shape}."
        )
        raise ValueError(msg)

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


# ---------------------------------------------------------------------------
# Broadcasting helpers
# ---------------------------------------------------------------------------


def _intersect_range(a: MITRange, b: MITRange) -> MITRange:
    """Return the MIT-range intersection of two same-frequency ranges.

    If the two ranges do not overlap, returns an empty range starting at
    ``max(a.start, b.start)``.
    """
    freq = a.frequency
    start_value = max(a.start.value, b.start.value)
    stop_value = min(a.stop.value, b.stop.value)
    if start_value > stop_value:
        return MITRange(MIT(freq, start_value), MIT(freq, start_value - 1))
    return MITRange(MIT(freq, start_value), MIT(freq, stop_value))


def _intersect_axes(a: MVTSeries, b: MVTSeries) -> tuple[MITRange, list[str]]:
    """Return ``(range, columns)`` intersection of two MVTSeries (a's column order)."""
    start_value = max(a._firstdate.value, b._firstdate.value)
    stop_value = min(a.lastdate.value, b.lastdate.value)
    freq = a.frequency
    if start_value > stop_value:
        rng = MITRange(MIT(freq, start_value), MIT(freq, start_value - 1))
    else:
        rng = MITRange(MIT(freq, start_value), MIT(freq, stop_value))
    cols = [n for n in a._columns if n in b._columns]
    return rng, cols


def _column_indices(
    a: MVTSeries,
    b: MVTSeries,
    cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return integer index arrays into ``a._values`` and ``b._values`` for ``cols``."""
    a_keys = list(a._columns.keys())
    b_keys = list(b._columns.keys())
    a_inds = np.asarray([a_keys.index(n) for n in cols], dtype=np.intp)
    b_inds = np.asarray([b_keys.index(n) for n in cols], dtype=np.intp)
    return a_inds, b_inds


def _build_like(
    proto: MVTSeries,
    values: np.ndarray,
    *,
    firstdate: MIT | None = None,
    names: list[str] | None = None,
) -> MVTSeries:
    """Build an MVTSeries with proto's first-date and column names by default."""
    fd = firstdate if firstdate is not None else proto._firstdate
    nm = names if names is not None else list(proto._columns.keys())
    if values.shape[1] != len(nm):
        msg = (
            f"_build_like: values shape {values.shape} does not match number of columns {len(nm)}."
        )
        raise ValueError(msg)
    return MVTSeries(fd, nm, values)


def _mvts_dispatch_ufunc(
    ufunc: np.ufunc,
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Compute the common axes and apply ``ufunc`` to aligned operand slices."""
    mvts_inputs = [x for x in inputs if isinstance(x, MVTSeries)]
    if not mvts_inputs:
        return NotImplemented  # pragma: no cover

    freq = mvts_inputs[0].frequency
    for m in mvts_inputs[1:]:
        if m.frequency != freq:
            raise _mixed_freq_error(mvts_inputs[0], m)
    # TSeries inputs must also share the frequency.
    for x in inputs:
        if isinstance(x, TSeries) and x.frequency != freq:
            raise _mixed_freq_error(mvts_inputs[0], x)

    # 1) Compute the intersected MVTSeries axes (range and column names).
    m0 = mvts_inputs[0]
    rng = m0.range
    names = list(m0._columns.keys())
    for m in mvts_inputs[1:]:
        rng = _intersect_range(rng, m.range)
        names = [n for n in names if n in m._columns]

    # 2) Intersect range with any TSeries operand.
    for x in inputs:
        if isinstance(x, TSeries):
            rng = _intersect_range(rng, x.range)

    n = len(rng)
    ncols = len(names)

    if n == 0 or ncols == 0:
        # Empty intersection — return an empty MVTSeries with right shape and dtype.
        # Probe the ufunc with empty operands to get the output dtype right.
        empty_inputs = []
        for x in inputs:
            if isinstance(x, MVTSeries):
                empty_inputs.append(np.zeros((0, 0), dtype=x._values.dtype))
            elif isinstance(x, TSeries):
                empty_inputs.append(np.zeros((0,), dtype=x.values.dtype))
            elif isinstance(x, np.ndarray):
                empty_inputs.append(x.astype(x.dtype, copy=False))
            else:
                empty_inputs.append(x)
        try:
            probe = ufunc(*empty_inputs, **kwargs)
            out_dtype = probe.dtype if isinstance(probe, np.ndarray) else np.float64
        except Exception:
            out_dtype = np.float64
        empty = np.empty((n, ncols), dtype=out_dtype)
        return MVTSeries(rng.start, names, empty)

    # 3) Materialise each input aligned to (rng, names) and call the ufunc.
    materialized: list[Any] = []
    for x in inputs:
        if isinstance(x, MVTSeries):
            row_off = rng.start.value - x._firstdate.value
            x_keys = list(x._columns.keys())
            col_inds = np.asarray([x_keys.index(c) for c in names], dtype=np.intp)
            materialized.append(x._values[row_off : row_off + n][:, col_inds])
            continue
        if isinstance(x, TSeries):
            off = rng.start.value - x.firstdate.value
            # TSeries operand broadcasts across columns.
            materialized.append(x.values[off : off + n][:, np.newaxis])
            continue
        if isinstance(x, np.ndarray):
            if x.ndim == 0:
                materialized.append(x)
                continue
            # Accept shapes that line up: full-matrix, row-vector (1, ncols), col-vector (n, 1).
            if x.shape == (n, ncols):
                materialized.append(x)
                continue
            if x.ndim == 1 and x.shape[0] == ncols:
                materialized.append(x[np.newaxis, :])
                continue
            if x.shape in ((1, ncols), (n, 1)):
                materialized.append(x)
                continue
            msg = (
                f"Cannot broadcast array of shape {x.shape} against MVTSeries shape ({n}, {ncols})."
            )
            raise ValueError(msg)
        materialized.append(x)

    result = ufunc(*materialized, **kwargs)
    if isinstance(result, tuple):
        return tuple(_maybe_wrap_result(r, rng, names) for r in result)
    return _maybe_wrap_result(result, rng, names)


def _maybe_wrap_result(result: Any, rng: MITRange, names: list[str]) -> Any:
    if not isinstance(result, np.ndarray):
        return result
    if result.ndim == 2 and result.shape == (len(rng), len(names)):
        return MVTSeries(rng.start, names, result)
    return result


# ---------------------------------------------------------------------------
# __array_function__ handlers
# ---------------------------------------------------------------------------


def _np_concatenate(*args: Any, **kwargs: Any) -> Any:
    if "out" in kwargs and kwargs["out"] is not None:
        msg = "np.concatenate with out= is not supported for MVTSeries."
        raise TypeError(msg)
    arrays = args[0]
    if not isinstance(arrays, (list, tuple)):
        msg = "np.concatenate requires a sequence of arrays."
        raise TypeError(msg)
    axis = kwargs.pop("axis", 0)
    if axis == 0:
        return vcat(arrays[0], *arrays[1:])
    if axis == 1:
        # Column concat: delegate to hcat if first is MVTSeries.
        head = arrays[0]
        if isinstance(head, MVTSeries):
            return hcat(*[a for a in arrays if isinstance(a, MVTSeries)])
    return np.concatenate([np.asarray(a) for a in arrays], axis=axis, **kwargs)


def _np_array_equal(*args: Any, **kwargs: Any) -> bool:
    a, b = args[0], args[1]
    a_arr = a._values if isinstance(a, MVTSeries) else a
    b_arr = b._values if isinstance(b, MVTSeries) else b
    return bool(np.array_equal(a_arr, b_arr, **kwargs))


def _np_allclose(*args: Any, **kwargs: Any) -> bool:
    a, b = args[0], args[1]
    a_arr = a._values if isinstance(a, MVTSeries) else a
    b_arr = b._values if isinstance(b, MVTSeries) else b
    return bool(np.allclose(a_arr, b_arr, **kwargs))


_ARRAY_FUNCTION_HANDLERS: Final[dict[Any, Any]] = {
    np.concatenate: _np_concatenate,
    np.array_equal: _np_array_equal,
    np.allclose: _np_allclose,
}


# ---------------------------------------------------------------------------
# rename_columns_inplace
# ---------------------------------------------------------------------------


def rename_columns_inplace(
    mvts: MVTSeries,
    new_names_or_map_or_func: list[str]
    | tuple[str, ...]
    | Mapping[str, str]
    | Callable[[str], str]
    | None = None,
    *,
    replace: tuple[str, str] | list[tuple[str, str]] | None = None,
    prefix: str | None = None,
    suffix: str | None = None,
) -> MVTSeries:
    """Rename the columns of ``mvts`` in place. Mirrors Julia ``rename_columns!``.

    Parameters
    ----------
    mvts
        The MVTSeries to mutate. Returned for chaining.
    new_names_or_map_or_func
        One of:

        * a list / tuple of strings — full replacement, length must match;
        * a mapping ``old -> new`` — partial rename (missing keys keep their name);
        * a callable ``f(old) -> new`` — applied to every column name;
        * ``None`` — use the kwargs ``replace`` / ``prefix`` / ``suffix``.
    replace
        ``(old, new)`` pair, or a list of such pairs, applied as
        ``str.replace`` against each column name in order. Cannot be combined
        with the positional ``new_names`` or mapping forms.
    prefix
        String prepended to each column name (after ``replace``).
    suffix
        String appended to each column name (after ``replace``).

    Returns
    -------
    MVTSeries
        The same ``mvts`` instance, with its column dictionary rebuilt.
    """
    arg = new_names_or_map_or_func
    if arg is None and replace is None and prefix is None and suffix is None:
        msg = (
            "rename_columns_inplace requires one of: new names, mapping, function, "
            "or keyword(s) replace / prefix / suffix."
        )
        raise ValueError(msg)
    if arg is not None and (replace is not None or prefix is not None or suffix is not None):
        msg = "Cannot combine positional rename argument with keyword forms."
        raise ValueError(msg)

    old_names = list(mvts._columns.keys())
    if arg is None:
        rename_fn = _rename_fn_from_kwargs(replace=replace, prefix=prefix, suffix=suffix)
        new_names = [rename_fn(n) for n in old_names]
    elif callable(arg) and not isinstance(arg, Mapping):
        new_names = [arg(n) for n in old_names]
    elif isinstance(arg, Mapping):
        new_names = [arg.get(n, n) for n in old_names]
    else:
        seq = list(arg)
        if len(seq) != len(old_names):
            msg = f"rename_columns_inplace: expected {len(old_names)} new names, got {len(seq)}."
            raise ValueError(msg)
        new_names = [str(n) for n in seq]

    if len(set(new_names)) != len(new_names):
        msg = f"rename_columns_inplace produced duplicate column names: {new_names!r}."
        raise ValueError(msg)
    # Rebuild the column dict, reusing the existing TSeries anchors (so column
    # views — which hold strided slices of mvts._values — keep working).
    new_columns: dict[str, TSeries] = {}
    for new_name, series in zip(new_names, mvts._columns.values(), strict=True):
        new_columns[new_name] = series
    mvts._columns = new_columns
    return mvts


def _rename_fn_from_kwargs(
    *,
    replace: tuple[str, str] | list[tuple[str, str]] | None,
    prefix: str | None,
    suffix: str | None,
) -> Callable[[str], str]:
    """Build a name-transformer closing over ``replace`` / ``prefix`` / ``suffix``."""
    if (
        isinstance(replace, tuple)
        and len(replace) == 2
        and all(isinstance(x, str) for x in replace)
    ):
        replace_pairs: list[tuple[str, str]] = [replace]
    elif isinstance(replace, list):
        replace_pairs = [(str(a), str(b)) for a, b in replace]
    elif replace is None:
        replace_pairs = []
    else:
        msg = "replace must be a (old, new) tuple or a list of such tuples."
        raise TypeError(msg)
    p = "" if prefix is None else str(prefix)
    s = "" if suffix is None else str(suffix)

    def fn(name: str) -> str:
        out = name
        for old, new in replace_pairs:
            out = out.replace(old, new)
        return f"{p}{out}{s}"

    return fn


# ---------------------------------------------------------------------------
# hcat / vcat
# ---------------------------------------------------------------------------


def hcat(*mvts: MVTSeries, **columns: Any) -> MVTSeries:
    """Concatenate MVTSeries by columns, optionally adding more via kwargs.

    All positional MVTSeries must share frequency. The output range is the
    span of every input MVTSeries (gaps filled with the dtype's NaN). The
    output dtype is the common-promoted dtype across inputs and kwargs.

    Column-name collisions raise ``ValueError`` — duplicates are not allowed.
    """
    if not mvts:
        msg = "hcat requires at least one MVTSeries argument."
        raise ValueError(msg)
    freq = mvts[0].frequency
    for m in mvts[1:]:
        if m.frequency != freq:
            raise _mixed_freq_error(mvts[0], m)

    # Promote dtype across MVTSeries values + kwarg values that carry dtypes.
    dtypes: list[np.dtype[Any]] = [m._values.dtype for m in mvts]
    for v in columns.values():
        if isinstance(v, TSeries):
            dtypes.append(v.values.dtype)
        elif isinstance(v, np.ndarray):
            dtypes.append(v.dtype)
        elif _is_scalar_number(v):
            dtypes.append(np.asarray(v).dtype)
    out_dtype = np.result_type(*dtypes) if dtypes else np.dtype(np.float64)

    # Range = span of all MVTSeries + TSeries / MITRange kwarg values.
    span_inputs: list[MITRange] = [m.range for m in mvts if m.range]
    for v in columns.values():
        if isinstance(v, TSeries):
            span_inputs.append(v.range)
        elif isinstance(v, MITRange):
            span_inputs.append(v)
    out_range = rangeof_span(*span_inputs) if span_inputs else mvts[0].range

    # Collect ordered (name, source) pairs.
    out_names: list[str] = []
    out_values = np.full((len(out_range), 0), typenan(out_dtype), dtype=out_dtype)
    seen: set[str] = set()

    def _append(name: str, col_data: np.ndarray) -> None:
        nonlocal out_values
        if name in seen:
            msg = f"hcat: duplicate column name {name!r}."
            raise ValueError(msg)
        seen.add(name)
        out_names.append(name)
        out_values = np.hstack([out_values, col_data.reshape(-1, 1).astype(out_dtype, copy=False)])

    for m in mvts:
        for name, anchor in m._columns.items():
            full = np.full(len(out_range), typenan(out_dtype), dtype=out_dtype)
            row_off = m._firstdate.value - out_range.start.value
            full[row_off : row_off + m._values.shape[0]] = anchor.values
            _append(name, full)

    for name, v in columns.items():
        full = np.full(len(out_range), typenan(out_dtype), dtype=out_dtype)
        if isinstance(v, TSeries):
            if v.frequency != freq:
                raise _mixed_freq_error(freq, v.frequency)
            lo = max(out_range.start.value, v.firstdate.value)
            hi = min(out_range.stop.value, v.lastdate.value)
            if lo <= hi:
                rng_off = lo - out_range.start.value
                src_off = lo - v.firstdate.value
                n = hi - lo + 1
                full[rng_off : rng_off + n] = v.values[src_off : src_off + n]
        elif _is_scalar_number(v):
            full[:] = v
        else:
            arr = np.asarray(v)
            if arr.ndim != 1 or arr.shape[0] != len(out_range):
                msg = (
                    f"hcat: kwarg {name!r} vector length {arr.shape[0]} does not "
                    f"match output range length {len(out_range)}."
                )
                raise ValueError(msg)
            full[:] = arr
        _append(name, full)

    return MVTSeries(out_range.start, out_names, out_values)


def vcat(mvts: MVTSeries, *blocks: _ArrayLike | MVTSeries) -> MVTSeries:
    """Append row blocks (matrices, MVTSeries, or 2-D arrays) below ``mvts``.

    Result keeps ``mvts.firstdate`` and column names; additional rows are
    appended in order. Column count must match.
    """
    pieces: list[np.ndarray] = [mvts._values]
    ncols = mvts._values.shape[1]
    for b in blocks:
        if isinstance(b, MVTSeries):
            arr = b._values
        else:
            arr = np.asarray(b)
            if arr.ndim == 1:
                if ncols != 1:
                    msg = (
                        f"vcat: 1-D block (length {arr.shape[0]}) cannot be appended to "
                        f"MVTSeries with {ncols} columns."
                    )
                    raise ValueError(msg)
                arr = arr.reshape(-1, 1)
        if arr.shape[1] != ncols:
            msg = f"vcat: block has {arr.shape[1]} columns, expected {ncols}."
            raise ValueError(msg)
        pieces.append(arr)
    out = np.vstack(pieces)
    return MVTSeries(mvts._firstdate, list(mvts._columns.keys()), out)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


_DEFAULT_DISPLAY_HEIGHT: Final[int] = 24
_DEFAULT_DISPLAY_WIDTH: Final[int] = 80
_NAME_TRUNCATE_AT: Final[int] = 10


def _format_mvtseries(
    m: MVTSeries,
    *,
    display_size: tuple[int, int] | None = None,
    limit: bool = True,
) -> str:
    """Pretty-print an MVTSeries.

    Mirrors ``mvts_show.jl``: per-column right-alignment based on the widest
    formatted value in that column (or the column-name width, whichever is
    larger); long names are truncated at :data:`_NAME_TRUNCATE_AT` characters
    plus an ellipsis. When the natural total width exceeds the available
    terminal width, columns from the middle are dropped and replaced with
    a ``…`` separator. When the natural row count exceeds the available
    terminal height, rows from the middle are dropped and replaced with
    a ``⋮`` line.

    ``limit=False`` shows every row and column unconditionally.
    """
    nrows, ncols = m.shape
    dtype_part = "" if m._values.dtype == np.float64 else f",{m._values.dtype}"
    type_str = f"MVTSeries{{{prettyprint_frequency(m.frequency)}{dtype_part}}}"
    names_all = list(m._columns.keys())
    vars_str = _summary_vars(names_all)
    head = f"{nrows}×{ncols} {type_str} with range {m.range} and {vars_str}"
    if nrows == 0 or ncols == 0:
        return head

    if display_size is None:
        dheight, dwidth = _DEFAULT_DISPLAY_HEIGHT, _DEFAULT_DISPLAY_WIDTH
    else:
        dheight, dwidth = display_size

    # Truncate long display names (matching the Julia rule).
    disp_names = [
        n if len(n) < _NAME_TRUNCATE_AT else n[:_NAME_TRUNCATE_AT] + "…" for n in names_all
    ]

    # Pre-format every cell and compute per-column display widths.
    cell_strs: list[list[str]] = [
        [_format_cell(m._values[i, j]) for j in range(ncols)] for i in range(nrows)
    ]
    col_widths = [
        max([len(disp_names[j])] + [len(cell_strs[i][j]) for i in range(nrows)])
        for j in range(ncols)
    ]
    mit_strs = [str(m._firstdate + i) for i in range(nrows)]
    mitpad = max(len(s) for s in mit_strs)
    prefix_width = mitpad + 3  # "<mit> : "
    sep = "  "
    natural_width = prefix_width + sum(col_widths) + sep_total(col_widths, sep)

    show_cols: list[int]
    use_split = False
    if limit and natural_width > dwidth and ncols > 1:
        show_cols, left_idx, right_idx = _pick_visible_columns(
            col_widths, sep, prefix_width, dwidth
        )
        if show_cols == list(range(ncols)):
            # Fallback when the split-pass would not actually save space.
            pass
        else:
            use_split = True
    if not use_split:
        show_cols = list(range(ncols))
        left_idx = list(range(ncols))
        right_idx = []

    # Decide visible rows. Mirrors Julia's `mvts_show.jl`:
    # truncate when `nrow > dheight - 6`; in that branch, top-row count is
    # `(dheight - 6) // 2` and bottom-row count is the remainder. This keeps
    # the total line count equal to ``min(nrow + 2, dheight - 3)`` (or 3).
    use_vdots = limit and (nrows > dheight - 6)
    if not use_vdots:
        top_rows = list(range(nrows))
        bot_rows: list[int] = []
    else:
        top_n = max(0, (dheight - 6) // 2)
        bot_n = max(0, (dheight - 6) - top_n)
        top_rows = list(range(top_n))
        bot_rows = list(range(nrows - bot_n, nrows))

    lines: list[str] = [head + ":"]
    lines.append(_render_header(disp_names, col_widths, left_idx, right_idx, prefix_width, sep))
    for i in top_rows:
        lines.append(
            _render_row(mit_strs[i], mitpad, cell_strs[i], col_widths, left_idx, right_idx, sep)
        )
    if use_vdots:
        lines.append(_render_vdots(col_widths, left_idx, right_idx, prefix_width, sep))
    for i in bot_rows:
        lines.append(
            _render_row(mit_strs[i], mitpad, cell_strs[i], col_widths, left_idx, right_idx, sep)
        )
    return "\n".join(lines)


def _summary_vars(names: list[str]) -> str:
    ncols = len(names)
    if ncols == 0:
        return "no variables"
    out = "variables (" + names[0]
    for i in range(1, ncols):
        extra = "," + names[i]
        if len(out) + len(extra) > 20:
            out += ",…"
            break
        out += extra
    return out + ")"


def _format_cell(v: Any) -> str:
    if isinstance(v, (np.floating, float)):
        if np.isnan(v):
            return "NaN"
        if np.isinf(v):
            return "Inf" if v > 0 else "-Inf"
        return format(float(v), ".4g")
    return str(v)


def sep_total(col_widths: list[int], sep: str) -> int:
    return max(0, len(col_widths) - 1) * len(sep)


def _pick_visible_columns(
    col_widths: list[int],
    sep: str,
    prefix_width: int,
    dwidth: int,
) -> tuple[list[int], list[int], list[int]]:
    ncols = len(col_widths)
    hdots = " … "  # 3 chars including spaces; matches Julia's hdots
    sep_len = len(sep)
    # Greedy: take left-most columns until adding the next would exceed half-width,
    # then right-most columns from the end.
    budget = dwidth - prefix_width - len(hdots)
    left: list[int] = []
    right: list[int] = []
    left_w = 0
    for j in range(ncols):
        addition = col_widths[j] + (sep_len if left else 0)
        if left_w + addition > budget // 2 + 1:
            break
        left.append(j)
        left_w += addition
    remaining = budget - left_w
    right_w = 0
    for j in range(ncols - 1, max(len(left) - 1, -1), -1):
        addition = col_widths[j] + (sep_len if right else 0)
        if right_w + addition > remaining:
            break
        right.insert(0, j)
        right_w += addition
    visible = left + right
    if not right and len(left) == ncols:
        return list(range(ncols)), list(range(ncols)), []
    return visible, left, right


def _render_header(
    disp_names: list[str],
    col_widths: list[int],
    left: list[int],
    right: list[int],
    prefix_width: int,
    sep: str,
) -> str:
    left_part = sep.join(disp_names[j].rjust(col_widths[j]) for j in left)
    if not right:
        return " " * prefix_width + left_part
    right_part = sep.join(disp_names[j].rjust(col_widths[j]) for j in right)
    return " " * prefix_width + left_part + " … " + right_part


def _render_row(
    mit_str: str,
    mitpad: int,
    cells: list[str],
    col_widths: list[int],
    left: list[int],
    right: list[int],
    sep: str,
) -> str:
    left_part = sep.join(cells[j].rjust(col_widths[j]) for j in left)
    head = f"{mit_str.rjust(mitpad)} : "
    if not right:
        return head + left_part
    right_part = sep.join(cells[j].rjust(col_widths[j]) for j in right)
    return head + left_part + " … " + right_part


def _render_vdots(
    col_widths: list[int],
    left: list[int],
    right: list[int],
    prefix_width: int,
    sep: str,
) -> str:
    left_part = sep.join("⋮".rjust(col_widths[j]) for j in left)
    if not right:
        return " " * prefix_width + left_part
    right_part = sep.join("⋮".rjust(col_widths[j]) for j in right)
    return " " * prefix_width + left_part + " ⋱ " + right_part


# Keep imports alive for documentation cross-refs.
_ = deepcopy
