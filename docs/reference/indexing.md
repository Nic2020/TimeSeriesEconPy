# Indexing

## Date ranges

`TSeries` and `MVTSeries` select the dates enumerated by an inclusive `MITRange`,
including positive strides and reverse order. A forward unit-step range returns
an independent labeled series; other steps return an independent NumPy array.
Table column selections follow the same rule.

```python
from tsecon import MITRange, TSeries, qq

s = TSeries(qq(2020, 1), [10., 20., 30., 40.])
backward = MITRange(qq(2020, 4), qq(2020, 1), -1)
assert s[backward].tolist() == [40., 30., 20., 10.]
s[backward] = [4., 3., 2., 1.]
assert s.values.tolist() == [1., 2., 3., 4.]
```

Empty ranges select no observations and assignments do not extend storage.
Frequency checks still apply. Nonempty reads require every selected date to be
stored. An owning `TSeries` can grow for date-range assignment; table range
assignment requires existing rows. Labeled assignment sources are selected by
date, while array values follow the requested order.

This corrects earlier reverse selections that omitted observations and table
selections that ignored the step. Code relying on those incorrect results should
use the actual selected dates; strided table results now have no misleading
contiguous date labels. Range assignment validates shape and casting before
changing the destination.

## NumPy array operations

Direct calls such as `np.mean(series)` and `np.copy(series)` retain their raw
NumPy results. Registered operations such as `np.concatenate([series, series])`
retain their existing series behavior.

Unsupported nested operations such as `np.stack([series, series])` now raise
`TypeError` instead of recursing. Convert explicitly when you want raw arrays:

```python
import numpy as np
from tsecon import TSeries, qq

s = TSeries(qq(2020, 1), [1., 2., 3.])
stacked = np.stack([np.asarray(s), np.asarray(s)])
assert stacked.shape == (2, 3)
```

Fallback delegation accepts direct series and plain numerical arrays, scalars
and built-in argument containers. It declines residual nested series, cyclic
containers, object arrays and unknown objects or iterators. Foreign array
overrides retain dispatch control. Non-`None` keyword `out=` is unsupported.

## Gather

`tsecon.lookup(t, keys)` is the vectorised "gather a list of MITs (or integer offsets)
from a TSeries" entry point. Routes through a Cython gather kernel when the compiled
extension is importable; introspect with `lookup_is_cython()`. See
[design/cython_strategy.md](../design/cython_strategy.md) for why the kernel speedup
is small here (~1.1× over NumPy) — `np.take` already runs in C.

::: tsecon.indexing
