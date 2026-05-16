# Indexing

`tsecon.lookup(t, keys)` is the vectorised "gather a list of MITs (or integer offsets)
from a TSeries" entry point. Routes through a Cython gather kernel when the compiled
extension is importable; introspect with `lookup_is_cython()`. See
[design/cython_strategy.md](../design/cython_strategy.md) for why the kernel speedup
is small here (~1.1× over NumPy) — `np.take` already runs in C.

::: tsecon.indexing
