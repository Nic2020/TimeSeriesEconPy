# Workspace

Ordered, heterogeneously-typed container for time series and scalars. Built on
Python's insertion-ordered `dict` (no `OrderedDict` needed since CPython 3.7+),
with dotted attribute access (`w.gdp`) sugar over `__getitem__` / `__setitem__`.

::: tsecon.workspace
