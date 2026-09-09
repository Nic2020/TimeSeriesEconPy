# DataEcon: first supported series slice

`tsecon.dataecon` reads and writes **nonempty monthly float64 TSeries** through
the DataEcon 0.4.0 C library. This is a limited integration: scalars, other
frequencies/dtypes, empty series, catalogs, workspaces and general attributes
are not supported yet. Existing JSON I/O is unchanged.

Native DataEcon support currently requires a configured local Windows x86-64
build. Ordinary CI wheels do not enable it yet. The integration uses a thin
Cython extension; CFFI and Julia are not runtime dependencies. Only the native
parts are compiled: the Python file API and conversion code remain Python.

```python
import numpy as np
from tsecon import TSeries, mm
from tsecon.dataecon import open_dataecon

series = TSeries(mm(2024, 1), np.array([1.25, -2.5, 0.0, 4.75], dtype=np.float64))
# Use a new file/name: "a" appends without overwriting existing objects.
with open_dataecon("example.daec", "a") as db:
    db.write_series("sample", series)

with open_dataecon("example.daec") as db:  # read-only by default
    restored = db.read_series("sample")

assert restored.firstdate == mm(2024, 1)
assert restored.lastdate == mm(2024, 4)
np.testing.assert_array_equal(restored.values, series.values)
```

Reads return owning TSeries arrays that remain usable after closing the file.
Writes copy the input, including strided arrays and column views. NaNs are
preserved as floating-point missing values. Names must be nonempty root names
without `/` or NUL; Unicode object names are supported. **Windows file paths must
be ASCII**: the pinned native library's narrow `fopen` existence check does not
reliably recognize UTF-8 paths when reopening files. The adapter rejects such
paths before opening or creating anything.

The payload limit is 128 MiB per series. Dates must fit the native encoding;
extreme boundary years can also be rejected before native date arithmetic.
Only little-endian hosts are currently supported. Julia `jtype`/`jeltype`
reconstruction attributes are rejected, never evaluated. Custom attributes are
outside this API's interchange contract.

## Closing and errors

Use a context manager or call `close()` explicitly. Successful close is
idempotent. Missing objects and duplicate names raise `DataEconError`, which
retains `code`, `operation`, `path`, `name` and `native_message`. Positive native
codes come from SQLite. Invalid or unsupported Python inputs raise
`TypeError`/`ValueError`; writes through an explicitly read-only owner are rejected
before calling C. No truncation or replacement mode is provided.

A native write failure may leave a partial object or unused axis; the adapter
does not promise rollback or delete the file. A failed native close makes the
owner unusable and is not retried: upstream statement-finalization retry safety
is not guaranteed. Native resources may remain until process exit in that case.
If a context body also raised, that original exception is preserved and the
cleanup failure is attached as an exception note. Garbage-collection cleanup is
best effort, not a substitute for explicit close.

All native calls through this extension share a lock, including error capture
and copying borrowed buffers. This does not synchronize a separate binding or
Julia runtime embedded in the same process. Interchange verification runs Julia
as a separate process.

## Local Windows build

Use CPython x86-64 and MSVC Build Tools. Install the project's build requirements
(Cython, NumPy, setuptools, hatchling; `build` for the command below). Obtain the
**DataEcon_jll 0.4.0+0 Windows x86-64 artifact** from its
[release](https://github.com/JuliaBinaryWrappers/DataEcon_jll.jl/releases/tag/DataEcon-v0.4.0%2B0).
The archive `DataEcon.v0.4.0.x86_64-w64-mingw32.tar.gz` has SHA-256
`c4d965008d46ab204fa35bbaee3238f3a15393850f0993e1d3cda52dba9af465`.
Verify it and extract it to a local directory. An already installed matching
Julia artifact can also supply these build inputs.

```powershell
python scripts/prepare_dataecon_windows.py <extracted-artifact-directory>
$env:TSECON_DATAECON_ROOT = (Resolve-Path build/dataecon).Path
python -m build --wheel --no-isolation
```

The preparation script checks the header and DLL hashes and generates an MSVC
**import library**, the linker's description of the DLL's exported functions.
The build hook compiles the extension against the actual header and bundles the
DLL in the wheel. It does not download anything during build or runtime. The
default build without `TSECON_DATAECON_ROOT` omits DataEcon native support.
Source distributions include this preparation script and the adapter sources;
the external native inputs/build tools are still required to enable the feature.

Install the resulting `.whl` in a fresh environment and test outside the checkout.
For a wheel containing this native support, users need neither a compiler nor a
separate DataEcon/Julia installation. A **wheel** is the installable built package;
GitHub Actions runs build/test jobs, and publishing is a separate operation.
All-platform native source builds and bundled CI wheels remain future work.

The bundled native library uses the BSD-3-Clause DataEcon license and public-domain
SQLite; notices ship beside the adapter. Cython code uses the package's MIT license.
