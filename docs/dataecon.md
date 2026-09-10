# DataEcon interchange

`tsecon.dataecon` reads and writes **Float64 scalars and monthly/quarterly float64 TSeries,
including empty series**, through the DataEcon 0.4.0 C library. Other scalar types,
frequencies/dtypes, catalogs, workspaces and general attributes
are not supported yet. Existing JSON I/O is unchanged.

Native DataEcon support is configured in the wheel workflow for CPython 3.11–3.13:
Windows x86-64, Linux x86-64 and macOS arm64. Native wheel builds and Julia
monthly, empty and scalar interchange checks have passed on all three platforms.
Quarterly interchange is included in the configured wheel checks. Successful CI builds
are separate from a published release.
The integration uses a thin
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
Only little-endian hosts are currently supported. Julia reconstruction attributes
are never evaluated. `jtype` is rejected; `jeltype` is accepted only when its
value is exactly `Float64` on an empty supported float series. Custom attributes
are outside this API's interchange contract.

## Quarterly series and fiscal anchors

Quarterly series support all three canonical calendars. `end_month` is the
ending month of the first quarter: 1 (January), 2 (February) or 3 (March,
the default). The anchor remains part of the frequency when data is saved.

```python
from tsecon import MIT, Quarterly

start = MIT.from_yp(Quarterly(end_month=1), 2024, 4)
quarterly = TSeries(start, np.array([1.25, -2.5, 0.0, 4.75], dtype=np.float64))
with open_dataecon("quarterly-example.daec", "a") as db:
    db.write_series("quarterly", quarterly)
with open_dataecon("quarterly-example.daec") as db:
    restored_quarterly = db.read_series("quarterly")
assert restored_quarterly.frequency == Quarterly(end_month=1)
assert restored_quarterly.firstdate == start
assert restored_quarterly.lastdate == MIT.from_yp(Quarterly(end_month=1), 2025, 3)
np.testing.assert_array_equal(restored_quarterly.values, quarterly.values)
```

Quarterly native date codes are `4 * year + period - 1`. The reliable range is
`-131200` through `2147483647`, inclusive; both observation endpoints must fit.
An empty series checks its stored first-date anchor, while its synthetic last
date may lie below that limit. Invalid dates raise `ValueError` before storage.
Integer date codes alone cannot distinguish fiscal anchors, so the adapter
preserves the native frequency codes 65, 66 and 67 explicitly. Other encodings
are rejected. Quarterly negative and zero years do not require conversion to
Python `datetime.date`.

## Empty series

An empty series retains its first-date anchor and owns an empty float64 array:

```python
empty = TSeries(mm(2024, 1), np.empty(0, dtype=np.float64))
with open_dataecon("empty-example.daec", "a") as db:
    db.write_series("empty", empty)
with open_dataecon("empty-example.daec") as db:
    restored_empty = db.read_series("empty")

assert restored_empty.firstdate == mm(2024, 1)
assert restored_empty.lastdate == mm(2023, 12)
assert restored_empty.values.shape == (0,)
```

The last date follows the core empty-range convention: one period before the
first date. It does not identify a stored observation. Empty inputs retain the
same read-only, closed-file, duplicate-name and native date-range checks.

The pinned TimeSeriesEcon.jl 0.7.4 writer stores `jeltype="Float64"` for an empty
series. Its loader reconstructs a plain vector when that marker is present,
although the file retains the dated axis. Python reads that exact marker and
preserves the TSeries. Python writes omit the redundant marker so the pinned
Julia loader also returns a dated Float64 TSeries. Explicitly different markers,
including `Float32`, are rejected. Without a marker, the zero-byte native float
representation defaults to float64, as it does in Julia.

## Float64 scalars

A scalar is a single value without a date axis; it is distinct from a
one-observation TSeries. Use the same file owner:

```python
with open_dataecon("scalar-example.daec", "a") as db:
    db.write_scalar("rate", 1.25)
with open_dataecon("scalar-example.daec") as db:
    rate = db.read_scalar("rate")
assert type(rate) is float
assert rate == 1.25
```

`write_scalar` accepts exact Python `float` and NumPy `float64` values.
Integers (including booleans), Float32, Decimal, complex numbers, scalar
subclasses and arrays are rejected without implicit conversion. Reads return
independent Python floats. NaN, infinities and the sign of zero are preserved;
arbitrary signaling-NaN states or payload bits are not an interchange guarantee.

Scalars share the root namespace with series: existing names are never
overwritten, and wrong-class reads raise `DataEconError`. Scalar `jtype` and
`jeltype` attributes are always rejected, including the literal `Float64`.
The empty-series attribute exception does not apply to scalars. Native scalar
payloads must be exactly eight bytes with no frequency metadata.

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

Use CPython x86-64 and Visual Studio 2022 MSVC Build Tools with a Windows SDK.
Install the project's build requirements (Cython, NumPy, setuptools, hatchling;
`build` for the wheel command). The preferred route builds DataEcon and its bundled
SQLite directly from pinned source using MSVC, matching the Cython toolchain.

Download the [pinned source ZIP](https://codeload.github.com/bankofcanada/DataEcon/zip/1a108688a044380f808bebf64079e32dbb9cd1a4)
to `downloads/dataecon-source.zip`. Its SHA-256 is
`d2b48df3a47d173c43354fe033438253bb46d878ae6c3824c6224fbd4e6a2d71`.
The helper verifies this hash before extracting or compiling. Use a new output
directory for each build; an existing one is never overwritten.

```powershell
python scripts/build_dataecon_windows.py downloads/dataecon-source.zip --output build/dataecon-source
$env:TSECON_DATAECON_ROOT = (Resolve-Path build/dataecon-source).Path
python -m build --wheel --no-isolation
```

The helper compiles the 13 DataEcon C files and bundled SQLite 3.50.2 amalgamation,
links `bin/libdaec.dll`, and generates `lib/daec.lib`. This **import library** is
the linker's description of the DLL's exported functions. It exports the 41 public
functions declared in `daec.h`, copies that header and notices, and writes
`build-info.json` with source/output hashes, compiler/SDK details and build flags.
That manifest is included with the DLL in the configured wheel.

One guarded portability adjustment is applied only to the extracted source:
`static const uint32_t EPOCH_s = 82;` becomes `#define EPOCH_s UINT32_C(82)`.
MSVC requires a constant expression in the dependent file-scope initializers;
the unsigned value, formulas and public ABI are unchanged. The compiler uses
C11 mode and its standard `/MD` shared C runtime. The native DLL therefore uses
the Microsoft C runtime, as does the Cython extension; dependency auditing and
installed-wheel checks remain required for each supported platform target.

The build steps are repeatable from pinned inputs and record the selected toolchain.
This is not yet a bit-for-bit reproducible build: linker timestamps are not
normalized, and repeated builds can have different binary hashes even with the
same compiler/SDK. Each manifest identifies its actual output binaries.
Upstream narrowing-conversion warnings are left visible; the adapter's existing
date/payload guards and limited supported scope still apply. No native source
algorithm, file-format, type support or Windows path policy is expanded here.

### Alternative: verified upstream binary

For comparison or a local fallback, obtain the
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
Source distributions include both native build helpers and the adapter sources;
the external native inputs/build tools are still required to enable the feature.

Install the resulting `.whl` in a fresh environment and test outside the checkout.
For a wheel containing this native support, users need neither a compiler nor a
separate DataEcon/Julia installation. A **wheel** is the installable built package;
GitHub Actions runs build/test jobs, and publishing is a separate operation.
Windows source builds are verified locally and in wheel CI. The Linux/macOS
source-build configuration below is also verified in platform CI.

The bundled native library uses the BSD-3-Clause DataEcon license and public-domain
SQLite; notices ship beside the adapter. Cython code uses the package's MIT license.

## Linux and macOS source builds

`scripts/build_dataecon_unix.py` uses the same verified source ZIP/header and
guarded constant-expression patch. It compiles all 13 DataEcon sources and bundled
SQLite into a position-independent **static archive**, `lib/libdaec.a`. The linker
incorporates that code into the Cython extension; no separate DataEcon shared
library or runtime search path is needed. Native symbols have hidden visibility
to isolate this SQLite copy from other extensions in the process. Windows retains
its verified DLL/import-library route; both routes use the same Cython binding.

```sh
# macOS arm64 only: export MACOSX_DEPLOYMENT_TARGET=11.0
python scripts/build_dataecon_unix.py downloads/dataecon-source.zip --output build/dataecon-native
export TSECON_DATAECON_ROOT="$PWD/build/dataecon-native"
python -m build --wheel
```

The native helper needs a C compiler and `ar`. Its manifest records the compiler,
flags, source/header hashes, portability patch and static archive hash. The build
hook checks the header/archive against that manifest before linking. The archive
hash identifies a build input, not the final extension: wheel repair can modify
the extension. The archive itself does not ship in the wheel; the manifest and
license notices do. Updating native code requires rebuilding the extension.

Linux CI builds inside the explicitly selected manylinux_2_28 x86-64 container
(glibc 2.28 floor), then uses cibuildwheel's auditwheel repair. A local Linux wheel
build alone does not establish manylinux compatibility. macOS uses the explicit
`macos-15` Apple Silicon runner, targets macOS 11.0, and retains delocate repair
and deployment-target validation. Intel macOS and universal2 are outside this
matrix. See [cibuildwheel options](https://cibuildwheel.pypa.io/en/v3.4.1/options/)
and [manylinux platforms](https://github.com/pypa/manylinux).

## Wheel CI checks

Each Windows wheel job builds the pinned native source with MSVC and includes
the DLL, build manifest and notices. `scripts/check_dataecon_wheel.py` runs after
installation: it rejects imports from the source checkout, verifies the pinned
manifest and DLL hash, checks the DLL's actual loaded location, reads the Julia
fixture and writes a result for separate Julia verification. The build log and
manifest record the native runtime dependency names.

The workflow sets `TSECON_REQUIRE_DATAECON=1`, so an absent extension is a pytest
configuration error rather than an optional skip. The ordinary core-only test
jobs leave this setting unset. The checker can also verify a core-only wheel when
the setting is unset. Linux/macOS installed checks inspect the repaired extension
with `readelf`/`otool` and `nm`: external DataEcon/SQLite dependencies or exposed
native symbols fail the job. Existing ABI-layout, owning-buffer, lifecycle and
error tests run against each installed native wheel.

After the installed tests pass, each platform job checks its output with Julia
1.12.5 and the pinned TimeSeriesEcon.jl checkout. Julia and DataEcon_jll are CI
verification dependencies, not dependencies of the Python wheel. The Julia setup
helper pins DataEcon_jll to 0.4.0+0; the interchange script verifies the reference
checkout identity and nonempty/empty value, type and metadata assertions before
wheel upload. Each output contains the nonempty sample and empty series anchored
at 2024M1 and 2025M7; the Julia loader must preserve all three as dated TSeries.
The same file contains seven Float64 scalars covering finite values, signed zero,
NaN and infinities; Julia checks their types, metadata and values as well.
Quarterly objects cover all three fiscal anchors, year transitions, negative and
zero years, and nonempty/empty anchors at the supported date limits. Their
frequency and first/last dates must also survive the Julia read.

For a local installed-wheel check, use a fresh output directory and the installed
environment's Python from outside the source package:

```powershell
$env:TSECON_REQUIRE_DATAECON = '1'
python scripts/check_dataecon_wheel.py --fixture tests/dataecon/fixtures/julia_monthly.daec --output-dir build/interchange-output
```

Use an absolute path to the script/fixture when running outside the checkout.
Without `--output-dir`, the checker uses `TSECON_DATAECON_OUTPUT_DIR` when set,
otherwise `build/dataecon-interchange` beside the project scripts directory.
Linux CI sets that variable to the host workspace through cibuildwheel's `/host`
mount. Its `/project` directory is a container copy, so files written there are
not automatically available to the later Julia step on the host.

The output is named for the CPython version tag (for example `cp311.daec`), and an existing
output is not overwritten. CI builds and artifact checks do not publish to PyPI;
publication remains a separate release workflow.
