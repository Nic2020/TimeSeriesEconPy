# Release notes

## 0.0.1.dev3

This development release adds DataEcon file interchange through idiomatic
Python interfaces, with explicit storage-preserving representations and
conversion methods. It also improves numerical robustness and core-only
installation coverage.

### DataEcon interchange

- Read and write scalars, dated series, multivariate series, plain arrays
  through five dimensions, text arrays and lossless range representations.
- Preserve wide integers, ComplexF16, raw text and reconstruction metadata
  with `StoredScalar`, `StoredSeries`, `StoredMVTSeries`, `StoredArray` and
  `StoredText`. Familiar supported values continue to use Python/NumPy types.
- Interpret verified rational, complex, calendar, abstract and empty-type
  markers explicitly. Unknown markers are preserved without evaluating code.
- Exchange nested `Workspace` objects through catalogs, with structured
  skip reports by default and an explicit strict mode.
- Address nested objects by path, inspect object IDs, list catalogs, manage
  string attributes, and use read-only, append, truncate and in-memory modes.
- Construct Diagonal, Symmetric and Hermitian storage forms explicitly.
- Verify Python/Julia interchange using pinned reference fixtures and
  installed-wheel tests on all nine supported platform/Python combinations.

### Numerical and installation improvements

- Scale correlation inputs by binary exponents before summation and products,
  handling finite subnormal and large-magnitude cases without an overflowing
  standalone scale factor. Unix statistics builds disable multiply/add
  contraction to match the per-operation reference tests.
- Official wheels bundle the compiled numerical kernels, X-13 and DataEcon.
  Core-only builds continue to import and operate without the optional
  DataEcon extension; native operations report its absence explicitly.

### Compatibility limits

Storage preservation and explicit interpretation are separate operations.
Some values can be preserved and rewritten even when conversion to an
ordinary Python value is unavailable. Notable limits include:

- Verified native date/frequency bounds and the documented low-rank tensor
  restrictions; Windows database file paths must be ASCII.
- No execution of Julia expressions or reconstruction of arbitrary
  third-party type semantics, and no general arbitrary-precision floating API.
- Protected reserved attributes; generic in-place reconstruction-marker
  editing/removal and version-metadata modification are not exposed.
- Refusal of lossy arbitrary-start range inputs and new empty structure
  wrappers whose requested element width cannot survive reconstruction.
- Refusal of undefined Float16 truncation, terminal-size-dependent Symbol
  display reconstruction, and the documented character/Unicode formatting
  cases where an exact Python string interpretation is unavailable.

Strict Workspace mode is not a transaction. Overwrites and marker-write
failures can leave partial results; consult the lifecycle/error documentation.
The [DataEcon guide](https://Nic2020.github.io/TimeSeriesEconPy/dataecon/)
contains the detailed compatibility rules and executable examples.

### Install this version

After publication to PyPI:

```bash
python -m pip install --upgrade "TimeSeriesEconPy==0.0.1.dev3"
```

Prebuilt wheels target CPython 3.11-3.13 on Windows x86-64, Linux x86-64
(glibc 2.28+), and macOS arm64 (11+). Other source-build configurations are
not covered by this wheel matrix. Import the package as `tsecon`.
