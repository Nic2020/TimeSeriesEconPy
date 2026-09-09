# Saved compatibility fixtures

These small, trusted JSON and protocol-4 pickle files were generated locally from
the source commit and Python/NumPy versions in `manifest.json`. They contain
synthetic values only. Tests load the committed bytes; they must not regenerate
them with the implementation being tested. SHA-256 hashes identify the originals.

The objects cover a February fiscal-quarter anchor, a duration, a strided date
range, floating/integer/Boolean/empty series, a two-column table and a nested
workspace. Tests check metadata and values independently, including table column
write-through after loading. This is a bounded compatibility sample, not a promise
to load arbitrary Python pickles or external resource handles.

`exports.json` records explicit `__all__` lists from public module paths. New
exports are allowed; removal of a recorded entry requires a compatibility decision.
Private modules are excluded even when they declare their own `__all__`.
