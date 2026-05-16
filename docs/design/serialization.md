# JSON serialization

!!! info "Work in progress"
    This page is a stub. Paper-voice content arrives in a follow-up writing session.

`tsecon.io.json` provides lossless round-trip of `Frequency` / `MIT` / `MITRange`
/ `TSeries` / `MVTSeries` / `Workspace` to and from JSON.

Each composite carries a `_type` field that names the class so the decoder can
rebuild it. Plain Python scalars (`int`, `float`, `str`, `bool`, `None`) and
homogeneous lists pass through unwrapped — no `_type` overhead when it isn't
needed.

```python exec="true" source="material-block" session="design-ser"
import json

from tsecon import qq, TSeries
from tsecon.io.json import dumps, loads

t = TSeries(qq(2020, 1), [1.0, 2.0, 3.0, 4.0])
encoded = dumps(t)
roundtripped = loads(encoded)

assert roundtripped.equals(t)
print(json.dumps(json.loads(encoded), indent=2))
```

## Why JSON, not pickle?

- **Portable.** Pickle is Python-specific; JSON is consumable by every language
  in the BoC pipeline.
- **Inspectable.** A failing round-trip is debugged by reading the JSON, not by
  running `pickletools`.
- **Versionable.** The `_type` discriminator is forward-compatible: a future
  version can rename a class and ship a migration shim without breaking
  on-disk artefacts.
