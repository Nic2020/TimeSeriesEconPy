# JSON serialization

Lossless round-trip of `TSeries` / `MVTSeries` / `Workspace` / `MIT` / `MITRange` /
`Frequency` to and from JSON. Each composite type carries a `_type` discriminator
so the decoder can rebuild the right class.

::: tsecon.io.json
