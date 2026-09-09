# SPDX-License-Identifier: MIT
# julia --project=<isolated-project> setup_dataecon_julia.jl <reference-checkout>
using Pkg

length(ARGS) == 1 || error("Supply the pinned TimeSeriesEcon checkout path.")
Pkg.develop(path=abspath(ARGS[1]))
Pkg.add(PackageSpec(name="DataEcon_jll", version=v"0.4.0+0"))
Pkg.pin(PackageSpec(name="DataEcon_jll"))
Pkg.instantiate()
