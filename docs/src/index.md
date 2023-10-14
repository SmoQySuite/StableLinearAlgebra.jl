```@meta
CurrentModule = StableLinearAlgebra
```

# StableLinearAlgebra.jl

Documentation for [StableLinearAlgebra.jl](https://github.com/SmoQySuite/StableLinearAlgebra.jl).

This package exports an [`LDR`](@ref) matrix factorization type for square matrices, along with a corresponding collection of functions for calculating numerically stable matrix products and matrix inverses. The methods exported by the package are essential
for implementing a determinant quantum Monte Carlo (DQMC) code for simulating interacting itinerant electrons on a lattice.

A very similar Julia package implementing and exporting many of the same algorithms is [`StableDQMC.jl`](https://github.com/carstenbauer/StableDQMC.jl).

## Installation
To install [`StableLinearAlgebra.jl`](https://github.com/SmoQySuite/StableLinearAlgebra.jl) run following in the Julia REPL:

```julia
julia> ]
pkg> add StableLinearAlgebra
```

## References

```@bibliography
*
```