# StableLinearAlgebra.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://SmoQySuite.github.io/StableLinearAlgebra.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://SmoQySuite.github.io/StableLinearAlgebra.jl/dev)
[![Build Status](https://github.com/SmoQySuite/StableLinearAlgebra.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/SmoQySuite/StableLinearAlgebra.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/SmoQySuite/StableLinearAlgebra.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/SmoQySuite/StableLinearAlgebra.jl)

This package exports an LDR matrix factorization type for square matrices, along with a corresponding collection of functions for calculating numerically stable matrix products and matrix inverses. The methods exported by the package are essential
for implementing a determinant quantum Monte Carlo (DQMC) code for simulating interacting itinerant electrons on a lattice.

A very similar Julia package implementing and exporting many of the same algorithms is [`StableDQMC.jl`](https://github.com/carstenbauer/StableDQMC.jl).

## Installation
To install [`StableLinearAlgebra.jl`](https://github.com/SmoQySuite/StableLinearAlgebra.jl) run following in the Julia REPL:

```julia
julia> ]
pkg> add StableLinearAlgebra
```