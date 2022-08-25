## Public API

### LDR Factorization

```@docs
LDR
ldr
ldr!
```

### Overloaded Functions

```@docs
Base.size
Base.copyto!
LinearAlgebra.mul!
LinearAlgebra.lmul!
LinearAlgebra.rmul!
LinearAlgebra.det
```

### Exported Functions

```@docs
chain_mul!
chain_lmul!
chain_rmul!
inv!
inv_IpA!
inv_UpV!
inv_invUpV!
sign_det
abs_det
abs_det_ratio
```

## Developer API

```@docs
StableLinearAlgebra.mul_D!
StableLinearAlgebra.lmul_D!
StableLinearAlgebra.rmul_D!
StableLinearAlgebra.ldiv_D!
StableLinearAlgebra.mul_P!
StableLinearAlgebra.inv_P!
```