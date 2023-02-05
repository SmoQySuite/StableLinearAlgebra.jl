module StableLinearAlgebra

using LinearAlgebra
import LinearAlgebra: adjoint!, lmul!, rmul!, mul!, ldiv!, rdiv!, logabsdet
import Base: eltype, size, copyto!

# define developer functions/methods
include("developer_functions.jl")

# imports for wrapping LAPACK functions to avoid dynamic memory allocations
using Base: require_one_based_indexing
using LinearAlgebra: checksquare
using LinearAlgebra: BlasInt, BlasFloat
using LinearAlgebra.BLAS: @blasfunc
using LinearAlgebra.LAPACK: chklapackerror, chkstride1, chktrans
using LinearAlgebra.LAPACK
const liblapack = "libblastrampoline"
include("qr.jl") # wrap LAPACK column-pivoted QR factorization
export QRWorkspace
include("lu.jl") # wrap LAPACK LU factorization (for determinants and matrix inversion)
export LUWorkspace, inv_lu!, det_lu!, ldiv_lu!

# define LDR factorization
include("LDR.jl")
export LDR, LDRWorkspace, ldr, ldr!, ldrs, ldrs!, ldr_workspace

# define overloaded functions/methods
include("overloaded_functions.jl")

# define exported functions/methods
include("exported_functions.jl")
export inv_IpA!, inv_IpUV!, inv_UpV!, inv_invUpV!

end