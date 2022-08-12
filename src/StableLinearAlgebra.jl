module StableLinearAlgebra

using LinearAlgebra
using DocStringExtensions

# wraps QR decomposition methods from LAPACK so as to avoid allocations
include("qr.jl")

# define LDR decomposition
include("LDR.jl")
export LDR, ldr, ldr!, chain_mul!, inv!, inv_IpF!

end