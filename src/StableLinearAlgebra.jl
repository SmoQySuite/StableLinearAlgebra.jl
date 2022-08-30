module StableLinearAlgebra

using LinearAlgebra
using FastLapackInterface

# define LDR decomposition
include("LDR.jl")

# define developer functions/methods
include("developer_functions.jl")

# define overloaded functions/methods
import Base: size, copyto!
import LinearAlgebra: mul!, lmul!, rmul!, ldiv!, det
include("overloaded_functions.jl")

# define exported functions/methods
include("exported_functions.jl")
export LDR, ldr, ldr!, ldrs, ldrs!
export inv!, inv_IpA!, inv_UpV!, inv_invUpV!
export abs_det, sign_det, abs_det_ratio

end