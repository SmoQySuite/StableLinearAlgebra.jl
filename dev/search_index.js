var documenterSearchIndex = {"docs":
[{"location":"public_api/#Public-API","page":"Public API","title":"Public API","text":"","category":"section"},{"location":"public_api/#LDR-Factorization","page":"Public API","title":"LDR Factorization","text":"","category":"section"},{"location":"public_api/","page":"Public API","title":"Public API","text":"LDR\nLDRWorkspace\nldr\nldr!\nldrs\nldrs!\nldr_workspace","category":"page"},{"location":"public_api/","page":"Public API","title":"Public API","text":"LDR\nLDRWorkspace\nldr\nldr!\nldrs\nldrs!\nldr_workspace","category":"page"},{"location":"public_api/#StableLinearAlgebra.LDR","page":"Public API","title":"StableLinearAlgebra.LDR","text":"LDR{T<:Number, E<:Real} <: Factorization{T}\n\nRepresents the matrix factorization A = L D R for a square matrix A, where L is a unitary matrix, D is a diagonal matrix of strictly positive real numbers, and R is defined such that det R = 1.\n\nThis factorization is based on a column-pivoted QR decomposition A P = Q R such that\n\nbeginalign*\nL = Q \nD = vert textrmdiag(R) vert \nR = vert textrmdiag(R) vert^-1 R P^T\nendalign*\n\nFields\n\nL::Matrix{T}: The left unitary matrix L in a LDR factorization.\nd::Vector{E}: A vector representing the diagonal matrix D in a LDR facotorization.\nR::Matrix{T}: The right matrix R in a LDR factorization.\n\n\n\n\n\n","category":"type"},{"location":"public_api/#StableLinearAlgebra.LDRWorkspace","page":"Public API","title":"StableLinearAlgebra.LDRWorkspace","text":"LDRWorkspace{T<:Number}\n\nA workspace to avoid dyanmic memory allocations when performing computations with a LDR factorization.\n\nFields\n\nqr_ws::QRWorkspace{T,E}: QRWorkspace for calculating column pivoted QR factorization without dynamic memory allocations.\nlu_ws::LUWorkspace{T}: LUWorkspace for calculating LU factorization without dynamic memory allocations.\nM::Matrix{T}: Temporary storage matrix for avoiding dynamic memory allocations. This matrix is used/modified when a LDR factorization is calculated.\nM′::Matrix{T}: Temporary storage matrix for avoiding dynamic memory allocations.\nM″::Matrix{T}: Temporary storage matrix for avoiding dynamic memory allocations.\nv::Vector{T}: Temporary storage vector for avoiding dynamic memory allocations.\n\n\n\n\n\n","category":"type"},{"location":"public_api/#StableLinearAlgebra.ldr","page":"Public API","title":"StableLinearAlgebra.ldr","text":"ldr(A::AbstractMatrix{T}) where {T}\n\nAllocate an LDR factorization based on A, but does not calculate its LDR factorization, instead initializing the factorization to the identity matrix.\n\n\n\n\n\nldr(A::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}\n\nReturn the LDR factorization of the matrix A.\n\n\n\n\n\nldr(F::LDR{T}, ignore...) where {T}\n\nReturn a copy of the LDR factorization F.\n\n\n\n\n\n","category":"function"},{"location":"public_api/#StableLinearAlgebra.ldr!","page":"Public API","title":"StableLinearAlgebra.ldr!","text":"ldr!(F::LDR{T}, A::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the LDR factorization F for the matrix A.\n\n\n\n\n\nldr!(F::LDR{T}, I::UniformScaling, ignore...) where {T}\n\nSet the LDR factorization equal to the identity matrix.\n\n\n\n\n\nldr!(Fout::LDR{T}, Fin::LDR{T}, ignore...) where {T}\n\nCopy the LDR factorization Fin to Fout.\n\n\n\n\n\nldr!(F::LDR, ws::LDRWorkspace{T}) where {T}\n\nCalculate the LDR factorization for the matrix F.L.\n\n\n\n\n\n","category":"function"},{"location":"public_api/#StableLinearAlgebra.ldrs","page":"Public API","title":"StableLinearAlgebra.ldrs","text":"ldrs(A::AbstractMatrix{T}, N::Int) where {T}\n\nReturn a vector of LDR factorizations of length N, where each one represents the identity matrix of the same size as A.\n\n\n\n\n\nldrs(A::AbstractMatrix{T}, N::Int, ws::LDRWorkspace{T,E}) where {T,E}\n\nReturn a vector of LDR factorizations of length N, where each one represents the matrix A.\n\n\n\n\n\nldrs(A::AbstractArray{T,3}, ws::LDRWorkspace{T,E}) where {T,E}\n\nReturn a vector of LDR factorizations of length size(A, 3), where there is an LDR factorization for each matrix A[:,:,i].\n\n\n\n\n\n","category":"function"},{"location":"public_api/#StableLinearAlgebra.ldrs!","page":"Public API","title":"StableLinearAlgebra.ldrs!","text":"ldrs!(Fs::AbstractVector{LDR{T,E}}, A::AbstractArray{T,3}, ws::LDRWorkspace{T,E}) where {T,E}\n\nCalculate the LDR factorization Fs[i] for the matrix A[:,:,i].\n\n\n\n\n\n","category":"function"},{"location":"public_api/#StableLinearAlgebra.ldr_workspace","page":"Public API","title":"StableLinearAlgebra.ldr_workspace","text":"ldr_workspace(A::AbstractMatrix)\n\nldr_workspace(F::LDR)\n\nReturn a LDRWorkspace that can be used to avoid dynamic memory allocations.\n\n\n\n\n\n","category":"function"},{"location":"public_api/#Overloaded-Functions","page":"Public API","title":"Overloaded Functions","text":"","category":"section"},{"location":"public_api/","page":"Public API","title":"Public API","text":"eltype\nsize\ncopyto!\nadjoint!\nlmul!\nrmul!\nmul!\nldiv!\nrdiv!\nlogabsdet","category":"page"},{"location":"public_api/","page":"Public API","title":"Public API","text":"Base.eltype\nBase.size\nBase.copyto!\nLinearAlgebra.adjoint!\nLinearAlgebra.lmul!\nLinearAlgebra.rmul!\nLinearAlgebra.mul!\nLinearAlgebra.ldiv!\nLinearAlgebra.rdiv!\nLinearAlgebra.logabsdet","category":"page"},{"location":"public_api/#Base.eltype","page":"Public API","title":"Base.eltype","text":"eltype(LDR{T}) where {T}\n\nReturn the matrix element type T of the LDR factorization F.\n\n\n\n\n\n","category":"function"},{"location":"public_api/#Base.size","page":"Public API","title":"Base.size","text":"size(F::LDR, dim...)\n\nReturn the size of the LDR factorization F.\n\n\n\n\n\n","category":"function"},{"location":"public_api/#Base.copyto!","page":"Public API","title":"Base.copyto!","text":"copyto!(U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nCopy the matrix represented by the LDR factorization V into the matrix U.\n\n\n\n\n\ncopyto!(U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T})\n\ncopyto!(U::LDR, I::UniformScaling, ignore...)\n\nCopy the matrix V to the LDR factorization U, calculating the LDR factorization to represent V.\n\n\n\n\n\ncopyto!(U::LDR{T}, V::LDR{T}, ignore...) where {T}\n\nCopy the 'LDR' factorization V to U.\n\n\n\n\n\n","category":"function"},{"location":"public_api/#LinearAlgebra.adjoint!","page":"Public API","title":"LinearAlgebra.adjoint!","text":"adjoint!(Aᵀ::AbstractMatrix{T}, A::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nGiven an LDR factorization A, construct the matrix representing its adjoint A^dagger\n\n\n\n\n\n","category":"function"},{"location":"public_api/#LinearAlgebra.lmul!","page":"Public API","title":"LinearAlgebra.lmul!","text":"lmul!(U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate V = U V where U is a LDR factorization and V is a matrix.\n\n\n\n\n\nlmul!(U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the numerically stable product V = U V where U is a matrix and V is an LDR factorization.\n\nAlgorithm\n\nCalculate V = U V using the procedure\n\nbeginalign*\nV=  UV\n=  oversetL_0D_0R_0overbraceUL_vD_vR_v\n=  oversetL_1overbraceL_0oversetD_1overbraceD_0oversetR_1overbraceR_0R_v\n=  L_1D_1R_1\nendalign*\n\n\n\n\n\nlmul!(U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the numerically stable product V = U V where U and V are both LDR factorizations.\n\nAlgorithm\n\nCalculate V = U V using the procedure\n\nbeginalign*\nV=  UV\n=  L_uD_uoversetMoverbraceR_uL_vD_vR_v\n=  L_uoversetL_0D_0R_0overbraceD_uMD_vR_v\n=  oversetL_1overbraceL_uL_0oversetD_1overbraceD_0oversetR_1overbraceR_0R_v\n=  L_1D_1R_1\nendalign*\n\n\n\n\n\n","category":"function"},{"location":"public_api/#LinearAlgebra.rmul!","page":"Public API","title":"LinearAlgebra.rmul!","text":"rmul!(U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate U = U V where U is a matrix and V is a LDR factorization.\n\n\n\n\n\nrmul!(U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the numerically stable product U = U V where U is a LDR factorization and V is a matrix.\n\nAlgorithm\n\nCalculate U = U V using the procedure\n\nbeginalign*\nU=  UV\n=  L_uoversetL_0D_0R_0overbraceD_uR_uV\n=  oversetL_1overbraceL_uL_0oversetD_1overbraceD_0oversetR_1overbraceR_0\n=  L_1D_1R_1\nendalign*\n\n\n\n\n\nrmul!(U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the numerically stable product U = U V where both U and V are LDR factorizations.\n\nAlgorithm\n\nCalculate U = U V using the procedure\n\nbeginalign*\nU=  UV\n=  L_uD_uoversetMoverbraceR_uL_vD_vR_v\n=  L_uoversetL_0D_0R_0overbraceD_uMD_vR_v\n=  oversetL_1overbraceL_uL_0oversetD_1overbraceD_0oversetR_1overbraceR_0R_v\n=  L_1D_1R_1\nendalign*\n\n\n\n\n\n","category":"function"},{"location":"public_api/#LinearAlgebra.mul!","page":"Public API","title":"LinearAlgebra.mul!","text":"mul!(H::AbstractMatrix{T}, U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the matrix product H = U V, where H and V are matrices and U is a LDR factorization.\n\n\n\n\n\nmul!(H::AbstractMatrix{T}, U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the matrix product H = U V, where H and U are matrices and V is a LDR factorization.\n\n\n\n\n\nmul!(H::LDR{T}, U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the numerically stable product H = U V, where U is matrix, and H and V are both LDR factorization. For the algorithm refer to documentation for lmul!.\n\n\n\n\n\nmul!(H::LDR{T}, U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the numerically stable product H = U V, where V is matrix, and H and U are both LDR factorizations. For the algorithm refer to the documentation for rmul!.\n\n\n\n\n\nmul!(H::LDR{T}, U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the numerically stable matrix product H = U V where H U and V are all LDR factorizations. For the algorithm refer to the documentation for lmul!.\n\n\n\n\n\n","category":"function"},{"location":"public_api/#LinearAlgebra.ldiv!","page":"Public API","title":"LinearAlgebra.ldiv!","text":"ldiv!(U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate V = U^-1 V where V is a matrix, and U is an LDR factorization.\n\n\n\n\n\nldiv!(H::AbstractMatrix{T}, U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate H = U^-1 V where H and V are matrices, and U is an LDR factorization.\n\n\n\n\n\nldiv!(U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the numerically stable product V = U^-1V where both U and V are LDR factorizations. Note that an intermediate LU factorization is required to calucate the matrix inverse R_u^-1 in addition to the intermediate LDR factorization that needs to occur.\n\nAlgorithm\n\nCalculate V = U^-1V using the procedure\n\nbeginalign*\nV=  U^-1V\n=  L_uD_uR_u^-1L_vD_vR_v\n=  R_u^-1D_u^-1oversetMoverbraceL_u^daggerL_vD_vR_v\n=  oversetL_0D_0R_0overbraceR_u^-1D_u^-1MD_vR_v\n=  oversetL_1overbraceL_0oversetD_1overbraceD_0^phantom1oversetR_1overbraceR_0R_v^phantom1\n=  L_1D_1R_1\nendalign*\n\n\n\n\n\nldiv!(H::LDR{T}, U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the numerically stable product H = U^-1 V where H U and V are all LDR factorizations. Note that an intermediate LU factorization is required to calucate the matrix inverse R_u^-1 in addition to the intermediate LDR factorization that needs to occur.\n\n\n\n\n\nldiv!(U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the numerically stable product V = U^-1 V where U is a matrix and V is a LDR factorization. Note that an intermediate LU factorization is required as well to calucate the matrix inverse U^-1 in addition to the intermediate LDR factorization that needs to occur.\n\nAlgorithm\n\nThe numerically stable procdure used to evaluate V = U^-1 V is\n\nbeginalign*\nV=  U^-1V\n=  oversetL_0D_0R_0overbraceU^-1L_vD_vR_v\n=  oversetL_1overbraceL_0oversetD_1overbraceD_0oversetR_1overbraceR_0R_v\n=  L_1D_1R_1\nendalign*\n\n\n\n\n\nldiv!(H::LDR{T}, U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the numerically stable product H = U^-1 V where H and V are LDR factorizations and U is a matrix. Note that an intermediate LU factorization is required to calculate U^-1 in addition to the intermediate LDR factorization that needs to occur.\n\n\n\n\n\n","category":"function"},{"location":"public_api/#LinearAlgebra.rdiv!","page":"Public API","title":"LinearAlgebra.rdiv!","text":"rdiv!(U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the matrix product U = U V^-1 where V is an LDR factorization and U is a matrix. Note that this requires two intermediate LU factorizations to calculate L_v^-1 and R_v^-1.\n\n\n\n\n\nrdiv!(H::AbstractMatrix{T}, U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the matrix product H = U V^-1 where H and U are matrices and V is a LDR factorization. Note that this requires two intermediate LU factorizations to calculate L_v^-1 and R_v^-1.\n\n\n\n\n\nrdiv!(U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the numerically stable product U = U V^-1 where both U and V are LDR factorizations. Note that an intermediate LU factorization is required to calucate the matrix inverse L_v^-1 in addition to the intermediate LDR factorization that needs to occur.\n\nAlgorithm\n\nCalculate U = UV^-1 using the procedure\n\nbeginalign*\nU=  UV^-1\n=  L_uD_uR_uL_vD_vR_v^-1\n=  L_uD_uoversetMoverbraceR_uR_v^-1D_v^-1L_v^dagger\n=  L_uoversetL_0D_0R_0overbraceD_uMD_v^-1L_v^dagger\n=  oversetL_1overbraceL_uL_0^phantom1oversetD_1overbraceD_0^phantom1oversetR_1overbraceR_0L_v^dagger\n=  L_1D_1R_1\nendalign*\n\n\n\n\n\nrdiv!(H::LDR{T}, U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the numerically stable product H = U V^-1 where H U and V are all LDR factorizations. Note that an intermediate LU factorization is required to calucate the matrix inverse L_v^-1 in addition to the intermediate LDR factorization that needs to occur.\n\n\n\n\n\nrdiv!(U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace) where {T}\n\nCalculate the numerically stable product U = U V^-1 where V is a matrix and U is an LDR factorization. Note that an intermediate LU factorization is required as well to calucate the matrix inverse V^-1 in addition to the intermediate LDR factorization that needs to occur.\n\nAlgorithm\n\nThe numerically stable procdure used to evaluate U = U V^-1 is\n\nbeginalign*\nU=  UV^-1\n=  L_uoversetL_0D_0R_0overbraceD_uR_uV^-1\n=  oversetL_1overbraceL_uL_0oversetD_1overbraceD_0oversetR_1overbraceR_0\n=  L_1D_1R_1\nendalign*\n\n\n\n\n\nrdiv!(H::LDR{T}, U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the numerically stable product H = U V^-1 where V is a matrix and H and U is an LDR factorization. Note that an intermediate LU factorization is required as well to calucate the matrix inverse V^-1 in addition to the intermediate LDR factorization that needs to occur.\n\n\n\n\n\n","category":"function"},{"location":"public_api/#LinearAlgebra.logabsdet","page":"Public API","title":"LinearAlgebra.logabsdet","text":"logabsdet(A::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate log(vert det A vert) and textrmsign(det A) for the LDR factorization A\n\n\n\n\n\n","category":"function"},{"location":"public_api/#Exported-Function","page":"Public API","title":"Exported Function","text":"","category":"section"},{"location":"public_api/","page":"Public API","title":"Public API","text":"inv_IpA!\ninv_IpUV!\ninv_UpV!\ninv_invUpV!","category":"page"},{"location":"public_api/","page":"Public API","title":"Public API","text":"inv_IpA!\ninv_IpUV!\ninv_UpV!\ninv_invUpV!","category":"page"},{"location":"public_api/#StableLinearAlgebra.inv_IpA!","page":"Public API","title":"StableLinearAlgebra.inv_IpA!","text":"inv_IpA!(G::AbstractMatrix{T}, A::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the numerically stable inverse G = I + A^-1 where G is a matrix, and A is represented by a LDR factorization. This method also returns log( vert det Gvert ) and textrmsign(det G)\n\nAlgorithm\n\nThe numerically stable inverse G = I + A^-1 is calculated using the procedure\n\nbeginalign*\nG=  I+A^-1\n=  I+L_aD_aR_a^-1\n=  I+L_aD_aminD_amaxR_a^-1\n=  (R_a^-1D_amax^-1+L_aD_amin)D_amaxR_a^-1\n=  R_a^-1D_amax^-1oversetMoverbraceR_a^-1D_amax^-1+L_aD_amin^-1\n=  R_a^-1D_amax^-1M^-1\nendalign*\n\nwhere D_amin = min(D_a 1) and D_amax = max(D_a 1) Intermediate matrix inversions and relevant determinant calculations are performed via LU factorizations with partial pivoting.\n\n\n\n\n\n","category":"function"},{"location":"public_api/#StableLinearAlgebra.inv_IpUV!","page":"Public API","title":"StableLinearAlgebra.inv_IpUV!","text":"inv_IpUV!(G::AbstractMatrix{T}, U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the numerically stable inverse G = I + UV^-1 where G is a matrix and U and V are represented by LDR factorizations. This method also returns log( vert det Gvert ) and textrmsign(det G)\n\nAlgorithm\n\nThe numerically stable inverse G = I + UV^-1 is calculated using the procedure\n\nbeginalign*\nG=  I+UV^-1\n=  I+L_uD_uR_uL_vD_vR_v^-1\n=  R_v^-1oversetMoverbraceL_u^daggerR_v^-1+D_uR_uL_vD_v^-1L_u^dagger\n=  R_v^-1M^-1L_u^dagger\nendalign*\n\nIntermediate matrix inversions and relevant determinant calculations are performed via LU factorizations with partial pivoting.\n\n\n\n\n\n","category":"function"},{"location":"public_api/#StableLinearAlgebra.inv_UpV!","page":"Public API","title":"StableLinearAlgebra.inv_UpV!","text":"inv_UpV!(G::AbstractMatrix{T}, U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the numerically stable inverse G = U+V^-1 where G is a matrix and U and V are represented by LDR factorizations. This method also returns log( vert det Gvert ) and textrmsign(det G)\n\nAlgorithm\n\nThe numerically stable inverse G = U+V^-1 is calculated using the procedure\n\nbeginalign*\nG=  U+V^-1\n=  oversetD_umaxD_uminL_uoverbraceD_uR_u+oversetD_vminD_vmaxL_voverbraceD_vR_v^-1\n=  L_uD_umaxD_uminR_u+L_vD_vminD_vmaxR_v^-1\n=  L_uD_umax(D_uminR_uR_v^-1D_vmax^-1+D_umax^-1L_u^daggerL_vD_vmin)D_vmaxR_v^-1\n=  R_v^-1D_vmax^-1oversetMoverbraceD_uminR_uR_v^-1D_vmax^-1+D_umax^-1L_u^daggerL_vD_vmin^-1D_umax^-1L_u^dagger\n=  R_v^-1D_vmax^-1M^-1D_umax^-1L_u^dagger\nendalign*\n\nwhere\n\nbeginalign*\nD_umin =  min(D_u1)\nD_umax =  max(D_u1)\nD_vmin =  min(D_v1)\nD_vmax =  max(D_v1)\nendalign*\n\nand all intermediate matrix inversions and determinant calculations are performed via LU factorizations with partial pivoting.\n\n\n\n\n\n","category":"function"},{"location":"public_api/#StableLinearAlgebra.inv_invUpV!","page":"Public API","title":"StableLinearAlgebra.inv_invUpV!","text":"inv_invUpV!(G::AbstractMatrix{T}, U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}\n\nCalculate the numerically stable inverse G = U^-1+V^-1 where G is a matrix and U and V are represented by LDR factorizations. This method also returns log( vert det Gvert ) and textrmsign(det G)\n\nAlgorithm\n\nThe numerically stable inverse G = U^-1+V^-1 is calculated using the procedure\n\nbeginalign*\nG=  U^-1+V^-1\n=  oversetD_umaxD_umin(L_uoverbraceD_uR_u)^-1+oversetD_vminD_vmaxL_voverbraceD_vR_v^-1\n=  (L_uD_umaxD_uminR_u)^-1+L_vD_vminD_vmaxR_v^-1\n=  R_u^-1D_umin^-1D_umax^-1L_u^dagger+L_vD_vminD_vmaxR_v^-1\n=  R_u^-1D_umin^-1(D_umax^-1L_u^daggerR_v^-1D_vmax^-1+D_uminR_uL_vD_vmin)D_vmaxR_v^-1\n=  R_v^-1D_vmax^-1oversetMoverbraceD_umax^-1L_u^daggerR_v^-1D_vmax^-1+D_uminR_uL_vD_vmin^-1D_uminR_u\n=  R_v^-1D_vmax^-1M^-1D_uminR_u\nendalign*\n\nwhere\n\nbeginalign*\nD_umin =  min(D_u1)\nD_umax =  max(D_u1)\nD_vmin =  min(D_v1)\nD_vmax =  max(D_v1)\nendalign*\n\nand all intermediate matrix inversions and determinant calculations are performed via LU factorizations with partial pivoting.\n\n\n\n\n\n","category":"function"},{"location":"developer_api/#Developer-API","page":"Developer API","title":"Developer API","text":"","category":"section"},{"location":"developer_api/","page":"Developer API","title":"Developer API","text":"StableLinearAlgebra.det_D\nStableLinearAlgebra.mul_D!\nStableLinearAlgebra.div_D!\nStableLinearAlgebra.lmul_D!\nStableLinearAlgebra.rmul_D!\nStableLinearAlgebra.ldiv_D!\nStableLinearAlgebra.rdiv_D!\nStableLinearAlgebra.mul_P!\nStableLinearAlgebra.inv_P!\nStableLinearAlgebra.perm_sign","category":"page"},{"location":"developer_api/#StableLinearAlgebra.det_D","page":"Developer API","title":"StableLinearAlgebra.det_D","text":"det_D(d::AbstractVector{T}) where {T}\n\nGiven a diagonal matrix D represented by the vector d, return textrmsign(det D) and log(det A)\n\n\n\n\n\n","category":"function"},{"location":"developer_api/#StableLinearAlgebra.mul_D!","page":"Developer API","title":"StableLinearAlgebra.mul_D!","text":"mul_D!(A, d, B)\n\nCalculate the matrix product A = D cdot B where D is a diagonal matrix represented by the vector d.\n\n\n\n\n\nmul_D!(A, B, d)\n\nCalculate the matrix product A = B cdot D where D is a diagonal matrix represented by the vector d.\n\n\n\n\n\n","category":"function"},{"location":"developer_api/#StableLinearAlgebra.div_D!","page":"Developer API","title":"StableLinearAlgebra.div_D!","text":"div_D!(A, d, B)\n\nCalculate the matrix product A = D^-1 cdot B where D is a diagonal matrix represented by the vector d.\n\n\n\n\n\ndiv_D!(A, B, d)\n\nCalculate the matrix product A = B cdot D^-1 where D is a diagonal matrix represented by the vector d.\n\n\n\n\n\n","category":"function"},{"location":"developer_api/#StableLinearAlgebra.lmul_D!","page":"Developer API","title":"StableLinearAlgebra.lmul_D!","text":"lmul_D!(d, M)\n\nIn-place calculation of the matrix product M = D cdot M where D is a diagonal matrix represented by the vector d.\n\n\n\n\n\n","category":"function"},{"location":"developer_api/#StableLinearAlgebra.rmul_D!","page":"Developer API","title":"StableLinearAlgebra.rmul_D!","text":"rmul_D!(M, d)\n\nIn-place calculation of the matrix product M = M cdot D where D is a diagonal matrix represented by the vector d.\n\n\n\n\n\n","category":"function"},{"location":"developer_api/#StableLinearAlgebra.ldiv_D!","page":"Developer API","title":"StableLinearAlgebra.ldiv_D!","text":"ldiv_D!(d, M)\n\nIn-place calculation of the matrix product M = D^-1 cdot M where D is a diagonal matrix represented by the vector d.\n\n\n\n\n\n","category":"function"},{"location":"developer_api/#StableLinearAlgebra.rdiv_D!","page":"Developer API","title":"StableLinearAlgebra.rdiv_D!","text":"rdiv_D!(M, d)\n\nIn-place calculation of the matrix product M = M cdot D^-1 where D is a diagonal matrix represented by the vector d.\n\n\n\n\n\n","category":"function"},{"location":"developer_api/#StableLinearAlgebra.mul_P!","page":"Developer API","title":"StableLinearAlgebra.mul_P!","text":"mul_P!(A, p, B)\n\nEvaluate the matrix product A = P cdot B where P is a permutation matrix represented by the vector of integers p.\n\n\n\n\n\nmul_P!(A, B, p)\n\nEvaluate the matrix product A = B cdot P where P is a permutation matrix represented by the vector of integers p.\n\n\n\n\n\n","category":"function"},{"location":"developer_api/#StableLinearAlgebra.inv_P!","page":"Developer API","title":"StableLinearAlgebra.inv_P!","text":"inv_P!(p⁻¹, p)\n\nCalculate the inverse/transpose P^-1=P^T of a permuation P represented by the vector p, writing the result to p⁻¹.\n\n\n\n\n\n","category":"function"},{"location":"developer_api/#StableLinearAlgebra.perm_sign","page":"Developer API","title":"StableLinearAlgebra.perm_sign","text":"perm_sign(p::AbstractVector{Int})\n\nCalculate the sign/parity of the permutation p, textrmsgn(p) = pm 1\n\n\n\n\n\n","category":"function"},{"location":"developer_api/#LAPACK-LinearAlgebra","page":"Developer API","title":"LAPACK LinearAlgebra","text":"","category":"section"},{"location":"developer_api/","page":"Developer API","title":"Developer API","text":"StableLinearAlgebra.QRWorkspace\nStableLinearAlgebra.LUWorkspace\nStableLinearAlgebra.inv_lu!\nStableLinearAlgebra.ldiv_lu!\nStableLinearAlgebra.det_lu!","category":"page"},{"location":"developer_api/#StableLinearAlgebra.QRWorkspace","page":"Developer API","title":"StableLinearAlgebra.QRWorkspace","text":"QRWorkspace{T<:Number, E<:Real}\n\nAllocated space for calcuating the pivoted QR factorization using the LAPACK routines geqp3! and orgqr! while avoiding dynamic memory allocations.\n\n\n\n\n\n","category":"type"},{"location":"developer_api/#StableLinearAlgebra.LUWorkspace","page":"Developer API","title":"StableLinearAlgebra.LUWorkspace","text":"LUWorkspace{T<:Number, E<:Real}\n\nAllocated space for calcuating the pivoted QR factorization using the LAPACK routine getrf!. Also interfaces with the getri! and getrs! routines for inverting matrices and solving linear systems respectively.\n\n\n\n\n\n","category":"type"},{"location":"developer_api/#StableLinearAlgebra.inv_lu!","page":"Developer API","title":"StableLinearAlgebra.inv_lu!","text":"inv_lu!(A::AbstractMatrix{T}, ws::LUWorkspace{T}) where {T}\n\nCalculate the inverse of the matrix A, overwriting A in-place. Also return log(det A^-1) and textrmsign(det A^-1)\n\n\n\n\n\n","category":"function"},{"location":"developer_api/#StableLinearAlgebra.ldiv_lu!","page":"Developer API","title":"StableLinearAlgebra.ldiv_lu!","text":"ldiv_lu!(A::AbstractMatrix{T}, B::AbstractMatrix{T}, ws::LUWorkspace{T}) where {T}\n\nCalculate B= A^-1 B modifying B in-place. The matrix A is over-written as well. Also return log(det A^-1) and textrmsign(det A^-1)\n\n\n\n\n\n","category":"function"},{"location":"developer_api/#StableLinearAlgebra.det_lu!","page":"Developer API","title":"StableLinearAlgebra.det_lu!","text":"det_lu!(A::AbstractMatrix{T}, ws::LUWorkspace) where {T}\n\nReturn log(det A) and textrmsign(det A) Note that A is left modified by this function.\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = StableLinearAlgebra","category":"page"},{"location":"#StableLinearAlgebra.jl","page":"Home","title":"StableLinearAlgebra.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for StableLinearAlgebra.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This package exports an LDR matrix factorization type for square matrices, along with a corresponding collection of functions for calculating numerically stable matrix products and matrix inverses. The methods exported by the package are essential for implementing a determinant quantum Monte Carlo (DQMC) code for simulating interacting itinerant electrons on a lattice.","category":"page"},{"location":"","page":"Home","title":"Home","text":"A very similar Julia package implementing and exporting many of the same algorithms is StableDQMC.jl.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To install StableLinearAlgebra.jl run following in the Julia REPL:","category":"page"},{"location":"","page":"Home","title":"Home","text":"] add StableLinearAlgebra","category":"page"},{"location":"#References","page":"Home","title":"References","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}
