import LinearAlgebra: mul!, lmul!, rmul!, adjoint!
import Base: size, copyto!

"""
    LDR{T} <: Factorization{T}

Represents the matrix factorization ``A P = L D R`` for a square matrix ``A``, which may equivalently be
written as ``A = (L D R) P^{-1} = (L D R) P^T``.
In the above ``L`` is a unitary matrix, ``D`` is a diagonal matrix, ``R`` is an upper triangular matrix.
Lastly, ``P`` is a permutation matrix, for which ``P^{-1}=P^T``.
This factorization is based on a column-pivoted QR decomposition ``A P = Q R`` such that
``L = Q``, ``D = Diag(R)``, ``R = Diag(R)^{-1} R`` and ``P = P``. 

# Fields
$(TYPEDFIELDS)
"""
struct LDR{T<:Number, E<:Real} <: Factorization{T}

    "The left unitary matrix ``L``."
    L::Matrix{T}

    "Vector representing diagonal matrix ``D``."
    d::Vector{E}

    "The right upper triangular matrix ``R``."
    R::Matrix{T}

    "Permutation vector to represent permuation matrix ``P^T``."
    pᵀ::Vector{Int}

    "Stores the elementary reflectors for calculatng ``AP = QR`` decomposition."
    τ::Vector{T}

    "Workspace for calculating QR decomposition using LAPACK without allocations."
    ws::QRWorkspace{T, E}

    "A matrix for temporarily storing intermediate results so as to avoid dynamic memory allocations."
    M_tmp::Matrix{T}

    "A vector for temporarily storing intermediate results so as to avoid dynamic memory allocations."
    p_tmp::Vector{Int}
end

function ldr(A::AbstractMatrix{T})::LDR{T} where {T}

    @assert size(A,1) == size(A,2)

    # matrix dimension
    n = size(A,1)

    # allocate relevant arrays
    L =  zeros(T,n,n)
    R  = zeros(T,n,n)
    if T <: Complex
        E = T.types[1]
        d = zeros(E,n)
    else
        d = zeros(T,n)
    end

    # allocate workspace for QR decomposition
    copyto!(L,A)
    ws = QRWorkspace(L)
    pᵀ = ws.jpvt
    τ  = ws.τ

    # allocate array for storing intermediate results to avoid
    # dynamic memory allocations
    M_tmp = zeros(T,n,n)
    p_tmp = zeros(Int,n)

    # instantiate LDR decomposition
    F = LDR(L,d,R,pᵀ,τ,ws,M_tmp,p_tmp)

    # calculate LDR decomposition
    ldr!(F, A)
    
    return F
end

function ldr!(F::LDR{T}, A::AbstractMatrix{T}) where {T}

    @assert size(F) == size(A)

    copyto!(F.L, A)
    ldr!(F)

    return nothing
end

# update/calculate the LDR decomposition F using the matrix F.L
function ldr!(F::LDR{T}) where {T}

    (; L, d, R, ws) = F

    # calclate QR decomposition
    geqp3!(L, ws)

    # extract upper triangular matrix R
    R′ = UpperTriangular(L)
    copyto!(R, R′)

    # set D = Diag(R), represented by vector d
    @inbounds for i in 1:size(L,1)
        d[i] = abs(R[i,i])
    end

    # calculate R = D⁻¹⋅R
    D = Diagonal(d)
    ldiv!(D, R)

    # construct L (same as Q) matrix
    orgqr!(L, ws)

    return nothing
end

size(F::LDR)       = size(F.L)
size(F::LDR, dims) = size(F.L, dims)


function copyto!(A::AbstractMatrix{T}, F::LDR{T}) where {T}

    @assert size(A) == size(F)

    (; L, pᵀ) = F
    D = Diagonal(F.d)
    R = UpperTriangular(F.R)
    A′ = F.M_tmp

    # A = L⋅D⋅R⋅Pᵀ
    copyto!(A′, L) # A = L
    rmul!(A′, D) # A = L⋅D
    rmul!(A′, R) # A = (L⋅D)⋅R
    mul_P!(A, pᵀ, A′) # A = (L⋅D⋅R)⋅Pᵀ

    return nothing
end


function chain_mul!(F::LDR{T}, B::AbstractArray{T,3}) where {T}

    @assert size(B,1) == size(B,2) == size(F,1)
    L  = size(B, 3)
    B₁ = @view B[:,:,1]
    ldr!(F, B₁) # construct LDR decomposition for first matrix in chain
    for l in 2:L
        Bₗ = @view B[:,:,l]
        lmul!(Bₗ, F) # stabalized multiplication by next matrix in chain
    end

    return nothing
end


function inv!(A::AbstractMatrix{T}, F::LDR{T}) where {T}

    # Calculate LDR decomposition A = L⋅D⋅R⋅Pᵀ
    ldr!(F, A)

    # calculate inverse A⁻¹ = P⋅R⁻¹⋅D⁻¹⋅Lᵀ
    A⁻¹ = A # renaming
    L   = F.L
    D   = Diagonal(F.d)
    R   = UpperTriangular(F.R)
    M   = F.M_tmp
    p   = F.p_tmp
    inv_P!(p, F.pᵀ)
    adjoint!(A⁻¹, F.L) # A⁻¹ = Lᵀ
    ldiv!(D, A⁻¹) # A⁻¹ = D⁻¹⋅Lᵀ
    ldiv!(R, A⁻¹) # A⁻¹ = R⁻¹⋅D⁻¹⋅Lᵀ
    mul_P!(A⁻¹, p, A⁻¹) # A⁻¹ = P⋅R⁻¹⋅D⁻¹⋅Lᵀ

    return nothing
end


function inv!(F::LDR)

    # calculate [L⋅D⋅R⋅Pᵀ]⁻¹ = P⋅R⁻¹⋅D⁻¹⋅Lᵀ
    p  = F.p_tmp
    Lᵀ = F.M_tmp
    adjoint!(Lᵀ, F.L)
    D = Diagonal(F.d)
    ldiv!(F.L,D,Lᵀ) # D⁻¹⋅Lᵀ
    R = UpperTriangular(F.R)
    ldiv!(R,F.L) # R⁻¹⋅D⁻¹⋅Lᵀ
    inv_P!(p, F.pᵀ)
    mul_P!(F.L, p, F.L) # P⋅R⁻¹⋅D⁻¹⋅Lᵀ
    ldr!(F)

    return nothing
end


function inv_IpF!(G::AbstractMatrix{T}, F::LDR{T}) where {T}

    # construct Dmin = min(D,1) and Dmax⁻¹ = [max(D,1)]⁻¹ matrices
    dmin     = @view G[:,1]
    inv_dmax = @view G[:,2]
    @inbound @fastmath for i in 1:length(F.d)
        if abs(F.d[i]) > 1
            dmin[i]  = 1
            inv_dmax = 1/F.d[i]
        else
            dmin[i]  = F.d[i]
            inv_dmax = 1
        end
    end
    Dmin   = Diagonal(dmin)
    Dmax⁻¹ = Diagonal(inv_dmax)

    # store the original [P⋅R⁻¹]₀
    p = F.p_tmp
    inv_P!(p, F.pᵀ) # P
    copyto!(F.M_tmp, F.R)
    R⁻¹ = UpperTriangular(F.M_tmp)
    LinearAlgebra.inv!(R⁻¹)

    # caclulate L⋅Dmin
    LDmin = F.L
    rmul!(LDmin, Dmin)

    # calculate [P⋅R⁻¹]₀⋅Dmax⁻¹
    PR⁻¹Dmax⁻¹ = F.R
    mul!(PR⁻¹Dmax⁻¹, R⁻¹, Dmax⁻¹)
    mul_P!(PR⁻¹Dmax⁻¹, p, PR⁻¹Dmax⁻¹)

    # calculate LDR decomposition of L⋅D⋅R⋅Pᵀ = [P⋅R⁻¹⋅Dmax⁻¹ + L⋅Dmin]
    @. F.L = PR⁻¹Dmax⁻¹ + LDmin
    ldr!(F)

    # invert the LDR decomposition, [L⋅D⋅R⋅Pᵀ]⁻¹ = P⋅R⁻¹⋅D⁻¹⋅Lᵀ
    adjoint!(F.L) # Lᵀ
    D = Diagonal(F.d)
    ldiv!(D,F.L) # D⁻¹⋅Lᵀ
    R = UpperTriangular(F.R)
    ldiv!(R,F.L) # R⁻¹⋅D⁻¹⋅Lᵀ
    inv_P!(F.pᵀ)
    mul_P!(F.L, F.pᵀ, F.L) # F.L = P⋅R⁻¹⋅D⁻¹⋅Lᵀ

    # F.L = Dmax⁻¹⋅[P⋅R⁻¹⋅D⁻¹⋅Lᵀ]
    lmul!(Dmax⁻¹, F.L)

    # calculate new LDR decomposition
    ldr!(F)

    # G = [P⋅R⁻¹]₀⋅F
    lmul!(R⁻¹, F.L)
    mul_P!(F.L, p, F.L)
    copyto!(G, F)

    return nothing
end


function lmul!(B::AbstractMatrix{T}, F::LDR{T}) where {T}

    (; L, pᵀ, M_tmp, p_tmp) = F
    D = Diagonal(F.d)
    R = F.R
    pᵀ_prev = p_tmp

    # B⋅L
    mul!(M_tmp, B, L)

    # B⋅L⋅D
    mul!(L, M_tmp, D)

    # store current R and pᵀ arrays
    copyto!(M_tmp, R)
    R_prev = UpperTriangular(M_tmp)
    copyto!(pᵀ_prev, pᵀ)

    # calculate new L′⋅D′⋅R′⋅P′ᵀ decomposition
    ldr!(F)

    # R′ = R′⋅P′ᵀ⋅R
    mul_P!(R, R, pᵀ)
    rmul!(R, R_prev)

    # P′ᵀ = Pᵀ
    copyto!(pᵀ, pᵀ_prev)

    return nothing
end


function mul!(A::AbstractMatrix{T}, F::LDR{T}, B::AbstractMatrix{T}) where {T}

    (; L, pᵀ, M) = F
    D = Diagonal(F.d)
    R = UpperTriangular(F.R)

    # calculate A = (L⋅D⋅R⋅Pᵀ)⋅B
    mul_P!(M_tmp, pᵀ, B)
    lmul!(R, M_tmp)
    lmul!(D, M_tmp)
    mul!(A, L, M_tmp)

    return nothing
end


function mul!(F′::LDR{T}, B::AbstractMatrix{T}, F::LDR{T}) where {T}

    L  = F.L
    D  = Diagonal(F.d)
    R  = UpperTriangular(F.R)
    pᵀ = F.pᵀ
    p  = F.p

    L′  = F′.L
    D′  = Diagonal(F′.d)
    R′  = F′.R
    p′ᵀ = F′.pᵀ
    p′  = F′.p

    # calculate L′ = B⋅L⋅D
    M_tmp = R′
    mul!(M_tmp, L, D) # L⋅D
    mul!(L′, B, M_tmp) # B⋅(L⋅D)

    # update/calculate F′ decomposition for (B⋅L⋅D)
    ldr!(F′)

    # calculate R′ = R′⋅P′ᵀ⋅R
    mul_P!(R′, p′ᵀ, R′) # (R′⋅P′ᵀ)
    rmul!(R′, R) # (R′⋅P′ᵀ)⋅R

    # set P′ᵀ = Pᵀ (P′ = P)
    copyto!(p′ᵀ, pᵀ)
    copyto!(p′, p)

    return nothing
end


function mul!(F′::LDR{T}, F₂::LDR{T}, F₁::LDR{T}) where {T}

    M_tmp = F′.M_tmp

    # calulcate R₂⋅P₂ᵀ⋅L₁
    mul_P!(F′.L, F₂.pᵀ, F₁.L) # P₂ᵀ⋅L₁
    R₂ = LowerTriangular(F₂.R)
    rmul!(F′.L, R₂) # R₂⋅(P₂ᵀ⋅L₁)

    # calculate (R₂⋅P₂ᵀ⋅L₁)⋅D₁
    D₁ = Diagonal(F₁.d)
    rmul!(F′.L, D₁)

    # calculate D₂⋅(R₂⋅P₂ᵀ⋅L₁⋅D₁)
    D₂ = Diagonal(F₂.d)
    lmul!(D₂, F′.L)

    # calculate the decomposition of (D₂⋅R₂⋅P₂ᵀ⋅L₁⋅D₁)
    ldr!(F′)

    # calculate L′ = L₂⋅L′
    mul!(M_tmp, F₂.L, F′.L)
    copyto!(F′.L, M_tmp)

    # calculate R′ = R′⋅P′ᵀ⋅R₁
    R₁ = LowerTriangular(F₁.R)
    mul_P!(F′.R, F′.pᵀ, F′.R) # R′⋅Pᵀ
    rmul!(F′.R′, R₁) # (R′⋅Pᵀ)⋅R₁

    # P′ᵀ = P₁ᵀ (P′ = P₁)
    copyto!(F′.pᵀ, F₁.pᵀ)
    copyto!(F′.p, F₁.p)

    return nothing
end


#######################
## DEVELOPER METHODS ##
#######################

# evaluate the matrix-product A = P⋅B, modifying A in-place,
# where P is a permutation matrix represented by p.
# Note that A and B can be the same matrix.
function mul_P!(A::AbstractMatrix, p::AbstractVector{Int}, B::AbstractMatrix)

    @views @. A = B[p,:]
    return nothing
end

# evaluate the matrix-product A = B⋅P, modifying A in-place,
# where P is a permutation matrix represented by p.
# Note that A and B can be the same matrix.
function mul_P!(A::AbstractMatrix, B::AbstractMatrix, p::AbstractVector{Int})

    A′ = @view A[:,p]
    @. A′ = B

    return nothing
end

# in-place inversion of a permutation matrix P represented by the vector p.
function inv_P!(p::Vector{Int})

    p′ = @view p[p]
    for i in 1:length(p)
        p′[i] = i
    end

    return nothing
end

# calculate inver of permutation matrix P represented by p,
# saving the result to p⁻¹=pᵀ
function inv_P!(pᵀ::Vector{Int}, p::Vector{Int})

    sortperm!(pᵀ, p)

    return nothing
end

# in-place adjoint of square matrix
function adjoint!(A::AbstractMatrix)

    @inbounds @simd for i in 1:size(A,1)
        for j in i:size(A,1)
            tmp    = A[i,j]
            A[i,j] = conj(A[j,i])
            A[j,i] = conj(tmp)
        end
    end

    return nothing
end