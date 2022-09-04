@doc raw"""
    LDR{T<:Number, E<:Real} <: Factorization{T}

Represents the matrix factorization ``A P = L D R`` for a square matrix ``A,`` which may equivalently be
written as ``A = (L D R) P^{-1} = (L D R) P^T``.

In the above ``L`` is a unitary matrix, ``D`` is a diagonal matrix of strictly positive real numbers,
and ``R`` is an upper triangular matrix. Lastly, ``P`` is a permutation matrix, for which ``P^{-1}=P^T``.

This factorization is based on a column-pivoted QR decomposition ``A P = Q R,`` such that
```math
\begin{align*}
L &= Q \\
D &= \vert \textrm{diag}(R) \vert \\
R &= \vert \textrm{diag}(R) \vert^{-1} R \\
P &= P.
\end{align*}
```

# Fields

- `L::Matrix{T}`: The unitary matrix ``L.``
- `d::Vector{E}`: A vector representing the diagonal matrix ``D.``
- `R::Matrix{T}`: The upper triangular matrix ``R.``
- `pᵀ::Vector{Int}`: A permutation vector representing the permuation matrix ``P^T.``
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

    "Workspace for calculating QR decomposition using LAPACK without allocations."
    ws::QRPivotedWs{T, E}
end


@doc raw"""
    LDRWorkspace{T<:Number}

A workspace to avoid temporary memory allocations when performing computations
with an [`LDR`](@ref) factorization.
"""
struct LDRWorkspace{T<:Number}

    "Temporary storage matrix."
    M::Matrix{T}

    "Temporary storage matrix."
    M′::Matrix{T}

    "Temporary storage matrix."
    M″::Matrix{T}

    "Temporary storage vector for permuation."
    p::Vector{Int}

    "Temporary storage vector for permuation."
    p′::Vector{Int}

    "Temporary storage vector."
    v::Vector{T}

    "Temporary storage vector."
    v′::Vector{T}

    "Temporary storage vector."
    v″::Vector{T}

    "Temporary storage vector."
    v‴::Vector{T}
end


@doc raw"""
    ldr(A::AbstractMatrix)

Calculate and return the [`LDR`](@ref) decomposition for the matrix `A`.
"""
function ldr(A::AbstractMatrix{T})::LDR{T} where {T}

    # make sure A is a square matrix
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
    ws = QRPivotedWs(L)
    pᵀ = ws.jpvt

    # instantiate LDR decomposition
    F = LDR(L,d,R,pᵀ,ws)

    # calculate LDR decomposition
    ldr!(F, A)
    
    return F
end


@doc raw"""
    ldr(F::LDR)

Return a new [`LDR`](@ref) factorization that is a copy of `F`.
"""
function ldr(F::LDR)

    F′ = ldr(F.L)
    copyto!(F′, F)

    return F′
end


@doc raw"""
    ldr!(F::LDR, A::AbstractMatrix)

Calculate the [`LDR`](@ref) decomposition `F` for the matrix `A`.
"""
function ldr!(F::LDR{T}, A::AbstractMatrix{T}) where {T}

    @assert size(F) == size(A)

    copyto!(F.L, A)
    ldr!(F)

    return nothing
end


@doc raw"""
    ldr!(F::LDR)

Re-calculate the [`LDR`](@ref) factorization `F` in-place based on the current contents
of the matrix `F.L`.
"""
function ldr!(F::LDR)

    (; L, d, R, ws) = F

    # calclate QR decomposition
    LAPACK.geqp3!(ws, L)

    # extract upper triangular matrix R
    R′ = UpperTriangular(L)
    copyto!(R, R′)

    # set D = Diag(R), represented by vector d
    @inbounds for i in 1:size(L,1)
        d[i] = abs(R[i,i])
    end

    # calculate R = D⁻¹⋅R
    ldiv_D!(d, R)

    # construct L (same as Q) matrix
    LAPACK.orgqr!(ws, L)

    return nothing
end


"""
    ldr!(F::LDR, I::UniformScaling)

Update the [`LDR`](@ref) factorization `F` to reflect the identity matrix.
"""
function ldr!(F::LDR, I::UniformScaling)

    N = length(F.d)
    copyto!(F.L, I)
    @. F.d = 1
    copyto!(F.R, I)
    @. F.pᵀ = 1:N

    return nothing
end


@doc raw"""
    ldrs(A::AbstractMatrix{T}, N::Int) where {T}

Return a vector of `N` [`LDR`](@ref) factorizations, where each one represent the matrix `A`.
"""
function ldrs(A::AbstractMatrix{T}, N::Int) where {T}
    
    Fs = Vector{LDR{T}}(undef,0)
    for i in 1:N
        push!(Fs, ldr(A))
    end
    return Fs
end


@doc raw"""
    ldrs(A::AbstractArray{T,3}) where {T}

Return a vector of [`LDR`](@ref) factorizations of length `size(A, 3)`, where there is an
[`LDR`](@ref) factorization for each matrix `A[:,:,i]`.
"""
function ldrs(A::AbstractArray{T,3}) where {T}
    
    Fs = Vector{LDR{T}}(undef,0)
    N  = size(A,3)
    for i in 1:N
        Aᵢ = @view A[:,:,i]
        push!(Fs, ldr(Aᵢ))
    end
    
    return Fs
end


@doc raw"""
    ldrs!(Fs::Vector{LDR{T}}, A::AbstractArray{T,3}) where {T}

Update the vector `Fs` of [`LDR`](@ref) factorization based on the sequence of matrices
contained in `A`.
"""
function ldrs!(Fs::Vector{LDR{T}}, A::AbstractArray{T,3}) where {T}
    
    for i in 1:size(A,3)
        Aᵢ = @view A[:,:,i]
        ldr!(Fs[i], Aᵢ)
    end
    
    return nothing
end

@doc raw"""

    ldr_workspace(A::AbstractMatrix)

    ldr_workspace(F::LDR)
    
    ldr_workspace(Fs::Vector{LDR})

Return a [`LDRWorkspace`](@ref) that can be used to avoid dynamic memory allocations.
"""
function ldr_workspace(A::AbstractMatrix)

    @assert size(A,1) == size(A,2)

    N  = size(A,1)
    M  = similar(A)
    M′ = similar(A)
    M″ = similar(A)
    p  = zeros(Int, N)
    p′ = zeros(Int, N)
    v  = zeros(eltype(A), N)
    v′ = zeros(eltype(A), N)
    v″ = zeros(eltype(A), N)
    v‴ = zeros(eltype(A), N)
    ws = LDRWorkspace(M, M′, M″, p, p′, v, v′, v″, v‴)

    return ws
end

ldr_workspace(F::LDR) = ldr_workspace(F.L)
ldr_workspace(Fs::Vector{LDR}) = ldr_workspace(Fs[1])