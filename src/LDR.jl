@doc raw"""
    LDR{T<:Number, E<:Real} <: Factorization{T}

Represents the matrix factorization ``A = L D R`` for a square matrix ``A``,
where ``L`` is a unitary matrix, ``D`` is a diagonal matrix of strictly positive real numbers,
and ``R`` is defined such that ``|\det R| = 1``.

This factorization is based on a column-pivoted QR decomposition ``A P = Q R',`` such that
```math
\begin{align*}
L &:= Q \\
D &:= \vert \textrm{diag}(R') \vert \\
R &:= \vert \textrm{diag}(R') \vert^{-1} R' P^T\\
\end{align*}
```

# Fields

- `L::Matrix{T}`: The left unitary matrix ``L`` in a [`LDR`](@ref) factorization.
- `d::Vector{E}`: A vector representing the diagonal matrix ``D`` in a [`LDR`](@ref) facotorization.
- `R::Matrix{T}`: The right matrix ``R`` in a [`LDR`](@ref) factorization.
"""
struct LDR{T<:Number, E<:AbstractFloat} <: Factorization{T}

    "The left unitary matrix ``L``."
    L::Matrix{T}

    "Vector representing diagonal matrix ``D``."
    d::Vector{E}

    "The right upper triangular matrix ``R``."
    R::Matrix{T}
end


@doc raw"""
    LDRWorkspace{T<:Number}

A workspace to avoid dyanmic memory allocations when performing computations
with a [`LDR`](@ref) factorization.

# Fields

- `qr_ws::QRWorkspace{T,E}`: [`QRWorkspace`](@ref) for calculating column pivoted QR factorization without dynamic memory allocations.
- `lu_ws::LUWorkspace{T}`: [`LUWorkspace`](@ref) for calculating LU factorization without dynamic memory allocations.
- `M::Matrix{T}`: Temporary storage matrix for avoiding dynamic memory allocations. This matrix is used/modified when a [`LDR`](@ref) factorization is calculated.
- `M′::Matrix{T}`: Temporary storage matrix for avoiding dynamic memory allocations.
- `M″::Matrix{T}`: Temporary storage matrix for avoiding dynamic memory allocations.
- `v::Vector{T}`: Temporary storage vector for avoiding dynamic memory allocations.
"""
struct LDRWorkspace{T<:Number, E<:AbstractFloat}

    "Workspace for calculating column pivoted QR factorization without allocations."
    qr_ws::QRWorkspace{T,E}

    "Workspace for calculating LU factorization without allocations."
    lu_ws::LUWorkspace{T}

    "Temporary storage matrix. This matrix is used/modified when a [`LDR`](@ref) factorization are calculated."
    M::Matrix{T}

    "Temporary storage matrix."
    M′::Matrix{T}

    "Temporary storage matrix."
    M″::Matrix{T}

    "Temporary storage vector."
    v::Vector{E}
end


@doc raw"""
    ldr(A::AbstractMatrix{T}) where {T}

Allocate an [`LDR`](@ref) factorization based on `A`, but does not calculate its [`LDR`](@ref) factorization,
instead initializing the factorization to the identity matrix.
"""
function ldr(A::AbstractMatrix{T}) where {T}

    n = checksquare(A)
    E = real(T)
    L = zeros(T,n,n)
    d = zeros(E,n)
    R = zeros(T,n,n)
    F = LDR{T,E}(L,d,R)
    ldr!(F, I)
    return F
end

@doc raw"""
    ldr(A::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}

Return the [`LDR`](@ref) factorization of the matrix `A`.
"""
function ldr(A::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}

    F = ldr(A)
    ldr!(F, A, ws)
    return F
end

@doc raw"""
    ldr(F::LDR{T}, ignore...) where {T}

Return a copy of the [`LDR`](@ref) factorization `F`.
"""
function ldr(F::LDR{T}, ignore...) where {T}

    L = copy(F.L)
    d = copy(F.d)
    R = copy(F.R)
    return LDR(L,d,R)
end


@doc raw"""
    ldr!(F::LDR{T}, A::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}

Calculate the [`LDR`](@ref) factorization `F` for the matrix `A`.
"""
function ldr!(F::LDR{T}, A::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}

    copyto!(F.L, A)
    return ldr!(F, ws)
end

@doc raw"""
    ldr!(F::LDR{T}, I::UniformScaling, ignore...) where {T}

Set the [`LDR`](@ref) factorization equal to the identity matrix.
"""
function ldr!(F::LDR{T}, I::UniformScaling, ignore...) where {T}

    (; L, d, R) = F
    copyto!(L, I)
    fill!(d, 1)
    copyto!(R, I)
    return nothing
end

@doc raw"""
    ldr!(Fout::LDR{T}, Fin::LDR{T}, ignore...) where {T}

Copy the [`LDR`](@ref) factorization `Fin` to `Fout`.
"""
function ldr!(Fout::LDR{T}, Fin::LDR{T}, ignore...) where {T}

    copyto!(Fout.L, Fin.L)
    copyto!(Fout.d, Fin.d)
    copyto!(Fout.R, Fin.R)
    return nothing
end

@doc raw"""
    ldr!(F::LDR, ws::LDRWorkspace{T}) where {T}

Calculate the [`LDR`](@ref) factorization for the matrix `F.L`.
"""
function ldr!(F::LDR, ws::LDRWorkspace{T}) where{T}

    (; qr_ws, M) = ws
    (; L, d, R) = F

    # calclate QR decomposition
    LAPACK.geqp3!(L, qr_ws)

    # extract upper triangular matrix R
    copyto!(M, L)
    triu!(M)

    # set D = Diag(R), represented by vector d
    @fastmath @inbounds for i in 1:size(L,1)
        d[i] = abs(M[i,i])
    end

    # calculate R = D⁻¹⋅R
    ldiv_D!(d, M)

    # calculate R⋅Pᵀ
    mul_P!(R, M, qr_ws.jpvt)

    # construct L (same as Q) matrix
    LAPACK.orgqr!(L, qr_ws)

    return nothing
end


@doc raw"""
    ldrs(A::AbstractMatrix{T}, N::Int) where {T}

Return a vector of [`LDR`](@ref) factorizations of length `N`, where each one represents the
identity matrix of the same size as ``A``.
"""
function ldrs(A::AbstractMatrix{T}, N::Int) where {T}
    
    Fs = LDR{T,real(T)}[]
    for i in 1:N
        push!(Fs, ldr(A))
    end

    return Fs
end


@doc raw"""
    ldrs(A::AbstractMatrix{T}, N::Int, ws::LDRWorkspace{T,E}) where {T,E}

Return a vector of [`LDR`](@ref) factorizations of length `N`, where each one represents the matrix `A`.
"""
function ldrs(A::AbstractMatrix{T}, N::Int, ws::LDRWorkspace{T,E}) where {T,E}
    
    Fs = LDR{T,E}[]
    F = ldr(A, ws)
    push!(Fs, F)
    for i in 2:N
        push!(Fs, ldr(F))
    end

    return Fs
end


@doc raw"""
    ldrs(A::AbstractArray{T,3}, ws::LDRWorkspace{T,E}) where {T,E}

Return a vector of [`LDR`](@ref) factorizations of length `size(A, 3)`, where there is an
[`LDR`](@ref) factorization for each matrix `A[:,:,i]`.
"""
function ldrs(A::AbstractArray{T,3}, ws::LDRWorkspace{T,E}) where {T,E}
    
    Fs = LDR{T,E}[]
    for i in axes(A,3)
        Aᵢ = @view A[:,:,i]
        push!(Fs, ldr(Aᵢ, ws))
    end
    
    return Fs
end


@doc raw"""
    ldrs!(Fs::AbstractVector{LDR{T,E}}, A::AbstractArray{T,3}, ws::LDRWorkspace{T,E}) where {T,E}

Calculate the [`LDR`](@ref) factorization `Fs[i]` for the matrix `A[:,:,i]`.
"""
function ldrs!(Fs::AbstractVector{LDR{T,E}}, A::AbstractArray{T,3}, ws::LDRWorkspace{T,E}) where {T,E}

    for i in eachindex(Fs)
        Aᵢ = @view A[:,:,i]
        ldr!(Fs[i], Aᵢ, ws)
    end

    return nothing
end


@doc raw"""
    ldr_workspace(A::AbstractMatrix)

    ldr_workspace(F::LDR)

Return a [`LDRWorkspace`](@ref) that can be used to avoid dynamic memory allocations.
"""
function ldr_workspace(A::AbstractMatrix{T}) where {T}

    E  = real(T)
    n  = checksquare(A)
    M  = zeros(T, n, n)
    M′ = zeros(T, n, n)
    M″ = zeros(T, n, n)
    v  = zeros(E, n)
    copyto!(M, I)
    qr_ws = QRWorkspace(M)
    lu_ws = LUWorkspace(M)

    return LDRWorkspace(qr_ws, lu_ws, M, M′, M″, v)
end

ldr_workspace(F::LDR) = ldr_workspace(F.L)


@doc raw"""
    copyto!(ldrws_out::LDRWorkspace{T,E}, ldrws_in::LDRWorkspace{T,E}) where {T,E}

Copy the contents of `ldrws_in` into `ldrws_out`.
"""
function copyto!(ldrws_out::LDRWorkspace{T,E}, ldrws_in::LDRWorkspace{T,E}) where {T,E}

    copyto!(ldrws_out.qr_ws, ldrws_in.qr_ws)
    copyto!(ldrws_out.lu_ws, ldrws_in.lu_ws)
    copyto!(ldrws_out.M, ldrws_in.M)
    copyto!(ldrws_out.M′, ldrws_in.M′)
    copyto!(ldrws_out.M″, ldrws_in.M″)
    copyto!(ldrws_out.v, ldrws_in.v)

    return nothing
end