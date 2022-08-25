@doc raw"""
    mul_D!(A, d, B)

Calculate the matrix product ``A = D \cdot B,`` where ``D`` is a diagonal matrix
represented by the vector `d`.
"""
function mul_D!(A::AbstractMatrix, d::AbstractVector, B::AbstractMatrix)

    @inbounds @simd for c in eachindex(d)
        for r in eachindex(d)
            A[r,c] = d[r] * B[r,c]
        end
    end

    return nothing
end


@doc raw"""
    mul_D!(A, B, d)

Calculate the matrix product ``A = B \cdot D,`` where ``D`` is a diagonal matrix
represented by the vector `d`.
"""
function mul_D!(A::AbstractMatrix, B::AbstractMatrix, d::AbstractVector)

    @inbounds @simd for c in eachindex(d)
        for r in eachindex(d)
            A[r,c] = B[r,c] * d[c]
        end
    end

    return nothing
end


@doc raw"""
    lmul_D!(d, M)

In-place calculation of the matrix product ``M = D \cdot M,`` where ``D`` is a diagonal
matrix represented by the vector `d`.
"""
function lmul_D!(d::AbstractVector, M::AbstractMatrix)

    @inbounds @simd for c in eachindex(d)
        for r in eachindex(d)
            M[r,c] *= d[r]
        end
    end

    return nothing
end


@doc raw"""
    rmul_D!(M, d)

In-place calculation of the matrix product ``M = M \cdot D,`` where ``D`` is a diagonal
matrix represented by the vector `d`.
"""
function rmul_D!(M::AbstractMatrix, d::AbstractVector)

    @inbounds @simd for c in eachindex(d)
        for r in eachindex(d)
            M[r,c] *= d[c]
        end
    end

    return nothing
end


@doc raw"""
    ldiv_D!(d, M)

In-place calculation of the matrix product ``M = D^{-1} \cdot M,`` where ``D`` is a diagonal
matrix represented by the vector `d`.
"""
function ldiv_D!(d::AbstractVector, M::AbstractMatrix)

    @inbounds @simd for c in eachindex(d)
        for r in eachindex(d)
            M[r,c] /= d[r]
        end
    end

    return nothing
end


@doc raw"""
    mul_P!(A, p, B)

Evaluate the matrix product ``A = P \cdot B,`` where ``P`` is a permutation matrix
represented by the vector of integers `p`.
"""
function mul_P!(A::AbstractMatrix, p::AbstractVector{Int}, B::AbstractMatrix)

    @views @. A = B[p,:]
    return nothing
end


@doc raw"""
    mul_P!(A, B, p)

Evaluate the matrix product ``A = B \cdot P,`` where ``P`` is a permutation matrix
represented by the vector of integers `p`.
"""
function mul_P!(A::AbstractMatrix, B::AbstractMatrix, p::AbstractVector{Int})

    A′ = @view A[:,p]
    @. A′ = B

    return nothing
end


@doc raw"""
    inv_P!(p⁻¹, p)

Calculate the inverse/tranpose ``P^{-1}=P^T`` of a permuation ``P`` represented by
the vector `p`, writing the result to `p⁻¹`.
"""
function inv_P!(p⁻¹::Vector{Int}, p::Vector{Int})

    sortperm!(p⁻¹, p)

    return nothing
end