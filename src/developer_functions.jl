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

Calculate the inverse/transpose ``P^{-1}=P^T`` of a permuation ``P`` represented by
the vector `p`, writing the result to `p⁻¹`.
"""
function inv_P!(p⁻¹::Vector{Int}, p::Vector{Int})

    sortperm!(p⁻¹, p)

    return nothing
end

@doc raw"""
    perm_sign(p::AbstractVector{Int})

Calculate the sign/parity of the permutation `p`, ``\textrm{sgn}(p) = \pm 1.``
"""
function perm_sign(p::AbstractVector{Int})
   
    N = length(p)
    sgn = 0
    # iterate over elements in permuation
    @fastmath @inbounds for i in eachindex(p)
        # if element has not been assigned to a cycle
        if p[i] <= N
            k = 1
            j = i
            # calculate cycle containing current element
            while p[j] != i
                tmp = j
                j = p[j]
                p[tmp] += N # mark element as assigned to cycle
                k += 1
            end
            p[j] += N # mark element as assigned to cycle
            sgn += (k-1)%2
        end
    end
    # set sgn = ±1
    sgn = 1 - 2*(sgn%2)
    # restore permutation
    @. p = p - N
    
    return sgn
end