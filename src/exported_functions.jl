@doc raw"""
    chain_mul!(F::LDR{T}, B::AbstractArray{T,3};
               tmp=similar(F.L), reversed::Bool=false) where {T}

Calculate a numerically stable product of a chain of matrices ``B_L B_{L-1} \dots B_l \dots B_2 B_1,`` where the ``B_l``
matrix in the sequence is given by `B[:,:,l]`, such that the final product is represented by
the LDR decomposition `F`. If `reversed = true`, when iterate over `B[:,:,l]` in reverse.
Internally uses the [`lmul!`](@ref) method to perform the matrix multiplications.
"""
function chain_mul!(F::LDR{T}, B::AbstractArray{T,3};
                    tmp::AbstractMatrix{T}=similar(F.L), reversed::Bool=false) where {T}

    @assert size(B,1) == size(B,2) == size(F,1)

    # number of B matrices
    L  = size(B, 3)

    # defining whether to iterate over B matrices in forward or reverse direction
    if reversed
        index_start = L
        index_step  = -1
        index_end   = 1
    else
        index_start = 1
        index_step  = 1
        index_end   = L
    end

    B₁ = @view B[:,:,index_start]
    ldr!(F, B₁) # construct LDR decomposition for first matrix in chain
    for l in (index_start+index_step):index_step:index_end
        Bₗ = @view B[:,:,l]
        lmul!(Bₗ, F, tmp=tmp) # stabalized multiplication by next matrix in chain
    end

    return nothing
end


@doc raw"""
    chain_lmul!(B::AbstractArray{T,3}, F::LDR{T}, B̄::AbstractArray{T};
                tmp=similar(F.L), reversed::Bool=false) where {T}

Calculate a numerically stable product of a chain of matrices ``B_L \dots B_l \dots B_2 B_1``
going from right to left, where the ``B_l`` matrix in the sequence is given by `B[:,:,l]`,
such that the final product is represented by the LDR decomposition `F`. Additionally, let
`B̄` contain the partial products such that `B̄[:,:,l]` corresponds to the partial product
``B_l B_{l-1} \dots B_1.`` If `reversed = true`, when iterate over `B[:,:,l]` and `B̄[:,:,l]` in reverse.
Internally uses the [`lmul!`](@ref) method to perform the matrix multiplications.
"""
function chain_lmul!(B::AbstractArray{T,3}, F::LDR{T}, B̄::AbstractArray{T,3};
                     tmp::AbstractMatrix{T}=similar(F.L), reversed::Bool=false) where {T}

    @assert size(B,1) == size(B,2) == size(F,1)
    @assert size(B) == size(B̄)

    # get number of B matrices
    L = size(B, 3)

    # defining whether to iterate over B matrices in forward
    # or reverse direction
    if reversed
        index_start = L
        index_end   = 1
        index_step  = -1
    else
        index_start = 1
        index_end   = L
        index_step  = 1
    end 

    B₁ = @view B[:,:,index_start]
    B̄₁ = @view B[:,:,index_start]
    copyto!(B̄₁, B₁)
    ldr!(F, B₁) # construct LDR decomposition for first matrix in chain
    for l in (index_start+index_step):index_step:index_end
        Bₗ = @view B[:,:,l]
        B̄ₗ = @view B̄[:,:,l]
        lmul!(Bₗ, F, tmp=tmp) # stabalized multiplication by next matrix in chain
        copyto!(B̄ₗ, F) # record the partial product
    end

    return nothing
end


@doc raw"""
    chain_rmul!(F::LDR{T}, B::AbstractArray{T,3}, B̄::AbstractArray{T,3}, tmp=similar(F.L);
                tmp::AbstractMatrix{T}=similar(F.L), reversed::Bool=false) where {T}

Calculate a numerically stable product of a chain of matrices ``B_1 B_2 \dots B_l \dots B_L``
going from left to right, where the ``B_l`` matrix in the sequence is given by `B[:,:,l]`,
such that the final product is represented by the LDR decomposition `F`. Additionally, let
`B̄` contain the partial products such that `B̄[:,:,l]` corresponds to the partial product
``B_1 B_2 \dots B_l.`` If `reversed = true`, when iterate over `B[:,:,l]` and `B̄[:,:,l]` in reverse.
Internally uses the [`rmul!`](@ref) method to perform the matrix multiplications.
"""
function chain_rmul!(F::LDR{T}, B::AbstractArray{T,3}, B̄::AbstractArray{T,3};
                     tmp::AbstractMatrix{T}=similar(F.L), reversed::Bool=false) where {T}

    @assert size(B,1) == size(B,2) == size(F,1)
    @assert size(B) == size(B̄)

    # get number of B matrices
    L = size(B, 3)

    # defining whether to iterate over B matrices in forward
    # or reverse direction
    if reversed
        index_start = L
        index_end   = 1
        index_step  = -1
    else
        index_start = 1
        index_end   = L
        index_step  = 1
    end 

    B₁ = @view B[:,:,index_start]
    B̄₁ = @view B[:,:,index_start]
    copyto!(B̄₁, B₁)
    ldr!(F, B₁) # construct LDR decomposition for first matrix in chain
    for l in (index_start+index_step):index_step:index_end
        Bₗ = @view B[:,:,l]
        B̄ₗ = @view B̄[:,:,l]
        rmul!(F, Bₗ, tmp=tmp) # stabalized multiplication by next matrix in chain
        copyto!(B̄ₗ, F) # record the partial product
    end

    return nothing
end


@doc raw"""
    inv!(A::AbstractMatrix, F::LDR)

Calculate the inverse of a matrix ``A`` represented of the LDR decomposition `F`,
writing the inverse matrix `A⁻¹`.
"""
function inv!(A⁻¹::AbstractMatrix{T}, F::LDR{T}) where {T}

    # calculate inverse A⁻¹ = P⋅R⁻¹⋅D⁻¹⋅Lᵀ
    L   = F.L
    d   = F.d
    R   = UpperTriangular(F.R)
    M   = F.M_tmp
    p   = F.p_tmp
    inv_P!(p, F.pᵀ)
    adjoint!(A⁻¹, F.L) # A⁻¹ = Lᵀ
    ldiv_D!(d, M) # A⁻¹ = D⁻¹⋅Lᵀ
    ldiv!(R, M) # A⁻¹ = R⁻¹⋅D⁻¹⋅Lᵀ
    mul_P!(A⁻¹, p, M) # A⁻¹ = P⋅R⁻¹⋅D⁻¹⋅Lᵀ

    return nothing
end


@doc raw"""
    inv!(F::LDR)

Invert the LDR decomposition `F` in-place.
"""
function inv!(F::LDR)

    # given F = [L⋅D⋅R⋅Pᵀ], calculate F⁻¹ = [L⋅D⋅R⋅Pᵀ]⁻¹ = P⋅R⁻¹⋅D⁻¹⋅Lᵀ in-place
    p  = F.p_tmp
    Lᵀ = F.M_tmp
    adjoint!(Lᵀ, F.L)
    ldiv_D!(F.d, Lᵀ) # D⁻¹⋅Lᵀ
    R = UpperTriangular(F.R)
    ldiv!(R, Lᵀ) # R⁻¹⋅D⁻¹⋅Lᵀ
    inv_P!(p, F.pᵀ)
    mul_P!(F.L, p, Lᵀ) # P⋅R⁻¹⋅D⁻¹⋅Lᵀ
    ldr!(F)

    return nothing
end


@doc raw"""
    inv_IpA!(G::AbstractMatrix, F::LDR; d_min=similar(F.d), inv_d_max=similar(F.d))

Given a matrix ``A`` represented by the LDR factorization `F`, calculate the numerically stabalized inverse
```math
G = (I + A)^{-1},
```
storing the result in the matrix `G`. Note that `F` is left modified by this method, so that it now
corresponds to the matrix `G` instead of the original matrix `A` it previously represented.

# Algorithm

Given an LDR factorization of ``A``, calculate ``G = (I + A)^{-1}`` using the procedure
```math
\begin{align*}
G = & \left(I+A\right)^{-1}\\
  = & (I+\overset{D_{a,\min}D_{a,\max}}{L_{a}\overbrace{D_{a}}R_{a}}P_{a}^{T})^{-1}\\
  = & \left(I+L_{a}D_{a,\min}D_{a,\max}R_{a}P_{a}^{T}\right)^{-1}\\
  = & \left(\left[P_{a}R_{a}^{-1}D_{a,\max}^{-1}+L_{a}D_{a,\min}\right]D_{a,\max}R_{a}P_{a}^{T}\right)^{-1}\\
  = & P_{a}R_{a}^{-1}D_{a,\max}^{-1}(\overset{L_{0}D_{0}R_{0}P_{0}^{T}}{\overbrace{P_{a}R_{a}^{-1}D_{a,\max}^{-1}+L_{a}D_{a,\min}}})^{-1}\\
  = & P_{a}R_{a}^{-1}D_{a,\max}^{-1}(L_{0}D_{0}R_{0}P_{0}^{T})^{-1}\\
  = & P_{a}R_{a}^{-1}\overset{L_{1}D_{1}R_{1}P_{1}^{T}}{\overbrace{D_{a,\max}^{-1}P_{0}R_{0}^{-1}D_{0}^{-1}L_{0}^{\dagger}}}\\
  = & P_{a}R_{a}^{-1}L_{1}D_{1}R_{1}P_{1}^{T},
\end{align*}
```
where ``D_{\min} = \min(D, 1)`` and ``D_{\max} = \max(D, 1).``
"""
function inv_IpA!(G::AbstractMatrix{T}, F::LDR{T};
                  d_min::AbstractVector{T}=similar(F.d),
                  inv_d_max::AbstractVector{T}=similar(F.d)) where {T}

    @assert length(F.pᵀ) == length(tmp)

    # construct Dmin = min(D,1) and Dmax⁻¹ = [max(D,1)]⁻¹ matrices
    @inbounds @fastmath for i in eachindex(F.d)
        if abs(F.d[i]) > 1
            d_min[i]     = 1
            inv_d_max[i] = 1/F.d[i]
        else
            d_min[i]     = F.d[i]
            inv_d_max[i] = 1
        end
    end

    # store the original [P⋅R⁻¹]₀
    p = F.p_tmp
    inv_P!(p, F.pᵀ) # P
    copyto!(F.M_tmp, F.R)
    R⁻¹ = UpperTriangular(F.M_tmp)
    LinearAlgebra.inv!(R⁻¹)

    # caclulate L⋅Dmin
    LDmin = F.L
    rmul!(LDmin, d_min)

    # calculate [P⋅R⁻¹]₀⋅Dmax⁻¹
    PR⁻¹Dmax⁻¹ = F.R
    mul!(PR⁻¹Dmax⁻¹, R⁻¹, inv_d_max)
    mul_P!(PR⁻¹Dmax⁻¹, p, PR⁻¹Dmax⁻¹)

    # calculate LDR decomposition of L⋅D⋅R⋅Pᵀ = [P⋅R⁻¹⋅Dmax⁻¹ + L⋅Dmin]
    @. F.L = PR⁻¹Dmax⁻¹ + LDmin
    ldr!(F)

    # invert the LDR decomposition, [L⋅D⋅R⋅Pᵀ]⁻¹ = P⋅R⁻¹⋅D⁻¹⋅Lᵀ
    adjoint!(G, F.L) # Lᵀ
    ldiv_D!(F.d, G) # D⁻¹⋅Lᵀ
    R = UpperTriangular(F.R)
    ldiv!(R, G) # R⁻¹⋅D⁻¹⋅Lᵀ
    inv_P!(p, F.pᵀ)
    mul_P!(F.R, p, G) # P⋅R⁻¹⋅D⁻¹⋅Lᵀ

    # F.L = Dmax⁻¹⋅[P⋅R⁻¹⋅D⁻¹⋅Lᵀ]
    mul!(F.L, inv_d_max, F.R)

    # calculate new LDR decomposition
    ldr!(F)

    # G = [P⋅R⁻¹]₀⋅F
    lmul!(R⁻¹, F.L)
    mul_P!(F.M_tmp, p, F.L)
    copyto!(F.L, F.M_tmp)
    copyto!(G, F)

    return nothing
end


@doc raw"""
    inv_UpV(G::AbstractMatrix, Fᵤ::LDR, Fᵥ::LDR;
            F::LDR=ldr(Fᵤ),
            dᵤ_min::AbstractVector=similar(Fᵤ.d), inv_dᵤ_max::AbstractVector=similar(Fᵤ.d),
            dᵥ_min::AbstractVector=similar(Fᵥ.d), inv_dᵥ_max::AbstractVector=similar(Fᵥ.d))

Calculate the numerically stable inverse ``G = (U + V)^{-1},`` where the matrices ``U`` and
``V`` are represented by the LDR factorizations `Fᵤ` and `Fᵥ` respectively.

# Algorithm

Letting ``U = [L_u D_u R_u] P_u^T`` and ``V = [L_v D_v R_v] P_v^T,`` the inverse matrix
``G = (U + V)^{-1}`` is calculated using the procedure
```math
\begin{align*}
G = & \left(U+V\right)^{-1}\\
  = & \overset{D_{u,\max}D_{u,\min}}{(L_{u}\overbrace{D_{u}}R_{u}}P_{u}^{T}+\overset{D_{v,\min}D_{v,\max}}{L_{v}\overbrace{D_{v}}R_{v}}P_{v}^{T})^{-1}\\
  = & \left(L_{u}D_{u,\max}D_{u,\min}R_{u}P_{u}^{T}+L_{v}D_{v,\min}D_{v,\max}R_{v}P_{v}^{T}\right)^{-1}\\
  = & \left(L_{u}D_{u,\max}\left[D_{u,\min}R_{u}P_{u}^{T}P_{v}R_{v}^{-1}D_{v,\max}^{-1}+D_{u,\max}^{-1}L_{u}^{\dagger}L_{v}D_{v,\min}\right]D_{v,\max}R_{v}P_{v}^{T}\right)^{-1}\\
  = & P_{v}R_{v}^{-1}D_{v,\max}^{-1}(\overset{L_{0}D_{0}R_{0}P_{0}^{T}}{\overbrace{D_{u,\min}R_{u}P_{u}^{T}P_{v}R_{v}^{-1}D_{v,\max}^{-1}+D_{u,\max}^{-1}L_{u}^{\dagger}L_{v}D_{v,\min}}})^{-1}D_{u,\max}^{-1}L_{u}^{\dagger}\\
  = & P_{v}R_{v}^{-1}D_{v,\max}^{-1}\left(L_{0}D_{0}R_{0}P_{0}^{T}\right)^{-1}D_{u,\max}^{-1}L_{u}^{\dagger}\\
  = & P_{v}R_{v}^{-1}\overset{L_{1}D_{1}R_{1}P_{1}^{T}}{\overbrace{D_{v,\max}^{-1}P_{0}R_{0}^{-1}D_{0}^{-1}L_{0}^{\dagger}D_{u,\max}^{-1}}}L_{u}^{\dagger}\\
  = & P_{v}R_{v}^{-1}L_{1}D_{1}R_{1}P_{1}^{T}L_{u}^{\dagger},
\end{align*}
```
where ``D_\textrm{min} = \min(D,1)`` and ``D_\textrm{max} = \max(D,1).``
"""
function inv_UpV!(G::AbstractMatrix{T}, Fᵤ::LDR{T}, Fᵥ::LDR{T};
                  F::LDR{T}=ldr(Fᵤ),
                  dᵤ_min::AbstractVector{T}=similar(Fᵤ.d),
                  inv_dᵤ_max::AbstractVector{T}=similar(Fᵤ.d),
                  dᵥ_min::AbstractVector{T}=similar(Fᵥ.d),
                  inv_dᵥ_max::AbstractVector{T}=similar(Fᵥ.d)) where {T}

    # calculate Dᵤ₋ = min(Dᵤ,1) and Dᵤ₊⁻¹ = [max(Dᵤ,1)]⁻¹
    @inbounds @fastmath for i in eachindex(Fᵤ.d)
        if abs(Fᵤ.d[i]) > 1
            dᵤ_min[i]     = 1
            inv_dᵤ_max[i] = 1/Fᵤ.d[i]
        else
            dᵤ_min[i]     = Fᵤ.d[i]
            inv_dᵤ_max[i] = 1
        end
    end

    # calculate Dᵥ₋ = min(Dᵥ,1) and Dᵥ₊⁻¹ = [max(Dᵥ,1)]⁻¹
    @inbounds @fastmath for i in eachindex(Fᵥ.d)
        if abs(Fᵥ.d[i]) > 1
            dᵥ_min[i]     = 1
            inv_dᵥ_max[i] = 1/Fᵥ.d[i]
        else
            dᵥ_min[i]     = Fᵥ.d[i]
            inv_dᵥ_max[i] = 1
        end
    end

    # calculate Dᵤ₋⋅Rᵤ⋅Pᵤᵀ⋅Pᵥ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹
    M  = F.M_tmp
    pᵥ = Fᵥ.p_tmp
    Rᵥ = UpperTriangular(Fᵥ.R)
    inv_P!(pᵥ, Fᵥ.pᵀ)
    copyto!(F.L, I) # I
    lmul_D!(inv_dᵥ_max, F.L) # Dᵥ₊⁻¹
    ldiv!(Rᵥ, F.L) # Rᵥ⁻¹⋅Dᵥ₊⁻¹
    mul_P!(M, pᵥ, F.L) # Pᵥ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹
    mul_P!(F.L, Fᵤ.pᵀ, M) # Pᵤᵀ⋅Pᵥ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹
    lmul!(Fᵤ.R, F.L) # Rᵤ⋅Pᵤᵀ⋅Pᵥ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹
    lmul_D!(dᵤ_min, F.L) # Dᵤ₋⋅Rᵤ⋅Pᵤᵀ⋅Pᵥ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹

    # calculate Dᵤ₊⁻¹⋅Lᵤᵀ⋅Lᵥ⋅Dᵥ₋
    M′  = Fᵥ.M_tmp
    Lᵤᵀ = Fᵤ.M_tmp
    adjoint!(Lᵤᵀ, Fᵤ.L)
    copyto!(M, I) # I
    lmul_D!(dᵥ_min, M) # Dᵥ₋
    mul!(M′, Fᵥ.L, M) # Lᵥ⋅Dᵥ₋
    mul!(M, Lᵤᵀ, M′) # Lᵤᵀ⋅Lᵥ⋅Dᵥ₋
    lmul_D!(inv_dᵤ_max, M) # Dᵤ₊⁻¹⋅Lᵤᵀ⋅Lᵥ⋅Dᵥ₋

    # calculate Dᵤ₋⋅Rᵤ⋅Pᵤᵀ⋅Pᵥ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹ + Dᵤ₊⁻¹⋅Lᵤᵀ⋅Lᵥ⋅Dᵥ₋
    @. F.L = F.L + M

    # calculate [L₀⋅D₀⋅R₀]⋅P₀ᵀ = [Dᵤ₋⋅Rᵤ⋅Pᵤᵀ⋅Pᵥ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹ + Dᵤ₊⁻¹⋅Lᵤᵀ⋅Lᵥ⋅Dᵥ₋]
    ldr!(F)

    # calculate Dᵥ₊⁻¹⋅[L₀⋅D₀⋅R₀⋅P₀ᵀ]⁻¹⋅Dᵤ₊⁻¹
    inv!(M′, F) # [L₀⋅D₀⋅R₀⋅P₀ᵀ]⁻¹
    mul_D!(F.L, M′, inv_dᵤ_max) # [L₀⋅D₀⋅R₀⋅P₀ᵀ]⁻¹⋅Dᵤ₊⁻¹
    lmul_D!(inv_dᵥ_max, F.L) # Dᵥ₊⁻¹⋅[L₀⋅D₀⋅R₀⋅P₀ᵀ]⁻¹⋅Dᵤ₊⁻¹

    # calculate [L₁⋅D₁⋅R₁]⋅P₁ᵀ = Dᵥ₊⁻¹⋅[L₀⋅D₀⋅R₀⋅P₀ᵀ]⁻¹⋅Dᵤ₊⁻¹
    ldr!(F)

    # calculate Pᵥ⋅Rᵥ⁻¹⋅[L₁⋅D₁⋅R₁⋅P₁ᵀ]⋅Lᵤᵀ
    mul!(M′, F, Lᵤᵀ) # [L₁⋅D₁⋅R₁⋅P₁ᵀ]⋅Lᵤᵀ
    ldiv!(Rᵥ, M′) # Rᵥ⁻¹⋅[L₁⋅D₁⋅R₁⋅P₁ᵀ]⋅Lᵤᵀ
    mul_P!(G, pᵥ, M′) # G = Pᵥ⋅Rᵥ⁻¹⋅[L₁⋅D₁⋅R₁⋅P₁ᵀ]⋅Lᵤᵀ

    return nothing
end


@doc raw"""
    inv_invUpV(G::AbstractMatrix, Fᵤ::LDR, Fᵥ::LDR;
               F::LDR=ldr(Fᵤ),
               dᵤ_min::AbstractVector=similar(Fᵤ.d), inv_dᵤ_max::AbstractVector=similar(Fᵤ.d),
               dᵥ_min::AbstractVector=similar(Fᵥ.d), inv_dᵥ_max::AbstractVector=similar(Fᵥ.d))

Calculate the numerically stable inverse ``G = (U^{-1} + V)^{-1},`` where the matrices ``U`` and
``V`` are represented by the LDR factorizations `Fᵤ` and `Fᵥ` respectively.

# Algorithm

Letting ``U = [L_u D_u R_u] P_u^T`` and ``V = [L_v D_v R_v] P_v^T,`` the inverse matrix
``G = (U^{-1} + V)^{-1}`` is calculated using the procedure
```math
\begin{align*}
G = & \left(U^{-1}+V\right)^{-1}\\
  = & ([\stackrel{D_{u,\max}D_{u,\min}}{L_{u}\overbrace{D_{u}}R_{u}}P_{u}^{T}]^{-1}+\overset{D_{v,\min}D_{v,\max}}{L_{v}\overbrace{D_{v}}R_{v}}P_{v}^{T})^{-1}\\
  = & \left(\left[L_{u}D_{u,\max}D_{u,\min}R_{u}P_{u}^{T}\right]^{-1}+L_{v}D_{v,\min}D_{v,\max}R_{v}P_{v}^{T}\right)^{-1}\\
  = & \left(P_{u}R_{u}^{-1}D_{u,\min}^{-1}D_{u,\max}^{-1}L_{u}^{\dagger}+L_{v}D_{v,\min}D_{v,\max}R_{v}P_{v}^{T}\right)^{-1}\\
  = & \left(P_{u}R_{u}^{-1}D_{u,\min}^{-1}\left[D_{u,\max}^{-1}L_{u}^{\dagger}P_{v}R_{v}^{-1}D_{v,\max}^{-1}+D_{u,\min}R_{u}P_{u}^{T}L_{v}D_{v,\min}\right]D_{v,\max}R_{v}P_{v}^{T}\right)^{-1}\\
  = & P_{v}R_{v}^{-1}D_{v,\max}^{-1}(\overset{L_{0}D_{0}R_{0}P_{0}^{T}}{\overbrace{D_{u,\max}^{-1}L_{u}^{\dagger}P_{v}R_{v}^{-1}D_{v,\max}^{-1}+D_{u,\min}R_{u}P_{u}^{T}L_{v}D_{v,\min}}})^{-1}D_{u,\min}R_{u}P_{u}^{T}\\
  = & P_{v}R_{v}^{-1}D_{v,\max}^{-1}\left(L_{0}D_{0}R_{0}P_{0}^{T}\right)^{-1}D_{u,\min}R_{u}P_{u}^{T}\\
  = & P_{v}R_{v}^{-1}\overset{L_{1}D_{1}R_{1}P_{1}^{T}}{\overbrace{D_{v,\max}^{-1}P_{0}R_{0}^{-1}D_{0}^{-1}L_{0}^{\dagger}D_{u,\min}}}R_{u}P_{u}^{T}\\
  = & P_{v}R_{v}^{-1}L_{1}D_{1}R_{1}P_{1}^{T}R_{u}P_{u}^{T},
\end{align*}
```
where ``D_\textrm{min} = \min(D,1)`` and ``D_\textrm{max} = \max(D,1).``
"""
function inv_invUpV!(G::AbstractMatrix{T}, Fᵤ::LDR{T}, Fᵥ::LDR{T};
                     F::LDR{T}=ldr(Fᵤ),
                     dᵤ_min::AbstractVector{T}=similar(Fᵤ.d),
                     inv_dᵤ_max::AbstractVector{T}=similar(Fᵤ.d),
                     dᵥ_min::AbstractVector{T}=similar(Fᵥ.d),
                     inv_dᵥ_max::AbstractVector{T}=similar(Fᵥ.d)) where {T}

    # calculate Dᵤ₋ = min(Dᵤ,1) and Dᵤ₊⁻¹ = [max(Dᵤ,1)]⁻¹
    @inbounds @fastmath for i in eachindex(Fᵤ.d)
        if abs(Fᵤ.d[i]) > 1
            dᵤ_min[i]     = 1
            inv_dᵤ_max[i] = 1/Fᵤ.d[i]
        else
            dᵤ_min[i]     = Fᵤ.d[i]
            inv_dᵤ_max[i] = 1
        end
    end

    # calculate Dᵥ₋ = min(Dᵥ,1) and Dᵥ₊⁻¹ = [max(Dᵥ,1)]⁻¹
    @inbounds @fastmath for i in eachindex(Fᵥ.d)
        if abs(Fᵥ.d[i]) > 1
            dᵥ_min[i]     = 1
            inv_dᵥ_max[i] = 1/Fᵥ.d[i]
        else
            dᵥ_min[i]     = Fᵥ.d[i]
            inv_dᵥ_max[i] = 1
        end
    end

    # calculate Dᵤ₊⁻¹⋅Lᵤᵀ⋅Pᵥ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹
    pᵥ = Fᵥ.p_tmp
    inv_P!(pᵥ, Fᵥ.pᵀ)
    Lᵤᵀ = Fᵤ.M_tmp
    adjoint!(Lᵤᵀ, Fᵤ.L)
    Rᵥ = UpperTriangular(Fᵥ.R)
    M = F.M_tmp
    copyto!(F.L, I) # I
    lmul_D!(inv_dᵥ_max, F.L) # Dᵥ₊⁻¹
    ldiv!(Rᵥ, F.L) # Rᵥ⁻¹⋅Dᵥ₊⁻¹
    mul!(M, pᵥ, F.L) # Pᵥ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹
    mul!(F.L, Lᵤᵀ, M) # Lᵤᵀ⋅Pᵥ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹
    lmul_D!(inv_dᵤ_max, F.L) # Dᵤ₊⁻¹⋅Lᵤᵀ⋅Pᵥ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹

    # calculate [L₀⋅D₀⋅R₀⋅P₀ᵀ] = [Dᵤ₊⁻¹⋅Lᵤᵀ⋅Pᵥ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹]
    ldr!(F)

    # calculate Dᵥ₊⁻¹⋅[L₀⋅D₀⋅R₀⋅P₀ᵀ]⁻¹⋅Dᵤ₋
    inv!(M, F) # [L₀⋅D₀⋅R₀⋅P₀ᵀ]⁻¹
    mul_D!(F.L, dᵤ_min, M) # [L₀⋅D₀⋅R₀⋅P₀ᵀ]⁻¹⋅Dᵤ₋
    lmul_D!(inv_dᵥ_max, F.L) # Dᵥ₊⁻¹⋅[L₀⋅D₀⋅R₀⋅P₀ᵀ]⁻¹⋅Dᵤ₋

    # calculate [L₁⋅D₁⋅R₁⋅P₁ᵀ] = Dᵥ₊⁻¹⋅[L₀⋅D₀⋅R₀⋅P₀ᵀ]⁻¹⋅Dᵤ₋
    ldr!(F)

    # calculate Pᵥ⋅Rᵥ⁻¹⋅[L₁⋅D₁⋅R₁⋅P₁ᵀ]⋅Rᵤ⋅Pᵤᵀ
    Rᵤ = UpperTriangular(Fᵤ.R)
    copyto!(G, F) # [L₁⋅D₁⋅R₁⋅P₁ᵀ]
    rmul!(G, Rᵤ) # [L₁⋅D₁⋅R₁⋅P₁ᵀ]⋅Rᵤ
    mul_P!(M, G, Fᵤ.pᵀ) # [L₁⋅D₁⋅R₁⋅P₁ᵀ]⋅Rᵤ⋅Pᵤᵀ
    ldiv!(Rᵥ, M) # Rᵥ⁻¹⋅[L₁⋅D₁⋅R₁⋅P₁ᵀ]⋅Rᵤ⋅Pᵤᵀ
    mul_P!(G, pᵥ, M) # G = Pᵥ⋅Rᵥ⁻¹⋅[L₁⋅D₁⋅R₁⋅P₁ᵀ]⋅Rᵤ⋅Pᵤᵀ

    return nothing
end

"""
    sign_det(F::LDR)

Returns the sign/phase factor of the determinant for a matrix represented by the
LDR factization `F`, which is calculated as the product of the diagonal matrix
elements of `F.R`.
"""
function sign_det(F::LDR{T}) where {T}

    sgn::T = 0
    # calculate the product of diagonal elements of R matrix
    for i in eachindex(F.d)
        rᵢ  = F.R[i,i]/abs(F.R[i,i])
        sgn = sgn * rᵢ
    end
    sgn = sgn/abs(sgn)

    return sgn
end


@doc raw"""
    abs_det(F::LDR; as_log::Bool=false)

Calculate the absolute value of determinant of the LDR factorization `F`.
If `as_log=true`, then the log of the absolute value of the determinant is
returned instead.

# Algorithm

Given an LDR factorization ``[L D R]P^T,`` calculate the absolute value of the determinant as
```math
\exp\left\{ \sum_i \log(D[i]) \right\},
```
where ``D`` is a diagonal matrix with strictly positive real matrix elements.
"""
function abs_det(F::LDR; as_log::Bool=false)

    # calculate log(|det(A)|)
    absdet = 0.0
    for i in eachindex(F.d)
        absdet += log(F.d[i])
    end

    # |det(A)|
    if !as_log
        absdet = exp(absdet)
    end

    return absdet
end


@doc raw"""
    abs_det_ratio(F₂::LDR, F₁::LDR, as_log::Bool=false)

Given two matrices ``A_2`` and ``A_1`` represented by the LDR factorizations
`F₂` and `F₁` respectively, calculate the absolute value of the determinant ratio
``\vert\det(A_2/A_1)\vert`` in a numerically stable fashion. If `as_log=true`, then
this function instead returns ``\log \left( \vert\det(A_2/A_1)\vert \right).``

# Algorithm

Let ``A_1 = [L_1 D_1 R_1] P_1^T`` and ``A_2 = [L_2 D_2 R_2] P_1^T`` be ``N \times N``
square matrices each represented by their respective LDR factorizations.
Let us define perumations ``p_1^{(1)} \dots p_1^{(i)} \dots p_1^{(N)}`` and 
``p_2^{(1)} \dots p_2^{(i)} \dots p_2^{(N)}`` that sort the diagonal elements
of ``D_1`` and ``D_2`` from smallest to largest. Then a numerically stable expression
for evaulating the absolute value of the determinant ratio is
```math
\vert \det(A_2/A_1) \vert = \exp\left\{ \sum_i \left( \log(D_2[p_2^{(i)}])
- \log(D_1[p_1^{(i)}]) \right) \right\},
```
keeping in mind that the diagonal elements of ``D_1`` and ``D_2`` are stictly
positive real numbers.
"""
function abs_det_ratio(F₂::LDR{T}, F₁::LDR{T}; as_log::Bool=false) where {T}

    @assert size(F′) == size(F)
    p₁ = F₁.p_tmp
    d₁ = F₁.d
    p₂ = F₂.p_tmp
    d₂ = F₂.d

    # sort the "pseudo-eigenvalues" from smallest to largest
    sortperm!(p₁, d₁)
    sortperm!(p₂, d₂)

    # calculat the log(|det(A₂/A₁)|) = log(|det(A₂)|) - log(|det(A₁)|)
    lndetR = 0.0
    for i in eachindex(p)
        lndetR = log(d₂[p₂[i]]) - log(d₁[p₁[i]])
    end

    if as_log
        R = lndetR
    else
        # calculate |det(A₂/A₁)|
        R = exp(lndetR)
    end

    return R
end