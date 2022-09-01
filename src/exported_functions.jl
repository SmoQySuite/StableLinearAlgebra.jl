@doc raw"""
    inv!(A⁻¹::AbstractMatrix{T}, F::LDR{T}, ws::LDRWorkspace{T}) where {T}
    inv!(A⁻¹::AbstractMatrix{T}, F::LDR{T};
         M::AbstractMatrix{T}=similar(F.L),
         p::AbstractVector{Int}=similar(F.pᵀ)) where {T}

Calculate the inverse of a matrix ``A`` represented of the [`LDR`](@ref) decomposition `F`,
writing the inverse matrix `A⁻¹`.
"""
function inv!(A⁻¹::AbstractMatrix{T}, F::LDR{T}, ws::LDRWorkspace{T}) where {T}

    inv!(A⁻¹, F, M=ws.M, p=ws.p)
    return nothing
end

function inv!(A⁻¹::AbstractMatrix{T}, F::LDR{T};
              M::AbstractMatrix{T}=similar(F.L),
              p::AbstractVector{Int}=similar(F.pᵀ)) where {T}

    # calculate inverse A⁻¹ = P⋅R⁻¹⋅D⁻¹⋅Lᵀ
    L   = F.L
    d   = F.d
    R   = UpperTriangular(F.R)
    inv_P!(p, F.pᵀ)
    adjoint!(M, L) # A⁻¹ = Lᵀ
    ldiv_D!(d, M) # A⁻¹ = D⁻¹⋅Lᵀ
    ldiv!(R, M) # A⁻¹ = R⁻¹⋅D⁻¹⋅Lᵀ
    mul_P!(A⁻¹, p, M) # A⁻¹ = P⋅R⁻¹⋅D⁻¹⋅Lᵀ

    return nothing
end


@doc raw"""
    inv!(F::LDR{T}, ws::LDRWorkspace{T}) where {T}
    inv!(F::LDR{T};
         M::AbstractMatrix{T},
         p::AbstractVector{Int}) where {T}

Invert the [`LDR`](@ref) decomposition `F` in-place.
"""
function inv!(F::LDR{T}, ws::LDRWorkspace{T}) where {T}

    inv!(F, M=ws.M, p=ws.p)
    return nothing
end

function inv!(F::LDR{T};
              M::AbstractMatrix{T},
              p::AbstractVector{Int}) where {T}

    # given F = [L⋅D⋅R⋅Pᵀ], calculate F⁻¹ = [L⋅D⋅R⋅Pᵀ]⁻¹ = P⋅R⁻¹⋅D⁻¹⋅Lᵀ in-place
    Lᵀ = M
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
    inv_IpA!(G::AbstractMatrix{T}, F::LDR{T}, ws::LDRWorkspace{T};
             F′::LDR{T}=ldr(F)) where {T}
    inv_IpA!(G::AbstractMatrix{T}, F::LDR{T};
             F′::LDR{T}=ldr(F),
             d_min::AbstractVector{T}=similar(F.d),
             d_max::AbstractVector{T}=similar(F.d),
             M::AbstractMatrix{T}=similar(F.L),
             p::AbstractVector{Int}=similar(F.pᵀ)) where {T}

Given a matrix ``A`` represented by the [`LDR`](@ref) factorization `F`, calculate the numerically stabalized inverse
```math
G = (I + A)^{-1},
```
storing the result in the matrix `G`.

# Algorithm

Given an [`LDR`](@ref) factorization of ``A``, calculate ``G = (I + A)^{-1}`` using the procedure
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
function inv_IpA!(G::AbstractMatrix{T}, F::LDR{T}, ws::LDRWorkspace{T};
                  F′::LDR{T}=ldr(F)) where {T}

    inv_IpA!(G, F, F′=F′, d_min=ws.v, d_max=ws.v′, M=ws.M, p=ws.p)
    return nothing
end

function inv_IpA!(G::AbstractMatrix{T}, F::LDR{T};
                  F′::LDR{T}=ldr(F),
                  d_min::AbstractVector{T}=similar(F.d),
                  d_max::AbstractVector{T}=similar(F.d),
                  M::AbstractMatrix{T}=similar(F.L),
                  p::AbstractVector{Int}=similar(F.pᵀ)) where {T}

    # construct Dmin = min(D,1) and Dmax⁻¹ = [max(D,1)]⁻¹ matrices
    @. d_min = min(F.d, 1)
    @. d_max = max(F.d, 1)

    # define the original P₀, R₀⁻¹
    p₀ = p
    inv_P!(p₀, F.pᵀ)
    R₀ = UpperTriangular(F.R)

    # caclulate L₀⋅Dmin
    mul_D!(F′.L, F.L, d_min)

    # calculate P₀⋅R₀⁻¹⋅Dmax⁻¹
    copyto!(G, I)
    ldiv_D!(d_max, G) # Dmax⁻¹
    ldiv!(R₀, G) # R₀⁻¹⋅Dmax⁻¹
    mul_P!(M, p₀, G) # P₀⋅R₀⁻¹⋅Dmax⁻¹

    # calculate LDR decomposition of L⋅D⋅R⋅Pᵀ = [P₀⋅R₀⁻¹⋅Dmax⁻¹ + L₀⋅Dmin]
    @. F′.L = F′.L + M
    ldr!(F′)

    # invert the LDR decomposition, [L⋅D⋅R⋅Pᵀ]⁻¹ = P⋅R⁻¹⋅D⁻¹⋅Lᵀ
    inv!(G, F′)

    # Dmax⁻¹⋅[P⋅R⁻¹⋅D⁻¹⋅Lᵀ]
    div_D!(F′.L, d_max, G)

    # calculate LDR of Dmax⁻¹⋅[P⋅R⁻¹⋅D⁻¹⋅Lᵀ]
    ldr!(F′)

    # G = P₀⋅R₀⁻¹⋅F
    copyto!(M, F′)
    ldiv!(R₀, M)
    mul_P!(G, p₀, M)

    return nothing
end


@doc raw"""
    inv_UpV!(G::AbstractMatrix{T}, Fᵤ::LDR{T}, Fᵥ::LDR{T}, ws::LDRWorkspace{T};
             F::LDR{T}=ldr(Fᵥ)) where {T}
    inv_UpV!(G::AbstractMatrix{T}, Fᵤ::LDR{T}, Fᵥ::LDR{T};
             F::LDR{T}=ldr(Fᵤ),
             dᵤ_min::AbstractVector{T}=similar(Fᵤ.d),
             dᵤ_max::AbstractVector{T}=similar(Fᵤ.d),
             dᵥ_min::AbstractVector{T}=similar(Fᵥ.d),
             dᵥ_max::AbstractVector{T}=similar(Fᵥ.d),
             M::AbstractMatrix{T}=similar(Fᵥ.L),
             M′::AbstractMatrix{T}=similar(Fᵥ.L),
             M″::AbstractMatrix{T}=similar(Fᵥ.L),
             p::AbstractVector{Int}=similar(Fᵥ.pᵀ),
             p′::AbstractVector{Int}=similar(Fᵥ.pᵀ)) where {T}

Calculate the numerically stable inverse ``G = (U + V)^{-1},`` where the matrices ``U`` and
``V`` are represented by the [`LDR`](@ref) factorizations `Fᵤ` and `Fᵥ` respectively.

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
function inv_UpV!(G::AbstractMatrix{T}, Fᵤ::LDR{T}, Fᵥ::LDR{T}, ws::LDRWorkspace{T};
                  F::LDR{T}=ldr(Fᵥ)) where {T}

    inv_UpV!(G, Fᵤ, Fᵥ, F=F, dᵤ_min=ws.v, dᵤ_max=ws.v′, dᵥ_min=ws.v″, dᵥ_max=ws.v‴,
             M=ws.M, M′=ws.M′, M″=ws.M″, p=ws.p, p′=ws.p′)
    return nothing
end

function inv_UpV!(G::AbstractMatrix{T}, Fᵤ::LDR{T}, Fᵥ::LDR{T};
                  F::LDR{T}=ldr(Fᵤ),
                  dᵤ_min::AbstractVector{T}=similar(Fᵤ.d),
                  dᵤ_max::AbstractVector{T}=similar(Fᵤ.d),
                  dᵥ_min::AbstractVector{T}=similar(Fᵥ.d),
                  dᵥ_max::AbstractVector{T}=similar(Fᵥ.d),
                  M::AbstractMatrix{T}=similar(Fᵥ.L),
                  M′::AbstractMatrix{T}=similar(Fᵥ.L),
                  M″::AbstractMatrix{T}=similar(Fᵥ.L),
                  p::AbstractVector{Int}=similar(Fᵥ.pᵀ),
                  p′::AbstractVector{Int}=similar(Fᵥ.pᵀ)) where {T}

    # calculate Dᵤ₋ = min(Dᵤ,1) and Dᵤ₊⁻¹ = [max(Dᵤ,1)]⁻¹
    @. dᵤ_min = min(Fᵤ.d, 1)
    @. dᵤ_max = max(Fᵤ.d, 1)
    @. dᵥ_min = min(Fᵥ.d, 1)
    @. dᵥ_max = max(Fᵥ.d, 1)

    # calculate Rᵤ⋅Pᵤᵀ⋅Pᵥ⋅Rᵥ⁻¹
    pᵥ = p
    Rᵥ = UpperTriangular(Fᵥ.R)
    Rᵤ = UpperTriangular(Fᵤ.R)
    inv_P!(pᵥ, Fᵥ.pᵀ)
    copyto!(F.L, I) # I
    ldiv!(Rᵥ, F.L) # Rᵥ⁻¹
    mul_P!(M, pᵥ, F.L) # Pᵥ⋅Rᵥ⁻¹
    mul_P!(F.L, Fᵤ.pᵀ, M) # Pᵤᵀ⋅Pᵥ⋅Rᵥ⁻¹
    lmul!(Rᵤ, F.L) # Rᵤ⋅Pᵤᵀ⋅Pᵥ⋅Rᵥ⁻¹

    # calculate Dᵤ₋⋅[Rᵤ⋅Pᵤᵀ⋅Pᵥ⋅Rᵥ⁻¹]⋅Dᵥ₊⁻¹
    @fastmath @inbounds for i in eachindex(dᵥ_max)
        for j in eachindex(dᵤ_min)
            F.L[j,i] *= dᵤ_min[j]/dᵥ_max[i]
        end
    end

    # calcualte Lᵤᵀ⋅Lᵥ
    Lᵤᵀ = M
    adjoint!(Lᵤᵀ, Fᵤ.L)
    mul!(F.R, Lᵤᵀ, Fᵥ.L)

    # calculate Dᵤ₊⁻¹⋅[Lᵤᵀ⋅Lᵥ]⋅Dᵥ₋
    @fastmath @inbounds for i in eachindex(dᵥ_min)
        for j in eachindex(dᵤ_max)
            F.R[j,i] *= dᵥ_min[i]/dᵤ_max[j]
        end
    end

    # calculate Dᵤ₋⋅Rᵤ⋅Pᵤᵀ⋅Pᵥ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹ + Dᵤ₊⁻¹⋅Lᵤᵀ⋅Lᵥ⋅Dᵥ₋
    @. F.L = F.L + F.R

    # calculate [L₀⋅D₀⋅R₀]⋅P₀ᵀ = [Dᵤ₋⋅Rᵤ⋅Pᵤᵀ⋅Pᵥ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹ + Dᵤ₊⁻¹⋅Lᵤᵀ⋅Lᵥ⋅Dᵥ₋]
    ldr!(F)

    # calculate [L₀⋅D₀⋅R₀⋅P₀ᵀ]⁻¹
    inv!(M′, F, M=M″, p=p′)

    # calculate Dᵥ₊⁻¹⋅[L₀⋅D₀⋅R₀⋅P₀ᵀ]⁻¹⋅Dᵤ₊⁻¹
    @fastmath @inbounds for i in eachindex(dᵤ_max)
        for j in eachindex(dᵥ_max)
            F.L[j,i] = M′[j,i] / dᵤ_max[i] / dᵥ_max[j]
        end
    end

    # calculate [L₁⋅D₁⋅R₁]⋅P₁ᵀ = Dᵥ₊⁻¹⋅[L₀⋅D₀⋅R₀⋅P₀ᵀ]⁻¹⋅Dᵤ₊⁻¹
    ldr!(F)

    # calculate Pᵥ⋅Rᵥ⁻¹⋅[L₁⋅D₁⋅R₁⋅P₁ᵀ]⋅Lᵤᵀ
    mul!(M′, F, Lᵤᵀ) # [L₁⋅D₁⋅R₁⋅P₁ᵀ]⋅Lᵤᵀ
    ldiv!(Rᵥ, M′) # Rᵥ⁻¹⋅[L₁⋅D₁⋅R₁⋅P₁ᵀ]⋅Lᵤᵀ
    mul_P!(G, pᵥ, M′) # G = Pᵥ⋅Rᵥ⁻¹⋅[L₁⋅D₁⋅R₁⋅P₁ᵀ]⋅Lᵤᵀ

    return nothing
end


@doc raw"""
    inv_invUpV!(G::AbstractMatrix{T}, Fᵤ::LDR{T}, Fᵥ::LDR{T}, ws::LDRWorkspace{T};
                F::LDR{T}=ldr(F)) where {T}
    inv_invUpV!(G::AbstractMatrix{T}, Fᵤ::LDR{T}, Fᵥ::LDR{T};
                F::LDR{T}=ldr(Fᵤ),
                dᵤ_min::AbstractVector{T}=similar(Fᵤ.d),
                dᵤ_max::AbstractVector{T}=similar(Fᵤ.d),
                dᵥ_min::AbstractVector{T}=similar(Fᵥ.d),
                dᵥ_max::AbstractVector{T}=similar(Fᵥ.d),
                M::AbstractMatrix{T}=similar(Fᵥ.L),
                M′::AbstractMatrix{T}=similar(Fᵥ.L),
                p::AbstractVector{Int}=similar(Fᵥ.pᵀ),
                p′::AbstractVector{Int}=similar(Fᵥ.pᵀ)) where {T}

Calculate the numerically stable inverse ``G = (U^{-1} + V)^{-1},`` where the matrices ``U`` and
``V`` are represented by the [`LDR`](@ref) factorizations `Fᵤ` and `Fᵥ` respectively.

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
function inv_invUpV!(G::AbstractMatrix{T}, Fᵤ::LDR{T}, Fᵥ::LDR{T}, ws::LDRWorkspace{T};
                     F::LDR{T}=ldr(F)) where {T}

    inv_invUpV!(G, Fᵤ, Fᵥ, F=F, dᵤ_min=ws.v, dᵤ_max=ws.v′, dᵥ_min=ws.v″, dᵥ_max=ws.v‴,
                M=ws.M, M′=ws.M′, p=ws.p, p′=ws.p′)
    return nothing
end

function inv_invUpV!(G::AbstractMatrix{T}, Fᵤ::LDR{T}, Fᵥ::LDR{T};
                     F::LDR{T}=ldr(Fᵤ),
                     dᵤ_min::AbstractVector{T}=similar(Fᵤ.d),
                     dᵤ_max::AbstractVector{T}=similar(Fᵤ.d),
                     dᵥ_min::AbstractVector{T}=similar(Fᵥ.d),
                     dᵥ_max::AbstractVector{T}=similar(Fᵥ.d),
                     M::AbstractMatrix{T}=similar(Fᵥ.L),
                     M′::AbstractMatrix{T}=similar(Fᵥ.L),
                     p::AbstractVector{Int}=similar(Fᵥ.pᵀ),
                     p′::AbstractVector{Int}=similar(Fᵥ.pᵀ)) where {T}

    # calculate Dᵤ₋ = min(Dᵤ,1) and Dᵤ₊⁻¹ = [max(Dᵤ,1)]⁻¹
    @. dᵤ_min = min(Fᵤ.d, 1)
    @. dᵤ_max = max(Fᵤ.d, 1)
    @. dᵥ_min = min(Fᵥ.d, 1)
    @. dᵥ_max = max(Fᵥ.d, 1)

    # calculate Lᵤᵀ⋅Pᵥ⋅Rᵥ⁻¹
    pᵥ = p
    inv_P!(pᵥ, Fᵥ.pᵀ)
    Lᵤᵀ = M
    adjoint!(Lᵤᵀ, Fᵤ.L)
    Rᵥ = UpperTriangular(Fᵥ.R)
    copyto!(F.L, I) # I
    ldiv!(Rᵥ, F.L) # Rᵥ⁻¹
    mul_P!(M′, pᵥ, F.L) # Pᵥ⋅Rᵥ⁻¹
    mul!(F.L, Lᵤᵀ, M′) # Lᵤᵀ⋅Pᵥ⋅Rᵥ⁻¹

    # calculate Dᵤ₊⁻¹⋅[Lᵤᵀ⋅Pᵥ⋅Rᵥ⁻¹]⋅Dᵥ₊⁻¹
    @fastmath @inbounds for i in eachindex(dᵥ_max)
        for j in eachindex(dᵤ_max)
            F.L[j,i] = F.L[j,i] / dᵤ_max[j] / dᵥ_max[i]
        end
    end

    # calculate Rᵤ⋅Pᵤᵀ⋅Lᵥ
    Rᵤ = UpperTriangular(Fᵤ.R)
    copyto!(F.R, Fᵥ.L) # Lᵥ
    mul_P!(M′, Fᵤ.pᵀ, F.R) # Pᵤᵀ⋅Lᵥ
    mul!(F.R, Rᵤ, M′) # Rᵤ⋅Pᵤᵀ⋅Lᵥ

    # calculate Dᵤ₋⋅[Rᵤ⋅Pᵤᵀ⋅Lᵥ]⋅Dᵥ₋
    @fastmath @inbounds for i in eachindex(dᵥ_min)
        for j in eachindex(dᵤ_min)
            F.R[j,i] = F.R[j,i] * dᵤ_min[j] * dᵥ_min[i]
        end
    end

    # calculate Dᵤ₊⁻¹⋅Lᵤᵀ⋅Pᵥ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹ + Dᵤ₋⋅Rᵤ⋅Pᵤᵀ⋅Lᵥ⋅Dᵥ₋
    @. F.L = F.L + F.R

    # calculate [L₀⋅D₀⋅R₀⋅P₀ᵀ] = [Dᵤ₊⁻¹⋅Lᵤᵀ⋅Pᵥ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹ + Dᵤ₋⋅Rᵤ⋅Pᵤᵀ⋅Lᵥ⋅Dᵥ₋]
    ldr!(F)

    # calculate [L₀⋅D₀⋅R₀⋅P₀ᵀ]⁻¹⋅
    inv!(M, F, M=M′, p=p′)

    # calculate Dᵥ₊⁻¹⋅[L₀⋅D₀⋅R₀⋅P₀ᵀ]⁻¹⋅Dᵤ₋
    @fastmath @inbounds for i in eachindex(dᵤ_min)
        for j in eachindex(dᵥ_max)
            F.L[j,i] = M[j,i] * dᵤ_min[i] / dᵥ_max[j]
        end
    end

    # calculate [L₁⋅D₁⋅R₁⋅P₁ᵀ] = Dᵥ₊⁻¹⋅[L₀⋅D₀⋅R₀⋅P₀ᵀ]⁻¹⋅Dᵤ₋
    ldr!(F)

    # calculate Pᵥ⋅Rᵥ⁻¹⋅[L₁⋅D₁⋅R₁⋅P₁ᵀ]⋅Rᵤ⋅Pᵤᵀ
    copyto!(G, F) # [L₁⋅D₁⋅R₁⋅P₁ᵀ]
    rmul!(G, Rᵤ) # [L₁⋅D₁⋅R₁⋅P₁ᵀ]⋅Rᵤ
    mul_P!(M, G, Fᵤ.pᵀ) # [L₁⋅D₁⋅R₁⋅P₁ᵀ]⋅Rᵤ⋅Pᵤᵀ
    ldiv!(Rᵥ, M) # Rᵥ⁻¹⋅[L₁⋅D₁⋅R₁⋅P₁ᵀ]⋅Rᵤ⋅Pᵤᵀ
    mul_P!(G, pᵥ, M) # G = Pᵥ⋅Rᵥ⁻¹⋅[L₁⋅D₁⋅R₁⋅P₁ᵀ]⋅Rᵤ⋅Pᵤᵀ

    return nothing
end


@doc raw"""
    sign_det(F::LDR{T}, ws::LDRWorkspace{T}) where {T}
    sign_det(F::LDR{T}) where {T<:Real}
    sign_det(F::LDR{T}; M::AbstractMatrix{T}=similar(F.L)) where {T<:Complex}

Returns the sign/phase factor of the determinant for a matrix ``A`` represented by the
[`LDR`](@ref) factorization `F`, which is calculated as
```math
\textrm{sgn}(\det A) = \det L \cdot \left( \prod_i R_{i,i} \right) \cdot \det P^T,
```
where ``A = [L D R]P^T.``
"""
function sign_det(F::LDR{T}, ws::LDRWorkspace{T}) where {T<:Real}

    return sign_det(F)
end

function sign_det(F::LDR{T}) where {T<:Real}

    sgn::T = 1
    # calculate the product of diagonal elements of R matrix
    for i in eachindex(F.d)
        rᵢ  = F.R[i,i]
        sgn = sgn * rᵢ
    end
    # account for sign of L
    sgn = -sgn
    # multiply by det(Pᵀ) <==> sign/parity of pᵀ
    sgn = sgn * sign_P(F.pᵀ)
    # normalize
    sgn = sgn/abs(sgn)

    return sgn
end

function sign_det(F::LDR{T}, ws::LDRWorkspace{T}) where {T<:Complex}

    return sign_det(F, M=ws.M)
end

function sign_det(F::LDR{T};
                  M::AbstractMatrix{T}=similar(F.L)) where {T<:Complex}

    sgn::T = 1
    # calculate the product of diagonal elements of R matrix
    for i in eachindex(F.d)
        rᵢ  = F.R[i,i]
        sgn = sgn * rᵢ
    end
    # account of det(L) phase
    copyto!(M,F.L)
    sgn = sgn * det(LinearAlgebra.lu!(M))
    # multiply by det(Pᵀ) <==> sign/parity of pᵀ
    sgn = sgn * sign_P(F.pᵀ)
    # normalize
    sgn = sgn/abs(sgn)

    return sgn
end


@doc raw"""
    abs_det(F::LDR{T}; as_log::Bool=false) where {T}

Calculate the absolute value of determinant of the [`LDR`](@ref) factorization `F`.
If `as_log=true`, then the log of the absolute value of the determinant is
returned instead.

# Algorithm

Given an [`LDR`](@ref) factorization ``[L D R]P^T,`` calculate the absolute value of the determinant as
```math
\exp\left\{ \sum_i \log(D[i]) \right\},
```
where ``D`` is a diagonal matrix with strictly positive real matrix elements.
"""
function abs_det(F::LDR{T}; as_log::Bool=false) where {T}

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
    abs_det_ratio(F₂::LDR{T}, F₁::LDR{T}, ws::LDRWorkspace{T};
                  as_log::Bool=false) where {T}
    abs_det_ratio(F₂::LDR{T}, F₁::LDR{T};
                  as_log::Bool=false,
                  p::AbstractVector{Int}=similar(F₁.pᵀ),
                  p′::AbstractVector{Int}=similar(F₁.pᵀ)) where {T}

Given two matrices ``A_2`` and ``A_1`` represented by the [`LDR`](@ref) factorizations
`F₂` and `F₁` respectively, calculate the absolute value of the determinant ratio
``\vert\det(A_2/A_1)\vert`` in a numerically stable fashion. If `as_log=true`, then
this function instead returns ``\log \left( \vert\det(A_2/A_1)\vert \right).``

# Algorithm

Let ``A_1 = [L_1 D_1 R_1] P_1^T`` and ``A_2 = [L_2 D_2 R_2] P_1^T`` be ``N \times N``
square matrices each represented by their respective [`LDR`](@ref) factorizations.
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
function abs_det_ratio(F₂::LDR{T}, F₁::LDR{T}, ws::LDRWorkspace{T};
                       as_log::Bool=false) where {T}

    return abs_det_ratio(F₂, F₁, p=ws.p, p′=ws.p′, as_log=as_log)
end

function abs_det_ratio(F₂::LDR{T}, F₁::LDR{T};
                       as_log::Bool=false,
                       p::AbstractVector{Int}=similar(F₁.pᵀ),
                       p′::AbstractVector{Int}=similar(F₁.pᵀ)) where {T}

    @assert size(F₂) == size(F₁)

    # sort diagonal matrix elements of D₁
    sortperm!(p, F₁.d)
    d₁ = @view F₁.d[p]

    # sort diagonal matrix elements of D₂
    sortperm!(p′, F₂.d)
    d₂ = @view F₂.d[p′]

    # calculat the log(|det(A₂/A₁)|) = log(|det(A₂)|) - log(|det(A₁)|)
    lndetR = 0.0
    for i in eachindex(d₁)
        lndetR += log(d₂[i]) - log(d₁[i])
    end

    if as_log
        R = lndetR
    else
        # calculate |det(A₂/A₁)|
        R = exp(lndetR)
    end

    return R
end