# @doc raw"""
#     adjoint_inv_ldr!(A⁻ᵀ::LDR{T}, A::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}

# Given a matrix ``A,`` calculate the [`LDR`](@ref) factorization ``(A^{-1})^{\dagger},``
# storing the result in `A⁻ᵀ`.

# # Algorithm

# By calculating the pivoted QR factorization of `A`, we can calculate the [`LDR`](@ref)
# facorization `(A^{-1})^{\dagger} = L_0 D_0 R_0` using the procedure

# ```math
# \begin{align*}
# (A^{-1})^{\dagger}:= & ([QRP^{T}]^{-1})^{\dagger}\\
# = & ([Q\,\textrm{|diag}(R)|\,|\textrm{diag}(R)|^{-1}\,RP^{T}]^{-1})^{\dagger}\\
# = & [\overset{R_{0}^{\dagger}}{\overbrace{PR^{-1}|\textrm{diag}(R)}|}\,\overset{D_{0}}{\overbrace{|\textrm{diag}(R)|^{-1}}}\,\overset{L_{0}^{\dagger}}{\overbrace{Q^{\dagger}}}]^{\dagger}\\
# = & [R_{0}^{\dagger}D_{0}L_{0}^{\dagger}]^{\dagger}\\
# = & L_{0}D_{0}R_{0}.
# \end{align*}
# ```
# """
# function adjoint_inv_ldr!(A⁻ᵀ::LDR{T}, A::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}

#     # calculate A = Q⋅R⋅Pᵀ
#     copyto!(A⁻ᵀ.L, A)
#     LAPACK.geqp3!(A⁻ᵀ.L, ws.qr_ws)
#     copyto!(A⁻ᵀ.R, A⁻ᵀ.L)
#     triu!(A⁻ᵀ.R) # R (set lower triangular to 0)
#     pᵀ = ws.qr_ws.jpvt
#     p  = ws.lu_ws.ipiv
#     inv_P!(p, pᵀ) # P
#     LAPACK.orgqr!(Aᵀ.L, ws.qr_ws) # L₀ = Q
#     @fastmath @inbounds for i in eachindex(A⁻ᵀ.d)
#         A⁻ᵀ.d[i] = inv(abs(A⁻ᵀ.R[i,i])) # D₀ = |diag(R)|⁻¹
#     end
#     R′ = A⁻ᵀ.R
#     lmul_D!(A⁻ᵀ.d, R′) # R′ := |diag(R)|⁻¹⋅R
#     R″ = R′
#     LinearAlgebra.inv!(UpperTriangular(R″)) # R″ = (R′)⁻¹ = R⁻¹⋅|diag(R)|
#     R₀ᵀ = ws.M
#     mul_P!(R₀ᵀ, p, R″) # R₀ᵀ = P⋅R″ = P⋅R⁻¹⋅diag(R)
#     adjoint(A⁻ᵀ.R, R₀ᵀ) # R₀ = [P⋅R⁻¹⋅diag(R)]ᵀ

#     return nothing
# end


@doc raw"""
    inv_IpA!(G::AbstractMatrix{T}, A::LDR{T}, ws::LDRWorkspace{T}) where {T}

Calculate the numerically stable inverse ``G := [I + A]^{-1},`` where ``G`` is a matrix,
and ``A`` is represented by a [`LDR`](@ref) factorization. This method also returns
``\log( \vert \det G\vert )`` and ``\textrm{sign}(\det G).``

# Algorithm

The numerically stable inverse ``G := [I + A]^{-1}`` is calculated using the procedure
```math
\begin{align*}
G:= & [I+A]^{-1}\\
= & [I+L_{a}D_{a}R_{a}]^{-1}\\
= & [I+L_{a}D_{a,\min}D_{a,\max}R_{a}]^{-1}\\
= & [(R_{a}^{-1}D_{a,\max}^{-1}+L_{a}D_{a,\min})D_{a,\max}R_{a}]^{-1}\\
= & R_{a}^{-1}D_{a,\max}^{-1}[\overset{M}{\overbrace{R_{a}^{-1}D_{a,\max}^{-1}+L_{a}D_{a,\min}}}]^{-1}\\
= & R_{a}^{-1}D_{a,\max}^{-1}M^{-1},
\end{align*}
```
where ``D_{a,\min} = \min(D_a, 1)`` and ``D_{a,\max} = \max(D_a, 1).``
Intermediate matrix inversions and relevant determinant calculations are performed
via LU factorizations with partial pivoting.
"""
function inv_IpA!(G::AbstractMatrix{T}, A::LDR{T}, ws::LDRWorkspace{T}) where {T}

    Lₐ = A.L
    dₐ = A.d
    Rₐ = A.R

    # calculate Rₐ⁻¹
    Rₐ⁻¹ = ws.M′
    copyto!(Rₐ⁻¹, Rₐ)
    logdetRₐ⁻¹, sgndetRₐ⁻¹ = inv_lu!(Rₐ⁻¹, ws.lu_ws)

    # calculate D₋ = min(Dₐ, 1)
    d₋ = ws.v
    @. d₋ = min(dₐ, 1)

    # calculate Lₐ⋅D₋
    mul_D!(ws.M, Lₐ, d₋)

    # calculate D₊ = max(Dₐ, 1)
    d₊ = ws.v
    @. d₊ = max(dₐ, 1)

    # calculate sign(det(D₊)) and log(|det(D₊)|)
    logdetD₊, sgndetD₊ = det_D(d₊)

    # calculate Rₐ⁻¹⋅D₊⁻¹
    Rₐ⁻¹D₊ = Rₐ⁻¹
    rdiv_D!(Rₐ⁻¹D₊, d₊)

    # calculate M = Rₐ⁻¹⋅D₊⁻¹ + Lₐ⋅D₋
    @. ws.M += Rₐ⁻¹D₊

    # calculate M⁻¹ = [Rₐ⁻¹⋅D₊⁻¹ + Lₐ⋅D₋]⁻¹
    M⁻¹ = ws.M
    logdetM⁻¹, sgndetM⁻¹ = inv_lu!(M⁻¹, ws.lu_ws)

    # calculate G = Rₐ⁻¹⋅D₊⁻¹⋅M⁻¹
    mul!(G, Rₐ⁻¹D₊, M⁻¹)

    # calculate sign(det(G)) and log(|det(G)|)
    sgndetG = sgndetRₐ⁻¹ * conj(sgndetD₊) * sgndetM⁻¹
    logdetG = -logdetD₊  + logdetM⁻¹

    return logdetG, sgndetG
end


@doc raw"""
    inv_IpUV!(G::AbstractMatrix{T}, U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

Calculate the numerically stable inverse ``G := [I + UV]^{-1},`` where ``G`` is a matrix and
``U`` and ``V`` are represented by [`LDR`](@ref) factorizations. This method also returns
``\log( \vert \det G\vert )`` and ``\textrm{sign}(\det G).``

# Algorithm

The numerically stable inverse ``G := [I + UV]^{-1}`` is calculated using the procedure
```math
\begin{align*}
G:= & [I+UV]^{-1}\\
= & [I+L_{u}D_{u}R_{u}L_{v}D_{v}R_{v}]^{-1}\\
= & R_{v}^{-1}[\overset{M}{\overbrace{L_{u}^{\dagger}R_{v}^{-1}+D_{u}R_{u}L_{v}D_{v}}}]^{-1}L_{u}^{\dagger}\\
= & R_{v}^{-1}M^{-1}L_{u}^{\dagger}.
\end{align*}
```
Intermediate matrix inversions and relevant determinant calculations are performed
via LU factorizations with partial pivoting.
"""
function inv_IpUV!(G::AbstractMatrix{T}, U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

    Lᵤ = U.L
    dᵤ = U.d
    Rᵤ = V.R
    Lᵥ = V.L
    dᵥ = V.d
    Rᵥ = V.R

    # calculate sign(det(Lᵤ)) and log(det(Lᵤ))
    copyto!(ws.M, Lᵤ)
    logdetLᵤ, sgndetLᵤ = det_lu!(ws.M, ws.lu_ws)

    # calculate Rᵥ⁻¹, sign(det(Rᵥ⁻¹)) and log(det(Rᵥ⁻¹))
    Rᵥ⁻¹ = ws.M′
    copyto!(Rᵥ⁻¹, Rᵥ)
    logdetRᵥ⁻¹, sgndetRᵥ⁻¹ = inv_lu!(Rᵥ⁻¹, ws.lu_ws)

    # represent Lᵤᵀ
    Lᵤᵀ = adjoint(Lᵤ)

    # calculate Dᵤ⋅Rᵤ⋅Lᵥ⋅Dᵥ
    mul!(ws.M, Rᵤ, Lᵥ) # Rᵤ⋅Lᵥ
    lmul_D!(dᵤ, ws.M) # Dᵤ⋅[Rᵤ⋅Lᵥ]
    rmul_D!(ws.M, dᵥ) # [Dᵤ⋅Rᵤ⋅Lᵥ]⋅Dᵥ

    # calculate Lᵤᵀ⋅Rᵥ⁻¹
    mul!(G, Lᵤᵀ, Rᵥ⁻¹)

    # calculate M = Lᵤᵀ⋅Rᵥ⁻¹ + Dᵤ⋅Rᵤ⋅Lᵥ⋅Dᵥ
    @. G += ws.M

    # calculate M⁻¹ = [Lᵤᵀ⋅Rᵥ⁻¹ + Dᵤ⋅Rᵤ⋅Lᵥ⋅Dᵥ]⁻¹
    M⁻¹ = G
    logdetM⁻¹, sgndetM⁻¹ = inv_lu!(M⁻¹, ws.lu_ws)

    # calculate G := Rᵥ⁻¹⋅M⁻¹⋅Lᵤᵀ
    mul!(ws.M, M⁻¹, Lᵤᵀ) # M⁻¹⋅Lᵤᵀ
    mul!(G, Rᵥ⁻¹, ws.M) # G := Rᵥ⁻¹⋅[M⁻¹⋅Lᵤᵀ]

    # calculate sign(det(G)) and log(|det(G)|)
    sgndetG = sgndetRᵥ⁻¹ * sgndetM⁻¹ * conj(sgndetLᵤ)
    logdetG = logdetM⁻¹

    return logdetG, sgndetG
end


@doc raw"""
    inv_UpV!(G::AbstractMatrix{T}, U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

Calculate the numerically stable inverse ``G := [U+V]^{-1},`` where ``G`` is a matrix and ``U``
and ``V`` are represented by [`LDR`](@ref) factorizations. This method also returns
``\log( \vert \det G\vert )`` and ``\textrm{sign}(\det G).``

# Algorithm

The numerically stable inverse ``G := [U+V]^{-1}`` is calculated using the procedure
```math
\begin{align*}
G:= & [U+V]^{-1}\\
= & [\overset{D_{u,\max}D_{u,\min}}{L_{u}\overbrace{D_{u}}R_{u}}+\overset{D_{v,\min}D_{v,\max}}{L_{v}\overbrace{D_{v}}R_{v}}]^{-1}\\
= & [L_{u}D_{u,\max}D_{u,\min}R_{u}+L_{v}D_{v,\min}D_{v,\max}R_{v}]^{-1}\\
= & [L_{u}D_{u,\max}(D_{u,\min}R_{u}R_{v}^{-1}D_{v,\max}^{-1}+D_{u,\max}^{-1}L_{u}^{\dagger}L_{v}D_{v,\min})D_{v,\max}R_{v}]^{-1}\\
= & R_{v}^{-1}D_{v,\max}^{-1}[\overset{M}{\overbrace{D_{u,\min}R_{u}R_{v}^{-1}D_{v,\max}^{-1}+D_{u,\max}^{-1}L_{u}^{\dagger}L_{v}D_{v,\min}}}]^{-1}D_{u,\max}^{-1}L_{u}^{\dagger}\\
= & R_{v}^{-1}D_{v,\max}^{-1}M^{-1}D_{u,\max}^{-1}L_{u}^{\dagger},
\end{align*}
```
where
```math
\begin{align*}
D_{u,\min} = & \min(D_u,1)\\
D_{u,\max} = & \max(D_u,1)\\
D_{v,\min} = & \min(D_v,1)\\
D_{v,\max} = & \max(D_v,1),
\end{align*}
```
and all intermediate matrix inversions and determinant calculations are performed via LU factorizations with partial pivoting.
"""
function inv_UpV!(G::AbstractMatrix{T}, U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

    Lᵤ = U.L
    dᵤ = U.d
    Rᵤ = U.R
    Lᵥ = V.L
    dᵥ = V.d
    Rᵥ = V.R

    # calculate sign(det(Lᵤ)) and log(|det(Lᵤ)|)
    copyto!(ws.M, Lᵤ)
    logdetLᵤ, sgndetLᵤ = det_lu!(ws.M, ws.lu_ws)

    # calculate Rᵥ⁻¹, sign(det(Rᵥ⁻¹)) and log(|det(Rᵥ⁻¹)|)
    Rᵥ⁻¹ = ws.M′
    copyto!(Rᵥ⁻¹, Rᵥ)
    logdetRᵥ⁻¹, sgndetRᵥ⁻¹ = inv_lu!(Rᵥ⁻¹, ws.lu_ws)

    # calcuate Dᵥ₊ = max(Dᵥ, 1)
    dᵥ₊ = ws.v
    @. dᵥ₊ = max(dᵥ, 1)

    # calculate sign(det(Dᵥ₊)) and log(|det(Dᵥ₊)|)
    logdetDᵥ₊, sgndetDᵥ₊ = det_D(dᵥ₊)

    # calculate Rᵥ⁻¹⋅Dᵥ₊⁻¹
    rdiv_D!(Rᵥ⁻¹, dᵥ₊)
    Rᵥ⁻¹Dᵥ₊⁻¹ = Rᵥ⁻¹

    # calcuate Dᵤ₊ = max(Dᵤ, 1)
    dᵤ₊ = ws.v
    @. dᵤ₊ = max(dᵤ, 1)

    # calculate sign(det(Dᵥ₊)) and log(|det(Dᵥ₊)|)
    logdetDᵤ₊, sgndetDᵤ₊ = det_D(dᵤ₊)
    
    # calcualte Dᵤ₊⁻¹⋅Lᵤᵀ
    adjoint!(ws.M, Lᵤ)
    ldiv_D!(dᵤ₊, ws.M)
    Dᵤ₊⁻¹Lᵤᵀ = ws.M

    # calculate Dᵤ₋ = min(Dᵤ, 1)
    dᵤ₋ = ws.v
    @. dᵤ₋ = min(dᵤ, 1)
    
    # calculate Dᵤ₋⋅Rᵤ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹
    mul!(G, Rᵤ, Rᵥ⁻¹Dᵥ₊⁻¹) # Rᵤ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹
    lmul_D!(dᵤ₋, G) # Dᵤ₋⋅[Rᵤ⋅Rᵥ⁻¹⋅Dᵥ₊]

    # calculate Dᵥ₋ = min(Dᵥ, 1)
    dᵥ₋ = ws.v
    @. dᵥ₋ = min(dᵥ, 1)

    # calculate Dᵤ₊⁻¹⋅Lᵤᵀ⋅Lᵥ⋅Dᵥ₋
    mul!(ws.M″, Dᵤ₊⁻¹Lᵤᵀ, Lᵥ)
    rmul_D!(ws.M″, dᵥ₋)

    # calculate M = Dᵤ₋⋅Rᵤ⋅Rᵥ⁻¹⋅Dᵥ₊ + Dᵤ₊⁻¹⋅Lᵤᵀ⋅Lᵥ⋅Dᵥ₋
    M = G
    @. M = M + ws.M″

    # calculate M⁻¹, sign(det(M)) and log(|det(M)|)
    M⁻¹ = G
    logdetM⁻¹, sgndetM⁻¹ = inv_lu!(M⁻¹, ws.lu_ws)

    # calculate G = Rᵥ⁻¹⋅Dᵥ₊⁻¹⋅M⁻¹⋅Dᵤ₊⁻¹⋅Lᵤᵀ
    mul!(ws.M″, Rᵥ⁻¹Dᵥ₊⁻¹, M⁻¹) # [Rᵥ⁻¹⋅Dᵥ₊⁻¹]⋅M⁻¹
    mul!(G, ws.M″, Dᵤ₊⁻¹Lᵤᵀ) # G = [Rᵥ⁻¹⋅Dᵥ₊⁻¹⋅M⁻¹]⋅Dᵤ₊⁻¹⋅Lᵤᵀ

    # calculate sign(det(G)) and log(|det(G)|)
    sgndetG = sgndetRᵥ⁻¹ * conj(sgndetDᵥ₊) * sgndetM⁻¹ * conj(sgndetDᵤ₊) * conj(sgndetLᵤ)
    logdetG = -logdetDᵥ₊ + logdetM⁻¹ - logdetDᵤ₊

    return logdetG, sgndetG
end


@doc raw"""
    inv_invUpV!(G::AbstractMatrix{T}, U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

Calculate the numerically stable inverse ``G := [U^{-1}+V]^{-1},`` where ``G`` is a matrix and ``U``
and ``V`` are represented by [`LDR`](@ref) factorizations. This method also returns
``\log( \vert \det G\vert )`` and ``\textrm{sign}(\det G).``

# Algorithm

The numerically stable inverse ``G := [U^{-1}+V]^{-1}`` is calculated using the procedure
```math
\begin{align*}
G:= & [U^{-1}+V]^{-1}\\
= & [\overset{D_{u,\max}D_{u,\min}}{(L_{u}\overbrace{D_{u}}R_{u})^{-1}}+\overset{D_{v,\min}D_{v,\max}}{L_{v}\overbrace{D_{v}}R_{v}}]^{-1}\\
= & [(L_{u}D_{u,\max}D_{u,\min}R_{u})^{-1}+L_{v}D_{v,\min}D_{v,\max}R_{v}]^{-1}\\
= & [R_{u}^{-1}D_{u,\min}^{-1}D_{u,\max}^{-1}L_{u}^{\dagger}+L_{v}D_{v,\min}D_{v,\max}R_{v}]^{-1}\\
= & [R_{u}^{-1}D_{u,\min}^{-1}(D_{u,\max}^{-1}L_{u}^{\dagger}R_{v}^{-1}D_{v,\max}^{-1}+D_{u,\min}R_{u}L_{v}D_{v,\min})D_{v,\max}R_{v}]^{-1}\\
= & R_{v}^{-1}D_{v,\max}^{-1}[\overset{M}{\overbrace{D_{u,\max}^{-1}L_{u}^{\dagger}R_{v}^{-1}D_{v,\max}^{-1}+D_{u,\min}R_{u}L_{v}D_{v,\min}}}]^{-1}D_{u,\min}R_{u}\\
= & R_{v}^{-1}D_{v,\max}^{-1}M^{-1}D_{u,\min}R_{u}
\end{align*}
```
where
```math
\begin{align*}
D_{u,\min} = & \min(D_u,1)\\
D_{u,\max} = & \max(D_u,1)\\
D_{v,\min} = & \min(D_v,1)\\
D_{v,\max} = & \max(D_v,1),
\end{align*}
```
and all intermediate matrix inversions and determinant calculations are performed via LU factorizations with partial pivoting.
"""
function inv_invUpV!(G::AbstractMatrix{T}, U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

    Lᵤ = U.L
    dᵤ = U.d
    Rᵤ = V.R
    Lᵥ = V.L
    dᵥ = V.d
    Rᵥ = V.R

    # calcualte sign(det(Rᵤ)) and log(|det(Rᵤ)|)
    copyto!(ws.M′, Rᵤ)
    logdetRᵤ, sgndetRᵤ = det_lu!(ws.M′, ws.lu_ws)

    # calculate Dᵤ₋ = min(Dᵤ, 1)
    dᵤ₋ = ws.v
    @. dᵤ₋ = min(dᵤ, 1)

    # calculate sign(det(Dᵤ₋)) and log(|det(Dᵤ₋)|)
    logdetDᵤ₋, sgndetDᵤ₋ = det_D(dᵤ₋)

    # calculate Dᵤ₋⋅Rᵤ
    Dᵤ₋Rᵤ = ws.M′
    copyto!(Dᵤ₋Rᵤ, Rᵤ)
    lmul_D!(dᵤ₋, Dᵤ₋Rᵤ)

    # calculate Rᵥ⁻¹, sign(det(Rᵥ⁻¹)) and log(|det(Rᵥ⁻¹)|)
    Rᵥ⁻¹ = ws.M″
    copyto!(Rᵥ⁻¹, Rᵥ)
    logdetRᵥ⁻¹, sgndetRᵥ⁻¹ = inv_lu!(Rᵥ⁻¹, ws.lu_ws)

    # calculate Dᵥ₊ = max(Dᵥ, 1)
    dᵥ₊ = ws.v
    @. dᵥ₊ = max(dᵥ, 1)

    # calculate sign(det(Dᵥ₊)) and log(|det(Dᵥ₊)|)
    logdetDᵥ₊, sgndetDᵥ₊ = det_D(dᵥ₊)

    # calculate Rᵥ⁻¹⋅Dᵥ₊⁻¹
    Rᵥ⁻¹Dᵥ₊⁻¹ = Rᵥ⁻¹
    rdiv_D!(Rᵥ⁻¹Dᵥ₊⁻¹, dᵥ₊)

    # calculate Dᵥ₋ = min(Dᵥ, 1)
    dᵥ₋ = ws.v
    @. dᵥ₋ = min(dᵥ, 1)

    # calculate Dᵤ₋⋅Rᵤ⋅Lᵥ⋅Dᵥ₋
    mul!(G, Dᵤ₋Rᵤ, Lᵥ) # Dᵤ₋⋅Rᵤ⋅Lᵥ
    rmul_D!(G, dᵥ₋) # [Dᵤ₋⋅Rᵤ⋅Lᵥ]⋅Dᵥ₋

    # calculate Dᵤ₊ = max(Dᵤ, 1)
    dᵤ₊ = ws.v
    @. dᵤ₊ = max(dᵤ, 1)

    # calculate Dᵤ₊⁻¹⋅Lᵤᵀ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹
    Lᵤᵀ = adjoint(Lᵤ)
    mul!(ws.M, Lᵤᵀ, Rᵥ⁻¹Dᵥ₊⁻¹) # Lᵤᵀ⋅[Rᵥ⁻¹⋅Dᵥ₊⁻¹]
    ldiv_D!(dᵤ₊, ws.M) # Dᵤ₊⁻¹⋅[Lᵤᵀ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹]

    # calculate Dᵤ₊⁻¹⋅Lᵤᵀ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹ + Dᵤ₋⋅Rᵤ⋅Lᵥ⋅Dᵥ₋
    @. G += ws.M

    # calculate M⁻¹, sign(det(M⁻¹)) and log(|det(M⁻¹)|)
    M⁻¹ = G
    logdetM⁻¹, sgndetM⁻¹ = inv_lu!(M⁻¹, ws.lu_ws)

    # calculate G := Rᵥ⁻¹⋅Dᵥ₊⁻¹⋅M⁻¹⋅Dᵤ₋⋅Rᵤ
    mul!(ws.M, M⁻¹, Dᵤ₋Rᵤ) # M⁻¹⋅Dᵤ₋⋅Rᵤ
    mul!(G, Rᵥ⁻¹Dᵥ₊⁻¹, ws.M) # G := Rᵥ⁻¹⋅Dᵥ₊⁻¹⋅[M⁻¹⋅Dᵤ₋⋅Rᵤ]

    # calcualte sign(det(G)) and log(|det(G)|)
    sgndetG = sgndetRᵥ⁻¹ * conj(sgndetDᵥ₊) * sgndetM⁻¹ * sgndetDᵤ₋ * sgndetRᵤ
    logdetG = -logdetDᵥ₊ + logdetM⁻¹ + logdetDᵤ₋

    return logdetG, sgndetG
end