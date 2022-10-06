###############################
## OVERLOADING Base.eltype() ##
###############################

@doc raw"""
    eltype(LDR{T}) where {T}

Return the matrix element type `T` of the [`LDR`](@ref) factorization `F`.
"""
eltype(F::LDR{T}) where {T} = T


#############################
## OVERLOADING Base.size() ##
#############################

@doc raw"""
    size(F::LDR, dim...)

Return the size of the [`LDR`](@ref) factorization `F`.
"""
size(F::LDR, dim...) = size(F.L, dim...)


################################
## OVERLOADING Base.copyto!() ##
################################

@doc raw"""
    copyto!(U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

Copy the matrix represented by the [`LDR`](@ref) factorization `V` into the matrix `U`.
"""
function copyto!(U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

    (; L, d, R) = V
    (; M) = ws

    copyto!(M, R) # R
    lmul_D!(d, M) # D⋅R
    mul!(U, L, M) # U = L⋅D⋅R

    return nothing
end

@doc raw"""
    copyto!(U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T})

    copyto!(U::LDR, I::UniformScaling, ignore...)

Copy the matrix `V` to the [`LDR`](@ref) factorization `U`, calculating the
[`LDR`](@ref) factorization to represent `V`.
"""
copyto!(U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T} = ldr!(U,V,ws)
copyto!(U::LDR, I::UniformScaling, ignore...) = ldr!(U,I)

@doc raw"""
    copyto!(U::LDR{T}, V::LDR{T}, ignore...) where {T}

Copy the ['LDR'](@ref) factorization `V` to `U`.
"""
copyto!(U::LDR{T}, V::LDR{T}, ignore...) where {T} = ldr!(U, V)


##########################################
## OVERLOADING LinearAlgebra.adjoint!() ##
##########################################

@doc raw"""
    adjoint!(Aᵀ::AbstractMatrix{T}, A::LDR{T}, ws::LDRWorkspace{T}) where {T}

Given an [`LDR`](@ref) factorization ``A``, construct the matrix representing its adjoint ``A^{\dagger}.``
"""
function adjoint!(Aᵀ::AbstractMatrix{T}, A::LDR{T}, ws::LDRWorkspace{T}) where {T}

    (; L, d, R) = A
    Rᵀ = ws.M′
    adjoint!(Rᵀ, R)
    adjoint!(ws.M, L) # Lᵀ
    lmul_D!(d, ws.M) # D⋅Lᵀ
    mul!(Aᵀ, Rᵀ, ws.M) # Rᵀ⋅D⋅Lᵀ

    return nothing
end

#######################################
## OVERLOADING LinearAlgebra.lmul!() ##
#######################################

@doc raw"""
    lmul!(U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}

Calculate ``V := U V`` where ``U`` is a [`LDR`](@ref) factorization and ``V`` is a matrix.
"""
function lmul!(U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}

    # calculate V := Lᵤ⋅Dᵤ⋅Rᵤ⋅V
    mul!(ws.M, U.R, V) # Rᵤ⋅V
    lmul_D!(U.d, ws.M) # Dᵤ⋅Rᵤ⋅V
    mul!(V, U.L, ws.M) # V := Lᵤ⋅Dᵤ⋅Rᵤ⋅V

    return nothing
end

@doc raw"""
    lmul!(U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

Calculate the numerically stable product ``V := U V,`` where ``U`` is a matrix and ``V``` is an [`LDR`](@ref) factorization.

# Algorithm

Calculate ``V := U V`` using the procedure

```math
\begin{align*}
V:= & UV\\
= & \overset{L_{0}D_{0}R_{0}}{\overbrace{U[L_{v}D_{v}}}R_{v}]\\
= & \overset{L_{1}}{\overbrace{L_{0}}}\,\overset{D_{1}}{\overbrace{D_{0}}}\,\overset{R_{1}}{\overbrace{R_{0}R_{v}}}\\
= & L_{1}D_{1}R_{1}.
\end{align*}
```
"""
function lmul!(U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

    # record original Rₐ matrix
    Rᵥ = ws.M′
    copyto!(Rᵥ, V.R)

    # calculate product U⋅Lᵥ⋅Dᵥ
    mul!(ws.M, U, V.L) # U⋅Lᵥ
    mul_D!(V.L, ws.M, V.d) # U⋅Lᵥ⋅Dᵥ

    # calcualte [L₀⋅D₀⋅R₀] = U⋅Lᵥ⋅Dᵥ
    ldr!(V, ws)

    # calcualte R₁ = R₀⋅Rₐ
    mul!(ws.M, V.R, Rᵥ)
    copyto!(V.R, ws.M)

    return nothing
end

@doc raw"""
    lmul!(U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

Calculate the numerically stable product ``V := U V,`` where ``U`` and ``V``` are both [`LDR`](@ref) factorizations.

# Algorithm

Calculate ``V := U V`` using the procedure

```math
\begin{align*}
V:= & UV\\
= & [L_{u}D_{u}\overset{M}{\overbrace{R_{u}][L_{v}}}D_{v}R_{v}]\\
= & L_{u}\overset{L_{0}D_{0}R_{0}}{\overbrace{D_{u}MD_{v}}}R_{v}\\
= & \overset{L_{1}}{\overbrace{L_{u}L_{0}}}\,\overset{D_{1}}{\overbrace{D_{0}}}\,\overset{R_{1}}{\overbrace{R_{0}R_{v}}}\\
= & L_{1}D_{1}R_{1}.
\end{align*}
```
"""
function lmul!(U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

    # record original Rᵥ
    Rᵥ = ws.M′
    copyto!(Rᵥ, V.R)

    # calculate M = Rᵤ⋅Lᵥ
    mul!(ws.M, U.R, V.L)

    # calculate Dᵤ⋅M⋅Dᵥ
    rmul_D!(ws.M, V.d) # M⋅Dᵥ
    mul_D!(V.L, U.d, ws.M) # Dᵤ⋅M⋅Dᵥ

    # calculate [L₀⋅D₀⋅R₀] = Dᵤ⋅M⋅Dᵥ
    ldr!(V, ws)

    # calculate L₁ = Lᵤ⋅L₀
    mul!(ws.M, U.L, V.L)
    copyto!(V.L, ws.M)

    # calculate R₁ = R₀⋅Rᵥ
    mul!(ws.M, V.R, Rᵥ)
    copyto!(V.R, ws.M)

    return nothing
end


#######################################
## OVERLOADING LinearAlgebra.rmul!() ##
#######################################

@doc raw"""
    rmul!(U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

Calculate ``U := U V`` where ``U`` is a matrix and ``V`` is a [`LDR`](@ref) factorization.
"""
function rmul!(U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

    # calculate U := U⋅Lᵥ⋅Dᵥ⋅Rᵥ
    mul!(ws.M, U, V.L) # U⋅Lᵥ
    rmul_D!(ws.M, V.d) # U⋅Lᵥ⋅Dᵥ
    mul!(U, ws.M, V.R) # U := U⋅Lᵥ⋅Dᵥ⋅Rᵥ

    return nothing
end

@doc raw"""
    rmul!(U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}

Calculate the numerically stable product ``U := U V,`` where ``U`` is a [`LDR`](@ref) factorization and ``V`` is a matrix.

# Algorithm

Calculate ``U := U V`` using the procedure

```math
\begin{align*}
U:= & UV\\
= & [L_{u}\overset{L_{0}D_{0}R_{0}}{\overbrace{D_{u}R_{u}]V}}\\
= & \overset{L_{1}}{\overbrace{L_{u}L_{0}}}\,\overset{D_{1}}{\overbrace{D_{0}}}\,\overset{R_{1}}{\overbrace{R_{0}}}\\
= & L_{1}D_{1}R_{1}.
\end{align*}
```
"""
function rmul!(U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}

    # record intial Lₐ
    Lᵤ = ws.M′
    copyto!(Lᵤ, U.L)

    # calculate Dᵤ⋅Rᵤ⋅V
    mul!(U.L, U.R, V)
    lmul_D!(U.d, U.L)

    # calculate [L₀⋅D₀⋅R₀] = Dᵤ⋅Rᵤ⋅V
    ldr!(U, ws)

    # calculate L₁ = Lᵤ⋅L₀
    mul!(ws.M, Lᵤ, U.L)
    copyto!(U.L, ws.M)

    return nothing
end

@doc raw"""
    rmul!(U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

Calculate the numerically stable product ``U := U V,`` where both ``U`` and ``V`` are [`LDR`](@ref) factorizations.

# Algorithm

Calculate ``U := U V`` using the procedure

```math
\begin{align*}
U:= & UV\\
= & [L_{u}D_{u}\overset{M}{\overbrace{R_{u}][L_{v}}}D_{v}R_{v}]\\
= & L_{u}\overset{L_{0}D_{0}R_{0}}{\overbrace{D_{u}MD_{v}}}R_{v}\\
= & \overset{L_{1}}{\overbrace{L_{u}L_{0}}}\,\overset{D_{1}}{\overbrace{D_{0}}}\,\overset{R_{1}}{\overbrace{R_{0}R_{v}}}\\
= & L_{1}D_{1}R_{1}.
\end{align*}
```
"""
function rmul!(U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

    # record initial Lᵤ
    Lᵤ = ws.M′
    copyto!(Lᵤ, U.L)

    # calculate M = Rᵤ⋅Lᵥ
    mul!(ws.M, U.R, V.L)

    # calculate Dᵤ⋅Rᵤ⋅Lᵥ⋅Dᵥ
    rmul_D!(ws.M, V.d)
    mul_D!(U.L, U.d, ws.M)

    # calculate [L₀⋅D₀⋅R₀] = Dᵤ⋅Rᵤ⋅Lᵥ⋅Dᵥ
    ldr!(U, ws)

    # L₁ = Lᵤ⋅L₀
    mul!(ws.M, Lᵤ, U.L)
    copyto!(U.L, ws.M)

    # R₁ = R₀⋅Rᵥ
    mul!(ws.M, U.R, V.R)
    copyto!(U.R, ws.M)

    return nothing
end


######################################
## OVERLOADING LinearAlgebra.mul!() ##
######################################

@doc raw"""
    mul!(H::AbstractMatrix{T}, U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}

Calculate the matrix product ``H := U V``, where ``H`` and ``V`` are matrices and ``U`` is
a [`LDR`](@ref) factorization.
"""
function mul!(H::AbstractMatrix{T}, U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}

    copyto!(H, V)
    lmul!(U, H, ws)

    return nothing
end

@doc raw"""
    mul!(H::AbstractMatrix{T}, U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

Calculate the matrix product ``H := U V``, where ``H`` and ``U`` are matrices and ``V`` is
a [`LDR`](@ref) factorization.
"""
function mul!(H::AbstractMatrix{T}, U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

    copyto!(H, U)
    rmul!(H, V, ws)

    return nothing
end

@doc raw"""
    mul!(H::LDR{T}, U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

Calculate the numerically stable product ``H := U V``, where ``U`` is matrix, and ``H`` and
``V`` are both [`LDR`](@ref) factorization. For the algorithm refer to documentation for [`lmul!`](@ref).
"""
function mul!(H::LDR{T}, U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

    copyto!(H, V)
    lmul!(U, H, ws)

    return nothing
end

@doc raw"""
    mul!(H::LDR{T}, U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}

Calculate the numerically stable product ``H := U V``, where ``V`` is matrix, and ``H`` and
``U`` are both [`LDR`](@ref) factorizations. For the algorithm refer to the documentation for [`rmul!`](@ref).
"""
function mul!(H::LDR{T}, U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}

    copyto!(H, U)
    rmul!(H, V, ws)

    return nothing
end

@doc raw"""
    mul!(H::LDR{T}, U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

Calculate the numerically stable matrix product ``H := U V,`` where ``H,`` ``U`` and ``V`` are all
[`LDR`](@ref) factorizations. For the algorithm refer to the documentation for [`lmul!`](@ref).
"""
function mul!(H::LDR{T}, U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

    copyto!(H, V)
    lmul!(U, H, ws)

    return nothing
end


#######################################
## OVERLOADING LinearAlgebra.ldiv!() ##
#######################################

@doc raw"""
    ldiv!(U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}

Calculate ``V := U^{-1} V,`` where ``V`` is a matrix, and ``U`` is an [`LDR`](@ref) factorization.
"""
function ldiv!(U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}

    # calculate V := U⁻¹⋅V = [Lᵤ⋅Dᵤ⋅Rᵤ]⁻¹⋅V = Rᵤ⁻¹⋅Dᵤ⁻¹⋅Lᵤ⁻¹⋅V
    Lᵤ = ws.M
    copyto!(Lᵤ, U.L)
    ldiv_lu!(Lᵤ, V, ws.lu_ws) # Lᵤ⁻¹⋅V
    ldiv_D!(U.d, V) # Dᵤ⁻¹⋅Lᵤ⁻¹⋅V
    Rᵤ = ws.M
    copyto!(Rᵤ, U.R)
    ldiv_lu!(Rᵤ, V, ws.lu_ws) # V := Rᵤ⁻¹⋅Dᵤ⁻¹⋅Lᵤ⁻¹⋅V

    return nothing
end

@doc raw"""
    ldiv!(H::AbstractMatrix{T}, U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}

Calculate ``H := U^{-1} V,`` where ``H`` and ``V`` are matrices, and ``U`` is an [`LDR`](@ref) factorization.
"""
function ldiv!(H::AbstractMatrix{T}, U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}

    copyto!(H, V)
    ldiv!(U, H, ws)
    return nothing
end

@doc raw"""
    ldiv!(U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

Calculate the numerically stable product ``V := U^{-1}V`` where both ``U`` and ``V`` are [`LDR`](@ref) factorizations.
Note that an intermediate LU factorization is required to calucate the matrix inverse ``R_u^{-1},`` in addition to the
intermediate [`LDR`](@ref) factorization that needs to occur.

# Algorithm

Calculate ``V := U^{-1}V`` using the procedure

```math
\begin{align*}
V:= & U^{-1}V\\
= & [L_{u}D_{u}R_{u}]^{-1}[L_{v}D_{v}R_{v}]\\
= & R_{u}^{-1}D_{u}^{-1}\overset{M}{\overbrace{L_{u}^{-1}L_{v}}}D_{v}R_{v}\\
= & \overset{L_{0}D_{0}R_{0}}{\overbrace{R_{u}^{-1}D_{u}^{-1}MD_{v}}}R_{v}\\
= & \overset{L_{1}}{\overbrace{L_{0}}}\,\overset{D_{1}}{\overbrace{D_{0}^{\phantom{1}}}}\,\overset{R_{1}}{\overbrace{R_{0}R_{v}^{\phantom{1}}}}\\
= & L_{1}D_{1}R_{1}.
\end{align*}
```
"""
function ldiv!(U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

    # calculate Lᵤᵀ⋅Lᵥ
    Lᵤᵀ = adjoint(U.L)
    mul!(ws.M, Lᵤᵀ, V.L)
    copyto!(V.L, ws.M)

    # record initial Rᵥ
    Rᵥ = ws.M′
    copyto!(Rᵥ, V.R)

    # calculate Rᵤ⁻¹⋅Dᵤ⁻¹⋅Lᵤᵀ⋅Lᵥ⋅Dᵥ
    rmul_D!(V.L, V.d) # [Lᵤᵀ⋅Lᵥ]⋅Dᵥ
    ldiv_D!(U.d, V.L) # Dᵤ⁻¹⋅[Lᵤᵀ⋅Lᵥ⋅Dᵥ]
    Rᵤ = ws.M
    copyto!(Rᵤ, U.R)
    ldiv_lu!(Rᵤ, V.L, ws.lu_ws) # Rᵤ⁻¹⋅[Dᵤ⁻¹⋅Lᵤᵀ⋅Lᵥ⋅Dᵥ]

    # calculate [L₀⋅D₀⋅R₀] = Rᵤ⁻¹⋅Dᵤ⁻¹⋅Lᵤᵀ⋅Lᵥ⋅Dᵥ
    ldr!(V, ws)

    # calculate R₁ = R₀⋅Rᵥ
    mul!(ws.M, V.R, Rᵥ)
    copyto!(V.R, ws.M)

    return nothing
end

@doc raw"""
    ldiv!(H::LDR{T}, U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

Calculate the numerically stable product ``H := U^{-1} V,`` where ``H,`` ``U`` and ``V`` are all [`LDR`](@ref) factorizations.
Note that an intermediate LU factorization is required to calucate the matrix inverse ``R_u^{-1},`` in addition to the
intermediate [`LDR`](@ref) factorization that needs to occur.
"""
function ldiv!(H::LDR{T}, U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

    copyto!(H, V)
    ldiv!(U, H, ws)

    return nothing
end

@doc raw"""
    ldiv!(U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

Calculate the numerically stable product ``V := U^{-1} V,`` where ``U`` is a matrix and ``V`` is a [`LDR`](@ref) factorization.
Note that an intermediate LU factorization is required as well to calucate the matrix inverse ``U^{-1},`` in addition to the
intermediate [`LDR`](@ref) factorization that needs to occur.

# Algorithm

The numerically stable procdure used to evaluate ``V := U^{-1} V`` is

```math
\begin{align*}
V:= & U^{-1}V\\
= & \overset{L_{0}D_{0}R_{0}}{\overbrace{U^{-1}[L_{v}D_{v}}}R_{v}]\\
= & \overset{L_{1}}{\overbrace{L_{0}}}\,\overset{D_{1}}{\overbrace{D_{0}}}\,\overset{R_{1}}{\overbrace{R_{0}R_{v}}}\\
= & L_{1}D_{1}R_{1}.
\end{align*}
```
"""
function ldiv!(U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

    # store Rᵥ for later
    Rᵥ = ws.M′
    copyto!(Rᵥ, V.R)

    # calculate U⁻¹⋅Lᵥ⋅Dᵥ
    rmul_D!(V.L, V.d) # Lᵥ⋅Dᵥ
    copyto!(ws.M, U)
    ldiv_lu!(ws.M, V.L, ws.lu_ws) # U⁻¹⋅Lᵥ⋅Dᵥ
    
    # calculate [L₀⋅D₀⋅R₀] = U⁻¹⋅Lᵥ⋅Dᵥ
    ldr!(V, ws)

    # R₁ = R₀⋅Rᵥ
    mul!(ws.M, V.R, Rᵥ)
    copyto!(V.R, ws.M)

    return nothing
end

@doc raw"""
    ldiv!(H::LDR{T}, U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

Calculate the numerically stable product ``H := U^{-1} V,`` where ``H`` and ``V`` are [`LDR`](@ref)
factorizations and ``U`` is a matrix. Note that an intermediate LU factorization is required to
calculate ``U^{-1},`` in addition to the intermediate [`LDR`](@ref) factorization that needs to occur.
"""
function ldiv!(H::LDR{T}, U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

    copyto!(H, V)
    ldiv!(U, H, ws)

    return nothing
end

#######################################
## OVERLOADING LinearAlgebra.rdiv!() ##
#######################################

@doc raw"""
    rdiv!(U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

Calculate the matrix product ``U := U V^{-1},`` where ``V`` is an [`LDR`](@ref) factorization and ``U`` is a matrix.
Note that this requires two intermediate LU factorizations to calculate ``L_v^{-1}`` and ``R_v^{-1}``.
"""
function rdiv!(U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

    # U := U⋅Rᵥ⁻¹⋅Dᵥ⁻¹⋅Lᵥ⁻¹
    copyto!(ws.M, I)
    Lᵥ = ws.M′
    copyto!(Lᵥ, V.L)
    ldiv_lu!(Lᵥ, ws.M, ws.lu_ws) # Lᵥ⁻¹
    ldiv_D!(V.d, ws.M) # Dᵥ⁻¹⋅Lᵥ⁻¹
    Rᵥ = ws.M′
    copyto!(Rᵥ, V.R)
    ldiv_lu!(Rᵥ, ws.M, ws.lu_ws) # Rᵥ⁻¹⋅Dᵥ⁻¹⋅Lᵥ⁻¹
    mul!(ws.M′, U, ws.M) # U⋅Rᵥ⁻¹⋅Dᵥ⁻¹⋅Lᵥ⁻¹
    copyto!(U, ws.M′) # U := U⋅Rᵥ⁻¹⋅Dᵥ⁻¹⋅Lᵥ⁻¹

    return nothing
end

@doc raw"""
    rdiv!(H::AbstractMatrix{T}, U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

Calculate the matrix product ``H := U V^{-1},`` where ``H`` and ``U`` are matrices and ``V`` is a [`LDR`](@ref) factorization.
Note that this requires two intermediate LU factorizations to calculate ``L_v^{-1}`` and ``R_v^{-1}``.
"""
function rdiv!(H::AbstractMatrix{T}, U::AbstractMatrix{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

    copyto!(H, U)
    rdiv!(H, V, ws)

    return nothing
end

@doc raw"""
    rdiv!(U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

Calculate the numerically stable product ``U := U V^{-1}`` where both ``U`` and ``V`` are [`LDR`](@ref) factorizations.
Note that an intermediate LU factorization is required to calucate the matrix inverse ``L_v^{-1},`` in addition to the
intermediate [`LDR`](@ref) factorization that needs to occur.

# Algorithm

Calculate ``U := UV^{-1}`` using the procedure

```math
\begin{align*}
U:= & UV^{-1}\\
= & [L_{u}D_{u}R_{u}][L_{v}D_{v}R_{v}]^{-1}\\
= & L_{u}D_{u}\overset{M}{\overbrace{R_{u}R_{v}^{-1}}}D_{v}^{-1}L_{v}^{\dagger}\\
= & L_{u}\overset{L_{0}D_{0}R_{0}}{\overbrace{D_{u}MD_{v}^{-1}}}L_{v}^{\dagger}\\
= & \overset{L_{1}}{\overbrace{L_{u}L_{0}^{\phantom{1}}}}\,\overset{D_{1}}{\overbrace{D_{0}^{\phantom{1}}}}\,\overset{R_{1}}{\overbrace{R_{0}L_{v}^{\dagger}}}\\
= & L_{1}D_{1}R_{1}.
\end{align*}
```
"""
function rdiv!(U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

    # calculate Rᵥ⁻¹
    Rᵥ⁻¹ = ws.M′
    copyto!(Rᵥ⁻¹, V.R)
    inv_lu!(Rᵥ⁻¹, ws.lu_ws)

    # calculate M = Rᵤ⋅Rᵥ⁻¹
    mul!(ws.M, U.R, Rᵥ⁻¹)

    # record original Lᵤ matrix
    Lᵤ = ws.M′
    copyto!(Lᵤ, U.L)

    # calculate Dᵤ⋅M⋅Dᵥ⁻¹
    div_D!(U.L, ws.M, V.d)
    lmul_D!(U.d, U.L)

    # calculate [L₀⋅D₀⋅R₀] = Dᵤ⋅M⋅Dᵥ⁻¹
    ldr!(U, ws)

    # L₁ = Lᵤ⋅L₀
    mul!(ws.M, Lᵤ, U.L)
    copyto!(U.L, ws.M)

    # calculate Rᵥ = R₀⋅Lᵥᵀ
    Lᵥᵀ = adjoint(V.L)
    mul!(ws.M, U.R, Lᵥᵀ)
    copyto!(U.R, ws.M)

    return nothing
end

@doc raw"""
    rdiv!(H::LDR{T}, U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

Calculate the numerically stable product ``H := U V^{-1}`` where ``H,`` ``U`` and ``V`` are all [`LDR`](@ref) factorizations.
Note that an intermediate LU factorization is required to calucate the matrix inverse ``L_v^{-1},`` in addition to the
intermediate [`LDR`](@ref) factorization that needs to occur.
"""
function rdiv!(H::LDR{T}, U::LDR{T}, V::LDR{T}, ws::LDRWorkspace{T}) where {T}

    copyto!(H, U)
    rdiv!(H, V, ws)

    return nothing
end

@doc raw"""
    rdiv!(U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace) where {T}

Calculate the numerically stable product ``U := U V^{-1},`` where ``V`` is a matrix and ``U`` is an [`LDR`](@ref) factorization.
Note that an intermediate LU factorization is required as well to calucate the matrix inverse ``V^{-1},`` in addition to the
intermediate [`LDR`](@ref) factorization that needs to occur.

# Algorithm

The numerically stable procdure used to evaluate ``U := U V^{-1}`` is

```math
\begin{align*}
U:= & UV^{-1}\\
= & [L_{u}\overset{L_{0}D_{0}R_{0}}{\overbrace{D_{u}R_{u}]V^{-1}}}\\
= & \overset{L_{1}}{\overbrace{L_{u}L_{0}}}\,\overset{D_{1}}{\overbrace{D_{0}}}\,\overset{R_{1}}{\overbrace{R_{0}}}\\
= & L_{1}D_{1}R_{1}.
\end{align*}
```
"""
function rdiv!(U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}

    # record intial Lᵤ
    Lᵤ = ws.M′
    copyto!(Lᵤ, U.L)

    # calculate Dᵤ⋅Rᵤ⋅V⁻¹
    copyto!(ws.M, V)
    inv_lu!(ws.M, ws.lu_ws) # V⁻¹
    mul!(U.L, U.R, ws.M) # Rᵤ⋅V⁻¹
    lmul_D!(U.d, U.L) # Dᵤ⋅Rᵤ⋅V⁻¹

    # calcualte [L₀⋅D₀⋅R₀] = Dᵤ⋅Rᵤ⋅V⁻¹
    ldr!(U, ws)

    # calculate L₁ = Lᵤ⋅L₀
    mul!(ws.M, Lᵤ, U.L)
    copyto!(U.L, ws.M)

    return nothing
end

@doc raw"""
    rdiv!(H::LDR{T}, U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}

Calculate the numerically stable product ``H := U V^{-1},`` where ``V`` is a matrix and ``H`` and ``U`` is an [`LDR`](@ref) factorization.
Note that an intermediate LU factorization is required as well to calucate the matrix inverse ``V^{-1},`` in addition to the
intermediate [`LDR`](@ref) factorization that needs to occur.
"""
function rdiv!(H::LDR{T}, U::LDR{T}, V::AbstractMatrix{T}, ws::LDRWorkspace{T}) where {T}

    copyto!(H, U)
    rdiv!(H, V, ws)

    return nothing
end

###########################################
## OVERLOADING LinearAlgebra.logabsdet() ##
###########################################

@doc raw"""
    logabsdet(A::LDR{T}, ws::LDRWorkspace{T}) where {T}

Calculate ``\log(\vert \det A \vert)`` and ``\textrm{sign}(\det A)`` for the
[`LDR`](@ref) factorization ``A.``
"""
function logabsdet(A::LDR{T}, ws::LDRWorkspace{T}) where {T}

    copyto!(ws.M, A.L)
    logdetL, sgndetL = det_lu!(ws.M, ws.lu_ws)
    logdetD, sgndetD = det_D(A.d)
    copyto!(ws.M, A.R)
    logdetR, sgndetR = det_lu!(ws.M, ws.lu_ws)
    logdetA = logdetL + logdetD + logdetR
    sgndetA = sgndetL * sgndetD * sgndetR

    return logdetA, sgndetA
end