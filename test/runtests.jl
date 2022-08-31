using StableLinearAlgebra
using LinearAlgebra
using Test

# calculate eigenenergies and eigenstates for 1D chain tight-binding model
function hamiltonian(N, t, μ)
        
    H  = zeros(typeof(t),N,N)
    for i in 1:N
        j = mod1(i+1,N)
        H[i,i] = -μ
        H[i,j] = -t
        H[j,i] = conj(H[i,j])
    end
    ϵ, U = eigen(H)
    
    return ϵ, U, H
end

# calculate the greens function given the eigenenergies and eigenstates
function greens(τ,β,ϵ,U)
    
    gτ = similar(ϵ)
    @. gτ = exp(-τ*ϵ)/(1+exp(-β*ϵ))
    Gτ = U * Diagonal(gτ) * adjoint(U)
    
    return Gτ
end

# calculate propagator matrix B(τ) given eigenenergies and eigenstates
function propagator(τ,ϵ,U)
    
    bτ = similar(ϵ)
    @. bτ = exp(-τ*ϵ)
    Bτ = U * Diagonal(bτ) * adjoint(U)
    
    return Bτ
end

@testset "StableLinearAlgebra.jl" begin
    
    # system parameters
    N     = 12 # system size
    t     = 1.0 # nearest neighbor hopping
    μ     = 0.5 # chemical potential
    β     = 20.0 # inverse temperature
    Δτ    = 0.1 # discretization in imaginary time
    L     = round(Int,β/Δτ) # length of imaginary time axis
    nstab = 10 # stabalization frequency
    Nstab = L ÷ nstab # number of reduced propagator matrices

    # hamitlonian eigenenergies and eigenstates
    ϵ, U, H = hamiltonian(N, t, μ)

    # propagator matrix
    B = propagator(Δτ, ϵ, U)

    # inverse propagator matrix
    B⁻¹ = propagator(-Δτ, ϵ, U)

    # greens functions
    G_0   = greens(0, β, ϵ, U) # G(τ=0)
    G_βo2 = greens(β/2, β, ϵ, U) # G(τ=β/2)

    # temporary matrices
    A = similar(B)
    M = similar(B)
    M′ = similar(B)
    M″ = similar(B)

    # temporary permutation vectors
    p = ones(Int, N)
    p′ = ones(Int, N)

    # partial propagator matrix product
    B̄ = Matrix{eltype(B)}(I,N,N)
    for i in 1:nstab
        mul!(M, B, B̄)
        copyto!(B̄, M)
    end

    # array of partial propagator matrix producs
    B̄s = zeros(eltype(B),N,N,Nstab)
    for i in 1:Nstab
        B̄ᵢ = @view B̄s[:,:,i]
        copyto!(B̄ᵢ, B̄)
    end

    # partial inverse propagator matrix product
    B̄⁻¹ = Matrix{eltype(B⁻¹)}(I,N,N)
    for i in 1:nstab
        mul!(M, B⁻¹, B̄⁻¹)
        copyto!(B̄⁻¹, M)
    end

    # temporary storage vector
    v_tmp  = zeros(eltype(B), N)
    v′_tmp = zeros(eltype(B), N)
    u_tmp  = zeros(eltype(B), N)
    u′_tmp = zeros(eltype(B), N)

    # test ldr(::AbstractMatrix) and copyto!
    F = ldr(B̄)
    copyto!(A, F, M=M)
    @test A ≈ B̄

    # test ldr(::LDR) and copyto!
    F′ = ldr(F)
    copyto!(A, F′, M=M)
    @test A ≈ B̄

    # testing LDR factorization with new matrix
    ldr!(F, B̄)
    copyto!(A, F, M=M)
    @test A ≈ B̄

    # test copying identity matrix to existing LDR factorization
    ldr!(F, I)
    copyto!(A, F, M=M)
    @test A ≈ I

    # testing lmul!(::AbstractMatrix, ::LDR) and inv_IpA!
    ldr!(F, I)
    for n in 1:Nstab
        lmul!(B̄, F, M=M, M′=M′, p=p)
    end
    inv_IpA!(A, F, F′=F′, d_min=u_tmp, d_max=u′_tmp, M=M, p=p)
    @test A ≈ G_0

    # test rmul!(::LDR, ::AbstractMatrix) and inv_IpA!
    ldr!(F, I)
    for n in 1:Nstab
        rmul!(F, B̄, M=M, M′=M′)
    end
    inv_IpA!(A, F, F′=F′, d_min=u_tmp, d_max=u′_tmp, M=M, p=p)
    @test A ≈ G_0

    # test ldrs(::AbstractMatrix, ::Int), lmul!(::LDR, ::LDR) and inv_IpA!
    Fs = ldrs(B̄, Nstab)
    ldr!(F, I)
    for n in 1:Nstab
        lmul!(Fs[n], F, M=M, M′=M′, p=p) 
    end
    inv_IpA!(A, F, F′=F′, d_min=u_tmp, d_max=u′_tmp, M=M, p=p)
    @test A ≈ G_0

    # test ldrs!(::Vector{LDR}, ::AbstractArray{3}), rmul!(::LDR, ::LDR) and inv_IpA!
    ldrs!(Fs, B̄s)
    ldr!(F, I)
    for n in 1:Nstab
        rmul!(F, Fs[n], M=M, M′=M′) 
    end
    inv_IpA!(A, F, F′=F′, d_min=u_tmp, d_max=u′_tmp, M=M, p=p)
    @test A ≈ G_0

    # test mul!(::LDR, ::AbstractMatrix, ::LDR)
    ldr!(F, I)
    for n in 1:Nstab
        mul!(F′, B̄, F, M=M)
        copyto!(F, F′)
    end
    inv_IpA!(A, F, F′=F′, d_min=u_tmp, d_max=u′_tmp, M=M, p=p)
    @test A ≈ G_0

    # test mul!(::LDR, ::LDR, ::AbstractMatrix)
    ldr!(F, I)
    for n in 1:Nstab
        mul!(F′, F, B̄, M=M)
        copyto!(F, F′)
    end
    inv_IpA!(A, F, F′=F′, d_min=u_tmp, d_max=u′_tmp, M=M, p=p)
    @test A ≈ G_0

    # test mul!(::LDR, ::LDR, ::LDR)
    Fs = ldrs(B̄, Nstab)
    ldr!(F, I)
    for n in 1:Nstab
        mul!(F′, Fs[n], F, M=M)
        copyto!(F, F′)
    end
    inv_IpA!(A, F, F′=F′, d_min=u_tmp, d_max=u′_tmp, M=M, p=p)
    @test A ≈ G_0

    # test inv_UpV!
    F″ = ldr(B)
    ldr!(F, I)
    ldr!(F′, I)
    for n in 1:Nstab÷2
        lmul!(B̄⁻¹, F′, M=M, M′=M′, p=p)
        rmul!(F, B̄, M=M, M′=M′)
    end
    inv_UpV!(A, F′, F,
             dᵤ_min=u_tmp, dᵤ_max=u′_tmp, dᵥ_min=v_tmp, dᵥ_max=v′_tmp,
             F=F″, M=M, M′=M′, M″=M″, p=p, p′=p′)
    @test A ≈ G_βo2

    # test inv_invUpV!
    ldr!(F, I)
    ldr!(F′, I)
    for n in 1:Nstab÷2
        rmul!(F, B̄, M=M, M′=M′)
    end
    copyto!(F′, F)
    inv_invUpV!(A, F, F′,
                dᵤ_min=u_tmp, dᵤ_max=u′_tmp, dᵥ_min=v_tmp, dᵥ_max=v′_tmp,
                F=F″, M=M, M′=M′, p=p, p′=p′)
    @test A ≈ G_βo2

    # test determinant calculation
    A = rand(Float64,10,10)
    Fa = ldr(A)
    @test det(A) ≈ det(Fa)

    # test determinant calculation
    A = rand(Complex{Float64},10,10)
    Fa = ldr(A)
    @test det(A) ≈ det(Fa)

    # test determinant ratio calculation
    A = rand(Float64,10,10)
    B = rand(Float64,10,10) 
    Fa = ldr(A)
    Fb = ldr(B)
    @test abs(det(A)/det(B)) ≈ abs_det_ratio(Fa,Fb)

    # test determinant ratio calculation
    A = rand(Complex{Float64},10,10)
    B = rand(Complex{Float64},10,10) 
    Fa = ldr(A)
    Fb = ldr(B)
    @test abs(det(A)/det(B)) ≈ abs_det_ratio(Fa,Fb)
end
