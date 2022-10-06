using StableLinearAlgebra
using LinearAlgebra
using Test
using LatticeUtilities

# construct and diagonalize hamiltonian matrix for square lattice tight binding model
function hamiltonian(L, t, μ)

    unit_cell = UnitCell(lattice_vecs = [[1.,0.],[0.,1.]], basis_vecs = [[0.,0.]])
    lattice = Lattice(L = [L,L], periodic = [true,true])
    bond_x = Bond(orbitals = (1,1), displacement = [1,0])
    bond_y = Bond(orbitals = (1,1), displacement = [0,1])
    neighbor_table = build_neighbor_table([bond_x, bond_y], unit_cell, lattice)
    Nsites = get_num_sites(unit_cell, lattice)
    Nbonds = size(neighbor_table, 2)
    H = zeros(typeof(t),Nsites,Nsites)
    for n in 1:Nbonds
        i = neighbor_table[1,n]
        j = neighbor_table[2,n]
        H[j,i] = -t
        H[i,j] = conj(-t)
    end
    for i in 1:Nsites
        H[i,i] = -μ
    end
    ϵ, U = eigen(H)
    
    return ϵ, U, H
end

# calculate the greens function given the eigenenergies and eigenstates
function greens(τ,β,ϵ,U)
    
    gτ = similar(ϵ)
    @. gτ = exp(-τ*ϵ)/(1+exp(-β*ϵ))
    Gτ = U * Diagonal(gτ) * adjoint(U)

    logdetGτ, sgndetGτ = logabsdet(Diagonal(gτ))
    
    return Gτ, sgndetGτ, logdetGτ
end

# calculate propagator matrix B(τ) given eigenenergies and eigenstates
function propagator(τ,ϵ,U)
    
    bτ = similar(ϵ)
    @. bτ = exp(-τ*ϵ)
    Bτ = U * Diagonal(bτ) * adjoint(U)


    logdetBτ, sgndetBτ = logabsdet(Diagonal(bτ))
    logdetBτ = -τ * sum(ϵ)

    return Bτ, logdetBτ, sgndetBτ
end

@testset "StableLinearAlgebra.jl" begin

    # system parameters
    L  = 4 # linear system size
    t  = 1.0 # nearest neighbor hopping
    μ  = 0.0 # chemical potential
    β  = 40.0 # inverse temperature
    Δτ = 0.1 # discretization in imaginary time
    Lτ = round(Int,β/Δτ) # length of imaginary time axis
    nₛ = 10 # stabalization frequency
    Nₛ = Lτ ÷ nₛ # number of reduced propagator matrices

    # hamitlonian eigenenergies and eigenstates
    ϵ, U, H = hamiltonian(L, t, μ)

    # number of sites in lattice
    N = size(H,1)

    # propagator matrix
    B, logdetB, sgndetB = propagator(Δτ, ϵ, U)

    # inverse propagator matrix
    B⁻¹, logdetB⁻¹, sgndetB⁻¹ = propagator(-Δτ, ϵ, U)

    # long propagators
    B_β = propagator(β, ϵ, U)

    # greens functions
    G_0, sgndetG_0, logdetG_0 = greens(0, β, ϵ, U) # G(τ=0)
    G_βo2, sgndetG_βo2, logdetG_βo2 = greens(β/2, β, ϵ, U) # G(τ=β/2)

    # temporary storage matrices
    A = similar(B)
    G = similar(B)

    # partial propagator matrix product
    B̄ = Matrix{eltype(B)}(I,N,N)
    B̄⁻¹ = Matrix{eltype(B)}(I,N,N)
    for i in 1:nₛ
        mul!(A, B, B̄)
        copyto!(B̄, A)
        mul!(A, B̄⁻¹, B⁻¹)
        copyto!(B̄⁻¹, A)
    end

    # initialize LDR workspace
    ws = ldr_workspace(B̄)

    # testing ldr and logabsdet
    A = rand(N,N)
    A = I + 0.1*A
    F = ldr(A, ws)
    logdetF, sgndetF = logabsdet(F, ws)
    logdetA, sgndetA = logabsdet(A)
    logdetF ≈ logdetA
    sgndetF ≈ sgndetA

    # testing ldr and copyto!
    F = ldr(B̄)
    copyto!(A, F, ws)
    @test A ≈ I

    # testing ldr and copyto!
    F = ldr(B̄, ws)
    copyto!(A, F, ws)
    @test A ≈ B̄

    # testing ldr!
    ldr!(F, B̄, ws)
    copyto!(A, F, ws)
    @test A ≈ B̄

    # testing ldrs
    n = 3
    Fs = ldrs(B̄, n)
    @testset for i in 1:n
        copyto!(A, Fs[i], ws)
        @test A ≈ I
    end

    # testing ldrs
    n = 3
    Fs = ldrs(B̄, n, ws)
    @testset for i in 1:n
        copyto!(A, Fs[i], ws)
        @test A ≈ B̄
    end

    # testing adjoint!
    A = randn(N,N)
    ldr!(F, A, ws)
    adjoint!(G, F, ws)
    @test G ≈ adjoint(A)

    # testing lmul! and inv_IpA!
    fill!(G, 0)
    copyto!(F, I, ws)
    for i in 1:Nₛ
        lmul!(B̄, F, ws)
    end
    logdetG, sgndetG = inv_IpA!(G, F, ws)
    @test G ≈ G_0
    @test sgndetG ≈ sgndetG_0
    @test logdetG ≈ logdetG_0

    # testing lmul! and inv_IpA!
    copyto!(F, I, ws)
    F_B̄ = ldr(B̄, ws)
    for i in 1:Nₛ
        lmul!(F_B̄, F, ws)
    end
    logdetG, sgndetG = inv_IpA!(G, F, ws)
    @test G ≈ G_0
    @test sgndetG ≈ sgndetG_0
    @test logdetG ≈ logdetG_0

    # testing rmul! and inv_IpA!
    copyto!(F, I, ws)
    for i in 1:Nₛ
        rmul!(F, B̄, ws)
    end
    logdetG, sgndetG = inv_IpA!(G, F, ws)
    @test G ≈ G_0
    @test sgndetG ≈ sgndetG_0
    @test logdetG ≈ logdetG_0

    # testing rmul! and inv_IpA!
    copyto!(F, I, ws)
    F_B̄ = ldr(B̄, ws)
    for i in 1:Nₛ
        rmul!(F, F_B̄, ws)
    end
    logdetG, sgndetG = inv_IpA!(G, F, ws)
    @test G ≈ G_0
    @test sgndetG ≈ sgndetG_0
    @test logdetG ≈ logdetG_0

    # testing ldiv! and inv_IpA!
    copyto!(F, I, ws)
    F_B̄⁻¹ = ldr(B̄⁻¹, ws)
    for i in 1:Nₛ
        ldiv!(F_B̄⁻¹, F, ws)
    end
    logdetG, sgndetG = inv_IpA!(G, F, ws)
    @test G ≈ G_0
    @test sgndetG ≈ sgndetG_0
    @test logdetG ≈ logdetG_0

    # testing ldiv! and inv_IpA!
    copyto!(F, I, ws)
    for i in 1:Nₛ
        ldiv!(B̄⁻¹, F, ws)
    end
    logdetG, sgndetG = inv_IpA!(G, F, ws)
    @test G ≈ G_0
    @test sgndetG ≈ sgndetG_0
    @test logdetG ≈ logdetG_0

    # testing ldiv!
    F_B̄ = ldr(B̄, ws)
    copyto!(A, B̄)
    ldiv!(F_B̄, A, ws)
    @test A ≈ I

    # testing rdiv! and inv_IpA!
    copyto!(F, I, ws)
    F_B̄⁻¹ = ldr(B̄⁻¹, ws)
    for i in 1:Nₛ
        rdiv!(F, F_B̄⁻¹, ws)
    end
    logdetG, sgndetG = inv_IpA!(G, F, ws)
    @test G ≈ G_0
    @test sgndetG ≈ sgndetG_0
    @test logdetG ≈ logdetG_0

    # testing rdiv! and inv_IpA!
    copyto!(F, I, ws)
    for i in 1:Nₛ
        rdiv!(F, B̄⁻¹, ws)
    end
    logdetG, sgndetG = inv_IpA!(G, F, ws)
    @test G ≈ G_0
    @test sgndetG ≈ sgndetG_0
    @test logdetG ≈ logdetG_0

    # testing rdiv!
    F_B̄ = ldr(B̄, ws)
    copyto!(A, B̄)
    rdiv!(A, F_B̄, ws)
    @test A ≈ I

    # testing inv_IpUV!
    copyto!(F, I, ws)
    for i in 1:Nₛ÷2
        lmul!(B̄, F, ws)
    end
    logdetG, sgndetG = inv_IpUV!(G, F, F, ws)
    @test G ≈ G_0
    @test sgndetG ≈ sgndetG_0
    @test logdetG ≈ logdetG_0

    # testing inv_UpV!
    F′ = ldr(B̄⁻¹)
    copyto!(F, I, ws)
    for i in 1:Nₛ÷2
        lmul!(B̄, F, ws)
        rmul!(F′, B̄⁻¹, ws)
    end
    logdetG, sgndetG = inv_UpV!(G, F′, F, ws)
    @test G ≈ G_βo2
    @test sgndetG ≈ sgndetG_βo2
    @test logdetG ≈ logdetG_βo2

    # testing inv_invUpV!
    copyto!(F, I, ws)
    for i in 1:Nₛ÷2
        lmul!(B̄, F, ws)
    end
    logdetG, sgndetG = inv_invUpV!(G, F, F, ws)
    @test G ≈ G_βo2
    @test sgndetG ≈ sgndetG_βo2
    @test logdetG ≈ logdetG_βo2
end
