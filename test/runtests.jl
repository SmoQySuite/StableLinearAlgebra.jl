using StableLinearAlgebra
using LinearAlgebra
using Test
using LatticeUtilities

# calculate eigenenergies and eigenstates for square lattice
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
    L     = 4 # linear system size
    t     = 1.0 # nearest neighbor hopping
    μ     = 0.0 # chemical potential
    β     = 40.0 # inverse temperature
    Δτ    = 0.1 # discretization in imaginary time
    Lτ    = round(Int,β/Δτ) # length of imaginary time axis
    nstab = 10 # stabalization frequency
    Nstab = Lτ  ÷ nstab # number of reduced propagator matrices

    # hamitlonian eigenenergies and eigenstates
    ϵ, U, H = hamiltonian(L, t, μ)

    # number of sites in lattice
    N = size(H,1)

    # propagator matrix
    B = propagator(Δτ, ϵ, U)

    # inverse propagator matrix
    B⁻¹ = propagator(-Δτ, ϵ, U)

    # greens functions
    G_0   = greens(0, β, ϵ, U) # G(τ=0)
    G_βo2 = greens(β/2, β, ϵ, U) # G(τ=β/2)

    # temporary matrices
    A = similar(B)

    # partial propagator matrix product
    B̄ = Matrix{eltype(B)}(I,N,N)
    for i in 1:nstab
        mul!(A, B, B̄)
        copyto!(B̄, A)
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
        mul!(A, B⁻¹, B̄⁻¹)
        copyto!(B̄⁻¹, A)
    end

    # test ldr(::AbstractMatrix) and copyto!
    F = ldr(B̄)
    ws = ldr_workspace(B̄)
    copyto!(A, F, ws)
    @test A ≈ B̄

    # test ldr(::LDR) and copyto!
    F′ = ldr(F)
    ws = ldr_workspace(B̄)
    copyto!(A, F′, ws)
    @test A ≈ B̄

    # testing LDR factorization with new matrix
    ldr!(F, B̄)
    copyto!(A, F, ws)
    @test A ≈ B̄

    # test copying identity matrix to existing LDR factorization
    ldr!(F, I)
    copyto!(A, F, ws)
    @test A ≈ I

    # test inv!
    F = ldr(B̄)
    inv!(A, F, ws)
    @test A ≈ B̄⁻¹

    # test ldiv!
    F = ldr(B̄)
    C = similar(A)
    copyto!(A, I)
    ldiv!(C, F, A, ws)
    @test (C * B̄) ≈ I

    # test ldiv!
    F = ldr(B̄)
    F′ = ldr(F)
    F″ = ldr(F)
    copyto!(F′, I)
    ldiv!(F″, F, F′, ws)
    copyto!(A, F″, ws)
    @test A ≈ B̄⁻¹

    # test rdiv!
    F = ldr(B̄)
    C = similar(A)
    copyto!(A, I)
    rdiv!(C, A, F, ws)
    @test (C * B̄) ≈ I

    # test rdiv!
    F = ldr(B̄)
    F′ = ldr(F)
    F″ = ldr(F)
    copyto!(F′, I)
    rdiv!(F″, F′, F, ws)
    copyto!(A, F″)
    @test (A * B̄) ≈ I

    # testing lmul!(::AbstractMatrix, ::LDR) and inv_IpA!
    F′ = ldr(F)
    ldr!(F, I)
    for n in 1:Nstab
        lmul!(B̄, F, ws)
    end
    inv_IpA!(A, F, ws, F′=F′)
    @test A ≈ G_0

    # test rmul!(::LDR, ::AbstractMatrix) and inv_IpA!
    ldr!(F, I)
    for n in 1:Nstab
        rmul!(F, B̄, ws)
    end
    inv_IpA!(A, F, ws, F′=F′)
    @test A ≈ G_0

    # test ldrs(::AbstractMatrix, ::Int), lmul!(::LDR, ::LDR) and inv_IpA!
    Fs = ldrs(B̄, Nstab)
    ldr!(F, I)
    for n in 1:Nstab
        lmul!(Fs[n], F, ws) 
    end
    inv_IpA!(A, F, ws, F′=F′)
    @test A ≈ G_0

    # test ldrs!(::Vector{LDR}, ::AbstractArray{3}), rmul!(::LDR, ::LDR) and inv_IpA!
    ldrs!(Fs, B̄s)
    ldr!(F, I)
    for n in 1:Nstab
        rmul!(F, Fs[n], ws) 
    end
    inv_IpA!(A, F, ws, F′=F′)
    @test A ≈ G_0

    # test mul!(::LDR, ::AbstractMatrix, ::LDR)
    ldr!(F, I)
    for n in 1:Nstab
        mul!(F′, B̄, F, ws)
        copyto!(F, F′)
    end
    inv_IpA!(A, F, ws, F′=F′)
    @test A ≈ G_0

    # test mul!(::LDR, ::LDR, ::AbstractMatrix)
    ldr!(F, I)
    for n in 1:Nstab
        mul!(F′, F, B̄, ws)
        copyto!(F, F′)
    end
    inv_IpA!(A, F, ws, F′=F′)
    @test A ≈ G_0

    # test mul!(::LDR, ::LDR, ::LDR)
    Fs = ldrs(B̄, Nstab)
    ldr!(F, I)
    for n in 1:Nstab
        mul!(F′, Fs[n], F, ws)
        copyto!(F, F′)
    end
    inv_IpA!(A, F, ws, F′=F′)
    @test A ≈ G_0

    # test inv_UpV!
    F″ = ldr(B)
    ldr!(F, I)
    ldr!(F′, I)
    for n in 1:Nstab÷2
        lmul!(B̄⁻¹, F′, ws)
        rmul!(F, B̄, ws)
    end
    inv_UpV!(A, F′, F, ws, F=F″)
    @test A ≈ G_βo2

    # test inv_invUpV!
    ldr!(F, I)
    ldr!(F′, I)
    for n in 1:Nstab÷2
        rmul!(F, B̄, ws)
    end
    copyto!(F′, F)
    inv_invUpV!(A, F, F′, ws, F=F″)
    @test A ≈ G_βo2

    # test determinant calculation
    A  = rand(Float64,10,10)
    ws = ldr_workspace(A)
    Fa = ldr(A)
    @test det(A) ≈ det(Fa, ws)

    # test determinant calculation
    A  = rand(Complex{Float64},10,10)
    ws = ldr_workspace(A)
    Fa = ldr(A)
    @test det(A) ≈ det(Fa, ws)

    # test determinant ratio calculation
    A  = rand(Float64,10,10)
    B  = rand(Float64,10,10)
    ws = ldr_workspace(A)
    Fa = ldr(A)
    Fb = ldr(B)
    @test abs(det(A)/det(B)) ≈ abs_det_ratio(Fa, Fb, ws)

    # test determinant ratio calculation
    A  = rand(Complex{Float64},10,10)
    B  = rand(Complex{Float64},10,10)
    ws = ldr_workspace(A)
    Fa = ldr(A)
    Fb = ldr(B)
    @test abs(det(A)/det(B)) ≈ abs_det_ratio(Fa, Fb, ws)
end
