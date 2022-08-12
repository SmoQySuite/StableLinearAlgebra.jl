using Base: require_one_based_indexing
using LinearAlgebra: BlasInt, BlasFloat
using LinearAlgebra.BLAS: @blasfunc
using LinearAlgebra.LAPACK: chklapackerror, chkstride1
using LinearAlgebra.LAPACK

const liblapack = "libblastrampoline"

# pre-allocated work-space for calculating QR decomposition without any memory allocations
struct QRWorkspace{T<:Number, E<:Real}

    geqp3_work::Vector{T}

    geqp3_rwork::Vector{E}

    orgqr_work::Vector{T}

    τ::Vector{T}

    jpvt::Vector{Int}

    info::Base.RefValue{Int}
end

# wrap geqp3 and orgqr LAPACK methods
for (geqp3, orgqr, elty, relty) in ((:dgeqp3_, :dorgqr_, :Float64, :Float64),
                             (:sgeqp3_, :sorgqr_, :Float32, Float32),
                             (:zgeqp3_, :zungqr_, :ComplexF64, Float64),
                             (:cgeqp3_, :cungqr_, :ComplexF32, Float32))
    @eval begin

        function QRWorkspace(A::StridedMatrix{$elty})

            # allocate for geqp3
            require_one_based_indexing(A)
            chkstride1(A)
            m = size(A, 1)
            n = size(A, 2)
            Rlda = max(1, stride(A, 2))
            jpvt = zeros(BlasInt, n)
            τ = zeros($elty, min(m, n))
            geqp3_work = Vector{$elty}(undef, 1)
            geqp3_rwork = Vector{$relty}(undef, 2n)
            info = Ref{BlasInt}()
            if eltype(A) <: Complex
                ccall((@blasfunc($geqp3), liblapack), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}),
                      m, n, A, Rlda, jpvt, τ, geqp3_work, -1, geqp3_rwork, info)
            else
                ccall((@blasfunc($geqp3), liblapack), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}),
                      m, n, A, Rlda, jpvt, τ, geqp3_work, -1, info)
            end
            chklapackerror(info[])
            resize!(geqp3_work, BlasInt(real(geqp3_work[1])))

            # allocate for orgqr
            k = length(τ)
            orgqr_work  = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            ccall((@blasfunc($orgqr), liblapack), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                  m, n, k, A, Rlda, τ, orgqr_work, -1, info)
            chklapackerror(info[])
            resize!(orgqr_work, BlasInt(real(orgqr_work[1])))

            return QRWorkspace(geqp3_work, geqp3_rwork, orgqr_work, τ, jpvt, info)
        end

        function geqp3!(A::AbstractMatrix{$elty}, ws::QRWorkspace{$elty})

            m = size(A, 1)
            n = size(A, 2)
            lda = stride(A, 2)
            if eltype(A) <: Complex
                ccall((@blasfunc($geqp3), liblapack), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}),
                      m, n, A, lda, ws.jpvt, ws.τ, ws.geqp3_work, length(ws.geqp3_work), ws.geqp3_rwork, ws.info)
            else
                ccall((@blasfunc($geqp3), liblapack), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}),
                      m, n, A, lda, ws.jpvt, ws.τ, ws.geqp3_work, length(ws.geqp3_work), ws.info)
            end
            chklapackerror(ws.info[])

            return nothing
        end

        function orgqr!(A::AbstractMatrix{$elty}, ws::QRWorkspace{$elty})

            require_one_based_indexing(A, ws.τ)
            chkstride1(A, ws.τ)
            k = length(ws.τ)
            m = size(A, 1)
            n = min(m, size(A, 2))
            ccall((@blasfunc($orgqr), liblapack), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                  m, n, k, A, max(1,stride(A,2)), ws.τ, ws.orgqr_work, length(ws.orgqr_work), ws.info)
            chklapackerror(ws.info[])

            return nothing
        end
    end
end