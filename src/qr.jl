import LinearAlgebra.LAPACK: geqp3!, orgqr!

@doc raw"""
    QRWorkspace{T<:Number, E<:Real}

Allocated space for calcuating the pivoted QR factorization using the LAPACK
routines `geqp3!` and `orgqr!` while avoiding dynamic memory allocations.
"""
struct QRWorkspace{T<:Number, E<:Real}

    work::Vector{T}
    rwork::Vector{E}
    τ::Vector{T}
    jpvt::Vector{Int}
end

# copy the state of one QRWorkspace into another
function copyto!(qrws_out::QRWorkspace{T,E}, qrws_in::QRWorkspace{T,E}) where {T,E}

    copyto!(qrws_out.work, qrws_in.work)
    copyto!(qrws_out.rwork, qrws_in.rwork)
    copyto!(qrws_out.τ, qrws_in.τ)
    copyto!(qrws_out.jpvt, qrws_in.jpvt)

    return nothing
end

# wrap geqp3 and orgqr LAPACK methods
for (geqp3, orgqr, elty, relty) in ((:dgeqp3_, :dorgqr_, :Float64,    :Float64),
                                    (:sgeqp3_, :sorgqr_, :Float32,    :Float32),
                                    (:zgeqp3_, :zungqr_, :ComplexF64, :Float64),
                                    (:cgeqp3_, :cungqr_, :ComplexF32, :Float32))
    @eval begin

        # method returns QRWorkspace for given matrix A
        function QRWorkspace(A::StridedMatrix{$elty})

            # allocate for geqp3/QR decomposition calculation
            n = checksquare(A)
            require_one_based_indexing(A)
            chkstride1(A)
            A′ = copy(A)
            Rlda = max(1, stride(A′, 2))
            jpvt = zeros(BlasInt, n)
            τ = zeros($elty, n)
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()
            if $elty <: Complex
                rwork = Vector{$relty}(undef, 2n)
                ccall((@blasfunc($geqp3), liblapack), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}),
                      n, n, A′, Rlda, jpvt, τ, work, lwork, rwork, info)
            else
                rwork = Vector{$relty}(undef, 0)
                ccall((@blasfunc($geqp3), liblapack), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}),
                      n, n, A′, Rlda, jpvt, τ, work, lwork, info)
            end
            chklapackerror(info[])
            lwork = BlasInt(real(work[1]))
            resize!(work, lwork)

            return QRWorkspace(work, rwork, τ, jpvt)
        end

        # method for calculating QR decomposition
        function geqp3!(A::AbstractMatrix{$elty}, ws::QRWorkspace{$elty})

            require_one_based_indexing(A)
            chkstride1(A)
            n = checksquare(A)
            lda = stride(A, 2)
            info = Ref{BlasInt}()
            fill!(ws.jpvt, 0) # not sure why I need to do this, but I do
            if $elty <: Complex
                ccall((@blasfunc($geqp3), liblapack), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}),
                      n, n, A, lda, ws.jpvt, ws.τ, ws.work, length(ws.work), ws.rwork, info)
            else
                ccall((@blasfunc($geqp3), liblapack), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}),
                      n, n, A, lda, ws.jpvt, ws.τ, ws.work, length(ws.work), info)
            end
            chklapackerror(info[])

            return nothing
        end

        # method for constructing Q matrix
        function orgqr!(A::AbstractMatrix{$elty}, ws::QRWorkspace{$elty})

            require_one_based_indexing(A, ws.τ)
            chkstride1(A, ws.τ)
            n = checksquare(A)
            k = length(ws.τ)
            info = Ref{BlasInt}()
            ccall((@blasfunc($orgqr), liblapack), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                  n, n, k, A, max(1,stride(A,2)), ws.τ, ws.work, length(ws.work), info)
            chklapackerror(info[])

            return nothing
        end
    end
end