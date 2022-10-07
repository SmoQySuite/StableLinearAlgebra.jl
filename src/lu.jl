import LinearAlgebra.LAPACK: getrf!, getri!, getrs!

@doc raw"""
    LUWorkspace{T<:Number, E<:Real}

Allocated space for calcuating the pivoted QR factorization using the LAPACK
routine `getrf!`. Also interfaces with the `getri!` and `getrs!` routines for
inverting matrices and solving linear systems respectively.
"""
struct LUWorkspace{T<:Number}

    work::Vector{T}
    ipiv::Vector{Int}
end

# wrap geqp3 and orgqr LAPACK methods
for (getrf, getri, getrs, elty, relty) in ((:dgetrf_, :dgetri_, :dgetrs_, :Float64,    :Float64),
                                           (:sgetrf_, :sgetri_, :sgetrs_, :Float32,    :Float32),
                                           (:zgetrf_, :zgetri_, :zgetrs_, :ComplexF64, :Float64),
                                           (:cgetrf_, :cgetri_, :cgetrs_, :ComplexF32, :Float32))

    @eval begin

        # returns LUWorkspace
        function LUWorkspace(A::StridedMatrix{$elty})

            # calculate LU factorization
            n = checksquare(A)
            require_one_based_indexing(A)
            chkstride1(A)
            A′ = copy(A)
            lda = max(1, stride(A′, 2))
            info = Ref{BlasInt}()
            ipiv = zeros(Int, n)
            ccall((@blasfunc($getrf), liblapack), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                  n, n, A′, lda, ipiv, info)
            chklapackerror(info[])
            # perform matrix inversion method once to resize workspace
            lwork = BlasInt(-1)
            work = Vector{$elty}(undef, 1)
            ccall((@blasfunc($getri), liblapack), Cvoid,
                  (Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                  n, A, lda, ipiv, work, lwork, info)
            chklapackerror(info[])
            lwork = BlasInt(real(work[1]))
            resize!(work, lwork)
            return LUWorkspace(work, ipiv)
        end
        
        # calculates LU factorization
        function getrf!(A::AbstractMatrix{$elty}, ws::LUWorkspace{$elty})

            require_one_based_indexing(A)
            chkstride1(A)
            n = checksquare(A)
            lda = max(1, stride(A, 2))
            info = Ref{BlasInt}()
            fill!(ws.ipiv, 0)
            ccall((@blasfunc($getrf), liblapack), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                  n, n, A, lda, ws.ipiv, info)
            chklapackerror(info[])
            return nothing
        end

        # calculate matrix inverse of LU factorization
        function getri!(A::AbstractMatrix{$elty}, ws::LUWorkspace{$elty})

            require_one_based_indexing(A, ws.ipiv)
            chkstride1(A, ws.ipiv)
            n = checksquare(A)
            lda = max(1,stride(A, 2))
            info = Ref{BlasInt}()
            ccall((@blasfunc($getri), liblapack), Cvoid,
                    (Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                    Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                    n, A, lda, ws.ipiv, ws.work, length(ws.work), info)
            chklapackerror(info[])
            return nothing
        end

        # solve the linear system A⋅X = B, where A is represented by LU factorization and B is overwritten in-place
        function getrs!(A::AbstractMatrix{$elty}, B::AbstractVecOrMat{$elty}, ws::LUWorkspace{$elty}, trans::AbstractChar='N')

            require_one_based_indexing(A, B)
            chktrans(trans)
            chkstride1(A, B)
            n = checksquare(A)
            @assert n == length(ws.ipiv) WorkspaceSizeError(length(ws.ipiv), n)
            if n != size(B, 1)
                throw(DimensionMismatch("B has leading dimension $(size(B,1)), but needs $n"))
            end
            info = Ref{BlasInt}()
            ccall((@blasfunc($getrs), liblapack), Cvoid,
                  (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Clong),
                  trans, n, size(B,2), A, max(1,stride(A,2)), ws.ipiv, B, max(1,stride(B,2)), info, 1)
            chklapackerror(info[])
            return nothing
        end
    end
end

@doc raw"""
    det_lu!(A::AbstractMatrix{T}, ws::LUWorkspace) where {T}

Return ``\log(|\det A|)`` and ``\textrm{sign}(\det A).``
Note that ``A`` is left modified by this function.
"""
function det_lu!(A::AbstractMatrix{T}, ws::LUWorkspace{T}) where {T}

    # calculate LU factorization
    LAPACK.getrf!(A, ws)

    # calculate det(A)
    logdetA = zero(real(T)) # logdetA = 0
    sgndetA = oneunit(T) # sgndetA = 1
    @fastmath @inbounds for i in axes(A,1)
        logdetA += log(abs(A[i,i]))
        sgndetA *= sign(A[i,i])
        if i != ws.ipiv[i]
            sgndetA = -sgndetA
        end
    end

    return logdetA, sgndetA
end


@doc raw"""
    inv_lu!(A::AbstractMatrix{T}, ws::LUWorkspace{T}) where {T}

Calculate the inverse of the matrix `A`, overwriting `A` in-place.
Also return ``\log(|\det A^{-1}|)`` and ``\textrm{sign}(\det A^{-1}).``
"""
function inv_lu!(A::AbstractMatrix{T}, ws::LUWorkspace{T}) where {T}

    # calculate LU factorization of A and determinant at the same time
    logdetA, sgndetA = det_lu!(A, ws)

    # calculate matrix inverse
    LAPACK.getri!(A, ws)

    return -logdetA, conj(sgndetA)
end


@doc raw"""
    ldiv_lu!(A::AbstractMatrix{T}, B::AbstractMatrix{T}, ws::LUWorkspace{T}) where {T}

Calculate ``B:= A^{-1} B,`` modifying ``B`` in-place. The matrix ``A`` is over-written as well.
Also return ``\log(|\det A^{-1}|)`` and ``\textrm{sign}(\det A^{-1}).``
""" 
function ldiv_lu!(A::AbstractMatrix{T}, B::AbstractMatrix{T}, ws::LUWorkspace{T}) where {T}

    # calculate LU factorization of A and determinant at the same time
    logdetA, sgndetA = det_lu!(A, ws)

    # calculate the matrix product A⁻¹⋅B
    LAPACK.getrs!(A, B, ws)

    return -logdetA, conj(sgndetA)
end