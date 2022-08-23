

"""
JMatrix{T, ±1}(n)

Creates an `AbstractMatrix{T}` of size `n x n`, representing a
block-diagonal matrix whose diagonal blocks are `±[0 1; -1 0]`.
If `n` is odd, then the last block is the `1 x 1` zero block.
The `±1` parameter allows us to transpose and invert the matrix,
and corresponds to an overall multiplicative sign factor.
"""
struct JMatrix{T<:Real, SGN} <: AbstractMatrix{T}
    n::Int    # size of the square matrix
    function JMatrix{T, SGN}(n::Integer) where {T, SGN}
        n ≥ 0 || throw("size $n must be ≥ 0")
        (SGN === +1 || SGN === -1) || throw("SGN parameter must be ±1")
        new{T, SGN}(n)
    end
end

Base.size(J::JMatrix) = (J.n, J.n)
Base.size(J::JMatrix, n::Integer) = n in (1,2) ? J.n : 1

function Base.Matrix(J::JMatrix{T, SGN}) where {T, SGN}
    M = zeros(T, J.n, J.n)
    for i = 1:2:J.n-1
        M[i+1,i] = -SGN
        M[i,i+1] = SGN
    end
    return M
end

Base.Array(A::JMatrix) = Matrix(A)

function SkewHermTridiagonal(J::JMatrix{T, SGN}) where {T, SGN}
    ev = zeros(T, J.n-1)
    ev[1:2:end] .= -SGN
    return SkewHermTridiagonal(ev)
end

Base.@propagate_inbounds function Base.getindex(J::JMatrix{T, SGN}, i::Integer, j::Integer) where {T, SGN}
    @boundscheck checkbounds(J, i, j)
    if i == j + 1 && iseven(i)
        return T(-SGN)
    elseif i + 1 == j && iseven(j)
        return T(SGN)
    else
        return zero(T)
    end
end

function Base.:*(J::JMatrix{T,SGN}, A::StridedVecOrMat) where {T,SGN}
    LA.require_one_based_indexing(A)
    m, k = size(A, 1), size(A, 2)
    if m != J.n
        throw(DimensionMismatch("J has second dimension $(size(J,2)), A has first dimension $(size(A,1))"))
    end
    B = similar(A, typeof(one(T) * oneunit(eltype(A))), J.n, k)
    @inbounds for j = 1:k, i = 1:2:J.n-1
        B[i,j] = SGN * A[i+1,j]
        B[i+1,j] = (-SGN) * A[i,j]
    end
    if isodd(J.n)
        B[J.n,:] .= 0
    end
    return B
end

function Base.:*(A::StridedVecOrMat, J::JMatrix{T,SGN}) where {T,SGN}
    LA.require_one_based_indexing(A)
    m, k = size(A, 1), size(A, 2)
    if k != J.n
        throw(DimensionMismatch("A has second dimension $(size(A,2)), J has first dimension $(size(J,1))"))
    end
    B = similar(A, typeof(one(T) * oneunit(eltype(A))), m, J.n)
    @inbounds for i = 1:2:J.n-1, j = 1:m
        B[j,i] = (-SGN) * A[j,i+1]
        B[j,i+1] = SGN * A[j,i]
    end
    if isodd(J.n)
         B[:,J.n] .= 0
    end
    return B
end

Base.:\(J::JMatrix, A::StridedVecOrMat) = inv(J) * A
Base.:/(A::StridedVecOrMat, J::JMatrix) =  A * inv(J)

Base.:-(J::JMatrix{T,+1}) where T = JMatrix{T,-1}(J.n)
Base.:-(J::JMatrix{T,-1}) where T = JMatrix{T,+1}(J.n)
LA.transpose(J::JMatrix) = -J
LA.adjoint(J::JMatrix) = -J
function LA.inv(J::JMatrix)
    iseven(J.n) || throw(LA.SingularException(J.n))
    return -J
end
LA.tr(J::JMatrix{T}) where T = zero(T)
LA.det(J::JMatrix{T}) where T = T(iseven(J.n))

function LA.diag(J::JMatrix{T,SGN}, k::Integer=0) where {T,SGN}
    v = zeros(T, max(0, J.n - abs(k)))
    if k == 1
        v[1:2:J.n-1] .= SGN
    elseif k == -1
        v[1:2:J.n-1] .= -SGN
    end
    return v
end