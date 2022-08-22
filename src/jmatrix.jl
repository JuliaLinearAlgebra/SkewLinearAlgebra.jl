
struct JMatrix{T, N<:Integer, SGN} <: AbstractMatrix{T}
    n::N    #size of the square matrix
    sgn::SGN #+-1, allows to transpose,invert easily the matrix.
    function JMatrix{T, N, SGN}(n, sgn) where {T, N<:Integer, SGN}
        (sgn == T(1) || sgn == T(-1) ) || throw("sgn argument must be +-1")
        new{T, N, SGN}(n,sgn)
    end
end

JMatrix(T::Type, n::Integer) = JMatrix{T,typeof(n),Any}(n, T(1))
JMatrix(T::Type, n::Integer, sgn) = JMatrix{T,typeof(n), Any}(n, T(sgn))

Base.similar(J::JMatrix,::Type{T}) where {T} = JMatrix(T, J.n, J.sgn)
Base.similar(J::JMatrix, ::Type{T}, dims::Union{Dims{1},Dims{2}}) where {T} = zeros(T, dims...)
Base.copy(J::JMatrix{T}) where T = JMatrix(T, J.n, J.sgn)

Base.size(J::JMatrix) = (J.n, J.n)
function Base.size(J::JMatrix, n::Integer)
    if n == 1 || n == 2
        return J.n
    else 
        return 1
    end
end

function Base.Matrix(J::JMatrix{T}) where {T}
    M = similar(J, T, J.n, J.n)
    for i =1 : 2 : J.n-1
        M[i+1,i] = -J.sgn
        M[i,i+1] = J.sgn
    end
    return M
end

Base.Array(A::JMatrix) = Matrix(A)

function SkewHermTridiagonal(J::JMatrix{T}) where {T}
    vec = zeros(T, J.n - 1)
    for i = 1 : 2 : J.n - 1
        vec[i] = -1
    end
    return SkewHermTridiagonal(vec)
end

Base.@propagate_inbounds function Base.getindex(J::JMatrix{T}, i::Integer, j::Integer) where T

    @boundscheck checkbounds(J, i, j)
    if i == j + 1 && i%2 == 0
        return -J.sgn
    elseif i + 1 == j && j%2 == 0
        return J.sgn
    else
        return zero(T)
    end
end
#=
function Base.:*(J::JMatrix,A::AbstractVecOrMat)
    m, k = size(A, 1), size(1, 2)
    if !(m == J.n)
        throw(DimensionMismatch("A has first dimension $(size(S,1)), B has $(size(B,1)), C has $(size(C,1)) but all must match"))
    end
    B = similar(A,J.n, k)
    for i = 1 : 2 : J.n-1
        B[i,:] .= A[i+1,:]
        B[i+1,:] .= -A[i,:]
    end
    return B
end
function Base.:*(A::AbstractVecOrMat, J::JMatrix)
    m, k = size(A, 1), size(1, 2)
    if !(k == J.n)
        throw(DimensionMismatch("A has first dimension $(size(S,1)), B has $(size(B,1)), C has $(size(C,1)) but all must match"))
    end
    B = similar(A,m, J.n)
    for i = 1 : 2 : J.n-1
        B[:,i] .= A[:,i+1]
        B[:, i+1] .= -A[:, i]
    end
    return B
end

\(J::JMatrix,A::AbstractVecOrMat) = - J * A
=#
Base.:-(J::JMatrix{T}) where T = JMatrix(T, J.n, -J.sgn)
LA.transpose(J::JMatrix) = -J
LA.adjoint(J::JMatrix) = -J
function LA.inv(J::JMatrix) 
        iszero(J.n %2) ||throw(SingularException)
        return -J
end
LA.diag(J::JMatrix{T}) where T = zeros(T, J.n)
LA.tr(J::JMatrix{T}) where T = zero(T)


        