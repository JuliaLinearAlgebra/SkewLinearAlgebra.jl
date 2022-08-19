
struct JMatrix{T<:Number,N<:Integer} <: AbstractMatrix{T}
    n::N
    function JMatrix{T,N}(n) where {T<:Number, N<:Integer}
        new{T,N}(n)
    end
end

JMatrix(T::Type, n::Integer) = JMatrix{T,typeof(n)}(n)
Base.similar(J::JMatrix,::Type{T}) where {T} = JMatrix(T, J.n)
Base.similar(J::JMatrix, ::Type{T}, dims::Union{Dims{1},Dims{2}}) where {T} = zeros(T, dims...)

function Base.Matrix(J::JMatrix{T}) where {T}
    M = similar(J, T, J.n, J.n)
    for i =1 : 2 : J.n-1
        M[i+1,i] = -1
        M[i,i+1] = 1
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
Base.:\(J::JMatrix,A::AbstractVecOrMat) = - J * A

