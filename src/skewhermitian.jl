# This file is a part of Julia. License is MIT: https://julialang.org/license

struct SkewHermitian{T<:Number,S<:AbstractMatrix{<:T}} <: AbstractMatrix{T}
    data::S

    function SkewHermitian{T,S}(data) where {T,S<:AbstractMatrix{<:T}}
        LA.require_one_based_indexing(data)
        new{T,S}(data)
    end
end

"""
    SkewHermitian(A) <: AbstractMatrix

Construct a `SkewHermitian` view of the skew-Hermitian matrix `A` (`A == -A'`),
which allows one to exploit efficient operations for eigenvalues, exponentiation,
and more.

Takes "ownership" of the matrix `A`.  See also [`skewhermitian`](@ref), which takes the
skew-hermitian part of `A`, and [`skewhermitian!`](@ref), which does this in-place,
along with [`isskewhermitian`](@ref) which checks whether `A == -A'`.
"""
function SkewHermitian(A::AbstractMatrix)
    isskewhermitian(A) || throw(ArgumentError("matrix `A` must be skew-Hermitian (equal `-A')"))
    return SkewHermitian{eltype(A),typeof(A)}(A)
end

"""
    skewhermitian(A)

Returns the skew-Hermitian part of A, i.e. `(A-A')/2`.  See also
[`skewhermitian!`](@ref), which does this in-place.
"""
skewhermitian(A::AbstractMatrix) = skewhermitian!(Base.copymutable(A))
skewhermitian(a::Number) = imag(a)

Base.@propagate_inbounds Base.getindex(A::SkewHermitian, i::Integer, j::Integer) = A.data[i,j]
Base.@propagate_inbounds function Base.setindex!(A::SkewHermitian, v, i::Integer, j::Integer)
    if i == j
        real(v) == 0 || throw(ArgumentError("diagonal elements must be zero"))
    else
        A.data[i,j] = v
        A.data[j,i] = -v'
    end
    return v
end

Base.similar(A::SkewHermitian, ::Type{T}) where {T} = SkewHermitian(similar(parent(A), T) .= 0)
Base.similar(A::SkewHermitian) = SkewHermitian(similar(parent(A)) .= 0)

# Conversion
Base.Matrix(A::SkewHermitian) = Matrix(A.data)
Base.Array(A::SkewHermitian) = Matrix(A)

Base.parent(A::SkewHermitian) = A.data
SkewHermitian{T,S}(A::SkewHermitian{T,S}) where {T,S} = A
SkewHermitian{T,S}(A::SkewHermitian) where {T,S<:AbstractMatrix{T}} = SkewHermitian{T,S}(S(A.data))
Base.AbstractMatrix{T}(A::SkewHermitian) where {T} = SkewHermitian(AbstractMatrix{T}(A.data))

Base.copy(A::SkewHermitian) = SkewHermitian(copy(A.data))
function Base.copyto!(dest::SkewHermitian, src::SkewHermitian)
    copyto!(dest.data, src.data)
    return dest
end
function Base.copyto!(dest::SkewHermitian, src::AbstractMatrix)
    isskewhermitian(src) || throw(ArgumentError("can only copy skew-Hermitian data to SkewHermitian"))
    copyto!(dest.data, src)
    return dest
end
Base.copyto!(dest::AbstractMatrix, src::SkewHermitian) = copyto!(dest, src.data)

Base.size(A::SkewHermitian,n) = size(A.data,n)
Base.size(A::SkewHermitian) = size(A.data)

"""
    isskewhermitian(A)

Returns whether `A` is skew-Hermitian, i.e. whether `A == -A'`.
"""
function isskewhermitian(A::AbstractMatrix{<:Number})
    axes(A,1) == axes(A,2) || throw(ArgumentError("axes $(axes(A,1)) and $(axex(A,2)) do not match"))
    @inbounds for i in axes(A,1)
        for j = firstindex(A, 1):i
            A[i,j] == -A[j,i]' || return false
        end
    end
    return true
end
isskewhermitian(A::SkewHermitian) = true
isskewhermitian(a::Number) = a == -a'

"""
    skewhermitian!(A)

Transforms `A` in-place to its skew-Hermitian part `(A-A')/2`,
and returns a [`SkewHermitian`](@ref) view.
"""
function skewhermitian!(A::AbstractMatrix{T}) where {T<:Number}
    LA.require_one_based_indexing(A)
    n = LA.checksquare(A)
    two = T(2)
    @inbounds for i in 1:n
        A[i,i] = T isa Real ? zero(T) : complex(zero(real(T)),imag(A[i,i]))
        for j = 1:i-1
            a = (A[i,j] - A[j,i]')/two
            A[i,j] = a
            A[j,i] = -a'
        end
    end
    return SkewHermitian(A)
end
LA.Tridiagonal(A::SkewHermitian) = Tridiagonal(A.data)

Base.isreal(A::SkewHermitian) = isreal(A.data)
Base.transpose(A::SkewHermitian) = SkewHermitian(transpose(A.data))
Base.adjoint(A::SkewHermitian) = SkewHermitian(A.data')
Base.real(A::SkewHermitian{<:Real}) = A
Base.real(A::SkewHermitian) = SkewHermitian(real(A.data))
Base.imag(A::SkewHermitian) = LA.Hermitian(imag(A.data))

Base.conj(A::SkewHermitian) = SkewHermitian(conj(A.data))
Base.conj!(A::SkewHermitian) = SkewHermitian(conj!(A.data))
LA.tr(A::SkewHermitian{<:Real}) = zero(eltype(A))
LA.tr(A::SkewHermitian) = tr(A.data)

LA.tril!(A::SkewHermitian) = tril!(A.data)
LA.tril(A::SkewHermitian)  = tril!(copy(A))
LA.triu!(A::SkewHermitian) = triu!(A.data)
LA.triu(A::SkewHermitian)  = triu!(copy(A))
LA.tril!(A::SkewHermitian,k::Integer) = tril!(A.data,k)
LA.tril(A::SkewHermitian,k::Integer)  = tril!(copy(A),k)
LA.triu!(A::SkewHermitian,k::Integer) = triu!(A.data,k)
LA.triu(A::SkewHermitian,k::Integer)  = triu!(copy(A),k)

function LA.dot(A::SkewHermitian, B::SkewHermitian)
    n = size(A, 2)
    T = eltype(A)
    two = T(2)

    if n != size(B, 2)
        throw(DimensionMismatch("A has size $(size(A)) but B has size $(size(B))"))
    end
    dotprod = zero(dot(first(A), first(B)))
    @inbounds for j = 1:n 
        for i = 1:j-1
            dotprod += two * dot(A.data[i, j], B.data[i, j])
        end
        dotprod += dot(A.data[j, j], B.data[j, j])
    end
    return dotprod
end

Base.:-(A::SkewHermitian) = SkewHermitian(-A.data)

for f in (:+, :-)
    @eval begin
        Base.$f(A::SkewHermitian, B::SkewHermitian) = SkewHermitian($f(A.data, B.data))
   end
end

## Matvec
LA.mul!(y::StridedVecOrMat, A::SkewHermitian, x::StridedVecOrMat, α::Number, β::Number) =
    LA.mul!(y, A.data, x, α, β)
LA.mul!(y::StridedVecOrMat, A::SkewHermitian, x::StridedVecOrMat) =
    LA.mul!(y, A.data, x)
LA.mul!(y::StridedVecOrMat, A::SkewHermitian, B::SkewHermitian, α::Number, β::Number) =
    LA.mul!(y, A.data, B.data, α, β)
LA.mul!(y::StridedVecOrMat, A::SkewHermitian, B::SkewHermitian) =
    LA.mul!(y, A.data, B.data)
LA.mul!(y::StridedVecOrMat, A::StridedMatrix, B::SkewHermitian, α::Number, β::Number) =
    LA.mul!(y, A, B.data, α, β)
LA.mul!(y::StridedVecOrMat, A::StridedMatrix, B::SkewHermitian) =
    LA.mul!(y, A, B.data)

function LA.dot(x::AbstractVector, A::SkewHermitian, y::AbstractVector)
    LA.require_one_based_indexing(x, y)
    (length(x) == length(y) == size(A, 1)) || throw(DimensionMismatch())
    r = zero(eltype(x)) * zero(eltype(A)) * zero(eltype(y))
    @inbounds for j = 1:length(y)
        r += dot(x[j], A.data[j,j], y[j]) # zero if A is real
        @simd for i = 1:j-1
            Aij = A.data[i,j]
            r += dot(x[i], Aij, y[j]) + dot(x[j], -Aij', y[i])
        end
    end
    return r
end

# Scaling:
for op in (:*, :/, :\)
    if op in (:*, :/)
        @eval Base.$op(A::SkewHermitian, x::Number) = $op(A.data, x)
        @eval Base.$op(A::SkewHermitian, x::Real) = SkewHermitian($op(A.data, x))
    end
    if op in (:*, :\)
        @eval Base.$op(x::Number, A::SkewHermitian) = $op(x, A.data)
        @eval Base.$op(x::Real, A::SkewHermitian) = SkewHermitian($op(x, A.data))
    end
end
function checkreal(x::Number)
    isreal(x) || throw(ArgumentError("in-place scaling of SkewHermitian requires a real scalar"))
    return real(x)
end
LA.rdiv!(A::SkewHermitian, b::Number) = rdiv!(A.data, checkreal(b))
LA.ldiv!(b::Number, A::SkewHermitian) = ldiv!(checkreal(b), A.data)
LA.rmul!(A::SkewHermitian, b::Number) = rmul!(A.data, checkreal(b))
LA.lmul!(b::Number, A::SkewHermitian) = lmul!(checkreal(b), A.data)

for f in (:det, :logdet, :lu, :lu!, :lq, :lq!, :qr, :qr!)
    @eval LA.$f(A::SkewHermitian) = LA.$f(A.data)
end
@eval LA.inv(A::SkewHermitian) = skewhermitian!(LA.inv(A.data))

LA.kron(A::SkewHermitian,B::StridedMatrix) = kron(A.data,B)
LA.kron(A::StridedMatrix,B::SkewHermitian) = kron(A,B.data)

@views function LA.schur!(A::SkewHermitian{<:Real})
    F = eigen!(A)
    return Schur(typeof(F.vectors)(Diagonal(F.values)), F.vectors, F.values)

end
LA.schur(A::SkewHermitian{<:Real})= LA.schur!(copyeigtype(A))