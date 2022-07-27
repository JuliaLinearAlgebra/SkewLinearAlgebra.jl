using LinearAlgebra
# This file is a part of Julia. License is MIT: https://julialang.org/license

# Real SkewSymmetric and Hermitian matrices
struct SkewSymmetric{T<:Real,S<:AbstractMatrix{<:T}} <: AbstractMatrix{T}
    data::S

    function SkewSymmetric{T,S}(data) where {T,S<:AbstractMatrix{<:T}}
        LinearAlgebra.require_one_based_indexing(data)
        new{T,S}(data)
    end
end
"""
SkewSymmetric(A)
Construct a `SkewSymmetric` view of the upper (if `uplo = :U`) or lower (if `uplo = :L`)
triangle of the matrix `A`.
# Examples
```jldoctest
julia> A = [0 0 2 0 3; 0 0 0 5 0; 6 0 0 0 8; 0 9 0 0 0; 2 0 3 0 0]
5×5 Matrix{Int64}:
0  0  2  0  3
0  0  0  5  0
6  0  0  0  8
0  9  0  0  0
2  0  3  0  0
julia> Supper = SkewSymmetric(A)
5×5 SkewSymmetric{Int64, Matrix{Int64}}:
0  0  2  0  3
0  0  0  5  0
-2  0  0  0  8
0  -5  0  0  0
-3  0  -8  0  0
julia> Slower = SkewSymmetric(A, :L)
5×5 SkewSymmetric{Int64, Matrix{Int64}}:
0  0 -6  0 -2
0  0  0 -9  0
6  0  0  0 -3
0  9  0  0  0
2  0  3  0  0
```
Note that `Supper` will not be equal to `Slower` unless `A` is itself symmetric (e.g. if `A == transpose(A)`).
"""
function SkewSymmetric(A::AbstractMatrix)
    LinearAlgebra.checksquare(A)
    return skewsymmetric_type(typeof(A))(A)
end

"""
skewsymmetric(A)
Construct a skewsymmetric view of `A`. If `A` is a matrix, `uplo` controls whether the upper
(if `uplo = :U`) or lower (if `uplo = :L`) triangle of `A` is used to implicitly fill the
other one. If `A` is a `Number`, it is returned as is.
If a skewsymmetric view of a matrix is to be constructed of which the elements are neither
matrices nor numbers, an appropriate method of `symmetric` has to be implemented. In that
case, `symmetric_type` has to be implemented, too.
"""
skewsymmetric(A::AbstractMatrix) = SkewSymmetric(A)
skewsymmetric(A::Number) = A

"""
symmetric_type(T::Type)
The type of the object returned by `symmetric(::T, ::Symbol)`. For matrices, this is an
appropriately typed `Symmetric`, for `Number`s, it is the original type. If `symmetric` is
implemented for a custom type, so should be `symmetric_type`, and vice versa.
"""
function skewsymmetric_type(::Type{T}) where {S, T<:AbstractMatrix{S}}
    return SkewSymmetric{Union{S, promote_op(transpose, S), skewsymmetric_type(S)}, T}
end
function skewsymmetric_type(::Type{T}) where {S<:Number, T<:AbstractMatrix{S}}
    return SkewSymmetric{S, T}
end
function skewsymmetric_type(::Type{T}) where {S<:AbstractMatrix, T<:AbstractMatrix{S}}
    return SkewSymmetric{AbstractMatrix, T}
end
skewsymmetric_type(::Type{T}) where {T<:Number} = T

@inline function getindex(A::SkewSymmetric, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    return @inbounds A.data[i,j]
end
# For A<:Union{SkewSymmetric}, similar(A[, neweltype]) should yield a matrix with the same
# skewsymmetry type, uplo flag, and underlying storage type as A. The following methods cover these cases.
similar(A::SkewSymmetric, ::Type{T}) where {T} = SkewSymmetric(similar(parent(A), T))

# On the other hand, similar(A, [neweltype,] shape...) should yield a matrix of the underlying
# storage type of A (not wrapped in a symmetry type). The following method covers these cases.
"""
similar(A::Union{Symmetric,Hermitian}, ::Type{T}, dims::Dims{N}) where {T,N} = similar(parent(A), T, dims)
"""
# Conversion
function Matrix(A::SkewSymmetric)
    B = copy(A.data)
    return B
end
Array(A::SkewSymmetric) = convert(Matrix, A)

parent(A::SkewSymmetric) = A.data
SkewSymmetric{T,S}(A::SkewSymmetric{T,S}) where {T,S<:AbstractMatrix{T}} = A
SkewSymmetric{T,S}(A::SkewSymmetric) where {T,S<:AbstractMatrix{T}} = SkewSymmetric{T,S}(convert(S,A.data))
AbstractMatrix{T}(A::SkewSymmetric) where {T} = SkewSymmetric(convert(AbstractMatrix{T}, A.data))

copy(A::SkewSymmetric{T,S}) where {T,S} = (B = copy(A.data); SkewSymmetric{T,typeof(B)}(B))

function copyto!(dest::SkewSymmetric, src::SkewSymmetric)
    copyto!(dest.data, src.data)
    return dest
end

function skewsize(A::SkewSymmetric,n::Integer=1)
    return LinearAlgebra.size(A.data,n)
end
# fill[stored]!
fill!(A::SkewSymmetric,uplo::Char, x) = fillstored!(A,uplo, x)
function fillstored!(A::SkewSymmetric{T},uplo::Char, x) where T
    xT = convert(T, x)
    if uplo != 'U' &&uplo !='L'
        throw(ArgumentError("uplo must be 'U' or 'L'"))
    end
    if uplo == 'U'
        fillband!(A.data, xT, 0, skewsize(A,2)-1)
        fillband!(-A.data, xT, skewsize(A,2)-1,0)
    else # A.uplo == 'L'
        fillband!(A.data, xT, 1-skewsize(A,1), 0)
        fillband!(-A.data, xT,0, 1-skewsize(A,1))
    end
    return A
end

isskewsymmetric(A::SkewSymmetric) = true

#adjoint(A::Hermitian) = A
transpose(A::SkewSymmetric) = -A
adjoint(A::SkewSymmetric{<:Real}) = -A
adjoint(A::SkewSymmetric) = -A

real(A::SkewSymmetric{<:Real}) = A
real(A::SkewSymmetric) = A#SkewSymmetric(real(A.data), sym_uplo(A.uplo))
imag(A::SkewSymmetric) = SkewSymmetric(imag(A.data))

Base.copy(A::Adjoint{<:Any,<:SkewSymmetric}) =
SkewSymmetric(copy(adjoint(A.parent.data)))

tr(A::SkewSymmetric) = 0#real(tr(A.data))

Base.conj(A::SkewSymmetric) = typeof(A)(A.data)
Base.conj!(A::SkewSymmetric) = A#typeof(A)(conj!(A.data), A.uplo)

# tril/triu
function tril(A::SkewSymmetric, k::Integer=0)
    """
    if A.uplo == 'U' && k <= 0
        return tril!(copy(transpose(A.data)),k)
    elseif A.uplo == 'U' && k > 0
        return tril!(copy(transpose(A.data)),-1) + tril!(triu(A.data),k)
    elseif A.uplo == 'L' && k <= 0
        return tril(A.data,k)
    else
        return tril(A.data,-1) + tril!(triu!(copy(transpose(A.data))),k)
    end
    """
    return tril!(A.data,k)
end

function triu(A::SkewSymmetric, k::Integer=0)
    """
    if A.uplo == 'U' && k >= 0
        return triu(A.data,k)
    elseif A.uplo == 'U' && k < 0
        return triu(A.data,1) + triu!(tril!(copy(transpose(A.data))),k)
    elseif A.uplo == 'L' && k >= 0
        return triu!(copy(transpose(A.data)),k)
    else
        return triu!(copy(transpose(A.data)),1) + triu!(tril(A.data),k)
    end
    """
    return triu!(A.data,k)
end

for (T, trans, real) in [(:SkewSymmetric, :transpose, :identity)]
    @eval begin
        function dot(A::$T, B::$T)
            n = skewsize(A, 2)
            if n != skewsize(B, 2)
                throw(DimensionMismatch("A has dimensions $(skewsize(A)) but B has dimensions $(skewsize(B))"))
            end

            dotprod = zero(dot(first(A), first(B)))
            @inbounds for j in 1:n
                for i in 1:(j - 1)
                    dotprod += 2 * $real(dot(A.data[i, j], B.data[i, j]))
                end
            end
            
            return dotprod
        end
    end
end

(-)(A::SkewSymmetric) = SkewSymmetric(-A.data)




#for f in (:+, :-)
#    @eval begin
#        $f(A::SymTridiagonal, B::Symmetric) = Symmetric($f(A, B.data), sym_uplo(B.uplo))
#        $f(A::Symmetric, B::SymTridiagonal) = Symmetric($f(A.data, B), sym_uplo(A.uplo))
#   end
#end

## Matvec
@inline function mul!(y::StridedVector{T}, A::SkewSymmetric{T,<:StridedMatrix}, x::StridedVector{T},
            α::Number, β::Number) where {T<:LinearAlgebra.BlasFloat}
    alpha, beta = promote(α, β, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        return BLAS.gemv!('N', alpha, A.data, x, beta, y)
    else
        return generic_matvecmul!(y, 'N', A, x, MulAddMul(α, β))
    end
end
## Matmat
@inline function mul!(C::StridedMatrix{T}, A::SkewSymmetric{T,<:StridedMatrix}, B::StridedMatrix{T},
            α::Number, β::Number) where {T<:LinearAlgebra.BlasFloat}
    alpha, beta = promote(α, β, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        return BLAS.gemm!('N', 'N', alpha, A.data, B, beta, C)
    else
        return generic_matmatmul!(C, 'N', 'N', A, B, MulAddMul(alpha, beta))
    end
end
@inline function mul!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::SkewSymmetric{T,<:StridedMatrix},
         α::Number, β::Number) where {T<:LinearAlgebra.BlasFloat}
    alpha, beta = promote(α, β, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        return BLAS.gemm!('N', 'N', alpha, B.data, A, beta, C)
    else
        return generic_matmatmul!(C, 'N', 'N', A, B, MulAddMul(alpha, beta))
    end
end


*(A::SkewSymmetric, B::SkewSymmetric) = A * copyto!(similar(parent(B)), B)

function dot(x::AbstractVector, A::SkewSymmetric, y::AbstractVector)
    require_one_based_indexing(x, y)
    (length(x) == length(y) == skewsize(A, 1)) || throw(DimensionMismatch())
    data = A.data
    r = zero(eltype(x)) * zero(eltype(A)) * zero(eltype(y))
    @inbounds for j = 1:length(y)
        @simd for i = 1:j-1
            Aij = data[i,j]
            r += dot(x[i], Aij, y[j]) + dot(x[j], -Aij, y[i])
        end
    end
    
    return r
end

# Scaling with Number
*(A::SkewSymmetric, x::Number) = SkewSymmetric(A.data*x)
*(x::Number, A::SkewSymmetric) = SkewSymmetric(x*A.data)
/(A::SkewSymmetric, x::Number) = SkewSymmetric(A.data/x)
