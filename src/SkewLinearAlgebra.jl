# This file is a part of Julia. License is MIT: https://julialang.org/license
"""
This module based on the LinearAlgebra module provides specialized functions
and types for skew-symmetricmatrices, i.e A=-A^T
"""
module SkewLinearAlgebra

import LinearAlgebra as LA
import Base: \, /, *, ^, +, -, ==, copy
export 
    #Types
    SkewSymmetric,
    SkewHessenberg,
    #functions
    isskewsymmetric

struct SkewSymmetric{T<:Real,S<:AbstractMatrix{<:T}} <: AbstractMatrix{T}
    data::S

    function SkewSymmetric{T,S}(data) where {T,S<:AbstractMatrix{<:T}}
        LA.require_one_based_indexing(data)
        new{T,S}(data)  
    end
end

"""
    SkewSymmetric(A)
Transform matrix A in a Skewsymmetric structure. A is assumed to be correctly 
build as a skew-symmetric matrix. 'isskewsymmetric(A)' allows to verify skew-symmetry
"""

function SkewSymmetric(A::AbstractMatrix)
    LA.checksquare(A)
    n=LA.size(A,1)
    n>1 || throw("Skew-symmetric cannot be of size  1x1")
    return skewsymmetric_type(typeof(A))(A)
end

skewsymmetric(A::AbstractMatrix) = SkewSymmetric(A)
skewsymmetric(A::Number) = throw("Number cannot be skewsymmetric")

"""
    skewsymmetric_type(T::Type)
The type of the object returned by `skewsymmetric(::T, ::Symbol)`. For matrices, this is an
appropriately typed `SkewSymmetric`. If `skewsymmetric` is
implemented for a custom type, so should be `skewsymmetric_type`, and vice versa.
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
skewsymmetric_type(::Type{T}) where {T<:Number} = throw("Number cannot be skewsymmetric")

@inline function Base.getindex(A::SkewSymmetric, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    return @inbounds A.data[i,j]
end 

function Base.setindex!(A::SkewSymmetric, v, i::Integer, j::Integer)
    i!=j || throw("Cannot modify zero diagonal element")
    Base.setindex!(A.data, v, i, j)
    Base.setindex!(A.data, -v, j, i)
end

Base.similar(A::SkewSymmetric, ::Type{T}) where {T} = SkewSymmetric(LA.similar(parent(A), T))
Base.similar(A::SkewSymmetric) = SkewSymmetric(LA.similar(parent(A)))

# Conversion
function Matrix(A::SkewSymmetric)
    B =copy(A.data)
    return B
end
Array(A::SkewSymmetric) = convert(Matrix, A)

Base.parent(A::SkewSymmetric) = A.data
SkewSymmetric{T,S}(A::SkewSymmetric{T,S}) where {T,S<:AbstractMatrix{T}} = A
SkewSymmetric{T,S}(A::SkewSymmetric) where {T,S<:AbstractMatrix{T}} = SkewSymmetric{T,S}(convert(S,A.data))
#AbstractMatrix{T}(A::SkewSymmetric) where {T} = SkewSymmetric(convert(AbstractMatrix{T}, A.data))

Base.copy(A::SkewSymmetric{T,S}) where {T,S} = (B = Base.copy(A.data); SkewSymmetric{T,typeof(B)}(B))

function Base.copyto!(dest::SkewSymmetric, src::SkewSymmetric)
    Base.copyto!(dest.data, src.data)
    return dest
end
Base.size(A::SkewSymmetric,n) = size(A.data,n)
Base.size(A::SkewSymmetric) = size(A.data)
# fill[stored]!
fill!(A::SkewSymmetric,uplo::Char, x) = fillstored!(A,uplo, x)
function fillstored!(A::SkewSymmetric{T},uplo::Char, x) where T
    xT = convert(T, x)
    if uplo != 'U' &&uplo !='L'
        throw(ArgumentError("uplo must be 'U' or 'L'"))
    end
    if uplo == 'U'
        LA.fillband!(A.data, xT, 0, size(A,2)-1)
        LA.fillband!(-A.data, xT, size(A,2)-1,0)
    else # A.uplo == 'L'
        LA.fillband!(A.data, xT, 1-size(A,1), 0)
        LA.fillband!(-A.data, xT,0, 1-size(A,1))
    end
    return A
end

function isskewsymmetric(A::SkewSymmetric) 
    n=size(A,1)
    for i=1:n
        Base.getindex(A,i,i) == 0 || return false
        for j=1:i-1
            Base.getindex(A,i,j) == -Base.getindex(A,j,i) ||return false
        end
    end
    return true
end

#adjoint(A::Hermitian) = A
LA.transpose(A::SkewSymmetric) = SkewSymmetric(-A.data)
LA.adjoint(A::SkewSymmetric{<:Real}) = SkewSymmetric(-A.data)
LA.adjoint(A::SkewSymmetric) = SkewSymmetric(-A.data)
LA.real(A::SkewSymmetric{<:Real}) = A
LA.real(A::SkewSymmetric) = A
LA.imag(A::SkewSymmetric) = SkewSymmetric(LA.imag(A.data))

Base.copy(A::SkewSymmetric) =SkewSymmetric(Base.copy(parent(A)))
Base.display(A::SkewSymmetric) = display(A.data)
Base.conj(A::SkewSymmetric) = typeof(A)(A.data)
Base.conj!(A::SkewSymmetric) = typeof(A)(A.data)
LA.tr(A::SkewSymmetric) = 0



for (T, trans, real) in [(:SkewSymmetric, :transpose, :identity)]
    @eval begin
        function dot(A::$T, B::$T)
            n = size(A, 2)
            if n != size(B, 2)
                throw(DimensionMismatch("A has dimensions $(size(A)) but B has dimensions $(size(B))"))
            end

            dotprod = zero(dot(first(A), first(B)))
            @inbounds for j in 1:n
                for i in 1:(j - 1)
                    dotprod += 2 * $real(LA.dot(A.data[i, j], B.data[i, j]))
                end
            end
            
            return dotprod
        end
    end
end

Base. -(A::SkewSymmetric) = SkewSymmetric(- A.data)




#for f in (:+, :-)
#    @eval begin
#        $f(A::SymTridiagonal, B::Symmetric) = Symmetric($f(A, B.data), sym_uplo(B.uplo))
#        $f(A::Symmetric, B::SymTridiagonal) = Symmetric($f(A.data, B), sym_uplo(A.uplo))
#   end
#end

## Matvec
@inline function LA.mul!(y::StridedVector{T}, A::SkewSymmetric{T,<:StridedMatrix}, x::StridedVector{T},
            α::Number, β::Number) where {T<:LA.BlasFloat}
    alpha, beta = promote(α, β, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        return LA.BLAS.gemv!('N', alpha, A.data, x, beta, y)
    else
        return generic_matvecmul!(y, 'N', A, x, MulAddMul(α, β))
    end
end
## Matmat
@inline function LA.mul!(C::StridedMatrix{T}, A::SkewSymmetric{T,<:StridedMatrix}, B::StridedMatrix{T},
            α::Number, β::Number) where {T<:LA.BlasFloat}
    alpha, beta = promote(α, β, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        return LA.BLAS.gemm!('N', 'N', alpha, A.data, B, beta, C)
    else
        return generic_matmatmul!(C, 'N', 'N', A, B, MulAddMul(alpha, beta))
    end
end
@inline function LA.mul!(C::StridedMatrix{T}, A::StridedMatrix{T}, B::SkewSymmetric{T,<:StridedMatrix},
        α::Number, β::Number) where {T<:LA.BlasFloat}
    alpha, beta = promote(α, β, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        return LA.BLAS.gemm!('N', 'N', alpha, B.data, A, beta, C)
    else
        return generic_matmatmul!(C, 'N', 'N', A, B, MulAddMul(alpha, beta))
    end
end
@inline function LA.mul!(C::StridedMatrix{T}, A::SkewSymmetric{T,<:StridedMatrix}, B::SkewSymmetric{T,<:StridedMatrix},
        α::Number, β::Number) where {T<:LA.BlasFloat}
    alpha, beta = promote(α, β, zero(T))
    if alpha isa Union{Bool,T} && beta isa Union{Bool,T}
        return LA.BLAS.gemm!('N', 'N', alpha, A.data, B.data, beta, C)
    else
        return generic_matmatmul!(C, 'N', 'N', A, B, MulAddMul(alpha, beta))
    end
end


Base. *(A::SkewSymmetric, B::SkewSymmetric) = LA.Symmetric(Base. *(A.data,  B.data))
Base. *(A::SkewSymmetric, B::AbstractMatrix) = Base. *(A.data,  B)
Base. *(A::AbstractMatrix, B::SkewSymmetric) = Base. *(A,  B.data)

function LA.dot(x::AbstractVector, A::SkewSymmetric, y::AbstractVector)
    LA.require_one_based_indexing(x, y)
    (length(x) == length(y) == size(A, 1)) || throw(DimensionMismatch())
    data = A.data
    r = *( zero(eltype(x)),  zero(eltype(A)) , zero(eltype(y)))
    @inbounds for j = 1:length(y)
        @simd for i = 1:j-1
            Aij = data[i,j]
            r += LA.dot(x[i], Aij, y[j]) + LA.dot(x[j], -Aij, y[i])
        end
    end
    
    return r
end

# Scaling with Number
Base. *(A::SkewSymmetric, x::Number) = SkewSymmetric(A.data*x)
Base. *(x::Number, A::SkewSymmetric) = SkewSymmetric(x*A.data)
Base. /(A::SkewSymmetric, x::Number) = SkewSymmetric(A.data/x)

include("hessenberg.jl")
include("eigen.jl")
include("exp.jl")

end


