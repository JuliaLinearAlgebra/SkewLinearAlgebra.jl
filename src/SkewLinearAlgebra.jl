# This file is a part of Julia. License is MIT: https://julialang.org/license
"""
This module based on the LinearAlgebra module provides specialized functions
and types for skew-symmetric matrices, i.e A=-A^T
"""
module SkewLinearAlgebra

import LinearAlgebra as LA
import LinearAlgebra: similar,require_one_based_indexing, BlasReal,BlasFloat, 
        checksquare,transpose, adjoint,real,imag,dot,tr,tril,
        tril!,triu,triu!,mul!,axpy!,norm,eigtype,eigvals!,eigvals,eigen,eigen!,
        eigmax,eigmin,det,inv,inv!,lu,lu!,rmul!,lmul!,rdiv!,ldiv!, 
        hessenberg,hessenberg!,Tridiagonal,UnitLowerTriangular,UpperHessenberg,Diagonal,Hermitian,Matrix,diagm,Array,SymTridiagonal
        lu,lu!
import LinearAlgebra.BLAS: gemv!,ger!
import Base: \, /, *, ^, +, -, ==, copy,copyto!, size,setindex!,getindex,display,conj,conj!,similar,
        isreal,cos,sin,cosh,sinh,tanh,cis
export 
    #Types
    SkewSymmetric,
    SkewHessenberg,
    #functions
    isskewsymmetric,
    getQ

struct SkewSymmetric{T<:Real,S<:AbstractMatrix{<:T}} <: AbstractMatrix{T}
    data::S

    function SkewSymmetric{T,S}(data) where {T,S<:AbstractMatrix{<:T}}
        require_one_based_indexing(data)
        new{T,S}(data)  
    end
end

"""
    SkewSymmetric(A)
Transform matrix A in a Skewsymmetric structure. A is assumed to be correctly 
build as a skew-symmetric matrix. 'isskewsymmetric(A)' allows to verify skew-symmetry
"""

function SkewSymmetric(A::AbstractMatrix)
    checksquare(A)
    n=size(A,1)
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

"""
    getindex(A,i,j)

Returns the value A(i,j)
"""

@inline function Base.getindex(A::SkewSymmetric, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    return @inbounds A.data[i,j]
end 

"""
    setindex!(A,v,i,j)
Set A(i,j)=v and A(j,i)=-v to conserve skew-symmetry
"""

function Base.setindex!(A::SkewSymmetric, v, i::Integer, j::Integer)
    i!=j || throw("Cannot modify zero diagonal element")
    setindex!(A.data, v, i, j)
    setindex!(A.data, -v, j, i)
end

similar(A::SkewSymmetric, ::Type{T}) where {T} = SkewSymmetric(LA.similar(parent(A), T))
similar(A::SkewSymmetric) = SkewSymmetric(similar(parent(A)))

# Conversion
function Matrix(A::SkewSymmetric)
    B = copy(A.data)
    return B
end
Array(A::SkewSymmetric) = convert(Matrix, A)

parent(A::SkewSymmetric) = A.data
SkewSymmetric{T,S}(A::SkewSymmetric{T,S}) where {T,S<:AbstractMatrix{T}} = A
SkewSymmetric{T,S}(A::SkewSymmetric) where {T,S<:AbstractMatrix{T}} = SkewSymmetric{T,S}(convert(S,A.data))
Base.AbstractMatrix{T}(A::SkewSymmetric) where {T} = SkewSymmetric(convert(AbstractMatrix{T}, A.data))

copy(A::SkewSymmetric{T,S}) where {T,S} = (B = Base.copy(A.data); SkewSymmetric{T,typeof(B)}(B))

function copyto!(dest::SkewSymmetric, src::SkewSymmetric)
    copyto!(dest.data, src.data)
    return dest
end
size(A::SkewSymmetric,n) = size(A.data,n)
size(A::SkewSymmetric) = size(A.data)


"""
    isskewsymmetric(A)

Verifies skew-symmetry of a matrix
"""

function isskewsymmetric(A::SkewSymmetric) 
    n=size(A,1)
    for i=1:n
        getindex(A,i,i) == 0 || return false
        for j=1:i-1
            getindex(A,i,j) == -getindex(A,j,i) ||return false
        end
    end
    return true
end

#Classic operators on a matrix
Base.isreal(A::SkewSymmetric)=true
transpose(A::SkewSymmetric) = SkewSymmetric(-A.data)
adjoint(A::SkewSymmetric{<:Real}) = SkewSymmetric(-A.data)
adjoint(A::SkewSymmetric) = SkewSymmetric(-A.data)
real(A::SkewSymmetric{<:Real}) = A
real(A::SkewSymmetric) = A
imag(A::SkewSymmetric) = SkewSymmetric(LA.imag(A.data))

copy(A::SkewSymmetric) =SkewSymmetric(Base.copy(parent(A)))
display(A::SkewSymmetric) = display(A.data)
conj(A::SkewSymmetric) = typeof(A)(A.data)
conj!(A::SkewSymmetric) = typeof(A)(A.data)
tr(A::SkewSymmetric) = 0


tril!(A::SkewSymmetric) = tril!(A.data)
tril(A::SkewSymmetric)  = tril(A.data)
triu!(A::SkewSymmetric) = triu!(A.data)
triu(A::SkewSymmetric)  = triu(A.data)
tril!(A::SkewSymmetric,k::Integer) = tril!(A.data,k)
tril(A::SkewSymmetric,k::Integer)  = tril(A.data,k)
triu!(A::SkewSymmetric,k::Integer) = triu!(A.data,k)
triu(A::SkewSymmetric,k::Integer)  = triu(A.data,k)


function LA.dot(A::SkewSymmetric, B::SkewSymmetric)
    n = size(A, 2)
    if n != size(B, 2)
        throw(DimensionMismatch("A has dimensions $(size(A)) but B has dimensions $(size(B))"))
    end
    dotprod = zero(dot(first(A), first(B)))
    @inbounds for j = 1:n
        for i = 1:(j - 1)
            dotprod += 2 *(dot(A.data[i, j], B.data[i, j]))
        end
    end
    
    return dotprod
end

Base. -(A::SkewSymmetric) = SkewSymmetric(- A.data)


for f in (:+, :-)
    @eval begin
        $f(A::SkewSymmetric, B::SkewSymmetric) = SkewSymmetric($f(A.data, B.data))
   end
end

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
            r += dot(x[i], Aij, y[j]) + dot(x[j], -Aij, y[i])
        end
    end
    
    return r
end
include("hessenberg.jl")
include("eigen.jl")
include("exp.jl")
# Scaling with Number
Base. *(A::SkewSymmetric, x::Number) = SkewSymmetric(A.data*x)
Base. *(x::Number, A::SkewSymmetric) = SkewSymmetric(x*A.data)
Base. /(A::SkewSymmetric, x::Number) = SkewSymmetric(A.data/x)
Base. \(A::SkewSymmetric,b) = \(A.data,b)

det(A::SkewSymmetric) = det(A.data)
logdet(A::SkewSymmetric) = logdet(A.data)
inv(A::SkewSymmetric)  = inv(A.data)
inv!(A::SkewSymmetric)  = inv!(A.data)

lu(A::SkewSymmetric)  = lu(A.data)
lu!(A::SkewSymmetric) = lu!(A.data)
lu(A::SkewSymmetric)  = lq(A.data)
lq!(A::SkewSymmetric) = lq!(A.data)
qr(A::SkewSymmetric)  = qr(A.data)
qr!(A::SkewSymmetric) = qr!(A.data)
schur(A::SkewSymmetric)=schur(A.data)
schur!(A::SkewSymmetric)=schur!(A.data)
svd(A::SkewSymmetric; full::Bool = false, alg::Algorithm = default_svd_alg(A))  = svd(A.data;full,alg)
svd!(A::SkewSymmetric; full::Bool = false, alg::Algorithm = default_svd_alg(A))  = svd!(A.data;full,alg)
svdvals(A::SkewSymmetric)=svdvals(A)
svdvals!(A::SkewSymmetric)=svdvals!(A)
diag(A::SkewSymmetric, k::Integer=0)=diag(A,k)
rank(A::SkewSymmetric; atol::Real=0, rtol::Real=atol>0 ? 0 : n*ϵ)=rank(A.data;atol,rtol)
rank(A::SkewSymmetric, rtol::Real)=rank(A.data,rtol)
norm(A::SkewSymmetric, p::Real=2)=norm(A,p)

rdiv!(A::SkewSymmetric,b::Number) = rdiv!(A.data,b)
ldiv!(A::SkewSymmetric,b::Number) = ldiv!(A.data,b)
rmul!(A::SkewSymmetric,b::Number) = rmul!(A.data,b)
lmul!(A::SkewSymmetric,b::Number) = lmul!(A.data,b)


end

kron(A::SkewSymmetric,B::AbstractMatrix)=kron(A.data,B)
kron(A::AbstractMatrix,B::SkewSymmetric)=kron(A,B.data)

