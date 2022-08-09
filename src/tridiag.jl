

# This file is a part of Julia. License is MIT: https://julialang.org/license

#### Specialized matrix types ####

struct SkewHermTridiagonal{T, V<:AbstractVector{T}} <: AbstractMatrix{T}
    ev::V                        # subdiagonal
    function SkewHermTridiagonal{T, V}(ev) where {T, V<:AbstractVector{T}}

        LA.require_one_based_indexing(ev)

        new{T, V}(ev)
    end
end

"""
    SkewHermTridiagonal(ev::V) where V <: AbstractVector
Construct a skewhermitian tridiagonal matrix from the subdiagonal (`ev`).
The result is of type `SkewHermTridiagonal`
and provides efficient specialized eigensolvers, but may be converted into a
regular matrix with [`convert(Array, _)`](@ref) (or `Array(_)` for short).
# Examples
```jldoctest
julia> ev = [7, 8, 9]
3-element Vector{Int64}:
 7
 8
 9
julia> SkewHermTridiagonal(ev)
4×4 SkewHermTridiagonal{Int64, Vector{Int64}}:
 0 -7  ⋅  ⋅
 7  . -8  ⋅
 ⋅  8  . -9
 ⋅  ⋅  9  .
```
"""
SkewHermTridiagonal(ev::V) where {T,V<:AbstractVector{T}} = SkewHermTridiagonal{T}(ev)
SkewHermTridiagonal{T}(ev::V) where {T,V<:AbstractVector{T}} = SkewHermTridiagonal{T,V}(ev)
function SkewHermTridiagonal{T}(ev::AbstractVector) where {T}
    SkewHermTridiagonal(convert(AbstractVector{T}, ev)::AbstractVector{T})
end

"""
    SkewHermTridiagonal(A::AbstractMatrix)
Construct a skewhermitian tridiagonal matrix from first subdiagonal
of the skewhermitian matrix `A`.
# Examples
```jldoctest
julia> A = [1 2 3; 2 4 5; 3 5 6]
3×3 Matrix{Int64}:
 1  2  3
 2  4  5
 3  5  6
julia> SkewHermTridiagonal(A)
3×3 SkewHermTridiagonal{Int64, Vector{Int64}}:
 . -2  ⋅
 2  . -5
 ⋅  5  .
```
"""
function SkewHermTridiagonal(A::AbstractMatrix)
    if diag(A, 1) == - adjoint.(diag(A, -1))
        SkewHermTridiagonal(diag(A, -1))
    else
        throw(ArgumentError("matrix is not skew-hermitian; cannot convert to SkewHermTridiagonal"))
    end
end

SkewHermTridiagonal{T,V}(S::SkewHermTridiagonal{T,V}) where {T,V<:AbstractVector{T}} = S
SkewHermTridiagonal{T,V}(S::SkewHermTridiagonal) where {T,V<:AbstractVector{T}} =
    SkewHermTridiagonal(convert(V, S.ev)::V)
SkewHermTridiagonal{T}(S::SkewHermTridiagonal{T}) where {T} = S
SkewHermTridiagonal{T}(S::SkewHermTridiagonal) where {T} =
    SkewHermTridiagonal(convert(AbstractVector{T}, S.ev)::AbstractVector{T})
SkewHermTridiagonal(S::SkewHermTridiagonal) = S

AbstractMatrix{T}(S::SkewHermTridiagonal) where {T} =
    SkewHermTridiagonal(convert(AbstractVector{T}, S.ev)::AbstractVector{T})

function Base.Matrix{T}(M::SkewHermTridiagonal) where T
    n = size(M, 1)
    Mf = zeros(T, n, n)
    n == 0 && return Mf
    @inbounds for i = 1:n-1
        Mf[i+1,i] = M.ev[i]
        Mf[i,i+1] = -M.ev[i]'
    end
    return Mf
end
Base.Matrix(M::SkewHermTridiagonal{T}) where {T} = Matrix{promote_type(T, typeof(zero(T)))}(M)
Base.Array(M::SkewHermTridiagonal) = Matrix(M)

Base.size(A::SkewHermTridiagonal) = (length(A.ev)+1,length(A.ev)+1)

function Base.size(A::SkewHermTridiagonal, d::Integer)
    if d < 1
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    elseif d<=2
        return length(A.ev)+1
    else
        return 1
    end
end


Base.similar(S::SkewHermTridiagonal, ::Type{T}) where {T} = SkewHermTridiagonal(similar(S.ev, T))

Base.similar(S::SkewHermTridiagonal, ::Type{T}, dims::Union{Dims{1},Dims{2}}) where {T} = zeros(T, dims...)

Base.copyto!(dest::SkewHermTridiagonal, src::SkewHermTridiagonal) =
    (copyto!(dest.ev, src.ev); dest)

#Elementary operations
for func in (:conj, :copy, :real, :imag)
    @eval Base.$func(M::SkewHermTridiagonal) = SkewHermTridiagonal(($func)(M.ev))
end

Base.transpose(S::SkewHermTridiagonal) = -S
Base.adjoint(S::SkewHermTridiagonal{<:Real}) = -S
Base.adjoint(S::SkewHermTridiagonal) = .-conj.(S)
Base.copy(S::LA.Adjoint{<:Any,<:SkewHermTridiagonal}) = SkewHermTridiagonal(map(x -> copy.(adjoint.(x)), (S.parent.ev))...)

isskewhermitian(S::SkewHermTridiagonal) = true


Base.:+(A::SkewHermTridiagonal, B::SkewHermTridiagonal) = SkewHermTridiagonal(A.ev+B.ev)
Base.:-(A::SkewHermTridiagonal, B::SkewHermTridiagonal) = SkewHermTridiagonal(A.ev-B.ev)
Base.:-(A::SkewHermTridiagonal) = SkewHermTridiagonal(-A.ev)
Base.:*(A::SkewHermTridiagonal, B::Number) = SkewHermTridiagonal(A.ev*B)
Base.:*(B::Number, A::SkewHermTridiagonal) = SkewHermTridiagonal(B*A.ev)
Base.:/(A::SkewHermTridiagonal, B::Number) = SkewHermTridiagonal(A.ev/B)
Base.:\(B::Number, A::SkewHermTridiagonal) = SkewHermTridiagonal(B\A.ev)
# ==(A::SkewHermTridiagonal, B::SkewHermTridiagonal) = (A.ev==B.ev)

@inline LA.mul!(A::StridedVecOrMat, B::SkewHermTridiagonal, C::StridedVecOrMat,
             alpha::Number, beta::Number) =
    _mul!(A, B, C, LA.MulAddMul(alpha, beta))

@inline function _mul!(C::StridedVecOrMat, S::SkewHermTridiagonal, B::StridedVecOrMat,
                          _add::LA.MulAddMul)
    m, n = size(B, 1), size(B, 2)
    if !(m == size(S, 1) == size(C, 1))
        throw(DimensionMismatch("A has first dimension $(size(S,1)), B has $(size(B,1)), C has $(size(C,1)) but all must match"))
    end
    if n != size(C, 2)
        throw(DimensionMismatch("second dimension of B, $n, doesn't match second dimension of C, $(size(C,2))"))
    end

    if m == 0
        return C
    elseif iszero(_add.alpha)
        return LA._rmul_or_fill!(C, _add.beta)
    end

    β = S.ev
    @inbounds begin
        for j = 1:n
            x₊ = B[1, j]
            x₀ = zero(x₊)
            # If m == 1 then β[1] is out of bounds
            β₀ = m > 1 ? zero(β[1]) : zero(eltype(β))
            for i = 1:m - 1
                x₋, x₀, x₊ = x₀, x₊, B[i + 1, j]
                β₋, β₀ = β₀, β[i]
                LA._modify!(_add, β₋*x₋ -adjoint(β₀)*x₊, C, (i, j))
            end
            LA._modify!(_add, β[m-1]*x₀ , C, (m, j))
        end
    end

    return C
end

function LA.dot(x::AbstractVector, S::SkewHermTridiagonal, y::AbstractVector)

    LA.require_one_based_indexing(x, y)

    nx, ny = length(x), length(y)
    (nx == size(S, 1) == ny) || throw(DimensionMismatch())
    if iszero(nx)
        return dot(zero(eltype(x)), zero(eltype(S)), zero(eltype(y)))
    end
    ev = S.ev
    x₀ = x[1]
    x₊ = x[2]
    sub = ev[1]
    r = dot( adjoint(sub)*x₊, y[1])
    @inbounds for j in 2:nx-1
        x₋, x₀, x₊ = x₀, x₊, x[j+1]
        sup, sub = -adjoint(sub), ev[j]
        r += dot(adjoint(sup)*x₋ + adjoint(sub)*x₊, y[j])
    end
    r += dot(adjoint(-adjoint(sub))*x₀, y[nx])
    return r
end

#Base.:\(T::SkewHermTridiagonal, B::StridedVecOrMat) = Base.ldlt(T)\B

@views function LA.eigvals!(A::SkewHermTridiagonal, sortby::Union{Function,Nothing}=nothing)
    vals = skeweigvals!(A)
    !isnothing(sortby) && sort!(vals, by=sortby)
    return complex.(0, vals)
end

@views function LA.eigvals!(A::SkewHermTridiagonal, irange::UnitRange)
    vals = skewtrieigvals!(A,irange)
    return complex.(0, vals)
end

@views function LA.eigvals!(A::SkewHermTridiagonal, vl::Real,vh::Real)
    vals = skewtrieigvals!(A,-vh,-vl)
    return complex.(0, vals)
end

LA.eigvals(A::SkewHermTridiagonal, irange::UnitRange) =
    LA.eigvals!(copyeigtype(A), irange)
LA.eigvals(A::SkewHermTridiagonal, vl::Real,vh::Real) =
    LA.eigvals!(copyeigtype(A), vl,vh)



@views function skewtrieigvals!(S::SkewHermTridiagonal)
    n = size(S,1)
    H = SymTridiagonal(zeros(eltype(S.ev),n),S.ev)
    vals = eigvals!(H)
    return vals .= .-vals

end

@views function skewtrieigvals!(S::SkewHermTridiagonal,irange::UnitRange)
    n = size(S,1)
    H = SymTridiagonal(zeros(eltype(S.ev),n),S.ev)
    vals = eigvals!(H,irange)
    return vals .= .-vals

end

@views function skewtrieigvals!(S::SkewHermTridiagonal,vl::Real,vh::Real)
    n = size(S,1)
    H = SymTridiagonal(zeros(eltype(S.ev),n),S.ev)
    vals = eigvals!(H,vl,vh)
    return vals .= .-vals
end

@views function skewtrieigen!(S::SkewHermTridiagonal)

    n = size(S,1)
    H = SymTridiagonal(zeros(eltype(S.ev),n),S.ev)
    trisol = eigen!(H)

    vals  = trisol.values*1im
    vals .*= -1

    Qdiag = similar(trisol.vectors,n,n)*1im
    c = 1
    @inbounds for j=1:n
        c = 1
        @simd for i=1:2:n-1
            Qdiag[i,j]  = trisol.vectors[i,j]*c
            Qdiag[i+1,j] = trisol.vectors[i+1,j]*c*1im
            c *= (-1)
        end

    end
    if n%2==1
        Qdiag[n,:]=trisol.vectors[n,:]*c

    end
    return Eigen(vals,Qdiag)
end


@views function LA.eigen!(A::SkewHermTridiagonal)
     return skewtrieigen!(A)
end

LA.eigen(A::SkewHermTridiagonal) = LA.eigen!(copyeigtype(A))

LA.eigvecs(A::SkewHermTridiagonal) = eigen(A).vectors
@views function LA.svdvals!(A::SkewHermTridiagonal)
    n=size(A,1)
    vals = skewtrieigvals!(A)
    vals .= abs.(vals)
    return sort!(vals; rev=true)
end

@views function LA.svd!(A::SkewHermTridiagonal)
    n=size(A,1)
    E=eigen!(A)
    U=E.vectors
    vals = imag.(E.values)
    I=sortperm(vals;by=abs,rev=true)
    permute!(vals,I)
    Base.permutecols!!(U,I)
    V = U .* -1im
    @inbounds for i=1:n
        if vals[i] < 0
            vals[i]=-vals[i]
            @simd for j=1:n
                V[j,i]=-V[j,i]
            end
        end
    end
    return LA.SVD(U,vals,adjoint(V))
end

LA.svd(A::SkewHermTridiagonal) = svd!(copyeigtype(A))

###################
# Generic methods #
###################

# det with optional diagonal shift for use with shifted Hessenberg factorizations
#det(A::SkewHermTridiagonal; shift::Number=false) = det_usmani(A.ev, A.dv, A.ev, shift)
#logabsdet(A::SkewHermTridiagonal; shift::Number=false) = logabsdet(ldlt(A; shift=shift))

# show a "⋅" for structural zeros when printing
function Base.replace_in_print_matrix(A::SkewHermTridiagonal, i::Integer, j::Integer, s::AbstractString)
    i==j-1||i==j||i==j+1 ? s : Base.replace_with_centered_mark(s)
end

@inline function Base.getindex(A::SkewHermTridiagonal{T}, i::Integer, j::Integer) where T
    @boundscheck checkbounds(A, i, j)
    if i == j + 1
        return @inbounds A.ev[j]
    elseif i + 1 == j
        return @inbounds -A.ev[i]'
    else
        return zero(T)
    end
end

@inline function Base.setindex!(A::SkewHermTridiagonal, x, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    if i == j+1
        @inbounds A.ev[j] = x
    elseif i == j-1
        @inbounds A.ev[i] = -x'
    else
        throw(ArgumentError("cannot set off-diagonal entry ($i, $j)"))
    end
    return x
end



