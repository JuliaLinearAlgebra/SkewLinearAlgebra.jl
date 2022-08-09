

# This file is a part of Julia. License is MIT: https://julialang.org/license

#### Specialized matrix types ####

struct SkewHermTridiagonal{T, V<:AbstractVector{T}} <: AbstractMatrix{T}
    ev::V                        # subdiagonal
    function SkewHermTridiagonal{T, V}(ev) where {T, V<:AbstractVector{T}}
        require_one_based_indexing(ev)
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
        SkewHermTridiagonal(diag(A, 1))
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

Base.similar(S::SkewHermTridiagonal, ::Type{T}) where {T} = SkewHermTridiagonal(similar(S.dv, T), similar(S.ev, T))
Base.similar(S::SkewHermTridiagonal, ::Type{T}, dims::Union{Dims{1},Dims{2}}) where {T} = zeros(T, dims...)

Base.copyto!(dest::SkewHermTridiagonal, src::SkewHermTridiagonal) =
    (copyto!(dest.ev, src.ev); dest)

#Elementary operations
for func in (:conj, :copy, :real, :imag)
    @eval Base.$func(M::SkewHermTridiagonal) = SkewHermTridiagonal(($func)(M.ev))
end

Base.transpose(S::SkewHermTridiagonal) = -S
Base.adjoint(S::SkewHermTridiagonal{<:Real}) = -S
Base.adjoint(S::SkewHermTridiagonal) = -conj.(S)
#=
permutedims(S::SkewHermTridiagonal) = S
function permutedims(S::SkewHermTridiagonal, perm)
    Base.checkdims_perm(S, S, perm)
    NTuple{2}(perm) == (2, 1) ? permutedims(S) : S
end
=#
Base.copy(S::SkewHermTridiagonal)=SkewHermTridiagonal(copy(S.ev))
Base.copy(S::LA.Adjoint{<:Any,<:SkewHermTridiagonal}) = SkewHermTridiagonal(map(x -> copy.(adjoint.(x)), (S.parent.ev))...)

#isskewhermitian(S::SkewHermTridiagonal) = true

function diag(M::SkewHermTridiagonal{T}, n::Integer=0) where T<:Number
    # every branch call similar(..., ::Int) to make sure the
    # same vector type is returned independent of n
    absn = abs(n)
    if absn == 0
        return zeros(eltype(M.ev),length(M.ev)+1)
    elseif absn == 1
        return copyto!(similar(M.ev, length(M.ev)), M.ev)
    elseif absn <= size(M,1)
        return fill!(similar(M.ev, size(M,1)-absn), zero(T))
    else
        throw(ArgumentError(string("requested diagonal, $n, must be at least $(-size(M, 1)) ",
            "and at most $(size(M, 2)) for an $(size(M, 1))-by-$(size(M, 2)) matrix")))
    end
end
function diag(M::SkewHermTridiagonal, n::Integer=0)
    # every branch call similar(..., ::Int) to make sure the
    # same vector type is returned independent of n
    absn = abs(n)
    if absn == 0
        return zeros(eltype(M.ev),length(M.ev)+1)
    elseif absn == 1
        return copyto!(similar(M.ev, length(M.ev)), M.ev)
    elseif absn <= size(M,1)
        return fill!(similar(M.ev, size(M,1)-absn), zero(T))
    else
        throw(ArgumentError(string("requested diagonal, $n, must be at least $(-size(M, 1)) ",
            "and at most $(size(M, 2)) for an $(size(M, 1))-by-$(size(M, 2)) matrix")))
    end
end
Base.:+(A::SkewHermTridiagonal, B::SkewHermTridiagonal) = SkewHermTridiagonal(A.ev+B.ev)
Base.:-(A::SkewHermTridiagonal, B::SkewHermTridiagonal) = SkewHermTridiagonal(A.ev-B.ev)
Base.:-(A::SkewHermTridiagonal) = SkewHermTridiagonal(-A.ev)
Base.:*(A::SkewHermTridiagonal, B::Number) = SkewHermTridiagonal(A.ev*B)
Base.:*(B::Number, A::SkewHermTridiagonal) = SkewHermTridiagonal(B*A.ev)
Base.:/(A::SkewHermTridiagonal, B::Number) = SkewHermTridiagonal(A.ev/B)
Base.:\(B::Number, A::SkewHermTridiagonal) = SkewHermTridiagonal(B\A.ev)
#Base.:==(A::SkewHermTridiagonal, B::SkewHermTridiagonal) = (A.ev==B.ev)

@inline mul!(A::StridedVecOrMat, B::SkewHermTridiagonal, C::StridedVecOrMat,
             alpha::Number, beta::Number) =
    _mul!(A, B, C, LA.MulAddMul(alpha, beta))
####To review ########################
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
        return _rmul_or_fill!(C, _add.beta)
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
                _modify!(_add, β₋*x₋ + β₀*x₊, C, (i, j))
            end
            _modify!(_add, β₀*x₀ , C, (m, j))
        end
    end

    return C
end

function dot(x::AbstractVector, S::SkewHermTridiagonal, y::AbstractVector)
    require_one_based_indexing(x, y)
    nx, ny = length(x), length(y)
    (nx == size(S, 1) == ny) || throw(DimensionMismatch())
    if iszero(nx)
        return dot(zero(eltype(x)), zero(eltype(S)), zero(eltype(y)))
    end
    ev = S.ev
    x₀ = x[1]
    x₊ = x[2]
    sub = transpose(ev[1])
    r = dot(adjoint(dv[1])*x₀ + adjoint(sub)*x₊, y[1])
    @inbounds for j in 2:nx-1
        x₋, x₀, x₊ = x₀, x₊, x[j+1]
        sup, sub = transpose(sub), transpose(ev[j])
        r += dot(adjoint(sup)*x₋ + adjoint(dv[j])*x₀ + adjoint(sub)*x₊, y[j])
    end
    r += dot(adjoint(transpose(sub))*x₀ + adjoint(dv[nx])*x₊, y[nx])
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
    n = size(S.data,1)
    n = size(S,1)
    H = SymTridiagonal(zeros(eltype(S.ev),n),S.ev)
    trisol = eigen!(H)

    vals  = trisol.values*1im
    vals .*= -1
    Qdiag = zeros(eltype(trisol.vectors),n,n)*1im
    c = 1
    @inbounds for j=1:n
        @simd for i=1:2:n-1
            k=(i+1)÷2
            Qdiag[i,j].=trisol.vectors[i,j]*c
            Qdiag[i+1,j].=trisol.vectors[i+1,j]*c*1im
        end
        c *= (-1)
    end
    if n%2==1
        Qdiag[n,:].=trisol.vectors[i,j]*c
    end
    return Eigen(vals)
end


@views function LA.eigen!(A::SkewHermTridiagonal)
     return skewtrieigen!(A)
end

copyeigtype(A) = copyto!(similar(A, LA.eigtype(eltype(A))), A)

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

# tril and triu

function LA.istriu(M::SkewHermTridiagonal, k::Integer=0)
    if k <= -1
        return true
    elseif k == 0
        return iszero(M.ev)
    else # k >= 1
        return iszero(M.ev)
    end
end
LA.istril(M::SkewHermTridiagonal, k::Integer) = istriu(M, -k)
Base.iszero(M::SkewHermTridiagonal) =  iszero(M.ev)
Base.isone(M::SkewHermTridiagonal) =  false
LA.isdiag(M::SkewHermTridiagonal) =  iszero(M.ev)


function LA.tril!(M::SkewHermTridiagonal{T}, k::Integer=0) where T
    n = length(M.ev)+1
    if !(-n - 1 <= k <= n - 1)
        throw(ArgumentError(string("the requested diagonal, $k, must be at least ",
            "$(-n - 1) and at most $(n - 1) in an $n-by-$n matrix")))
    elseif k < -1
        fill!(M.ev, zero(T))
        return Tridiagonal(M.ev,zeros(eltype(M.ev),n),copy(M.ev))
    elseif k == -1
        fill!(M.dv, zero(T))
        return Tridiagonal(M.ev,zeros(eltype(M.ev),n),zero(M.ev))
    elseif k == 0
        return Tridiagonal(M.ev,zeros(eltype(M.ev),n),zero(M.ev))
    elseif k >= 1
        return Tridiagonal(M.ev,zeros(eltype(M.ev),n),copy(adjoint.(M.ev)))
    end
end
#=
function Base.triu!(M::SkewHermTridiagonal{T}, k::Integer=0) where T
    n = length(M.dv)
    if !(-n + 1 <= k <= n + 1)
        throw(ArgumentError(string("the requested diagonal, $k, must be at least ",
            "$(-n + 1) and at most $(n + 1) in an $n-by-$n matrix")))
    elseif k > 1
        fill!(M.ev, zero(T))
        fill!(M.dv, zero(T))
        return Tridiagonal(M.ev,M.dv,copy(M.ev))
    elseif k == 1
        fill!(M.dv, zero(T))
        return Tridiagonal(zero(M.ev),M.dv,M.ev)
    elseif k == 0
        return Tridiagonal(M.ev,zeros(eltype(M.ev),n),zero(M.ev))
    elseif k <= -1
        return Tridiagonal(M.ev,M.dv,copy(M.ev))
    end
end
=#
###################
# Generic methods #
###################

# det with optional diagonal shift for use with shifted Hessenberg factorizations
#det(A::SkewHermTridiagonal; shift::Number=false) = det_usmani(A.ev, A.dv, A.ev, shift)
#logabsdet(A::SkewHermTridiagonal; shift::Number=false) = logabsdet(ldlt(A; shift=shift))

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

#=
function Base._sum(A::SkewHermTridiagonal, dims::Integer)
    res = Base.reducedim_initarray(A, dims, zero(eltype(A)))
    n = length(A.dv)
    if n == 0
        return res
    elseif n == 1
        res[1] = A.dv[1]
        return res
    end
    @inbounds begin
        if dims == 1
            res[1] = transpose(A.ev[1]) + symmetric(A.dv[1], :U)
            for i = 2:n-1
                res[i] = transpose(A.ev[i]) + symmetric(A.dv[i], :U) + A.ev[i-1]
            end
            res[n] = symmetric(A.dv[n], :U) + A.ev[n-1]
        elseif dims == 2
            res[1] = symmetric(A.dv[1], :U) + A.ev[1]
            for i = 2:n-1
                res[i] = transpose(A.ev[i-1]) + symmetric(A.dv[i], :U) + A.ev[i]
            end
            res[n] = transpose(A.ev[n-1]) + symmetric(A.dv[n], :U)
        elseif dims >= 3
            for i = 1:n-1
                res[i,i+1] = A.ev[i]
                res[i,i]   = symmetric(A.dv[i], :U)
                res[i+1,i] = transpose(A.ev[i])
            end
            res[n,n] = symmetric(A.dv[n], :U)
        end
    end
    res
end
=#
