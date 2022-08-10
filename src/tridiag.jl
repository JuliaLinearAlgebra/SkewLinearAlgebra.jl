

# This file is a part of Julia. License is MIT: https://julialang.org/license

#### Specialized matrix types ####
struct SkewHermTridiagonal{T, V<:AbstractVector{T}, Vim<:Union{AbstractVector{<:Real},Nothing}} <: AbstractMatrix{T}
    ev::V                        # subdiagonal
    dvim::Vim               # diagonal imaginary parts (may be nothing if T is real)
    function SkewHermTridiagonal{T, V, Vim}(ev, dvim) where {T, V<:AbstractVector{T}, Vim}
        LA.require_one_based_indexing(ev)
        if Vim !== Nothing
            LA.require_one_based_indexing(dvim)
            eltype(dvim) === real(T) || throw(ArgumentError("mismatch between $(real(T)) and $(eltype(dvim))"))
        end
        new{T, V, Vim}(ev, dvim)
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

# real skew-symmetric case
SkewHermTridiagonal(ev::AbstractVector{T}) where {T<:Real} = SkewHermTridiagonal{T, typeof(ev), Nothing}(ev, nothing)
SkewHermTridiagonal{T}(ev::AbstractVector) where {T<:Real} =
    SkewHermTridiagonal(convert(AbstractVector{T}, ev)::AbstractVector{T})

# complex skew-hermitian case
SkewHermTridiagonal(ev::AbstractVector{Complex{T}}, dvim::AbstractVector{T}) where T = SkewHermTridiagonal{Complex{T}, typeof(ev), typeof(dvim)}(ev, dvim)
SkewHermTridiagonal{Complex{T}}(ev::AbstractVector, dvim::AbstractVector) where T =
    SkewHermTridiagonal(convert(AbstractVector{Complex{T}}, ev)::AbstractVector{Complex{T}}, convert(AbstractVector{T}, dvim)::AbstractVector{T})
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
    if iszero(real(diag(A))) && !iszero(imag(diag(A)))
        if diag(A, 1) == - adjoint.(diag(A, -1))
            SkewHermTridiagonal(diag(A, -1),imag(diag(A)))
        else
            throw(ArgumentError("matrix is not skew-hermitian; cannot convert to SkewHermTridiagonal"))
        end
    else
        if diag(A, 1) == - adjoint.(diag(A, -1))
            SkewHermTridiagonal(diag(A, -1))
        else
            throw(ArgumentError("matrix is not skew-hermitian; cannot convert to SkewHermTridiagonal"))
        end
    end


end

SkewHermTridiagonal{T,V,Vim}(S::SkewHermTridiagonal{T,V,Vim}) where {T,V<:AbstractVector{T}, Vim<:Union{AbstractVector{<:Real},Nothing}} = S
SkewHermTridiagonal{T,V,Vim}(S::SkewHermTridiagonal) where {T,V<:AbstractVector{T}, Vim<:Union{AbstractVector{<:Real},Nothing}} =
    SkewHermTridiagonal(convert(V, S.ev)::V,convert(Vim, S.dvim)::Vim)
SkewHermTridiagonal{T}(S::SkewHermTridiagonal{T}) where {T} = S
SkewHermTridiagonal{T}(S::SkewHermTridiagonal) where {T} =
    SkewHermTridiagonal(convert(AbstractVector{T}, S.ev)::AbstractVector{T},convert(AbstractVector{<:Real}, S.dvim)::AbstractVector{<:Real})
SkewHermTridiagonal(S::SkewHermTridiagonal) = S

AbstractMatrix{T}(S::SkewHermTridiagonal) where {T} =
    SkewHermTridiagonal(convert(AbstractVector{T}, S.ev)::AbstractVector{T},convert(AbstractVector{<:Real}, S.dvim)::AbstractVector{<:Real})
    
function Base.Matrix{T}(M::SkewHermTridiagonal) where T
    n = size(M, 1)
    Mf = zeros(T, n, n)
    n == 0 && return Mf
    if M.dvim !== nothing
        @inbounds for i = 1:n-1
            Mf[i,i]= M.dvim[i]*1im
            Mf[i+1,i] = M.ev[i]
            Mf[i,i+1] = -M.ev[i]'
        end
        Mf[n,n]=M.dvim[n]*1im
    else
        @inbounds for i = 1:n-1
            Mf[i+1,i] = M.ev[i]
            Mf[i,i+1] = -M.ev[i]'
        end
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


Base.similar(S::SkewHermTridiagonal, ::Type{T}) where {T} = SkewHermTridiagonal(similar(S.ev, T),similar(S.dvim,T))

Base.similar(S::SkewHermTridiagonal, ::Type{T}, dims::Union{Dims{1},Dims{2}}) where {T} = zeros(T, dims...)
function Base.copyto!(dest::SkewHermTridiagonal, src::SkewHermTridiagonal)
    (copyto!(dest.ev, src.ev);dest)
    if src.dvim !== nothing
        (copyto!(dest.dvim, src.dvim);dest)
    end
end

#Elementary operations
for func in (:conj, :copy)
    @eval Base.$func(M::SkewHermTridiagonal) = SkewHermTridiagonal(($func)(M.ev),($func)(M.dvim))
end
Base.imag(M::SkewHermTridiagonal) = LA.SymTridiagonal(imag.(M.dvim),imag.(M.ev))
Base.real(M::SkewHermTridiagonal) = SkewHermTridiagonal(real.(M.ev))

Base.transpose(S::SkewHermTridiagonal) = -S
Base.adjoint(S::SkewHermTridiagonal{<:Real}) = -S
Base.adjoint(S::SkewHermTridiagonal) = -conj.(S)

Base.copy(S::SkewHermTridiagonal)=SkewHermTridiagonal(copy(S.ev),copy(S.dvim))
Base.copy(S::LA.Adjoint{<:Any,<:SkewHermTridiagonal}) = SkewHermTridiagonal(map(x -> copy.(adjoint.(x)), (S.parent.ev,S.parent.dvim))...)

isskewhermitian(S::SkewHermTridiagonal) = true


function Base.:+(A::SkewHermTridiagonal, B::SkewHermTridiagonal) 
    if A.dvim !== nothing && B.dvim !== nothing
        return SkewHermTridiagonal(A.ev+B.ev,A.dvim+B.dvim)
    elseif A.dvim === nothing && B.dvim !== nothing
        return SkewHermTridiagonal(A.ev+B.ev,B.dvim)
    elseif B.dvim === nothing && A.dvim !== nothing
        return SkewHermTridiagonal(A.ev+B.ev,A.dvim)
    else
        return SkewHermTridiagonal(A.ev+B.ev)
    end
end
function Base.:-(A::SkewHermTridiagonal, B::SkewHermTridiagonal) 
    if A.dvim !== nothing && B.dvim !== nothing
        return SkewHermTridiagonal(A.ev-B.ev,A.dvim-B.dvim)
    elseif A.dvim === nothing && B.dvim !== nothing
        return SkewHermTridiagonal(A.ev-B.ev,-B.dvim)
    elseif B.dvim === nothing && A.dvim !== nothing
        return SkewHermTridiagonal(A.ev-B.ev,A.dvim)
    else
        return SkewHermTridiagonal(A.ev-B.ev)
    end
end
function Base.:-(A::SkewHermTridiagonal) 
    if A.dvim !== nothing 
        return SkewHermTridiagonal(-A.ev,-A.dvim)
    else
        return SkewHermTridiagonal(-A.ev)
    end
end

function Base.:*(A::SkewHermTridiagonal, B::Number) 
    if A.dvim !== nothing 
        return SkewHermTridiagonal(A.ev*B,A.dvim*B)
    else
        return SkewHermTridiagonal(A.ev*B)
    end
end
function Base.:*(B::Number,A::SkewHermTridiagonal) 
    if A.dvim !== nothing 
        return SkewHermTridiagonal(B*A.ev,B*A.dvim)
    else
        return SkewHermTridiagonal(B*A.ev)
    end
end
function Base.:/(A::SkewHermTridiagonal, B::Number) 
    if A.dvim !== nothing 
        return SkewHermTridiagonal(A.ev/B,A.dvim/B)
    else
        return SkewHermTridiagonal(A.ev/B)
    end
end
function Base.:\(B::Number,A::SkewHermTridiagonal) 
    if A.dvim !== nothing 
        return SkewHermTridiagonal(B\A.ev,B \A.dvim)
    else
        return SkewHermTridiagonal(B\A.ev)
    end
end
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
    α =S.dvim
    β = S.ev
    if α === nothing
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
    else
        @inbounds begin
            for j = 1:n
                x₊ = B[1, j]
                x₀ = zero(x₊)
                # If m == 1 then β[1] is out of bounds
                β₀ = m > 1 ? zero(β[1]) : zero(eltype(β))
                for i = 1:m - 1
                    x₋, x₀, x₊ = x₀, x₊, B[i + 1, j]
                    β₋, β₀ = β₀, β[i]
                    LA._modify!(_add, β₋*x₋ +α[i]*x₀*1im -adjoint(β₀)*x₊, C, (i, j))
                end
                LA._modify!(_add, β[m-1]*x₀+α[m]*x₊*1im , C, (m, j))
            end
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
    dv=S.dvim
    ev = S.ev
    x₀ = x[1]
    x₊ = x[2]
    sub = ev[1]
    if dv !== nothing
        r = dot( adjoint(sub)*x₊+complex(zero(dv[1]),-dv[1])*x₀, y[1])
        @inbounds for j in 2:nx-1
            x₋, x₀, x₊ = x₀, x₊, x[j+1]
            sup, sub = -adjoint(sub), ev[j]
            r += dot(adjoint(sup)*x₋+complex(zero(dv[j]),-dv[j])*x₀ + adjoint(sub)*x₊, y[j])
        end
        r += dot(-sub*x₀+complex(zero(dv[nx]),-dv[nx])*x₊, y[nx])
    else
        r = dot( adjoint(sub)*x₊, y[1])
        @inbounds for j in 2:nx-1
            x₋, x₀, x₊ = x₀, x₊, x[j+1]
            sup, sub = -adjoint(sub), ev[j]
            r += dot(adjoint(sup)*x₋ + adjoint(sub)*x₊, y[j])
        end
        r += dot(adjoint(-adjoint(sub))*x₀, y[nx])
    end
    return r
end

#Base.:\(T::SkewHermTridiagonal, B::StridedVecOrMat) = Base.ldlt(T)\B

@views function LA.eigvals!(A::SkewHermTridiagonal{T,V,Vim}, sortby::Union{Function,Nothing}=nothing) where {T<:Real,V,Vim<:Nothing}
    vals = skeweigvals!(A)
    !isnothing(sortby) && sort!(vals, by=sortby)
    return complex.(0, vals)
end

@views function LA.eigvals!(A::SkewHermTridiagonal{T,V,Vim}, irange::UnitRange) where {T<:Real,V,Vim<:Nothing}
    vals = skewtrieigvals!(A,irange)
    return complex.(0, vals)
end

@views function LA.eigvals!(A::SkewHermTridiagonal{T,V,Vim}, vl::Real,vh::Real) where {T<:Real,V,Vim<:Nothing}
    vals = skewtrieigvals!(A,-vh,-vl)
    return complex.(0, vals)
end

LA.eigvals(A::SkewHermTridiagonal{T,V,Vim}, irange::UnitRange) where {T<:Real,V,Vim<:Nothing} =
    LA.eigvals!(copyeigtype(A), irange)
LA.eigvals(A::SkewHermTridiagonal{T,V,Vim}, vl::Real,vh::Real)  where {T<:Real,V,Vim<:Nothing}=
    LA.eigvals!(copyeigtype(A), vl,vh)



@views function skewtrieigvals!(S::SkewHermTridiagonal{T,V,Vim}) where {T<:Real,V,Vim<:Nothing}
    n = size(S,1)
    H = SymTridiagonal(zeros(eltype(S.ev),n),S.ev)
    vals = eigvals!(H)
    return vals .= .-vals

end

@views function skewtrieigvals!(S::SkewHermTridiagonal{T,V,Vim},irange::UnitRange) where {T<:Real,V,Vim<:Nothing}
    n = size(S,1)
    H = SymTridiagonal(zeros(eltype(S.ev),n),S.ev)
    vals = eigvals!(H,irange)
    return vals .= .-vals

end

@views function skewtrieigvals!(S::SkewHermTridiagonal{T,V,Vim},vl::Real,vh::Real) where {T<:Real,V,Vim<:Nothing}
    n = size(S,1)
    H = SymTridiagonal(zeros(eltype(S.ev),n),S.ev)
    vals = eigvals!(H,vl,vh)
    return vals .= .-vals
end

@views function skewtrieigen!(S::SkewHermTridiagonal{T,V,Vim}) where {T<:Real,V,Vim<:Nothing}

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


@views function LA.eigen!(A::SkewHermTridiagonal{T,V,Vim}) where {T<:Real,V,Vim<:Nothing}
     return skewtrieigen!(A)
end

function copyeigtype(A::SkewHermTridiagonal)  
    temp=similar(A,LA.eigtype(eltype(A.ev)))
    copyto!(temp ,A)
    display(temp)
    return temp
end
#fix copyeigtype
LA.eigen(A::SkewHermTridiagonal{T,V,Vim}) where {T<:Real,V,Vim<:Nothing}=LA.eigen!(A)

LA.eigvecs(A::SkewHermTridiagonal{T,V,Vim})  where {T<:Real,V,Vim<:Nothing}= eigen(A).vectors

@views function LA.svdvals!(A::SkewHermTridiagonal{T,V,Vim}) where {T<:Real,V,Vim<:Nothing}
    n=size(A,1)
    vals = skewtrieigvals!(A)
    vals .= abs.(vals)
    return sort!(vals; rev=true)
end
LA.svdvals(A::SkewHermTridiagonal{T,V,Vim}) where {T<:Real,V,Vim<:Nothing}=svdvals!(A)

@views function LA.svd!(A::SkewHermTridiagonal{T,V,Vim}) where {T<:Real,V,Vim<:Nothing}
    n=size(A,1)
    E=eigen!(A)
    U=E.vectors
    vals = imag.(E.values)
    I=sortperm(vals;by=abs,rev=true)
    permute!(vals,I)
    Base.permutecols!!(U,I)
    V2 = U .* -1im
    @inbounds for i=1:n
        if vals[i] < 0
            vals[i]=-vals[i]
            @simd for j=1:n
                V2[j,i]=-V2[j,i]
            end
        end
    end
    return LA.SVD(U,vals,adjoint(V2))
end

LA.svd(A::SkewHermTridiagonal{T,V,Vim})  where {T<:Real,V,Vim<:Nothing}= svd!(A)

###################
# Generic methods #
###################

# det with optional diagonal shift for use with shifted Hessenberg factorizations
#det(A::SkewHermTridiagonal; shift::Number=false) = det_usmani(A.ev, A.dv, A.ev, shift)
#logabsdet(A::SkewHermTridiagonal; shift::Number=false) = logabsdet(ldlt(A; shift=shift))

Base.@propagate_inbounds function Base.getindex(A::SkewHermTridiagonal{T}, i::Integer, j::Integer) where T
    @boundscheck checkbounds(A, i, j)
    if i == j + 1
        return @inbounds A.ev[j]
    elseif i + 1 == j
        return @inbounds -A.ev[i]'
    elseif T <: Complex && i == j
        return complex(zero(real(T)), A.dvim[i])
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
    elseif T <: Complex && i == j && isreal(x)
        @inbounds A.dvim[i]== x
    elseif T <: Complex && i == j && imag(x)!=0
        @inbounds A.dvim[i]== imag(x)
    else
        throw(ArgumentError("cannot set off-diagonal entry ($i, $j)"))
    end
    return x
end



