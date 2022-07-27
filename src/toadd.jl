## Addition/subtraction
for f ∈ (:+, :-), (Wrapper, conjugation) ∈ ((:SkewSymmetric, :transpose))
    @eval begin
        function $f(A::$Wrapper, B::$Wrapper)
            return $Wrapper($f($conjugation(parent(A)), $conjugation(parent(B))))
            
        end
    end
end
"""
factorize(A::SkewSymmetric) = _factorize(A)
function _factorize(A::SkewSymmetric{T}; check::Bool=true) where T
    TT = typeof(sqrt(oneunit(T)))
    if TT <: BlasFloat
        return bunchkaufman(A; check=check)
    else # fallback
        return lu(A; check=check)
    end
end

det(A::RealHermSymComplexHerm) = real(det(_factorize(A; check=false)))
det(A::Symmetric{<:Real}) = det(_factorize(A; check=false))
det(A::Symmetric) = det(_factorize(A; check=false))
"""
#\(A::HermOrSym{<:Any,<:StridedMatrix}, B::AbstractVector) = \(factorize(A), B)
# Bunch-Kaufman solves can not utilize BLAS-3 for multiple right hand sides
# so using LU is faster for AbstractMatrix right hand side
#\(A::HermOrSym{<:Any,<:StridedMatrix}, B::AbstractMatrix) = \(lu(A), B)
"""
function _inv(A::HermOrSym)
    n = checksquare(A)
    B = inv!(lu(A))
    conjugate = isa(A, Hermitian)
    # symmetrize
    if A.uplo == 'U' # add to upper triangle
        @inbounds for i = 1:n, j = i:n
            B[i,j] = conjugate ? (B[i,j] + conj(B[j,i])) / 2 : (B[i,j] + B[j,i]) / 2
        end
    else # A.uplo == 'L', add to lower triangle
        @inbounds for i = 1:n, j = i:n
            B[j,i] = conjugate ? (B[j,i] + conj(B[i,j])) / 2 : (B[j,i] + B[i,j]) / 2
        end
    end
    B
end

inv(A::Hermitian{<:Any,<:StridedMatrix}) = Hermitian(_inv(A), sym_uplo(A.uplo))
inv(A::Symmetric{<:Any,<:StridedMatrix}) = Symmetric(_inv(A), sym_uplo(A.uplo))
"""
function svd(A::RealHermSymComplexHerm; full::Bool=false)
vals, vecs = eigen(A)
I = sortperm(vals; by=abs, rev=true)
permute!(vals, I)
Base.permutecols!!(vecs, I)         # left-singular vectors
V = copy(vecs)                      # right-singular vectors
# shifting -1 from singular values to right-singular vectors
@inbounds for i = 1:length(vals)
    if vals[i] < 0
        vals[i] = -vals[i]
        for j = 1:size(V,1); V[j,i] = -V[j,i]; end
    end
end
return SVD(vecs, vals, V')
end

function svdvals!(A::RealHermSymComplexHerm)
vals = eigvals!(A)
for i = 1:length(vals)
    vals[i] = abs(vals[i])
end
return sort!(vals, rev = true)
end

# Matrix functions
^(A::Symmetric{<:Real}, p::Integer) = sympow(A, p)
^(A::Symmetric{<:Complex}, p::Integer) = sympow(A, p)
function sympow(A::Symmetric, p::Integer)
if p < 0
    return Symmetric(Base.power_by_squaring(inv(A), -p))
else
    return Symmetric(Base.power_by_squaring(A, p))
end
end
function ^(A::Symmetric{<:Real}, p::Real)
isinteger(p) && return integerpow(A, p)
F = eigen(A)
if all(λ -> λ ≥ 0, F.values)
    return Symmetric((F.vectors * Diagonal((F.values).^p)) * F.vectors')
else
    return Symmetric((F.vectors * Diagonal((complex(F.values)).^p)) * F.vectors')
end
end
function ^(A::Symmetric{<:Complex}, p::Real)
isinteger(p) && return integerpow(A, p)
return Symmetric(schurpow(A, p))
end
function ^(A::Hermitian, p::Integer)
if p < 0
    retmat = Base.power_by_squaring(inv(A), -p)
else
    retmat = Base.power_by_squaring(A, p)
end
for i = 1:size(A,1)
    retmat[i,i] = real(retmat[i,i])
end
return Hermitian(retmat)
end
function ^(A::Hermitian{T}, p::Real) where T
isinteger(p) && return integerpow(A, p)
F = eigen(A)
if all(λ -> λ ≥ 0, F.values)
    retmat = (F.vectors * Diagonal((F.values).^p)) * F.vectors'
    if T <: Real
        return Hermitian(retmat)
    else
        for i = 1:size(A,1)
            retmat[i,i] = real(retmat[i,i])
        end
        return Hermitian(retmat)
    end
else
    return (F.vectors * Diagonal((complex(F.values).^p))) * F.vectors'
end
end

for func in (:exp, :cos, :sin, :tan, :cosh, :sinh, :tanh, :atan, :asinh, :atanh)
@eval begin
    function ($func)(A::HermOrSym{<:Real})
        F = eigen(A)
        return Symmetric((F.vectors * Diagonal(($func).(F.values))) * F.vectors')
    end
    function ($func)(A::Hermitian{<:Complex})
        n = checksquare(A)
        F = eigen(A)
        retmat = (F.vectors * Diagonal(($func).(F.values))) * F.vectors'
        for i = 1:n
            retmat[i,i] = real(retmat[i,i])
        end
        return Hermitian(retmat)
    end
end
end

function cis(A::Union{RealHermSymComplexHerm,SymTridiagonal{<:Real}})
F = eigen(A)
# The returned matrix is unitary, and is complex-symmetric for real A
return F.vectors .* cis.(F.values') * F.vectors'
end

for func in (:acos, :asin)
@eval begin
    function ($func)(A::HermOrSym{<:Real})
        F = eigen(A)
        if all(λ -> -1 ≤ λ ≤ 1, F.values)
            retmat = (F.vectors * Diagonal(($func).(F.values))) * F.vectors'
        else
            retmat = (F.vectors * Diagonal(($func).(complex.(F.values)))) * F.vectors'
        end
        return Symmetric(retmat)
    end
    function ($func)(A::Hermitian{<:Complex})
        n = checksquare(A)
        F = eigen(A)
        if all(λ -> -1 ≤ λ ≤ 1, F.values)
            retmat = (F.vectors * Diagonal(($func).(F.values))) * F.vectors'
            for i = 1:n
                retmat[i,i] = real(retmat[i,i])
            end
            return Hermitian(retmat)
        else
            return (F.vectors * Diagonal(($func).(complex.(F.values)))) * F.vectors'
        end
    end
end
end

function acosh(A::HermOrSym{<:Real})
F = eigen(A)
if all(λ -> λ ≥ 1, F.values)
    retmat = (F.vectors * Diagonal(acosh.(F.values))) * F.vectors'
else
    retmat = (F.vectors * Diagonal(acosh.(complex.(F.values)))) * F.vectors'
end
return Symmetric(retmat)
end
function acosh(A::Hermitian{<:Complex})
n = checksquare(A)
F = eigen(A)
if all(λ -> λ ≥ 1, F.values)
    retmat = (F.vectors * Diagonal(acosh.(F.values))) * F.vectors'
    for i = 1:n
        retmat[i,i] = real(retmat[i,i])
    end
    return Hermitian(retmat)
else
    return (F.vectors * Diagonal(acosh.(complex.(F.values)))) * F.vectors'
end
end

function sincos(A::HermOrSym{<:Real})
n = checksquare(A)
F = eigen(A)
S, C = Diagonal(similar(A, (n,))), Diagonal(similar(A, (n,)))
for i in 1:n
    S.diag[i], C.diag[i] = sincos(F.values[i])
end
return Symmetric((F.vectors * S) * F.vectors'), Symmetric((F.vectors * C) * F.vectors')
end
function sincos(A::Hermitian{<:Complex})
    n = checksquare(A)
    F = eigen(A)
    S, C = Diagonal(similar(A, (n,))), Diagonal(similar(A, (n,)))
    for i in 1:n
        S.diag[i], C.diag[i] = sincos(F.values[i])
    end
    retmatS, retmatC = (F.vectors * S) * F.vectors', (F.vectors * C) * F.vectors'
    for i = 1:n
        retmatS[i,i] = real(retmatS[i,i])
        retmatC[i,i] = real(retmatC[i,i])
    end
    return Hermitian(retmatS), Hermitian(retmatC)
end


for func in (:log, :sqrt)
    # sqrt has rtol arg to handle matrices that are semidefinite up to roundoff errors
    rtolarg = func === :sqrt ? Any[Expr(:kw, :(rtol::Real), :(eps(real(float(one(T))))*size(A,1)))] : Any[]
    rtolval = func === :sqrt ? :(-maximum(abs, F.values) * rtol) : 0
    @eval begin
        function ($func)(A::HermOrSym{T}; $(rtolarg...)) where {T<:Real}
            F = eigen(A)
            λ₀ = $rtolval # treat λ ≥ λ₀ as "zero" eigenvalues up to roundoff
            if all(λ -> λ ≥ λ₀, F.values)
                retmat = (F.vectors * Diagonal(($func).(max.(0, F.values)))) * F.vectors'
            else
                retmat = (F.vectors * Diagonal(($func).(complex.(F.values)))) * F.vectors'
            end
            return Symmetric(retmat)
        end

        function ($func)(A::Hermitian{T}; $(rtolarg...)) where {T<:Complex}
            n = checksquare(A)
            F = eigen(A)
            λ₀ = $rtolval # treat λ ≥ λ₀ as "zero" eigenvalues up to roundoff
            if all(λ -> λ ≥ λ₀, F.values)
                retmat = (F.vectors * Diagonal(($func).(max.(0, F.values)))) * F.vectors'
                for i = 1:n
                    retmat[i,i] = real(retmat[i,i])
                end
                return Hermitian(retmat)
            else
                retmat = (F.vectors * Diagonal(($func).(complex(F.values)))) * F.vectors'
                return retmat
            end
        end
    end
end

