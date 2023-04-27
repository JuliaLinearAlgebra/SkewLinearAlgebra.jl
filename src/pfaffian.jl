# This file is a part of Julia. License is MIT: https://julialang.org/license

#using LinearAlgebra: exactdiv
if isdefined(LA,:exactdiv)
    const exactdiv = LA.exactdiv
else
    exactdiv(a,b) = a / b
    exactdiv(a::Integer, b::Integer) = div(a, b)
end

# in-place O(n³) algorithm to compute the exact Pfaffian of
# a skew-symmetric matrix over integers (or potentially any ring supporting exact division).
#
#     G. Galbiati & F. Maffioli, "On the computation of pfaffians,"
#     Discrete Appl. Math. 51, 269–275 (1994).
#     https://doi.org/10.1016/0166-218X(92)00034-J
function _exactpfaffian!(A::AbstractMatrix)
    n = size(A,1)
    isodd(n) && return zero(eltype(A))
    c = one(eltype(A))
    signflip = false
    n = n ÷ 2
    while n > 1
        # find last k with A[2n-1,k] ≠ 0
        k = 2n
        while k > 0 && iszero(A[2n-1,k]); k -= 1; end
        iszero(k) && return zero(eltype(A))

        # swap rows/cols k and 2n
        if k != 2n
            for i = 1:2n
                A[k,i], A[2n,i] = A[2n,i], A[k,i] # swap rows
            end
            for i = 1:2n
                A[i,k], A[i,2n] = A[i,2n], A[i,k] # swap cols
            end
            signflip = !signflip
        end

        # update, A, c, n
        for j = 1:2n-2, i = 1:j-1
            δ = A[2n-1,2n]*A[i,j] - A[i,2n-1]*A[j,2n] + A[j,2n-1]*A[i,2n]
            A[j,i] = -(A[i,j] = exactdiv(δ, c))
            # @assert A[i,j] * c == δ
        end
        c = A[2n-1,2n]
        n -= 1
    end
    return signflip ? -A[1,2] : A[1,2]
end

function exactpfaffian!(A::AbstractMatrix)
    LinearAlgebra.require_one_based_indexing(A)
    isskewhermitian(A) || throw(ArgumentError("Pfaffian requires a skew-Hermitian matrix"))
    return _exactpfaffian!(A)
end
exactpfaffian!(A::SkewHermitian) = _exactpfaffian!(A.data)
exactpfaffian(A::AbstractMatrix) = exactpfaffian!(copyto!(similar(A), A))

const ExactRational = Union{BigInt,Rational{BigInt}}
pfaffian!(A::AbstractMatrix{<:ExactRational}) = exactpfaffian!(A)
pfaffian(A::AbstractMatrix{<:ExactRational}) = pfaffian!(copy(A))

# prevent method ambiguities:
pfaffian(A::SkewHermitian{<:ExactRational}) = pfaffian!(copy(A))
pfaffian!(A::SkewHermitian{<:ExactRational}) = exactpfaffian!(A)

function _pfaffian!(A::SkewHermitian{<:Real})
    n = size(A,1)
    isodd(n) && return zero(eltype(A))
    H = hessenberg!(A)
    pf = one(eltype(A))
    T = H.H
    for i=1:2:n-1
        pf *= -T.ev[i]
    end
    return pf * sign(det(H.Q))
end

function _pfaffian!(A::SkewHermTridiagonal{<:Real})
    n = size(A,1)
    isodd(n) && return zero(eltype(A.ev))
    pf = one(eltype(A.ev))
    for i=1:2:n-1
        pf *= -A.ev[i]
    end
    return pf
end

pfaffian!(A::Union{SkewHermitian{<:Real},SkewHermTridiagonal{<:Real}})= _pfaffian!(A)
pfaffian(A::Union{SkewHermitian{<:Real},SkewHermTridiagonal{<:Real}})= pfaffian!(copyeigtype(A))

"""
    pfaffian(A)

Returns the pfaffian of `A` where a is a real skew-Hermitian matrix.
If `A` is not of type `SkewHermitian{<:Real}`, then `isskewhermitian(A)`
is checked to ensure that `A == -A'`
"""
pfaffian(A::AbstractMatrix{<:Real}) = pfaffian!(copyeigtype(A))

"""
    pfaffian!(A)

Similar to [`pfaffian`](@ref), but overwrites `A` in-place with intermediate calculations.
"""
function pfaffian!(A::AbstractMatrix{<:Real})
    isskewhermitian(A) || throw(ArgumentError("Pfaffian requires a skew-Hermitian matrix"))
    return _pfaffian!(SkewHermitian(A))
end

function _logabspfaffian!(A::SkewHermitian{<:Real})
    n = size(A, 1)
    isodd(n) && return convert(eltype(A), -Inf), zero(eltype(A))
    H = hessenberg!(A)
    logpf = zero(eltype(H))
    T = H.H
    sgn = one(eltype(H))
    for i=1:2:n-1
        logpf += log(abs(T.ev[i]))
        sgn *= sign(-T.ev[i])
    end
    return logpf, sgn*sign(det(H.Q))
end
function _logabspfaffian!(A::SkewHermTridiagonal{<:Real})
    n = size(A, 1)
    isodd(n) && return convert(eltype(A.ev), -Inf), zero(eltype(A.ev))
    logpf = zero(eltype(A.ev))
    sgn = one(eltype(A.ev))
    for i=1:2:n-1
        logpf += log(abs(A.ev[i]))
        sgn *= sign(-A.ev[i])
    end
    return logpf, sgn
end
logabspfaffian!(A::Union{SkewHermitian{<:Real},SkewHermTridiagonal{<:Real}})= _logabspfaffian!(A)
logabspfaffian(A::Union{SkewHermitian{<:Real},SkewHermTridiagonal{<:Real}})= logabspfaffian!(copyeigtype(A))

"""
    logabspfaffian(A)

Returns a tuple `(log|Pf A|, sign)`, with the log of the absolute value of the pfaffian of `A` as first output
and the sign (±1) of the pfaffian as second output. A must be a real skew-Hermitian matrix.
If `A` is not of type `SkewHermitian{<:Real}`, then `isskewhermitian(A)`
is checked to ensure that `A == -A'`
"""
logabspfaffian(A::AbstractMatrix{<:Real}) = logabspfaffian!(copyeigtype(A))


"""
    logabspfaffian!(A)

Similar to [`logabspfaffian`](@ref), but overwrites `A` in-place with intermediate calculations.
"""
function logabspfaffian!(A::AbstractMatrix{<:Real})
    isskewhermitian(A) || throw(ArgumentError("Pfaffian requires a skew-Hermitian matrix"))
    return _logabspfaffian!(SkewHermitian(A))
end
