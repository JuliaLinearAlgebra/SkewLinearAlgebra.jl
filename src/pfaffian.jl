#using LinearAlgebra: exactdiv
if isdefined(LA,:exactdiv)
    const exactdiv = LA.exactdiv
else
    exactdiv(a,b) = a/ b
    excatdiv(a::Integer, b::Integer) = div(a, b)
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
                A[i,k], A[i,2n] = A[i,2n], A[i,k]
                A[k,i], A[2n,i] = A[2n,i], A[k,i]
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

exactpfaffian!(A::SkewHermitian{<:BigInt}) = _exactpfaffian!(A.data)
pfaffian!(A::SkewHermitian{<:BigInt}) = _exactpfaffian!(A.data)
pfaffian(A::SkewHermitian{<:BigInt}) = pfaffian!(copy(A))
pfaffian!(A::AbstractMatrix{<:BigInt}) = exactpfaffian!(A)
pfaffian(A::AbstractMatrix{<:BigInt}) = pfaffian!(copy(A))

function _pfaffian!(A::SkewHermitian{<:Real})
    n=size(A,1)
    if n%2==1
        return convert(eltype(A.data),0)
    end 
    H=hessenberg(A)
    pf=convert(eltype(A.data),1)
    T=H.H
    for i=1:2:n-1
        pf *= -T.ev[i]
    end
    return pf
end

pfaffian!(A::SkewHermitian{<:Real})= _pfaffian!(A)
pfaffian(A::SkewHermitian{<:Real})= pfaffian!(copyeigtype(A))
pfaffian(A::AbstractMatrix{<:Real}) = pfaffian!(copy(A))

function pfaffian!(A::AbstractMatrix{<:Real})
    isskewhermitian(A) || throw(ArgumentError("Pfaffian requires a skew-Hermitian matrix"))
    return _pfaffian!(SkewHermitian(A))
end

