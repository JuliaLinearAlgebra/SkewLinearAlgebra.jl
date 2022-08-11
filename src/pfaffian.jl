using LinearAlgebra: exactdiv

# in-place O(n³) algorithm to compute the exact Pfaffian of
# a skew-symmetric matrix over integers (or potentially any ring supporting exact division).
#
#     G. Galbiati & F. Maffioli, "On the computation of pfaffians,"
#     Discrete Appl. Math. 51, 269–275 (1994).
#     https://doi.org/10.1016/0166-218X(92)00034-J
function exactpfaffian!(A::AbstractMatrix{<:Integer})
    LinearAlgebra.require_one_based_indexing(A)
    isskewhermitian(A) || throw(ArgumentError("Pfaffian requires a skew-Hermitian matrix"))
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
exactpfaffian(A::AbstractMatrix{<:Integer}) =
    exactpfaffian!(copyto!(similar(A, typeof(exactdiv(zero(eltype(A))^2, one(eltype(A))))), A))

function realpfaffian(A::SkewHermitian{<:Real})
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