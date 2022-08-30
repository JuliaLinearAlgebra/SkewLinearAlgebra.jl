# This file is a part of Julia. License is MIT: https://julialang.org/license

struct SkewCholesky{T,R<:UpperTriangular{<:T},J<:JMatrix{<:T},P<:AbstractVector{<:Integer}}
    R::R #Uppertriangular matrix
    J::J # Block diagonal skew-symmetric matrix of type JMatrix
    p::P #Permutation vector

    function SkewCholesky{T,R,J,P}(Rm,Jm,pv) where {T,R,J,P}
        LA.require_one_based_indexing(Rm)
        new{T,R,J,P}(Rm,Jm,pv)
    end
end

"""
    SkewCholesky(R,p)

Construct a `SkewCholesky` structure from the `UpperTriangular`
matrix `R` and the permutation vector `p`. A matrix `J` of type `JMatrix`
is build calling this function.
The `SkewCholesky` structure has three arguments: `R`,`J` and `p`.
"""
function SkewCholesky(R::UpperTriangular{<:T},p::AbstractVector{<:Integer}) where {T<:Real}
    n = size(R, 1)
    return SkewCholesky{T,typeof(R),JMatrix{T,+1},typeof(p)}(R, JMatrix{T,+1}(n), p)

end

function _skewchol!(A::SkewHermitian{<:Real})
    @views B = A.data
    tol = 1e-15 * norm(B)
    m = size(B,1)
    m == 1 && return [1]
    J2 = similar(B,2,2)
    J2[1,1] = 0; J2[2,1] = -1; J2[1,2] = 1; J2[2,2] = 0
    ii = 0; jj = 0; kk = 0
    P = Array(1:m)
    tempM = similar(B,2,m-2)
    for j = 1:m÷2
        j2 = 2*j
        M = findmax(B[j2-1:m,j2-1:m])
        ii = M[2][1] + j2 - 2
        jj = M[2][2] + j2 - 2

        abs(B[ii,jj])<tol && return P

        kk= (jj == j2-1 ? ii : jj)

        if ii != j2-1
            P[ii],P[j2-1] = P[j2-1],P[ii]
            for t = 1:m
                B[t,ii], B[t,j2-1] = B[t,j2-1], B[t,ii]
            end
            for t = 1:m
                B[ii,t], B[j2-1,t] = B[j2-1,t], B[ii,t]
            end

        end
        if kk != j2
            P[kk],P[j2] = P[j2],P[kk]
            for t = 1:m
                B[t,kk], B[t,j2] = B[t,j2], B[t,kk]
            end
            for t = 1:m
                B[kk,t], B[j2,t] = B[j2,t], B[kk,t]
            end
        end

        l = m-j2
        r = sqrt(B[j2-1,j2])
        B[j2-1,j2-1] = r
        B[j2,j2] = r
        B[j2-1,j2] = 0
        @views mul!(tempM[:,1:l], J2, B[j2-1:j2,j2+1:m])
        B[j2-1:j2,j2+1:m] .= tempM[:,1:l]
        B[j2-1:j2,j2+1:m] .*= (-1/r)
        @views mul!(tempM[:,1:l], J2, B[j2-1:j2,j2+1:m])
        @views mul!(B[j2+1:m,j2+1:m], transpose(B[j2-1:j2,j2+1:m]), tempM[:,1:l],-1,1)

    end
    return P
end
copyeigtype(A::AbstractMatrix) = copyto!(similar(A, LA.eigtype(eltype(A))), A)

@views function skewchol!(A::SkewHermitian)
    P = _skewchol!(A)
    return SkewCholesky(UpperTriangular(A.data), P)
end

skewchol(A::SkewHermitian) = skewchol!(copyeigtype(A))
skewchol!(A::AbstractMatrix) = @views  skewchol!(SkewHermitian(A))

"""
    skewchol(A)

Computes a Cholesky-like factorization of the real skew-symmetric matrix `A`.
The function returns a `SkewCholesky` structure composed of three fields:
`R`,`J`,`p`. `R` is `UpperTriangular`, `J` is a `JMatrix`,
`p` is an array of integers. Let `S` be the returned structure, then the factorization
is such that `S.R'*S.J*S.R = A[S.p,S.p]`

This factorization (and the underlying algorithm) is described in from P. Benner et al,
"[Cholesky-like factorizations of skew-symmetric matrices](https://etna.ricam.oeaw.ac.at/vol.11.2000/pp85-93.dir/pp85-93.pdf)"(2000).
"""
function skewchol(A::AbstractMatrix)
    isskewhermitian(A) || throw(ArgumentError("Pfaffian requires a skew-Hermitian matrix"))
    return skewchol!(SkewHermitian(copyeigtype(A)))
end



