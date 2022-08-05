# This file is a part of Julia. License is MIT: https://julialang.org/license

function skewexp!(A::SkewHermitian)
    n = size(A,1)
    n == 1 && return fill(exp(A.data[1,1]), 1,1)
    E = skeweigen!(A)

    temp2 = similar(A,n,n)
    Q1=similar(A,n,n)
    Q2=similar(A,n,n)
    Cos=similar(A,n)
    Sin=similar(A,n)

    @simd for i=1:n
        @inbounds Sin[i],Cos[i]=sincos(imag(E.values[i]))
    end
    C=Diagonal(Cos)
    S=Diagonal(Sin)

    mul!(Q1,E.realvectors,C)
    mul!(Q2,E.imagvectors,S)
    Q1 .-= Q2
    mul!(temp2,Q1,transpose(E.realvectors))
    mul!(Q1,E.realvectors,S)
    mul!(Q2,E.imagvectors,C)
    Q1 .+= Q2
    mul!(Q2,Q1,transpose(E.imagvectors))
    temp2 .+= Q2
    return temp2
end

function Base.exp(A::SkewHermitian)
    return skewexp!(copyeigtype(A))
end

function skewcis!(A::SkewHermitian)
    n = size(A,1)
    n == 1 && return fill(cis(A.data[1,1]), 1,1)

    Eig = skeweigen!(A)
    Q = Eig.imagvectors.*1im
    Q.+=Eig.realvectors
    temp = similar(Q,n,n)
    temp2 = similar(Q,n,n)
    eig = @. exp(-imag(Eig.values))
    E=Diagonal(eig)
    mul!(temp,Q,E)
    mul!(temp2,temp,adjoint(Q))
    return temp2
end

@views function skewcos!(A::SkewHermitian)
    n = size(A,1)
    if n == 1
        return exp(A.data*1im)
    end
    Eig = skeweigen!(A)

    temp2 = similar(A,n,n)
    Q1=similar(A,n,n)
    Q2=similar(A,n,n)

    eig = @. exp(-imag(Eig.values))
    E=Diagonal(eig)

    mul!(Q1,Eig.realvectors,E)
    mul!(Q2,Eig.imagvectors,E)
    mul!(temp2,Q1,transpose(Eig.realvectors))
    mul!(Q1,Q2,transpose(Eig.imagvectors))
    Q1 .+= temp2
    return Q1
end
@views function skewsin!(A::SkewHermitian)
    n = size(A,1)
    if n == 1
        return exp(A.data*1im)
    end
    Eig = skeweigen!(A)

    temp2 = similar(A,n,n)
    Q1=similar(A,n,n)
    Q2=similar(A,n,n)

    eig = @. exp(-imag(Eig.values))
    E=Diagonal(eig)

    mul!(Q1,Eig.realvectors,E)
    mul!(Q2,Eig.imagvectors,E)
    mul!(temp2,Q1,transpose(Eig.imagvectors))
    mul!(Q1,Q2,transpose(Eig.realvectors))
    Q1 .-= temp2
    return Q1
end
Base.cis(A::SkewHermitian) = skewcis!(copyeigtype(A))
Base.cos(A::SkewHermitian) = skewcos!(copyeigtype(A))
Base.sin(A::SkewHermitian) = skewsin!(copyeigtype(A))

function Base.tan(A::SkewHermitian)
    E=cis(A)
    S=imag(E)
    C=real(E)
    return C\S
end
Base.sinh(A::SkewHermitian) = skewhermitian!(exp(A))
Base.cosh(A::SkewHermitian) = hermitian!(exp(A))

function Base.tanh(A::SkewHermitian)
    E = skewexp!(2A)
    return (E+LA.I)\(E-LA.I)
end

# someday this should be in LinearAlgebra: https://github.com/JuliaLang/julia/pull/31836
function hermitian!(A::AbstractMatrix{<:Number})
    LA.require_one_based_indexing(A)
    n = LA.checksquare(A)
    @inbounds for i in 1:n
        A[i,i] = real(A[i,i])
        for j = 1:i-1
            A[i,j] = A[j,i] = (A[i,j] + A[j,i]')/2
        end
    end
    return LA.Hermitian(A)
end

