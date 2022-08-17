# This file is a part of Julia. License is MIT: https://julialang.org/license

function skewexp!(A::SkewHermitian{<:Real})
    n = size(A, 1)
    n == 1 && return fill(exp(A.data[1,1]), 1, 1)
    vals, Qr, Qim = skeweigen!(A)

    temp2 = similar(A, n, n)
    Q1 = similar(A, n, n)
    Q2 = similar(A, n, n)
    Cos = similar(A, n)
    Sin = similar(A, n)

    @simd for i = 1 : n
        @inbounds Sin[i], Cos[i] = sincos(imag(vals[i]))
    end
    C = Diagonal(Cos)
    S = Diagonal(Sin)

    mul!(Q1, Qr, C)
    mul!(Q2, Qim, S)
    Q1 .-= Q2
    mul!(temp2, Q1, transpose(Qr))
    mul!(Q1, Qr, S)
    mul!(Q2, Qim, C)
    Q1 .+= Q2
    mul!(Q2, Q1, transpose(Qim))
    temp2 .+= Q2
    return temp2
end

@views function skewexp!(A::SkewHermitian{<:Complex})
    n = size(A, 1)
    n == 1 && return fill(exp(A.data[1,1]), 1, 1)
    Eig = eigen!(A)
    eig = exp.(Eig.values)
    temp = similar(A, n, n)
    Exp = similar(A, n, n)
    mul!(temp, Diagonal(eig), Eig.vectors')
    mul!(Exp,Eig.vectors,temp)
    return Exp
end

function Base.exp(A::SkewHermitian)
    return skewexp!(copyeigtype(A))
end

@views function skewcis!(A::SkewHermitian{<:Real})
    n = size(A, 1)
    n == 1 && return fill(cis(A.data[1,1]), 1, 1)
    Eig = eigen!(A)
    Q = Eig.vectors
    temp = similar(Q, n, n)
    temp2 = similar(Q, n, n)
    eig = @. exp(-imag(Eig.values))
    E = Diagonal(eig)
    mul!(temp, Q, E)
    mul!(temp2, temp, adjoint(Q))
    return Hermitian(temp2)
end

@views function skewcis!(A::SkewHermitian{<:Complex})
    n = size(A,1)
    n == 1 && return fill(exp(A.data[1,1]), 1,1)
    Eig = eigen!(A)
    eig = @. exp(-imag(Eig.values))
    Cis = similar(A, n, n)
    temp = similar(A, n, n)
    mul!(temp, Eig.vectors, Diagonal(eig))
    mul!(Cis, temp, Eig.vectors')
    return Hermitian(Cis)
end

@views function skewcos!(A::SkewHermitian{<:Real})
    n = size(A,1)
    if n == 1
        return exp(A.data*1im)
    end
    vals, Qr, Qim = skeweigen!(A)
    temp2 = similar(A, n, n)
    Q1 = similar(A, n, n)
    Q2 = similar(A, n, n)
    eig = @. exp(-imag(vals))
    E = Diagonal(eig)
    mul!(Q1, Qr, E)
    mul!(Q2, Qim, E)
    mul!(temp2, Q1, transpose(Qr))
    mul!(Q1, Q2, transpose(Qim))
    Q1 .+= temp2
    return Symmetric(Q1)
end

@views function skewcos!(A::SkewHermitian{<:Complex})
    n = size(A,1)
    n == 1 && return fill(exp(A.data[1,1]), 1,1)
    Eig = eigen!(A)
    eig1 = @. exp(-imag(Eig.values))
    eig2 = @. exp(imag(Eig.values))
    Cos = similar(A, n, n)
    temp = similar(A, n, n)
    temp2 = similar(A, n, n)
    mul!(temp, Eig.vectors, Diagonal(eig1))
    mul!(temp2, temp, Eig.vectors')
    mul!(temp, Eig.vectors, Diagonal(eig2))
    mul!(Cos, temp, Eig.vectors')
    Cos .+= temp2
    Cos ./= 2
    return Hermitian(Cos)
end

@views function skewsin!(A::SkewHermitian{<:Real})
    n = size(A, 1)
    if n == 1
        return exp(A.data * 1im)
    end
    vals, Qr, Qim = skeweigen!(A)
    temp2 = similar(A, n, n)
    Q1 = similar(A, n, n)
    Q2 = similar(A, n, n)
    eig = @. exp(-imag(vals))
    E = Diagonal(eig)
    mul!(Q1, Qr, E)
    mul!(Q2, Qim, E)
    mul!(temp2, Q1, transpose(Qim))
    mul!(Q1, Q2, transpose(Qr))
    Q1 .-= temp2
    return Q1
end

@views function skewsin!(A::SkewHermitian{<:Complex})
    n = size(A,1)
    n == 1 && return fill(exp(A.data[1,1]), 1, 1)
    Eig = eigen!(A)
    eig1 = @. exp(-imag(Eig.values))
    eig2 = @. exp(imag(Eig.values))
    Sin = similar(A,n,n)
    temp = similar(A,n,n)
    temp2 = similar(A,n,n)
    mul!(temp,Eig.vectors,Diagonal(eig1))
    mul!(Sin,temp,Eig.vectors')
    mul!(temp,Eig.vectors,Diagonal(eig2))
    mul!(temp2,temp,Eig.vectors')
    Sin .-= temp2
    Sin ./= -2
    Sin .*= 1im
    return Sin
end

Base.cis(A::SkewHermitian) = skewcis!(copyeigtype(A))
Base.cos(A::SkewHermitian) = skewcos!(copyeigtype(A))
Base.sin(A::SkewHermitian) = skewsin!(copyeigtype(A))

@views function skewsincos!(A::SkewHermitian{<:Complex})
    n = size(A, 1)
    n == 1 && return fill(exp(A.data[1,1]), 1,1)
    Eig = eigen!(A)
    eig1 = @. exp(-imag(Eig.values))
    eig2 = @. exp(imag(Eig.values))
    Sin = similar(A, n, n)
    Cos = similar(A, n, n)
    temp = similar(A, n, n)
    temp2 = similar(A, n, n)
    mul!(temp, Eig.vectors, Diagonal(eig1))
    mul!(Sin, temp, Eig.vectors')
    mul!(temp, Eig.vectors, Diagonal(eig2))
    mul!(temp2, temp, Eig.vectors')
    Cos .= Sin
    Cos .+= temp2
    Cos ./= 2
    Sin .-= temp2
    Sin .*= -1im/2
    return Sin, Hermitian(Cos) 
end

function Base.tan(A::SkewHermitian{<:Real})
    E=cis(A)
    C=real(A)
    S=imag(A)
    return C\S
end

function Base.tan(A::SkewHermitian{<:Complex})
    Sin, Cos =skewsincos!(copyeigtype(A))
    return Cos\Sin
end

Base.sinh(A::SkewHermitian) = skewhermitian!(exp(A))
Base.cosh(A::SkewHermitian{<:Real}) = hermitian!(exp(A))
@views function Base.cosh(A::SkewHermitian{<:Complex}) 
    B = hermitian!(exp(A))
    Cosh=complex.(real(B),-imag(B))
    return Cosh
end

function Base.tanh(A::SkewHermitian{<:Real})
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

