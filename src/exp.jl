# This file is a part of Julia. License is MIT: https://julialang.org/license
"""
@views function skewexpm(A::SkewSymmetric)
    n = size(A.data,1)
    if n == 1
        return exp(A.data)
    end

    vals,Qr,Qim = skeweigen!(A)
    temp2 = similar(A.data,n,n)
    QrS = copy(Qr)
    QrC = copy(Qr)
    QimS = copy(Qim)
    QimC = copy(Qim)
    for i=1:n
        c = cos(imag(vals[i]))
        s = sin(imag(vals[i]))
        QrS[:,i] .*=s
        QimS[:,i] .*=s
        QrC[:,i] .*=c
        QimC[:,i] .*=c
    end
    mul!(temp2,QrC-QimS,transpose(Qr))
    mul!(A.data,QrS+QimC,transpose(Qim))
    temp2 += A.data
    return temp2
end

"""
@views function skewexpm(A::SkewSymmetric)
    n = size(A,1)
    if n == 1
        return exp(A.data)
    end
    vals,Qr,Qim = skeweigen!(A)

    temp2 = similar(A,n,n)
    Q1=similar(A,n,n)
    Q2=similar(A,n,n)
    Cos=similar(A,n)
    Sin=similar(A,n)

    @simd for i=1:n
        @inbounds Cos[i]=cos(imag(vals[i]))
        @inbounds Sin[i]=sin(imag(vals[i]))
    end
    C=Diagonal(Cos)
    S=Diagonal(Sin)
    
    mul!(Q1,Qr,C)
    mul!(Q2,Qim,S)
    Q1 .-= Q2
    mul!(temp2,Q1,transpose(Qr))
    mul!(Q1,Qr,S)
    mul!(Q2,Qim,C)
    Q1 .+= Q2
    mul!(Q2,Q1,transpose(Qim))
    temp2 .+= Q2
    return temp2
end


"""
    exp(A)
Returns the matrix exponential of A skew-symmetric using the eigenvalue decomposition.
"""
@views function LA.exp(A::SkewSymmetric)
    return skewexpm(copy(A))
end