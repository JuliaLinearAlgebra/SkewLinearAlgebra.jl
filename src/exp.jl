# This file is a part of Julia. License is MIT: https://julialang.org/license

@views function skewexpm(A::SkewSymmetric)
    n=LA.size(A.data,1)
    if n==1
        return exp(A.data)
    end

    vals,Qr,Qim = skeweigen!(A)
    
    temp1=LA.similar(A.data,n,n)
    temp2=LA.similar(A.data,n,n)
    QrS = copy(Qr)
    QrC = copy(Qr)
    QimS = copy(Qim)
    QimC = copy(Qim)
    for i=1:n
        c=cos(imag(vals[i]))
        s=sin(imag(vals[i]))
        QrS[:,i].*=s
        QimS[:,i].*=s
        QrC[:,i].*=c
        QimC[:,i].*=c
    end
    LA.mul!(temp1,QrC-QimS,LA.transpose(Qr))
    LA.mul!(temp2,QrS+QimC,LA.transpose(Qim))
    return temp1+temp2
end

@views function LA.exp!(A::SkewSymmetric)
    return skewexpm(A)
end
function LA.exp(A::SkewSymmetric)
    return skewexpm(copy(A))
end