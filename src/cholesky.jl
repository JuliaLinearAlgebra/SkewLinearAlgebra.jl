
@views function skewchol!(A::SkewHermitian)
    B=A.data
    tol=1e-15
    m=size(B,1)
    J2=[0 1;-1 0]
    ii=0; jj=0; kk=0
    P=Array(1:m)
    temp=similar(B,m)
    tempM=similar(B,2,m-2)
    for j=1:m÷2
        j2=2*j
        M=maximum(B[j2-1:m,j2-1:m])
        for i1=j2-1:m
            for i2=j2-1:m
                if B[i1,i2] == M
                    ii = i1
                    jj = i2
                end
            end
        end
        if abs(B[ii,jj])<tol
            rank = j2-2
            return P
        end
        if jj==j2-1
            kk=ii
        else
            kk=jj
        end
        if ii!=j2-1

            I = Array(1:m)
            I[ii] = j2-1
            I[j2-1] = ii
            
            temp2 = P[ii]
            P[ii] = P[j2-1]
            P[j2-1] = temp2
            
            Base.permutecols!!(B,I)
            temp .= B[ii,:]
            B[ii,:] .= B[j2-1,:]
            B[j2-1,:] .= temp
        end
        if kk!= j2

            I=Array(1:m)
            I[kk]=j2
            I[j2]=kk
            
            temp3 = P[kk]
            P[kk] = P[j2]
            P[j2] = temp3
            
            Base.permutecols!!(B,I)
            temp .= B[kk,:]
            B[kk,:] .= B[j2,:]
            B[j2,:] .= temp
        end
        
        l=m-j2
        r = sqrt(B[j2-1,j2])
        B[j2-1,j2-1] = r
        B[j2,j2] = r
        B[j2-1,j2] = 0
        mul!(tempM[:,1:l],J2,B[j2-1:j2,j2+1:m])
        B[j2-1:j2,j2+1:m] .= tempM[:,1:l]
        B[j2-1:j2,j2+1:m] .*= (-1/r)
        mul!(tempM[:,1:l],J2,B[j2-1:j2,j2+1:m])
        mul!(B[j2+1:m,j2+1:m],transpose(B[j2-1:j2,j2+1:m]),tempM[:,1:l],-1,1)
    end
    r=2*(m÷2)
    return P
end

@views function skewsolve(A::SkewHermitian,b::AbstractVector)
    
    n = size(A,1)
    P = skewchol!(A)
    y1 = similar(A.data,n)
    y2 = similar(A.data,n)
    R = UpperTriangular(A.data)
    vec=zeros(n-1)
    for i=1:2:n-1
        vec[i]=1
    end
    Jt = Tridiagonal(vec,zeros(n),-vec)
    Base.permute!(b,P)
    y1 .= transpose(R)\b
    mul!(y2,Jt,y1)
    y1 .= R\y2
    Base.permute!(y1,P)
    return y1
end

"""
Q^TR^TJRQx=b
=>RQx=J^T R^(-T) Q^Tb
"""
