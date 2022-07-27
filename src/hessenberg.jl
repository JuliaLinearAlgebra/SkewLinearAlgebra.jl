# This file is a part of Julia. License is MIT: https://julialang.org/license
struct SkewHessenberg{SH<:Union{LA.Tridiagonal,LA.UpperHessenberg},S<:LA.UnitLowerTriangular,tau<:AbstractVector}
    H::SH # Tridiagonal
    V::S # reflector data in unit lower triangular
    τ::tau # more Q (reflector) data
end

SkewHessenberg(F::SkewHessenberg) = F

Base.copy(F::SkewHessenberg{<:Any,<:LA.UpperHessenberg}) = SkewHessenberg(copy(F.H),copy(F.V), copy(F.τ))
Base.copy(F::SkewHessenberg{<:Any,<:LA.Tridiagonal}) = SkewHessenberg(copy(F.H),copy(F.V), copy(F.τ))
LA.size(F::SkewHessenberg, d) = LA.size(F.H, d)
LA.size(F::SkewHessenberg) = LA.size(F.H)

@views function LA.hessenberg!(A::SkewSymmetric{<:LA.BlasFloat})
    tau,E = sktrd!(A)
    n=size(A,1)
    return SkewHessenberg(LA.Tridiagonal(E,zeros(n),-E),LA.UnitLowerTriangular(A.data[2:end,1:end-1]),tau)
end

function LA.hessenberg(A::SkewSymmetric{<:LA.BlasFloat})
    return LA.hessenberg!(copy(A))
end

Base.display(F::SkewHessenberg) = display(F.H)

@views function getQ(H::SkewHessenberg)
    n=size(H.H,1)
    Q = LA.Matrix(LA.diagm(ones(n)))
    s=similar(Q,n)
    for i=1:n-2
        t=H.τ[n-i-1]
        leftHouseholder!(Q[n-i:n,n-i-1:n],LA.Array(H.V[n-i-1:n-1,n-i-1]),s[n-i-1:n],t)
    end
    return Q
end
@views function householder_reflector!(x,v,n)
    v[2:n]=x[2:n]
    v .*= 1/(x[1]+sign(x[1])*LA.norm(x))
    v[1]=1
    tau = 2/((LA.norm(v)^2))
    return v,tau
end

@views function leftHouseholder!(A::AbstractMatrix,v::AbstractArray,s::AbstractArray,tau::Number)
    LA.BLAS.gemv!('T',1.0,A,v,0.0,s)
    LA.BLAS.ger!(-tau,v,s,A)
    return
end

@views function rightHouseholder!(A::AbstractMatrix,v::AbstractVector,s::AbstractVector,tau::Number)
    LA.BLAS.gemv!('N',1.0,A,v,0.0,s)
    LA.BLAS.ger!(-tau,s,v,A)
    return 
end

@views function skewhess!(A::AbstractMatrix,tau::AbstractVector,E::AbstractVector)
    n=LA.size(A,1)
    atmp = LA.similar(A,n)
    vtmp = LA.similar(atmp)
    for i=1:n-2
        v,stau = householder_reflector!(A[i+1:end,i], vtmp[i+1:end],n-i)

        A[i+1,i] -= stau*v[1]*LA.dot(v,A[i+1:end,i])
        E[i] = A[i+1,i]
        tau[i]=stau
        for j=i+2:n
            A[j,i]=v[j-i]
        end
        A[i,i+1]=-A[i+1,i]
        for j=i+2:n
            A[i,j]=0#Purely esthetic
        end

        leftHouseholder!(A[i+1:end,i+1:end],v,atmp[i+1:end],stau)

        s = LA.mul!(atmp[i+1:end], A[i+1:end,i+1:end], v)
        A[i+1,i+1]=0
        for j=i+2:n
            @inbounds A[j,j]=0
            @inbounds for k=i+1:j-1
                A[j,k] -= stau*s[j-i]*v[k-i]
                A[k,j]  = -A[j,k]
            end
        end
    end
    return
end
@views function latrd!(A::AbstractMatrix,E::AbstractVector,W::AbstractMatrix,V::AbstractVector,tau::AbstractVector,n::Number,nb::Number)

    @inbounds for i=1:nb
        #update A[i:n,i]

        if i>1
            LA.BLAS.gemv!('n',1.0,A[i:n,1:i-1],W[i,1:i-1],1.0,A[i:n,i])
            LA.BLAS.gemv!('n',-1.0,W[i:n,1:i-1],A[i,1:i-1],1.0,A[i:n,i])
        end
        
        #Generate elementary reflector H(i) to annihilate A(i+2:n,i)
        
        v,stau = householder_reflector!(A[i+1:n,i],V[i:n-1],n-i)
        A[i+1,i] -= stau*v[1]*LA.dot(v,A[i+1:end,i])
        A[i+2:end,i] = v[2:end]
        tau[i] = stau
        E[i]   = A[i+1,i]
        A[i+1,i] = 1

        #Compute W[i+1:n,i]
        LA.mul!(W[i+1:n,i],A[i+1:n,i+1:n], A[i+1:n,i])  #Key point 60% of running time of sktrd!
        if i>1
            LA.mul!(W[1:i-1,i],LA.transpose(W[i+1:n,1:i-1]),A[i+1:n,i])
            LA.BLAS.gemv!('n',1.0,A[i+1:n,1:i-1],W[1:i-1,i],1.0,W[i+1:n,i])
            LA.mul!(W[1:i-1,i],LA.transpose(A[i+1:n,1:i-1]),A[i+1:n,i])
            LA.BLAS.gemv!('n',-1.0,W[i+1:n,1:i-1],W[1:i-1,i],1.0,W[i+1:n,i])
        end
        
        W[i+1:n,i].*=stau
        alpha=-0.5*stau*LA.dot(W[i+1:n,i],A[i+1:n,i])
        LA.axpy!(alpha,A[i+1:n,i],W[i+1:n,i])
        
    end
    return 
end
function set_nb(n::Number)
    if n<=12
        return max(n-3,1)
    elseif n<=100
        return 10
    else
        return 40
    end
    return 1
end

@views function sktrd!(S::SkewSymmetric)
    #println("begin\n")
    n = LA.size(S.data,1)

    if n==1
        return 0, S.data
    end

    nb = set_nb(n)

    A = S.data
    E = LA.similar(A,n-1)
    tau = LA.similar(A,n-2)
    W = LA.similar(A,n,nb)
    update = LA.similar(A,n-nb,n-nb)
    V = LA.similar(A,n-1)

    oldi=0
    
    @inbounds for i=1:nb:n-nb-2
        size=n-i+1

        latrd!(A[i:n,i:n],E[i:i+nb-1],W,V,tau[i:i+nb-1],size,nb)
        LA.mul!(update[1:n-nb-i+1,1:n-nb-i+1],A[i+nb:n,i:i+nb-1],LA.transpose(W[nb+1:size,:]))

        s=i+nb-1
        @inbounds for j=1:n-s
            for k=1:j-1
                A[s+j,s+k] += update[j,k]-update[k,j]
                A[s+k,s+j] =- A[s+j,s+k]
            end
            A[s+j,s+j]=0
        end
        @simd for j=i:i+nb-1
            @inbounds A[j+1,j]=E[j]
        end
        oldi=i
    end
    oldi+=nb
    if oldi<n
        skewhess!(A[oldi:n,oldi:n],tau[oldi:end],E[oldi:end])
    end
    E[end]=A[end,end-1]

    return tau, E
    
end

