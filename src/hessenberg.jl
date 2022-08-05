# Based on hessenberg.jl in Julia. License is MIT: https://julialang.org/license

"""
    SkewHessenberg

Type returned by hessenberg(A) when A is skew-symmetric.
Contains the tridiagonal reduction in H, the reflectors in V that is UnitLowerTriangular.
τ contains the scalars relative to the householder reflector i.e Q_i= I - τ_i v_i v_i^T.
"""
struct SkewHessenberg{SH<:Tridiagonal,S<:UnitLowerTriangular,tau<:AbstractVector}
    H::SH # Tridiagonal
    V::S # reflector data in unit lower triangular
    τ::tau # more Q (reflector) data
end

SkewHessenberg(F::SkewHessenberg) = F
Base.copy(F::SkewHessenberg) = SkewHessenberg(copy(F.H), copy(F.V), copy(F.τ))
Base.size(F::SkewHessenberg, d) = size(F.H, d)
Base.size(F::SkewHessenberg) = size(F.H)

@views function LA.hessenberg!(A::SkewHermitian)
    tau,E = sktrd!(A)
    n = size(A,1)
    return SkewHessenberg(Tridiagonal(E,zeros(n),-E),UnitLowerTriangular(A.data[2:end,1:end-1]),tau)
end

function Base.show(io::IO, mime::MIME"text/plain", F::SkewHessenberg)
    summary(io, F)
    println(io, "\nV factor:")
    show(io, mime, F.V)
    println(io, "\nH factor:")
    show(io, mime, F.H)
end

# fixme: to get the Q factor, F.Q should return a type that
# implicitly multiplies by reflectors, and you should convert it
# to a matrix with Matrix(F.Q), eliminating getQ.

"""
    getQ(H)

Allows to reconstruct orthogonal matrix Q from reflectors V.
Returns Q.
"""
@views function getQ(H::SkewHessenberg)
    n = size(H.H,1)
    Q = diagm(ones(n))
    s = similar(Q,n)
    for i = 1:n-2
        t = H.τ[n-i-1]
        leftHouseholder!(Q[n-i:n,n-i-1:n],Array(H.V[n-i-1:n-1,n-i-1]),s[n-i-1:n],t)
    end
    return Q
end

@views function householder_reflector!(x,v,n)
    div=1/(x[1]+sign(x[1])*norm(x))
    v[1] = 1
    """
    @simd for j=2:n
        @inbounds v[j] = x[j]*div
    end
    """
    @inbounds v[2:end]=x[2:end]
    v[2:end].*=div
    tau = 2/((norm(v)^2))
    return v,tau
end
@inline @views function ger2!(tau::Number , v::StridedVector{T} , s::StridedVector{T},
    A::StridedMatrix{T}) where {T<:LA.BlasFloat}
    tau2 = promote(tau, zero(T))[1]
    if tau2 isa Union{Bool,T}
        return LA.BLAS.ger!(tau2, v, s, A)
    else
        m=length(v)
        n=length(s)
        @inbounds for j=1:n
            temp=tau2*s[j]
            @simd for i=1:m
                A[i,j] += v[i]*temp
            end
        end

    end
end

@views function leftHouseholder!(A::AbstractMatrix,v::AbstractArray,s::AbstractArray,tau::Number)
    mul!(s,transpose(A),v)
    ger2!(-tau,v,s,A)
    return
end

@views function skewhess!(A::AbstractMatrix,tau::AbstractVector,E::AbstractVector)
    n = size(A,1)
    atmp = similar(A,n)
    vtmp = similar(atmp)
    @inbounds (for i=1:n-2
        v,stau = householder_reflector!(A[i+1:end,i], vtmp[i+1:end],n-i)

        A[i+1,i] -= stau*dot(v,A[i+1:end,i])
        E[i] = A[i+1,i]
        A[i+1:end,i]=v
        leftHouseholder!(A[i+1:end,i+1:end],v,atmp[i+1:end],stau)

        s = mul!(atmp[i+1:end], A[i+1:end,i+1:end], v)
        A[i+1,i+1]=  0
        for j=i+2:n
            A[j,j]=0
            @simd for k=i+1:j-1
                A[j,k] -= stau*s[j-i]*v[k-i]
                A[k,j]  = -A[j,k]
            end
        end
        tau[i] = stau
    end)
    return
end
@views function skmv!(A::AbstractMatrix,x::AbstractVector,y::AbstractVector,n::Integer)
    @simd for j=1:n
        @inbounds y[j]=0
    end
    """
    nb=60
    oldk=0
    for j=1:n
        temp = x[j]
        for k=j+1:nb:n-nb
            k2=k+nb-1
            @inbounds axpy!(temp,A[k:k2,j],y[k:k2])
            @inbounds y[j] -= dot(A[k:k2,j],x[k:k2])
            oldk=k
        end
        oldk+=nb
        if oldk<n
            @inbounds axpy!(temp,A[oldk:n,j],y[oldk:n])
            @inbounds y[j] -= dot(A[oldk:n,j],x[oldk:n])
        end
    end
    """

    for j=1:n
        @inbounds axpy!(x[j],A[j+1:n,j],y[j+1:n])
        @inbounds y[j] -= dot(A[j+1:n,j],x[j+1:n])
    end

    """
    nb=10
    @inbounds for j=1:n
        temp1=x[j]
        temp2=0
        oldi=0
        for i=j+1:nb:n-nb
            @simd for k=0:nb-1
                i2=i+k
                @inbounds y[i2] += temp1*A[i2,j]
                @inbounds temp2 += A[i2,j]*x[i2]
            end
            old=i
        end
        oldi+=nb
        if oldi<n
            @simd for i=oldi:n
                @inbounds y[i] += temp1*A[i,j]
                @inbounds temp2 += A[i,j]*x[i]
            end
        end
        y[j] -= temp2
    end
    """
end
@views function gemv2!(A::AbstractMatrix,x::AbstractVector,y::AbstractVector,n::Integer)
    @simd for j=1:n
        @inbounds y[j]=0
    end
    @inbounds(for j=1:n
        temp1=x[j]
        @simd for i=1:n
            y[i] += temp1*A[i,j]
        end
    end)
end
@views function latrd!(A::AbstractMatrix,E::AbstractVector,W::AbstractMatrix,V::AbstractVector,tau::AbstractVector,n::Number,nb::Number)

    @inbounds(for i=1:nb
        #update A[i:n,i]

        if i>1
            mul!(A[i:n,i],A[i:n,1:i-1],W[i,1:i-1],1,1)
            mul!(A[i:n,i],W[i:n,1:i-1],A[i,1:i-1],-1,1)
        end

        #Generate elementary reflector H(i) to annihilate A(i+2:n,i)

        v,stau = householder_reflector!(A[i+1:n,i],V[i:n-1],n-i)
        A[i+1,i] -= stau*dot(v,A[i+1:n,i])
        E[i]   = A[i+1,i]
        A[i+1:end,i] = v

        mul!(W[i+1:n,i],A[i+1:n,i+1:n], A[i+1:n,i])  #Key point 60% of running time of sktrd!
        #skmv!(A[i+1:n,i+1:n], A[i+1:n,i],W[i+1:n,i],n-i)
        if i>1
            mul!(W[1:i-1,i],transpose(W[i+1:n,1:i-1]),A[i+1:n,i])
            mul!(W[i+1:n,i],A[i+1:n,1:i-1],W[1:i-1,i],1,1)
            mul!(W[1:i-1,i],transpose(A[i+1:n,1:i-1]),A[i+1:n,i])
            mul!(W[i+1:n,i],W[i+1:n,1:i-1],W[1:i-1,i],-1,1)
        end
        W[i+1:n,i] .*= stau

        alpha = -stau*dot(W[i+1:n,i],A[i+1:n,i])/2
        W[i+1:n,i].+=alpha.*A[i+1:n,i]
        tau[i] = stau


    end)
    return
end
function set_nb(n::Integer)
    if n<=12
        return max(n-4,1)
    elseif n<=100
        return 10
    else
        return 60
    end
    return 1
end

@views function sktrd!(S::SkewHermitian{<:Real})
    #println("begin\n")
    n = size(S.data,1)

    if n == 1
        return 0, S.data
    end

    nb  = set_nb(n)
    A   = S.data

    E   = similar(A,n-1)
    tau = similar(A,n-2)
    W   = similar(A, n, nb)
    update = similar(A, n-nb, n-nb)
    V   = similar(A, n-1)

    oldi = 0

    @inbounds(for i = 1:nb:n-nb-2
        size = n-i+1

        latrd!(A[i:n,i:n],E[i:i+nb-1],W,V,tau[i:i+nb-1],size,nb)
        mul!(update[1:n-nb-i+1,1:n-nb-i+1],A[i+nb:n,i:i+nb-1],transpose(W[nb+1:size,:]))

        s = i+nb-1
        """
        for j = 1:n-s
            @simd for k = 1:j-1
                @inbounds A[s+j,s+k] += update[j,k]-update[k,j]
                @inbounds A[s+k,s+j] = - A[s+j,s+k]
            end
            @inbounds A[s+j,s+j] = 0
        end
        """

        for k = 1:n-s
            A[s+k,s+k] = 0
            @simd for j = k+1:n-s
                A[s+j,s+k] += update[j,k]-update[k,j]
                A[s+k,s+j] = - A[s+j,s+k]
            end

        end

        """
        N=n-nb-i+1
        @inbounds (for j=1:N
            k1=s+j
            A[s+j,s+j]=0
            for l=1:nb
                k2=i-1+l
                temp1 = W[nb+j,l]
                temp2 = A[k1,k2]
                @simd for t=j+1:N
                    A[s+t,k1] += A[s+t,k2]*temp1-W[nb+t,l]*temp2
                end
            end

            @simd for t=j+1:N
                A[k1,s+t]=-A[s+t,k1]
            end
        end)
        """
        """
        A[s+1:n,s+1:n].+= update[1:n-s,1:n-s]
        A[s+1:n,s+1:n].-= transpose(update[1:n-s,1:n-s])
        """
        oldi = i
    end)
    oldi += nb
    if oldi < n
        skewhess!(A[oldi:n,oldi:n],tau[oldi:end],E[oldi:end])
    end
    E[end] = A[end,end-1]

    return tau, E

end

