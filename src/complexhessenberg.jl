
@views function LA.hessenberg!(A::SkewHermitian)
    tau,E = sktrd!(A)
    n = size(A,1)
    tau2=similar(tau,n-1)
    tau2[1:n-2].=tau
    tau2[n-1]=0  
    T=SkewHermTridiagonal(E)
    return  Hessenberg{typeof(zero(eltype(A.data))),typeof(T),typeof(A.data),typeof(tau2),typeof(false)}(T, 'L', A.data, tau2, false)
end


@views function complex_householder_reflector!(x,v,n)
    xnorm=norm(x[2:end])
    alpha=x[1]
    alphar=real(alpha)
    alphaim=imag(alpha)
    if xnorm > 1e-15
        if alphar>0
            beta=-sqrt(alphar*alphar+alphaim*alphaim+xnorm*xnorm)
        else
            beta=sqrt(alphar*alphar+alphaim*alphaim+xnorm*xnorm)
        end
        tau=(beta-alphar)/beta-alphaim/beta*1im
        alpha=1/(alpha-beta)
        v[1]=1
        @inbounds v[2:end]=x[2:end]
        v[2:end] .*= alpha
        alpha=beta
    else
        tau = convert(eltype(x),0)
        v = zeros(eltype(x),n)
        alpha=0
    end
    return v,tau, alpha
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

@views function complexleftHouseholder!(A::AbstractMatrix,v::AbstractArray,s::AbstractArray,tau::Number)
    mul!(s,transpose(A),v')
    ger2!(-tau,v,adjoint.(s),A)
    return
end

@views function complexskewhess!(A::AbstractMatrix,tau::AbstractVector,E::AbstractVector)
    n = size(A,1)
    atmp = similar(A,n)
    vtmp = similar(atmp)
    @inbounds (for i=1:n-2
        v,stau,alpha = complex_householder_reflector!(A[i+1:end,i], vtmp[i+1:end],n-i)

        #A[i+1,i] -= stau*dot(adjoint.(v),A[i+1:end,i])
        E[i] = alpha
        A[i+1:end,i]=v
        leftHouseholder!(A[i+1:end,i+1:end],v,atmp[i+1:end],stau)

        s = mul!(atmp[i+1:end], A[i+1:end,i+1:end], v)
        A[i+1,i+1]=  0
        for j=i+2:n
            A[j,j]=0
            @simd for k=i+1:j-1
                A[j,k] -= stau*s[j-i]*v[k-i]'
                A[k,j]  = -A[j,k]
            end
        end
        tau[i] = stau
    end)
    return
end




@views function complexlatrd!(A::AbstractMatrix,E::AbstractVector,W::AbstractMatrix,V::AbstractVector,tau::AbstractVector,n::Number,nb::Number)

    @inbounds(for i=1:nb
        #update A[i:n,i]

        if i>1
            display(A[i:n,1:i-1])
            display(adjoint(W[i,1:i-1]))

            #A[i,i]=imag(A[i,i])*1im
            #
            if i==2
                A[i:n,i].+=A[i:n,1:i-1].*(W[i,1:i-1]')
                A[i:n,i].+=W[i:n,1:i-1].*(A[i,1:i-1]')
            else
                mul!(A[i:n,i],A[i:n,1:i-1],adjoint(W[i,1:i-1]),1,1)
            #
                mul!(A[i:n,i],W[i:n,1:i-1],adjoint(A[i,1:i-1]),-1,1)
            end
            #A[i,i] = imag(A[i,i])*1im
        end

        #Generate elementary reflector H(i) to annihilate A(i+2:n,i)

        v,stau,alpha = complex_householder_reflector!(A[i+1:n,i],V[i:n-1],n-i)
        #A[i+1,i] -= stau*dot(adjoint.(v),A[i+1:n,i])
        E[i]   = real(alpha)
        A[i+1:n,i] = v
        display(E)
        mul!(W[i+1:n,i],A[i+1:n,i+1:n], A[i+1:n,i],1,0)  #Key point 60% of running time of sktrd!
        #skmv!(A[i+1:n,i+1:n], A[i+1:n,i],W[i+1:n,i],n-i)
        if i>1
            mul!(W[1:i-1,i],adjoint(W[i+1:n,1:i-1]),A[i+1:n,i])
            mul!(W[i+1:n,i],A[i+1:n,1:i-1],W[1:i-1,i],1,1)
            mul!(W[1:i-1,i],adjoint(A[i+1:n,1:i-1]),A[i+1:n,i])
            mul!(W[i+1:n,i],W[i+1:n,1:i-1],W[1:i-1,i],-1,1)
        end
        W[i+1:n,i] .*= stau

        alpha = -stau*dot(W[i+1:n,i],A[i+1:n,i])/2
        W[i+1:n,i].+= alpha.*A[i+1:n,i]
        tau[i] = stau


    end)
    return
end

@views function complexsktrd!(S::SkewHermitian{<:Complex})
    n = size(S.data,1)

    if n == 1
        return Hessenberg(Matrix(S.data),Vector{eltype(S.data)}(undef,0),LA.UpperHessenberg(S.data),'L')
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

        complexlatrd!(A[i:n,i:n],E[i:i+nb-1],W,V,tau[i:i+nb-1],size,nb)
        mul!(update[1:n-nb-i+1,1:n-nb-i+1],A[i+nb:n,i:i+nb-1],adjoint(W[nb+1:size,:]))

        s = i+nb-1

        for k = 1:n-s
            A[s+k,s+k] = 0
            @simd for j = k+1:n-s
                A[s+j,s+k] += update[j,k]-update[k,j]'
                A[s+k,s+j] = - A[s+j,s+k]
            end

        end
        oldi = i
    end)
    oldi += nb
    if oldi < n
        complexskewhess!(A[oldi:n,oldi:n],tau[oldi:end],E[oldi:end])
    end
    E[end] = A[end,end-1]

    return tau, E

end
