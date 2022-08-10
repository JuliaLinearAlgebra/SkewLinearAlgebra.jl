
@views function LA.hessenberg!(A::SkewHermitian{<:Complex})
    tau,E = complexsktrd!(A)
    n = size(A,1)
    T=SkewHermTridiagonal(E,imag(diag(A.data)))
    return  Hessenberg{typeof(zero(eltype(A.data))),typeof(T),typeof(A.data),typeof(tau),typeof(false)}(T, 'L', A.data, tau, false)
end


@views function complex_householder_reflector!(x,v,n)
    if n>1
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
    else
        alpha=x[1]
        alphar=real(alpha)
        alphaim=imag(alpha)
        if alphar>0
            beta=-sqrt(alphar*alphar+alphaim*alphaim)
        else
            beta=sqrt(alphar*alphar+alphaim*alphaim)
        end
        tau=(beta-alphar)/beta-alphaim/beta*1im
        alpha=1/(alpha-beta)
        v[1]=1
        alpha=beta
    end
    return v,tau, alpha
end

@views function complexleftHouseholder!(A::AbstractMatrix,v::AbstractArray,s::AbstractArray,tau::Number)
    mul!(s,adjoint(A),v)
    ger2!(-tau,v,s,A)
    return
end

@views function complexskewhess!(A::AbstractMatrix,tau::AbstractVector,E::AbstractVector)
    n = size(A,1)
    atmp = similar(A,n)
    vtmp = similar(atmp)
    @inbounds (for i=1:n-1
        v,stau,alpha = complex_householder_reflector!(A[i+1:end,i], vtmp[i+1:end],n-i)

        E[i] = alpha
        A[i+1:end,i]=v
        #leftHouseholder!(A[i+1:end,i+1:end],v,atmp[i+1:end],stau)
        q = mul!(atmp[i+1:end],adjoint(A[i+1:end,i+1:end]),v)
        for j=i+1:n
            for k=i+1:n
                A[k,j] -= stau'*q[j-i]'*v[k-i]
            end
        end

        s = mul!(atmp[i+1:end], A[i+1:end,i+1:end], v)
        for j=i+1:n
            for k=i+1:n
                A[k,j] -= stau*s[k-i]*v[j-i]'
            end
        end
        tau[i] = stau
    end)
    return
end




@views function complexlatrd!(A::AbstractMatrix,E::AbstractVector,W::AbstractMatrix,V::AbstractVector,tau::AbstractVector,n::Number,nb::Number)

    @inbounds(for i=1:nb
        if i>1
            #display(W[i,1:i-1])
            #display(transpose(W[i,1:i-1]'))
            A[i:n,i].+=A[i:n,1:i-1]*transpose(adjoint(W[i,1:i-1]))
            A[i:n,i].-=W[i:n,1:i-1]*transpose(adjoint(A[i,1:i-1]))
            """
            mul!(A[i:n,i],A[i:n,1:i-1],adjoint(W[i,1:i-1]),1,1)
            mul!(A[i:n,i],W[i:n,1:i-1],adjoint(A[i,1:i-1]),-1,1)
            """
            
        end

        #Generate elementary reflector H(i) to annihilate A(i+2:n,i)

        v,stau,alpha = complex_householder_reflector!(A[i+1:n,i],V[i:n-1],n-i)
        #A[i+1,i] -= stau*dot(adjoint.(v),A[i+1:n,i])
        E[i]   = real(alpha)
        A[i+1:n,i] = v
        #display(E)
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
        W[i+1:n,i].-= alpha.*A[i+1:n,i]
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
    tau = similar(A,n-1)
    W   = similar(A, n, nb)
    update = similar(A, n-nb, n-nb)
    V   = similar(A, n-1)

    oldi = 0

    @inbounds(for i = 1:nb:n-nb-1
        size = n-i+1

        complexlatrd!(A[i:n,i:n],E[i:i+nb-1],W,V,tau[i:i+nb-1],size,nb)
        mul!(update[1:n-nb-i+1,1:n-nb-i+1],A[i+nb:n,i:i+nb-1],adjoint(W[nb+1:size,:]))

        s = i+nb-1

        for k = 1:n-s
            A[s+k,s+k] += update[k,k]-update[k,k]'
            @simd for j = k+1:n-s
                A[s+j,s+k] += update[j,k]-update[k,j]'
                A[s+k,s+j] = - A[s+j,s+k]'
            end

        end
        oldi = i
    end)
    oldi += nb
    if oldi < n
        complexskewhess!(A[oldi:n,oldi:n],tau[oldi:end],E[oldi:end])
    end

    return tau, E

end

