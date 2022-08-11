
@views function LA.hessenberg!(A::SkewHermitian{<:Complex})
    tau,E = complexsktrd!(A)
    n = size(A,1)
    T=SkewHermTridiagonal(E,imag(diag(A.data)))
    return  Hessenberg{typeof(zero(eltype(A.data))),typeof(T),typeof(A.data),typeof(tau),typeof(false)}(T, 'L', A.data, tau, false)
end
LA.hessenberg(A::SkewHermitian{<:Complex})=hessenberg!(copy(A))


@views function complex_householder_reflector!(x,n)
    
    if n>1
        xnorm = norm(x[2:end])
    else
        xnorm = zero(real.(x[1]))
    end
    T=eltype(x)
    alpha=x[1]
    alphar=real.(alpha)
    alphaim=imag.(alpha)
    if xnorm > 1e-15 || n==1
        
        if alphar>0
            beta=-sqrt(alphar*alphar+alphaim*alphaim+xnorm*xnorm)
        else
            beta=sqrt(alphar*alphar+alphaim*alphaim+xnorm*xnorm)
        end
        tau = complex.((beta-alphar)/beta,-alphaim/beta)
        beta= convert(T,beta)
        alpha = 1/(alpha-beta)
        x[1] = convert(T,1)
        alpha=convert(T,alpha)
        
        if n>1
            @inbounds x[2:n].*=alpha
        end
        
        
        alpha=beta
        
    else
        tau = convert(eltype(x),0)
        x = zeros(eltype(x),n)
        alpha=convert(eltype(x),0)
    end
    
    return tau, alpha

end
@views function cger2!(tau::Number , v::StridedVector{T} , s::StridedVector{T},
    A::StridedMatrix{T}) where {T<:LA.BlasFloat}
    tau2 = promote(tau, zero(T))[1]

    if tau2 isa Union{Bool,T}
        return LA.BLAS.ger!(tau2, v, s, A)
    else
        m=length(v)
        n=length(s)
        @inbounds for j=1:n
            temp=tau2*s[j]'
            @simd for i=1:m
                A[i,j] += v[i]*temp
            end
        end

    end
end

@views function complexleftHouseholder!(A::AbstractMatrix,v::AbstractArray,s::AbstractArray,tau::Number)
    mul!(s,adjoint(A),v)
    cger2!(-tau',v,s,A)
    return
end

@views function complexskewhess!(A::AbstractMatrix,tau::AbstractVector,E::AbstractVector)
    n = size(A,1)
    atmp = similar(A,n)
    @inbounds (for i=1:n-1
        stau,alpha = complex_householder_reflector!(A[i+1:end,i],n-i)
        @views v=A[i+1:end,i]
        E[i] = alpha

        complexleftHouseholder!(A[i+1:end,i+1:end],v,atmp[i+1:end],stau)
        
        s = mul!(atmp[i+1:end], A[i+1:end,i+1:end], v)
        for j=i+1:n
            A[j,j] -= stau*s[j-i]*v[j-i]'
            for k=j+1:n
                A[k,j] -= stau*s[k-i]*v[j-i]'
                A[j,k]=-A[k,j]'
            end
        end
        tau[i] = stau
    end)
    return
end





@views function complexlatrd!(A::AbstractMatrix,E::AbstractVector,W::AbstractMatrix,tau::AbstractVector,tempconj::AbstractVector,n::Number,nb::Number)

    @inbounds(for i=1:nb
        if i>1
            @simd for j=1:i-1
                tempconj[j] = conj.(W[i,j])
            end
            mul!(A[i:n,i],A[i:n,1:i-1],tempconj[1:i-1],1,1)
            @simd for j=1:i-1
                tempconj[j] = conj.(A[i,j])
            end
            mul!(A[i:n,i],W[i:n,1:i-1],tempconj[1:i-1],-1,1)

            
            
        end

        #Generate elementary reflector H(i) to annihilate A(i+2:n,i)


        stau,alpha = complex_householder_reflector!(A[i+1:n,i],n-i)
        E[i]   = real(alpha)

        
        mul!(W[i+1:n,i],A[i+1:n,i+1:n], A[i+1:n,i],1,0)  
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
function set_nb2(n::Integer)
    if n<=12
        return max(n-4,1)
    elseif n<=100
        return 10
    else

        return 60

    end
    return 1
end

@views function complexsktrd!(S::SkewHermitian{<:Complex})
    
    n = size(S.data,1)

    if n == 1
        return Hessenberg(Matrix(S.data),Vector{eltype(S.data)}(undef,0),LA.UpperHessenberg(S.data),'L')
    end

    nb  = set_nb2(n)
    A   = S.data
    
    E   = similar(A,n-1)
    tau = similar(A,n-1)
    W   = similar(A, n, nb)
    update = similar(A, n-nb, n-nb)

    tempconj=similar(A,nb)


    oldi = 0

    @inbounds(for i = 1:nb:n-nb-1
        size = n-i+1


        complexlatrd!(A[i:n,i:n],E[i:i+nb-1],W,tau[i:i+nb-1],tempconj,size,nb)

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

