LA.HessenbergQ(F::Hessenberg{<:Any,<:SkewHermTridiagonal,S,W}) where {S,W} = LA.HessenbergQ{eltype(F.factors),S,W,true}(F.uplo, F.factors, F.τ)

@views function LA.hessenberg!(A::SkewHermitian{T}) where {T}
    tau,E = skewblockedhess!(A)
    n = size(A,1)
    if T <: Complex 
        Tr=SkewHermTridiagonal(E,imag(diag(A.data)))
    else
        Tr=SkewHermTridiagonal(E)
    end
    return  Hessenberg{typeof(zero(eltype(A.data))),typeof(Tr),typeof(A.data),typeof(tau),typeof(false)}(Tr, 'L', A.data, tau, false)
end

LA.hessenberg(A::SkewHermitian)=hessenberg!(copyeigtype(A))


@views function householder!(x::AbstractVector{T},n::Integer) where {T}
    if n==1 && T <:Real
        return convert(eltype(x), 0), x[1]
    end

    xnorm = (n > 1 ? norm(x[2:end]) : zero(real(x[1])))
    alpha = x[1]

    if !iszero(xnorm) || n==1
        
        beta=(real(alpha) > 0 ? -1 : +1)*hypot(abs(alpha),xnorm)
        tau = 1-alpha/beta#complex((beta-alphar)/beta,-alphaim/beta)
        beta= convert(T,beta)
        alpha = 1/(alpha-beta)
        x[1] = convert(T,1)
        alpha= convert(T,alpha)
        
        if n>1
            @inbounds x[2:n].*=alpha
        end
        
        
        alpha=beta
        
    else
        tau = convert(eltype(x), 0)
        x = zeros(eltype(x),n)
        alpha = convert(eltype(x), 0)
    end
    
    return tau, alpha

end
@views function ger2!(tau::Number , v::StridedVector{T} , s::StridedVector{T},
    A::StridedMatrix{T}) where {T<:LA.BlasFloat}
    tau2 = promote(tau, zero(T))[1]

    if tau2 isa Union{Bool,T}
        return LA.BLAS.ger!(tau2, v, s, A)
    else
        m=length(v)
        n=length(s)
        @inbounds for j=1:n
            temp = tau2 * s[j]'
            @simd for i=1:m
                A[i,j] += v[i] * temp
            end
        end

    end
end

@views function lefthouseholder!(A::AbstractMatrix,v::AbstractArray,s::AbstractArray,tau::Number)
    mul!(s,adjoint(A),v)
    ger2!(-tau',v,s,A)
    return
end

@views function skewhess!(A::AbstractMatrix{T},tau::AbstractVector,E::AbstractVector) where {T}
    n = size(A,1)
    atmp = similar(A,n)
    @inbounds (for i=1:n-1
        stau,alpha = householder!(A[i+1:end,i],n-i)
        @views v = A[i+1:end,i]
        E[i] = alpha

        lefthouseholder!(A[i+1:end,i+1:end],v,atmp[i+1:end],stau)
        
        s = mul!(atmp[i+1:end], A[i+1:end,i+1:end], v)
        for j=i+1:n
            A[j,j] -= stau*s[j-i]*v[j-i]'
            for k=j+1:n
                A[k,j] -= stau*s[k-i]*v[j-i]'
                A[j,k] =-A[k,j]'
            end
        end
        tau[i] = stau
    end)
    return
end





@views function skewlatrd!(A::AbstractMatrix{T},E::AbstractVector,W::AbstractMatrix,tau::AbstractVector,tempconj::AbstractVector,n::Number,nb::Number) where {T}

    @inbounds(for i=1:nb
        if i>1
            if T <: Complex
                @simd for j=1:i-1
                    tempconj[j] = conj(W[i,j])
                end
                mul!(A[i:n,i],A[i:n,1:i-1],tempconj[1:i-1],1,1)
                @simd for j=1:i-1
                    tempconj[j] = conj(A[i,j])
                end
                mul!(A[i:n,i],W[i:n,1:i-1],tempconj[1:i-1],-1,1)
            else
                mul!(A[i:n,i],A[i:n,1:i-1],W[i,1:i-1],1,1)
                mul!(A[i:n,i],W[i:n,1:i-1],A[i,1:i-1],-1,1)
            end

            
            
        end

        #Generate elementary reflector H(i) to annihilate A(i+2:n,i)


        stau,alpha = householder!(A[i+1:n,i],n-i)
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
        
        if T<:Complex
            W[i+1:n,i].-= alpha.*A[i+1:n,i]
        else
            W[i+1:n,i].+= alpha.*A[i+1:n,i]
        end

        tau[i] = stau
        


    end)
    return
end
function setnb(n::Integer)
    if n<=12
        return max(n-4,1)
    elseif n<=100
        return 10
    else

        return 60

    end
    return 1
end

@views function skewblockedhess!(S::SkewHermitian{T}) where {T}
    
    n = size(S.data,1)
    
    if n == 1
        return Hessenberg(Matrix(S.data),Vector{eltype(S.data)}(undef,0),LA.UpperHessenberg(S.data),'L')
    end

    nb  = setnb(n)
    A   = S.data
    
    E   = similar(A,n-1)
    tau = similar(A,n-1)
    W   = similar(A, n, nb)
    update = similar(A, n-nb, n-nb)

    tempconj=similar(A,nb)


    oldi = 0

    @inbounds(for i = 1:nb:n-nb-1
        size = n-i+1


        skewlatrd!(A[i:n,i:n],E[i:i+nb-1],W,tau[i:i+nb-1],tempconj,size,nb)

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
        skewhess!(A[oldi:n,oldi:n],tau[oldi:end],E[oldi:end])
    end

    return tau, E

end

