# This file is a part of Julia. License is MIT: https://julialang.org/license

LA.HessenbergQ(F::Hessenberg{<:Any,<:SkewHermTridiagonal,S,W}) where {S,W} = LA.HessenbergQ{eltype(F.factors),S,W,true}(F.uplo, F.factors, F.τ)

@views function LA.hessenberg!(A::SkewHermitian{T}) where {T}
    τ, η = skewblockedhess!(A)
    if T <: Complex 
        Tr=SkewHermTridiagonal(convert.(T, η), imag( diag(A.data)))
    else
        Tr=SkewHermTridiagonal(η)
    end
    return  Hessenberg{typeof(zero(eltype(A.data))),typeof(Tr),typeof(A.data),typeof(τ),typeof(false)}(Tr, 'L', A.data, τ, false)
end

LA.hessenberg(A::SkewHermitian)=hessenberg!(copyeigtype(A))

"""
    householder!(x,n)

Takes `x::AbstractVector{T}` and its size `n` as input.
Computes the associated householder reflector  v overwitten in x.
The reflector matrix is H = I-τ * v * v'.
Returns τ as first output and β as second output where β 
is the first element of the output vector of H*x.
"""
@views function householder!(x::AbstractVector{T},n::Integer) where {T}
    if n == 1 && T <:Real
        return T(0), real(x[1]) # no final 1x1 reflection for the real case
    end
    xnorm = norm(x[2:end]) 
    α = x[1]
    if !iszero(xnorm) || (n == 1 && !iszero(α))
        β = (real(α) > 0 ? -1 : +1) * hypot(α,xnorm)
        τ = 1 - α / β
        α = 1 / (α - β)
        x[1] = 1
        x[2:n] .*= α
    else 
        τ = T(0)
        x .= 0
        β = real(T)(abs(α))
    end
    return τ, β
end

@views function ger2!(τ::Number , v::StridedVector{T} , s::StridedVector{T},
    A::StridedMatrix{T}) where {T<:LA.BlasFloat}
    τ2 = promote(τ, zero(T))[1]
    if τ2 isa Union{Bool,T}
        return LA.BLAS.ger!(τ2, v, s, A)
    else
        iszero(τ2) && return 
        m = length(v)
        n = length(s)
        @inbounds for j = 1:n
            temp = τ2 * s[j]'
            @simd for i=1:m
                A[i,j] += v[i] * temp
            end
        end
    end
end

@views function lefthouseholder!(A::AbstractMatrix,v::AbstractArray,s::AbstractArray,τ::Number)
    mul!(s, adjoint(A), v)
    ger2!(-τ', v, s, A)
    return
end

@views function skewhess!(A::AbstractMatrix{T},τ::AbstractVector,η::AbstractVector) where {T}
    n = size(A, 1)
    atmp = similar(A, n)
    @inbounds (for i = 1:n-1
        τ_s, α = householder!(A[i+1:end,i], n - i)
        @views v = A[i+1:end,i]
        η[i] = α
        lefthouseholder!(A[i+1:end,i+1:end], v, atmp[i+1:end], τ_s)
        s = mul!(atmp[i+1:end], A[i+1:end,i+1:end], v)

        for j=i+1:n
            A[j,j] -= τ_s * s[j-i] * v[j-i]'
            for k=j+1:n
                A[k,j] -= τ_s * s[k-i] * v[j-i]'
                A[j,k] = -A[k,j]'
            end
        end
        τ[i] = τ_s
    end)
    return
end

@views function skewlatrd!(A::AbstractMatrix{T},η::AbstractVector,W::AbstractMatrix,τ::AbstractVector,tempconj::AbstractVector,n::Number,nb::Number) where {T}

    @inbounds(for i=1:nb
        if i>1
            if T <: Complex
                @simd for j = 1:i-1
                    tempconj[j] = conj(W[i,j])
                end
                mul!(A[i:n,i], A[i:n,1:i-1], tempconj[1:i-1], 1, 1)
                @simd for j=1:i-1
                    tempconj[j] = conj(A[i,j])
                end
                mul!(A[i:n,i], W[i:n,1:i-1], tempconj[1:i-1], -1, 1)
            else
                mul!(A[i:n,i], A[i:n,1:i-1], W[i,1:i-1], 1, 1)
                mul!(A[i:n,i], W[i:n,1:i-1], A[i,1:i-1], -1, 1)
            end 
        end

        #Generate elementary reflector H(i) to annihilate A(i+2:n,i)
        τ_s,α = householder!(A[i+1:n,i],n-i)
        η[i]   = α
        mul!(W[i+1:n,i], A[i+1:n,i+1:n], A[i+1:n,i], 1, 0)  
        if i>1
            mul!(W[1:i-1,i], adjoint(W[i+1:n,1:i-1]), A[i+1:n,i])
            mul!(W[i+1:n,i], A[i+1:n,1:i-1], W[1:i-1,i], 1, 1)
            mul!(W[1:i-1,i], adjoint(A[i+1:n,1:i-1]),A[i+1:n,i])
            mul!(W[i+1:n,i], W[i+1:n,1:i-1], W[1:i-1,i], -1, 1)
        end
        W[i+1:n,i] .*= τ_s
        
        if T<:Complex
            α = τ_s  * dot(W[i+1:n,i],A[i+1:n,i]) / 2
            W[i+1:n,i] .+= α.*A[i+1:n,i]
        end

        τ[i] = τ_s
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
    nb  = setnb(n)
    A   = S.data
    η   = similar(A, real(T), n - 1)
    τ = similar(A, n - 1)
    W   = similar(A, n, nb)
    Ω = similar(A, n - nb, n - nb)
    tempconj = similar(A, nb)
    oldi = 0
    @inbounds(for i = 1:nb:n-nb-1
        size = n-i+1
        skewlatrd!(A[i:n,i:n], η[i:i+nb-1], W, τ[i:i+nb-1], tempconj,size,nb)
        mul!(Ω[1:n-nb-i+1,1:n-nb-i+1], A[i+nb:n,i:i+nb-1], adjoint(W[nb+1:size,:]))
        s = i+nb-1
        for k = 1:n-s
            A[s+k,s+k] += Ω[k,k] - Ω[k,k]'
            @simd for j = k+1:n-s
                A[s+j,s+k] += Ω[j,k] - Ω[k,j]'
                A[s+k,s+j] = - A[s+j,s+k]'
            end
        end
        oldi = i
    end)
    oldi += nb
    if oldi < n
        skewhess!(A[oldi:n,oldi:n],τ[oldi:end],η[oldi:end])
    end
    return τ, η
end