# This file is a part of Julia. License is MIT: https://julialang.org/license


@views function LA.eigvals!(A::SkewSymmetric{<:LA.BlasReal,<:StridedMatrix}, sortby::Union{Function,Nothing}=nothing)
    vals = skeweigvals!(A)
    !isnothing(sortby) && sort!(vals, by=sortby)
    return vals
end

@views function LA.eigvals!(A::SkewSymmetric{<:LA.BlasReal,<:StridedMatrix,}, irange::UnitRange)
    return skeweigvals!(A,irange)
end

@views function LA.eigvals!(A::SkewSymmetric{<:LA.BlasReal,<:StridedMatrix}, vl::Real,vh::Real)
    return skeweigvals!(A,vl,vh)
end

function LA.eigvals(A::SkewSymmetric; sortby::Union{Function,Nothing}=nothing)
    T = eltype(A)
    S = LA.eigtype(T)
    LA.eigvals!(S != T ? convert(AbstractMatrix{S}, A) : copy(A), sortby)
end

function LA.eigvals(A::SkewSymmetric, irange::UnitRange)
    T = eltype(A)
    S = LA.eigtype(T)
    LA.eigvals!(S != T ? convert(AbstractMatrix{S}, A) : copy(A), irange)
end

function LA.eigvals(A::SkewSymmetric, vl::Real, vh::Real)
    T = eltype(A)
    S = LA.eigtype(T)
    LA.eigvals!(S != T ? convert(AbstractMatrix{S}, A) : copy(A), vl, vh)
end

LA.eigmax(A::SkewSymmetric{<:Real,<:StridedMatrix}) = LA.eigvals(A, size(A, 1):size(A, 1))[1]
LA.eigmin(A::SkewSymmetric{<:Real,<:StridedMatrix}) = LA.eigvals(A, 1:1)[1]

@views function skeweigvals!(S::SkewSymmetric)
    n = LA.size(S.data,1)
    E = sktrd!(S)[2]
    H = LA.SymTridiagonal(zeros(n),E)
    vals = LA.eigvals!(H)
    return -vals.*1im
    
end

@views function skeweigvals!(S::SkewSymmetric,irange::UnitRange)
    n = LA.size(S.data,1)
    E = sktrd!(S)[2]
    H = LA.SymTridiagonal(zeros(n),E)
    vals = LA.eigvals!(H,irange)
    return -vals.*1im
    
end

@views function skeweigvals!(S::SkewSymmetric,vl::Real,vh::Real)
    n = LA.size(S.data,1)
    E = sktrd!(S)[2]
    H = LA.SymTridiagonal(zeros(n),E)
    vals = LA.eigvals!(H,vl,vh)
    return -vals.*1im
end

@views function skeweigen!(S::SkewSymmetric)
    n = LA.size(S.data,1)
    tau,E = sktrd!(S)
    A = S.data
    s = LA.similar(A,n)
    H = LA.SymTridiagonal(zeros(n),E)
    trisol = LA.eigen!(H)
    
    vals  = -trisol.values.*1im
    Qdiag = trisol.vectors
    Qr   = LA.similar(A,(n+1)÷2,n)
    Qim  = LA.similar(A,n÷2,n)
    temp = LA.similar(A,n,n)
    
    Q = LA.Matrix(LA.diagm(ones(n)))
    Q1= LA.similar(A,n,(n+1)÷2)
    Q2= LA.similar(A,n,n÷2)
    for i=1:n-2
        t=tau[n-i-1]
        E[1:i+1]=A[n-i:n,n-i-1]
        E[1]=1
        leftHouseholder!(Q[n-i:n,n-i-1:n],E[1:i+1],s[n-i-1:n],t)
        
    end
    
    
    c=1
    #vec=similar(A,n)
    @inbounds for i=1:2:n-1
        k1=(i+1)÷2
        Qr[k1,:] = Qdiag[i,:]
        Qim[k1,:] = Qdiag[i+1,:]
        Qr[k1,:].*=c
        Qim[k1,:].*=c
        #vec=Q[:,(i+1)÷2]
        Q1[:,(i+1)÷2] = Q[:,i]
        #Q[:,i] = vec
        Q2[:,(i+1)÷2] = Q[:,i+1]
        c*=(-1)
    end
    if n%2==1
        Qr[(n+1)÷2,:] = Qdiag[n,:]
        Qr[(n+1)÷2,:].*=c
        Q1[:,(n+1)÷2] = Q[:,n]
    end
    
    
    LA.mul!(temp,Q1,Qr) #temp is Qr
    LA.mul!(Qdiag,Q2,Qim) #Qdiag is Qim
    
    return vals,temp,Qdiag

    
end

LA.eigen!(A::SkewSymmetric{<:LA.BlasReal,<:StridedMatrix}) = skeweigen!(A)

function LA.eigen(A::SkewSymmetric)
    T = eltype(A)
    S = LA.eigtype(T)
    return LA.eigen!(S != T ? convert(AbstractMatrix{S}, A) : copy(A))
end