# This file is a part of Julia. License is MIT: https://julialang.org/license


"""
    eigvals!(A,sortby)
Returns the eigenvalues of A where A is SkewSymmetric, 
imaginary part of eigenvalues sorted by sortby.
"""
@views function LA.eigvals!(A::SkewSymmetric{<:BlasReal,<:StridedMatrix}, sortby::Union{Function,Nothing}=nothing)
    vals = skeweigvals!(A)
    !isnothing(sortby) && sort!(vals, by=sortby)
    return vals.*1im
end
"""
    eigvals!(A,irange)
Returns the eigenvalues of A where A is SkewSymmetric, 
irange specifies the indices of the eigenvalues to search for.
"""

@views function LA.eigvals!(A::SkewSymmetric{<:BlasReal,<:StridedMatrix,}, irange::UnitRange)
    vals= skeweigvals!(A,irange)
    return vals.*1im
end

"""
    eigvals!(A,vl,vh)
Returns the eigenvalues of A where A is SkewSymmetric, 
[vl,vh] defines the range the imaginary part of the eigenvalues of A must be contained in.
"""
@views function LA.eigvals!(A::SkewSymmetric{<:BlasReal,<:StridedMatrix}, vl::Real,vh::Real)
    vals=skeweigvals!(A,vl,vh)
    return vals.*1im
end
"""
    eigvals(A,sortby)
Returns the eigenvalues of A where A is SkewSymmetric, 
imaginary part of eigenvalues sorted by sortby.
"""

function LA.eigvals(A::SkewSymmetric; sortby::Union{Function,Nothing}=nothing)
    T = eltype(A)
    S = eigtype(T)
    eigvals!(S != T ? convert(AbstractMatrix{S}, A) : copy(A), sortby)
end
"""
    eigvals(A,irange)
Returns the eigenvalues of A where A is SkewSymmetric, 
irange specifies the indices of the eigenvalues to search for.
"""

function LA.eigvals(A::SkewSymmetric, irange::UnitRange)
    T = eltype(A)
    S = eigtype(T)
    eigvals!(S != T ? convert(AbstractMatrix{S}, A) : copy(A), irange)
end

"""
    eigvals(A,vl,vh)
Returns the eigenvalues of A where A is SkewSymmetric, 
[vl,vh] defines the range the imaginary part of the eigenvalues of A must be contained in.
"""

function LA.eigvals(A::SkewSymmetric, vl::Real, vh::Real)
    T = eltype(A)
    S = eigtype(T)
    eigvals!(S != T ? convert(AbstractMatrix{S}, A) : copy(A), vl, vh)
end

eigmax(A::SkewSymmetric{<:Real,<:StridedMatrix}) = eigvals(A, size(A, 1):size(A, 1))[1]
eigmin(A::SkewSymmetric{<:Real,<:StridedMatrix}) = eigvals(A, 1:1)[1]

@views function skeweigvals!(S::SkewSymmetric)
    n = size(S.data,1)
    E = sktrd!(S)[2]
    H = SymTridiagonal(zeros(n),E)
    vals = eigvals!(H)
    return -vals
    
end

@views function skeweigvals!(S::SkewSymmetric,irange::UnitRange)
    n = size(S.data,1)
    E = sktrd!(S)[2]
    H = SymTridiagonal(zeros(n),E)
    vals = eigvals!(H,irange)
    return -vals
    
end

@views function skeweigvals!(S::SkewSymmetric,vl::Real,vh::Real)
    n = size(S.data,1)
    E = sktrd!(S)[2]
    H = SymTridiagonal(zeros(n),E)
    vals = eigvals!(H,vl,vh)
    return -vals
end
@views function WYform(A,tau,Q)
    n = size(A,1)
    W = zeros(n-1,n-2)
    Yt = zeros(n-2,n-1)
    temp = zeros(n-1)
    
    for i = 1:n-2
        t = tau[i]
        v = A[i+1:n,i]
        if i == 1
            W[i:end,i] = -t*v
            Yt[i,i:end] = v
        else
            mul!(temp[1:i-1],Yt[1:i-1,i:end],v)
            Yt[i,i:end] = v
            W[i:end,i] = v
            gemv!('n',1.0,W[:,1:i-1],temp[1:i-1],1.0,W[:,i])
            W[:,i] .*= -t
        end
    end
    mul!(Q,W,Yt)
    return
end
@views function skeweigen!(S::SkewSymmetric)
    n = size(S.data,1)
    tau,E = sktrd!(S)
    A = S.data
    s = similar(A,n)
    H = SymTridiagonal(zeros(n),E)
    trisol = eigen!(H)
    
    vals  = trisol.values*1im
    vals .*= -1
    Qdiag = trisol.vectors

    Qr   = similar(A,(n+1)÷2,n)
    Qim  = similar(A,n÷2,n)
    temp = similar(A,n,n)
    
    Q  = diagm(ones(n))
    Q1 = similar(A,n,(n+1)÷2)
    Q2 = similar(A,n,n÷2)
    
    
    for i=1:n-2
        t = tau[n-i-1]
        leftHouseholder!(Q[n-i:n,n-i-1:n], A[n-i:n,n-i-1], s[n-i-1:n], t)
    end
    
    """
    #T = similar(A,2,2)
    #G = similar(A,n-1,2)
    #F = similar(A,n-1,2)
    #temp2 = similar(A,2,n-1)
    oldi=0
    @inbounds for i=1:2:n-3
            k=n-i-1
            t1 = tau[k]
            t2=tau[k-1]
            T[1,1] = -t2
            T[2,2] = -t1
            T[1,2] = t1*t2/2
            T[2,1] = t1*t2/2
            G[1:i+2,:] = A[k:n,k-1:k]
            G[1,1] = 1
            G[2,2] = 1
            
            mul!(F[1:i+2,:],G[1:i+2,:],T)
            mul!(temp2[:,1:i+2],transpose(G[1:i+2,:]),Q[k:n,k:n])
            mul!(temp[k:n,k:n],F[1:i+2,:],temp2[:,1:i+2])
            @simd for d=k:n
                @simd for j=k:n
                    @inbounds Q[j,d] -= temp[j,d]
                end
            end
            
            oldi=i
            #leftHouseholder!(Q[n-i:n,n-i-1:n],E[1:i+1],s[n-i-1:n],t) 
    end
    if oldi==n-4
        t=tau[1]
        E=A[2:n,1]
        E[1]=1
        leftHouseholder!(Q[2:n,1:n],E,s[1:n],t)
    end
    """
    """
    WYform(A,tau,Q[2:n,2:n])
    Q += I
    """

    c = 1
    @inbounds for i=1:2:n-1
        k1 = (i+1)÷2
        Qr[k1,:] = Qdiag[i,:]
        Qr[k1,:] .*=c
        Qim[k1,:] = Qdiag[i+1,:]
        Qim[k1,:] .*=c
        Q1[:,(i+1)÷2] = Q[:,i]
        Q2[:,(i+1)÷2] = Q[:,i+1]
        c *= (-1)
    end
    if n%2==1
        Qr[(n+1)÷2,:] = Qdiag[n,:]
        Qr[(n+1)÷2,:] .*= c
        Q1[:,(n+1)÷2] = Q[:,n]
    end
    mul!(temp,Q1,Qr) #temp is Qr
    mul!(Qdiag,Q2,Qim) #Qdiag is Qim
    
    return vals,temp,Qdiag

    
end
"""
    eigen!(A)

Returns [val,Re(Q),Im(Q)], containing the eigenvalues in vals, 
the real part of the eigenvectors in Re(Q) and the Imaginary part of the eigenvectors in Im(Q)
"""

eigen!(A::SkewSymmetric{<:BlasReal,<:StridedMatrix}) = skeweigen!(A)
"""
    eigen(A)

Returns [val,Re(Q),Im(Q)], containing the eigenvalues in vals, 
the real part of the eigenvectors in Re(Q) and the Imaginary part of the eigenvectors in Im(Q)
"""

function LA.eigen(A::SkewSymmetric)
    T = eltype(A)
    S = eigtype(T)
    return eigen!(S != T ? convert(AbstractMatrix{S}, A) : copy(A))
end