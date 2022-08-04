# This file is a part of Julia. License is MIT: https://julialang.org/license


"""
    eigvals!(A,sortby)
Returns the eigenvalues of A where A is SkewSymmetric, 
imaginary part of eigenvalues sorted by sortby.
"""
@views function LA.eigvals!(A::SkewSymmetric{<:LA.BlasReal,<:StridedMatrix}, sortby::Union{Function,Nothing}=nothing)
    vals = skeweigvals!(A)
    !isnothing(sortby) && sort!(vals, by=sortby)
    return vals.*1im
end
"""
    eigvals!(A,irange)
Returns the eigenvalues of A where A is SkewSymmetric, 
irange specifies the indices of the eigenvalues to search for.
"""

@views function LA.eigvals!(A::SkewSymmetric{<:LA.BlasReal,<:StridedMatrix,}, irange::UnitRange)
    vals= skeweigvals!(A,irange)
    return vals.*1im
end

"""
    eigvals!(A,vl,vh)
Returns the eigenvalues of A where A is SkewSymmetric, 
[vl,vh] defines the range the imaginary part of the eigenvalues of A must be contained in.
"""
@views function LA.eigvals!(A::SkewSymmetric{<:LA.BlasReal,<:StridedMatrix}, vl::Real,vh::Real)
    vals=skeweigvals!(A,-vh,-vl)
    return vals.*1im
end
"""
    eigvals(A,sortby)
Returns the eigenvalues of A where A is SkewSymmetric, 
imaginary part of eigenvalues sorted by sortby.
"""

function LA.eigvals(A::SkewSymmetric; sortby::Union{Function,Nothing}=nothing)
    T = eltype(A)
    S = LA.eigtype(T)
    eigvals!(S != T ? convert(AbstractMatrix{S}, A) : copy(A), sortby)
end
"""
    eigvals(A,irange)
Returns the eigenvalues of A where A is SkewSymmetric, 
irange specifies the indices of the eigenvalues to search for.
"""

function LA.eigvals(A::SkewSymmetric, irange::UnitRange)
    T = eltype(A)
    S = LA.eigtype(T)
    eigvals!(S != T ? convert(AbstractMatrix{S}, A) : copy(A), irange)
end

"""
    eigvals(A,vl,vh)
Returns the eigenvalues of A where A is SkewSymmetric, 
[vl,vh] defines the range the imaginary part of the eigenvalues of A must be contained in.
"""

function LA.eigvals(A::SkewSymmetric, vl::Real, vh::Real)
    T = eltype(A)
    S = LA.eigtype(T)
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
            mul!(W[:,i],W[:,1:i-1],temp[1:i-1],1,1)
            W[:,i] .*= -t
        end
    end
    mul!(Q,W,Yt)
    Q.+=I
    return
end

@views function dlarft(k,n,V,tau,T)
    prevlastv = n
    for i=1:k
        prevlastv=max(i,prevlastv)
        if tau[i]==0
            println("coucou\n")
            for j=1:i
                T[j,i]=0
            end
        else
            lastv=n
            while lastv>i
                if V[lastv,i] != 0
                    break
                end
                lastv -= 1
            end
            for j=1:i-1
                T[j,i] = - tau[i] * V[i,j]
            end
            j = min(lastv,prevlastv)
            mul!(T[1:i-1,i],transpose(V[i:j,1:i-1]),V[i:j,i], -tau[i], 0)
            LA.BLAS.trmv!('U','N','N', T[1:i-1,1:i-1], T[1:i-1,i])
            T[i,i]=tau[i]
            if i>1
                prevlastv=max(prevlastv,lastv)
            else
                prevlastv=lastv
            end
        end
    end
end
function set_nb2(n::Integer)
    if n<=12
        return max(n-4,1)
    elseif n<=100
        return 10
    else
        return 30
    end
    return 1
end
@views function dormqr(n,k,Q,V,tau)
    nb=set_nb2(n)
    T=similar(Q,nb,nb)
    temp=similar(Q,n,nb)
    temp2=similar(temp,n,nb)
    for i=1:nb:k
        ib=min(nb,k-i+1)
        ni=n-i+1
        dlarft(ib,ni,V[i:n,i:i+ib-1],tau[i:i+ib-1],T[1:ib,1:ib])
        mul!(temp[:,1:ib],Q[1:n,i:n],V[i:n,i:i+ib-1])
        mul!(temp2[:,1:ib],temp[:,1:ib],T[1:ib,1:ib])
        mul!(Q[1:n,i:n],temp2[:,1:ib],transpose(V[i:n , i:i+ib-1]),-1,1)
    end
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

    Qr   = similar(A,n,(n+1)÷2)
    Qim  = similar(A,n,n÷2)
    temp = similar(A,n,n)
    
    Q  = diagm(ones(n))
    LA.LAPACK.ormqr!('L','N',A[2:n,1:n-2],tau,Q[2:end,2:end])
    
    Q1 = similar(A,(n+1)÷2,n)
    Q2 = similar(A,n÷2,n)
    @inbounds for j=1:n
        @simd for i=1:2:n-1
            k=(i+1)÷2
            Q1[k,j] = Qdiag[i,j]
            Q2[k,j] = Qdiag[i+1,j]
        end   
    end

    c = 1
    @inbounds for i=1:2:n-1
        k1 = (i+1)÷2
        @simd for j=1:n
            Qr[j,k1] = Q[j,i]*c
            Qim[j,k1] = Q[j,i+1]*c
        end
        c *= (-1)
    end
    
    if n%2==1
        k=(n+1)÷2
        @simd for j=1:n
            Qr[j,k] = Q[j,n]*c
        end
        Q1[k,:] = Qdiag[n,:]
    end
    mul!(temp,Qr,Q1) #temp is Qr
    mul!(Qdiag,Qim,Q2) #Qdiag is Qim
    
    return vals,temp,Qdiag

    
end
"""
    eigen!(A)

Returns [val,Re(Q),Im(Q)], containing the eigenvalues in vals, 
the real part of the eigenvectors in Re(Q) and the Imaginary part of the eigenvectors in Im(Q)
"""
LA.eigen!(A::SkewSymmetric{<:LA.BlasReal,<:StridedMatrix}) = skeweigen!(A)

"""
    eigen(A)

Returns [val,Re(Q),Im(Q)], containing the eigenvalues in vals, 
the real part of the eigenvectors in Re(Q) and the Imaginary part of the eigenvectors in Im(Q)
"""
function LA.eigen(A::SkewSymmetric)
    T = eltype(A)
    S = LA.eigtype(T)
    return eigen!(S != T ? convert(AbstractMatrix{S}, A) : copy(A))
end

@views function LA.svdvals!(A::SkewSymmetric)
    n=size(A,1)
    vals = skeweigvals!(A)
    @simd for i=1:n
        vals[i]=abs(vals[i])
    end
    sort!(vals;rev=true)
    return vals
end

function LA.svdvals(A::SkewSymmetric)
    return svdvals!(copy(A))
end
@views function LA.svd!(A::SkewSymmetric)
    n=size(A,1)
    eig,Qr,Qim=eigen!(A)
    vals=similar(A,n)
    U=Qim*1im
    U+=Qr
    @simd for i=1:n
        vals[i]=imag(eig[i])
    end
    I=sortperm(vals;by=abs,rev=true)
    permute!(vals,I)
    Base.permutecols!!(U,I)
    V=copy(U)*1im
    V.*= -1
    @inbounds for i=1:n
        if vals[i] < 0
            vals[i]=-vals[i]
            @simd for j=1:n
                V[j,i]=-V[j,i]
            end
        end
    end
    return LA.SVD(U,vals,adjoint(V))
end
function LA.svd(A::SkewSymmetric)
    return svd!(copy(A))
end
