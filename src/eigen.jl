# Based on eigen.jl in Julia. License is MIT: https://julialang.org/license
@views function LA.eigvals!(A::SkewHermitian{<:Real}, sortby::Union{Function,Nothing}=nothing)
    vals = skeweigvals!(A)
    !isnothing(sortby) && sort!(vals, by = sortby)
    return complex.(0, vals)
end

@views function LA.eigvals!(A::SkewHermitian{<:Real}, irange::UnitRange)
    vals = skeweigvals!(A, irange)
    return complex.(0, vals)
end

@views function LA.eigvals!(A::SkewHermitian{<:Real}, vl::Real,vh::Real)
    vals = skeweigvals!(A, -vh, -vl)
    return complex.(0, vals)
end

@views function LA.eigvals!(A::SkewHermitian{<:Complex}, sortby::Union{Function,Nothing}=nothing)
    H = Hermitian(A.data.*1im)
    if sortby === nothing
        return complex.(0, - eigvals!(H))
    end
    vals = eigvals!(H, sortby)
    reverse!(vals)
    vals.= .-vals
    return complex.(0, vals)
end

@views function LA.eigvals!(A::SkewHermitian{<:Complex}, irange::UnitRange)
    H = Hermitian(A.data.*1im)
    vals = eigvals!(H,-irange)
    vals .= .-vals
    return complex.(0, vals)
end

@views function LA.eigvals!(A::SkewHermitian{<:Complex}, vl::Real,vh::Real)
    H = Hermitian(A.data.*1im)
    vals = eigvals!(H,-vh,-vl)
    vals .= .-vals
    return complex.(0, vals)
end

LA.eigvals(A::SkewHermitian, irange::UnitRange) =
    LA.eigvals!(copyeigtype(A), irange)
LA.eigvals(A::SkewHermitian, vl::Real,vh::Real) =
    LA.eigvals!(copyeigtype(A), vl,vh)

# no need to define LA.eigen(...) since the generic methods should work

@views function skeweigvals!(S::SkewHermitian{<:Real})
    n = size(S.data, 1)
    E = skewblockedhess!(S)[2]
    H = SymTridiagonal(zeros(eltype(E), n), E)
    vals = eigvals!(H)
    return vals .= .-vals
end

@views function skeweigvals!(S::SkewHermitian{<:Real},irange::UnitRange)
    n = size(S.data,1)
    E = skewblockedhess!(S)[2]
    H = SymTridiagonal(zeros(eltype(E), n), E)
    vals = eigvals!(H,irange)
    return vals .= .-vals
end

@views function skeweigvals!(S::SkewHermitian{<:Real},vl::Real,vh::Real)
    n = size(S.data,1)
    E = skewblockedhess!(S)[2]
    H = SymTridiagonal(zeros(eltype(E), n), E)
    vals = eigvals!(H,vl,vh)
    return vals .= .-vals
end

@views function skeweigen!(S::SkewHermitian{<:Real})
    n = size(S.data, 1)

    tau,E = skewblockedhess!(S)
    T = SkewHermTridiagonal(E)
    H1 = Hessenberg{typeof(zero(eltype(S.data))),typeof(T),typeof(S.data),typeof(tau),typeof(false)}(T, 'L', S.data, tau, false)
    A = S.data
    H = SymTridiagonal(zeros(eltype(E), n), E)
    trisol = eigen!(H)

    vals  = trisol.values * 1im
    vals .*= -1
    Qdiag = trisol.vectors

    Qr   = similar(A, n, (n+1)÷2)
    Qim  = similar(A, n, n÷2)
    temp = similar(A, n, n)
    Q=Matrix(H1.Q)
    Q1 = similar(A, (n+1)÷2, n)
    Q2 = similar(A, n÷2, n)

    @inbounds for j = 1:n
        @simd for i = 1:2:n-1
            k = (i+1)÷2
            Q1[k,j] = Qdiag[i,j]
            Q2[k,j] = Qdiag[i+1,j]
        end
    end

    c = 1
    @inbounds for i=1:2:n-1
        k1 = (i+1)÷2
        @simd for j=1:n
            Qr[j,k1] = Q[j,i] * c
            Qim[j,k1] = Q[j,i+1] * c
        end
        c *= (-1)
    end
    if n%2==1
        k=(n+1)÷2
        @simd for j=1:n
            Qr[j,k] = Q[j,n] * c
        end
        Q1[k,:] = Qdiag[n,:]
    end

    mul!(temp,Qr,Q1) #temp is Qr
    mul!(Qdiag,Qim,Q2) #Qdiag is Qim
    
    return vals,temp,Qdiag
end


@views function LA.eigen!(A::SkewHermitian{<:Real})
     vals, Qr, Qim = skeweigen!(A)
     return Eigen(vals,complex.(Qr,Qim))
end

copyeigtype(A::SkewHermitian) = copyto!(similar(A, LA.eigtype(eltype(A))), A)

@views function LA.eigen!(A::SkewHermitian{T}) where {T<:Complex}
    H = Hermitian(A.data.*1im)
    Eig = eigen!(H)
    skew_Eig = Eigen(complex.(0,-Eig.values), Eig.vectors)
    return skew_Eig
end

LA.eigen(A::SkewHermitian) = LA.eigen!(copyeigtype(A))

@views function LA.svdvals!(A::SkewHermitian{<:Real})
    vals = skeweigvals!(A)
    vals .= abs.(vals)
    return sort!(vals; rev=true)
end

LA.svdvals!(A::SkewHermitian{<:Complex}) = svdvals!(Hermitian(A.data.*1im))
LA.svdvals(A::SkewHermitian) = svdvals!(copyeigtype(A))

@views function LA.svd!(A::SkewHermitian{<:Real})
    n = size(A, 1)
    E = eigen!(A)
    U = E.vectors
    vals = imag.(E.values)
    I = sortperm(vals; by = abs, rev = true)
    permute!(vals, I)
    Base.permutecols!!(U, I)
    V = U .* -1im
    @inbounds for i=1:n
        if vals[i] < 0
            vals[i]=-vals[i]
            @simd for j=1:n
                V[j,i]=-V[j,i]
            end
        end
    end
    return LA.SVD(U, vals, adjoint(V))
end
@views function LA.svd(A::SkewHermitian{T}) where {T<:Complex}
    H = Hermitian(A.data.*1im)
    Svd = svd(H)
    return SVD(Svd.U , Svd.S, (Svd.Vt).*(-1im))
end

LA.svd(A::SkewHermitian{<:Real}) = svd!(copyeigtype(A))
