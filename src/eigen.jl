# Based on eigen.jl in Julia. License is MIT: https://julialang.org/license
@views function LA.eigvals!(A::SkewHermitian{<:Real}, sortby::Union{Function,Nothing}=nothing)
    vals = imag.(skeweigvals!(A))
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

LA.eigvals(A::SkewHermitian, sortby::Union{Function,Nothing}) = eigvals!(copyeigtype(A), sortby)
LA.eigvals(A::SkewHermitian, irange::UnitRange) = eigvals!(copyeigtype(A), irange)
LA.eigvals(A::SkewHermitian, vl::Real,vh::Real) = eigvals!(copyeigtype(A), vl,vh)

# no need to define LA.eigen(...) since the generic methods should work

@views function skeweigvals!(S::SkewHermitian{<:Real})
    n = size(S.data, 1)
    n == 1 && return [S.data[1,1]]
    E = skewblockedhess!(S)[2]
    H = SkewHermTridiagonal(E)
    return skewtrieigvals!(H)
end

@views function skeweigvals!(S::SkewHermitian{<:Real},irange::UnitRange)
    n = size(S.data,1)
    n == 1 && return [S.data[1,1]]
    E = skewblockedhess!(S)[2]
    H = SymTridiagonal(zeros(eltype(E), n), E)
    vals = eigvals!(H,irange)
    return vals .= .-vals
end

@views function skeweigvals!(S::SkewHermitian{<:Real},vl::Real,vh::Real)
    n = size(S.data,1)
    n == 1 && imag(S.data[1,1]) > vl && imag(S.data[1,1]) < vh && return [S.data[1,1]]
    E = skewblockedhess!(S)[2]
    H = SymTridiagonal(zeros(eltype(E), n), E)
    vals = eigvals!(H,vl,vh)
    return vals .= .-vals
end

@views function skeweigen!(S::SkewHermitian{T}) where {T<:Real}
    n = size(S.data, 1)
    if n == 1
        return [S.data[1,1]], ones(T,1,1), zeros(T,1,1)
    end
    tau, E = skewblockedhess!(S)
    Tr = SkewHermTridiagonal(E)
    H1 = Hessenberg{typeof(zero(eltype(S.data))),typeof(Tr),typeof(S.data),typeof(tau),typeof(false)}(Tr, 'L', S.data, tau, false)   
    vectorsreal = similar(S, T, n, n)
    vectorsim = similar(S, T, n, n)
    Q = Matrix(H1.Q)

    vals, Qr, Qim = skewtrieigen_divided!(Tr)
    mul!(vectorsreal, Q, Qr)
    mul!(vectorsim, Q, Qim)
    return vals, vectorsreal, vectorsim
end


@views function LA.eigen!(A::SkewHermitian{<:Real})
     vals, Qr, Qim = skeweigen!(A)
     return Eigen(vals,complex.(Qr, Qim))
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
    vals = imag.(skeweigvals!(A))
    vals .= abs.(vals)
    return sort!(vals; rev = true)
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
