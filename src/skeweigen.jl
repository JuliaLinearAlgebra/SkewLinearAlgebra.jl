# This file is a part of Julia. License is MIT: https://julialang.org/license

function getgivens(a,b)
    nm = hypot(a, b)
    iszero(nm) && return 1, 0
    return a / nm , b / nm
end

#Eliminate the zero eigenvalue if odd size. 
#Proposed in Ward, R.C., Gray L.,J., Eigensystem computation for skew-symmetric matrices and a class of symmetric matrices, 1975.
@views function reducetozero(ev::AbstractVector{T}, G::AbstractVector{T}, n::Integer) where T
    n == 0 && return
    bulge = zero(T)
    if n > 2
        #Kill last row
        α = ev[n-2]
        β = ev[n-1]
        γ = ev[n]
        c, s = getgivens(-β, γ)
        G[n-1] = c ; G[n] = -s;
        ev[n-2] *= c
        ev[n-1] = c * β - s * γ
        ev[n] = 0
        bulge  = -s * α

        #Chase the bulge
        for i = n-2:-2:4
            α = ev[i-2]
            β = ev[i-1]
            c, s = getgivens(-β, bulge)
            G[i-1] = c; G[i] = -s
            ev[i-2] *= c
            ev[i-1] = c * β - s * bulge
            bulge  = - s * α
        end
    end

    #Make the bulge disappear
    β = (n == 2 ? ev[2] : bulge)
    α = ev[1]
    c, s = getgivens(-α, β)
    G[1] = c; G[2] = -s
    ev[1] = c * α -s * β
    if n == 2
        ev[2] = s * α + β * c
    end
end

@views function getoddvectors(Qodd::AbstractMatrix{T}, G::AbstractVector{T}, n::Integer) where T
    nn = div(n, 2) + 1
    @inbounds( for i = 1:2:n-1
        c = G[i]
        s = G[i+1]
        ii = div(i+1,2)
        for j = 1:nn
            σ = Qodd[ii, j]
            ω = Qodd[nn, j]
            Qodd[ii, j] = c * σ + s * ω
            Qodd[nn, j] = - s * σ + c * ω
        end
    end)
end

@views function eigofblock(k::Number, val::AbstractVector)
    val[1] = complex(0, k)
    val[2] = complex(0, -k)
end

function getshift(ev::AbstractVector{T}) where T
    return ev[2]^2
end

@views function implicitstep_novec(ev::AbstractVector{T} , n::Integer, start::Integer) where T
    bulge = zero(T)
    shift = getshift(ev[n-1:n])
    tol = T(1) * eps(T)
    @inbounds(for i = start:n-1
        α = (i > start ? ev[i-1] : zero(ev[i]))
        β = ev[i]
        γ = ev[i+1]
        x1 = - α * α - β * β + shift
        x2 = - α * bulge + β * γ
        c, s = (i > start ? getgivens(α, bulge) : getgivens(x1, x2))

        if i > 1
            ev[i-1] = c * α + s * bulge
        end

        ev[i] = c * β - s * γ
        ev[i+1] = s * β + c * γ

        if i < n-1
            ζ = ev[i+2]
            ev[i+2] *= c
            bulge = s * ζ
            if abs(bulge) < tol && abs(ev[i]) < tol
                start = i + 1
                return start
            end
        end
    end)
    return start
end

@views function skewtrieigvals!(A::SkewHermTridiagonal{T,V,Vim}) where {T<:Real,V<:AbstractVector{T},Vim<:Nothing}
    n = size(A, 1)
    values = complex(zeros(T, n))
    ev = A.ev
    if isodd(n)
        n -= 1
        Ginit = similar(A, T, n)
        reducetozero(ev, Ginit, n)
    end
    tol = eps(T) * T(10)
    max_iter = 30 * n
    iter = 0 ;
    mem = T(1); count_static = 0    #mem and count_static allow to detect the failure of Wilkinson shifts.
    start = 1                       #start remembers if a zero eigenvalue appeared in the middle of ev.
    while n > 2 && iter < max_iter
        start = implicitstep_novec(ev, n - 1, start)
        while n > 2 && abs(ev[n-1]) <= tol
                values[n] = 0
                n -= 1
        end
        while n > 2 && abs(ev[n - 2]) <= tol * abs(ev[n - 1])
            eigofblock(ev[n - 1], values[n-1:n] )
            n -= 2
        end
        if start > n-2
            start = 1
        end
        if n>1 && abs(mem - ev[n-1]) < T(0.0001) * abs(ev[n-1])
            count_static += 1
            if count_static > 4
                #Wilkinson shifts have failed, change strategy using LAPACK tridiagonal symmetric solver.
                values[1:n] .= complex.(0, skewtrieigvals_backup!(SkewHermTridiagonal(ev[1:n-1])))
                return values
            end
        else
            count_static = 0
        end
        mem = (n>1 ? ev[n-1] : T(0))
        iter += 1
    end
    if n == 2
        eigofblock(ev[1], values[1:2])
        return values
    elseif n == 1
        values[1] = 0
        return values
    elseif n == 0
        return values
    else
        error("Maximum number of iterations reached, the algorithm didn't converge")
    end
end

@views function implicitstep_vec!(ev::AbstractVector{T}, Qeven::AbstractMatrix{T}, Qodd::AbstractMatrix{T}, n::Integer, N::Integer, start::Integer) where T
    bulge = zero(T)
    shift = getshift(ev[n-1:n])
    tol = 10 * eps(T)
    @inbounds(for i = start:n-1
        α = (i > start ? ev[i-1] : zero(ev[i]))
        β = ev[i]
        γ = ev[i+1]

        x1 = - α * α - β * β + shift
        x2 = - α * bulge + β * γ
        c, s = (i > start ? getgivens(α,bulge) : getgivens(x1, x2))
        if i > 1
            ev[i-1] = c * α + s * bulge
        end
        ev[i] = c * β - s * γ
        ev[i+1] = s * β + c * γ
        if i < n-1
            ζ = ev[i+2]
            ev[i+2] *= c
            bulge = s * ζ
            if abs(bulge) < tol && abs(ev[i]) < tol
                start = i + 1
            end
        end
        Q = (isodd(i) ? Qodd : Qeven)
        k = div(i+1, 2)
        for j = 1:N
            σ = Q[j, k]
            ω = Q[j, k+1]
            Q[j, k] = c*σ + s*ω
            Q[j, k+1] = -s*σ + c*ω
        end
    end)
    return start
end

@views function skewtrieigen_merged!(A::SkewHermTridiagonal{T}) where {T<:Real}
    Backup = copy(A)
    n = size(A, 1)
    values = complex(zeros(T, n))
    vectors = similar(A, Complex{T}, n, n)
    Qodd = diagm(ones(T, div(n+1,2)))
    Qeven = diagm(ones(T, div(n,2)))
    ev = A.ev
    N = n
    if isodd(n)
        n -= 1
        Ginit = similar(A, T, n)
        reducetozero(ev, Ginit, n)
    end

    tol = eps(T) * T(10)
    max_iter = 30 * n
    iter = 0 ;
    halfN = div(n, 2)
    mem = T(1); count_static = 0    #mem and count_static allow to detect the failure of Wilkinson shifts.
    start = 1                       #start remembers if a zero eigenvalue appeared in the middle of ev.
    while n > 2 && iter < max_iter
        start = implicitstep_vec!(ev, Qeven, Qodd, n - 1, halfN, start)
        while n > 2 && abs(ev[n-1]) <= tol
            values[n] = 0
            n -= 1
        end
        while n > 2 && abs(ev[n - 2]) <= tol * abs(ev[n - 1])
            eigofblock(ev[n - 1], values[n-1:n] )
            n -= 2
        end
        if start > n-2
            start = 1
        end
        
        if n>1 && abs(mem - ev[n-1]) < T(0.0001) * abs(ev[n-1])
            count_static += 1
            if count_static > 4
                #Wilkinson shifts have failed, change strategy using LAPACK tridiagonal symmetric solver.
                values, Q = skewtrieigen_backup!(Backup)
                return Eigen(values, Q)
            end
        else
            count_static = 0
        end
        mem = (n>1 ? ev[n-1] : T(0))
        iter += 1
    end
    
    if n > 0
        if n == 2 
            eigofblock(ev[1], values[1:2])
        else
            values[1] = 0
        end
        if isodd(N)
            getoddvectors(Qodd, Ginit, N - 1)
        end

        s2 = T(1/sqrt(2))
        zero = T(0)
        i = 1
        countzeros = 0
        @inbounds(while i <= N 
            ii = div(i+1,2)
            if iszero(values[i])
                if iseven(countzeros)
                    for j = 1:2:N-1
                        jj = div(j+1, 2)
                        vectors[j, i] = complex(Qodd[jj, ii], zero)
                    end
                    if isodd(N)
                        vectors[N, i] = complex(Qodd[halfN+1, ii],zero)
                    end
                else
                    for j = 1:2:N-1
                        jj = div(j+1, 2)
                        vectors[j+1, i] = complex(Qeven[jj, ii], zero)
                    end
                end
                countzeros +=1
                i += 1
            else
                if isodd(countzeros)
                    for j = 1:2:N-1
                        jj = div(j+1, 2)
                        vectors[j, i] = complex(zero, -s2 * Qodd[jj, ii+1])
                        vectors[j+1, i] = complex(s2 * Qeven[jj, ii], zero)
                        vectors[j, i+1] = complex(-s2 * Qodd[jj, ii+1], zero)
                        vectors[j+1, i+1] = complex(zero,  s2 * Qeven[jj,ii])
                    end
                    if isodd(N)
                        vectors[N, i] = complex(zero, -s2 * Qodd[halfN+1, ii+1])
                        vectors[N, i+1] = complex(-s2 * Qodd[halfN+1, ii+1], zero)
                    end
                else
                    for j = 1:2:N-1
                        jj = div(j+1, 2)
                        vectors[j, i] = complex(s2 * Qodd[jj, ii], zero)
                        vectors[j+1, i] = complex(zero, - s2 * Qeven[jj, ii])
                        vectors[j, i+1] = complex(zero,s2 * Qodd[jj, ii])
                        vectors[j+1, i+1] = complex(- s2 * Qeven[jj,ii], zero)
                    end
                    if isodd(N)
                        vectors[N, i] = complex(s2 * Qodd[halfN+1, ii],zero)
                        vectors[N, i+1] = complex(zero, s2 * Qodd[halfN+1, ii])
                    end
                end
                i+=2
            end
                
        end)

        if isodd(N)
            @inbounds(for j = 1: 2: N
                jj = div(j+1,2)
                vectors[j, N] = complex(Qodd[jj, halfN+1], zero)
            end)
        end
        return Eigen(values, vectors)
    elseif n == 0
        return Eigen(values, complex.(Qodd))
    else
        error("Maximum number of iterations reached, the algorithm didn't converge")
    end
end

@views function skewtrieigen_divided!(A::SkewHermTridiagonal{T}) where {T<:Real}
    Backup = copy(A)
    n = size(A, 1)
    values = complex(zeros(T, n))
    vectorsreal = similar(A, n, n)
    vectorsim = similar(A, n, n)
    Qodd = diagm(ones(T, div(n+1,2)))
    Qeven = diagm(ones(T, div(n,2)))
    ev = A.ev
    N = n

    if isodd(n)
        n -= 1
        Ginit = similar(A, T, n)
        reducetozero(ev, Ginit, n)
    end
    tol = eps(T)*T(10)

    max_iter = 30 * n
    iter = 0 ;
    halfN = div(n, 2)
    mem = T(1); count_static = 0    #mem and count_static allow to detect the failure of Wilkinson shifts.
    start = 1                       #start remembers if a zero eigenvalue appeared in the middle of ev.
    while n > 2 && iter < max_iter
        start = implicitstep_vec!(ev, Qeven, Qodd, n - 1, halfN, start)
        while n > 2 && abs(ev[n-1]) <= tol
            values[n] = 0
            n -= 1
        end
        while n > 2 && abs(ev[n - 2]) <= tol * abs(ev[n - 1])
            eigofblock(ev[n - 1], values[n-1:n] )
            n -= 2
        end
        if n > 2 && abs(ev[n-1]-mem) < tol
            eigofblock(ev[n - 1], values[n-1:n] )
            n -= 2
        end
        if start > n-2
            start = 1
        end
        if n>1 && abs(mem - ev[n-1]) < T(0.0001) * abs(ev[n-1])
            count_static += 1
            if count_static > 4
                #Wilkinson shifts have failed, change strategy using LAPACK tridiagonal symmetric solver.
                Q  = complex.(vectorsreal, vectorsim)
                values, Q = skewtrieigen_backup!(Backup)
                vectorsreal .= real.(Q) 
                vectorsim   .= imag.(Q)
                return values, vectorsreal, vectorsim
            end
        else
            count_static = 0
        end
        mem = (n>1 ? ev[n-1] : T(0))
        iter += 1
    end
    if n > 0
        if n == 2 
            eigofblock(ev[1], values[1:2])
        else
            values[1] = 0
        end
        
        if isodd(N)
            getoddvectors(Qodd, Ginit, N - 1)
        end

        s2 = T(1/sqrt(2))
        NN = div(N+1, 2)

        i = 1
        countzeros = 0
        @inbounds(while i <= N 
            ii = div(i+1,2)
            if iszero(values[i])
                if iseven(countzeros)
                    for j = 1:2:N-1
                        jj = div(j+1, 2)
                        vectorsreal[j, i] = Qodd[jj, ii]
                    end
                    if isodd(N)
                        vectorsreal[N, i] = Qodd[halfN+1, ii]
                    end
                else
                    for j = 1:2:N-1
                        jj = div(j+1, 2)
                        vectorsreal[j+1, i] = Qeven[jj, ii]
                    end
                end
                countzeros +=1
                i += 1
            else
                if isodd(countzeros)
                    for j = 1:2:N-1
                        jj = div(j+1, 2)
                        vectorsim[j, i] =  -s2 * Qodd[jj, ii+1]
                        vectorsreal[j+1, i] = s2 * Qeven[jj, ii]
                        vectorsreal[j, i+1] = -s2 * Qodd[jj, ii+1]
                        vectorsim[j+1, i+1] =  s2 * Qeven[jj,ii]
                    end
                    if isodd(N)
                        vectorsreal[N, i] =  -s2 * Qodd[halfN+1, ii+1]
                        vectorsim[N, i+1] = -s2 * Qodd[halfN+1, ii+1]
                    end
                else
                    for j = 1:2:N-1
                        jj = div(j+1, 2)
                        vectorsreal[j, i] = s2 * Qodd[jj, ii]
                        vectorsim[j+1, i] =  - s2 * Qeven[jj, ii]
                        vectorsim[j, i+1] = s2 * Qodd[jj, ii]
                        vectorsreal[j+1, i+1] = - s2 * Qeven[jj,ii]
                    end
                    if isodd(N)
                        vectorsreal[N, i] = s2 * Qodd[halfN+1, ii]
                        vectorsim[N, i+1] =  s2 * Qodd[halfN+1, ii]
                    end
                end
                i+=2
            end
                
        end)

        if isodd(N)
            @inbounds(for j = 1:2:N
                jj = div(j+1,2)
                vectorsreal[j,N] = Qodd[jj, halfN+1]
            end)
        end
        return values, vectorsreal, vectorsim
    elseif n == 0
        return values, vectorsreal, vectorsim
    else
        error("Maximum number of iterations reached, the algorithm didn't converge")
    end

end

#The Wilkinson shifts have some pathological cases.
#In these cases, the skew-symmetric eigenvalue problem is solved as detailed in
#C. Penke, A. Marek, C. Vorwerk, C. Draxl, P. Benner, High Performance Solution of Skew-symmetric Eigenvalue Problems with Applications in Solving the Bethe-Salpeter Eigenvalue Problem, Parallel Computing, Volume 96, 2020.

@views function skewtrieigvals_backup!(S::SkewHermTridiagonal{T,V,Vim}) where {T<:Real,V<:AbstractVector{T},Vim<:Nothing}
    n = size(S,1)
    H = SymTridiagonal(zeros(eltype(S.ev), n), copy(S.ev))
    vals = eigvals!(H)
    return vals .= .-vals
end

@views function skewtrieigen_backup!(S::SkewHermTridiagonal{T,V,Vim}) where {T<:Real,V<:AbstractVector{T},Vim<:Nothing}

    n = size(S, 1)
    H = SymTridiagonal(zeros(T, n), S.ev)
    trisol = eigen!(H)
    vals  = complex.(0, -trisol.values)
    Qdiag = complex(zeros(T,n,n))
    c = 1
    @inbounds for j=1:n
        c = 1
        @simd for i=1:2:n-1
            Qdiag[i,j]  = trisol.vectors[i,j] * c
            Qdiag[i+1,j] = complex(0, trisol.vectors[i+1,j] * c)
            c *= (-1)
        end
    end
    if n % 2 == 1
        Qdiag[n,:] = trisol.vectors[n,:] * c
    end
    return vals, Qdiag
end