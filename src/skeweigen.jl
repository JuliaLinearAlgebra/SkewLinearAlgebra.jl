# This file is a part of Julia. License is MIT: https://julialang.org/license

function getgivens(a,b)
    nm = hypot(a, b)
    return a / nm , b / nm 
end

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

@views function implicitstep_novec(ev::AbstractVector{T} , n::Integer ) where T
    buldge = zero(T)
    shift = ev[n]^2
    @inbounds(for i=1:n-1
        α = (i > 1 ? ev[i-1] : zero(ev[i]))
        β = ev[i]
        γ = ev[i+1]

        x1 = - α * α - β * β + shift
        x2 = - α * buldge + β * γ 
        c, s = getgivens(x1, x2)
        if i > 1
            ev[i-1] = c * α + s * buldge
        end

        ev[i] = c * β - s * γ
        ev[i+1] = s * β + c * γ

        if i < n-1
            ζ = ev[i+2]
            ev[i+2] *= c
            buldge = s * ζ
        end
    end)
    return

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
    tol = eps(T) * norm(ev)
    max_iter = 30 * n
    iter = 0 ;
    N = n 

    while n > 2 && iter < max_iter
        implicitstep_novec(ev, n - 1)
        if abs(ev[n - 2]) < tol
            eigofblock(ev[n - 1], values[n-1:n] )
            n -= 2
        end
        iter += 1
    end    
    if n == 2
        eigofblock(ev[1], values[1:2])
        return values
    elseif n == 0
        return values
    else
        throw("Maximum number of iterations reached, the algorithm didn't converge")
    end


end

@views function implicitstep_vec!(ev::AbstractVector{T}, Qeven::AbstractMatrix{T}, Qodd::AbstractMatrix{T}, n::Integer, N::Integer) where T
    buldge = zero(T)
    shift = ev[n]^2
    @inbounds(for i=1:n-1
        α = (i > 1 ? ev[i-1] : zero(ev[i]))
        β = ev[i]
        γ = ev[i+1]

        x1 = - α * α - β * β + shift
        x2 = - α * buldge + β * γ 
        c, s = getgivens(x1, x2)
        if i > 1
            ev[i-1] = c * α + s * buldge
        end
        ev[i] = c * β - s * γ
        ev[i+1] = s * β + c * γ
        if i < n-1
            ζ = ev[i+2]
            ev[i+2] *= c
            buldge = s * ζ
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
    return
end

@views function skewtrieigen_merged!(A::SkewHermTridiagonal{T}) where {T<:Real}
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

    tol = eps(T) * norm(ev)
    max_iter = 100 * n
    iter = 0 ;
    halfN = div(n,2)

    while n > 2 && iter < max_iter
        implicitstep_vec!(ev, Qeven, Qodd, n - 1, halfN)
        if abs(ev[n - 2]) < tol
            eigofblock(ev[n - 1], values[n-1:n])
            n -= 2
        end
        iter += 1
    end   
    if n == 2
        eigofblock(ev[1], values[1:2])
        if isodd(N)
            getoddvectors(Qodd, Ginit, N - 1)
        end
        
        s2 = T(1/sqrt(2))
        zero = T(0)
        @inbounds(for i = 1:2:N-1
            ii = div(i+1,2)
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
        throw("Maximum number of iterations reached, the algorithm didn't converge")
    end
end

@views function skewtrieigen_divided!(A::SkewHermTridiagonal{T}) where {T<:Real}
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
    tol = eps(T) * norm(ev)

    max_iter = 30 * n
    iter = 0 ;
    halfN = div(n, 2)
    while n > 2 && iter < max_iter
        implicitstep_vec!(ev, Qeven, Qodd, n - 1, halfN)
        if abs(ev[n - 2]) < tol
            eigofblock(ev[n - 1], values[n-1:n])
            n -= 2
        end
        iter += 1
    end
    if n == 2
        eigofblock(ev[1], values[1:2])
        if isodd(N)
            getoddvectors(Qodd, Ginit, N - 1)
        end
        
        s2 = T(1/sqrt(2))
        NN = div(N+1, 2)
        @inbounds(for i = 1:2:N-1
            ii = div(i+1,2)
            for j = 1:2:N-1
                jj = div(j+1,2)
                vectorsreal[j, i] = s2*Qodd[jj, ii]
                vectorsreal[j+1, i+1] = -s2*Qeven[jj, ii]
                vectorsim[j, i+1] = vectorsreal[j, i]
                vectorsim[j+1, i] = vectorsreal[j+1, i+1] 
            end
            
            if isodd(N)
                vectorsreal[N, i] = s2 * Qodd[halfN+1, ii]
                vectorsim[N, i+1] = vectorsreal[N, i]
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
        throw("Maximum number of iterations reached, the algorithm didn't converge")
    end
      
end