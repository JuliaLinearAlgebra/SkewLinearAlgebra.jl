
@views function eig_of_skew_block(k::Number, val::AbstractVector)
    val[1] = complex(0, -k)
    val[2] = complex(0, k)
end 

@views function implicit_step(ev::AbstractVector{T} , n::Integer ) where T
    buldge = zero(T)
    shift = ev[n]^2
    for i=1:n-1

        α = (i > 1 ? ev[i-1] : zero(ev[i]))
        β = ev[i]
        γ = ev[i+1]

        x1 = - α * α - β * β + shift
        x2 = - α * buldge + β * γ 
        nm = sqrt(x1 * x1 + x2 * x2)
        c = x1/nm
        s = x2/nm

        if i > 1
            ev[i-1] = -c*α-s*buldge
        end

        ev[i] = -c*β+s*γ
        ev[i+1] = s*β+c*γ

        if i < n-1
            ζ = ev[i+2]
            ev[i+2] *= c
            buldge = -s*ζ
        end
    end
    return

end

@views function QR_with_shifts(A::SkewHermTridiagonal{T}) where {T<:Real}
    n = size(A, 1)

    ev = A.ev
    tol = T(1e-10) * norm(ev)
    max_iter = 16 * n
    iter = 0 ; n_converged = 0
    values = complex(zeros(T, n))
    N = n

    while n_converged < N && iter < max_iter
        implicit_step(ev, n - 1)
        if abs(ev[n - 2]) < tol
            n_converged += 2
            eig_of_skew_block(ev[n - 1], values[n_converged-1:n_converged] )
            n -= 2
        end
        if n == 2
            eig_of_skew_block(ev[1], values[end-1:end])
            return values
        end
        iter += 1
    end   
end

#=
BLAS.set_num_threads(1)
n = 999 #n must be kept odd for the moment

v = rand(n)
A = SkewHermTridiagonal(v)
@btime QR_with_shifts(copy(A)) 
@btime eigvals(A) 
a=1

=#
