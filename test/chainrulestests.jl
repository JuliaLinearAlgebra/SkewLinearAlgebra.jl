using ChainRulesTestUtils
using Random
using SkewLinearAlgebra
using LinearAlgebra
using FiniteDifferences

ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::SkewHermitian) = skewhermitian!(rand(rng, eltype(x), size(x)...))

# Required to make finite differences behave correctly
function FiniteDifferences.to_vec(x::SkewHermitian)
    m = size(x, 1)
    v = Vector{eltype(x)}(undef, m * (m - 1) ÷ 2)
    k = 1
    for i in 2:m, j in 1:i-1
        @inbounds v[k] = x[i, j]
        k += 1
    end

    function from_vec(v)
        y = zero(x)
        k = 1
        for i in 2:m, j in 1:i-1
            @inbounds y[i, j] = v[k]
            @inbounds y[j, i] = -v[k]
            k += 1
        end
        return y
    end
    return v, from_vec
end

m = 10
A = skewhermitian(rand(m, m))

test_rrule(pfaffian, A)

