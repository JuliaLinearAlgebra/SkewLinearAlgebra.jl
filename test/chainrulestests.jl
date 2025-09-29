using ChainRulesTestUtils
using Random
using SkewLinearAlgebra
using LinearAlgebra
using FiniteDifferences

ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::SkewHermitian) = skewhermitian!(rand(rng, eltype(x), size(x)...))

# Required to make finite differences behave correctly
function FiniteDifferences.to_vec(A::SkewHermitian)
    m = size(A, 1)
    v = [A[i,j] for i in 2:m for j in 1:i-1]
    function from_vec(v)
        B = zero(A)
        k = 1
        for i in 2:m, j in 1:i-1
            @inbounds B[i,j] = v[k]
            k += 1
        end
        return B
    end
    return v, from_vec
end

@testset "automatic differentiation" begin
    m = 10
    inds = [1,2]
    A = skewhermitian(rand(m, m))

    test_rrule(SkewHermitian, A) # test constructor

    test_rrule(pfaffian, A) # test pfaffian
    test_rrule(pfaffian, SkewHermitian(A[inds, inds])) # test pfaffian of submatrix
end