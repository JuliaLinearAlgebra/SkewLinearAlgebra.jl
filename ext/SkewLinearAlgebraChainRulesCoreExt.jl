module SkewLinearAlgebraChainRulesCoreExt

using LinearAlgebra
using SkewLinearAlgebra
using ChainRulesCore


function ChainRulesCore.rrule(::Type{SkewHermitian}, val) 
    y = SkewHermitian(val)
    function Foo_pb(ΔFoo) 
        return (NoTangent(), unthunk(ΔFoo).data)
    end
    return y, Foo_pb
end

function ChainRulesCore.rrule(::typeof(pfaffian), A::SkewHermitian)
    Ω = pfaffian(A)
    pfaffian_pullback(ΔΩ) = NoTangent(), SkewHermitian(rmul!(inv(A)', dot(Ω, ΔΩ))) #potentially need the 0.5 here !
    return Ω, pfaffian_pullback
end

function ChainRulesCore.ProjectTo{<:SkewHermitian}(x::AbstractArray)
    return skewhermitian(x)
end

end
