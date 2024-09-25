module SkewLinearAlgebraChainRulesCoreExt

function ChainRulesCore.rrule(::typeof(pfaffian), A::SkewHermitian)
    Ω = pfaffian(A)
    pfaffian_pullback(ΔΩ) = NoTangent(), SkewHermitian(rmul!(inv(A)', dot(Ω, ΔΩ)))
    return Ω, pfaffian_pullback
end

function ChainRulesCore.ProjectTo{<:SkewHermitian}(x::AbstractArray)
    return skewhermitian(x)
end

end
