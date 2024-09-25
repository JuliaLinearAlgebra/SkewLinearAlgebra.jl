using ChainRulesCore
using ChainRules
using ChainRulesTestUtils

function ChainRulesCore.rrule(::Type{SkewHermitian}, val) # constructor rrule
    y = SkewHermitian(val)
    function Foo_pb(ΔFoo) 
        return (NoTangent(), unthunk(ΔFoo).data)
    end
    return y, Foo_pb
end

function ChainRules.rrule(::typeof(pfaffian), x::Union{SkewHermitian})  #pfaffian rrule
    Ω = pfaffian(x)
    function pullback(ΔΩ)
        ∂x = #=0.5 *=# inv(x)' * dot(Ω, ΔΩ)  #we removed the 0.5 because changing element ij immediatelyu changes element ji hence this gives a factor 2.
        return (NoTangent(), SkewHermitian(∂x))
    end
    return Ω, pullback
end

#perhaps should add more options here ? Defenitely we will need + , - , * . At this point I'm happy though If I can just take the derivative of Pfaffian and it's constructor. All other operators I can do with dense matrices for now...