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

#we make a small test for this 
A = skewhermitian(rand(2,2))
B = skewhermitian(rand(2,2))

test_rrule(pfaffian, A)   #this one doesn't work since it seems to be getting non skew symmetric matrices at some point

test_rrule(pfaffian, A⊢B)  #SO I wanted to specify the tangent but the same happens, judging from our short conversation this might be because I'm ill defining the tangent B ?

test_rrule(SkewHermitian, A.data ⊢ B.data)#constructor false because ΔFoo supposedly has no field data. But ΔFoo should be a SkewHermitian object, so it should have a field data.

