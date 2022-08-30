using Documenter, SkewLinearAlgebra, LinearAlgebra
using .DocMeta: setdocmeta!

setdocmeta!(SkewLinearAlgebra, :DocTestSetup, :(using SkewLinearAlgebra, LinearAlgebra);
            recursive=true)

makedocs(modules=[SkewLinearAlgebra], sitename="SkewLinearAlgebra Documentation")

deploydocs(repo="github.com/JuliaLinearAlgebra/SkewLinearAlgebra.jl")
