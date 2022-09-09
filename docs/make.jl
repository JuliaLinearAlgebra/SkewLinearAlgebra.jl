using Documenter, SkewLinearAlgebra, LinearAlgebra
using .DocMeta: setdocmeta!

setdocmeta!(SkewLinearAlgebra, :DocTestSetup, :(using SkewLinearAlgebra, LinearAlgebra);
            recursive=true)

makedocs(
    modules = [SkewLinearAlgebra],
    clean = false,
    sitename = "SkewLinearAlgebra Documentation",
    authors = "Simon Mataigne, Steven G. Johnson, and contributors.",
    pages = [
        "Home" => "index.md",
        "Matrix Types" => "types.md",
        "Eigenproblems" => "eigen.md",
        "Exponential/Trigonometric functions" => "trig.md",
        "Pfaffians" => "pfaffian.md",
        "Skew-Cholesky" => "skewchol.md",
    ],
)

deploydocs(repo="github.com/JuliaLinearAlgebra/SkewLinearAlgebra.jl")
