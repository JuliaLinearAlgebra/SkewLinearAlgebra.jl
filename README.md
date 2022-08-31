
[![CI](https://github.com/JuliaLinearAlgebra
/SkewLinearAlgebra.jl/workflows/CI/badge.svg)](https://github.com/JuliaMath/SkewLinearAlgebra.jl/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/JuliaLinearAlgebra
/SkewLinearAlgebra.jl/badge.svg?branch=main)](https://coveralls.io/github/JuliaLinearAlgebra
/SkewLinearAlgebra.jl?branch=main)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://julialinearalgebra.github.io/SkewLinearAlgebra.jl/dev/)

# SkewLinearAlgebra

The `SkewLinearAlgebra` package provides specialized matrix types, optimized methods of `LinearAlgebra` functions, and a few entirely new functions for dealing with linear algebra on [skew-Hermitian matrices](https://en.wikipedia.org/wiki/Skew-Hermitian_matrix), especially for the case of [real skew-symmetric matrices](https://en.wikipedia.org/wiki/Skew-symmetric_matrix).

In particular, it defines new `SkewHermitian` and `SkewHermTridiagonal` matrix types supporting optimized eigenvalue/eigenvector,
Hessenberg factorization, and matrix exponential/trigonometric functions.  It also provides functions to compute the
[Pfaffian](https://en.wikipedia.org/wiki/Pfaffian) of real skew-symmetric matrices, along with a Cholesky-like factorization.

See the [Documentation](https://julialinearalgebra.github.io/SkewLinearAlgebra.jl/dev/) for details.
