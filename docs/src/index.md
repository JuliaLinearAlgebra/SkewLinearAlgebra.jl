# SkewLinearAlgebra

The `SkewLinearAlgebra` package provides specialized matrix types, optimized methods of `LinearAlgebra` functions, and a few entirely new functions for dealing with linear algebra on [skew-Hermitian matrices](https://en.wikipedia.org/wiki/Skew-Hermitian_matrix), especially for the case of [real skew-symmetric matrices](https://en.wikipedia.org/wiki/Skew-symmetric_matrix).

## Introduction

A skew-Hermitian matrix ``A`` is a square matrix that equals the negative of its conjugate-transpose: ``A=-\overline{A^{T}}=-A^{*}``, equivalent to `A == -A'` in Julia.  (In the real skew-symmetric case, this is simply ``A=-A^T``.)   Such matrices have special computational properties: orthogonal eigenvectors and purely imaginary eigenvalues, "skew-Cholesky" factorizations, and a relative of the determinant called the [Pfaffian](https://en.wikipedia.org/wiki/Pfaffian).

Although any skew-Hermitian matrix ``A`` can be transformed into a Hermitian matrix ``H=iA``, this transformation converts real matrices ``A`` into complex-Hermitian matrices ``H``, which entails at least a factor of two loss in performance and memory usage compared to the real case.   (And certain operations, like the Pfaffian, are *only* defined for the skew-symmetric case.)  `SkewLinearAlgebra` gives you access to the greater performance and functionality that are possible for purely real skew-symmetric matrices.

To achieve this `SkewLinearAlgebra` defines a new matrix type, [`SkewHermitian`](@ref SkewLinearAlgebra.SkewHermitian) (analogous to the [`LinearAlgebra.Hermitian`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.Hermitian) type in the Julia standard library) that gives you access to optimized methods and specialized functionality for skew-Hermitian matrices, especially in the real case.  It also provides a more specialized [`SkewHermTridiagonal`](@ref SkewLinearAlgebra.SkewHermTridiagonal) for skew-Hermitian tridiagonal matrices (analogous to the [`LinearAlgebra.SymTridiagonal`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.SymTridiagonal) type in the Julia standard library) .

## Contents

The `SkewLinearAlgebra` documentation is divided into the following sections:

* [Skew-Hermitian matrices](@ref): the `SkewHermitian` type, constructors, and basic operations
* [Skew-Hermitian tridiagonal matrices](@ref): the `SkewHermTridiagonal` type, constructors, and basic operations
* [Skew-Hermitian eigenproblems](@ref): optimized eigenvalues/eigenvectors (& Schur/SVD), and Hessenberg factorization
* [Trigonometric functions](@ref): optimized matrix exponentials and related functions (`exp`, `sin`, `cos`, etcetera)
* [Pfaffian calculations](@ref): computation of the Pfaffian and log-Pfaffian
* [Skew-Cholesky factorization](@ref): a skew-Hermitian analogue of Cholesky factorization

## Quick start

Here is a simple example demonstrating some of the features of the `SkewLinearAlgebra` package.   See the manual chapters outlines above for the complete details and explanations:
```jl
julia> using SkewLinearAlgebra, LinearAlgebra

julia> A = SkewHermitian([0  1 2
                         -1  0 3
                         -2 -3 0])
3×3 SkewHermitian{Int64, Matrix{Int64}}:
  0   1  2
 -1   0  3
 -2  -3  0

julia> eigvals(A) # optimized eigenvalue calculation (purely imaginary)
3-element Vector{ComplexF64}:
 0.0 - 3.7416573867739404im
 0.0 + 3.7416573867739404im
 0.0 + 0.0im

julia> Q = exp(A) # optimized matrix exponential
3×3 Matrix{Float64}:
  0.348107  -0.933192   0.0892929
 -0.63135   -0.303785  -0.713521
  0.692978   0.192007  -0.694921

julia> Q'Q ≈ I # the exponential of a skew-Hermitian matrix is unitary
true

julia> pfaffian(A) # the Pfaffian (always zero for odd-size skew matrices)
0.0
```

## Acknowledgements

The `SkewLinearAlgebra` package was initially created by [Simon Mataigne](https://github.com/smataigne) and [Steven G. Johnson](https://math.mit.edu/~stevenj/), with support from [UCLouvain](https://uclouvain.be/) and the [MIT–Belgium program](https://misti.mit.edu/mit-belgium).
