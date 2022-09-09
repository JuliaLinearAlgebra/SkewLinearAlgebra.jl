## Skew-Cholesky factorization

The package provides a Cholesky-like factorization for real skew-symmetric matrices as presented in P. Benner et al, "[Cholesky-like factorizations of skew-symmetric matrices](https://etna.ricam.oeaw.ac.at/vol.11.2000/pp85-93.dir/pp85-93.pdf)"(2000).
Every real skew-symmetric matrix ``A`` can be factorized as ``A=P^TR^TJRP`` where ``P`` is a permutation matrix, ``R`` is an `UpperTriangular` matrix and J is of a special type called `JMatrix` that is a tridiagonal skew-symmetric matrix composed of diagonal blocks of the form ``B=[0, 1; -1, 0]``. The `JMatrix` type implements efficient operations related to the shape of the matrix as matrix-matrix/vector multiplication and inversion.
The function `skewchol` implements this factorization and returns a `SkewCholesky` structure composed of the matrices `Rm` and `Jm` of type `UpperTriangular` and `JMatrix` respectively. The permutation matrix ``P`` is encoded as a permutation vector `Pv`.

```jl
julia> R = skewchol(A)
julia> R.Rm
4×4 LinearAlgebra.UpperTriangular{Float64, Matrix{Float64}}:
 2.82843  0.0      0.707107  -1.06066
  ⋅       2.82843  2.47487    0.353553
  ⋅        ⋅       1.06066    0.0
  ⋅        ⋅        ⋅         1.06066

julia> R.Jm
4×4 JMatrix{Float64, 1}:
   ⋅   1.0    ⋅    ⋅
 -1.0   ⋅     ⋅    ⋅
   ⋅    ⋅     ⋅   1.0
   ⋅    ⋅   -1.0   ⋅

julia> R.Pv
4-element Vector{Int64}:
 3
 2
 1
 4

 julia> transpose(R.Rm) * R.Jm * R.Rm ≈ A[R.Pv,R.Pv]
true
```

### Skew-Cholesky Reference

```@docs
SkewLinearAlgebra.skewchol
SkewLinearAlgebra.skewchol!
SkewLinearAlgebra.SkewCholesky
SkewLinearAlgebra.JMatrix
```