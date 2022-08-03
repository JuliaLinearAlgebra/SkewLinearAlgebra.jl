# SkewLinearAlgebra
[`tr`](@ref)
This package proposes specialized functions for real skew-symmetric matrices i.e $A=-A^T$.
It provides the structure SkewSymmetric and the basic linear operations on such 
matrices. The package fits in the framework given by the package LinearAlgebra.

In particular, the package provides de following functions for A [`SkewSymmetric`](@ref) :\
-Tridiagonal reduction: [`hessenberg`](@ref)\
-Eigensolvers: [`eigen`](@ref), [`eigvals`](@ref),[`eigmax`](@ref),[`eigmin`](@ref)\
-Trigonometric functions:[`cis`](@ref),[`cos`](@ref),[`sin`](@ref),[`tan`](@ref),[`sinh`](@ref),[`cosh`](@ref),[`tanh`](@ref)

The SkewSymmetric type uses the complete matrix as representation of its data. It doesn't verify automatically that the given matrix input is skew-symmetric. In particular, in-place methods could destroy the skew-symmetry. The provided function isskewsymmetric(A) allows to verify that A is indeed skew-symmetric.
```
julia> A = [0 2 -7 4; -2 0 -8 3; 7 8 0 1;-4 -3 -1 0]
3×3 Matrix{Int64}:
  0  2 -7  4
 -2  0 -8  3
  7  8  0  1
  -4 -3 -1 0
julia> A = SkewSymmetric(A)
julia> tr(A)
0

julia> det(A)
81.0

julia> inv(A)
4×4 Matrix{Float64}:
 -0.0        0.111111  -0.333333     -0.888889
 -0.111111   0.0        0.444444      0.777778
  0.333333  -0.444444   2.77556e-17   0.222222
  0.888889  -0.777778  -0.222222      0.0
  ```
