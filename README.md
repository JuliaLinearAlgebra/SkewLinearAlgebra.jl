# SkewLinearAlgebra
[`tr`](@ref)
This package proposes specialized functions for real skew-symmetric matrices i.e A=-A^T.
It provides the structure SkewSymmetric and the basic linear operations on such 
matrices. The package fits in the framework given by the package LinearAlgebra.

In particular, the package provides de following functions for A::SkewSymmetric :\
-Tridiagonal reduction: [`hessenberg`](@ref)\
-Eigensolvers: [`eigen`](@ref), [`eigvals`](@ref),[`eigmax`](@ref),[`eigmin`](@ref)\
-Trigonometric functions:[`cis`](@ref),[`cos`](@ref),[`sin`](@ref),[`tan`](@ref),[`sinh`](@ref),[`cosh`](@ref),[`tanh`](@ref)

The SkewSymmetric type uses the complete matrix as representation of its data. It doesn't verify automatically that the given matrix input is skew-symmetric. In particular, in-place methods could destroy the skew-symmetry. The provided function isskewsymmetric(A) allows to verify that A is indeed skew-symmetric.
