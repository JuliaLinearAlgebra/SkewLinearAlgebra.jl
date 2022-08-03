# SkewLinearAlgebra
This package proposes specialized functions for real skew-symmetric matrices i.e A=-A^T.
It provides the structure SkewSymmetric and the basic linear operations on such 
matrices. The package fits in the framework given by the package LinearAlgebra.

In particular, the package provides de following functions for A::SkewSymmetric :\
-Tridiagonal reduction: hessenberg(A)\
-Eigensolvers: eigen(A), eigvals(A),eigmin(A),eigmax(A)\
-Trigonometric functions: cis(A),cos(A),sin(A),tan(A),sinh(A),cosh(A),tanh(A)

The SkewSymmetric type uses the complete matrix as representation of its data. It doesn't verify automatically that the given matrix input is skew-symmetric. In particular, in-place methods could destroy the skew-symmetry. The provided function isskewsymmetric(A) allows to verify that A is indeed skew-symmetric.
