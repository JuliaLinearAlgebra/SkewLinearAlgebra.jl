# SkewLinearAlgebra

This package proposes specialized functions for dense real skew-symmetric matrices i.e $A=-A^T$.
It provides the structure SkewSymmetric and the basic linear operations on such 
matrices. The package fits in the framework given by the LinearAlgebra package.

In particular, the package provides de following functions for $A::$ [`SkewSymmetric`](@ref) :\

-Tridiagonal reduction: [`hessenberg`](@ref)\
-Eigensolvers: [`eigen`](@ref), [`eigvals`](@ref),[`eigmax`](@ref),[`eigmin`](@ref)\
-Trigonometric functions:[`cis`](@ref),[`cos`](@ref),[`sin`](@ref),[`tan`](@ref),[`sinh`](@ref),[`cosh`](@ref),[`tanh`](@ref)

The SkewSymmetric type uses the complete matrix representation as data. It doesn't verify automatically that the given matrix input is skew-symmetric. In particular, in-place methods could destroy the skew-symmetry. The provided function isskewsymmetric(A) allows to verify that A is indeed skew-symmetric.
Here is a basic example to initialize a [`SkewSymmetric`](@ref
```
julia> A = [0 2 -7 4; -2 0 -8 3; 7 8 0 1;-4 -3 -1 0]
3×3 Matrix{Int64}:
  0  2 -7  4
 -2  0 -8  3
  7  8  0  1
  -4 -3 -1 0
julia> A = SkewSymmetric(A)
4×4 Matrix{Int64}:
  0   2  -7  4
 -2   0  -8  3
  7   8   0  1
 -4  -3  -1  0
 
julia> isskewsymmetric(A)
true

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
  
julia> x=[1;2;3;4]
4-element Vector{Int64}:
 1
 2
 3
 4

julia> A\x
4-element Vector{Float64}:
 -4.333333333333334
  4.333333333333334
  0.3333333333333336
 -1.3333333333333333
  ```
  
  The functions from the LinearAlgebra package can be used in the same fashion:
  ```
julia> hessenberg(A)
4×4 Tridiagonal{Float64, Vector{Float64}}:
 0.0      -8.30662    ⋅        ⋅
 8.30662   0.0      -8.53382   ⋅
  ⋅        8.53382   0.0      1.08347
  ⋅         ⋅       -1.08347  0.0
3×3 UnitLowerTriangular{Float64, SubArray{Float64, 2, Matrix{Float64}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}}:
  1.0         ⋅         ⋅ 
 -0.679175   1.0        ⋅
  0.3881    -0.185238  1.0
2-element Vector{Float64}:
 1.2407717061715382
 1.9336503566410876
  
 julia> eigvals(A)
4-element Vector{ComplexF64}:
  0.0 + 11.93445871397423im
  0.0 + 0.7541188264752853im
 -0.0 - 0.7541188264752877im
 -0.0 - 11.934458713974225im
 
 julia> eigmax(A)
-0.0 - 11.934458713974223im
  
  ```
  \usepackage{amsymb}\
The hessenberg reduction performs a reduction $A=QHQ^T$ where $Q=\prod_i I-\tau_i v_iv_i^T$\
The hessenberg function returns a structure of type [`SkewHessenberg`](@ref) containing the [`Tridiagonal`](@ref) reduction $H\in \mathbb{R}^{n\times n}$, the householder reflectors $v_i$ in a  [`UnitLowerTriangular`](@ref) $V\in \mathbb{R}^{n-1\times n-1}$  and the $n-2$ scalars $\tau_i$ associated to the reflectors. A function [`getQ`](@ref) is provided to retrieve the orthogonal transformation Q. 

  ```
  julia> H=hessenberg(A)
4×4 Tridiagonal{Float64, Vector{Float64}}:
 0.0      -8.30662    ⋅        ⋅
 8.30662   0.0      -8.53382   ⋅
  ⋅        8.53382   0.0      1.08347
  ⋅         ⋅       -1.08347  0.0
3×3 UnitLowerTriangular{Float64, SubArray{Float64, 2, Matrix{Float64}, Tuple{UnitRange{Int64}, UnitRange{Int64}}, false}}:
  1.0         ⋅         ⋅ 
 -0.679175   1.0        ⋅
  0.3881    -0.185238  1.0
2-element Vector{Float64}:
 1.2407717061715382
 1.9336503566410876
 
 julia> Q=getQ(H)
4×4 Matrix{Float64}:
 1.0   0.0        0.0         0.0
 0.0  -0.240772  -0.95927    -0.14775
 0.0   0.842701  -0.282138    0.458534
 0.0  -0.481543  -0.0141069   0.876309
 
 ```
  
  
  
