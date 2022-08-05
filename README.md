# SkewLinearAlgebra

WARNING: Package still in development!

This package provides specialized functions for dense real skew-symmetric matrices i.e $A=-A^T$.
It provides the type [`SkewHermitian`](@ref) and the classical linear operations on such
matrices. The package fits in the framework given by the LinearAlgebra package.

In particular, the package provides de following functions for $A::$ [`SkewHermitian`](@ref) :

-Tridiagonal reduction: [`hessenberg`](@ref)\
-Eigensolvers: [`eigen`](@ref), [`eigvals`](@ref),[`eigmax`](@ref),[`eigmin`](@ref)\
-SVD: [`svd`](@ref), [`svdvals`](@ref)\
-Trigonometric functions:[`exp`](@ref), [`cis`](@ref),[`cos`](@ref),[`sin`](@ref),[`tan`](@ref),[`sinh`](@ref),[`cosh`](@ref),[`tanh`](@ref)

The `SkewHermitian(A)` wraps an existing matrix `A`, which *must* already be skew-Hermitian,
in the `SkewHermitian` type, which supports fast specialized operations noted above.  You
can use the function `isskewhermitian(A)` to check whether `A` is skew-Hermitian (`A == -A'`).

Alternatively, you can use the funcition `skewhermitian(A)` to take the skew-Hermitian *part*
of `A`, given by `(A - A')/2`, and wrap it in a `SkewHermitian` view.  Alternatively, the
function `skewhermitian!(A)` does the same operation in-place on `A`.

Here is a basic example to initialize a [`SkewHermitian`](@ref)
```jl
julia> A = [0 2 -7 4; -2 0 -8 3; 7 8 0 1;-4 -3 -1 0]
3×3 Matrix{Int64}:
  0  2 -7  4
 -2  0 -8  3
  7  8  0  1
  -4 -3 -1 0

julia> isskewhermitian(A)
true

julia> A = SkewHermitian(A)
4×4 SkewHermitian{Int64, Matrix{Int64}}:
  0   2  -7  4
 -2   0  -8  3
  7   8   0  1
 -4  -3  -1  0

julia> tr(A)
0

julia> det(A)
81.0

julia> inv(A)
4×4 SkewHermitian{Float64, Matrix{Float64}}:
  0.0        0.111111  -0.333333  -0.888889
 -0.111111   0.0        0.444444   0.777778
  0.333333  -0.444444   0.0        0.222222
  0.888889  -0.777778  -0.222222   0.0

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
 ## Hessenberg/tridiagonal reduction
The hessenberg reduction performs a reduction $A=QHQ^T$ where $Q=\prod_i I-\tau_i v_iv_i^T$ is an orthonormal matrix.
The [`hessenberg`](@ref) function returns a structure of type [`SkewHessenberg`](@ref) containing the [`Tridiagonal`](@ref) reduction $H\in \mathbb{R}^{n\times n}$, the householder reflectors $v_i$ in a  [`UnitLowerTriangular`](@ref) $V\in \mathbb{R}^{n-1\times n-1}$  and the $n-2$ scalars $\tau_i$ associated to the reflectors. A function [`getQ`](@ref) is provided to retrieve the orthogonal transformation Q.

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
\usepackage{hyperref}

 ## Eigenvalues and eigenvectors

 The package also provides eigensolvers for  [`SkewHermitian`](@ref) matrices. The method to solve the eigenvalue problem is issued from: \url{C.Penke,A.Marek,C.Vorwerk,C.Draxl,P.Benner, \textit{ High Performance Solution of Skew-symmetric Eigenvalue Problems with Applications in Solving Bethe-Salpeter Eigenvalue Problem}, 2020.}

  The function [`eigen`](@ref) returns the eigenvalues plus the real part of the eigenvectors and the imaginary part separeted.
 ```
  julia> val,Qr,Qim=eigen(A)

  julia> val
4-element Vector{ComplexF64}:
  0.0 + 11.934458713974193im
  0.0 + 0.7541188264752758im
 -0.0 - 0.7541188264752989im
 -0.0 - 11.93445871397423im

julia> Qr
4×4 Matrix{Float64}:
 -0.49111     -0.508735     0.508735    0.49111
 -0.488014     0.471107    -0.471107    0.488014
 -0.143534     0.138561    -0.138561    0.143534
 -0.00717668   0.00692804  -0.00692804  0.00717668

julia> Qim
4×4 Matrix{Float64}:
  0.0        0.0         0.0         0.0
 -0.176712   0.0931315   0.0931315  -0.176712
  0.615785  -0.284619   -0.284619    0.615785
 -0.299303  -0.640561   -0.640561   -0.299303

 ```
 The function [`eigvals`](@ref) provides de eigenvalues of $A$. The eigenvalues can be sorted and found partially with imaginary part in some given real range or by order.
 ```
 julia> eigvals(A)
4-element Vector{ComplexF64}:
  0.0 + 11.93445871397423im
  0.0 + 0.7541188264752853im
 -0.0 - 0.7541188264752877im
 -0.0 - 11.934458713974225im

julia> eigvals(A,0,15)
2-element Vector{ComplexF64}:
 0.0 + 11.93445871397414im
 0.0 + 0.7541188264752858im

julia> eigvals(A,1:3)
3-element Vector{ComplexF64}:
  0.0 + 11.93445871397423im
  0.0 + 0.7541188264752989im
 -0.0 - 0.7541188264752758im
 ```
 ## SVD

 Specialized SVD using the eigenvalue decomposition is implemented for [`SkewSymmetric`](@ref) type.
 These functions can be used using the LinearAlgebra syntax.

 ```
 julia> svd(A)
SVD{ComplexF64, Float64, Matrix{ComplexF64}}
U factor:
4×4 Matrix{ComplexF64}:
    0.49111+0.0im          -0.49111+0.0im          0.508735+0.0im         -0.508735+0.0im
   0.488014-0.176712im    -0.488014-0.176712im    -0.471107+0.0931315im    0.471107+0.0931315im
   0.143534+0.615785im    -0.143534+0.615785im    -0.138561-0.284619im     0.138561-0.284619im
 0.00717668-0.299303im  -0.00717668-0.299303im  -0.00692804-0.640561im   0.00692804-0.640561im
singular values:
4-element Vector{Float64}:
 11.93445871397423
 11.934458713974193
  0.7541188264752989
  0.7541188264752758
Vt factor:
4×4 Matrix{ComplexF64}:
 0.0-0.49111im     0.176712-0.488014im  -0.615785-0.143534im   0.299303-0.00717668im
 0.0-0.49111im    -0.176712-0.488014im   0.615785-0.143534im  -0.299303-0.00717668im
 0.0-0.508735im  -0.0931315+0.471107im   0.284619+0.138561im   0.640561+0.00692804im
 0.0-0.508735im   0.0931315+0.471107im  -0.284619+0.138561im  -0.640561+0.00692804im

 julia> svdvals(A)
4-element Vector{Float64}:
 11.93445871397423
 11.934458713974225
  0.7541188264752877
  0.7541188264752853

 ```
 ## Trigonometric functions

 The package implements special versions of the trigonometric functions using the eigenvalue decomposition. The provided functions are [`exp`](@ref), [`cis`](@ref),[`cos`](@ref),[`sin`](@ref),[`tan`](@ref),[`sinh`](@ref),[`cosh`](@ref),[`tanh`](@ref).
 ```
 julia> exp(A)
4×4 Matrix{Float64}:
 -0.317791  -0.816528    -0.268647   0.400149
 -0.697298   0.140338     0.677464   0.187414
  0.578289  -0.00844255   0.40033    0.710807
  0.279941  -0.559925     0.555524  -0.547275

 julia> cis(A)
4×4 Matrix{ComplexF64}:
   5.95183+0.0im       3.21734+1.80074im     -0.658082-3.53498im      -1.4454+5.61775im
   3.21734-1.80074im   4.00451+1.0577e-17im   -1.42187-1.41673im     0.791701+4.77348im
 -0.658082+3.53498im  -1.42187+1.41673im       2.89938+7.7327e-18im  -2.69134-1.61285im
   -1.4454-5.61775im  0.791701-4.77348im      -2.69134+1.61285im      6.92728+2.40436e-16im

julia> cos(A)
4×4 Matrix{Float64}:
  5.95183    3.21734   -0.658082  -1.4454
  3.21734    4.00451   -1.42187    0.791701
 -0.658082  -1.42187    2.89938   -2.69134
 -1.4454     0.791701  -2.69134    6.92728

julia> cosh(A)
4×4 Matrix{Float64}:
 -0.317791  -0.756913  0.154821   0.340045
 -0.756913   0.140338  0.334511  -0.186256
  0.154821   0.334511  0.40033    0.633165
  0.340045  -0.186256  0.633165  -0.547275

 ```




