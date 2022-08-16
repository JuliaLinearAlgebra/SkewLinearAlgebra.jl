# SkewLinearAlgebra

WARNING: Package still in development!

This package provides specialized algorithms for dense real skew-symmetric matrices i.e $A=-A^T$ and skew-hermitian matrices i.e $A=-A^*$.
It provides the matrix types `SkewHermitian` and `SkewHermTridiagonal`and implements the usual linear operations on such
matrices by extending functions from Julia's `LinearAlgebra` standard library, including optimized
algorithms that exploit this special matrix structure.

In particular, the package provides the following optimized functions for `SkewHermitian` matrices:

- Tridiagonal reduction: `hessenberg`
- Eigensolvers: `eigen`, `eigvals`
- SVD: `svd`, `svdvals`
- Trigonometric functions:`exp`, `cis`,`cos`,`sin`,`tan`,`sinh`,`cosh`,`tanh`
- Cholesky-like factorization: `skewchol`

(Currently, we only provide specialized algorithms for real skew-Hermitian/skew-symmetric matrices.
Methods for complex skew-Hermitian matrices transform these at negligible cost in complex `Hermitian` 
matrices by multiplying by $i$. This allows to use efficient LAPACK algorithms for hermitian matrices.
Note, however that for real skew-Hermitian matrices this would force you to use complex arithmetic.  
Hence, the benefits of specialized algorithms are greatest for real skew-Hermitian matrices.)

The `SkewHermitian(A)` wraps an existing matrix `A`, which *must* already be skew-Hermitian,
in the `SkewHermitian` type, which supports fast specialized operations noted above.  You
can use the function `isskewhermitian(A)` to check whether `A` is skew-Hermitian (`A == -A'`).

Alternatively, you can use the funcition `skewhermitian(A)` to take the skew-Hermitian *part*
of `A`, given by `(A - A')/2`, and wrap it in a `SkewHermitian` view.  Alternatively, the
function `skewhermitian!(A)` does the same operation in-place on `A`.

Here is a basic example to initialize a `SkewHermitian`
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
```jl
julia> hessenberg(A)
Hessenberg{Float64, SkewHermTridiagonal{Float64, Vector{Float64}, Nothing}, Matrix{Float64}, Vector{Float64}, Bool}
Q factor:
4×4 LinearAlgebra.HessenbergQ{Float64, Matrix{Float64}, Vector{Float64}, true}:
 1.0   0.0        0.0         0.0
 0.0  -0.240772  -0.95927    -0.14775
 0.0   0.842701  -0.282138    0.458534
 0.0  -0.481543  -0.0141069   0.876309
H factor:
4×4 SkewHermTridiagonal{Float64, Vector{Float64}, Nothing}:
 0.0      -8.30662   0.0       0.0 
 8.30662   0.0      -8.53382   0.0 
 0.0       8.53382   0.0       1.08347
 0.0       0.0      -1.08347   0.0

 julia> eigvals(A)
4-element Vector{ComplexF64}:
  0.0 + 11.93445871397423im
  0.0 + 0.7541188264752853im
 -0.0 - 0.7541188264752877im
 -0.0 - 11.934458713974225im

```

 ## Hessenberg/Tridiagonal reduction
The Hessenberg reduction performs a reduction $A=QHQ^T$ where $Q=\prod_i I-\tau_i v_iv_i^T$ is an orthonormal matrix.
The `hessenberg` function computes the Hessenberg decomposition of `A` and return a `Hessenberg` object. If `F` is the
factorization object, the unitary matrix can be accessed with `F.Q` (of type `LinearAlgebra.HessenbergQ`)
and the Hessenberg matrix with `F.H` (of type `SkewHermTridiagonal`), either of
which may be converted to a regular matrix with `Matrix(F.H)` or `Matrix(F.Q)`.

```jl
julia> hessenberg(A)
Hessenberg{Float64, Tridiagonal{Float64, Vector{Float64}}, Matrix{Float64}, Vector{Float64}, Bool}
Q factor:
4×4 LinearAlgebra.HessenbergQ{Float64, Matrix{Float64}, Vector{Float64}, true}:
 1.0   0.0        0.0         0.0
 0.0  -0.240772  -0.95927    -0.14775
 0.0   0.842701  -0.282138    0.458534
 0.0  -0.481543  -0.0141069   0.876309
H factor:
4×4 SkewHermTridiagonal{Float64, Vector{Float64}, Nothing}:
 0.0      -8.30662   0.0       0.0 
 8.30662   0.0      -8.53382   0.0 
 0.0       8.53382   0.0       1.08347
 0.0       0.0      -1.08347   0.0
```

 ## Eigenvalues and eigenvectors

 The package also provides eigensolvers for  `SkewHermitian` matrices. The method to solve the eigenvalue problem is based on the algorithm described in Penke et al, "[High Performance Solution of Skew-symmetric Eigenvalue Problems with Applications in Solving Bethe-Salpeter Eigenvalue Problem](https://arxiv.org/abs/1912.04062)" (2020).

The function `eigen` returns the eigenvalues plus the real part of the eigenvectors and the imaginary part separeted.
```jl
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
 The function `eigvals` provides the eigenvalues of $A$. The eigenvalues can be sorted and found partially with imaginary part in some given real range or by order.
```jl
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

 A specialized SVD using the eigenvalue decomposition is implemented for `SkewSymmetric` type.
 These functions can be called using the `LinearAlgebra` syntax.

```jl
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

 The package implements special versions of the trigonometric functions using the eigenvalue decomposition. The provided functions are `exp`, `cis`,`cos`,`sin`,`tan`,`sinh`,`cosh`,`tanh`.
```jl
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
```jl




