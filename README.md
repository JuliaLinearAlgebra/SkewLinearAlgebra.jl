# SkewLinearAlgebra

To use this package, using the `LinearAlgebra` standard library is required.
```jl
using LinearAlgebra
using SkewLinearAlgebra
```
WARNING: Package still in development!
## SkewHermitian and SkewHermTridiagonal types

This package provides specialized algorithms for dense real skew-symmetric matrices i.e $A=-A^T$ and complex skew-hermitian matrices i.e $A=-A^*$.
It provides the matrix types `SkewHermitian` and `SkewHermTridiagonal`and implements the usual linear operations on such
matrices by extending functions from Julia's `LinearAlgebra` standard library, including optimized
algorithms that exploit this special matrix structure.

In particular, the package provides the following optimized functions for `SkewHermitian` and `SkewHermTridiagonal` matrices:

- Tridiagonal reduction: `hessenberg`
- Eigensolvers: `eigen`, `eigvals`
- SVD: `svd`, `svdvals`
- Trigonometric functions:`exp`, `cis`,`cos`,`sin`,`sinh`,`cosh`,`sincos`

Only for `SkewHermitian` matrices:
- Cholesky-like factorization: `skewchol`
- Pfaffian of real `SkewHermitian`: `pfaffian`, `logabspfaffian`

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

The `SkewHermTridiagonal(ev,dvim)`creates a abstract version of a tridiagonal skew-Hermitian matrix
where `ev` is the subdiagonal and `dvim` is a `Real` vector representing the pure imaginary diagonal of the matrix.
Real skew-symmetric matrices having zero diagonal elements, the constructor allows to only give the subdiagonal as argument.

Here is a basic example to initialize a `SkewHermTridiagonal`
```jl
julia> A=SkewHermTridiagonal(rand(ComplexF64,4), rand(5))
5×5 SkewHermTridiagonal{ComplexF64, Vector{ComplexF64}, Vector{Float64}}:
      0.0+0.150439im  -0.576265+0.23126im          0.0+0.0im             0.0+0.0im             0.0+0.0im
 0.576265+0.23126im         0.0+0.0833022im  -0.896415+0.6846im          0.0+0.0im             0.0+0.0im
      0.0+0.0im        0.896415+0.6846im           0.0+0.868229im  -0.593476+0.421484im        0.0+0.0im
      0.0+0.0im             0.0+0.0im         0.593476+0.421484im        0.0+0.995528im  -0.491818+0.32038im
      0.0+0.0im             0.0+0.0im              0.0+0.0im        0.491818+0.32038im         0.0+0.241177im
      
julia> SkewHermTridiagonal(randn(ComplexF32, 4))
5×5 SkewHermTridiagonal{ComplexF32, Vector{ComplexF32}, Nothing}:
       0.0+0.0im        0.343935+0.292369im         0.0+0.0im             0.0+0.0im             0.0+0.0im
 -0.343935+0.292369im        0.0+0.0im       -0.0961587-0.282884im        0.0+0.0im             0.0+0.0im
       0.0+0.0im       0.0961587-0.282884im         0.0+0.0im       -0.397075+0.518492im        0.0+0.0im
       0.0+0.0im             0.0+0.0im         0.397075+0.518492im        0.0+0.0im       -0.405492+0.679622im
       0.0+0.0im             0.0+0.0im              0.0+0.0im        0.405492+0.679622im        0.0+0.0im

julia> SkewHermTridiagonal(randn(4))
5×5 SkewHermTridiagonal{Float64, Vector{Float64}, Nothing}:
  0.0      1.93717    0.0        0.0       0.0
 -1.93717  0.0       -0.370536   0.0       0.0
  0.0      0.370536   0.0       -0.964014  0.0
  0.0      0.0        0.964014   0.0       1.33282
  0.0      0.0        0.0       -1.33282   0.0
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
The `hessenberg` function computes the Hessenberg decomposition of `A` and returns a `Hessenberg` object. If `F` is the
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

The package also provides eigensolvers for  `SkewHermitian` and `SkewHermTridiagonal` matrices. The method to solve the eigenvalue problem is based on the algorithm described in Penke et al, "[High Performance Solution of Skew-symmetric Eigenvalue Problems with Applications in Solving Bethe-Salpeter Eigenvalue Problem](https://arxiv.org/abs/1912.04062)" (2020).

The function `eigen` returns a `Eigen`structure as the LinearAlgebra standard library:
```jl
julia> E = eigen(A)
Eigen{ComplexF64, ComplexF64, Matrix{ComplexF64}, Vector{ComplexF64}}
values:
4-element Vector{ComplexF64}:
  0.0 + 11.934458713974193im
  0.0 + 0.7541188264752741im
 -0.0 - 0.7541188264752989im
 -0.0 - 11.934458713974236im
vectors:
4×4 Matrix{ComplexF64}:
    -0.49111+0.0im        -0.508735+0.0im           0.508735+0.0im           0.49111+0.0im
   -0.488014-0.176712im    0.471107+0.0931315im    -0.471107+0.0931315im    0.488014-0.176712im
   -0.143534+0.615785im    0.138561-0.284619im     -0.138561-0.284619im     0.143534+0.615785im
 -0.00717668-0.299303im  0.00692804-0.640561im   -0.00692804-0.640561im   0.00717668-0.299303im
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

 A specialized SVD using the eigenvalue decomposition is implemented for `SkewHermitian` and `SkewHermTridiagonal` type.
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
```

## Cholesky-like factorization

The package provides a Cholesky-like factorization for real skew-symmetric matrices as presented in P. Benner et al, "[Cholesky-like factorizations of skew-symmetric matrices](https://etna.ricam.oeaw.ac.at/vol.11.2000/pp85-93.dir/pp85-93.pdf)"(2000). 
Every real skew-symmetric matrix $A$ can be factorized as $A=P^TR^TJRP$ where $P$ is a permutation matrix, $R$ is an `UpperTriangular` matrix and J is of a special type called `JMatrix` that is a tridiagonal skew-symmetric matrix composed of diagonal blocks of the form $B=[0, 1; -1, 0]$. The `JMatrix` type implements efficient operations related to the shape of the matrix as matrix-matrix/vector multiplication and inversion. 
The function `skewchol` implements this factorization and returns a `SkewCholesky` structure composed of the matrices `Rm` and `Jm` of type `UpperTriangular` and `JMatrix` respectively. The permutation matrix $P$ is encoded as a permutation vector `Pv`.


```jl
julia> R = skewchol(A)
SkewCholesky{Float64, LinearAlgebra.UpperTriangular{var"#s24", S} where {var"#s24"<:Float64, S<:AbstractMatrix{var"#s24"}}, JMatrix{var"#s6", N, SGN} where {var"#s6"<:Float64, N<:Integer, SGN}, AbstractVector{var"#s3"} where var"#s3"<:Integer}([2.8284271247461903 0.0 0.7071067811865475 -1.0606601717798212; 0.0 2.8284271247461903 2.474873734152916 0.35355339059327373; 0.0 0.0 1.0606601717798216 0.0; 0.0 0.0 0.0 1.0606601717798216], [0.0 1.0 0.0 0.0; -1.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0; 0.0 0.0 -1.0 0.0], [3, 2, 1, 4])

julia> R.Rm
4×4 LinearAlgebra.UpperTriangular{Float64, Matrix{Float64}}:
 2.82843  0.0      0.707107  -1.06066
  ⋅       2.82843  2.47487    0.353553
  ⋅        ⋅       1.06066    0.0
  ⋅        ⋅        ⋅         1.06066

julia> R.Jm
4×4 JMatrix{Float64, Int64, Any}:
  0.0  1.0   0.0  0.0
 -1.0  0.0   0.0  0.0
  0.0  0.0   0.0  1.0
  0.0  0.0  -1.0  0.0

julia> R.Pv
4-element Vector{Int64}:
 3
 2
 1
 4
 
 julia> transpose(R.Rm) * R.Jm * R.Rm ≈ A[R.Pv,R.Pv]
true
```

## Pfaffian

The determinant of a real skew-Hermitian maxtrix is a perfect square. 
The pfaffian of A is a signed number such that `pfaffian(A)^2 = det(A)`.
Since the pfaffian may overflow, it may be convenient to compute the logarithm
of its absolute value. `logabspfaffian(A)` returns a tuple containing the logarithm 
of the absolute value of the pfaffian and the sign of the pfaffian.
```jl
julia> A = skewhermitian(rand(4,4))
4×4 SkewHermitian{Float64, Matrix{Float64}}:
 0.0       -0.133862  -0.356458  -0.405602
 0.133862   0.0        0.164159   0.117686
 0.356458  -0.164159   0.0        0.144498
 0.405602  -0.117686  -0.144498   0.0

julia> pfaffian(A)
-0.043975773548597816

julia> logabspfaffian(A)
(-3.124116397868594, -1.0)
```
