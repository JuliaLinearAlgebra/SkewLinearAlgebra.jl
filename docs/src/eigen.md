## Skew-Hermitian eigenproblems

A skew-Hermitian matrix ``A = -A^*`` is very special with respect to its eigenvalues/vectors and related properties:

* It has **purely imaginary** eigenvalues.  (If ``A`` is real, these come in **± pairs** or are zero.)
* We can always find [orthonormal](https://en.wikipedia.org/wiki/Orthonormality) eigenvectors (``A`` is [normal](https://en.wikipedia.org/wiki/Normal_matrix)).

By wrapping a matrix in the [`SkewHermitian`](@ref) or [`SkewHermTridiagonal`](@ref) types, you can exploit optimized methods
for eigenvalue calculations (extending the functions defined in Julia's `LinearAlgebra` standard library).   Especially for *real*
skew-symmetric ``A=-A^T``, these optimized methods are generally *much faster* than the alternative of forming the complex-Hermitian
matrix ``iA``, computing its diagonalization, and multiplying the eigenvalues by ``-i``.

In particular, optimized methods are provided for [`eigen`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.eigen) (returning a factorization object storing both eigenvalues and eigenvectors), [`eigvals`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.eigvals) (just eigenvalues), [`eigvecs`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.eigvecs) (just eigenvectors), and their in-place variants [`eigen!`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.eigen!)/[`eigvals!`] (which overwrite the matrix data).

Since the [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition) and [Schur](https://en.wikipedia.org/wiki/Schur_decomposition) factorizations can be trivially computed from the eigenvectors/eigenvalues for any normal matrix, we also provide optimized methods for [`svd`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.svd), [`svdvals`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.svdvals), [`schur`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.schur), and their in-place variants [`svd!`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.svd!)/[`svdvals!`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.svdvals!)/[`schur!`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.schur!).

A key intermediate step in solving eigenproblems is computing the Hessenberg tridiagonal reduction of the matrix, and we expose this functionality by providing optimized [`hessenberg`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.hessenberg) and [`hessenberg!`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.hessenberg!) methods for `SkewHermitian` matrices as described below.   (The Hessenberg tridiagonalization is sometimes useful in its own right for matrix computations.)

 ### Eigenvalues and eigenvectors

The package also provides eigensolvers for  `SkewHermitian` and `SkewHermTridiagonal` matrices. A fast and sparse specialized QR algorithm is implemented for `SkewHermTridiagonal` matrices and also for `SkewHermitian` matrices using the `hessenberg` reduction.

The function `eigen` returns a `Eigen`structure as the LinearAlgebra standard library:
```jl
julia> using SkewLinearAlgebra, LinearAlgebra

julia> A = SkewHermitian([0 2 -7 4; -2 0 -8 3; 7 8 0 1;-4 -3 -1 0]);

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

 The function `eigvals` provides the eigenvalues of ``A``. The eigenvalues can be sorted and found partially with imaginary part in some given real range or by order.
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

 ### SVD

 A specialized SVD using the eigenvalue decomposition is implemented for `SkewHermitian` and `SkewHermTridiagonal` type.
 These functions can be called using the `LinearAlgebra` syntax.
```jl
julia> using SkewLinearAlgebra, LinearAlgebra

julia> A = SkewHermitian([0 2 -7 4; -2 0 -8 3; 7 8 0 1;-4 -3 -1 0]);

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

### Hessenberg tridiagonalization

The Hessenberg reduction performs a reduction ``A=QHQ^T`` where ``Q=\prod_i I-\tau_i v_iv_i^T`` is an orthonormal matrix.
The `hessenberg` function computes the Hessenberg decomposition of `A` and returns a `Hessenberg` object. If `F` is the
factorization object, the unitary matrix can be accessed with `F.Q` (of type `LinearAlgebra.HessenbergQ`)
and the Hessenberg matrix with `F.H` (of type `SkewHermTridiagonal`), either of
which may be converted to a regular matrix with `Matrix(F.H)` or `Matrix(F.Q)`.

```jl
julia> using SkewLinearAlgebra, LinearAlgebra

julia> A = SkewHermitian([0 2 -7 4; -2 0 -8 3; 7 8 0 1;-4 -3 -1 0]);

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
