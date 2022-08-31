## Skew-Hermitian matrices

[`SkewHermitian(A)`](@ref) wraps an existing matrix `A`, which *must* already be skew-Hermitian,
in the `SkewHermitian` type (a subtype of `AbstractMatrix`), which supports fast specialized operations noted below.  You
can use the function [`isskewhermitian(A)`](@ref) to check whether `A` is skew-Hermitian (`A == -A'`).

`SkewHermitian(A)` *requires* that `A == -A'` and throws an error if it is not.
Alternatively, you can use the funcition (`skewhermitian(A)`](@ref) to take the skew-Hermitian *part*
of `A`, defined by `(A - A')/2`, and wrap it in a `SkewHermitian` view.  The
function [`skewhermitian!(A)`](@ref) does the same operation in-place on `A` (overwriting `A` with its
skew-Hermitian part).

Here is a basic example to initialize a `SkewHermitian` matrix:
```jl
julia> using SkewLinearAlgebra, LinearAlgebra

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
```

Basic linear-algebra operations are supported:
```jl
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

A `SkewHermitian` matrix `A` is simply a wrapper around an underlying matrix (in which both the upper and lower triangles are stored, despite the redundancy, to support fast matrix operations).  You can extract this underlying matrix with the Julia [`parent(A)`](https://docs.julialang.org/en/v1/base/arrays/#Base.parent) function (this does *not* copy the data:
mutating the parent will modify `A`).  Alternatively, you can *copy* the data to an ordinary `Matrix` ([2d `Array`](https://docs.julialang.org/en/v1/manual/arrays/)) with `Matrix(A)`.

### Operations on `SkewHermitian`

The `SkewHermitian` type supports the basic operations for any Julia `AbstractMatrix` (indexing, iteration, multiplication, addition, scaling, and so on).   Matrix–matrix and matrix–vector multiplications are performed using the underlying
parent matrix, so they are fast.

We try to preserve the `SkewHermitian` wrapper, if possible.  For example, adding two `SkewHermitian` matrices or scaling by a real number yields another `SkewHermitian` matrix.  Similarly for `real(A)`, `conj(A)`, `inv(A)`, or `-A`.

The package provides several optimized methods for `SkewHermitian` matrices,
based on the functions defined by the Julia `LinearAlgebra` package:

- Tridiagonal reduction: `hessenberg`
- Eigensolvers: `eigen`, `eigvals` (also `schur`, `svd`, `svdvals`)
- Trigonometric functions:`exp`, `cis`,`cos`,`sin`,`sinh`,`cosh`,`sincos`

We also define the following *new* functions for real skew-symmetric matrices only:
- Cholesky-like factorization: [`skewchol`](@ref)
- Pfaffian: [`pfaffian`](@ref), [`logabspfaffian`](@ref)

## Skew-Hermitian tridiagonal matrices

In the special case of a [tridiagonal](https://en.wikipedia.org/wiki/Tridiagonal_matrix) skew-Hermitian matrix,
many calculations can be performed very quickly, typically with $O(n)$ operations for an $n\times n$ matrix.
Such optimizations are supported by the `SkewLinearAlgebra` package using the `SkewHermTridiagonal` matrix type.

A complex tridiagonal skew-Hermitian matrix is of the form:
$
A=\left(\begin{array}{ccccc}
id_{1} & -e_{1}^{*}\\
e_{1} & id_{2} & -e_{2}^{*}\\
 & e_{2} & id_{3} & \ddots\\
 &  & \ddots & \ddots & -e_{n-1}^{*}\\
 &  &  & e_{n-1} & id_{n}
\end{array}\right)=-A^{*}
$
with purely imaginary diagonal entries $id_k$.   This s represented in the `SkewLinearAlgebra` by calling the `SkewHermTridiagonal(ev,dvim)` constructor, where `ev` is the (complex) vector of $n-1$ subdiagonal entries $e_k$
and `dvim` is the (real) vector of $n$ diagonal imaginary parts $d_k$:
```jl
julia> SkewHermTridiagonal([1+2im,3+4im],[5,6,7])
3×3 SkewHermTridiagonal{Complex{Int64}, Vector{Complex{Int64}}, Vector{Int64}}:
 0+5im  -1+2im     ⋅
 1+2im   0+6im  -3+4im
   ⋅     3+4im   0+7im
```
In the case of a *real* matrix, the diagonal entries are zero, and the matrix takes the form:
$
\text{real }A=\left(\begin{array}{ccccc}
0 & -e_{1}\\
e_{1} & 0 & -e_{2}\\
 & e_{2} & 0 & \ddots\\
 &  & \ddots & \ddots & -e_{n-1}\\
 &  &  & e_{n-1} & 0
\end{array}\right)=-A^{T}
$
In this case, you need not store the zero diagonal entries, and can simply call `SkewHermTridiagonal(ev)`
with the *real* vector `ev` of the $n-1$ subdiagonal entries:
```jl
julia> A = SkewHermTridiagonal([1,2,3])
4×4 SkewHermTridiagonal{Int64, Vector{Int64}, Nothing}:
 ⋅  -1   ⋅   ⋅
 1   ⋅  -2   ⋅
 ⋅   2   ⋅  -3
 ⋅   ⋅   3   ⋅
```
(Notice that zero values that are not stored (“structural zeros”) are shown as a `⋅`.)

A `SkewHermTridiagonal` matrix can also be converted to the [`LinearAlgebra.Tridiagonal`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.SymTridiagonal) type in the Julia standard library:
```jl
julia> Tridiagonal(A)
4×4 Tridiagonal{Int64, Vector{Int64}}:
 0  -1   ⋅   ⋅
 1   0  -2   ⋅
 ⋅   2   0  -3
 ⋅   ⋅   3   0
```
 which may support a wider range of linear-algebra functions, but does not optimized for the skew-Hermitian structure.


### Operations on `SkewHermTridiagonal`

The `SkewHermTridiagonal` type is modeled on the [`LinearAlgebra.SymTridiagonal`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.SymTridiagonal) type in the Julia standard library) and supports typical matrix operations
(indexing, iteration, scaling, and so on).

The package provides several optimized methods for `SkewHermTridiagonal` matrices,
based on the functions defined by the Julia `LinearAlgebra` package:

- Matrix-vector `A*x` and `dot(x,A,y)` products; also solves `A\x` (via conversion to `Tridiagonal`)
- Eigensolvers: `eigen`, `eigvals` (also `svd`, `svdvals`)
- Trigonometric functions:`exp`, `cis`,`cos`,`sin`,`sinh`,`cosh`,`sincos`

## Types Reference

```@docs
SkewLinearAlgebra.SkewHermitian
SkewLinearAlgebra.skewhermitian
SkewLinearAlgebra.skewhermitian!
SkewLinearAlgebra.SkewHermTridiagonal
```
