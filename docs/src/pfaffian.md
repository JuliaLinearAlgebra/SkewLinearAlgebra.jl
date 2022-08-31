# Pfaffian calculations

A real skew-symmetrix matrix $A = -A^T$ has a special property: its determinant is the square of a polynomial function of the
matrix entries, called the [Pfaffian](https://en.wikipedia.org/wiki/Pfaffian).   That is, $\mathrm{det}(A) = \mathrm{Pf}(A)^2$, but
knowing the Pfaffian itself (and its sign, which is lost in the determinant) is useful for a number of applications.

We provide a function `pfaffian(A)` to compute the Pfaffian of a real skew-symmetric matrix `A`.
```jl
julia> using SkewLinearAlgebra, LinearAlgebra

julia> A = [0 2 -7 4; -2 0 -8 3; 7 8 0 1;-4 -3 -1 0]
4×4 Matrix{Int64}:
  0   2  -7  4
 -2   0  -8  3
  7   8   0  1
 -4  -3  -1  0

julia> pfaffian(A)
-9.000000000000002

julia> det(A) # exact determinant is (-9)^2
80.99999999999999
```

By default, this computation is performed approximately using floating-point calculations, similar to Julia's [`det`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.det) algorithm for the determinant.  However, for a [`BigInt` matrix](https://docs.julialang.org/en/v1/base/numbers/#BigFloats-and-BigInts), `pfaffian(A)` is computed exactly using an algorithm by
[Galbiati and Maffioli (1994)](https://doi.org/10.1016/0166-218X(92)00034-J):

```jl
julia> pfaffian(BigInt.(A))
-9

julia> det(big.(A)) # also exact for BigInt
81
```

Note that you need not (but may) pass a `SkewHermitian` matrix type to `pfaffian`.  However, because the Pfaffian is only
defined for skew-symmetric matrices, it will give an error if you pass it a non-skewsymmetric matrix:

```jl
julia> pfaffian([1 2 3; 4 5 6; 7 8 9])
ERROR: ArgumentError: Pfaffian requires a skew-Hermitian matrix
```

We also provide a function `pfaffian!(A)` that overwrites `A` in-place (with undocumented values), rather than making a copy of
the matrix for intermediate calculations:
```jl
julia> pfaffian!(BigInt[0 2 -7; -2 0 -8; 7 8 0])
0
```
(Note that the Pfaffian is *always zero* for any *odd* size skew-symmetric matrix.)

Since the computation of the pfaffian can easily overflow/underflow the maximum/minimum representable floating-point value, we also provide a function `logabspfaffian` (along with an in-place variant `logabspfaffian!`) that returns a tuple `(logpf, sign)` such
that the Pfaffian is `sign * exp(logpf)`.   (This is similar to the [`logabsdet`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.logabsdet) function in Julia's `LinearAlgebra` library to compute the log of the determinant.)

```jl
julia> logpf, sign = logabspfaffian(A)
(2.1972245773362196, -1.0)

julia> sign * exp(logpf) # matches pfaffian(A), up to floating-point rounding errors
-9.000000000000002

julia> B = triu(rand(-9:9, 500,500)); B = B - B'; # 500×500 skew-symmetric matrix

julia> pfaffian(B) # overflows double-precision floating point
Inf

julia> pf = pfaffian(big.(B)) # exact answer in BigInt precision (slow)
-149678583522522720601879230931167588166888361287738234955688347466367975777696295859892310371985561723944757337655733584612691078889626269612647408920674699424393216780756729980039853434268507566870340916969614567968786613166601938742927283707974123631646016992038329261449437213872613766410239159659548127386325836018158542150965421313640795710036050440344289340146687857870477701301808699453999823930142237829465931054145755710674564378910415127367945223991977718726

julia> Float64(log(abs(pf))) # exactly rounded log(abs(pfaffian(B)))
1075.7105584607807

julia> logabspfaffian(B) # matches log and sign!
(1075.71055846078, -1.0)
```

### Pfaffian Reference

```@docs
SkewLinearAlgebra.pfaffian
SkewLinearAlgebra.pfaffian!
SkewLinearAlgebra.logabspfaffian
SkewLinearAlgebra.logabspfaffian!
```