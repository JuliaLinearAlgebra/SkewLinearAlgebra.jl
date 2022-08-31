## Trigonometric functions

The package implements special methods of trigonometric matrix functions using our optimized eigenvalue decomposition for `SkewHermitian` and `SkewHermTridiagonal` matrices: `exp`, `cis`, `cos`, `sin`, `sincos`, `sinh`, and `cosh`.

For example:
```jl
julia> using SkewLinearAlgebra, LinearAlgebra

julia> A = SkewHermitian([0 2 -7 4; -2 0 -8 3; 7 8 0 1;-4 -3 -1 0]);

julia> Q = exp(A)
4×4 Matrix{Float64}:
-0.317791  -0.816528    -0.268647   0.400149
-0.697298   0.140338     0.677464   0.187414
0.578289  -0.00844255   0.40033    0.710807
0.279941  -0.559925     0.555524  -0.547275
```

Note that the exponential of a skew-Hermitian matrix is very special: it is unitary.  That is, if $A^* = -A$, then $(e^A)^* = (e^A)^{-1}$:
```jl
julia> Q' ≈ Q^-1
true
```

Several of the other matrix trigonometric functions also have special return types, in addition to being
optimized for performance.
```jl
julia> cis(A)
4×4 Hermitian{ComplexF64, Matrix{ComplexF64}}:
 36765.0+0.0im       36532.0+13228.5im  …   537.235+22406.2im
 36532.0-13228.5im   41062.9+0.0im          8595.76+22070.7im
 10744.7+46097.2im  -5909.58+49673.4im     -27936.2+7221.87im
 537.235-22406.2im   8595.76-22070.7im      13663.9+0.0im

julia> cos(A)
4×4 Symmetric{Float64, Matrix{Float64}}:
 36765.0    36532.0    10744.7      537.235
 36532.0    41062.9    -5909.58    8595.76
 10744.7    -5909.58   60940.6   -27936.2
   537.235   8595.76  -27936.2    13663.9

julia> cosh(A)
4×4 Hermitian{Float64, Matrix{Float64}}:
 0.766512      0.0374       0.011        0.000550001
 0.0374        0.770912    -0.00605001   0.00880001
 0.011        -0.00605001   0.791262    -0.0286
 0.000550001   0.00880001  -0.0286       0.742862
```
