var documenterSearchIndex = {"docs":
[{"location":"skewchol/#Skew-Cholesky-factorization","page":"Skew-Cholesky","title":"Skew-Cholesky factorization","text":"","category":"section"},{"location":"skewchol/","page":"Skew-Cholesky","title":"Skew-Cholesky","text":"The package provides a Cholesky-like factorization for real skew-symmetric matrices as presented in P. Benner et al, \"Cholesky-like factorizations of skew-symmetric matrices\"(2000). Every real skew-symmetric matrix A can be factorized as A=P^TR^TJRP where P is a permutation matrix, R is an UpperTriangular matrix and J is of a special type called JMatrix that is a tridiagonal skew-symmetric matrix composed of diagonal blocks of the form B=0 1 -1 0. The JMatrix type implements efficient operations related to the shape of the matrix as matrix-matrix/vector multiplication and inversion. The function skewchol implements this factorization and returns a SkewCholesky structure composed of the matrices Rm and Jm of type UpperTriangular and JMatrix respectively. The permutation matrix P is encoded as a permutation vector Pv.","category":"page"},{"location":"skewchol/","page":"Skew-Cholesky","title":"Skew-Cholesky","text":"julia> R = skewchol(A)\njulia> R.Rm\n4×4 LinearAlgebra.UpperTriangular{Float64, Matrix{Float64}}:\n 2.82843  0.0      0.707107  -1.06066\n  ⋅       2.82843  2.47487    0.353553\n  ⋅        ⋅       1.06066    0.0\n  ⋅        ⋅        ⋅         1.06066\n\njulia> R.Jm\n4×4 JMatrix{Float64, 1}:\n   ⋅   1.0    ⋅    ⋅\n -1.0   ⋅     ⋅    ⋅\n   ⋅    ⋅     ⋅   1.0\n   ⋅    ⋅   -1.0   ⋅\n\njulia> R.Pv\n4-element Vector{Int64}:\n 3\n 2\n 1\n 4\n\n julia> transpose(R.Rm) * R.Jm * R.Rm ≈ A[R.Pv,R.Pv]\ntrue","category":"page"},{"location":"skewchol/#Skew-Cholesky-Reference","page":"Skew-Cholesky","title":"Skew-Cholesky Reference","text":"","category":"section"},{"location":"skewchol/","page":"Skew-Cholesky","title":"Skew-Cholesky","text":"SkewLinearAlgebra.skewchol\nSkewLinearAlgebra.skewchol!\nSkewLinearAlgebra.SkewCholesky\nSkewLinearAlgebra.JMatrix","category":"page"},{"location":"skewchol/#SkewLinearAlgebra.skewchol","page":"Skew-Cholesky","title":"SkewLinearAlgebra.skewchol","text":"skewchol(A)\n\nComputes a Cholesky-like factorization of the real skew-symmetric matrix A. The function returns a SkewCholesky structure composed of three fields: R,J,p. R is UpperTriangular, J is a JMatrix, p is an array of integers. Let S be the returned structure, then the factorization is such that S.R'*S.J*S.R = A[S.p,S.p]\n\nThis factorization (and the underlying algorithm) is described in from P. Benner et al, \"Cholesky-like factorizations of skew-symmetric matrices\"(2000).\n\n\n\n\n\n","category":"function"},{"location":"skewchol/#SkewLinearAlgebra.skewchol!","page":"Skew-Cholesky","title":"SkewLinearAlgebra.skewchol!","text":"skewchol!(A)\n\nSimilar to skewchol!, but overwrites A in-place with intermediate calculations.\n\n\n\n\n\n","category":"function"},{"location":"skewchol/#SkewLinearAlgebra.SkewCholesky","page":"Skew-Cholesky","title":"SkewLinearAlgebra.SkewCholesky","text":"SkewCholesky(R,p)\n\nConstruct a SkewCholesky structure from the UpperTriangular matrix R and the permutation vector p. A matrix J of type JMatrix is build calling this function. The SkewCholesky structure has three arguments: R,J and p.\n\n\n\n\n\n","category":"type"},{"location":"skewchol/#SkewLinearAlgebra.JMatrix","page":"Skew-Cholesky","title":"SkewLinearAlgebra.JMatrix","text":"JMatrix{T, ±1}(n) Creates an AbstractMatrix{T} of size n x n, representing a block-diagonal matrix whose diagonal blocks are ±[0 1; -1 0]. If n is odd, then the last block is the 1 x 1 zero block. The ±1 parameter allows us to transpose and invert the matrix, and corresponds to an overall multiplicative sign factor.\n\n\n\n\n\n","category":"type"},{"location":"eigen/#Skew-Hermitian-eigenproblems","page":"Eigenproblems","title":"Skew-Hermitian eigenproblems","text":"","category":"section"},{"location":"eigen/","page":"Eigenproblems","title":"Eigenproblems","text":"A skew-Hermitian matrix A = -A^* is very special with respect to its eigenvalues/vectors and related properties:","category":"page"},{"location":"eigen/","page":"Eigenproblems","title":"Eigenproblems","text":"It has purely imaginary eigenvalues.  (If A is real, these come in ± pairs or are zero.)\nWe can always find orthonormal eigenvectors (A is normal).","category":"page"},{"location":"eigen/","page":"Eigenproblems","title":"Eigenproblems","text":"By wrapping a matrix in the SkewHermitian or SkewHermTridiagonal types, you can exploit optimized methods for eigenvalue calculations (extending the functions defined in Julia's LinearAlgebra standard library).   Especially for real skew-symmetric A=-A^T, these optimized methods are generally much faster than the alternative of forming the complex-Hermitian matrix iA, computing its diagonalization, and multiplying the eigenvalues by -i.","category":"page"},{"location":"eigen/","page":"Eigenproblems","title":"Eigenproblems","text":"In particular, optimized methods are provided for eigen (returning a factorization object storing both eigenvalues and eigenvectors), eigvals (just eigenvalues), eigvecs (just eigenvectors), and their in-place variants eigen!/eigvals!.","category":"page"},{"location":"eigen/","page":"Eigenproblems","title":"Eigenproblems","text":"Since the SVD and Schur factorizations can be trivially computed from the eigenvectors/eigenvalues for any normal matrix, we also provide optimized methods for svd, svdvals, schur, and their in-place variants svd!/svdvals!/schur!.","category":"page"},{"location":"eigen/","page":"Eigenproblems","title":"Eigenproblems","text":"A key intermediate step in solving eigenproblems is computing the Hessenberg tridiagonal reduction of the matrix, and we expose this functionality by providing optimized hessenberg and hessenberg! methods for SkewHermitian matrices as described below.   (The Hessenberg tridiagonalization is sometimes useful in its own right for matrix computations.)","category":"page"},{"location":"eigen/#Eigenvalues-and-eigenvectors","page":"Eigenproblems","title":"Eigenvalues and eigenvectors","text":"","category":"section"},{"location":"eigen/","page":"Eigenproblems","title":"Eigenproblems","text":"The package also provides eigensolvers for  SkewHermitian and SkewHermTridiagonal matrices. A fast and sparse specialized QR algorithm is implemented for SkewHermTridiagonal matrices and also for SkewHermitian matrices using the hessenberg reduction.","category":"page"},{"location":"eigen/","page":"Eigenproblems","title":"Eigenproblems","text":"The function eigen returns a Eigenstructure as the LinearAlgebra standard library:","category":"page"},{"location":"eigen/","page":"Eigenproblems","title":"Eigenproblems","text":"julia> using SkewLinearAlgebra, LinearAlgebra\n\njulia> A = SkewHermitian([0 2 -7 4; -2 0 -8 3; 7 8 0 1;-4 -3 -1 0]);\n\njulia> E = eigen(A)\nEigen{ComplexF64, ComplexF64, Matrix{ComplexF64}, Vector{ComplexF64}}\nvalues:\n4-element Vector{ComplexF64}:\n  0.0 + 11.934458713974193im\n  0.0 + 0.7541188264752741im\n -0.0 - 0.7541188264752989im\n -0.0 - 11.934458713974236im\nvectors:\n4×4 Matrix{ComplexF64}:\n    -0.49111+0.0im        -0.508735+0.0im           0.508735+0.0im           0.49111+0.0im\n   -0.488014-0.176712im    0.471107+0.0931315im    -0.471107+0.0931315im    0.488014-0.176712im\n   -0.143534+0.615785im    0.138561-0.284619im     -0.138561-0.284619im     0.143534+0.615785im\n -0.00717668-0.299303im  0.00692804-0.640561im   -0.00692804-0.640561im   0.00717668-0.299303im","category":"page"},{"location":"eigen/","page":"Eigenproblems","title":"Eigenproblems","text":"The function eigvals provides the eigenvalues of A. The eigenvalues can be sorted and found partially with imaginary part in some given real range or by order.","category":"page"},{"location":"eigen/","page":"Eigenproblems","title":"Eigenproblems","text":" julia> eigvals(A)\n4-element Vector{ComplexF64}:\n  0.0 + 11.93445871397423im\n  0.0 + 0.7541188264752853im\n -0.0 - 0.7541188264752877im\n -0.0 - 11.934458713974225im\n\njulia> eigvals(A,0,15)\n2-element Vector{ComplexF64}:\n 0.0 + 11.93445871397414im\n 0.0 + 0.7541188264752858im\n\njulia> eigvals(A,1:3)\n3-element Vector{ComplexF64}:\n  0.0 + 11.93445871397423im\n  0.0 + 0.7541188264752989im\n -0.0 - 0.7541188264752758im","category":"page"},{"location":"eigen/#SVD","page":"Eigenproblems","title":"SVD","text":"","category":"section"},{"location":"eigen/","page":"Eigenproblems","title":"Eigenproblems","text":"A specialized SVD using the eigenvalue decomposition is implemented for SkewHermitian and SkewHermTridiagonal type.  These functions can be called using the LinearAlgebra syntax.","category":"page"},{"location":"eigen/","page":"Eigenproblems","title":"Eigenproblems","text":"julia> using SkewLinearAlgebra, LinearAlgebra\n\njulia> A = SkewHermitian([0 2 -7 4; -2 0 -8 3; 7 8 0 1;-4 -3 -1 0]);\n\n julia> svd(A)\nSVD{ComplexF64, Float64, Matrix{ComplexF64}}\nU factor:\n4×4 Matrix{ComplexF64}:\n    0.49111+0.0im          -0.49111+0.0im          0.508735+0.0im         -0.508735+0.0im\n   0.488014-0.176712im    -0.488014-0.176712im    -0.471107+0.0931315im    0.471107+0.0931315im\n   0.143534+0.615785im    -0.143534+0.615785im    -0.138561-0.284619im     0.138561-0.284619im\n 0.00717668-0.299303im  -0.00717668-0.299303im  -0.00692804-0.640561im   0.00692804-0.640561im\nsingular values:\n4-element Vector{Float64}:\n 11.93445871397423\n 11.934458713974193\n  0.7541188264752989\n  0.7541188264752758\nVt factor:\n4×4 Matrix{ComplexF64}:\n 0.0-0.49111im     0.176712-0.488014im  -0.615785-0.143534im   0.299303-0.00717668im\n 0.0-0.49111im    -0.176712-0.488014im   0.615785-0.143534im  -0.299303-0.00717668im\n 0.0-0.508735im  -0.0931315+0.471107im   0.284619+0.138561im   0.640561+0.00692804im\n 0.0-0.508735im   0.0931315+0.471107im  -0.284619+0.138561im  -0.640561+0.00692804im\n\n julia> svdvals(A)\n4-element Vector{Float64}:\n 11.93445871397423\n 11.934458713974225\n  0.7541188264752877\n  0.7541188264752853","category":"page"},{"location":"eigen/#Hessenberg-tridiagonalization","page":"Eigenproblems","title":"Hessenberg tridiagonalization","text":"","category":"section"},{"location":"eigen/","page":"Eigenproblems","title":"Eigenproblems","text":"The Hessenberg reduction performs a reduction A=QHQ^T where Q=prod_i I-tau_i v_iv_i^T is an orthonormal matrix. The hessenberg function computes the Hessenberg decomposition of A and returns a Hessenberg object. If F is the factorization object, the unitary matrix can be accessed with F.Q (of type LinearAlgebra.HessenbergQ) and the Hessenberg matrix with F.H (of type SkewHermTridiagonal), either of which may be converted to a regular matrix with Matrix(F.H) or Matrix(F.Q).","category":"page"},{"location":"eigen/","page":"Eigenproblems","title":"Eigenproblems","text":"julia> using SkewLinearAlgebra, LinearAlgebra\n\njulia> A = SkewHermitian([0 2 -7 4; -2 0 -8 3; 7 8 0 1;-4 -3 -1 0]);\n\njulia> hessenberg(A)\nHessenberg{Float64, Tridiagonal{Float64, Vector{Float64}}, Matrix{Float64}, Vector{Float64}, Bool}\nQ factor:\n4×4 LinearAlgebra.HessenbergQ{Float64, Matrix{Float64}, Vector{Float64}, true}:\n 1.0   0.0        0.0         0.0\n 0.0  -0.240772  -0.95927    -0.14775\n 0.0   0.842701  -0.282138    0.458534\n 0.0  -0.481543  -0.0141069   0.876309\nH factor:\n4×4 SkewHermTridiagonal{Float64, Vector{Float64}, Nothing}:\n 0.0      -8.30662   0.0       0.0\n 8.30662   0.0      -8.53382   0.0\n 0.0       8.53382   0.0       1.08347\n 0.0       0.0      -1.08347   0.0","category":"page"},{"location":"pfaffian/#Pfaffian-calculations","page":"Pfaffians","title":"Pfaffian calculations","text":"","category":"section"},{"location":"pfaffian/","page":"Pfaffians","title":"Pfaffians","text":"A real skew-symmetrix matrix A = -A^T has a special property: its determinant is the square of a polynomial function of the matrix entries, called the Pfaffian.   That is, mathrmdet(A) = mathrmPf(A)^2, but knowing the Pfaffian itself (and its sign, which is lost in the determinant) is useful for a number of applications.","category":"page"},{"location":"pfaffian/","page":"Pfaffians","title":"Pfaffians","text":"We provide a function pfaffian(A) to compute the Pfaffian of a real skew-symmetric matrix A.","category":"page"},{"location":"pfaffian/","page":"Pfaffians","title":"Pfaffians","text":"julia> using SkewLinearAlgebra, LinearAlgebra\n\njulia> A = [0 2 -7 4; -2 0 -8 3; 7 8 0 1;-4 -3 -1 0]\n4×4 Matrix{Int64}:\n  0   2  -7  4\n -2   0  -8  3\n  7   8   0  1\n -4  -3  -1  0\n\njulia> pfaffian(A)\n-9.000000000000002\n\njulia> det(A) # exact determinant is (-9)^2\n80.99999999999999","category":"page"},{"location":"pfaffian/","page":"Pfaffians","title":"Pfaffians","text":"By default, this computation is performed approximately using floating-point calculations, similar to Julia's det algorithm for the determinant.  However, for a BigInt matrix, pfaffian(A) is computed exactly using an algorithm by Galbiati and Maffioli (1994):","category":"page"},{"location":"pfaffian/","page":"Pfaffians","title":"Pfaffians","text":"julia> pfaffian(BigInt.(A))\n-9\n\njulia> det(big.(A)) # also exact for BigInt\n81","category":"page"},{"location":"pfaffian/","page":"Pfaffians","title":"Pfaffians","text":"Note that you need not (but may) pass a SkewHermitian matrix type to pfaffian.  However, because the Pfaffian is only defined for skew-symmetric matrices, it will give an error if you pass it a non-skewsymmetric matrix:","category":"page"},{"location":"pfaffian/","page":"Pfaffians","title":"Pfaffians","text":"julia> pfaffian([1 2 3; 4 5 6; 7 8 9])\nERROR: ArgumentError: Pfaffian requires a skew-Hermitian matrix","category":"page"},{"location":"pfaffian/","page":"Pfaffians","title":"Pfaffians","text":"We also provide a function pfaffian!(A) that overwrites A in-place (with undocumented values), rather than making a copy of the matrix for intermediate calculations:","category":"page"},{"location":"pfaffian/","page":"Pfaffians","title":"Pfaffians","text":"julia> pfaffian!(BigInt[0 2 -7; -2 0 -8; 7 8 0])\n0","category":"page"},{"location":"pfaffian/","page":"Pfaffians","title":"Pfaffians","text":"(Note that the Pfaffian is always zero for any odd size skew-symmetric matrix.)","category":"page"},{"location":"pfaffian/","page":"Pfaffians","title":"Pfaffians","text":"Since the computation of the pfaffian can easily overflow/underflow the maximum/minimum representable floating-point value, we also provide a function logabspfaffian (along with an in-place variant logabspfaffian!) that returns a tuple (logpf, sign) such that the Pfaffian is sign * exp(logpf).   (This is similar to the logabsdet function in Julia's LinearAlgebra library to compute the log of the determinant.)","category":"page"},{"location":"pfaffian/","page":"Pfaffians","title":"Pfaffians","text":"julia> logpf, sign = logabspfaffian(A)\n(2.1972245773362196, -1.0)\n\njulia> sign * exp(logpf) # matches pfaffian(A), up to floating-point rounding errors\n-9.000000000000002\n\njulia> B = triu(rand(-9:9, 500,500)); B = B - B'; # 500×500 skew-symmetric matrix\n\njulia> pfaffian(B) # overflows double-precision floating point\nInf\n\njulia> pf = pfaffian(big.(B)) # exact answer in BigInt precision (slow)\n-149678583522522720601879230931167588166888361287738234955688347466367975777696295859892310371985561723944757337655733584612691078889626269612647408920674699424393216780756729980039853434268507566870340916969614567968786613166601938742927283707974123631646016992038329261449437213872613766410239159659548127386325836018158542150965421313640795710036050440344289340146687857870477701301808699453999823930142237829465931054145755710674564378910415127367945223991977718726\n\njulia> Float64(log(abs(pf))) # exactly rounded log(abs(pfaffian(B)))\n1075.7105584607807\n\njulia> logabspfaffian(B) # matches log and sign!\n(1075.71055846078, -1.0)","category":"page"},{"location":"pfaffian/#Pfaffian-Reference","page":"Pfaffians","title":"Pfaffian Reference","text":"","category":"section"},{"location":"pfaffian/","page":"Pfaffians","title":"Pfaffians","text":"SkewLinearAlgebra.pfaffian\nSkewLinearAlgebra.pfaffian!\nSkewLinearAlgebra.logabspfaffian\nSkewLinearAlgebra.logabspfaffian!","category":"page"},{"location":"pfaffian/#SkewLinearAlgebra.pfaffian","page":"Pfaffians","title":"SkewLinearAlgebra.pfaffian","text":"pfaffian(A)\n\nReturns the pfaffian of A where a is a real skew-Hermitian matrix. If A is not of type SkewHermitian{<:Real}, then isskewhermitian(A) is checked to ensure that A == -A'\n\n\n\n\n\n","category":"function"},{"location":"pfaffian/#SkewLinearAlgebra.pfaffian!","page":"Pfaffians","title":"SkewLinearAlgebra.pfaffian!","text":"pfaffian!(A)\n\nSimilar to pfaffian, but overwrites A in-place with intermediate calculations.\n\n\n\n\n\n","category":"function"},{"location":"pfaffian/#SkewLinearAlgebra.logabspfaffian","page":"Pfaffians","title":"SkewLinearAlgebra.logabspfaffian","text":"logabspfaffian(A)\n\nReturns a tuple (log|Pf A|, sign), with the log of the absolute value of the pfaffian of A as first output and the sign (±1) of the pfaffian as second output. A must be a real skew-Hermitian matrix. If A is not of type SkewHermitian{<:Real}, then isskewhermitian(A) is checked to ensure that A == -A'\n\n\n\n\n\n","category":"function"},{"location":"pfaffian/#SkewLinearAlgebra.logabspfaffian!","page":"Pfaffians","title":"SkewLinearAlgebra.logabspfaffian!","text":"logabspfaffian!(A)\n\nSimilar to logabspfaffian, but overwrites A in-place with intermediate calculations.\n\n\n\n\n\n","category":"function"},{"location":"trig/#Trigonometric-functions","page":"Exponential/Trigonometric functions","title":"Trigonometric functions","text":"","category":"section"},{"location":"trig/","page":"Exponential/Trigonometric functions","title":"Exponential/Trigonometric functions","text":"The package implements special methods of trigonometric matrix functions using our optimized eigenvalue decomposition for SkewHermitian and SkewHermTridiagonal matrices: exp, cis, cos, sin, sincos, sinh, and cosh.","category":"page"},{"location":"trig/","page":"Exponential/Trigonometric functions","title":"Exponential/Trigonometric functions","text":"For example:","category":"page"},{"location":"trig/","page":"Exponential/Trigonometric functions","title":"Exponential/Trigonometric functions","text":"julia> using SkewLinearAlgebra, LinearAlgebra\n\njulia> A = SkewHermitian([0 2 -7 4; -2 0 -8 3; 7 8 0 1;-4 -3 -1 0]);\n\njulia> Q = exp(A)\n4×4 Matrix{Float64}:\n-0.317791  -0.816528    -0.268647   0.400149\n-0.697298   0.140338     0.677464   0.187414\n0.578289  -0.00844255   0.40033    0.710807\n0.279941  -0.559925     0.555524  -0.547275","category":"page"},{"location":"trig/","page":"Exponential/Trigonometric functions","title":"Exponential/Trigonometric functions","text":"Note that the exponential of a skew-Hermitian matrix is very special: it is unitary.  That is, if A^* = -A, then (e^A)^* = (e^A)^-1:","category":"page"},{"location":"trig/","page":"Exponential/Trigonometric functions","title":"Exponential/Trigonometric functions","text":"julia> Q' ≈ Q^-1\ntrue","category":"page"},{"location":"trig/","page":"Exponential/Trigonometric functions","title":"Exponential/Trigonometric functions","text":"Several of the other matrix trigonometric functions also have special return types, in addition to being optimized for performance.","category":"page"},{"location":"trig/","page":"Exponential/Trigonometric functions","title":"Exponential/Trigonometric functions","text":"julia> cis(A)\n4×4 Hermitian{ComplexF64, Matrix{ComplexF64}}:\n 36765.0+0.0im       36532.0+13228.5im  …   537.235+22406.2im\n 36532.0-13228.5im   41062.9+0.0im          8595.76+22070.7im\n 10744.7+46097.2im  -5909.58+49673.4im     -27936.2+7221.87im\n 537.235-22406.2im   8595.76-22070.7im      13663.9+0.0im\n\njulia> cos(A)\n4×4 Symmetric{Float64, Matrix{Float64}}:\n 36765.0    36532.0    10744.7      537.235\n 36532.0    41062.9    -5909.58    8595.76\n 10744.7    -5909.58   60940.6   -27936.2\n   537.235   8595.76  -27936.2    13663.9\n\njulia> cosh(A)\n4×4 Hermitian{Float64, Matrix{Float64}}:\n 0.766512      0.0374       0.011        0.000550001\n 0.0374        0.770912    -0.00605001   0.00880001\n 0.011        -0.00605001   0.791262    -0.0286\n 0.000550001   0.00880001  -0.0286       0.742862","category":"page"},{"location":"#SkewLinearAlgebra","page":"Home","title":"SkewLinearAlgebra","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The SkewLinearAlgebra package provides specialized matrix types, optimized methods of LinearAlgebra functions, and a few entirely new functions for dealing with linear algebra on skew-Hermitian matrices, especially for the case of real skew-symmetric matrices.","category":"page"},{"location":"#Introduction","page":"Home","title":"Introduction","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A skew-Hermitian matrix A is a square matrix that equals the negative of its conjugate-transpose: A=-overlineA^T=-A^*, equivalent to A == -A' in Julia.  (In the real skew-symmetric case, this is simply A=-A^T.)   Such matrices have special computational properties: orthogonal eigenvectors and purely imaginary eigenvalues, \"skew-Cholesky\" factorizations, and a relative of the determinant called the Pfaffian.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Although any skew-Hermitian matrix A can be transformed into a Hermitian matrix H=iA, this transformation converts real matrices A into complex-Hermitian matrices H, which entails at least a factor of two loss in performance and memory usage compared to the real case.   (And certain operations, like the Pfaffian, are only defined for the skew-symmetric case.)  SkewLinearAlgebra gives you access to the greater performance and functionality that are possible for purely real skew-symmetric matrices.","category":"page"},{"location":"","page":"Home","title":"Home","text":"To achieve this SkewLinearAlgebra defines a new matrix type, SkewHermitian (analogous to the LinearAlgebra.Hermitian type in the Julia standard library) that gives you access to optimized methods and specialized functionality for skew-Hermitian matrices, especially in the real case.  It also provides a more specialized SkewHermTridiagonal for skew-Hermitian tridiagonal matrices (analogous to the LinearAlgebra.SymTridiagonal type in the Julia standard library) .","category":"page"},{"location":"#Contents","page":"Home","title":"Contents","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The SkewLinearAlgebra documentation is divided into the following sections:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Skew-Hermitian matrices: the SkewHermitian type, constructors, and basic operations\nSkew-Hermitian tridiagonal matrices: the SkewHermTridiagonal type, constructors, and basic operations\nSkew-Hermitian eigenproblems: optimized eigenvalues/eigenvectors (& Schur/SVD), and Hessenberg factorization\nTrigonometric functions: optimized matrix exponentials and related functions (exp, sin, cos, etcetera)\nPfaffian calculations: computation of the Pfaffian and log-Pfaffian\nSkew-Cholesky factorization: a skew-Hermitian analogue of Cholesky factorization","category":"page"},{"location":"#Quick-start","page":"Home","title":"Quick start","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Here is a simple example demonstrating some of the features of the SkewLinearAlgebra package.   See the manual chapters outlines above for the complete details and explanations:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using SkewLinearAlgebra, LinearAlgebra\n\njulia> A = SkewHermitian([0  1 2\n                         -1  0 3\n                         -2 -3 0])\n3×3 SkewHermitian{Int64, Matrix{Int64}}:\n  0   1  2\n -1   0  3\n -2  -3  0\n\njulia> eigvals(A) # optimized eigenvalue calculation (purely imaginary)\n3-element Vector{ComplexF64}:\n 0.0 - 3.7416573867739404im\n 0.0 + 3.7416573867739404im\n 0.0 + 0.0im\n\njulia> Q = exp(A) # optimized matrix exponential\n3×3 Matrix{Float64}:\n  0.348107  -0.933192   0.0892929\n -0.63135   -0.303785  -0.713521\n  0.692978   0.192007  -0.694921\n\njulia> Q'Q ≈ I # the exponential of a skew-Hermitian matrix is unitary\ntrue\n\njulia> pfaffian(A) # the Pfaffian (always zero for odd-size skew matrices)\n0.0","category":"page"},{"location":"#Acknowledgements","page":"Home","title":"Acknowledgements","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The SkewLinearAlgebra package was initially created by Simon Mataigne and Steven G. Johnson, with support from UCLouvain and the MIT–Belgium program.","category":"page"},{"location":"types/#Skew-Hermitian-matrices","page":"Matrix Types","title":"Skew-Hermitian matrices","text":"","category":"section"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"SkewHermitian(A) wraps an existing matrix A, which must already be skew-Hermitian, in the SkewHermitian type (a subtype of AbstractMatrix), which supports fast specialized operations noted below.  You can use the function isskewhermitian(A) to check whether A is skew-Hermitian (A == -A').","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"SkewHermitian(A) requires that A == -A' and throws an error if it is not. Alternatively, you can use the funcition skewhermitian(A) to take the skew-Hermitian part of A, defined by (A - A')/2, and wrap it in a SkewHermitian view.  The function skewhermitian!(A) does the same operation in-place on A (overwriting A with its skew-Hermitian part).","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"Here is a basic example to initialize a SkewHermitian matrix:","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"julia> using SkewLinearAlgebra, LinearAlgebra\n\njulia> A = [0 2 -7 4; -2 0 -8 3; 7 8 0 1;-4 -3 -1 0]\n3×3 Matrix{Int64}:\n  0  2 -7  4\n -2  0 -8  3\n  7  8  0  1\n  -4 -3 -1 0\n\njulia> isskewhermitian(A)\ntrue\n\njulia> A = SkewHermitian(A)\n4×4 SkewHermitian{Int64, Matrix{Int64}}:\n  0   2  -7  4\n -2   0  -8  3\n  7   8   0  1\n -4  -3  -1  0","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"Basic linear-algebra operations are supported:","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"julia> tr(A)\n0\n\njulia> det(A)\n81.0\n\njulia> inv(A)\n4×4 SkewHermitian{Float64, Matrix{Float64}}:\n  0.0        0.111111  -0.333333  -0.888889\n -0.111111   0.0        0.444444   0.777778\n  0.333333  -0.444444   0.0        0.222222\n  0.888889  -0.777778  -0.222222   0.0\n\njulia> x=[1;2;3;4]\n4-element Vector{Int64}:\n 1\n 2\n 3\n 4\n\njulia> A\\x\n4-element Vector{Float64}:\n -4.333333333333334\n  4.333333333333334\n  0.3333333333333336\n -1.3333333333333333","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"A SkewHermitian matrix A is simply a wrapper around an underlying matrix (in which both the upper and lower triangles are stored, despite the redundancy, to support fast matrix operations).  You can extract this underlying matrix with the Julia parent(A) function (this does not copy the data: mutating the parent will modify A).  Alternatively, you can copy the data to an ordinary Matrix (2d Array) with Matrix(A).","category":"page"},{"location":"types/#Operations-on-SkewHermitian","page":"Matrix Types","title":"Operations on SkewHermitian","text":"","category":"section"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"The SkewHermitian type supports the basic operations for any Julia AbstractMatrix (indexing, iteration, multiplication, addition, scaling, and so on).   Matrix–matrix and matrix–vector multiplications are performed using the underlying parent matrix, so they are fast.","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"We try to preserve the SkewHermitian wrapper, if possible.  For example, adding two SkewHermitian matrices or scaling by a real number yields another SkewHermitian matrix.  Similarly for real(A), conj(A), inv(A), or -A.","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"The package provides several optimized methods for SkewHermitian matrices, based on the functions defined by the Julia LinearAlgebra package:","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"Tridiagonal reduction: hessenberg\nEigensolvers: eigen, eigvals (also schur, svd, svdvals)\nTrigonometric functions:exp, cis,cos,sin,sinh,cosh,sincos","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"We also define the following new functions for real skew-symmetric matrices only:","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"Cholesky-like factorization: skewchol\nPfaffian: pfaffian, logabspfaffian","category":"page"},{"location":"types/#Skew-Hermitian-tridiagonal-matrices","page":"Matrix Types","title":"Skew-Hermitian tridiagonal matrices","text":"","category":"section"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"In the special case of a tridiagonal skew-Hermitian matrix, many calculations can be performed very quickly, typically with O(n) operations for an ntimes n matrix. Such optimizations are supported by the SkewLinearAlgebra package using the SkewHermTridiagonal matrix type.","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"A complex tridiagonal skew-Hermitian matrix is of the form:","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"A=left(beginarrayccccc\nid_1  -e_1^*\ne_1  id_2  -e_2^*\n  e_2  id_3  ddots\n    ddots  ddots  -e_n-1^*\n      e_n-1  id_n\nendarrayright)=-A^*","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"with purely imaginary diagonal entries id_k.   This is represented in the SkewLinearAlgebra package by calling the SkewHermTridiagonal(ev,dvim) constructor, where ev is the (complex) vector of n-1 subdiagonal entries e_k and dvim is the (real) vector of n diagonal imaginary parts d_k:","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"julia> SkewHermTridiagonal([1+2im,3+4im],[5,6,7])\n3×3 SkewHermTridiagonal{Complex{Int64}, Vector{Complex{Int64}}, Vector{Int64}}:\n 0+5im  -1+2im     ⋅\n 1+2im   0+6im  -3+4im\n   ⋅     3+4im   0+7im","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"In the case of a real matrix, the diagonal entries are zero, and the matrix takes the form:","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"textreal A=left(beginarrayccccc\n0  -e_1\ne_1  0  -e_2\n  e_2  0  ddots\n    ddots  ddots  -e_n-1\n      e_n-1  0\nendarrayright)=-A^T","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"In this case, you need not store the zero diagonal entries, and can simply call SkewHermTridiagonal(ev) with the real vector ev of the n-1 subdiagonal entries:","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"julia> A = SkewHermTridiagonal([1,2,3])\n4×4 SkewHermTridiagonal{Int64, Vector{Int64}, Nothing}:\n ⋅  -1   ⋅   ⋅\n 1   ⋅  -2   ⋅\n ⋅   2   ⋅  -3\n ⋅   ⋅   3   ⋅","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"(Notice that zero values that are not stored (“structural zeros”) are shown as a ⋅.)","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"A SkewHermTridiagonal matrix can also be converted to the LinearAlgebra.Tridiagonal type in the Julia standard library:","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"julia> Tridiagonal(A)\n4×4 Tridiagonal{Int64, Vector{Int64}}:\n 0  -1   ⋅   ⋅\n 1   0  -2   ⋅\n ⋅   2   0  -3\n ⋅   ⋅   3   0","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"which may support a wider range of linear-algebra functions, but does not optimized for the skew-Hermitian structure.","category":"page"},{"location":"types/#Operations-on-SkewHermTridiagonal","page":"Matrix Types","title":"Operations on SkewHermTridiagonal","text":"","category":"section"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"The SkewHermTridiagonal type is modeled on the LinearAlgebra.SymTridiagonal type in the Julia standard library) and supports typical matrix operations (indexing, iteration, scaling, and so on).","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"The package provides several optimized methods for SkewHermTridiagonal matrices, based on the functions defined by the Julia LinearAlgebra package:","category":"page"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"Matrix-vector A*x and dot(x,A,y) products; also solves A\\x (via conversion to Tridiagonal)\nEigensolvers: eigen, eigvals (also svd, svdvals)\nTrigonometric functions:exp, cis,cos,sin,sinh,cosh,sincos","category":"page"},{"location":"types/#Types-Reference","page":"Matrix Types","title":"Types Reference","text":"","category":"section"},{"location":"types/","page":"Matrix Types","title":"Matrix Types","text":"SkewLinearAlgebra.SkewHermitian\nSkewLinearAlgebra.skewhermitian\nSkewLinearAlgebra.skewhermitian!\nSkewLinearAlgebra.SkewHermTridiagonal","category":"page"},{"location":"types/#SkewLinearAlgebra.SkewHermitian","page":"Matrix Types","title":"SkewLinearAlgebra.SkewHermitian","text":"SkewHermitian(A) <: AbstractMatrix\n\nConstruct a SkewHermitian view of the skew-Hermitian matrix A (A == -A'), which allows one to exploit efficient operations for eigenvalues, exponentiation, and more.\n\nTakes \"ownership\" of the matrix A.  See also skewhermitian, which takes the skew-hermitian part of A, and skewhermitian!, which does this in-place, along with isskewhermitian which checks whether A == -A'.\n\n\n\n\n\n","category":"type"},{"location":"types/#SkewLinearAlgebra.skewhermitian","page":"Matrix Types","title":"SkewLinearAlgebra.skewhermitian","text":"skewhermitian(A)\n\nReturns the skew-Hermitian part of A, i.e. (A-A')/2.  See also skewhermitian!, which does this in-place.\n\n\n\n\n\n","category":"function"},{"location":"types/#SkewLinearAlgebra.skewhermitian!","page":"Matrix Types","title":"SkewLinearAlgebra.skewhermitian!","text":"skewhermitian!(A)\n\nTransforms A in-place to its skew-Hermitian part (A-A')/2, and returns a SkewHermitian view.\n\n\n\n\n\n","category":"function"},{"location":"types/#SkewLinearAlgebra.SkewHermTridiagonal","page":"Matrix Types","title":"SkewLinearAlgebra.SkewHermTridiagonal","text":"SkewHermTridiagonal(ev::V, dvim::Vim) where {V <: AbstractVector, Vim <: AbstractVector{<:Real}}\n\nConstruct a skewhermitian tridiagonal matrix from the subdiagonal (ev) and the imaginary part of the main diagonal (dvim). The result is of type SkewHermTridiagonal and provides efficient specialized eigensolvers, but may be converted into a regular matrix with convert(Array, _) (or Array(_) for short).\n\nExamples\n\njulia> ev = complex.([7, 8, 9] , [7, 8, 9])\n3-element Vector{Complex{Int64}}:\n 7 + 7im\n 8 + 8im\n 9 + 9im\n julia> dvim =  [1, 2, 3, 4]\n 4-element Vector{Int64}:\n  1\n  2\n  3\n  4\njulia> SkewHermTridiagonal(ev, dvim)\n4×4 SkewHermTridiagonal{Complex{Int64}, Vector{Complex{Int64}}, Vector{Int64}}:\n 0+1im -7+7im  0+0im  0+0im\n 7-7im  0+2im -8+8im  0+0im\n 0+0im -8+8im  0+3im -9+9im\n 0+0im  0+0im  9+9im  0+4im\n\n\n\n\n\nSkewHermTridiagonal(A::AbstractMatrix)\n\nConstruct a skewhermitian tridiagonal matrix from first subdiagonal and main diagonal of the skewhermitian matrix A.\n\nExamples\n\njulia> A = [1 2 3; 2 4 5; 3 5 6]\n3×3 Matrix{Int64}:\n 1  2  3\n 2  4  5\n 3  5  6\njulia> SkewHermTridiagonal(A)\n3×3 SkewHermTridiagonal{Int64, Vector{Int64}}:\n 0 -2  0\n 2  0 -5\n 0  5  0\n\n\n\n\n\n","category":"type"}]
}
