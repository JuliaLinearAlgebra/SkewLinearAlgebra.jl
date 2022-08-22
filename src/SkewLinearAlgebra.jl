# This file is a part of Julia. License is MIT: https://julialang.org/license
"""
This module based on the LinearAlgebra module provides specialized functions
and types for skew-symmetricmatrices, i.e A=-A^T
"""
module SkewLinearAlgebra

using LinearAlgebra
import LinearAlgebra as LA
export
    #Types
    SkewHermitian,
    SkewHermTridiagonal,
    SkewCholesky,
    JMatrix,
    #functions
    isskewhermitian,
    skewhermitian,
    skewhermitian!,
    to_symtridiagonal,
    pfaffian,
    pfaffian!,
    logabspfaffian,
    logabspfaffian!,
    skewchol,
    skewchol!

include("skewhermitian.jl")
include("tridiag.jl")
include("jmatrix.jl")
include("hessenberg.jl")
include("eigen.jl")
include("exp.jl")
include("cholesky.jl")
include("pfaffian.jl")

end



