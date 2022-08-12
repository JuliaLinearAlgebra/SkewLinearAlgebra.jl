using LinearAlgebra, Random
import SkewLinearAlgebra as SLA
using Test

Random.seed!(314159) # use same pseudorandom stream for every test

@testset "README.md" begin # test from the README examples
    A = [0 2 -7 4; -2 0 -8 3; 7 8 0 1;-4 -3 -1 0]
    @test SLA.isskewhermitian(A)
    A = SLA.SkewHermitian(A)
    @test tr(A) == 0
    @test det(A) ≈ 81
    @test SLA.isskewhermitian(inv(A))
    @test inv(A) ≈ [0 1 -3 -8; -1 0 4 7; 3 -4 0 2; 8 -7 -2 0]/9
    @test A \ [1,2,3,4] ≈ [-13,13,1,-4]/3
    v = [8.306623862918073, 8.53382018538718, -1.083472677771923]
    @test hessenberg(A).H ≈ Tridiagonal(v,[0,0,0,0.],-v)
    iλ₁,iλ₂ = 11.93445871397423, 0.7541188264752862
    @test eigvals(A) ≈ [iλ₁,iλ₂,-iλ₂,-iλ₁]*im
    @test Matrix(hessenberg(A).Q) ≈ [1.0 0.0 0.0 0.0; 0.0 -0.2407717061715382 -0.9592700375676934 -0.14774972261267352; 0.0 0.8427009716003843 -0.2821382463434394 0.4585336219014009; 0.0 -0.48154341234307674 -0.014106912317171805 0.8763086996337883]
    @test eigvals(A, 0,15) ≈ [iλ₁,iλ₂]*im
    @test eigvals(A, 1:3) ≈ [iλ₁,iλ₂,-iλ₂]*im
    @test svdvals(A) ≈ [iλ₁,iλ₁,iλ₂,iλ₂]
end

@testset "SkewLinearAlgebra.jl" begin
    for n in [2,20,153,200]
        A = SLA.skewhermitian(randn(n,n))
        @test SLA.isskewhermitian(A)
        @test SLA.isskewhermitian(A.data)
        B = 2*Matrix(A)
        @test SLA.isskewhermitian(B)

        @test A == copy(A)::SLA.SkewHermitian
        @test size(A) == size(A.data)
        @test size(A,1) == size(A.data,1)
        @test size(A,2) == size(A.data,2)
        @test Matrix(A) == A.data
        @test tr(A) == 0
        @test (-A).data ==-(A.data)
        A2 = A.data*A.data
        @test A*A == A2 ≈ Symmetric(A2)
        @test A*B == A.data*B
        @test B*A == B*A.data
        if iseven(n) # for odd n, a skew-Hermitian matrix is singular
            @test inv(A)::SLA.SkewHermitian ≈ inv(A.data)
        end
        @test (A*2).data ==A.data*2
        @test (2*A).data ==2*A.data
        @test (A/2).data == A.data/2
        C = A + A
        @test C.data==A.data+A.data
        B = SLA.SkewHermitian(B)
        C = A - B
        @test C.data==-A.data
        B=triu(A)
        @test B≈triu(A.data)
        B = tril(A,n-2)
        @test B≈tril(A.data,n-2)
        k = dot(A,A)
        B = Matrix(A)
        for i=1:n
            for j=1:n
                B[i,j] *= B[i,j]
            end
        end
        @test k≈sum(B)

        if n>1
            @test getindex(A,2,1) == A.data[2,1]
        end

        setindex!(A,3,n,n-1)
        @test getindex(A,n,n-1) ==3
        @test getindex(A,n-1,n) ==-3
        @test parent(A) == A.data

        x = randn(n)
        y = zeros(n)
        mul!(y,A,x,2,0)
        @test y == 2*A.data*x
        k = dot(y,A,x)
        @test k ≈ transpose(y)*A.data*x
        k = copy(y)
        mul!(y,A,x,2,3)
        @test y ≈ 2*A*x+3*k
        B = copy(A)
        copyto!(B,A)
        @test B == A
        B = Matrix(A)
        @test B == A.data
        C = similar(B,n,n)
        mul!(C,A,B,2,0)
        @test C == 2*A.data*B
        mul!(C,B,A,2,0)
        @test C == 2*B*A.data
        B = SLA.SkewHermitian(B)
        mul!(C,B,A,2,0)
        @test C == 2*B.data*A.data
        A.data[n,n] = 4
        @test SLA.isskewhermitian(A.data) == false
        A.data[n,n] = 0
        A.data[n,1] = 4
        @test SLA.isskewhermitian(A.data) == false
        #LU=lu(A)
        #@test LU.L*LU.U≈A.data
        LQ = lq(A)
        @test LQ.L*LQ.Q ≈ A.data
        QR = qr(A)
        @test QR.Q*QR.R ≈ A.data
        A = SLA.skewhermitian(randn(n,n))
        F = schur(A)
        @test A.data ≈ F.vectors * F.Schur * F.vectors'

        Ac = SLA.skewhermitian!(randn(ComplexF64, n, n))
        for f in (real, imag)
            @test f(Ac) == f(Matrix(Ac))
        end
    end
end

@testset "hessenberg.jl" begin
    for n in [2,20,153,200]
        A = SLA.skewhermitian(randn(n,n))
        B = Matrix(A)
        HA = hessenberg(A)
        HB = hessenberg(B)
        @test Matrix(HA.H) ≈ Matrix(HB.H)
        @test Matrix(HA.Q) ≈ Matrix(HB.Q)
    end

    A=zeros(4,4)
    A[2:4,1]=ones(3)
    A[1,2:4]=-ones(3)
    A=SLA.SkewHermitian(A)
    B=Matrix(A)
    HA=hessenberg(A)
    HB=hessenberg(B)
    @test Matrix(HA.H)≈Matrix(HB.H)

end

@testset "eigen.jl" begin
    for n in [2,20,153,200]
        A = SLA.skewhermitian(randn(n,n))
        B = Matrix(A)

        valA = imag(eigvals(A))
        valB = imag(eigvals(B))
        sort!(valA)
        sort!(valB)
        @test valA ≈ valB
        Eig = eigen(A)
        valA = Eig.values
        Q2 = Eig.vectors
        valB,Q = eigen(B)
        @test real(Q2*diagm(valA)*adjoint(Q2)) ≈ A.data
        valA = imag(valA)
        valB = imag(valB)
        sort!(valA)
        sort!(valB)
        @test valA ≈ valB
        Svd = svd(A)
        @test Svd.U*Diagonal(Svd.S)*Svd.Vt ≈ A.data
        @test svdvals(A)≈svdvals(B)
    end
end
@testset "exp.jl" begin

    for n in [2,20,153,200]
        A=SLA.skewhermitian(randn(n,n))
        B=Matrix(A)
        @test exp(B) ≈ exp(A)
        @test cis(A) ≈ exp(Hermitian(A.data*1im))
        @test cos(B) ≈ cos(A)
        @test sin(B) ≈ sin(A)
        #@test tan(B)≈tan(A)
        @test sinh(B) ≈ sinh(A)
        @test cosh(B) ≈ cosh(A)
        @test tanh(B) ≈ tanh(A)
    end
end



@testset "tridiag.jl" begin 
    for n in [2,20,153,200]
        C=SLA.skewhermitian(randn(n,n))
        A=SLA.SkewHermTridiagonal(C)
        @test Tridiagonal(Matrix(A))≈Tridiagonal(Matrix(C))
        
        A=SLA.SkewHermTridiagonal(randn(n-1))
        

        C=randn(n,n)
        D1=randn(n,n)
        D2=copy(D1)
        mul!(D1,A,C,2,1)
        @test D1≈D2+2*Matrix(A)*C
        mul!(D1,A,C,2,0)
        @test D1≈2*Matrix(A)*C
        @test Matrix(A+A)==Matrix(2*A)
        @test Matrix(A-2*A)==Matrix(-A)
        x=randn(n)
        y=randn(n)
        @test dot(x,A,y)≈dot(x,Matrix(A),y)
        B=Matrix(A)
        
        @test size(A,1)==n
        A=SLA.SkewHermTridiagonal(randn(n-1))
        B=Matrix(A)
        EA=eigen(A)
        EB=eigen(B)
        Q = EA.vectors
        @test real(Q*diagm(EA.values)*adjoint(Q)) ≈ B
        valA = imag(EA.values)
        valB = imag(EB.values)
        sort!(valA)
        sort!(valB)
        @test valA ≈ valB
        A=SLA.SkewHermTridiagonal(randn(n-1))
        B=Matrix(A)
        Svd = svd(A)
        @test real(Svd.U*Diagonal(Svd.S)*Svd.Vt) ≈ B
        A=SLA.SkewHermTridiagonal(randn(n-1))
        B=Matrix(A)
        @test svdvals(A)≈svdvals(B)
        
        A=randn(n,n)+1im*randn(n,n)
        A=(A-A')/2
        A=SLA.SkewHermTridiagonal(A)
        C=randn(n,n)+1im*randn(n,n)
        D1=randn(n,n)+1im*randn(n,n)
        D2=copy(D1)
        mul!(D1,A,C,2,1)
        @test D1≈D2+2*Matrix(A)*C
        @test dot(x,A,y)≈dot(x,Matrix(A),y)


        B = SLA.SkewHermTridiagonal([3,4,5])
        @test B == [0 -3 0 0; 3 0 -4 0; 0 4 0 -5; 0 0 5 0]
        #@test repr("text/plain", B) == "4×4 SkewLinearAlgebra.SkewHermTridiagonal{$Int, Vector{$Int}}:\n 0  -3   ⋅   ⋅\n 3   0  -4   ⋅\n ⋅   4   0  -5\n ⋅   ⋅   5   0"

        Ac = SLA.SkewHermTridiagonal(randn(ComplexF64, n))
        for f in (real, imag)
            @test f(Ac) == f(Matrix(Ac))
        end

    end
end

@testset "complexhessenberg.jl" begin
    for n in [2,20,153,200]
        A = SLA.skewhermitian(randn(n,n)+1im*randn(n,n))
        B = Matrix(A)
        HA = hessenberg(A)
        HB = hessenberg(B)
        @test Matrix(HA.H) ≈ Matrix(HB.H)
        @test Matrix(HA.Q) ≈ Matrix(HB.Q)
    end
end
@testset "pfaffian.jl" begin
    for n in [2,3,4,5,6,8,10,20,40]
        A=SLA.skewhermitian(rand(-10:10,n,n)*2)
        Abig = BigInt.(A.data)
        @test SLA.pfaffian(A) ≈ SLA.pfaffian(Abig)  == SLA.pfaffian(SLA.SkewHermitian(Abig))
        if VERSION ≥ v"1.7" # for exact det of BigInt matrices
            @test SLA.pfaffian(Abig)^2 == det(Abig)
        end
        @test Float64(SLA.pfaffian(Abig)^2) ≈ iseven(n) ? det(Float64.(A)) : 0.0
    end

end


#=
using BenchmarkTools
n=1000
A = SLA.skewhermitian(randn(n,n)+1im*randn(n,n))
B = Hermitian(A.data*1im)

C=Matrix(A)
#@btime hessenberg(B)
@btime hessenberg(A)
#@btime hessenberg(C)
a=1

=#



