using LinearAlgebra
using BenchmarkTools
include("SkewLinearAlgebra.jl")
import .SkewLinearAlgebra as SLA
BLAS.set_num_threads(1)
n=1000
A=randn(n,n)
#B=zeros(n,n)
for i=1:n
    A[i,i]=0
    #B[i,i]=0
    for j=1:i-1
        #A[j,i]=i+j-1
        A[i,j]=-A[j,i]
        
        #B[j,i]=A[j,i]
        #B[i,j]=A[i,j]
        
    end
end
B=copy(A)
A=SLA.SkewSymmetric(A)
#B=SLA.SkewSymmetric(B)
#B=SLA.copyto!(B,A)
#B=SLA.Matrix(A)
#setindex!(A,2,3,2)
#display(A)
#C = SLA.times(A,B)
#display(eigen(A.data))
#display(hessenberg(B).H)
#@btime hessenberg(B)
#@btime hessenberg!(A) setup=(A = copy($A))
#hessenberg!(A)
#display(A)
#display(H.V)
#display(H.H)
#vals = eigvals(B)
#display(vals)
#vals = eigvals(A)
#display(vals)
#vals = eigmin(A)
#display(vals)
#display(exp(A.data))
#display(exp(A))
#display((2*A).data)
#A.data=(2*B).data
#display(A.data)
#B=B.*1im
#B=Hermitian(B)
#B=Symmetric(B)
@btime hessenberg!(A) setup=(A = copy($A))
@btime hessenberg(B)

#@btime exp(B) 
#@btime exp(A) 
a=1