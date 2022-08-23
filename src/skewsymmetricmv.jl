



using VectorizationBase, LinearAlgebra, Static

@generated function skewsymmetricmv!(y::AbstractVector{TY}, A::AbstractMatrix{TA}, x::AbstractVector{TX}) where {TY,TA,TX}
  T = promote_type(TY,TA,TX)
  W = VectorizationBase.pick_vector_width(T)
  RU = static(2)
  CU = RU*W
  registers_used = CU + RU
  registers_remaining = VectorizationBase.register_count() - registers_used

  WI = Int(W)
  RUI = Int(RU)
  CUI = Int(CU)
  quote
    N = LinearAlgebra.checksquare(A)
    @assert N == length(x) == length(y)
    # for now hard code block size
    GC.@preserve y A x begin
      py = VectorizationBase.zstridedpointer(y)
      pA = VectorizationBase.zstridedpointer(A)
      px = VectorizationBase.zstridedpointer(x)
      Nr = N ÷ $CU
      @assert Nr * $CU == N# "TODO: handle remainders!!!"
      r = 0
      for ro = 1:Nr
        # down rows
        # initialize accumulators
        Base.Cartesian.@nexprs $RUI u -> v_u = VectorizationBase.vzero($W, $T)
        Base.Cartesian.@nexprs $CUI u -> s_u = VectorizationBase.vzero($W, $T)
        c = 0
        for co = 1:Nr
          # across columns
          Base.Cartesian.@nexprs $RUI ri -> begin
            rowind_ri = MM($W, r + (ri-1)*$W)
            sx_ri = VectorizationBase.vload(px, (rowind_ri,))
          end

          Base.Cartesian.@nexprs $CUI ci -> begin
            colind = c+ci-1
            cx_ci = VectorizationBase.vbroadcast($W, VectorizationBase.vload(px, (colind,)))
            Base.Cartesian.@nexprs $RUI ri -> begin
              # rowind = MM($W, r + (ri-1)*$W)
              m = colind < rowind_ri #(ro == co ? colind < rowind_ri : 0 < rowind_ri)
              vL = vload(pA, (rowind_ri, colind), m)
              #@show  (rowind_ri, colind) vL
              @show s_ci
              @show sx_ri
              @show vL
              v_ri = VectorizationBase.vfmadd(vL, cx_ci, v_ri)
              s_ci = VectorizationBase.vfnmadd(vL, sx_ri, s_ci)
            end
          end
          c += $CUI
        end
        # we're storing and reloading, in hope that LLVM deletes the stores and reloads
        # because for now we're too lazy to implement a specialized method
        #TODO: don't be lazy
        # vus = VectorizationBase.VecUnroll( Base.Cartesian.@ntuple $CU u -> s_u)
        # VectorizationBase.vstore!(VectorizationBase.vsum, vy, vus, Unroll{$}((r,)))
        Base.Cartesian.@nexprs $RUI ri -> begin
          vus = VectorizationBase.VecUnroll( Base.Cartesian.@ntuple $WI u -> s_{u+(ri-1)*$WI})
          svreduced = VectorizationBase.reduce_to_onevec(+, VectorizationBase.transpose_vecunroll(vus))
          @show v_ri vus
          v_to_store = @fastmath v_ri + svreduced
          @show v_to_store
          VectorizationBase.vstore!(py, v_to_store, (MM($W, r + (ri-1)*$W),))
          @show y
        end
      
        r += $CUI
      end
      # #TODO: remainder
      # for i = Nr*CU:N
      # end
    end
  end
end

N = 16;
x = rand(N);
A = rand(N,N); A .-= A';
display(A)
y = similar(x);

yref = A*x

skewsymmetricmv!(y, A, x)




