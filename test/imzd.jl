function imzd(A::SkewHermTridiagonal{T}) where T
    n = size(A, 1)
    Z = zeros(T, n, n)
    e = zeros(T, n)
    for i=1:n
        Z[i,i] = 1
    end
    ierr = 0
    m = n
    mm1 = m - 1
    e[1] = 0
    its = 0
    if n >= 2
        m0 = m
        f = 0
        for i = 1:mm1
            j = m - i
            jp1 = j + 1 
            g = abs(e[jp1])
            tmag  = abs(e[j]) + f
            test = tmag + g
            if abs(test - tmag) < eps * tmag

end