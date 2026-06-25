# n=200000, L=100000. d=100000.
# A = [0,100000) sum sA = 92233720368547  (avg 922337203.68...)
# B = [100000,200000) sum sB = 92233720368548 = sA+1 (avg slightly higher => B is TRUE max)
# Each element in [-1e9,1e9]; cap per window = 1e14; sA,sB ~9.2e13 < 1e14 feasible.
d=100000
sA=92233720368547
sB=92233720368548
assert abs(sA)<=10**9*d and abs(sB)<=10**9*d
# realize a window of length d with given sum using values in [-1e9,1e9]:
def realize(total, length):
    # base value q, remainder r; q = total//length, distribute remainder
    q, r = divmod(total, length)
    vals=[q]*length
    for i in range(r): vals[i]+=1
    assert sum(vals)==total
    assert all(-10**9<=v<=10**9 for v in vals)
    return vals
vals = realize(sA,d) + realize(sB,d)
n=2*d
print(n, d)   # L = d = 100000
print(' '.join(map(str,vals)))
