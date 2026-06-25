# n=200000, L=100000. Two length-100000 windows:
#   A = indices [0,100000)  with sum sA
#   B = indices [100000,200000) with sum sB = sA+1  (so avgB > avgA; B is the TRUE max)
# Choose sA so that int64 cross-product comparison wraps and wrongly prefers A.
# From find_flip3: d=100000, sA=92233720368547, sB=92233720368548 give a flip.
# Feasibility: each window has 100000 elements in [-1e9,1e9]; cap = 1e11. sA~9.2e13 > 1e11!! infeasible.
# So that sB is too big for length 100000. Recompute a feasible flip:
#   product threshold near 2^63=9.223e18. With d=100000, need sB*d ~ 2^63 -> sB ~ 9.2e13,
#   but max sB with 100000 elems is 1e11. product = 1e11*1e5 = 1e16 << 2^63. NO overflow.
# Overflow needs den as large as the sum is large. The ONLY way num*den exceeds 2^63 is
# when BOTH the window is long AND values large: num~1e9*len, den=len -> 1e9*len^2 > 9.2e18
#   -> len^2 > 9.2e9 -> len > 95917. So a single window length ~96000+ with avg ~1e9.
# Thus the overflow-sensitive comparison is between a LONG near-max window and another.
# With n=200000 we can have ONE window of length ~100000+ and compare to others.
#
# Realize the flip with a SINGLE pair where one window is long. Use:
#   B = full window length dB, sum sB ; A = window length dA, sum sA.
# Pick dB=128000 (so 1e9*dB^2=1.6e19>2^63). sB=dB*1e9 => avgB=1e9 (all +1e9). product sB*?:
# Compare B (avg 1e9) against A (avg just under 1e9). int64: l=(ll)sA*dB, r=(ll)sB*dA.
# We want exact B-wins but int64 A-wins. Search feasible params:
def wrap(x):
    x &= (2**64-1)
    return x-2**64 if x>=2**63 else x
base=10**9; TWO63=2**63
best=None
n=200000
for dB in range(96000, 200001, 1000):
    sB = dB*base                      # avg exactly base (all +1e9), feasible
    # A: length dA, avg base - tiny so B truly wins. choose dA, sA=dA*base-1.
    for dA in range(96000, 200001, 1000):
        sA = dA*base - 1              # avg = base - 1/dA < base => B wins exactly
        if abs(sA)>base*dA: continue
        # exact: B wins iff sB*dA > sA*dB
        exact_B = sB*dA > sA*dB
        l=wrap(sA*dB); r=wrap(sB*dA)
        int64_B = r>l
        if exact_B and not int64_B:
            best=(dA,sA,dB,sB); break
    if best: break
print("flip params:", best)
