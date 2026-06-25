def wrap(x):
    x &= (2**64 - 1)
    if x >= 2**63: x -= 2**64
    return x
# Maximize products: use largest sums & lengths. num up to ~2e14, den up to 2e5.
# Compare windowA(sA,dA) vs windowB(sB,dB). int64 code: l=(ll)sA*dB ; r=(ll)sB*dA.
# Want product magnitude > 9.2e18: need sA*dB > 9.2e18 -> with dB=2e5, sA>4.6e13.
# That's reachable: sA = 1e9 * 5e4 = 5e13 with dA=5e4. dB=2e5 needs sB.
# Let's pick concrete near-tie that overflows.
base=10**9
# A: dA = 50000, all +base => sA = 50000*base = 5e13. avgA = base.
# B: dB = 200000, sB = 200000*base = 2e16?? no, that's > limit? sum cap: 2e5*1e9=2e14. ok sB up to 2e14.
# avgB = base => tie. break tie by making B avg base - 1/dB (one element base-1).
dA=50000; sA=dA*base            # avg = base exactly
dB=200000; sB=dB*base - 1       # avg = base - 1/dB  < base ; A should win
exact = sA*dB - sB*dA           # >0 means A wins
l=wrap(sA*dB); r=wrap(sB*dA)
print("sA*dB =", sA*dB, " > 9.2e18? ", sA*dB>9.2e18)
print("sB*dA =", sB*dA)
print("exact A wins?", exact>0)
print("int64 l=",l," r=",r," int64 picks A?", l>r)
