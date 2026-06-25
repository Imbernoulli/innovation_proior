# Find two windows A=(sA,dA), B=(sB,dB) with dA,dB in [L,n], sums achievable,
# such that exact: sA*dB > sB*dA (A better) but int64-wrapped products flip it.
M = 2**63
def wrap(x):
    x &= (2**64 - 1)
    if x >= 2**63: x -= 2**64
    return x
# Try: dA=dB=d large. sA, sB near d*1e9. exact sign of sA*d - sB*d = (sA-sB)*d.
# Same d => no overflow help (factor d cancels conceptually but products both overflow).
# Let dA=200000, dB=199999. sA=200000*1e9 - kA, sB=199999*1e9 - kB.
import itertools
d=200000
best=None
for dA in [200000]:
  for dB in [199990,199995,199999]:
    # choose sums so averages extremely close
    # avgA = avgB + epsilon ; let avgA slightly larger -> A is true winner
    # pick sB = round(dB * 1e9), sA = sB/dB*dA + small
    base=10**9
    sB = dB*base - 1          # avgB = base - 1/dB
    sA = dA*base - 1          # avgA = base - 1/dA ; since dA>dB, 1/dA<1/dB => avgA>avgB. A wins.
    exact = sA*dB - sB*dA
    wr = wrap(wrap(sA*dB) - wrap(sB*dA))
    # int64 code computes l=(ll)sA*(ll)dB ; r=(ll)sB*(ll)dA ; picks A if l>r
    l=wrap(sA*dB); r=wrap(sB*dA)
    int64_picksA = l>r
    exact_picksA = exact>0
    print(f"dA={dA} dB={dB} sA={sA} sB={sB} exact_picksA={exact_picksA} int64_picksA={int64_picksA} (l={l} r={r})")
