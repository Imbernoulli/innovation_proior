# TIER: trivial
# One vial filling the whole carrier: a single circle of radius R at the centre.
# Total radius = R, which exactly reproduces the checker's diameter-row baseline.
import sys

toks = sys.stdin.read().split()
N = int(toks[0])
R = float(toks[1])

print(1)
print("%.10f %.10f %.10f" % (0.0, 0.0, R))
