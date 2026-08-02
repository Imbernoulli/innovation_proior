# TIER: invalid
# Emits a correctly-shaped schedule but overspends the potassium budget several
# times over (and throws in one negative magnesium application) -> must score 0.
import sys

toks = sys.stdin.read().split()
p = iter(toks)
T = int(next(p)); P = int(next(p))
for _ in range(3):
    next(p)  # v
for _ in range(3):
    next(p)  # retain
next(p)  # kappa
BN = float(next(p)); BK = float(next(p)); BMg = float(next(p))

out = []
for t in range(T):
    if t == 0:
        out.append("%.6f %.6f %.6f" % (0.0, BK * 5.0, -1.0))
    else:
        out.append("0.0 0.0 0.0")
sys.stdout.write("\n".join(out) + "\n")
