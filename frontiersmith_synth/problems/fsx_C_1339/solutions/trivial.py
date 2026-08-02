# TIER: trivial
# Reproduces the checker's own baseline: dump the entire season's nutrient budget
# for N, K and Mg in a single application on day 1 ("efficient in labour" but
# leaches away before the uptake curve needs it, and floods K and Mg together).
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
# demand curves are unused by the trivial construction

out = []
for t in range(T):
    if t == 0:
        out.append("%.6f %.6f %.6f" % (BN, BK, BMg))
    else:
        out.append("0.0 0.0 0.0")
sys.stdout.write("\n".join(out) + "\n")
