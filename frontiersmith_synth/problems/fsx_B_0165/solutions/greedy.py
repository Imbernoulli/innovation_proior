# TIER: greedy
# Knows to sit on the true trade-off surface (distance vars = 0.5, so g=0),
# but spreads only ONE position variable while pinning the rest at 0.5. This
# sweeps a 1D arc of the Pareto front with a coarse handful of points -- better
# than a single point, but it leaves most of the M-1 dimensional surface (and
# the budget) unused.
import sys, json

inst = json.load(sys.stdin)
M = inst["M"]
n = inst["n"]
B = inst["B"]

c = min(B, 8)                       # coarse: at most 8 points along the arc
pts = []
for i in range(c):
    t = 0.0 if c == 1 else i / (c - 1)
    x = [0.5] * n
    x[0] = t                        # vary only the first position variable
    pts.append(x)
print(json.dumps({"points": pts}))
