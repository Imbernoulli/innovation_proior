# TIER: greedy
# Sensible-but-plain batch: lay a uniform grid over the trade-off (position)
# variables in DECISION space and pin every distance variable to 0.5, so every
# configuration lands on the cost frontier. This spreads prototypes and beats the
# single-point baseline. But because the surrogate warps the position variables
# through the biased power map theta=(x**alpha)*(pi/2), a uniform grid in x-space
# is a heavily CLUSTERED, uneven spread on the actual cost frontier (especially for
# alpha>1), leaving a lot of hypervolume on the table.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
n_pos = inst["n_pos"]
budget = inst["budget"]

pts = []
if n_pos == 1:
    for a in range(budget):
        x = [0.5] * n
        x[0] = (a + 0.5) / budget
        pts.append(x)
else:
    side = max(2, int(budget ** 0.5))
    for a in range(side):
        for b in range(side):
            if len(pts) >= budget:
                break
            x = [0.5] * n
            x[0] = (a + 0.5) / side
            x[1] = (b + 0.5) / side
            pts.append(x)
print(json.dumps({"points": pts}))
