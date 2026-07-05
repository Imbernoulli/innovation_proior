# TIER: greedy
# Sensible-but-plain batch: lay a uniform grid over the two trade-off (position)
# variables and pin every distance variable to the frontier centre, so every
# configuration lands on the cost frontier. This spreads trials and beats the
# single-point baseline, but a uniform grid in decision space is NOT an even/optimal
# spread on the curved cost frontier -- and it ignores the asymmetric reference -- so it
# leaves hypervolume on the table.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
n_pos = inst["n_pos"]
budget = inst["budget"]
center = inst["center"]

side = max(2, int(budget ** 0.5))
pts = []
for a in range(side):
    for b in range(side):
        if len(pts) >= budget:
            break
        x = [center] * n
        if n_pos >= 1:
            x[0] = (a + 0.5) / side
        if n_pos >= 2:
            x[1] = (b + 0.5) / side
        pts.append(x)
print(json.dumps({"points": pts}))
