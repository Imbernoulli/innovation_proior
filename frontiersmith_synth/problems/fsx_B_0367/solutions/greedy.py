# TIER: greedy
# Sensible-but-plain portfolio: lay a uniform grid over the THREE trade-off (position)
# variables and pin every distance variable to 0.5, so every mission profile lands on
# the cost frontier. This spreads the missions and beats the single-point baseline, but a
# uniform grid in decision space is NOT an even/optimal spread on the curved 3D cost
# frontier, so it leaves 4D hypervolume on the table.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
n_pos = inst["n_pos"]
budget = inst["budget"]

side = max(2, int(round(budget ** (1.0 / max(1, n_pos)))))
pts = []
for a in range(side):
    for b in range(side):
        for c in range(side):
            if len(pts) >= budget:
                break
            x = [0.5] * n
            if n_pos >= 1:
                x[0] = (a + 0.5) / side
            if n_pos >= 2:
                x[1] = (b + 0.5) / side
            if n_pos >= 3:
                x[2] = (c + 0.5) / side
            pts.append(x)
if not pts:
    pts = [[0.5] * n]
print(json.dumps({"points": pts[:budget]}))
