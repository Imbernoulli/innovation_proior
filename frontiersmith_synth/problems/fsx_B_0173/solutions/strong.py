# TIER: strong
import sys, json, math
inst = json.load(sys.stdin)
n = inst["n"]; budget = inst["budget"]
# Dense grid over the two position variables (x0 -> polar, x1 -> azimuth on the octant sphere),
# distance variables pinned to 0.5 so g=0 (points land exactly on the Pareto front). Endpoints
# 0 and 1 are included so the axis-extreme corners (which contribute the most hypervolume) are
# always hit. Use as many grid rows/cols as the budget allows.
g = max(2, int(math.isqrt(budget)))
axis = [i / (g - 1) for i in range(g)]  # linspace(0,1,g) including endpoints
pts = []
for a in axis:
    for b in axis:
        pts.append([a, b] + [0.5] * (n - 2))
pts = pts[:budget]
print(json.dumps({"points": pts}))
