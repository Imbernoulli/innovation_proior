# TIER: strong
# Space-filling design ON the trade-off surface: pin all distance vars at 0.5
# (g=0) and lay a near-uniform grid over ALL M-1 position variables, spending
# the budget to cover the full Pareto surface (including the extreme axis
# points that dominate the largest boxes). Much larger hypervolume than the
# 1D arc, but a fixed grid still leaves headroom vs. an HV-optimal placement.
import sys, json


def linspace(c):
    if c <= 1:
        return [0.5]
    return [i / (c - 1) for i in range(c)]


inst = json.load(sys.stdin)
M = inst["M"]
n = inst["n"]
B = inst["B"]
pdim = M - 1                        # number of position variables

# largest per-axis resolution c with c**pdim <= B
c = 1
while (c + 1) ** pdim <= B:
    c += 1
grid = linspace(c)

pts = []
if pdim == 1:
    for a in grid:
        x = [0.5] * n
        x[0] = a
        pts.append(x)
else:  # pdim == 2
    for a in grid:
        for b in grid:
            x = [0.5] * n
            x[0] = a
            x[1] = b
            pts.append(x)

# spend any leftover budget on the pure axis / edge-extreme directions
if len(pts) < B:
    extras = []
    for d in range(pdim):
        x = [0.5] * n
        for e in range(pdim):
            x[e] = 0.0
        x[d] = 1.0
        extras.append(x)
    for x in extras:
        if len(pts) >= B:
            break
        pts.append(x)

pts = pts[:B]
print(json.dumps({"points": pts}))
