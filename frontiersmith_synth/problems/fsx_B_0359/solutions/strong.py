# TIER: strong
# Push every layout onto the DTLZ2 sphere (distance vars = 0.5) AND spread the
# angle vars on a near-uniform grid that tiles the octant, including the
# extreme (axis) layouts.  Even, boundary-inclusive coverage of the front
# yields a much larger dominated hypervolume than random scatter -- yet a
# finite grid still leaves headroom below the continuous-front optimum.
import sys, json, math
inst = json.load(sys.stdin)
M, k, n, budget = inst["M"], inst["k"], inst["n"], inst["budget"]


def lin(a, b, m):
    if m <= 1:
        return [0.5]
    return [a + (b - a) * i / (m - 1) for i in range(m)]


pts = []
if M == 2:
    for t in lin(0.0, 1.0, budget):
        x = [0.0] * n
        x[0] = t
        for j in range(1, n):
            x[j] = 0.5
        pts.append(x)
else:
    # M == 3: grid over the two angle vars, side ~ sqrt(budget), endpoints incl.
    side = max(2, int(math.floor(math.sqrt(budget))))
    while side * side > budget and side > 2:
        side -= 1
    grid = lin(0.0, 1.0, side)
    for a in grid:
        for b in grid:
            if len(pts) >= budget:
                break
            x = [0.0] * n
            x[0] = a
            x[1] = b
            for j in range(2, n):
                x[j] = 0.5
            pts.append(x)
    # spend any leftover budget on staggered mid-cell samples for finer tiling
    if side >= 2:
        off = 0.5 / (side - 1)
        mids = [g + off for g in grid[:-1]]
        for a in mids:
            for b in mids:
                if len(pts) >= budget:
                    break
                x = [0.0] * n
                x[0] = a
                x[1] = b
                for j in range(2, n):
                    x[j] = 0.5
                pts.append(x)
            if len(pts) >= budget:
                break
pts = pts[:budget]
print(json.dumps({"points": pts}))
