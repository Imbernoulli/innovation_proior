# TIER: strong
# Pin every distance variable to 0.5 (all trials on the cost frontier), seed a uniform
# grid over the two trade-off variables, then run coordinate-wise LOCAL HYPERVOLUME
# ASCENT: repeatedly nudge each configuration's trade-off coordinates (multi-scale
# steps) and keep the move only if it increases the batch's exact dominated
# hypervolume. This reshapes the plain grid into a spread tuned to the reference point,
# capturing materially more volume than the uniform grid -- yet still short of the
# (intractable) global optimum, so it leaves headroom.
import sys, json, math

inst = json.load(sys.stdin)
n = inst["n"]
M = inst["M"]
n_pos = inst["n_pos"]
budget = inst["budget"]
ref = inst["ref"]


def dtlz2(x):
    g = 0.0
    for i in range(n_pos, n):
        d = x[i] - 0.5
        g += d * d
    theta = [xi * (math.pi / 2.0) for xi in x[:n_pos]]
    f = []
    for i in range(M):
        val = 1.0 + g
        for j in range(M - 1 - i):
            val *= math.cos(theta[j])
        if i > 0:
            val *= math.sin(theta[M - 1 - i])
        f.append(val)
    return f


def area2d(pts):
    nd = []
    miny = float("inf")
    for p in sorted(pts, key=lambda p: (p[0], p[1])):
        if p[1] < miny - 1e-15:
            nd.append(p)
            miny = p[1]
    area = 0.0
    prev_y = ref[1]
    for p in nd:
        area += (ref[0] - p[0]) * (prev_y - p[1])
        prev_y = p[1]
    return area


def hv3d(costs):
    pts = [c for c in costs if c[0] < ref[0] and c[1] < ref[1] and c[2] < ref[2]]
    if not pts:
        return 0.0
    pts.sort(key=lambda p: p[2])
    hv = 0.0
    m = len(pts)
    for i in range(m):
        z0 = pts[i][2]
        z1 = pts[i + 1][2] if i + 1 < m else ref[2]
        dz = z1 - z0
        if dz <= 0.0:
            continue
        hv += area2d(pts[:i + 1]) * dz
    return hv


side = max(2, int(budget ** 0.5))
pos = []
for a in range(side):
    for b in range(side):
        if len(pos) < budget:
            pos.append([(a + 0.5) / side, (b + 0.5) / side])


def full_hv():
    return hv3d([dtlz2([p[0], p[1]] + [0.5] * (n - n_pos)) for p in pos])


cur = full_hv()
for _ in range(60):
    improved = False
    for i in range(len(pos)):
        for d in range(2):
            for step in (0.06, -0.06, 0.02, -0.02, 0.008, -0.008):
                old = pos[i][d]
                nv = min(1.0, max(0.0, old + step))
                if nv == old:
                    continue
                pos[i][d] = nv
                h = full_hv()
                if h > cur + 1e-12:
                    cur = h
                    improved = True
                else:
                    pos[i][d] = old
    if not improved:
        break

pts = [[p[0], p[1]] + [0.5] * (n - n_pos) for p in pos]
print(json.dumps({"points": pts}))
