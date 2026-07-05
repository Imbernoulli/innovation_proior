# TIER: strong
# Pin every distance variable to 0.5 (all missions on the cost frontier), seed a spread
# over the three trade-off (position) variables, then run coordinate-wise LOCAL
# HYPERVOLUME ASCENT: repeatedly nudge each profile's trade-off coordinates (multi-scale
# steps) and keep a move only if it increases the portfolio's exact dominated 4D
# hypervolume. This reshapes the plain seed into a spread tuned to the reference point,
# capturing materially more volume than a uniform grid -- yet still short of the
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


def hv(points, rf):
    m = len(rf)
    pts = [p for p in points if all(p[i] < rf[i] for i in range(m))]
    if not pts:
        return 0.0
    if m == 1:
        return rf[0] - min(p[0] for p in pts)
    if m == 2:
        return area2d(pts)
    d = m - 1
    pts.sort(key=lambda p: p[d])
    vol = 0.0
    k = len(pts)
    for i in range(k):
        z0 = pts[i][d]
        z1 = pts[i + 1][d] if i + 1 < k else rf[d]
        dz = z1 - z0
        if dz <= 0.0:
            continue
        proj = [p[:d] for p in pts[:i + 1]]
        vol += hv(proj, rf[:d]) * dz
    return vol


# ---- seed: quasi-uniform spread over the n_pos position variables ----
side = max(2, int(round(budget ** (1.0 / max(1, n_pos)))))
pos = []
idx = 0
grid = []
def rec(prefix):
    if len(prefix) == n_pos:
        grid.append(list(prefix))
        return
    for a in range(side):
        rec(prefix + [(a + 0.5) / side])
rec([])
for row in grid:
    if len(pos) < budget:
        pos.append(row)
# if grid under-fills the budget, pad with jittered copies (deterministic)
j = 0
while len(pos) < budget and pos:
    base = pos[j % len(grid)]
    off = ((j * 0.111) % 0.9) - 0.45
    pos.append([min(1.0, max(0.0, v + 0.0)) for v in base])
    j += 1


def full_hv():
    return hv([dtlz2(p + [0.5] * (n - n_pos)) for p in pos], ref)


cur = full_hv()
steps = (0.11, -0.11, 0.045, -0.045)
for _ in range(3):
    improved = False
    for i in range(len(pos)):
        for d in range(n_pos):
            for step in steps:
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

pts = [p + [0.5] * (n - n_pos) for p in pos]
print(json.dumps({"points": pts[:budget]}))
