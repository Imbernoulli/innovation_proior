# TIER: strong
# Pin every distance variable to 0.5 (all prototypes on the cost frontier), then seed
# a spread that INVERTS the surrogate's bias: pick target trade-off angles evenly in
# [0,1] and set each position variable x_j = target ** (1/alpha), so theta_j spreads
# evenly on the frontier (undoing the (x**alpha) warp). Finally run coordinate-wise
# LOCAL HYPERVOLUME ASCENT: nudge each configuration's position coordinates
# (multi-scale steps) and keep a move only if it increases the batch's exact dominated
# hypervolume. This reshapes the seed into a reference-tuned spread capturing
# materially more volume than the uniform grid -- yet still short of the (intractable)
# global optimum, so it leaves headroom.
import sys, json, math

inst = json.load(sys.stdin)
n = inst["n"]
M = inst["M"]
n_pos = inst["n_pos"]
alpha = inst["alpha"]
budget = inst["budget"]
ref = inst["ref"]


def surrogate(x):
    g = 0.0
    for i in range(n_pos, n):
        d = x[i] - 0.5
        g += d * d
    theta = [(max(0.0, xi) ** alpha) * (math.pi / 2.0) for xi in x[:n_pos]]
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


def hv(costs):
    if M == 2:
        pts = [c[:2] for c in costs if c[0] < ref[0] and c[1] < ref[1]]
        if not pts:
            return 0.0
        return area2d(pts)
    pts = [c for c in costs if c[0] < ref[0] and c[1] < ref[1] and c[2] < ref[2]]
    if not pts:
        return 0.0
    pts.sort(key=lambda p: p[2])
    vol = 0.0
    m = len(pts)
    for i in range(m):
        z0 = pts[i][2]
        z1 = pts[i + 1][2] if i + 1 < m else ref[2]
        dz = z1 - z0
        if dz <= 0.0:
            continue
        vol += area2d([p[:2] for p in pts[:i + 1]]) * dz
    return vol


def invert(t):
    # target angle fraction t in [0,1] -> position variable value
    return min(1.0, max(0.0, t ** (1.0 / alpha)))


# ---- bias-inverted seeding ----
pos = []
if n_pos == 1:
    for a in range(budget):
        t = (a + 0.5) / budget
        pos.append([invert(t)])
else:
    side = max(2, int(budget ** 0.5))
    for a in range(side):
        for b in range(side):
            if len(pos) >= budget:
                break
            pos.append([invert((a + 0.5) / side), invert((b + 0.5) / side)])


def full_hv():
    return hv([surrogate(list(p) + [0.5] * (n - n_pos)) for p in pos])


cur = full_hv()
for _ in range(60):
    improved = False
    for i in range(len(pos)):
        for d in range(n_pos):
            for step in (0.08, -0.08, 0.03, -0.03, 0.01, -0.01):
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

pts = [list(p) + [0.5] * (n - n_pos) for p in pos]
print(json.dumps({"points": pts}))
