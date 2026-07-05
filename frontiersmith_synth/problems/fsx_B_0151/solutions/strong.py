# TIER: strong
# Iterative fully-stressed design (FSD): repeatedly re-solve the FEM and resize
# every strut to its CURRENT axial force so material tracks the redistributed
# load path, then add just enough uniform stiffness to satisfy the tip-sway
# limit. Converges near the stress-optimal mass with headroom to the true
# (sensitivity-optimal) minimum.
import sys, json, math


def fem(nodes, bars, fixed, loads, E, areas):
    N = len(nodes); ndof = 2 * N
    K = [[0.0] * ndof for _ in range(ndof)]
    lengths = []; dircos = []
    for (a, b), A in zip(bars, areas):
        x1, y1 = nodes[a]; x2, y2 = nodes[b]
        L = math.hypot(x2 - x1, y2 - y1); lengths.append(L)
        c = (x2 - x1) / L; s = (y2 - y1) / L; dircos.append((c, s))
        k = E * A / L; idx = [2 * a, 2 * a + 1, 2 * b, 2 * b + 1]
        cc, ss, cs = c * c, s * s, c * s
        ke = [[cc, cs, -cc, -cs], [cs, ss, -cs, -ss],
              [-cc, -cs, cc, cs], [-cs, -ss, cs, ss]]
        for ii in range(4):
            for jj in range(4):
                K[idx[ii]][idx[jj]] += k * ke[ii][jj]
    F = [0.0] * ndof
    for d, v in loads:
        F[d] += v
    fixedset = set(fixed)
    free = [d for d in range(ndof) if d not in fixedset]
    n = len(free)
    A_ = [[K[free[i]][free[j]] for j in range(n)] for i in range(n)]
    bvec = [F[free[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(A_[r][col]))
        if abs(A_[piv][col]) < 1e-18:
            return None
        if piv != col:
            A_[col], A_[piv] = A_[piv], A_[col]
            bvec[col], bvec[piv] = bvec[piv], bvec[col]
        pv = A_[col][col]
        for r in range(col + 1, n):
            f = A_[r][col] / pv
            if f == 0.0:
                continue
            for cc2 in range(col, n):
                A_[r][cc2] -= f * A_[col][cc2]
            bvec[r] -= f * bvec[col]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        acc = bvec[i]
        for j in range(i + 1, n):
            acc -= A_[i][j] * x[j]
        x[i] = acc / A_[i][i]
    u = [0.0] * ndof
    for i, d in enumerate(free):
        u[d] = x[i]
    stresses = []
    for (a, b), (c, s), L in zip(bars, dircos, lengths):
        elong = (u[2 * b] - u[2 * a]) * c + (u[2 * b + 1] - u[2 * a + 1]) * s
        stresses.append(E * elong / L)
    return u, stresses, lengths


inst = json.load(sys.stdin)
nodes = inst["nodes"]; bars = inst["bars"]; fixed = inst["fixed"]; loads = inst["loads"]
m = len(bars); E = inst["E"]; sigma = inst["sigma"]
a_min = inst["a_min"]; a_max = inst["a_max"]; u_max = inst["u_max"]

areas = [a_max] * m
# fully-stressed fixed-point iteration
for _ in range(60):
    res = fem(nodes, bars, fixed, loads, E, areas)
    if res is None:
        break
    u, st, Ls = res
    new = []
    for a_i, s_i in zip(areas, st):
        target = abs(s_i) / sigma * a_i * 1.02   # drive |stress| toward sigma
        new.append(min(max(target, a_min), a_max))
    if max(abs(nw - a_i) for nw, a_i in zip(new, areas)) < 1e-9:
        areas = new
        break
    areas = new

# add uniform stiffness until tip-sway (and any residual stress) is satisfied
for _ in range(120):
    res = fem(nodes, bars, fixed, loads, E, areas)
    u, st, _ = res
    umax = max(abs(v) for v in u)
    smax = max(abs(v) for v in st)
    if umax <= u_max * 0.9995 and smax <= sigma * 0.9995:
        break
    areas = [min(a * 1.03, a_max) for a in areas]

print(json.dumps({"areas": areas}))
