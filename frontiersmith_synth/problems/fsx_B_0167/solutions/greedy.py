# TIER: greedy
# Pure fully-stressed (yield-only) sizing. Run one FEM to recover the member forces
# (they are area-independent for this statically-determinate gantry), then size every bar
# to exactly its yield limit  area_e = |force_e| / sigma  (clamped to [a_min, a_max]).
# This is the lightest design that satisfies the YIELD gate -- but it ignores buckling and
# sag. On "loose" instances (stocky sections, generous clearance) it also happens to clear
# the buckling and sag gates and is very light -> high score. On "tight" instances the
# slender compression members BUCKLE and/or the thin gantry SAGS past disp_limit, so the
# whole design is rejected and scores 0.
import sys, json, math

inst = json.load(sys.stdin)
nodes = inst["nodes"]; bars = inst["bars"]; loads = inst["loads"]; fixed = inst["fixed"]
E = inst["E"]; sigma = inst["sigma"]; a_min = inst["a_min"]; a_max = inst["a_max"]
N = len(nodes); M = len(bars)


def fem(areas):
    ndof = 2 * N
    K = [[0.0] * ndof for _ in range(ndof)]
    geo = []
    for e in range(M):
        i, j = bars[e]
        dx = nodes[j][0] - nodes[i][0]; dy = nodes[j][1] - nodes[i][1]
        Le = math.hypot(dx, dy); c = dx / Le; s = dy / Le
        geo.append((c, s, Le))
        k = E * areas[e] / Le
        idx = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        ke = [[c * c, c * s, -c * c, -c * s],
              [c * s, s * s, -c * s, -s * s],
              [-c * c, -c * s, c * c, c * s],
              [-c * s, -s * s, c * s, s * s]]
        for a in range(4):
            row = K[idx[a]]
            for b in range(4):
                row[idx[b]] += k * ke[a][b]
    F = [0.0] * ndof
    for n in range(N):
        F[2 * n] = loads[n][0]; F[2 * n + 1] = loads[n][1]
    freefix = []
    for n in range(N):
        freefix.append(not fixed[n][0]); freefix.append(not fixed[n][1])
    free = [d for d in range(ndof) if freefix[d]]
    nf = len(free)
    A = [[K[free[a]][free[b]] for b in range(nf)] for a in range(nf)]
    rhs = [F[free[a]] for a in range(nf)]
    for col in range(nf):
        piv = max(range(col, nf), key=lambda rr: abs(A[rr][col]))
        A[col], A[piv] = A[piv], A[col]
        rhs[col], rhs[piv] = rhs[piv], rhs[col]
        d = A[col][col]
        for rr in range(nf):
            if rr == col:
                continue
            f = A[rr][col] / d
            if f == 0.0:
                continue
            for cc in range(col, nf):
                A[rr][cc] -= f * A[col][cc]
            rhs[rr] -= f * rhs[col]
    u = [0.0] * ndof
    for a in range(nf):
        u[free[a]] = rhs[a] / A[a][a]
    stress = []; length = []
    for e in range(M):
        i, j = bars[e]; c, s, Le = geo[e]
        elong = (-c) * u[2 * i] + (-s) * u[2 * i + 1] + c * u[2 * j] + s * u[2 * j + 1]
        stress.append(E * elong / Le); length.append(Le)
    return u, stress, length


u, stress, length = fem([a_max] * M)
forces = [stress[e] * a_max for e in range(M)]
areas = [min(max(abs(forces[e]) / sigma, a_min), a_max) for e in range(M)]
print(json.dumps({"areas": areas}))
