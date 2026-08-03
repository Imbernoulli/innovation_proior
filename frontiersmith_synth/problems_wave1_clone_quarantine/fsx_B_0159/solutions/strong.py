# TIER: strong
# Fully-stressed design PLUS an explicit displacement-gate repair. First size every
# member to the yield limit (area_e = |force_e|/sigma) -- for this determinate
# trellis that is the lightest stress-feasible sizing. Then, if the resulting joint
# sag exceeds the serviceability limit, uniformly scale all areas up by the measured
# overshoot and re-analyse until the sag gate is met (member forces are area-
# independent here, so stress stays feasible throughout). This keeps the design
# feasible on BOTH gates while staying far lighter than uniform max sizing; on
# slack-sag instances it reduces to the pure fully-stressed design.
import sys, json, math

inst = json.load(sys.stdin)
nodes = inst["nodes"]; bars = inst["bars"]; loads = inst["loads"]
fixed = inst["fixed"]; E = inst["E"]; sigma = inst["sigma"]
a_min = inst["a_min"]; a_max = inst["a_max"]; disp_limit = inst["disp_limit"]
N = len(nodes); M = len(bars); ndof = 2 * N


def fem(areas):
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
            for b in range(4):
                K[idx[a]][idx[b]] += k * ke[a][b]
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
    stress = []
    for e in range(M):
        i, j = bars[e]; c, s, Le = geo[e]
        elong = (-c) * u[2 * i] + (-s) * u[2 * i + 1] + c * u[2 * j] + s * u[2 * j + 1]
        stress.append(E * elong / Le)
    return u, stress


def max_disp(u):
    return max(math.hypot(u[2 * n], u[2 * n + 1]) for n in range(N))


# fully-stressed sizing (converges in one pass for a determinate truss; iterate a
# few times for robustness)
areas = [a_max] * M
for _ in range(6):
    u, st = fem(areas)
    areas = [min(max(abs(st[e] * areas[e]) / sigma, a_min), a_max) for e in range(M)]

# displacement-gate repair
for _ in range(40):
    u, st = fem(areas)
    md = max_disp(u)
    if md <= disp_limit * (1 - 1e-4):
        break
    factor = (md / disp_limit) * 1.02
    areas = [min(max(a * factor, a_min), a_max) for a in areas]

print(json.dumps({"areas": areas}))
