# TIER: greedy
# Fully-stressed design (single pass): analyse the uniform a_max trellis, read each
# member's axial force, then size that member to exactly meet the yield limit,
# area_e = |force_e| / sigma (clamped to [a_min, a_max]). This is much lighter than
# uniform sizing and always satisfies the STRESS gate, but it ignores the joint
# displacement (sag) gate: on instances with a tight sag limit the slimmed-down
# trellis sags too far and the whole design is rejected (score 0).
import sys, json, math

inst = json.load(sys.stdin)
nodes = inst["nodes"]; bars = inst["bars"]; loads = inst["loads"]
fixed = inst["fixed"]; E = inst["E"]; sigma = inst["sigma"]
a_min = inst["a_min"]; a_max = inst["a_max"]
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
    # Gaussian elimination on the reduced system
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


_, st = fem([a_max] * M)
areas = []
for e in range(M):
    force = st[e] * a_max
    ae = abs(force) / sigma
    areas.append(min(max(ae, a_min), a_max))
print(json.dumps({"areas": areas}))
