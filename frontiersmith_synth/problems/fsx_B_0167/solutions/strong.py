# TIER: strong
# Three-gate feasible sizing. (1) Run one FEM to recover the (area-independent) member
# forces. (2) Floor every bar to the larger of its two per-member minima:
#       yield min      a_yield_e    = |force_e| / sigma
#       buckling min   a_buckle_e   = sqrt(|force_e| * length_e^2 / (pi^2 * E * kappa))   (compression only)
#    clamped to [a_min, a_max]. This is the lightest design that satisfies BOTH the yield
#    and buckling gates member-by-member. (3) Sag repair: because the truss is determinate,
#    scaling ALL areas up by a factor divides every joint displacement by that factor while
#    only relaxing the stress/buckling gates, so if the design still sags past disp_limit we
#    scale all areas by (max_disp/disp_limit)*1.01 and re-check -- converging in a couple of
#    passes. The result is feasible on all three gates and far lighter than the uniform a_max
#    baseline; on loose instances it reduces to the plain fully-stressed design.
import sys, json, math

inst = json.load(sys.stdin)
nodes = inst["nodes"]; bars = inst["bars"]; loads = inst["loads"]; fixed = inst["fixed"]
E = inst["E"]; sigma = inst["sigma"]; a_min = inst["a_min"]; a_max = inst["a_max"]
kappa = inst["kappa"]; disp_limit = inst["disp_limit"]
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


def max_disp(u):
    return max(math.hypot(u[2 * n], u[2 * n + 1]) for n in range(N))


u, stress, length = fem([a_max] * M)
forces = [stress[e] * a_max for e in range(M)]
areas = []
for e in range(M):
    ya = abs(forces[e]) / sigma
    ba = math.sqrt(abs(forces[e]) * length[e] ** 2 / (math.pi ** 2 * E * kappa)) \
        if forces[e] < 0 else 0.0
    areas.append(min(max(max(ya, ba), a_min), a_max))

for _ in range(40):
    u, _, _ = fem(areas)
    md = max_disp(u)
    if md <= disp_limit * (1 - 1e-4):
        break
    factor = (md / disp_limit) * 1.01
    areas = [min(a * factor, a_max) for a in areas]

print(json.dumps({"areas": areas}))
