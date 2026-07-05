# TIER: strong
# Iterated fully-stressed design with force redistribution PLUS an explicit sag-gate
# repair. Because the X-braced frame is statically indeterminate, member forces move
# as the areas change, so a single fully-stressed pass is not self-consistent. Here we
# re-analyse and re-size repeatedly: area_e <- |force_e|/sigma using the CURRENT
# forces, with gentle damping so the redistribution converges to a fixed point where
# every member sits at (or below) yield. A safety margin (0.97) keeps us just inside
# the yield gate against tiny residual redistribution. Then, if the resulting deck sag
# exceeds the service limit, uniformly scale all areas up by the measured overshoot
# and re-analyse until the sag gate is met. This stays feasible on BOTH gates while
# far lighter than uniform max sizing; on slack-sag instances it reduces to the pure
# fully-stressed design.
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


target = sigma * 0.97  # stay just inside yield against residual redistribution
areas = [a_max] * M
# iterated fully-stressed design (damped fixed-point over force redistribution)
for _ in range(60):
    u, st = fem(areas)
    new = []
    for e in range(M):
        force = st[e] * areas[e]
        want = abs(force) / target
        # damp toward the target area to converge the indeterminate redistribution
        ae = 0.5 * areas[e] + 0.5 * want
        new.append(min(max(ae, a_min), a_max))
    if max(abs(new[e] - areas[e]) for e in range(M)) < 1e-9:
        areas = new
        break
    areas = new

# make sure we are strictly stress-feasible after convergence
for _ in range(20):
    u, st = fem(areas)
    worst = max(abs(st[e]) for e in range(M))
    if worst <= sigma * (1 - 1e-4):
        break
    factor = (worst / sigma) * 1.01
    areas = [min(max(a * factor, a_min), a_max) for a in areas]

# sag-gate repair
for _ in range(60):
    u, st = fem(areas)
    md = max_disp(u)
    if md <= disp_limit * (1 - 1e-4):
        break
    factor = (md / disp_limit) * 1.02
    areas = [min(max(a * factor, a_min), a_max) for a in areas]

print(json.dumps({"areas": areas}))
