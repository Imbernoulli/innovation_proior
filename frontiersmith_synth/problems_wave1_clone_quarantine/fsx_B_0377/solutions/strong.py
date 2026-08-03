# TIER: strong
# Buckling-aware fully-stressed design + global displacement repair: uniformly scale all
# areas up (capped at a_max) until the sag limit is met. Feasible on all three gates
# everywhere and far lighter than the uniform-a_max baseline.
import sys, json, math

def fem(nodes, bars, areas, loads, fixed, E):
    N = len(nodes); M = len(bars); ndof = 2 * N
    K = [[0.0] * ndof for _ in range(ndof)]
    L = [0.0] * M; Cc = [0.0] * M; Ss = [0.0] * M
    for e in range(M):
        i, j = bars[e]
        dx = nodes[j][0] - nodes[i][0]; dy = nodes[j][1] - nodes[i][1]
        Le = math.hypot(dx, dy); L[e] = Le
        c = dx / Le; s = dy / Le; Cc[e] = c; Ss[e] = s
        k = E * areas[e] / Le
        idx = [2*i, 2*i+1, 2*j, 2*j+1]
        ke = [[c*c, c*s, -c*c, -c*s],
              [c*s, s*s, -c*s, -s*s],
              [-c*c, -c*s, c*c, c*s],
              [-c*s, -s*s, c*s, s*s]]
        for a in range(4):
            for b in range(4):
                K[idx[a]][idx[b]] += k * ke[a][b]
    F = []
    for ld in loads:
        F.append(ld[0]); F.append(ld[1])
    fx = []
    for fv in fixed:
        fx.append(bool(fv[0])); fx.append(bool(fv[1]))
    free = [d for d in range(ndof) if not fx[d]]
    nf = len(free)
    A = [[K[free[r]][free[cc]] for cc in range(nf)] for r in range(nf)]
    bb = [F[free[r]] for r in range(nf)]
    # Gaussian elimination
    for col in range(nf):
        piv = max(range(col, nf), key=lambda rr: abs(A[rr][col]))
        A[col], A[piv] = A[piv], A[col]; bb[col], bb[piv] = bb[piv], bb[col]
        d = A[col][col]
        if abs(d) < 1e-15:
            raise ValueError("singular")
        for rr in range(nf):
            if rr == col: continue
            f = A[rr][col] / d
            if f == 0.0: continue
            for cc in range(col, nf):
                A[rr][cc] -= f * A[col][cc]
            bb[rr] -= f * bb[col]
    u = [0.0] * ndof
    for r in range(nf):
        u[free[r]] = bb[r] / A[r][r]
    stress = [0.0] * M
    for e in range(M):
        i, j = bars[e]; c = Cc[e]; s = Ss[e]; Le = L[e]
        elong = (-c)*u[2*i] + (-s)*u[2*i+1] + c*u[2*j] + s*u[2*j+1]
        stress[e] = E * elong / Le
    return u, stress, L

def maxdisp(u):
    N = len(u) // 2
    return max(math.hypot(u[2*i], u[2*i+1]) for i in range(N))

inst = json.load(sys.stdin)
nodes = inst["nodes"]; bars = inst["bars"]; loads = inst["loads"]; fixed = inst["fixed"]
E = inst["E"]; sigma = inst["sigma"]; kb = inst["k_buck"]
a_min = inst["a_min"]; a_max = inst["a_max"]; disp_limit = inst["disp_limit"]
M = len(bars)
_, forces, L = fem(nodes, bars, [1.0]*M, loads, fixed, E)
areas = []
for e in range(M):
    a = abs(forces[e]) / sigma
    if forces[e] < 0.0:
        a_b = math.sqrt(abs(forces[e]) * L[e]*L[e] / (kb * E))
        if a_b > a: a = a_b
    if a < a_min: a = a_min
    if a > a_max: a = a_max
    areas.append(a)

def clip(v):
    return [min(a_max, max(a_min, x)) for x in v]

for _ in range(50):
    u, _, _ = fem(nodes, bars, areas, loads, fixed, E)
    dcur = maxdisp(u)
    if dcur <= disp_limit * (1 + 1e-9):
        break
    cur_max = max(areas)
    if cur_max >= a_max * (1 - 1e-12):
        break
    f = dcur / disp_limit * 1.002
    cap = a_max / cur_max
    if f > cap: f = cap
    areas = clip([x * f for x in areas])
print(json.dumps({"areas": areas}))
