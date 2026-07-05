# TIER: strong
# Fully-stressed design: iterate the FEM and resize each pipe to |N_e|/sigma (member sized to its
# own force), then scale the whole design up just enough to satisfy the joint-deflection limit.
# Redistributes material (thick near the wall, thin at the tip) -> much lighter than uniform sizing,
# but still leaves headroom vs a true global optimum.
import sys, json, math

def fem(nodes, members, areas, fixed, loads, E):
    nn = len(nodes); ndof = 2 * nn
    K = [[0.0] * ndof for _ in range(ndof)]
    geo = []
    for e, (i, j) in enumerate(members):
        xi, yi = nodes[i]; xj, yj = nodes[j]
        dx = xj - xi; dy = yj - yi; L = math.hypot(dx, dy)
        c = dx / L; s = dy / L; geo.append((L, c, s))
        k = E * areas[e] / L
        d = (2 * i, 2 * i + 1, 2 * j, 2 * j + 1)
        km = ((c*c, c*s, -c*c, -c*s), (c*s, s*s, -c*s, -s*s),
              (-c*c, -c*s, c*c, c*s), (-c*s, -s*s, c*s, s*s))
        for a in range(4):
            for b in range(4):
                K[d[a]][d[b]] += k * km[a][b]
    fixed = set(fixed)
    free = [x for x in range(ndof) if x not in fixed]; nf = len(free)
    A = [[K[free[r]][free[c]] for c in range(nf)] for r in range(nf)]
    F = [float(loads.get(free[r], 0.0)) for r in range(nf)]
    for col in range(nf):
        piv = max(range(col, nf), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]; F[col], F[piv] = F[piv], F[col]
        pv = A[col][col]
        for r in range(nf):
            if r == col: continue
            f = A[r][col] / pv
            if f == 0.0: continue
            for cc in range(col, nf): A[r][cc] -= f * A[col][cc]
            F[r] -= f * F[col]
    u = [0.0] * ndof
    for r in range(nf): u[free[r]] = F[r] / A[r][r]
    N = []
    for e, (i, j) in enumerate(members):
        L, c, s = geo[e]
        du = (u[2*j]-u[2*i])*c + (u[2*j+1]-u[2*i+1])*s
        N.append(E * areas[e] / L * du)
    return u, N

def maxdisp(u):
    nn = len(u) // 2
    return max(math.hypot(u[2*i], u[2*i+1]) for i in range(nn))

inst = json.load(sys.stdin)
nodes = [tuple(v) for v in inst["nodes"]]
members = [tuple(v) for v in inst["members"]]
m = len(members)
fixed = inst["fixed_dofs"]; loads = {d: v for d, v in inst["loads"]}
E = inst["E"]; sig = inst["sigma_allow"]; dall = inst["disp_allow"]
Amin = inst["Amin"]; Amax = inst["Amax"]

areas = [Amax] * m
for _ in range(20):
    u, N = fem(nodes, members, areas, fixed, loads, E)
    areas = [min(Amax, max(Amin, abs(N[k]) / sig)) for k in range(m)]

# scale up until both stress and deflection limits hold (with a small safety margin)
for _ in range(30):
    u, N = fem(nodes, members, areas, fixed, loads, E)
    dm = maxdisp(u)
    smax = max(abs(N[k]) / areas[k] for k in range(m))
    if dm <= dall * (1 - 1e-7) and smax <= sig * (1 - 1e-9):
        break
    sc = max(dm / (dall * (1 - 1e-7)), smax / (sig * (1 - 1e-9)), 1.0) * 1.0000001
    areas = [min(Amax, a * sc) for a in areas]

areas = [max(Amin, min(Amax, a)) for a in areas]
print(json.dumps({"areas": areas}))
