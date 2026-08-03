# TIER: strong
# Sensitivity / virtual-work sizing. Because the tower is statically
# determinate, member forces are area-independent and the top sway equals
# exactly  u = sum_i (N_i n_i L_i)/(E A_i),  where N_i is the axial force under
# the real load and n_i under a unit load in the sway direction. Minimizing
# mass subject to a single displacement limit then has the closed form
# A_i proportional to sqrt(|N_i n_i|) (a Lagrange/virtual-work optimum), which
# spends stiffness only where it buys the most sway reduction per kilogram.
# We take the per-member MAX of the fully-stressed area and this sway-optimal
# area, and return whichever of {virtual-work design, uniform-scaled design}
# is lighter and feasible -> never worse than the greedy uniform strategy.
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
m = len(bars); E = inst["E"]; sigma = inst["sigma"]; rho = inst["rho"]
a_min = inst["a_min"]; a_max = inst["a_max"]; u_max = inst["u_max"]


def mass(areas):
    tot = 0.0
    for (a, b), A in zip(bars, areas):
        tot += rho * math.hypot(nodes[b][0] - nodes[a][0], nodes[b][1] - nodes[a][1]) * A
    return tot


def feasible(areas):
    res = fem(nodes, bars, fixed, loads, E, areas)
    if res is None:
        return False
    u, st, _ = res
    return (max(abs(v) for v in u) <= u_max * 0.999 and
            max(abs(v) for v in st) <= sigma * 0.999)


# reference solve -> real axial forces (area-independent for a determinate truss)
u0, st0, Ls = fem(nodes, bars, fixed, loads, E, [a_max] * m)
N = [s * a_max for s in st0]                      # axial force per member
Astress = [min(max(abs(nf) / sigma * 1.02, a_min), a_max) for nf in N]

# ---- candidate A: fully-stressed + uniform sway top-up (robust fallback) ----
gA = list(Astress)
for _ in range(400):
    if feasible(gA):
        break
    gA = [min(a * 1.02, a_max) for a in gA]

# ---- candidate B: virtual-work sway-optimal, combined with the stress design ----
# unit load in the real sway direction (horizontal, split over the two top nodes)
hloads = [[d, (1.0 if v > 0 else 0.0)] for d, v in loads if d % 2 == 0 and v != 0]
# normalize the unit load to total magnitude 1
tot = sum(v for _, v in hloads) or 1.0
hloads = [[d, v / tot] for d, v in hloads]
_, stv, _ = fem(nodes, bars, fixed, hloads, E, [a_max] * m)
nvir = [s * a_max for s in stv]                   # virtual forces
cvec = [abs(N[i] * nvir[i] * Ls[i] / E) for i in range(m)]
ssum = sum(math.sqrt(c) for c in cvec)
if ssum <= 0 or u_max <= 0:
    Asway = [a_min] * m
else:
    # scale so that sum_i c_i / A_i = u_max  with A_i ~ sqrt(c_i)
    Asway = [(math.sqrt(cvec[i]) * ssum / u_max) if cvec[i] > 0 else a_min for i in range(m)]
gB = [min(max(max(Astress[i], Asway[i]), a_min), a_max) for i in range(m)]
# tiny uniform top-up to absorb the max-displacement/vertical coupling
for _ in range(120):
    if feasible(gB):
        break
    gB = [min(a * 1.01, a_max) for a in gB]

# choose the lighter FEASIBLE design; fall back to the safe uniform one
best = None
for cand in (gB, gA):
    if feasible(cand):
        if best is None or mass(cand) < mass(best):
            best = cand
if best is None:
    best = [a_max] * m                            # last-resort feasible-ish
print(json.dumps({"areas": best}))
