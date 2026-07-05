import sys, json, math, random, isorun

# ------------------------------------------------------------------ FEM (pure python, deterministic)
def _fem(nodes, members, areas, fixed, loads, E):
    """Return (u, N, geo) or None if singular. geo[e]=(L,c,s)."""
    nn = len(nodes); ndof = 2 * nn
    K = [[0.0] * ndof for _ in range(ndof)]
    geo = []
    for e, (i, j) in enumerate(members):
        xi, yi = nodes[i]; xj, yj = nodes[j]
        dx = xj - xi; dy = yj - yi; L = math.hypot(dx, dy)
        if L <= 0.0:
            return None
        c = dx / L; s = dy / L
        geo.append((L, c, s))
        k = E * areas[e] / L
        dofs = (2 * i, 2 * i + 1, 2 * j, 2 * j + 1)
        km = ((c * c, c * s, -c * c, -c * s),
              (c * s, s * s, -c * s, -s * s),
              (-c * c, -c * s, c * c, c * s),
              (-c * s, -s * s, c * s, s * s))
        for a in range(4):
            Ka = K[dofs[a]]
            for b in range(4):
                Ka[dofs[b]] += k * km[a][b]
    fixed = set(fixed)
    free = [d for d in range(ndof) if d not in fixed]
    nf = len(free)
    A = [[K[free[r]][free[c]] for c in range(nf)] for r in range(nf)]
    F = [float(loads.get(free[r], 0.0)) for r in range(nf)]
    # Gaussian elimination with partial pivoting
    for col in range(nf):
        piv = max(range(col, nf), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) < 1e-14:
            return None
        A[col], A[piv] = A[piv], A[col]; F[col], F[piv] = F[piv], F[col]
        pv = A[col][col]
        for r in range(nf):
            if r == col:
                continue
            f = A[r][col] / pv
            if f == 0.0:
                continue
            Ar = A[r]; Ac = A[col]
            for cc in range(col, nf):
                Ar[cc] -= f * Ac[cc]
            F[r] -= f * F[col]
    u = [0.0] * ndof
    for r in range(nf):
        u[free[r]] = F[r] / A[r][r]
    N = []
    for e, (i, j) in enumerate(members):
        L, c, s = geo[e]
        du = (u[2 * j] - u[2 * i]) * c + (u[2 * j + 1] - u[2 * i + 1]) * s
        N.append(E * areas[e] / L * du)
    return u, N, geo


def _maxdisp(u):
    nn = len(u) // 2
    return max(math.hypot(u[2 * i], u[2 * i + 1]) for i in range(nn))


def _build(nb, H, L):
    nodes = [(i * L, 0.0) for i in range(nb + 1)] + [(i * L, H) for i in range(nb + 1)]
    def b(i): return i
    def t(i): return (nb + 1) + i
    members = []
    for i in range(nb): members.append((b(i), b(i + 1)))       # bottom chord
    for i in range(nb): members.append((t(i), t(i + 1)))       # top chord
    for i in range(nb + 1): members.append((b(i), t(i)))       # verticals
    for i in range(nb): members.append((b(i), t(i + 1)))       # diagonals
    fixed = [2 * b(0), 2 * b(0) + 1, 2 * t(0), 2 * t(0) + 1]
    return nodes, members, fixed


# ------------------------------------------------------------------ instances (seeded, deterministic)
E_MOD = 2.0e11
RHO = 7850.0
SIGMA = 2.5e8
AMIN = 1.5e-4
AMAX = 3.0e-3

# (bays, load-pattern along bottom chord, target stress fraction at all-Amax, disp slack factor)
_CFGS = [
    (4, [1.0],            0.55, 2.5),
    (5, [1.0, 0.6],       0.50, 2.0),
    (5, [1.0],            0.60, 1.6),
    (6, [1.0, 0.8, 0.5],  0.50, 2.2),
    (6, [1.0],            0.55, 1.5),
    (7, [1.0, 0.7],       0.50, 2.5),
    (4, [1.0, 0.3],       0.60, 1.4),
    (7, [1.0],            0.45, 1.8),
]


def make_instances():
    out = []
    for si, (nb, lp, tf, df) in enumerate(_CFGS):
        nodes, members, fixed = _build(nb, 1.0, 1.0)
        m = len(members)
        loads0 = {}
        for i in range(1, nb + 1):
            loads0[2 * i + 1] = -float(lp[(i - 1) % len(lp)])   # downward at bottom joints
        allmax = [AMAX] * m
        r = _fem(nodes, members, allmax, fixed, loads0, E_MOD)
        u, N, geo = r
        r0 = max(abs(N[k]) / AMAX for k in range(m))
        alpha = tf * SIGMA / r0                                  # scale loads to hit target stress
        loads = {d: v * alpha for d, v in loads0.items()}
        u, N, geo = _fem(nodes, members, allmax, fixed, loads, E_MOD)
        disp_allow = df * _maxdisp(u)
        pub = {
            "nodes": [[x, y] for (x, y) in nodes],
            "members": [[i, j] for (i, j) in members],
            "fixed_dofs": list(fixed),
            "loads": [[d, loads[d]] for d in sorted(loads)],
            "E": E_MOD, "rho": RHO, "sigma_allow": SIGMA,
            "disp_allow": disp_allow, "Amin": AMIN, "Amax": AMAX,
        }
        out.append({"public": pub, "hidden": {}})
    return out


def _weight(geo, areas):
    return RHO * sum(areas[k] * geo[k][0] for k in range(len(areas)))


def baseline(inst):
    p = inst["public"]
    nodes = [tuple(v) for v in p["nodes"]]
    members = [tuple(v) for v in p["members"]]
    m = len(members)
    _, _, geo = _fem(nodes, members, [p["Amax"]] * m, p["fixed_dofs"],
                     {d: v for d, v in p["loads"]}, p["E"])
    return _weight(geo, [p["Amax"]] * m)


def score(inst, ans):
    p = inst["public"]
    members = p["members"]; m = len(members)
    Amin = p["Amin"]; Amax = p["Amax"]
    if not isinstance(ans, dict) or "areas" not in ans:
        return False, 0.0
    areas = ans["areas"]
    if not isinstance(areas, list) or len(areas) != m:
        return False, 0.0
    tol = 1e-9
    clean = []
    for a in areas:
        if not isinstance(a, (int, float)):
            return False, 0.0
        a = float(a)
        if a != a or a in (float("inf"), float("-inf")):
            return False, 0.0
        if a < Amin - tol or a > Amax + tol:
            return False, 0.0
        clean.append(min(Amax, max(Amin, a)))
    nodes = [tuple(v) for v in p["nodes"]]
    mem = [tuple(v) for v in members]
    loads = {d: v for d, v in p["loads"]}
    r = _fem(nodes, mem, clean, p["fixed_dofs"], loads, p["E"])
    if r is None:
        return False, 0.0
    u, N, geo = r
    for val in u:
        if val != val or val in (float("inf"), float("-inf")):
            return False, 0.0
    # stress feasibility
    for k in range(m):
        if abs(N[k]) / clean[k] > p["sigma_allow"] * (1.0 + 1e-6):
            return False, 0.0
    # displacement feasibility
    if _maxdisp(u) > p["disp_allow"] * (1.0 + 1e-6):
        return False, 0.0
    W = _weight(geo, clean)
    if W <= 0.0 or W != W or W == float("inf"):
        return False, 0.0
    return True, W


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, st = isorun.run_candidate(cand, inst["public"], timeout=20)
        if st != "OK":
            vec.append(0.0); continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok, obj = False, 0.0
        if not ok:
            vec.append(0.0); continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
