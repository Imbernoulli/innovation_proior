import sys, json, math, random, isorun

# ==========================================================================
# fsx_B_0151 -- feasibility-gated-structural (Format B, isolated candidate)
# Theme: "mountain rescue relays" -- size the load-bearing struts of a
# cantilevered relay gantry (a 2D pin-jointed truss anchored to a cliff)
# so it carries the relay/antenna payload at its tip while staying within
# the material stress limit AND a maximum tip-sway (displacement) limit.
# Objective: MINIMIZE total steel mass. An infeasible design scores 0.
# The evaluator runs a pure-python 2D truss FEM to grade every design.
# ==========================================================================


def _fem(nodes, bars, fixed, loads, E, areas):
    """Linear-elastic 2D pin-jointed truss. Returns (u, stresses, lengths).
    u: list of 2*N nodal displacements; stresses: axial stress per bar."""
    N = len(nodes)
    ndof = 2 * N
    K = [[0.0] * ndof for _ in range(ndof)]
    lengths = []
    dircos = []
    for (a, b), A in zip(bars, areas):
        x1, y1 = nodes[a]
        x2, y2 = nodes[b]
        L = math.hypot(x2 - x1, y2 - y1)
        lengths.append(L)
        c = (x2 - x1) / L
        s = (y2 - y1) / L
        dircos.append((c, s))
        k = E * A / L
        idx = [2 * a, 2 * a + 1, 2 * b, 2 * b + 1]
        cc, ss, cs = c * c, s * s, c * s
        ke = [[cc, cs, -cc, -cs],
              [cs, ss, -cs, -ss],
              [-cc, -cs, cc, cs],
              [-cs, -ss, cs, ss]]
        for ii in range(4):
            for jj in range(4):
                K[idx[ii]][idx[jj]] += k * ke[ii][jj]
    F = [0.0] * ndof
    for d, v in loads:
        F[d] += v
    fixedset = set(fixed)
    free = [d for d in range(ndof) if d not in fixedset]
    # dense Gaussian elimination on the free sub-system
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


def _geometry(nbays, h):
    """Cantilever Warren-style truss anchored at the left (cliff) face."""
    Lb = 2.0
    nodes = []
    for i in range(nbays + 1):
        nodes.append([i * Lb, 0.0])          # bottom chord
    for i in range(nbays + 1):
        nodes.append([i * Lb, h])            # top chord
    bot = lambda i: i
    top = lambda i: nbays + 1 + i
    bars = []
    for i in range(nbays):
        bars.append([bot(i), bot(i + 1)])    # bottom chord bars
    for i in range(nbays):
        bars.append([top(i), top(i + 1)])    # top chord bars
    for i in range(nbays + 1):
        bars.append([bot(i), top(i)])        # verticals
    for i in range(nbays):
        bars.append([bot(i), top(i + 1)])    # diagonals
    fixed = [2 * bot(0), 2 * bot(0) + 1, 2 * top(0), 2 * top(0) + 1]
    return nodes, bars, fixed


def make_instances():
    E = 200e9
    rho = 7850.0
    sigma = 250e6
    a_min = 1e-4
    a_max = 2e-2
    # (nbays, height, load kN, disp-slack factor df):
    #   df >= 1  -> stress governs (fully-stressed design wins big)
    #   df <  1  -> tip-sway governs (must add material for stiffness)
    specs = [
        (4, 1.8, 45.0, 2.0),
        (5, 1.7, 50.0, 1.4),
        (6, 2.0, 42.0, 1.0),
        (7, 1.6, 55.0, 0.8),
        (8, 1.9, 48.0, 0.6),
        (4, 2.0, 52.0, 1.2),
        (5, 1.8, 40.0, 0.7),
        (6, 1.7, 58.0, 2.0),
        (7, 2.0, 47.0, 0.9),
        (8, 1.6, 53.0, 0.6),
    ]
    out = []
    for si, (nbays, h, Pk, df) in enumerate(specs):
        rng = random.Random(9000 + si)
        h = round(h + 0.05 * rng.random(), 4)
        P = 1000.0 * (Pk + 3.0 * rng.random())
        nodes, bars, fixed = _geometry(nbays, h)
        loads = [[2 * nbays + 1, -P]]         # downward payload at the tip bottom node
        # reference solve at a_max to calibrate a public tip-sway limit u_max
        u, st, Ls = _fem(nodes, bars, fixed, loads, E, [a_max] * len(bars))
        S_ref = max(abs(x) for x in st)
        U_ref = max(abs(x) for x in u)
        a_stress = S_ref * a_max / sigma       # uniform area if stress-limited
        u_at_stress = U_ref * a_max / a_stress  # tip sway at that uniform area
        u_max = round(df * u_at_stress, 8)
        pub = {
            "nodes": nodes, "bars": bars, "fixed": fixed, "loads": loads,
            "E": E, "rho": rho, "sigma": sigma,
            "a_min": a_min, "a_max": a_max, "u_max": u_max,
        }
        out.append({"public": pub, "hidden": {}})
    return out


def _uniform_star(pub):
    """Smallest single (uniform) area that satisfies both limits, and its mass."""
    nodes, bars, fixed, loads = pub["nodes"], pub["bars"], pub["fixed"], pub["loads"]
    E, sigma, a_max, u_max, rho = pub["E"], pub["sigma"], pub["a_max"], pub["u_max"], pub["rho"]
    u, st, Ls = _fem(nodes, bars, fixed, loads, E, [a_max] * len(bars))
    S_ref = max(abs(x) for x in st)
    U_ref = max(abs(x) for x in u)
    a_stress = S_ref * a_max / sigma
    a_disp = U_ref * a_max / u_max
    a = min(max(a_stress, a_disp), a_max)
    return a, a * rho * sum(Ls)


def baseline(inst):
    _, W = _uniform_star(inst["public"])
    return W


def score(inst, ans):
    pub = inst["public"]
    bars = pub["bars"]
    m = len(bars)
    a_min, a_max = pub["a_min"], pub["a_max"]
    if not isinstance(ans, dict) or "areas" not in ans:
        return False, 0.0
    areas = ans["areas"]
    if not isinstance(areas, list) or len(areas) != m:
        return False, 0.0
    clean = []
    for v in areas:
        if not isinstance(v, (int, float)):
            return False, 0.0
        v = float(v)
        if v != v or v in (float("inf"), float("-inf")):
            return False, 0.0
        if v < a_min - 1e-12 or v > a_max + 1e-9:
            return False, 0.0
        clean.append(v)
    res = _fem(pub["nodes"], bars, pub["fixed"], pub["loads"], pub["E"], clean)
    if res is None:
        return False, 0.0
    u, st, Ls = res
    smax = max(abs(x) for x in st)
    umax = max(abs(x) for x in u)
    if smax > pub["sigma"] * 1.001 or umax > pub["u_max"] * 1.001:
        return False, 0.0
    W = sum(pub["rho"] * L * A for L, A in zip(Ls, clean))
    if W != W or W <= 0.0:
        return False, 0.0
    return True, W


def main():
    cand = sys.argv[1]
    insts = make_instances()
    vec = []
    for inst in insts:
        ans, stt = isorun.run_candidate(cand, inst["public"], timeout=20)
        if stt != "OK":
            vec.append(0.0)
            continue
        try:
            ok, obj = score(inst, ans)
        except Exception:
            ok = False
        if not ok:
            vec.append(0.0)
            continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0 <= r <= 1) else 0.0)
    ratio = sum(vec) / len(vec)
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


main()
