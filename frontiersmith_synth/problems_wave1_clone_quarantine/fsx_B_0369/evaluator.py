import sys, json, math, random, isorun

# ==========================================================================
# fsx_B_0369 -- feasibility-gated-structural (Format B, isolated candidate)
# Theme: "wind-farm turbines" -- size the lattice members of a 2D pin-jointed
# steel support TOWER that carries a wind-turbine nacelle at its top. The
# turbine applies a horizontal rotor THRUST (wind) plus its dead WEIGHT at hub
# height. Choose the cross-sectional area of every lattice member so the tower
# is as LIGHT as possible while staying within BOTH a material stress limit
# and a maximum top-of-tower sway (displacement) limit.
# Objective: MINIMIZE total steel mass. An infeasible design scores 0.
# The evaluator runs a pure-python 2D truss FEM to grade every design.
#
# The tower is a vertical Pratt/Warren lattice: left + right chords, a
# horizontal tie at every level, and one diagonal per bay. Base nodes are
# pinned. It is statically determinate, so member forces are area-independent;
# the winning strategy sizes each member to its own load path instead of
# thickening the whole tower uniformly.
# ==========================================================================


def _fem(nodes, bars, fixed, loads, E, areas):
    """Linear-elastic 2D pin-jointed truss. Returns (u, stresses, lengths) or
    None if the assembled free system is singular."""
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


def _geometry(nstories, w, h):
    """Vertical lattice tower. Left chord at x=0, right chord at x=w; a level
    every h metres. Both base nodes pinned. Returns (nodes,bars,fixed,topL,topR)."""
    nodes = []
    for lvl in range(nstories + 1):
        nodes.append([0.0, lvl * h])          # left chord node -> index 2*lvl
        nodes.append([w, lvl * h])            # right chord node -> index 2*lvl+1
    Ln = lambda l: 2 * l
    Rn = lambda l: 2 * l + 1
    bars = []
    for l in range(nstories):
        bars.append([Ln(l), Ln(l + 1)])      # left chord segments
    for l in range(nstories):
        bars.append([Rn(l), Rn(l + 1)])      # right chord segments
    for l in range(1, nstories + 1):
        bars.append([Ln(l), Rn(l)])          # horizontal ties (above the base)
    for l in range(nstories):
        bars.append([Ln(l), Rn(l + 1)])      # diagonals (one per bay)
    fixed = [2 * Ln(0), 2 * Ln(0) + 1, 2 * Rn(0), 2 * Rn(0) + 1]  # base pinned
    return nodes, bars, fixed, Ln(nstories), Rn(nstories)


def make_instances():
    E = 200e9
    rho = 7850.0
    sigma = 250e6
    a_min = 1e-4
    a_max = 2e-2
    w = 2.5
    # (nstories, story_height, thrust_kN, weight_kN, sway-slack df):
    #   df >= 1  -> the stress limit governs (per-member sizing wins big)
    #   df <  1  -> the top-sway limit governs (must add stiffness material)
    specs = [
        (4, 3.0, 40.0, 60.0, 2.0),
        (5, 3.0, 45.0, 60.0, 1.4),
        (6, 3.0, 42.0, 60.0, 1.0),
        (7, 3.0, 50.0, 70.0, 0.8),
        (8, 3.0, 48.0, 70.0, 0.6),
        (4, 3.2, 52.0, 65.0, 1.2),
        (5, 2.8, 38.0, 55.0, 0.7),
        (6, 3.1, 58.0, 72.0, 2.0),
        (7, 2.9, 47.0, 68.0, 0.9),
        (8, 3.0, 53.0, 75.0, 0.6),
    ]
    out = []
    for si, (nstories, h, Pk, Wk, df) in enumerate(specs):
        rng = random.Random(6900 + si)
        h = round(h + 0.05 * rng.random(), 4)
        P = 1000.0 * (Pk + 4.0 * rng.random())     # horizontal rotor thrust
        Wt = 1000.0 * (Wk + 4.0 * rng.random())    # nacelle + rotor dead weight
        nodes, bars, fixed, topL, topR = _geometry(nstories, w, h)
        loads = [[2 * topL, P / 2.0], [2 * topR, P / 2.0],
                 [2 * topL + 1, -Wt / 2.0], [2 * topR + 1, -Wt / 2.0]]
        # reference solve at a_max to calibrate a public top-sway limit u_max
        u, st, Ls = _fem(nodes, bars, fixed, loads, E, [a_max] * len(bars))
        S_ref = max(abs(x) for x in st)
        U_ref = max(abs(x) for x in u)
        a_stress = S_ref * a_max / sigma            # uniform area if stress-limited
        u_at_stress = U_ref * a_max / a_stress      # top sway at that uniform area
        u_max = round(df * u_at_stress, 8)
        pub = {
            "nodes": nodes, "bars": bars, "fixed": fixed, "loads": loads,
            "E": E, "rho": rho, "sigma": sigma,
            "a_min": a_min, "a_max": a_max, "u_max": u_max,
        }
        out.append({"public": pub, "hidden": {}})
    return out


def _uniform_star(pub):
    """Smallest single (uniform) area that satisfies BOTH limits, and its mass."""
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
        if isinstance(v, bool) or not isinstance(v, (int, float)):
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
