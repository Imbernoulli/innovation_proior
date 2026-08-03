import sys, json, math, random, isorun

# ==========================================================================
# fsx_B_0361 -- feasibility-gated-structural (Format B, isolated candidate)
# Theme: "orbital debris cleanup".
#
# A servicer satellite extends a long 2D pin-jointed CAPTURE BOOM (a Pratt
# cantilever truss anchored at the servicer bus) to reach out and grapple a
# tumbling piece of debris. The boom must survive MULTIPLE capture load
# cases simultaneously:
#   LC0  net-drag: the capture net snags the debris, dragging the tip down.
#   LC1  de-tumble recoil: reeling the debris in pulls the tip back toward
#        the bus (axial) while the reaction lifts the top chord.
# For EVERY load case the axial stress in every strut must stay within the
# material limit AND the grapple head (the two tip nodes) must not drift
# beyond a pointing/displacement budget u_max (or it loses the target).
#
# The solver chooses a cross-section area for each of the ~45 struts.
# Objective: MINIMIZE total structural mass. Any design that is infeasible
# under ANY load case scores 0. A pure-python 2D truss FEM grades every
# design deterministically -- no wall-clock, no randomness at score time.
# ==========================================================================


def _fem(nodes, bars, fixed, loads, E, areas):
    """Linear-elastic 2D pin-jointed truss. Returns (u, stresses, lengths)
    or None if the reduced stiffness matrix is singular.
    u: 2*N nodal displacements; stresses: axial stress per bar."""
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


def _geometry(nbays, h, Lb=2.0):
    """Pratt cantilever boom anchored at the servicer bus (left face)."""
    nodes = []
    for i in range(nbays + 1):
        nodes.append([i * Lb, 0.0])          # bottom chord
    for i in range(nbays + 1):
        nodes.append([i * Lb, h])            # top chord
    bot = lambda i: i
    top = lambda i: nbays + 1 + i
    bars = []
    for i in range(nbays):
        bars.append([bot(i), bot(i + 1)])    # bottom chord
    for i in range(nbays):
        bars.append([top(i), top(i + 1)])    # top chord
    for i in range(nbays + 1):
        bars.append([bot(i), top(i)])        # verticals
    for i in range(nbays):
        bars.append([top(i), bot(i + 1)])    # Pratt diagonals
    fixed = [2 * bot(0), 2 * bot(0) + 1, 2 * top(0), 2 * top(0) + 1]
    return nodes, bars, fixed, bot, top


def _worst(nodes, bars, fixed, load_cases, E, areas, monitor):
    """Worst axial stress and worst monitored-node displacement magnitude
    across ALL load cases. Returns (Smax, Umax, lengths) or None."""
    Smax = 0.0
    Umax = 0.0
    lengths = None
    for lc in load_cases:
        res = _fem(nodes, bars, fixed, lc, E, areas)
        if res is None:
            return None
        u, st, Ls = res
        lengths = Ls
        for v in st:
            av = abs(v)
            if av > Smax:
                Smax = av
        for nd in monitor:
            d = math.hypot(u[2 * nd], u[2 * nd + 1])
            if d > Umax:
                Umax = d
    return Smax, Umax, lengths


def make_instances():
    E = 200e9
    rho = 7850.0
    sigma = 250e6
    a_min = 1e-4
    a_max = 2e-2
    # (nbays, height, drag kN P1, axial kN P2, lift kN P3, disp-slack df):
    #   df >= 1  -> stress governs  (per-strut sizing wins big)
    #   df <  1  -> pointing/displacement governs (must add stiffening mass)
    specs = [
        (11, 1.8, 40.0, 24.0, 15.0, 1.6),
        (11, 1.9, 46.0, 30.0, 18.0, 0.7),
        (12, 2.0, 38.0, 22.0, 14.0, 1.1),
        (12, 1.7, 52.0, 34.0, 20.0, 0.6),
        (13, 2.1, 44.0, 26.0, 16.0, 1.3),
        (11, 2.0, 50.0, 28.0, 19.0, 0.85),
        (13, 1.8, 42.0, 32.0, 17.0, 1.0),
        (12, 1.9, 48.0, 25.0, 21.0, 0.65),
        (14, 2.0, 40.0, 30.0, 15.0, 1.4),
        (14, 1.7, 55.0, 36.0, 22.0, 0.75),
    ]
    out = []
    for si, (nbays, h, P1k, P2k, P3k, df) in enumerate(specs):
        rng = random.Random(36100 + si)
        h = round(h + 0.05 * rng.random(), 4)
        P1 = 1000.0 * (P1k + 3.0 * rng.random())
        P2 = 1000.0 * (P2k + 2.0 * rng.random())
        P3 = 1000.0 * (P3k + 2.0 * rng.random())
        nodes, bars, fixed, bot, top = _geometry(nbays, h)
        tb = bot(nbays)
        tt = top(nbays)
        load_cases = [
            [[2 * tb + 1, -P1]],                       # LC0 net-drag (down)
            [[2 * tb, -P2], [2 * tt + 1, P3]],         # LC1 de-tumble recoil
        ]
        monitor = [tb, tt]
        # reference solve at a_max to calibrate the pointing budget u_max
        w = _worst(nodes, bars, fixed, load_cases, E, [a_max] * len(bars), monitor)
        S_ref, U_ref, _ = w
        a_stress = S_ref * a_max / sigma            # uniform area if stress-limited
        u_at_stress = U_ref * a_max / a_stress      # tip drift at that uniform area
        u_max = round(df * u_at_stress, 8)
        pub = {
            "nodes": nodes, "bars": bars, "fixed": fixed,
            "load_cases": load_cases, "monitor": monitor,
            "E": E, "rho": rho, "sigma": sigma,
            "a_min": a_min, "a_max": a_max, "u_max": u_max,
        }
        out.append({"public": pub, "hidden": {}})
    return out


def _uniform_star_mass(pub):
    """Mass of the smallest single (uniform) area feasible over all LCs."""
    w = _worst(pub["nodes"], pub["bars"], pub["fixed"], pub["load_cases"],
               pub["E"], [pub["a_max"]] * len(pub["bars"]), pub["monitor"])
    S_ref, U_ref, Ls = w
    a_stress = S_ref * pub["a_max"] / pub["sigma"]
    a_disp = U_ref * pub["a_max"] / pub["u_max"]
    a = min(max(a_stress, a_disp), pub["a_max"])
    return a * pub["rho"] * sum(Ls)


def baseline(inst):
    return _uniform_star_mass(inst["public"])


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
    w = _worst(pub["nodes"], bars, pub["fixed"], pub["load_cases"],
               pub["E"], clean, pub["monitor"])
    if w is None:
        return False, 0.0
    Smax, Umax, Ls = w
    if Smax > pub["sigma"] * 1.001 or Umax > pub["u_max"] * 1.001:
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
