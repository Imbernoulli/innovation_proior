#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0175 -- "Bakery Mezzanine Rack-Gantry Sizing".

Family: feasibility-gated-structural (Frontier-Eng StructuralOptimization), skinned
as the steel rack-gantry that carries the pallet mezzanine in a bakery distribution
warehouse. The deck (top chord) holds pallets of flour and sugar (downward loads)
plus a lateral forklift/seismic nudge; the frame spans between a pinned column base
and a roller base. The frame is a 2D pin-jointed truss, but -- unlike a plain
triangulated beam -- every panel is X-BRACED (two crossing diagonals), so the truss
is STATICALLY INDETERMINATE: the redundant diagonals mean member forces DEPEND on
the chosen cross-sections, not just on the loads.

The candidate must choose a CROSS-SECTIONAL AREA for every member so the total steel
weight is minimized subject to HARD limits:
    * every member's axial stress magnitude <= sigma        (yield gate)
    * every joint's displacement magnitude  <= disp_limit   (deck-sag / service gate)
    * every area in [a_min, a_max]
This is a feasibility-GATED objective: a design that violates ANY constraint is worth
0 (infeasible), otherwise its worth is its weight. Because the frame is indeterminate
there is no easy optimum: a single-pass fully-stressed design sized from the uniform
analysis is wrong -- as members are slimmed the load redistributes toward the stiffer
(larger-area) members, pushing their stress back over yield. A good solver must
iterate the force redistribution to a fixed point AND trade weight against the sag
gate, whose tightness varies per instance.

The evaluator runs a pure-numpy linear-elastic FEM (direct stiffness method) in the
PARENT process; the candidate is an UNTRUSTED stdin->stdout program run in an ISOLATED
subprocess via `isorun`, so it only ever sees the public instance and can never reach
the evaluator's frames/FEM/scorer.

Scoring (deterministic; no wall-time):
  weight(areas) = sum_e areas[e] * length[e]                 (density := 1)
  baseline b    = weight of the uniform a_max design (heaviest, always feasible)
  For a feasible answer with objective obj:  r = min(1, 0.1 * b / obj)
  -> the trivial uniform-a_max design maps to exactly 0.1; a design k times lighter
     than baseline maps to min(1, 0.1*k). Infeasible / malformed answer -> 0.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import numpy as np
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)
    return nxt


# ----------------------------- geometry ------------------------------------
def _build_gantry(npanel, h=3.0, d=4.0):
    """X-braced statically-indeterminate 2D truss (a rack gantry).
    Bottom chord nodes 0..P at (i*d, 0); top chord (deck) nodes P+1..2P+1 at (i*d, h).
    Members per panel: bottom chord, top chord, verticals, AND both diagonals (X).
    With P panels: n=2(P+1) nodes, m=5P+1 members; pin+roller reactions r=3, so the
    internal indeterminacy degree m+r-2n = P > 0 (one redundant diagonal per panel)."""
    P = npanel
    def B(i): return i            # bottom node index
    def T(i): return (P + 1) + i  # top node index
    nodes = [[float(i * d), 0.0] for i in range(P + 1)] + \
            [[float(i * d), float(h)] for i in range(P + 1)]
    bars = []
    for i in range(P):
        bars.append([B(i), B(i + 1)])      # bottom chord
        bars.append([T(i), T(i + 1)])      # top chord (deck)
    for i in range(P + 1):
        bars.append([B(i), T(i)])          # verticals
    for i in range(P):
        bars.append([B(i), T(i + 1)])      # diagonal /
        bars.append([T(i), B(i + 1)])      # diagonal \  (X-brace -> redundant)
    return nodes, bars


# ----------------------------- frozen FEM scaffold -------------------------
def _fem(nodes, bars, areas, loads, fixed, E):
    """2D pin-jointed truss, linear elastic, direct stiffness method.
    Returns (u, stress, L). Raises on singular system."""
    nodes = np.asarray(nodes, float)
    N = len(nodes); M = len(bars); ndof = 2 * N
    K = np.zeros((ndof, ndof))
    L = np.zeros(M); Cc = np.zeros(M); Ss = np.zeros(M)
    for e in range(M):
        i, j = bars[e]
        dx = nodes[j, 0] - nodes[i, 0]; dy = nodes[j, 1] - nodes[i, 1]
        Le = math.hypot(dx, dy)
        L[e] = Le
        c = dx / Le; s = dy / Le; Cc[e] = c; Ss[e] = s
        k = E * areas[e] / Le
        idx = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
        ke = k * np.array([[c * c, c * s, -c * c, -c * s],
                           [c * s, s * s, -c * s, -s * s],
                           [-c * c, -c * s, c * c, c * s],
                           [-c * s, -s * s, c * s, s * s]])
        for a in range(4):
            for b in range(4):
                K[idx[a], idx[b]] += ke[a, b]
    F = np.asarray(loads, float).reshape(-1)
    fixed = np.asarray(fixed, bool).reshape(-1)
    free = ~fixed
    u = np.zeros(ndof)
    Kff = K[np.ix_(free, free)]
    Ff = F[free]
    uf = np.linalg.solve(Kff, Ff)
    u[free] = uf
    stress = np.zeros(M)
    for e in range(M):
        i, j = bars[e]
        c = Cc[e]; s = Ss[e]; Le = L[e]
        elong = (-c) * u[2 * i] + (-s) * u[2 * i + 1] + c * u[2 * j] + s * u[2 * j + 1]
        stress[e] = E * elong / Le
    return u, stress, L


def _node_disp(u):
    u = np.asarray(u); N = len(u) // 2
    return np.array([math.hypot(u[2 * i], u[2 * i + 1]) for i in range(N)])


# ----------------------------- instance family -----------------------------
def make_instances():
    """Deterministic, seeded. Returns list of {'public':..., 'hidden':{}}.
    Material E and yield sigma are fixed; per-instance we set a_max/a_min from the
    UNIFORM member forces and a per-instance sag-gate tightness d_fact. Note: for a
    UNIFORM area vector member forces are area-independent (K scales with A), so the
    unit-area analysis gives the true uniform-sizing forces; indeterminacy only bites
    once the solver makes the areas non-uniform."""
    E = 200000.0
    sigma = 250.0
    AMAX_FACT = 2.2      # uniform a_max design is 2.2x the stress-critical member area
    specs = [
        (201, 4), (202, 4), (203, 5), (204, 5), (205, 6), (206, 6),
        # larger / held-out instances
        (207, 7), (208, 7), (209, 8), (210, 8), (211, 9), (212, 6),
    ]
    out = []
    for seed, npanel in specs:
        r = _rng(seed)
        nodes, bars = _build_gantry(npanel)
        N = len(nodes); M = len(bars)
        P = npanel
        fixed = [[False, False] for _ in range(N)]
        fixed[0] = [True, True]                        # pinned column base (bottom node 0)
        fixed[P] = [False, True]                        # roller base (bottom node P)
        loads = [[0.0, 0.0] for _ in range(N)]
        for i in range(P + 1):
            top = (P + 1) + i                           # deck node: pallet loads
            loads[top][1] -= r(60, 160) / 10.0
        # lateral forklift/seismic nudge at the loaded deck corner
        loads[(P + 1)][0] += r(20, 70) / 10.0
        # member forces from a unit-area (uniform) analysis -> the uniform-sizing forces
        u1, st1, L = _fem(nodes, bars, [1.0] * M, loads, fixed, E)
        maxf = float(np.max(np.abs(st1)))               # force = stress * (area=1)
        a_nom = maxf / sigma
        a_max = AMAX_FACT * a_nom
        a_min = 0.02 * a_nom
        disp_nom = float(np.max(_node_disp(u1))) / a_nom   # sag of uniform a_nom design
        d_fact = r(105, 150) / 100.0                    # per-instance sag-gate tightness
        disp_limit = d_fact * disp_nom
        public = {
            "nodes": nodes, "bars": bars, "loads": loads, "fixed": fixed,
            "E": E, "sigma": sigma, "disp_limit": disp_limit,
            "a_min": a_min, "a_max": a_max,
        }
        out.append({"public": public, "hidden": {}})
    return out


# ----------------------------- scoring -------------------------------------
def _lengths(nodes, bars):
    nodes = np.asarray(nodes, float)
    return np.array([math.hypot(nodes[j, 0] - nodes[i, 0], nodes[j, 1] - nodes[i, 1])
                     for i, j in bars])


def baseline(inst):
    p = inst["public"]
    L = _lengths(p["nodes"], p["bars"])
    return float(p["a_max"] * np.sum(L))


def score(inst, answer):
    """Strictly validate the answer against the instance; return (ok, obj)."""
    p = inst["public"]
    bars = p["bars"]; M = len(bars)
    if not isinstance(answer, dict):
        return False, None
    areas = answer.get("areas", None)
    if not isinstance(areas, list) or len(areas) != M:
        return False, None
    try:
        areas = [float(x) for x in areas]
    except (TypeError, ValueError):
        return False, None
    a = np.asarray(areas, float)
    if not np.all(np.isfinite(a)):
        return False, None
    if np.any(a < p["a_min"] - 1e-9) or np.any(a > p["a_max"] + 1e-9):
        return False, None
    try:
        u, st, L = _fem(p["nodes"], bars, a, p["loads"], p["fixed"], p["E"])
    except Exception:
        return False, None
    if not np.all(np.isfinite(u)):
        return False, None
    if float(np.max(np.abs(st))) > p["sigma"] * (1 + 1e-6):
        return False, None
    if float(np.max(_node_disp(u))) > p["disp_limit"] * (1 + 1e-6):
        return False, None
    obj = float(np.sum(a * L))
    return True, obj


def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <candidate.py>")
        sys.exit(2)
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
            ok, obj = False, None
        if not ok or obj is None or obj <= 0:
            vec.append(0.0); continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
