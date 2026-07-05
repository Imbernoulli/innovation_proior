#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0377 -- "Festival Stage Roof-Truss Rigging Sizing".

Family: feasibility-gated-structural (Frontier-Eng StructuralOptimization), skinned
as a temporary festival main-stage roof. A steel parallel-chord Pratt truss spans
the stage between a pinned tower and a roller tower. Its bottom chord carries the
downward rigging drops (line-array speakers, LED wall panels, moving-head lighting
bars) and the windward top corner takes a lateral gust load. The truss is a
statically-determinate 2D pin-jointed structure, so member axial forces are
INDEPENDENT of the chosen areas -- only stresses, displacements and buckling
capacities change.

The candidate must choose a CROSS-SECTIONAL AREA for every member so that the total
steel weight is minimized subject to THREE hard gates:
    * yield:        every member's |axial stress| <= sigma
    * buckling:     every COMPRESSION member's |stress| <= k_buck * E * area / L^2
                    (Euler-type: the allowable compressive stress GROWS with area,
                     so slender long struts must be thickened beyond fully-stressed)
    * serviceability: every joint displacement magnitude <= disp_limit
    * box:          every area in [a_min, a_max]
A design that violates ANY gate is infeasible and worth 0; otherwise its worth is its
weight. There is no easy optimum: a purely fully-stressed (area=|force|/sigma) design
is lightest but buckles slender struts; repairing buckling locally is still not enough
because a light truss can bust the global displacement gate on tight instances, so a
solver must trade weight against BOTH the local buckling gate and the global sag gate,
and the per-instance tightness varies.

The evaluator runs a pure-numpy linear-elastic FEM (direct stiffness method) in the
PARENT process; the candidate is an UNTRUSTED stdin->stdout program run in an ISOLATED
subprocess via `isorun`, so it only ever sees the public instance and can never reach
the evaluator's frames / FEM / scorer.

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

E_MOD = 200000.0
SIGMA = 250.0
K_BUCK = 0.6
AMAX_FACT = 1.7


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)
    return nxt


# ----------------------------- geometry ------------------------------------
def _build_truss(nb, h, d):
    """Parallel-chord Pratt truss, statically determinate (m = 4*nb+1 = 2n-3).
    bottom chord nodes 0..nb at y=0; top chord nodes nb+1..2nb+1 at y=h."""
    nodes = []
    for i in range(nb + 1):
        nodes.append([float(i * d), 0.0])
    for i in range(nb + 1):
        nodes.append([float(i * d), float(h)])
    bars = []
    for i in range(nb):                    # bottom chord
        bars.append([i, i + 1])
    for i in range(nb):                    # top chord
        bars.append([nb + 1 + i, nb + 1 + i + 1])
    for i in range(nb + 1):                # verticals
        bars.append([i, nb + 1 + i])
    for i in range(nb):                    # diagonals (bottom i -> top i+1)
        bars.append([i, nb + 1 + i + 1])
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


def _buck_area(force, L):
    return math.sqrt(abs(force) * L * L / (K_BUCK * E_MOD)) if force < 0 else 0.0


# ----------------------------- instance family -----------------------------
def make_instances():
    """Deterministic, seeded. Returns list of {'public':..., 'hidden':{}}.
    Material E and yield sigma are fixed; per-instance loads, span geometry (h,d,nb)
    and a displacement-gate tightness are varied. a_max is set large enough that the
    uniform a_max design always satisfies yield AND buckling (the trivial baseline);
    disp_limit = tight * (sag of the buckling-repaired fully-stressed design), clamped
    to comfortably clear the a_max design, so tightness < 1 makes a fully-stressed
    design bust the global sag gate while the baseline still passes."""
    specs = [
        (201, 11, 3.0, 4.0, 0.85), (202, 11, 3.2, 3.8, 1.15),
        (203, 11, 2.8, 4.2, 0.80), (204, 11, 3.0, 4.0, 1.10),
        (205, 11, 3.4, 3.6, 0.90), (206, 11, 2.6, 4.4, 1.20),
        (207, 11, 3.1, 4.1, 0.82), (208, 11, 3.3, 3.9, 1.05),
        # larger / smaller held-out spans
        (209, 13, 3.0, 4.0, 0.88), (210, 9, 3.0, 4.2, 1.12),
        (211, 13, 3.2, 3.8, 0.83), (212, 9, 2.8, 4.4, 1.08),
    ]
    out = []
    for seed, nb, h, d, tight in specs:
        r = _rng(seed)
        nodes, bars = _build_truss(nb, h, d)
        N = len(nodes); M = len(bars)
        fixed = [[False, False] for _ in range(N)]
        fixed[0] = [True, True]                 # pinned tower
        fixed[nb] = [False, True]               # roller tower (bottom-right)
        loads = [[0.0, 0.0] for _ in range(N)]
        for k in range(nb + 1):                 # rigging drops on the bottom chord
            loads[k][1] -= r(40, 140) / 10.0
        loads[nb + 1][0] += r(20, 60) / 10.0    # lateral gust at the windward top corner
        # member forces from a unit-area analysis (determinate -> area-independent)
        u1, st1, L = _fem(nodes, bars, [1.0] * M, loads, fixed, E_MOD)
        forces = st1
        a_nom = float(np.max(np.abs(forces))) / SIGMA
        a_buck_max = max(_buck_area(forces[e], L[e]) for e in range(M))
        a_max = AMAX_FACT * max(a_nom, a_buck_max)
        a_min = 0.02 * a_nom
        # buckling-repaired fully-stressed reference design (for sag tightness)
        a_ref = np.abs(forces) / SIGMA
        for e in range(M):
            if forces[e] < 0:
                a_ref[e] = max(a_ref[e], _buck_area(forces[e], L[e]))
        a_ref = np.clip(a_ref, a_min, a_max)
        u_ref, _, _ = _fem(nodes, bars, a_ref, loads, fixed, E_MOD)
        disp_ref = float(np.max(_node_disp(u_ref)))
        u_am, _, _ = _fem(nodes, bars, [a_max] * M, loads, fixed, E_MOD)
        disp_am = float(np.max(_node_disp(u_am)))
        disp_limit = max(tight * disp_ref, 1.15 * disp_am)   # baseline always clears
        public = {
            "nodes": nodes, "bars": bars, "loads": loads, "fixed": fixed,
            "E": E_MOD, "sigma": SIGMA, "k_buck": K_BUCK, "disp_limit": disp_limit,
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
    if not np.all(np.isfinite(u)) or not np.all(np.isfinite(st)):
        return False, None
    # yield gate
    if float(np.max(np.abs(st))) > p["sigma"] * (1 + 1e-6):
        return False, None
    # buckling gate (compression members only)
    kb = p["k_buck"]; E = p["E"]
    for e in range(M):
        if st[e] < 0.0:
            allow = kb * E * a[e] / (L[e] * L[e])
            if abs(st[e]) > allow * (1 + 1e-6):
                return False, None
    # serviceability (sag) gate
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
