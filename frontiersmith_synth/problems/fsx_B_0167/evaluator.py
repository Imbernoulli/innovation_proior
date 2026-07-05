#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0167 -- "Data-Center Overhead Cooling-Gantry Sizing".

Family: feasibility-gated-structural (Frontier-Eng StructuralOptimization), skinned as
a data-center facility problem. A hyperscale data hall carries its liquid-cooling plant
(coolant distribution units, chilled-water risers, manifold headers) on a long OVERHEAD
STEEL GANTRY that spans the aisle between two structural columns. The gantry is modelled
as a statically-determinate 2D pin-jointed truss (a triangulated Pratt-style beam):
joints connected by straight bar members, pinned at one column and on a roller at the
other. The suspended cooling hardware hangs from the bottom-chord joints (downward point
loads) and hot-aisle airflow / seismic bracing pushes laterally on the top chord.

The candidate chooses a CROSS-SECTIONAL AREA area_e in [a_min, a_max] for every bar so
that the gantry is as LIGHT (cheap) as possible while remaining safe. This is a
feasibility-GATED objective -- a design that violates ANY structural limit is worthless
(scores 0). Because the truss is statically determinate, the member axial forces depend
ONLY on the geometry and loads (not on the chosen areas); the joint displacements do
depend on the areas.

Given nodes, bar connectivity, joint loads, supports and Young's modulus E, the evaluator
assembles the global stiffness matrix, solves K u = F for joint displacements u, and
computes each bar's axial stress = E * elongation / length.

HARD CONSTRAINTS (all must hold, else the instance scores 0):
  1. Yield gate:      |stress_e| <= sigma            for every bar e.
  2. Buckling gate:   for every COMPRESSION bar (stress_e < 0),
                      |stress_e| <= sigma_cr_e,  sigma_cr_e = pi^2 * E * kappa * area_e / length_e^2
                      (Euler critical stress for a section family with I = kappa*area^2;
                       slender, long, lightly-thickened compression members buckle first).
  3. Sag gate:        every joint displacement magnitude sqrt(ux^2+uy^2) <= disp_limit.
  4. Bounds:          a_min <= area_e <= a_max        for every bar.

OBJECTIVE (minimize):  weight(areas) = sum_e area_e * length_e   (density = 1).

This is genuinely open-ended. Sizing purely to the yield gate (fully-stressed design,
area_e = |force_e|/sigma) is light and always passes yield, but leaves slender
compression members that BUCKLE, and on tight-clearance instances the thin gantry SAGS
past disp_limit -- both reject the whole design. A robust solver must respect all three
physical gates at once: floor each member to max(yield, buckling) area, then stiffen
(scale areas up) until the sag gate is met. The right trade-off varies per instance and
has no single closed form.

The candidate is UNTRUSTED model output: it runs in an ISOLATED subprocess via `isorun`,
sees ONLY the public instance on stdin, and returns ONLY its answer on stdout, so it can
never reach the evaluator's frames / scorer / baseline / held-out data.

Scoring (deterministic; no wall-time):
  baseline b = weight of the uniform a_max design (heaviest; always feasible).
  For a FEASIBLE answer with objective obj = weight(areas):  r = min(1, 0.1 * b / obj)
  -> the uniform a_max design maps to exactly 0.1; a design k times lighter than baseline
     maps to min(1, 0.1*k). Infeasible / malformed / out-of-bounds / non-finite -> 0.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)
    return nxt


# ----------------------------- truss FEM (pure python) ---------------------
def _build_truss(n_bays, bay_len, height):
    """Pratt-style double-chord span truss. Returns (nodes, bars).
    Bottom nodes 0..n_bays at y=0; top nodes (n_bays+1)+i at y=height.
    Members = 4*n_bays + 1  -> with pin+roller supports the truss is statically
    determinate (members + reactions == 2*nodes)."""
    nodes = []
    for i in range(n_bays + 1):
        nodes.append((i * bay_len, 0.0))
    for i in range(n_bays + 1):
        nodes.append((i * bay_len, height))
    def B(i): return i
    def T(i): return (n_bays + 1) + i
    bars = []
    for i in range(n_bays):     bars.append((B(i), B(i + 1)))   # bottom chord
    for i in range(n_bays):     bars.append((T(i), T(i + 1)))   # top chord
    for i in range(n_bays + 1): bars.append((B(i), T(i)))       # verticals
    for i in range(n_bays):     bars.append((B(i), T(i + 1)))   # diagonals
    return nodes, bars


def _fem(nodes, bars, loads, fixed, E, areas):
    """Assemble K, solve K u = F (Gaussian elim with partial pivot on free DOFs),
    return (u, stress, length). u is None if the system is singular."""
    N = len(nodes); M = len(bars); ndof = 2 * N
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
            row = K[idx[a]]
            kea = ke[a]
            for b in range(4):
                row[idx[b]] += k * kea[b]
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
        if abs(A[piv][col]) < 1e-12:
            return None, None, None
        A[col], A[piv] = A[piv], A[col]
        rhs[col], rhs[piv] = rhs[piv], rhs[col]
        d = A[col][col]
        for rr in range(nf):
            if rr == col:
                continue
            f = A[rr][col] / d
            if f == 0.0:
                continue
            Arr = A[rr]; Acol = A[col]
            for cc in range(col, nf):
                Arr[cc] -= f * Acol[cc]
            rhs[rr] -= f * rhs[col]
    u = [0.0] * ndof
    for a in range(nf):
        u[free[a]] = rhs[a] / A[a][a]
    stress = []; length = []
    for e in range(M):
        i, j = bars[e]; c, s, Le = geo[e]
        elong = (-c) * u[2 * i] + (-s) * u[2 * i + 1] + c * u[2 * j] + s * u[2 * j + 1]
        stress.append(E * elong / Le); length.append(Le)
    return u, stress, length


def _max_disp(u, N):
    return max(math.hypot(u[2 * n], u[2 * n + 1]) for n in range(N))


# ----------------------------- instance family -----------------------------
_E = 2.0e5
_SIGMA = 250.0
_A_MIN = 0.05
_A_MAX = 3.0


def make_instances():
    """Deterministic, seeded. Returns [{'public':..., 'hidden':{}}].
    A Pratt-style overhead cooling gantry with per-instance span (n_bays), bay geometry,
    hung-equipment load level (wbase), section slenderness (kappa) and sag-gate tightness
    (tight). 'Loose' instances (stocky sections + generous clearance) let the pure
    yield-only sizing stay feasible; 'tight' ones make it buckle and/or sag -> rejected.
    disp_limit is set to `tight` times the sag of the yield+buckling-floored design, so
    tight<1 forces extra stiffening beyond the per-member minimum."""
    # seed, n_bays, bay_len, height, wbase, kappa, tight
    specs = [
        (201,  9, 2.0, 1.4, 14.0, 0.020, 1.50),   # loose  (stocky, slack sag)
        (202, 10, 1.9, 1.5, 16.0, 0.020, 1.40),   # loose
        (203, 11, 2.0, 1.3, 13.0, 0.003, 0.75),   # tight  (slender, tight sag)
        (204, 11, 2.1, 1.4, 12.0, 0.003, 0.70),   # tight
        (205, 10, 1.8, 1.2, 18.0, 0.008, 0.90),   # mixed
        (206, 12, 2.0, 1.5, 12.0, 0.0035, 0.80),  # tight, larger
        (207, 11, 1.9, 1.3, 15.0, 0.020, 1.30),   # loose
        (208, 12, 2.0, 1.4, 11.0, 0.003, 0.70),   # tight, larger
        (209, 13, 2.0, 1.5, 12.0, 0.0035, 0.75),  # tight, held-out large
        (210, 13, 1.9, 1.3, 14.0, 0.006, 0.85),   # mixed, held-out large
    ]
    out = []
    for seed, n_bays, bay_len, height, wbase, kappa, tight in specs:
        r = _rng(seed)
        nodes, bars = _build_truss(n_bays, bay_len, height)
        N = len(nodes); M = len(bars)
        def T(i): return (n_bays + 1) + i
        loads = [[0.0, 0.0] for _ in range(N)]
        for i in range(1, n_bays):                       # hung cooling hardware (downward)
            loads[i][1] = -wbase * (0.7 + r(0, 60) / 100.0)
        for i in range(n_bays + 1):                      # lateral airflow/seismic on top chord
            loads[T(i)][0] = 0.15 * wbase * (0.4 + r(0, 60) / 100.0)
        fixed = [[False, False] for _ in range(N)]
        fixed[0] = [True, True]                          # pinned column
        fixed[n_bays] = [False, True]                    # roller column
        # reference: yield+buckling-floored design; disp_limit = tight * its sag
        u, stress, length = _fem(nodes, bars, loads, fixed, _E, [_A_MAX] * M)
        forces = [stress[e] * _A_MAX for e in range(M)]
        ref = []
        for e in range(M):
            ya = abs(forces[e]) / _SIGMA
            ba = math.sqrt(abs(forces[e]) * length[e] ** 2 / (math.pi ** 2 * _E * kappa)) \
                if forces[e] < 0 else 0.0
            ref.append(min(max(max(ya, ba), _A_MIN), _A_MAX))
        u2, _, _ = _fem(nodes, bars, loads, fixed, _E, ref)
        disp_limit = tight * _max_disp(u2, N)
        public = {
            "nodes": [list(p) for p in nodes],
            "bars": [list(b) for b in bars],
            "loads": [list(v) for v in loads],
            "fixed": [[int(a), int(b)] for a, b in fixed],
            "E": _E, "sigma": _SIGMA, "disp_limit": disp_limit,
            "a_min": _A_MIN, "a_max": _A_MAX, "kappa": kappa,
        }
        out.append({"public": public, "hidden": {}})
    return out


# ----------------------------- scoring -------------------------------------
def baseline(inst):
    """Uniform a_max gantry: heaviest, always feasible."""
    p = inst["public"]
    M = len(p["bars"])
    _, _, length = _fem(p["nodes"], p["bars"], p["loads"], p["fixed"], p["E"], [p["a_max"]] * M)
    return float(sum(p["a_max"] * length[e] for e in range(M)))


def score(inst, answer):
    """Strictly validate the answer against the instance; return (ok, obj)."""
    p = inst["public"]
    nodes = p["nodes"]; bars = p["bars"]; loads = p["loads"]; fixed = p["fixed"]
    N = len(nodes); M = len(bars)
    sigma = p["sigma"]; a_min = p["a_min"]; a_max = p["a_max"]
    E = p["E"]; kappa = p["kappa"]; disp_limit = p["disp_limit"]
    if not isinstance(answer, dict):
        return False, None
    areas = answer.get("areas", None)
    if not isinstance(areas, list) or len(areas) != M:
        return False, None
    try:
        areas = [float(x) for x in areas]
    except (TypeError, ValueError):
        return False, None
    for a in areas:
        if not math.isfinite(a) or a < a_min - 1e-9 or a > a_max + 1e-9:
            return False, None
    u, stress, length = _fem(nodes, bars, loads, fixed, E, areas)
    if u is None:
        return False, None
    for e in range(M):
        if not math.isfinite(stress[e]):
            return False, None
        if abs(stress[e]) > sigma + 1e-6:                       # yield gate
            return False, None
        if stress[e] < 0.0:                                     # buckling gate
            scr = math.pi * math.pi * E * kappa * areas[e] / (length[e] * length[e])
            if abs(stress[e]) > scr + 1e-6:
                return False, None
    if _max_disp(u, N) > disp_limit + 1e-9:                     # sag gate
        return False, None
    obj = sum(areas[e] * length[e] for e in range(M))
    if not math.isfinite(obj) or obj <= 0.0:
        return False, None
    return True, float(obj)


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
        if not ok or obj is None or obj <= 0.0:
            vec.append(0.0); continue
        b = baseline(inst)
        r = min(1.0, 0.1 * b / max(obj, 1e-12))
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
