#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0351 -- "Solar-Farm Inverter Design Sweep"
(budgeted multi-objective hypervolume, DTLZ4-biased variant).

Family: multiobjective-hypervolume (Frontier-Eng ReactionOptimisation / DTLZ2),
skinned as commissioning a fleet of grid-tie inverters for a solar farm. An
engineering team must pick a BATCH of inverter DESIGN CONFIGURATIONS to prototype
and characterise on a bench. Each configuration is a decision vector x in [0,1]^n;
a deterministic cost surrogate turns it into M competing costs to MINIMIZE:
    f1 = capital / bill-of-materials cost
    f2 = conversion power loss (heat)
    f3 = harmonic distortion (THD)          [only when M == 3]

This variant uses a *biased* DTLZ4-style surrogate: the trade-off ("position")
variables enter through a POWER map theta_j = (x_j ** alpha) * (pi/2). With
alpha > 1 the map is strongly non-linear, so a uniform grid in decision space
lands as a heavily CLUSTERED, uneven spread on the cost frontier -- a solver must
INVERT the bias (x_j = t ** (1/alpha)) to spread evenly. The remaining coordinates
are "distance" variables (thermal / control margins) whose cost g is minimized
(= 0) exactly when each equals 0.5, placing (f1..fM) on the unit-radius cost
frontier sum_i f_i^2 = 1 in the positive orthant.

Instances mix M = 2 (cost vs loss) and M = 3 (cost vs loss vs THD) fronts, and
different bias exponents alpha, budgets, distance-variable counts and references.

The team can only afford to prototype `budget` configurations. The QUALITY of a
chosen batch is its dominated HYPERVOLUME with respect to a fixed reference
(worst-tolerable) cost point `ref`: the volume of cost space dominated by at least
one prototyped configuration and bounded above by `ref`. Maximizing it rewards
BOTH pushing configurations onto the frontier AND spreading them across
complementary trade-offs. There is no easy optimum: placing a fixed number of
points on the curved (and bias-warped) frontier to maximize dominated volume is a
hard continuous global-optimization problem, and uniform grids, bias-inverted
angle-even spreads, and local hypervolume ascent all yield materially different
volumes.

The candidate is UNTRUSTED model output. It is run as a standalone stdin->stdout
program in an ISOLATED subprocess via `isorun`, sees ONLY the public instance, and
never touches the evaluator's frames / surrogate / scorer. The evaluator computes
the surrogate and the EXACT hypervolume itself, in the parent process.

Scoring (deterministic; no wall-time / GPU):
  obj(batch) = exact dominated hypervolume of the batch's cost vectors w.r.t. ref
  baseline b = hypervolume of the single "midpoint" configuration x = [0.5]*n
               (one Pareto-optimal but un-spread prototype -- the evaluator's own
                trivial construction)
  For a feasible answer with objective obj (a MAXIMIZATION):
        r = min(1, 0.1 * obj / b)
  -> a batch that only reproduces the single midpoint point maps to exactly 0.1;
     a batch whose spread earns k times the single-point volume maps to min(1, 0.1*k).
  Infeasible / malformed / over-budget answers -> 0.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# --------------------------- biased DTLZ cost surrogate ---------------------
def _surrogate(x, M, n_pos, alpha):
    """Deterministic M-objective DTLZ4-biased surrogate (minimization).
    x in [0,1]^n. Distance cost g is minimized (=0) when every distance variable
    equals 0.5, placing (f1..fM) on the unit-radius cost frontier. The position
    variables enter through the power map theta_j = (x_j**alpha)*(pi/2)."""
    g = 0.0
    for i in range(n_pos, len(x)):
        d = x[i] - 0.5
        g += d * d
    theta = [(max(0.0, xi) ** alpha) * (math.pi / 2.0) for xi in x[:n_pos]]
    f = []
    for i in range(M):
        val = 1.0 + g
        for j in range(M - 1 - i):
            val *= math.cos(theta[j])
        if i > 0:
            val *= math.sin(theta[M - 1 - i])
        f.append(val)
    return f


# --------------------------- exact hypervolume ------------------------------
def _area2d(pts, ref):
    """Exact area of the union of axis-aligned rectangles [p_x,ref_x] x [p_y,ref_y]
    (shared upper corner = ref): the 2D dominated hypervolume of `pts`."""
    nd = []
    miny = float("inf")
    for p in sorted(pts, key=lambda p: (p[0], p[1])):
        if p[1] < miny - 1e-15:
            nd.append(p)
            miny = p[1]
    area = 0.0
    prev_y = ref[1]
    for p in nd:
        area += (ref[0] - p[0]) * (prev_y - p[1])
        prev_y = p[1]
    return area


def _hv2d(points, ref):
    pts = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
    if not pts:
        return 0.0
    return _area2d(pts, ref)


def _hv3d(points, ref):
    """Exact 3D dominated hypervolume (minimization) w.r.t. reference `ref`,
    via the standard z-sweep: slice along f3 into layers where the active point
    set is constant, and integrate the exact 2D union-area of each layer."""
    pts = [list(p) for p in points
           if p[0] < ref[0] and p[1] < ref[1] and p[2] < ref[2]]
    if not pts:
        return 0.0
    pts.sort(key=lambda p: p[2])
    hv = 0.0
    n = len(pts)
    for i in range(n):
        z0 = pts[i][2]
        z1 = pts[i + 1][2] if i + 1 < n else ref[2]
        dz = z1 - z0
        if dz <= 0.0:
            continue
        hv += _area2d([p[:2] for p in pts[:i + 1]], ref) * dz
    return hv


def _hv(points, ref, M):
    if M == 2:
        return _hv2d([p[:2] for p in points], ref)
    return _hv3d(points, ref)


# --------------------------- instance family --------------------------------
def make_instances():
    """Deterministic, fixed reproducible specs. Each spec:
    (M objectives, k distance vars, alpha bias exponent, batch budget, R ref coord).
    n_pos = M-1. A mix of 2- and 3-objective fronts and bias exponents; the last
    two specs are harder / held-out (more distance vars, strong bias, tight ref)."""
    specs = [
        (3, 3, 1.0, 16, 1.10),
        (3, 3, 2.0, 20, 1.12),
        (3, 4, 3.0, 24, 1.15),
        (3, 4, 2.0, 14, 1.08),
        (3, 5, 1.0, 22, 1.18),
        (3, 5, 3.0, 18, 1.12),
        (2, 2, 1.0, 12, 1.10),
        (2, 3, 2.0, 16, 1.12),
        (2, 3, 3.0, 20, 1.14),
        (2, 4, 2.0, 10, 1.08),
        # harder / held-out
        (3, 6, 3.0, 26, 1.20),
        (2, 5, 3.0, 22, 1.16),
    ]
    out = []
    for (M, k, alpha, budget, R) in specs:
        n_pos = M - 1
        n = n_pos + k
        ref = [R] * M
        public = {
            "surrogate": "dtlz_biased",
            "M": M,
            "n": n,
            "n_pos": n_pos,
            "alpha": alpha,
            "budget": budget,
            "ref": ref,
            "note": ("Return {'points': [[x_1..x_n], ...]}, each x_i in [0,1], at most "
                     "'budget' configurations. Costs = biased-DTLZ surrogate with "
                     "theta_j=(x_j**alpha)*(pi/2); maximize the exact %dD dominated "
                     "hypervolume w.r.t. 'ref' (minimization). Frontier at "
                     "distance-vars=0.5, positive-orthant unit sphere." % M),
        }
        out.append({"public": public, "hidden": {}})
    return out


# --------------------------- scoring ----------------------------------------
def baseline(inst):
    p = inst["public"]
    mid = [0.5] * p["n"]
    f = _surrogate(mid, p["M"], p["n_pos"], p["alpha"])
    return float(_hv([f], p["ref"], p["M"]))


def score(inst, answer):
    """Strictly validate the answer; return (ok, objective_hypervolume)."""
    p = inst["public"]
    n = p["n"]
    M = p["M"]
    n_pos = p["n_pos"]
    alpha = p["alpha"]
    budget = p["budget"]
    ref = p["ref"]
    if not isinstance(answer, dict):
        return False, None
    pts = answer.get("points", None)
    if not isinstance(pts, list) or len(pts) < 1 or len(pts) > budget:
        return False, None
    costs = []
    for row in pts:
        if not isinstance(row, list) or len(row) != n:
            return False, None
        x = []
        for v in row:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return False, None
            v = float(v)
            if not (v == v) or v in (float("inf"), float("-inf")):
                return False, None
            if v < -1e-9 or v > 1.0 + 1e-9:
                return False, None
            x.append(min(1.0, max(0.0, v)))
        f = _surrogate(x, M, n_pos, alpha)
        if any((c != c) or c in (float("inf"), float("-inf")) for c in f):
            return False, None
        costs.append(f)
    hv = float(_hv(costs, ref, M))
    if not (hv == hv) or hv in (float("inf"), float("-inf")) or hv < 0.0:
        return False, None
    return True, hv


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
        r = min(1.0, 0.1 * obj / max(b, 1e-12))
        vec.append(r if (r == r and 0.0 <= r <= 1.0) else 0.0)
    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
