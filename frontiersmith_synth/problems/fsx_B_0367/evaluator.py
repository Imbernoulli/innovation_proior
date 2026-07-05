#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0367 -- "Orbital Debris Cleanup Mission Portfolio".

Family: multiobjective-hypervolume (Frontier-Eng ReactionOptimisation / DTLZ2),
LARGE-scale variant #5, skinned as an orbital-debris-removal campaign. A cleanup
operator must select a BATCH of debris-capture MISSION PROFILES to fly. Each profile
is a decision vector x in [0,1]^n; a deterministic 4-objective surrogate (a
shifted-radius DTLZ2 hypersurface, M=4) turns it into four campaign costs to MINIMIZE:

    f1 = propellant / delta-v budget
    f2 = time-to-deorbit (mission duration)
    f3 = collision / operational risk
    f4 = residual debris cross-section left uncleared

The first `n_pos` coordinates steer the TRADE-OFF DIRECTION along the cost
hypersurface (the DTLZ2 "position" variables); the remaining coordinates are
"distance" variables controlling how close the profile sits to the ideal cost
frontier (their cost is minimized at 0.5). A profile is Pareto-optimal iff every
distance variable equals 0.5, in which case (f1..f4) lands on the unit-radius cost
frontier f1^2 + f2^2 + f3^2 + f4^2 = 1 in the positive orthant.

The operator can only afford to fly `budget` profiles. The QUALITY of the chosen
portfolio is its dominated HYPERVOLUME in 4-objective cost space with respect to a
fixed reference (worst-tolerable) cost point `ref`: the 4D volume of cost space
dominated by at least one flown profile and bounded above by `ref`. Maximizing
hypervolume rewards BOTH pushing profiles onto the frontier AND spreading them to
cover complementary trade-offs. There is no easy optimum: for a fixed budget the
optimal placement of points on the curved 3D frontier to maximize dominated 4D volume
is a hard continuous global-optimization problem, and multiple strategies (uniform
grids, angle-even spreads, local hypervolume ascent) give materially different
volumes.

The candidate is UNTRUSTED model output. It is run as a standalone stdin->stdout
program in an ISOLATED subprocess via `isorun`, sees ONLY the public instance, and
never touches the evaluator's frames / surrogate / scorer. The evaluator computes the
DTLZ2 surrogate and the EXACT M-dimensional hypervolume itself, in the parent process.

Scoring (deterministic; no wall-time / GPU):
  obj(batch) = exact dominated 4D hypervolume of the batch's cost vectors w.r.t. ref
  baseline b = hypervolume of the single "midpoint" profile x = [0.5]*n
               (one Pareto-optimal but un-spread mission -- the evaluator's own
                trivial construction)
  For a feasible answer with objective obj (a MAXIMIZATION):
        r = min(1, 0.1 * obj / b)
  -> a portfolio that only reproduces the single midpoint point maps to exactly 0.1;
     a portfolio whose spread earns k times the single-point volume maps to
     min(1, 0.1*k).
  Infeasible / malformed / over-budget answers -> 0.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# --------------------------- DTLZ2 cost surrogate ---------------------------
def _dtlz2(x, M, n_pos):
    """Deterministic M-objective DTLZ2 (minimization). x in [0,1]^n.
    Distance cost g is minimized (=0) when every distance variable equals 0.5,
    placing (f1..fM) on the unit-radius cost frontier in the positive orthant."""
    g = 0.0
    for i in range(n_pos, len(x)):
        d = x[i] - 0.5
        g += d * d
    theta = [xi * (math.pi / 2.0) for xi in x[:n_pos]]
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
    (shared upper corner = ref), i.e. the 2D dominated hypervolume of `pts`."""
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


def _hv(points, ref):
    """Exact M-dimensional dominated hypervolume (minimization) w.r.t. reference
    `ref`, via recursive Hypervolume-by-Slicing: slice along the last objective into
    layers where the active (dominating) point set is constant, and integrate the
    exact (M-1)-D union hypervolume of each layer. Reduces to _area2d at M=2."""
    M = len(ref)
    pts = [p for p in points if all(p[i] < ref[i] for i in range(M))]
    if not pts:
        return 0.0
    if M == 1:
        return ref[0] - min(p[0] for p in pts)
    if M == 2:
        return _area2d(pts, ref)
    d = M - 1
    pts.sort(key=lambda p: p[d])
    vol = 0.0
    n = len(pts)
    for i in range(n):
        z0 = pts[i][d]
        z1 = pts[i + 1][d] if i + 1 < n else ref[d]
        dz = z1 - z0
        if dz <= 0.0:
            continue
        proj = [p[:d] for p in pts[:i + 1]]
        vol += _hv(proj, ref[:d]) * dz
    return vol


# --------------------------- instance family --------------------------------
def make_instances():
    """Deterministic (no RNG -- fixed, reproducible specs). M=4 objectives,
    n_pos=3 trade-off (position) variables fixed. Each spec: (k distance vars,
    batch budget, reference-cost coordinate R). Later specs use more distance
    variables and tighter/looser references -- these are the harder / held-out
    instances (generalization)."""
    M = 4
    n_pos = 3
    specs = [
        (2, 16, 1.10), (2, 20, 1.12), (3, 18, 1.15), (3, 12, 1.08),
        (4, 24, 1.20), (2, 14, 1.06), (3, 22, 1.18), (4, 16, 1.10),
        # harder / held-out: more distance vars, tighter/looser references
        (5, 20, 1.25), (5, 24, 1.22), (4, 18, 1.30), (3, 14, 1.12),
    ]
    out = []
    for (k, budget, R) in specs:
        n = n_pos + k
        public = {
            "surrogate": "dtlz2",
            "M": M,
            "n": n,
            "n_pos": n_pos,
            "budget": budget,
            "ref": [R, R, R, R],
            "note": ("Return {'points': [[x_1..x_n], ...]}, each x_i in [0,1], at most "
                     "'budget' mission profiles. Costs = shifted-DTLZ2 (M=4); maximize the "
                     "exact 4D dominated hypervolume w.r.t. 'ref' (minimization). Cost "
                     "frontier at distance-vars=0.5, positive-orthant unit hypersphere."),
        }
        out.append({"public": public, "hidden": {"M": M, "n_pos": n_pos}})
    return out


# --------------------------- scoring ----------------------------------------
def baseline(inst):
    p = inst["public"]
    mid = [0.5] * p["n"]
    f = _dtlz2(mid, p["M"], p["n_pos"])
    return float(_hv([f], p["ref"]))


def score(inst, answer):
    """Strictly validate the answer; return (ok, objective_hypervolume)."""
    p = inst["public"]
    n = p["n"]
    M = p["M"]
    n_pos = p["n_pos"]
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
        f = _dtlz2(x, M, n_pos)
        if any((c != c) or c in (float("inf"), float("-inf")) for c in f):
            return False, None
        costs.append(f)
    hv = float(_hv(costs, ref))
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
