#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_B_0375 -- "Harbor Container Port: Budgeted Berth-Plan Trials".

Family: multiobjective-hypervolume (Frontier-Eng ReactionOptimisation / DTLZ2),
skinned as a small container-port terminal. A port operator must pick a BATCH of berth /
crane / yard configurations to trial on the terminal's digital twin. Each configuration
is a decision vector x in [0,1]^n; a deterministic 3-objective surrogate (a shifted /
re-centred DTLZ2 surface) turns it into three costs to MINIMIZE:
    f1 = vessel turnaround time
    f2 = quay-crane energy draw
    f3 = landside yard congestion
The first `n_pos` coordinates steer the TRADE-OFF direction along the cost surface
(the DTLZ2 "position" variables); the remaining coordinates control how close the
configuration sits to the ideal cost frontier (the "distance" variables). Unlike the
textbook DTLZ2, the distance cost here is minimized at a per-instance value `center`
(the terminal's calibrated operating point), NOT always 0.5: a configuration is
Pareto-optimal iff every distance variable equals `center`, in which case (f1,f2,f3)
lands on the unit-radius cost frontier f1^2 + f2^2 + f3^2 = 1 in the positive octant.

The operator can only afford to trial `budget` configurations on the twin. The QUALITY
of the chosen batch is its dominated HYPERVOLUME with respect to a fixed reference
(worst-tolerable) cost point `ref` -- and here the reference is ASYMMETRIC per objective
(different tolerance for turnaround vs energy vs congestion), so the geometry of a good
spread is skewed and instance-specific. Maximizing hypervolume rewards BOTH pushing
configurations onto the frontier AND spreading them to cover complementary trade-offs.
There is no easy optimum: for a fixed budget the optimal placement of points on the
curved cost frontier to maximize dominated volume under an asymmetric reference is a
hard continuous global-optimization problem, and multiple strategies (uniform grids,
angle-even spreads, local hypervolume ascent) give materially different volumes.

The candidate is UNTRUSTED model output. It is run as a standalone stdin->stdout
program in an ISOLATED subprocess via `isorun`, sees ONLY the public instance, and
never touches the evaluator's frames / surrogate / scorer. The evaluator computes the
re-centred DTLZ2 surrogate and the EXACT 3D hypervolume itself, in the parent process.

Scoring (deterministic; no wall-time / GPU):
  obj(batch) = exact dominated hypervolume of the batch's cost vectors w.r.t. ref
  baseline b = hypervolume of the single "domain-midpoint" configuration x = [0.5]*n
               (one un-spread trial at the naive centre of the decision box -- the
                evaluator's own trivial construction; off-frontier whenever center!=0.5)
  For a feasible answer with objective obj (a MAXIMIZATION):
        r = min(1, 0.1 * obj / b)
  -> a batch that only reproduces the single domain-midpoint point maps to exactly 0.1;
     a batch whose spread earns k times the single-point volume maps to min(1, 0.1*k).
  Infeasible / malformed / over-budget answers -> 0.

CLI:  python3 evaluator.py <candidate.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# --------------------------- DTLZ2 cost surrogate ---------------------------
def _dtlz2(x, M, n_pos, center):
    """Deterministic 3-objective DTLZ2 (minimization), re-centred. x in [0,1]^n.
    Distance cost g is minimized (=0) when every distance variable equals `center`,
    placing (f1..fM) on the unit-radius cost frontier."""
    g = 0.0
    for i in range(n_pos, len(x)):
        d = x[i] - center
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
        hv += _area2d(pts[:i + 1], ref) * dz
    return hv


# --------------------------- instance family --------------------------------
def make_instances():
    """Deterministic (no RNG needed -- fixed, reproducible specs). Each instance:
    (k distance vars, batch budget, ASYMMETRIC reference cost triple, frontier
    `center`). M=3, n_pos=2 fixed. Later specs use more distance variables, tighter
    references, and off-centre frontiers -- the harder / held-out instances
    (generalization)."""
    M = 3
    n_pos = 2
    specs = [
        (3, 16, [1.10, 1.15, 1.08], 0.50),
        (3, 24, [1.20, 1.05, 1.12], 0.50),
        (4, 20, [1.12, 1.18, 1.10], 0.55),
        (4, 12, [1.06, 1.10, 1.14], 0.45),
        (5, 25, [1.15, 1.25, 1.12], 0.50),
        (3, 18, [1.08, 1.12, 1.20], 0.50),
        (4, 28, [1.22, 1.10, 1.16], 0.55),
        (5, 15, [1.10, 1.30, 1.08], 0.45),
        # harder / held-out: more distance vars, skewed references, off-centre fronts
        (6, 22, [1.28, 1.14, 1.20], 0.50),
        (6, 30, [1.18, 1.24, 1.14], 0.55),
        (5, 27, [1.30, 1.12, 1.26], 0.45),
        (4, 14, [1.14, 1.20, 1.10], 0.50),
    ]
    out = []
    for (k, budget, ref, center) in specs:
        n = n_pos + k
        public = {
            "surrogate": "dtlz2_recentred",
            "M": M,
            "n": n,
            "n_pos": n_pos,
            "budget": budget,
            "ref": [float(r) for r in ref],
            "center": float(center),
            "note": ("Return {'points': [[x_1..x_n], ...]}, each x_i in [0,1], at most "
                     "'budget' configurations. Costs = re-centred 3-objective DTLZ2 "
                     "(minimize). Frontier reached when every distance variable "
                     "(indices n_pos..n-1) equals 'center'; position variables "
                     "(indices 0..n_pos-1) sweep the trade-off direction. Maximize the "
                     "exact 3D dominated hypervolume w.r.t. the ASYMMETRIC 'ref'."),
        }
        out.append({"public": public, "hidden": {}})
    return out


# --------------------------- scoring ----------------------------------------
def baseline(inst):
    p = inst["public"]
    mid = [0.5] * p["n"]
    f = _dtlz2(mid, p["M"], p["n_pos"], p["center"])
    return float(_hv3d([f], p["ref"]))


def score(inst, answer):
    """Strictly validate the answer; return (ok, objective_hypervolume)."""
    p = inst["public"]
    n = p["n"]
    M = p["M"]
    budget = p["budget"]
    ref = p["ref"]
    center = p["center"]
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
        f = _dtlz2(x, M, p["n_pos"], center)
        if any((c != c) or c in (float("inf"), float("-inf")) for c in f):
            return False, None
        costs.append(f)
    hv = float(_hv3d(costs, ref))
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
