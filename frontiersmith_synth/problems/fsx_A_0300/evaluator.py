#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_A_0300 -- "Lechuguilla Deep: The Survey Loop"
(family: heuristic-contest-offline; format B, quality-metric).

THEME.  A caving expedition is mapping a large horizontal cave level.  A previous
trip left a set of numbered SURVEY STATIONS bolted to the rock at fixed 2-D
coordinates (metres on the survey grid).  Ground-penetrating sonar has also located
a set of FORMATIONS -- crystal galleries, gypsum chandeliers, aragonite bushes --
each at a known point and each worth an integer amount of "documentation value".

The expedition will string ONE closed survey loop: a rope traverse that visits an
ORDERED subset of the stations and returns to the start, forming a SIMPLE (non
self-intersecting) polygon.  Every formation that ends up STRICTLY INSIDE the loop is
inside the mapped chamber and its value is documented.  Stringing rope costs effort in
proportion to the loop's length.

    loop value  =  sum of documentation value of formations strictly inside the loop
                   -  lambda * (perimeter of the loop, in survey metres)

The expedition wants to MAXIMISE the loop value by choosing which stations to string,
and in what cyclic order.  This is an AtCoder-heuristic-contest-style offline
optimisation: a polygon-selection problem over a fixed instance, with a deterministic
contest scorer and no easy optimum.  Enclosing everything pays too much rope for the
far, low-value formations; a single tight triangle misses most galleries; the good
plans hug a chosen subset of formation clusters -- so region growing, cluster
selection, convex-hull hugging and local search all give genuinely different results.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "S": int, "lam": float,
             "stations": [[x,y], ...],        # station i is stations[i]  (integers)
             "features":  [[x,y,v], ...]}     # a formation at (x,y) worth v>0 (integers)
  stdout: ONE JSON object:
            {"tour": [i0, i1, ..., ik]}       # station indices, cyclic order of the loop

  A loop is VALID iff "tour" is a list of >=3 DISTINCT integer station indices, each in
  range 0<=i<len(stations), whose polygon (in the given cyclic order) is SIMPLE: no two
  non-adjacent edges intersect and no three consecutive vertices are collinear.  A
  self-intersecting loop, an out-of-range / duplicate / non-integer index, fewer than 3
  vertices, a crash, a timeout, or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance the evaluator computes:
    base = value of the BEST SINGLE TRIANGLE (loop over 3 stations) -- a weak reference
           that captures at most one cluster and pays a triangle's rope; > 0 by design.
    ub   = sum of ALL formation values, paying NO rope  -- an optimistic, unreachable
           bound (ignores both perimeter cost and the fact that no single simple loop
           can strictly enclose every formation cheaply).
    cand = loop value achieved by the candidate.
  and normalises with an affine anchor (weak triangle -> 0.1, ideal -> 1.0):
    r = clamp( 0.1 + 0.9 * (cand - base) / (ub - base), 0, 1 )
  A candidate that only strings the best triangle scores ~0.1; the (unreachable)
  all-formations-no-rope bound scores 1.0; doing worse than the best triangle scores
  < 0.1.  Because `ub` ignores rope AND enclosure geometry, even excellent loops stay
  well below 1.0 -> headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH OS-SANDBOXED SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance.  The references (base,
ub) and the simplicity / strict-containment checks are computed by THIS parent process,
so a frame-walking / introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt_int(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt_int


# ----------------------------- geometry primitives -------------------------
def _orient(a, b, c):
    """Twice the signed area of triangle abc (exact for integer coords)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_seg(a, b, p):
    """p is assumed collinear with a,b; return True if within the segment bbox."""
    return (min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and
            min(a[1], b[1]) <= p[1] <= max(a[1], b[1]))


def _seg_int(p1, p2, p3, p4):
    """True iff closed segments p1p2 and p3p4 share any point (proper or touching)."""
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    if d1 == 0 and _on_seg(p3, p4, p1):
        return True
    if d2 == 0 and _on_seg(p3, p4, p2):
        return True
    if d3 == 0 and _on_seg(p1, p2, p3):
        return True
    if d4 == 0 and _on_seg(p1, p2, p4):
        return True
    return False


def _is_simple(V):
    """V is a list of vertex coordinates in cyclic order.  Return True iff it is a
    simple (non self-intersecting) polygon with no collinear consecutive triple."""
    n = len(V)
    if n < 3:
        return False
    if len(set(map(tuple, V))) != n:            # repeated vertex coordinate
        return False
    for i in range(n):                          # no degenerate / spike vertices
        if _orient(V[i], V[(i + 1) % n], V[(i + 2) % n]) == 0:
            return False
    for i in range(n):                          # no two NON-adjacent edges may touch
        a1, a2 = V[i], V[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue                        # adjacent edges share a vertex: allowed
            b1, b2 = V[j], V[(j + 1) % n]
            if _seg_int(a1, a2, b1, b2):
                return False
    return True


def _strict_inside(p, V):
    """True iff point p is STRICTLY inside simple polygon V (boundary excluded)."""
    n = len(V)
    for i in range(n):                          # on an edge -> not strictly inside
        a, b = V[i], V[(i + 1) % n]
        if _orient(a, b, p) == 0 and _on_seg(a, b, p):
            return False
    x, y = p
    inside = False
    for i in range(n):
        x1, y1 = V[i]
        x2, y2 = V[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xint:
                inside = not inside
    return inside


def _pt_in_tri(p, a, b, c):
    """Strict point-in-triangle (boundary excluded), exact integer arithmetic."""
    d1 = _orient(a, b, p)
    d2 = _orient(b, c, p)
    d3 = _orient(c, a, p)
    return (d1 > 0 and d2 > 0 and d3 > 0) or (d1 < 0 and d2 < 0 and d3 < 0)


def _perim(V):
    n = len(V)
    s = 0.0
    for i in range(n):
        ax, ay = V[i]
        bx, by = V[(i + 1) % n]
        s += math.hypot(ax - bx, ay - by)
    return s


# ----------------------------- instance family -----------------------------
def _accept(stations, cand):
    """Accept a new station only if it is distinct AND in general position
    (forms no collinear triple with any existing pair).  This guarantees that any
    loop over distinct stations has no collinear-consecutive vertices, so the
    simplicity test never spuriously rejects a convex-hull loop."""
    for s in stations:
        if s[0] == cand[0] and s[1] == cand[1]:
            return False
    n = len(stations)
    for i in range(n):
        for j in range(i + 1, n):
            if _orient(stations[i], stations[j], cand) == 0:
                return False
    return True


def _gen_instance(seed, S, grid, clusters, cluster_pts, lam, scatter, cval, sval):
    ni = _rng(seed)
    # four corner stations (indices 0..3), then a jittered interior grid
    stations = [[0, 0], [S, 0], [S, S], [0, S]]
    for a in range(1, grid + 1):
        for b in range(1, grid + 1):
            x = a * S // (grid + 1) + ni(-18, 18)
            y = b * S // (grid + 1) + ni(-18, 18)
            cand = [x, y]
            if _accept(stations, cand):
                stations.append(cand)
    # formation clusters: spread apart so no single triangle covers two cheaply
    centers = []
    tries = 0
    while len(centers) < clusters and tries < 800:
        tries += 1
        cx = ni(220, S - 220)
        cy = ni(220, S - 220)
        if all(abs(cx - px) + abs(cy - py) >= 360 for (px, py) in centers):
            centers.append((cx, cy))
    features = []
    for (cx, cy) in centers:
        m = cluster_pts
        for _ in range(m):
            fx = min(S - 5, max(5, cx + ni(-45, 45)))
            fy = min(S - 5, max(5, cy + ni(-45, 45)))
            v = ni(cval[0], cval[1])
            features.append([fx, fy, v])
    # low-value formations parked near the far corners (rope to reach them rarely pays)
    for _ in range(scatter):
        corner = ni(0, 3)
        bx = [55, S - 55, S - 55, 55][corner]
        by = [55, 55, S - 55, S - 55][corner]
        fx = min(S - 5, max(5, bx + ni(-70, 70)))
        fy = min(S - 5, max(5, by + ni(-70, 70)))
        v = ni(sval[0], sval[1])
        features.append([fx, fy, v])
    return {"name": f"cave{seed}", "S": S, "lam": lam,
            "stations": stations, "features": features}


def _build_instances():
    # (seed, S, grid, clusters, cluster_pts, lam, scatter, cval, sval)
    specs = [
        (301, 1000, 5, 3, 4, 0.15, 4, (40, 90), (5, 15)),
        (302, 1000, 5, 4, 4, 0.15, 5, (40, 90), (5, 15)),
        (303, 1000, 5, 3, 5, 0.12, 4, (40, 90), (5, 15)),
        (304, 1000, 5, 4, 3, 0.20, 5, (40, 90), (5, 15)),   # high rope: be selective
        (305, 1000, 5, 4, 4, 0.15, 6, (40, 90), (5, 15)),
        (306, 1000, 5, 4, 5, 0.10, 4, (40, 90), (5, 15)),   # cheap rope: enclose more
        (307, 1200, 5, 4, 4, 0.18, 6, (40, 90), (5, 15)),
        (308, 1000, 5, 5, 3, 0.15, 7, (40, 90), (5, 15)),
        # harder / larger held-out instances
        (411, 1200, 5, 5, 4, 0.13, 8, (40, 95), (5, 15)),
        (412, 1200, 5, 4, 5, 0.22, 6, (40, 95), (5, 15)),   # high rope + dense clusters
        (413, 1400, 5, 5, 4, 0.16, 8, (45, 95), (5, 18)),
        (414, 1400, 5, 6, 3, 0.14, 10, (45, 95), (5, 18)),
    ]
    return [_gen_instance(*s) for s in specs]


# ----------------------------- references ----------------------------------
def _base(inst):
    """Value of the BEST single triangle over 3 stations (weak reference)."""
    st = inst["stations"]
    feats = inst["features"]
    lam = inst["lam"]
    n = len(st)
    best = None
    for i in range(n):
        ai = st[i]
        for j in range(i + 1, n):
            bj = st[j]
            for k in range(j + 1, n):
                ck = st[k]
                if _orient(ai, bj, ck) == 0:
                    continue
                val = 0
                for (fx, fy, fv) in feats:
                    if _pt_in_tri((fx, fy), ai, bj, ck):
                        val += fv
                v = val - lam * _perim([ai, bj, ck])
                if best is None or v > best:
                    best = v
    return best if best is not None else 0.0


def _ub(inst):
    return float(sum(fv for (_, _, fv) in inst["features"]))


# ----------------------------- validation / scoring ------------------------
def _loop_value(inst, ans):
    """Validate the answer; return the loop value (float) or None if infeasible."""
    if not isinstance(ans, dict):
        return None
    tour = ans.get("tour")
    if not isinstance(tour, list) or len(tour) < 3:
        return None
    st = inst["stations"]
    ns = len(st)
    seen = set()
    V = []
    for idx in tour:
        if isinstance(idx, bool) or not isinstance(idx, int):
            return None
        if idx < 0 or idx >= ns:
            return None
        if idx in seen:
            return None
        seen.add(idx)
        V.append(st[idx])
    if not _is_simple(V):
        return None
    val = 0
    for (fx, fy, fv) in inst["features"]:
        if _strict_inside((fx, fy), V):
            val += fv
    return float(val - inst["lam"] * _perim(V))


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        base = _base(inst)
        ub = _ub(inst)
        denom = ub - base
        if denom < 1e-9:
            denom = 1e-9
        public = {"name": inst["name"], "S": inst["S"], "lam": inst["lam"],
                  "stations": [list(p) for p in inst["stations"]],
                  "features": [list(f) for f in inst["features"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            val = _loop_value(inst, ans)
        except Exception:
            val = None
        if val is None:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (val - base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
