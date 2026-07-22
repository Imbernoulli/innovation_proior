#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0927 -- "One Circuit, No Nemesis" (family:
dispatcher-over-instance-families; format B, quality-metric).

THEME.  A courier's single delivery vehicle must complete a closed loop through
every stop in a territory before returning to start (a Euclidean-cycle "circuit"
routing problem). The courier keeps a *portfolio* of route-construction rules but
only ever has time to run one of them per territory, so the real design object is
the DISPATCH RULE that decides -- from cheap, quickly-computed structural signals
about the stop layout -- which construction to trust, before polishing whatever it
picks with a shared local-search pass. Territories are drawn from FIVE fixed
layout families (tight pockets, uniform scatter, winding corridors, radiating
hub-and-spoke networks, and separated grid blocks); the layout's family identity
is NEVER given to the candidate, only the raw stop coordinates, so a genuine
structural read is required. Scoring reports the MINIMUM family-mean ratio across
all five families (not a plain average) -- a rule that wins on four families and
collapses on the fifth is scored as if it always collapses, so the ONE recipe that
lands every family's "nemesis" case is the entire game.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n": N (int), "points": [[x0,y0], ..., [x_{N-1},y_{N-1}]]}
          (family identity is intentionally withheld)
  stdout: ONE JSON object:
            {"tour": [p_0, ..., p_{N-1}]}
          a permutation of 0..N-1 describing visit order of a closed cycle
          (after the last stop the courier returns to p_0).

  A tour is VALID iff it is a list of exactly N distinct integers, each in
  [0, N-1]. Invalid output, wrong length, a repeated/out-of-range index, a crash,
  a timeout, or non-JSON output -> that instance scores 0.0.

SCORING (deterministic; no wall-time). Per instance the evaluator computes,
itself, from the FULL point set:
    L_lb   = MST weight of the points          # a real lower bound on any cycle
    L_base = length of the "sort stops by x"   # a weak, deterministic reference
             sweep-line cycle                  # tour (anchors r = 0.1)
    L_cand = length of the candidate's cycle
and normalizes with an affine anchor:
    r = clamp( 0.1 + 0.9 * (L_base - L_cand) / max(1e-9, L_base - L_lb), 0, 1 )
Matching the x-sort reference scores ~0.1; reaching the (generally unreachable)
MST lower bound scores 1.0; doing worse than the x-sort reference scores < 0.1.

The per-instance ratios are grouped into their FIVE generating families (two
instances each). The reported **Ratio** is the MINIMUM of the five per-family
means (worst-family-approximation); **Vector** lists all ten per-instance ratios
in generation order.

ISOLATION. The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance. L_lb/L_base are
computed by THIS parent process from the full instance, so a frame-walking or
introspecting candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <min over the 5 family means, in [0,1]>
  Vector: [r_1, ..., r_10]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
def _rng(seed):
    state = [seed * 6364136223846793005 + 1442695040888963407 & ((1 << 64) - 1)]

    def nxt():
        state[0] = (state[0] * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return state[0]

    def randf():
        return (nxt() >> 11) / (1 << 53)

    def randr(lo, hi):
        return lo + randf() * (hi - lo)

    def randint(lo, hi):
        return lo + nxt() % (hi - lo + 1)

    return randf, randr, randint


def _shuffle(lst, randint):
    a = list(lst)
    for i in range(len(a) - 1, 0, -1):
        j = randint(0, i)
        a[i], a[j] = a[j], a[i]
    return a


# ----------------------------- family generators ----------------------------
def _gen_uniform(seed, n, box=100.0):
    _, randr, randint = _rng(seed)
    return [(randr(0, box), randr(0, box)) for _ in range(n)]


def _gen_hub_spoke(seed, k_spokes, per_spoke, spoke_len, jitter):
    _, randr, randint = _rng(seed)
    pts = []
    for k in range(k_spokes):
        ang = 2 * math.pi * k / k_spokes + randr(-0.05, 0.05)
        for j in range(1, per_spoke + 1):
            r = spoke_len * j / per_spoke
            jx = randr(-jitter, jitter)
            jy = randr(-jitter, jitter)
            pts.append((r * math.cos(ang) + jx, r * math.sin(ang) + jy))
    idx = _shuffle(list(range(len(pts))), randint)
    return [pts[i] for i in idx]


def _gen_clustered_outlier(seed, k_clusters, per_cluster, cluster_r, field, n_outliers):
    _, randr, randint = _rng(seed)
    centers = []
    gcols = int(math.ceil(math.sqrt(k_clusters)))
    cell = field / gcols
    for k in range(k_clusters):
        gx, gy = k % gcols, k // gcols
        cx = cell * (gx + 0.5) + randr(-cell * 0.15, cell * 0.15)
        cy = cell * (gy + 0.5) + randr(-cell * 0.15, cell * 0.15)
        centers.append((cx, cy))
    pts = []
    for (cx, cy) in centers:
        for _ in range(per_cluster):
            ang = randr(0, 2 * math.pi)
            rad = randr(0, cluster_r)
            pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    for i in range(n_outliers):
        c1 = centers[i % len(centers)]
        c2 = centers[(i + len(centers) // 2) % len(centers)]
        t = randr(0.35, 0.65)
        ox = c1[0] + t * (c2[0] - c1[0]) + randr(-cluster_r * 0.3, cluster_r * 0.3)
        oy = c1[1] + t * (c2[1] - c1[1]) + randr(-cluster_r * 0.3, cluster_r * 0.3)
        pts.append((ox, oy))
    idx = _shuffle(list(range(len(pts))), randint)
    return [pts[i] for i in idx]


def _gen_corridor(seed, n, seg_len, width, turns):
    _, randr, randint = _rng(seed)
    verts = [(0.0, 0.0)]
    ang = 0.0
    for t in range(turns):
        ang += randr(1.2, 2.0) * (1 if t % 2 == 0 else -1)
        vx, vy = verts[-1]
        verts.append((vx + seg_len * math.cos(ang), vy + seg_len * math.sin(ang)))
    total_len = seg_len * turns
    pts = []
    for i in range(n):
        t = randr(0, total_len)
        seg = min(int(t / seg_len), turns - 1)
        local = (t - seg * seg_len) / seg_len
        x0, y0 = verts[seg]
        x1, y1 = verts[seg + 1]
        bx = x0 + local * (x1 - x0)
        by = y0 + local * (y1 - y0)
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy) or 1.0
        px, py = -dy / L, dx / L
        w = randr(-width, width)
        pts.append((bx + px * w, by + py * w))
    idx = _shuffle(list(range(n)), randint)
    return [pts[i] for i in idx]


def _gen_grid_blocks(seed, blocks, rows, cols, dx, dy, gap):
    _, randr, randint = _rng(seed)
    pts = []
    bx = 0.0
    for b in range(blocks):
        ox = bx
        oy = randr(-dy * 0.5, dy * 0.5)
        for r in range(rows):
            for c in range(cols):
                jx = randr(-dx * 0.08, dx * 0.08)
                jy = randr(-dy * 0.08, dy * 0.08)
                pts.append((ox + c * dx + jx, oy + r * dy + jy))
        bx += cols * dx + gap
    idx = _shuffle(list(range(len(pts))), randint)
    return [pts[i] for i in idx]


def _build_instances():
    """5 families x 2 instances = 10 total. Family identity kept ONLY here
    (internal), never sent to the candidate."""
    specs = [
        ("clustered", lambda: _gen_clustered_outlier(3001, 5, 6, 3.0, 100.0, 2)),
        ("clustered", lambda: _gen_clustered_outlier(3002, 6, 5, 2.5, 110.0, 2)),
        ("uniform", lambda: _gen_uniform(3101, 32, 100.0)),
        ("uniform", lambda: _gen_uniform(3102, 36, 100.0)),
        ("corridor", lambda: _gen_corridor(3201, 30, 18.0, 3.0, 8)),
        ("corridor", lambda: _gen_corridor(3202, 34, 16.0, 2.5, 9)),
        ("hubspoke", lambda: _gen_hub_spoke(3301, 7, 5, 40.0, 1.2)),
        ("hubspoke", lambda: _gen_hub_spoke(3302, 8, 4, 45.0, 1.0)),
        ("gridblocks", lambda: _gen_grid_blocks(3401, 3, 4, 4, 6.0, 6.0, 40.0)),
        ("gridblocks", lambda: _gen_grid_blocks(3402, 2, 5, 5, 5.0, 5.0, 55.0)),
    ]
    out = []
    for i, (fam, gen) in enumerate(specs):
        pts = gen()
        out.append({"name": f"case{i:02d}_{fam}", "family": fam, "points": pts, "n": len(pts)})
    return out


# ----------------------------- geometry references --------------------------
def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _tour_length(pts, order):
    n = len(order)
    return sum(_dist(pts[order[i]], pts[order[(i + 1) % n]]) for i in range(n))


def _mst_weight(pts):
    n = len(pts)
    if n <= 1:
        return 0.0
    inmst = [False] * n
    key = [float("inf")] * n
    key[0] = 0.0
    total = 0.0
    for _ in range(n):
        u, best = -1, float("inf")
        for v in range(n):
            if not inmst[v] and key[v] < best:
                best, u = key[v], v
        inmst[u] = True
        total += best if best != float("inf") else 0.0
        for v in range(n):
            if not inmst[v]:
                d = _dist(pts[u], pts[v])
                if d < key[v]:
                    key[v] = d
    return total


def _xsort_order(pts):
    return sorted(range(len(pts)), key=lambda i: pts[i][0])


# ----------------------------- validation ------------------------------------
def _validate_tour(answer, n):
    if not isinstance(answer, dict):
        return None
    tour = answer.get("tour")
    if not isinstance(tour, list) or len(tour) != n:
        return None
    seen = set()
    out = []
    for v in tour:
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        if v < 0 or v >= n or v in seen:
            return None
        seen.add(v)
        out.append(v)
    return out


# ----------------------------- scoring driver --------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    fam_scores = {}
    for inst in instances:
        pts = inst["points"]
        n = inst["n"]
        L_lb = _mst_weight(pts)
        L_base = _tour_length(pts, _xsort_order(pts))
        denom = max(1e-9, L_base - L_lb)

        public = {"name": inst["name"], "n": n, "points": [list(p) for p in pts]}
        ans, st = isorun.run_candidate(cand, public, timeout=5)
        if st != "OK":
            r = 0.0
        else:
            try:
                order = _validate_tour(ans, n)
            except Exception:
                order = None
            if order is None:
                r = 0.0
            else:
                L_cand = _tour_length(pts, order)
                r = 0.1 + 0.9 * (L_base - L_cand) / denom
                if not (r == r) or r in (float("inf"), float("-inf")):
                    r = 0.0
                r = max(0.0, min(1.0, r))
        vec.append(r)
        fam_scores.setdefault(inst["family"], []).append(r)

    fam_means = [sum(v) / len(v) for v in fam_scores.values()]
    ratio = min(fam_means) if fam_means else 0.0

    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
