#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_S_0829 -- "Right Basin First: Regime-Aware Tour Building"
(family: regime-switch-tour-builder; format B, quality-metric).

THEME.  A point set (unlabeled -- could be strongly clustered, roughly uniform,
or a perturbed grid) must be visited by a single closed tour.  The candidate
submits ONE initial visiting order (a permutation).  This evaluator then
polishes that order itself, deterministically, with a HARD-BUDGETED local
search: up to `refine_budget` *accepted* improving moves drawn from a fixed
2-opt (edge exchange) + Or-opt (relocate a run of 1-3 consecutive stops)
neighborhood (mechanism: or-opt-2opt-refine).  The budget is small on purpose
-- enough to iron out local wrinkles, nowhere near enough to rebuild a bad
global plan from scratch.  So what really decides the score is the BASIN the
candidate's own construction starts in, not how a fixed, shared polishing pass
finishes it.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance) -- see statement.md.
          {"name":..., "n":.., "points":[[x,y],...], "refine_budget":<int>}
  stdout: ONE JSON object -- {"tour": [permutation of 0..n-1]}.
          Bad shape / not a permutation / non-finite -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance:
    final_len = length of the CANDIDATE's tour after this evaluator applies
                its own budgeted 2-opt + Or-opt refine (mechanism 1) to it.
    q_base    = same refine applied to the "no construction" identity-order
                tour (0,1,2,...,n-1) -- the weak reference (== trivial.py).
    q_target  = an internal, stronger reference: the BEST of several
                constructions (identity / nearest-neighbor / cluster-first /
                sweep), refined with a LARGER budget (never exposed to any
                candidate) -- an anchor that keeps the ceiling open.
    r = clamp( 0.1 + 0.75 * (q_base - final_len) / max(q_base - q_target, eps), 0, 1 )
  Ratio = mean(r) over 10 seeded instances.  Several instances are strongly
  clustered ON PURPOSE: naive nearest-neighbor construction there strands a
  few points, producing long late "return" jumps across the map that the
  small refine budget cannot repair (mechanism composition: the trap is a
  spatial regime that a construction blind to it walks straight into).

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
isorun.run_candidate; it only ever sees the PUBLIC instance.  All refinement,
scoring, and the internal stronger reference happen in THIS parent process.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <mean r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ----------------------------
def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        u = (state >> 11) / float(1 << 53)
        return lo + u * (hi - lo)

    return nxt


# ----------------------------- geometry helpers ------------------------------
def _dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def _tour_length(points, tour):
    n = len(tour)
    total = 0.0
    for i in range(n):
        total += _dist(points[tour[i]], points[tour[(i + 1) % n]])
    return total


# ----------------------------- point-set generators --------------------------
def _gen_uniform(seed, n, w, h):
    nx = _rng(seed)
    return [[nx(0.0, w), nx(0.0, h)] for _ in range(n)]


def _gen_grid(seed, gx, gy, spacing, jitter_frac):
    nx = _rng(seed)
    pts = []
    for i in range(gx):
        for j in range(gy):
            x = i * spacing + nx(-jitter_frac * spacing, jitter_frac * spacing)
            y = j * spacing + nx(-jitter_frac * spacing, jitter_frac * spacing)
            pts.append([x, y])
    return pts


def _gen_clustered(seed, per_cluster, spread, area_w, area_h, min_center_sep):
    nx = _rng(seed)
    centers = []
    tries = 0
    margin = spread * 3.0
    while len(centers) < len(per_cluster) and tries < 20000:
        tries += 1
        c = [nx(margin, area_w - margin), nx(margin, area_h - margin)]
        if all(_dist(c, c2) >= min_center_sep for c2 in centers):
            centers.append(c)
    while len(centers) < len(per_cluster):  # fallback if packing failed
        centers.append([nx(margin, area_w - margin), nx(margin, area_h - margin)])
    pts = []
    for ci, c in enumerate(centers):
        cnt = per_cluster[ci]
        for _ in range(cnt):
            ang = nx(0.0, 2.0 * math.pi)
            r = spread * math.sqrt(nx(0.0, 1.0))  # uniform-in-disk
            pts.append([c[0] + r * math.cos(ang), c[1] + r * math.sin(ang)])
    return pts


# ----------------------------- instance family --------------------------------
# (name, kind, seed, gen_kwargs, refine_budget).  refine_budget is a small, FIXED
# move count for the clustered ("trap") instances -- deliberately far below the
# number of cluster-transition edges a plain nearest-neighbor construction gets
# wrong, so limited 2-opt/Or-opt cannot fully repair it (mechanism composition:
# spatial-regime-classify -> strategy-switch-construction feeds directly into
# what or-opt-2opt-refine can and cannot fix).  Uniform/grid instances use a
# fraction of n instead -- there NN rarely creates more than a couple of bad
# edges, so a modest, size-scaled budget is already enough for near-ceiling
# repair regardless of which construction was used.
def _build_specs():
    return [
        ("uniA", "uniform", 101, dict(n=50, w=100.0, h=100.0), ("frac", 0.40)),
        ("gridA", "grid", 202, dict(gx=8, gy=7, spacing=13.0, jitter_frac=0.12), ("frac", 0.40)),
        ("uniB", "uniform", 404, dict(n=65, w=130.0, h=130.0), ("frac", 0.40)),
        ("gridB", "grid", 606, dict(gx=9, gy=8, spacing=15.0, jitter_frac=0.15), ("frac", 0.40)),
        ("uniC", "uniform", 808, dict(n=80, w=150.0, h=150.0), ("frac", 0.40)),   # held-out, larger
        ("clusA", "clustered", 94,
         dict(per_cluster=[4] * 20, spread=3.0, area_w=300.0, area_h=300.0,
              min_center_sep=42.0), ("fixed", 6)),                                # trap 1
        ("clusB", "clustered", 72,
         dict(per_cluster=[4] * 18, spread=2.5, area_w=260.0, area_h=260.0,
              min_center_sep=38.0), ("fixed", 6)),                                # trap 2
        ("clusC", "clustered", 27,
         dict(per_cluster=[4] * 22, spread=3.5, area_w=320.0, area_h=320.0,
              min_center_sep=44.0), ("fixed", 7)),                                # trap 3 (held-out, larger)
        ("clusD", "clustered", 69,
         dict(per_cluster=[6, 6, 2, 6, 2, 6, 2, 6, 2, 6, 2, 6, 2, 6, 2, 6, 2, 6, 2, 6],
              spread=3.0, area_w=300.0, area_h=300.0, min_center_sep=42.0),
         ("fixed", 7)),                                                            # trap 4: lumpy sizes
        ("clusE", "clustered", 19,
         dict(per_cluster=[4] * 20, spread=3.0, area_w=300.0, area_h=300.0,
              min_center_sep=42.0), ("fixed", 6)),                                # trap 5 (held-out)
    ]


# ----------------------------- constructions (evaluator's own, for anchors) --
def _construct_identity(n):
    return list(range(n))


def _construct_nn(points, start=0):
    n = len(points)
    visited = [False] * n
    visited[start] = True
    tour = [start]
    cur = start
    for _ in range(n - 1):
        best_j, best_d = -1, float("inf")
        for j in range(n):
            if not visited[j]:
                d = _dist(points[cur], points[j])
                if d < best_d:
                    best_d, best_j = d, j
        visited[best_j] = True
        tour.append(best_j)
        cur = best_j
    return tour


def _mst_clusters(points, k_target_ratio=3.0):
    """Prim MST, then cut the longest edges to split into natural clusters.
    Returns list of clusters (lists of point indices)."""
    n = len(points)
    if n <= 2:
        return [list(range(n))]
    in_tree = [False] * n
    dmin = [float("inf")] * n
    parent = [-1] * n
    dmin[0] = 0.0
    edges = []
    for _ in range(n):
        u = -1
        best = float("inf")
        for i in range(n):
            if not in_tree[i] and dmin[i] < best:
                best = dmin[i]
                u = i
        in_tree[u] = True
        if parent[u] != -1:
            edges.append((best, parent[u], u))
        for v in range(n):
            if not in_tree[v]:
                d = _dist(points[u], points[v])
                if d < dmin[v]:
                    dmin[v] = d
                    parent[v] = u
    if not edges:
        return [list(range(n))]
    lens = sorted(e[0] for e in edges)
    med = lens[len(lens) // 2]
    thresh = max(med * k_target_ratio, 1e-9)
    keep = [(a, b) for (d, a, b) in edges if d <= thresh]
    adj = {i: [] for i in range(n)}
    for a, b in keep:
        adj[a].append(b)
        adj[b].append(a)
    seen = [False] * n
    clusters = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for w in adj[u]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        clusters.append(comp)
    return clusters


def _order_clusters_by_centroid_nn(points, clusters):
    cents = []
    for c in clusters:
        cx = sum(points[i][0] for i in c) / len(c)
        cy = sum(points[i][1] for i in c) / len(c)
        cents.append([cx, cy])
    m = len(clusters)
    order = list(_construct_nn(cents, start=0)) if m > 1 else [0]
    if m > 2:  # cheap 2-opt polish on the small centroid-level chain (open path, not a cycle)
        improved = True
        while improved:
            improved = False
            for i in range(m - 1):
                for j in range(i + 2, m):
                    a, b = order[i], order[i + 1]
                    c, d = order[j], order[(j + 1) % m] if j + 1 < m else None
                    old = _dist(cents[a], cents[b])
                    if d is not None:
                        old += _dist(cents[c], cents[d])
                    new = _dist(cents[a], cents[c])
                    if d is not None:
                        new += _dist(cents[b], cents[d])
                    if new < old - 1e-9:
                        order[i + 1:j + 1] = reversed(order[i + 1:j + 1])
                        improved = True
    return order


def _path_from_entry(points, idxs, entry_idx):
    """Nearest-neighbor Hamiltonian PATH over idxs, starting at entry_idx."""
    remaining = set(idxs)
    remaining.discard(entry_idx)
    path = [entry_idx]
    cur = entry_idx
    while remaining:
        nxt = min(remaining, key=lambda j: _dist(points[cur], points[j]))
        remaining.discard(nxt)
        path.append(nxt)
        cur = nxt
    return path


def _construct_cluster_first(points):
    """Cluster-first construction with ENTRY/EXIT-aware chaining: order clusters
    coarsely (NN + light 2-opt on centroids), then for each cluster in that
    order build a nearest-neighbor PATH that *starts* at whichever of its own
    points is closest to the previous cluster's exit point -- so the link edge
    between consecutive clusters is chosen well, instead of an arbitrary
    independently-built sub-tour stitched together at whatever point happened
    to come first."""
    clusters = _mst_clusters(points)
    if len(clusters) <= 1:
        return _construct_nn(points)
    order = _order_clusters_by_centroid_nn(points, clusters)
    tour = []
    prev_point = None
    for ci in order:
        c = clusters[ci]
        if len(c) == 1:
            tour.extend(c)
            prev_point = points[c[0]]
            continue
        if prev_point is None:
            entry = c[0]
        else:
            entry = min(c, key=lambda i: _dist(prev_point, points[i]))
        path = _path_from_entry(points, c, entry)
        tour.extend(path)
        prev_point = points[path[-1]]
    return tour


def _construct_sweep(points):
    n = len(points)
    xs = [p[0] for p in points]
    strips = max(1, round(math.sqrt(n / 2.0)))
    xmin, xmax = min(xs), max(xs)
    width = max(xmax - xmin, 1e-9)
    buckets = [[] for _ in range(strips)]
    for i, p in enumerate(points):
        b = int((p[0] - xmin) / width * strips)
        if b >= strips:
            b = strips - 1
        if b < 0:
            b = 0
        buckets[b].append(i)
    tour = []
    for b in range(strips):
        pts_b = buckets[b]
        pts_b.sort(key=lambda i: points[i][1], reverse=(b % 2 == 1))
        tour.extend(pts_b)
    return tour


# ----------------------------- budgeted local search --------------------------
def _local_search(points, tour, budget):
    tour = list(tour)
    n = len(tour)
    if n < 4 or budget <= 0:
        return tour
    used = 0
    while used < budget:
        improved = False
        # --- 2-opt: first-improvement scan ---
        for i in range(n - 1):
            a, b = tour[i], tour[i + 1]
            dab = _dist(points[a], points[b])
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c, d = tour[j], tour[(j + 1) % n]
                delta = (_dist(points[a], points[c]) + _dist(points[b], points[d])) \
                    - (dab + _dist(points[c], points[d]))
                if delta < -1e-9:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    used += 1
                    improved = True
                    break
            if improved:
                break
        if improved:
            if used >= budget:
                break
            continue
        # --- Or-opt: relocate a run of 1..3 consecutive stops ---
        for seg_len in (1, 2, 3):
            if seg_len >= n - 2:
                continue
            moved = False
            for i in range(n):
                idxs = [(i + k) % n for k in range(seg_len)]
                prev_idx = (i - 1) % n
                next_idx = (i + seg_len) % n
                if prev_idx in idxs or next_idx in idxs:
                    continue
                p, s0, s1, q = tour[prev_idx], tour[idxs[0]], tour[idxs[-1]], tour[next_idx]
                gain_remove = (_dist(points[p], points[s0]) + _dist(points[s1], points[q])) \
                    - _dist(points[p], points[q])
                if gain_remove <= 1e-9:
                    continue
                best_delta, best_j, best_rev = 0.0, None, False
                for j in range(n):
                    if j in idxs or (j + 1) % n in idxs or j == prev_idx:
                        continue
                    aa, bb = tour[j], tour[(j + 1) % n]
                    base = _dist(points[aa], points[bb])
                    add_f = _dist(points[aa], points[s0]) + _dist(points[s1], points[bb])
                    df = (add_f - base) - gain_remove
                    if df < best_delta - 1e-9:
                        best_delta, best_j, best_rev = df, j, False
                    if seg_len > 1:
                        add_r = _dist(points[aa], points[s1]) + _dist(points[s0], points[bb])
                        dr = (add_r - base) - gain_remove
                        if dr < best_delta - 1e-9:
                            best_delta, best_j, best_rev = dr, j, True
                if best_j is not None:
                    seg = [tour[k] for k in idxs]
                    if best_rev:
                        seg.reverse()
                    rest = [tour[k] for k in range(n) if k not in idxs]
                    a_val = tour[best_j]
                    pos = rest.index(a_val)
                    rest[pos + 1:pos + 1] = seg
                    tour = rest
                    used += 1
                    moved = True
                    improved = True
                    break
            if moved:
                break
        if not improved:
            break
        if used >= budget:
            break
    return tour


# ----------------------------- instance materialization -----------------------
def _build_instances():
    out = []
    for name, kind, seed, kwargs, budget_spec in _build_specs():
        if kind == "uniform":
            pts = _gen_uniform(seed, **kwargs)
        elif kind == "grid":
            pts = _gen_grid(seed, **kwargs)
        else:
            pts = _gen_clustered(seed, **kwargs)
        n = len(pts)
        mode, val = budget_spec
        refine_budget = max(6, round(val * n)) if mode == "frac" else int(val)

        cand_id = _construct_identity(n)
        cand_nn = _construct_nn(pts, start=0)
        cand_cl = _construct_cluster_first(pts)
        cand_sw = _construct_sweep(pts)

        q_base = _tour_length(pts, _local_search(pts, cand_id, refine_budget))

        big_budget = max(refine_budget * 6 + 30, 60)
        q_target = min(
            _tour_length(pts, _local_search(pts, t, big_budget))
            for t in (cand_id, cand_nn, cand_cl, cand_sw)
        )

        out.append({
            "name": name, "n": n, "points": pts, "refine_budget": refine_budget,
            "q_base": q_base, "q_target": q_target,
        })
    return out


def baseline(inst):
    return inst["q_base"]


# ----------------------------- answer validation + scoring --------------------
def _validate_tour(answer, n):
    if not isinstance(answer, dict):
        return None
    t = answer.get("tour")
    if not isinstance(t, list) or len(t) != n:
        return None
    seen = set()
    out = []
    for v in t:
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        if v < 0 or v >= n or v in seen:
            return None
        seen.add(v)
        out.append(v)
    if len(out) != n:
        return None
    return out


def score(inst, answer):
    n = inst["n"]
    tour = _validate_tour(answer, n)
    if tour is None:
        return False, None
    refined = _local_search(inst["points"], tour, inst["refine_budget"])
    final_len = _tour_length(inst["points"], refined)
    if not (final_len == final_len) or final_len in (float("inf"), float("-inf")):
        return False, None
    return True, final_len


# ----------------------------- scoring driver ---------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        public = {
            "name": inst["name"], "n": inst["n"],
            "points": [list(p) for p in inst["points"]],
            "refine_budget": inst["refine_budget"],
        }
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            ok, final_len = score(inst, ans)
        except Exception:
            ok, final_len = False, None
        if not ok or final_len is None:
            vec.append(0.0)
            continue
        q_base = inst["q_base"]
        q_target = inst["q_target"]
        denom = max(q_base - q_target, 1e-9)
        # scale kept at 0.75 (not 0.9): even a candidate that reaches the internal
        # anchor q_target on an "easy" instance caps at 0.85, leaving headroom
        # above every reference solution (see AGENT_BRIEF_INNOVATION_ADDENDUM).
        r = 0.1 + 0.75 * (q_base - final_len) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            r = 0.0
        r = max(0.0, min(1.0, r))
        vec.append(r)

    ratio = sum(vec) / len(vec) if vec else 0.0
    print("Ratio: %.6f" % ratio)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
