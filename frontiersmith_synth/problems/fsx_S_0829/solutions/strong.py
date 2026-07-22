# TIER: strong
# Insight: classify the (unlabeled) layout from a CHEAP nearest-neighbor-
# distance histogram, then pick a constructor that matches the diagnosed
# regime -- cluster-first for strongly clustered layouts, a space-filling
# sweep for uniform/grid layouts -- instead of always reaching for plain
# nearest-neighbor. The instance also tells us `refine_budget`: the exact
# number of improving moves the evaluator will spend polishing whatever we
# submit. So rather than guess which construction is better from its raw
# length, we replicate the evaluator's own budgeted 2-opt + Or-opt refine
# locally and submit whichever candidate construction (plain NN vs the
# regime-matched one) comes out shorter AFTER that refine -- classification
# picks the right BASIN, and the local replay picks the right constructor
# for THIS budget, not just in principle.
import sys, json, math


def dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)


def tour_length(points, tour):
    n = len(tour)
    return sum(dist(points[tour[i]], points[tour[(i + 1) % n]]) for i in range(n))


def construct_nn(points, start=0):
    n = len(points)
    visited = [False] * n
    visited[start] = True
    tour = [start]
    cur = start
    for _ in range(n - 1):
        best_j, best_d = -1, float("inf")
        for j in range(n):
            if not visited[j]:
                d = dist(points[cur], points[j])
                if d < best_d:
                    best_d, best_j = d, j
        visited[best_j] = True
        tour.append(best_j)
        cur = best_j
    return tour


def construct_sweep(points):
    n = len(points)
    xs = [p[0] for p in points]
    strips = max(1, round(math.sqrt(n / 2.0)))
    xmin, xmax = min(xs), max(xs)
    width = max(xmax - xmin, 1e-9)
    buckets = [[] for _ in range(strips)]
    for i, p in enumerate(points):
        b = int((p[0] - xmin) / width * strips)
        b = min(max(b, 0), strips - 1)
        buckets[b].append(i)
    tour = []
    for b in range(strips):
        pts_b = buckets[b]
        pts_b.sort(key=lambda i: points[i][1], reverse=(b % 2 == 1))
        tour.extend(pts_b)
    return tour


def mst_clusters(points, k_target_ratio=3.0):
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
                d = dist(points[u], points[v])
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


def order_clusters(points, clusters):
    cents = []
    for c in clusters:
        cx = sum(points[i][0] for i in c) / len(c)
        cy = sum(points[i][1] for i in c) / len(c)
        cents.append([cx, cy])
    m = len(clusters)
    order = list(construct_nn(cents, start=0)) if m > 1 else [0]
    if m > 2:
        improved = True
        while improved:
            improved = False
            for i in range(m - 1):
                for j in range(i + 2, m):
                    a, b = order[i], order[i + 1]
                    c = order[j]
                    d = order[j + 1] if j + 1 < m else None
                    old = dist(cents[a], cents[b])
                    if d is not None:
                        old += dist(cents[c], cents[d])
                    new = dist(cents[a], cents[c])
                    if d is not None:
                        new += dist(cents[b], cents[d])
                    if new < old - 1e-9:
                        order[i + 1:j + 1] = reversed(order[i + 1:j + 1])
                        improved = True
    return order


def path_from_entry(points, idxs, entry_idx):
    remaining = set(idxs)
    remaining.discard(entry_idx)
    path = [entry_idx]
    cur = entry_idx
    while remaining:
        nxt = min(remaining, key=lambda j: dist(points[cur], points[j]))
        remaining.discard(nxt)
        path.append(nxt)
        cur = nxt
    return path


def construct_cluster_first(points):
    clusters = mst_clusters(points)
    if len(clusters) <= 1:
        return construct_nn(points)
    order = order_clusters(points, clusters)
    tour = []
    prev_point = None
    for ci in order:
        c = clusters[ci]
        if len(c) == 1:
            tour.extend(c)
            prev_point = points[c[0]]
            continue
        entry = c[0] if prev_point is None else min(c, key=lambda i: dist(prev_point, points[i]))
        path = path_from_entry(points, c, entry)
        tour.extend(path)
        prev_point = points[path[-1]]
    return tour


def classify_regime(points):
    """Cheap nearest-neighbor-distance histogram: compare the mean NN distance
    against the value expected for a uniform Poisson layout of the same
    density. Clustered layouts pack points far closer than that; a very
    regular (low-variance) grid sits notably farther apart than that."""
    n = len(points)
    nn_dists = []
    for i in range(n):
        best = min(dist(points[i], points[j]) for j in range(n) if j != i)
        nn_dists.append(best)
    mean_nn = sum(nn_dists) / n
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    area = max((max(xs) - min(xs)) * (max(ys) - min(ys)), 1e-9)
    density = n / area
    expected_uniform_nn = 0.5 / math.sqrt(density) if density > 0 else 1.0
    ratio = mean_nn / expected_uniform_nn if expected_uniform_nn > 0 else 1.0
    if ratio > 1.5:
        return "grid"
    if ratio < 0.5:
        return "clustered"
    return "uniform"


# ---- budgeted local search, IDENTICAL to the evaluator's, so we can predict
#      which raw construction will actually come out ahead after refine ----
def local_search(points, tour, budget):
    tour = list(tour)
    n = len(tour)
    if n < 4 or budget <= 0:
        return tour
    used = 0
    while used < budget:
        improved = False
        for i in range(n - 1):
            a, b = tour[i], tour[i + 1]
            dab = dist(points[a], points[b])
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c, d = tour[j], tour[(j + 1) % n]
                delta = (dist(points[a], points[c]) + dist(points[b], points[d])) \
                    - (dab + dist(points[c], points[d]))
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
                gain_remove = (dist(points[p], points[s0]) + dist(points[s1], points[q])) \
                    - dist(points[p], points[q])
                if gain_remove <= 1e-9:
                    continue
                best_delta, best_j, best_rev = 0.0, None, False
                for j in range(n):
                    if j in idxs or (j + 1) % n in idxs or j == prev_idx:
                        continue
                    aa, bb = tour[j], tour[(j + 1) % n]
                    base = dist(points[aa], points[bb])
                    add_f = dist(points[aa], points[s0]) + dist(points[s1], points[bb])
                    df = (add_f - base) - gain_remove
                    if df < best_delta - 1e-9:
                        best_delta, best_j, best_rev = df, j, False
                    if seg_len > 1:
                        add_r = dist(points[aa], points[s1]) + dist(points[s0], points[bb])
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


inst = json.load(sys.stdin)
points = inst["points"]
budget = inst["refine_budget"]

regime = classify_regime(points)
nn_tour = construct_nn(points, start=0)
if regime == "clustered":
    alt_tour = construct_cluster_first(points)
else:
    alt_tour = construct_sweep(points)

refined_nn = local_search(points, nn_tour, budget)
refined_alt = local_search(points, alt_tour, budget)

best_tour = refined_nn if tour_length(points, refined_nn) <= tour_length(points, refined_alt) else refined_alt
print(json.dumps({"tour": best_tour}))
