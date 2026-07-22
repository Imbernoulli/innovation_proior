#!/usr/bin/env python3
"""Checker for fsx_A_0943 -- Star Atlas Portfolio.
Usage: chk.py <in> <out> <ans>   (ans is unused; scorer problem).
Always exits 0; prints "... Ratio: <float>" -- the harness parses the LAST such token.
"""
import sys, math

def fail(reason, F=0.0, B=0.0):
    print(f"WA F={F:.6f} B={B:.6f} reason={reason} Ratio: 0.000000")
    sys.exit(0)


def read_tokens(path):
    with open(path, "r") as f:
        return f.read().split()


def cross(ox, oy, ax, ay, bx, by):
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def on_segment(px, py, qx, qy, rx, ry):
    # is r on segment pq, given p,q,r collinear
    return min(px, qx) - 1e-9 <= rx <= max(px, qx) + 1e-9 and \
           min(py, qy) - 1e-9 <= ry <= max(py, qy) + 1e-9


def segs_properly_bad(A, B, C, D):
    """A-B and C-D are two edges (each a (x,y) endpoint pair). Returns True iff they
    constitute a crossing-free-forest VIOLATION: any intersection other than touching at
    a single shared declared endpoint."""
    shared = []
    if A == C or A == D:
        shared.append(A)
    if B == C or B == D:
        shared.append(B)
    shared = [p for i, p in enumerate(shared) if p not in shared[:i]]

    if len(shared) >= 1:
        # they are allowed to touch only at the single shared vertex; check for
        # collinear overlap beyond that point.
        s = shared[0]
        other1 = B if A == s else A
        other2 = D if C == s else C
        cr = cross(s[0], s[1], other1[0], other1[1], other2[0], other2[1])
        if cr != 0:
            return False  # not collinear -> only touch at s, fine
        # collinear: bad if other2 lies on ray s->other1 (or vice versa) beyond s,
        # i.e. the two segments overlap in more than the single point s.
        d1 = (other1[0] - s[0], other1[1] - s[1])
        d2 = (other2[0] - s[0], other2[1] - s[1])
        dot = d1[0] * d2[0] + d1[1] * d2[1]
        return dot > 0  # same direction from s -> overlapping beyond the shared point
    # no shared endpoint: standard proper-intersection / touching test
    d1 = cross(C[0], C[1], D[0], D[1], A[0], A[1])
    d2 = cross(C[0], C[1], D[0], D[1], B[0], B[1])
    d3 = cross(A[0], A[1], B[0], B[1], C[0], C[1])
    d4 = cross(A[0], A[1], B[0], B[1], D[0], D[1])
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True  # proper crossing
    if d1 == 0 and on_segment(C[0], C[1], D[0], D[1], A[0], A[1]):
        return True
    if d2 == 0 and on_segment(C[0], C[1], D[0], D[1], B[0], B[1]):
        return True
    if d3 == 0 and on_segment(A[0], A[1], B[0], B[1], C[0], C[1]):
        return True
    if d4 == 0 and on_segment(A[0], A[1], B[0], B[1], D[0], D[1]):
        return True
    return False


def compute_shape(size, deg, tree_adj, pos):
    """branch_c, curve_c for one cluster given its tree adjacency (point-id -> list of nbrs)."""
    max_deg = max(deg.values()) if deg else 1
    branch = (max_deg - 1) / (size - 2) if size >= 3 else 0.0
    branch = min(1.0, max(0.0, branch))

    turns = []
    for v, nbrs in tree_adj.items():
        if len(nbrs) == 2:
            a, b = nbrs
            v1 = (pos[v][0] - pos[a][0], pos[v][1] - pos[a][1])
            v2 = (pos[b][0] - pos[v][0], pos[b][1] - pos[v][1])
            n1 = math.hypot(*v1); n2 = math.hypot(*v2)
            if n1 < 1e-9 or n2 < 1e-9:
                continue
            cosv = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
            cosv = min(1.0, max(-1.0, cosv))
            turn = math.acos(cosv)  # 0 = straight continuation, pi = full u-turn
            turns.append(turn / math.pi)
    curve = sum(turns) / len(turns) if turns else 0.0
    return branch, curve


def role_fit(role, branch, curve):
    if role == 0:    # PATH
        return (1 - branch) * (1 - curve)
    elif role == 1:  # STAR
        return branch
    else:            # ZIGZAG
        return (1 - branch) * curve


def compute_F(N, K, pos, cluster_of, cluster_pts, edges_by_cluster, roles):
    """pos: point-id(1..N) -> (x,y). cluster_pts: cid -> list of point ids.
    edges_by_cluster: cid -> list of (u,v). roles: cid -> int in {0,1,2}."""
    deg_global = {i: 0 for i in range(1, N + 1)}
    adj_global = {i: [] for i in range(1, N + 1)}
    all_edges = []
    for cid in range(K):
        for (u, v) in edges_by_cluster[cid]:
            deg_global[u] += 1; deg_global[v] += 1
            adj_global[u].append(v); adj_global[v].append(u)
            all_edges.append((u, v))

    # angular resolution: for every junction (vertex with 2+ incident segments), its own
    # angular resolution is the smallest angle between any two segments meeting there
    # (equivalently the smallest circular gap between their sorted directions). The atlas
    # score is the AVERAGE of this per-junction value (normalized by pi) over all
    # junctions -- rewarding generously spread-out junctions everywhere, not just the
    # single worst one, so one unlucky vertex cannot alone collapse the whole atlas.
    junction_scores = []
    for v in range(1, N + 1):
        if deg_global[v] < 2:
            continue
        angs = sorted(math.atan2(pos[u][1] - pos[v][1], pos[u][0] - pos[v][0]) for u in adj_global[v])
        m = len(angs)
        vmin = None
        for i in range(m):
            a1 = angs[i]
            a2 = angs[(i + 1) % m] + (2 * math.pi if i == m - 1 else 0)
            gap = a2 - a1
            if vmin is None or gap < vmin:
                vmin = gap
        junction_scores.append(min(1.0, max(0.0, vmin / math.pi)))
    ang_score = sum(junction_scores) / len(junction_scores) if junction_scores else 1.0

    # crossing-free-ness (soft, over all edge pairs)
    m_edges = len(all_edges)
    bad = 0
    total_pairs = m_edges * (m_edges - 1) // 2
    if total_pairs > 0:
        for i in range(m_edges):
            u1, v1 = all_edges[i]
            A, B = pos[u1], pos[v1]
            for j in range(i + 1, m_edges):
                u2, v2 = all_edges[j]
                C, D = pos[u2], pos[v2]
                if segs_properly_bad(A, B, C, D):
                    bad += 1
        cross_score = 1.0 - bad / total_pairs
    else:
        cross_score = 1.0

    # per-cluster shape + fit
    fits = []
    shapevecs = []
    for cid in range(K):
        size = len(cluster_pts[cid])
        tree_adj = {p: [] for p in cluster_pts[cid]}
        deg_c = {p: 0 for p in cluster_pts[cid]}
        for (u, v) in edges_by_cluster[cid]:
            tree_adj[u].append(v); tree_adj[v].append(u)
            deg_c[u] += 1; deg_c[v] += 1
        branch, curve = compute_shape(size, deg_c, tree_adj, pos)
        fits.append(role_fit(roles[cid], branch, curve))
        shapevecs.append((branch, curve))

    mean_fit = sum(fits) / len(fits) if fits else 0.0

    pair_cnt = 0
    dist_sum = 0.0
    for i in range(K):
        for j in range(i + 1, K):
            dx = shapevecs[i][0] - shapevecs[j][0]
            dy = shapevecs[i][1] - shapevecs[j][1]
            dist_sum += math.hypot(dx, dy)
            pair_cnt += 1
    div_score = (dist_sum / pair_cnt) / math.sqrt(2.0) if pair_cnt else 0.0
    div_score = min(1.0, max(0.0, div_score))

    F = 0.06 * ang_score + 0.06 * cross_score + 0.33 * mean_fit + 0.55 * div_score
    return F, (ang_score, cross_score, mean_fit, div_score)


def trivial_construction(N, K, pos, cluster_of, cluster_pts):
    """Baseline: connect each cluster's points as a simple path in file-appearance order;
    declare every role as PATH(0)."""
    edges_by_cluster = {}
    for cid in range(K):
        pts = cluster_pts[cid]
        edges_by_cluster[cid] = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    roles = {cid: 0 for cid in range(K)}
    return compute_F(N, K, pos, cluster_of, cluster_pts, edges_by_cluster, roles)


def main():
    inf, ouf = sys.argv[1], sys.argv[2]
    itok = read_tokens(inf)
    it = iter(itok)
    N = int(next(it)); K = int(next(it))
    pos = {}
    cluster_of = {}
    cluster_pts = {cid: [] for cid in range(K)}
    for i in range(1, N + 1):
        x = int(next(it)); y = int(next(it)); c = int(next(it))
        pos[i] = (x, y); cluster_of[i] = c
        cluster_pts[c].append(i)

    Fb, _ = trivial_construction(N, K, pos, cluster_of, cluster_pts)
    B = max(Fb, 1e-6)

    try:
        otok = read_tokens(ouf)
    except Exception:
        fail("cannot read output", 0.0, B)
    oit = iter(otok)

    def next_int(lo, hi, name):
        try:
            tok = next(oit)
        except StopIteration:
            fail(f"missing token for {name}", 0.0, B)
        try:
            v = int(tok)
        except ValueError:
            fail(f"non-integer token for {name}: {tok!r}", 0.0, B)
        if not (lo <= v <= hi):
            fail(f"{name}={v} out of range [{lo},{hi}]", 0.0, B)
        return v

    roles = {}
    for cid in range(K):
        roles[cid] = next_int(0, 2, f"role[{cid}]")

    m_expected = N - K
    edges_by_cluster = {cid: [] for cid in range(K)}
    seen_edges = set()
    dsu = list(range(N + 1))

    def find(x):
        while dsu[x] != x:
            dsu[x] = dsu[dsu[x]]; x = dsu[x]
        return x

    edge_count_per_cluster = {cid: 0 for cid in range(K)}
    for e in range(m_expected):
        u = next_int(1, N, f"edge[{e}].u")
        v = next_int(1, N, f"edge[{e}].v")
        if u == v:
            fail(f"self-loop at edge {e}: {u}", 0.0, B)
        cu, cv = cluster_of[u], cluster_of[v]
        if cu != cv:
            fail(f"edge {e} ({u},{v}) crosses clusters {cu}!={cv}", 0.0, B)
        key = (min(u, v), max(u, v))
        if key in seen_edges:
            fail(f"duplicate edge {key}", 0.0, B)
        seen_edges.add(key)
        ru, rv = find(u), find(v)
        if ru == rv:
            fail(f"edge {key} creates a cycle in cluster {cu}", 0.0, B)
        dsu[ru] = rv
        edges_by_cluster[cu].append((u, v))
        edge_count_per_cluster[cu] += 1

    # trailing garbage check
    try:
        extra = next(oit)
        fail(f"trailing token {extra!r}", 0.0, B)
    except StopIteration:
        pass

    for cid in range(K):
        size = len(cluster_pts[cid])
        if edge_count_per_cluster[cid] != size - 1:
            fail(f"cluster {cid} has {edge_count_per_cluster[cid]} edges, need {size - 1}", 0.0, B)
        # connectivity: all points of this cluster must share one DSU root
        roots = {find(p) for p in cluster_pts[cid]}
        if len(roots) != 1:
            fail(f"cluster {cid} spanning forest is disconnected", 0.0, B)

    F, comps = compute_F(N, K, pos, cluster_of, cluster_pts, edges_by_cluster, roles)
    if not all(math.isfinite(x) for x in (F, B)):
        fail("non-finite score", 0.0, B)

    sc = min(1000.0, 100.0 * F / B)
    ratio = sc / 1000.0
    a, cr, ft, dv = comps
    print(f"OK F={F:.6f} B={B:.6f} ang={a:.4f} cross={cr:.4f} fit={ft:.4f} div={dv:.4f} Ratio: {ratio:.6f}")
    sys.exit(0)


if __name__ == "__main__":
    main()
