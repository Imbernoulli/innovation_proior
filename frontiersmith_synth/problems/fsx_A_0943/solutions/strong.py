# TIER: strong
"""The insight: diversity across the PORTFOLIO is not something any single tree's local
optimization can create -- it has to be PLANNED before any tree is drawn. We (1) score
every constellation's natural elongation via PCA of its point cloud, (2) use that ranking
to assign a BALANCED spread of PATH/STAR/ZIGZAG roles across all K constellations (not a
uniform per-cluster local rule), then (3) build each tree with a role-SPECIFIC recipe that
actually realizes the declared grammar (a straight PCA-axis path, a centroid hub-and-spoke
star, or an axis-riffled zigzag) -- so the realized shapes genuinely diverge cluster to
cluster instead of all looking like "the same algorithm's blob"."""
import sys, math


def pca_axis(pts, pos):
    s = len(pts)
    cx = sum(pos[p][0] for p in pts) / s
    cy = sum(pos[p][1] for p in pts) / s
    sxx = sum((pos[p][0] - cx) ** 2 for p in pts) / s
    syy = sum((pos[p][1] - cy) ** 2 for p in pts) / s
    sxy = sum((pos[p][0] - cx) * (pos[p][1] - cy) for p in pts) / s
    tr = sxx + syy
    diff = sxx - syy
    disc = math.sqrt(diff * diff / 4 + sxy * sxy)
    l1 = tr / 2 + disc
    l2 = tr / 2 - disc
    theta = 0.5 * math.atan2(2 * sxy, diff) if not (sxy == 0 and diff == 0) else 0.0
    elong = (l1 - l2) / tr if tr > 1e-9 else 0.0
    return (cx, cy), (math.cos(theta), math.sin(theta)), elong


def build_path(pts, pos, axis):
    cx_cy, (ux, uy), _ = axis
    order = sorted(pts, key=lambda p: (pos[p][0] - cx_cy[0]) * ux + (pos[p][1] - cx_cy[1]) * uy)
    return [(order[i], order[i + 1]) for i in range(len(order) - 1)]


def build_zigzag(pts, pos, axis):
    """Sort along the PRIMARY axis into small consecutive windows (spatially local, so
    every step stays a short hop), and within each window order by the SECONDARY axis,
    alternating ascending/descending window to window -- a bounded, 'boustrophedic'
    switchback instead of a maximal front/back U-turn. Produces a consistently high
    turning signature without collapsing angular resolution the way a global riffle does."""
    cx_cy, (ux, uy), _ = axis
    vx, vy = -uy, ux
    scored = []
    for p in pts:
        dx, dy = pos[p][0] - cx_cy[0], pos[p][1] - cx_cy[1]
        scored.append((dx * ux + dy * uy, dx * vx + dy * vy, p))
    scored.sort(key=lambda t: t[0])
    W = 2
    seq = []
    for w0 in range(0, len(scored), W):
        chunk = scored[w0:w0 + W]
        chunk.sort(key=lambda t: t[1], reverse=((w0 // W) % 2 == 1))
        seq.extend(t[2] for t in chunk)
    return [(seq[k], seq[k + 1]) for k in range(len(seq) - 1)]


def circular_min_gap(angles):
    a = sorted(angles)
    m = len(a)
    best = None
    for i in range(m):
        a1 = a[i]
        a2 = a[(i + 1) % m] + (2 * math.pi if i == m - 1 else 0)
        g = a2 - a1
        if best is None or g < best:
            best = g
    return best if best is not None else 2 * math.pi


def build_star(pts, pos):
    """A full hub-and-spoke star, choosing the hub (among all candidates) that gives the
    most evenly angularly spread spokes -- same branching, better angular resolution."""
    best_hub, best_gap = None, -1.0
    for h in pts:
        angs = [math.atan2(pos[p][1] - pos[h][1], pos[p][0] - pos[h][0]) for p in pts if p != h]
        if len(angs) < 2:
            gap = 2 * math.pi
        else:
            gap = circular_min_gap(angs)
        if gap > best_gap:
            best_gap = gap; best_hub = h
    return [(best_hub, p) for p in pts if p != best_hub]


def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it)); K = int(next(it))
    pos = {}
    cluster_pts = {c: [] for c in range(K)}
    for i in range(1, N + 1):
        x = int(next(it)); y = int(next(it)); c = int(next(it))
        pos[i] = (x, y)
        cluster_pts[c].append(i)

    axes = {}
    for c in range(K):
        axes[c] = pca_axis(cluster_pts[c], pos)

    # portfolio-level, diversity-aware role assignment: rank clusters by elongation and
    # split into three balanced groups (most elongated -> PATH, most compact -> STAR,
    # the rest -> ZIGZAG) so all three grammars are actually used across the portfolio.
    order_by_elong = sorted(range(K), key=lambda c: axes[c][2], reverse=True)
    third = max(1, K // 3)
    roles = {}
    for rank, c in enumerate(order_by_elong):
        if rank < third:
            roles[c] = 0        # most elongated -> PATH
        elif rank >= K - third:
            roles[c] = 1        # most compact/round -> STAR
        else:
            roles[c] = 2        # in between -> ZIGZAG

    all_edges = []
    for c in range(K):
        pts = cluster_pts[c]
        r = roles[c]
        if r == 0:
            edges = build_path(pts, pos, axes[c])
        elif r == 1:
            edges = build_star(pts, pos)
        else:
            edges = build_zigzag(pts, pos, axes[c])
        all_edges.append(edges)

    out = [" ".join(str(roles[c]) for c in range(K))]
    for edges in all_edges:
        for (u, v) in edges:
            out.append(f"{u} {v}")
    print("\n".join(out))


if __name__ == "__main__":
    main()
