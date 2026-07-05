# TIER: greedy
# Lloyd's k-means (k-means++ seeded) on the raw 2-D feature space.  Centroid
# clustering nails compact BLOB spend-tiers, but it partitions space into convex
# Voronoi cells, so it shreds the non-convex MOON arcs and the concentric RING
# shells -- it beats the structure-blind coordinate threshold on blobs yet stays
# mediocre on the manifold-shaped segments.  Deterministic: the init RNG is
# seeded from the instance size, and label ids are irrelevant to ARI.
import sys, json


def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


def _uni(ni):
    return ni(1, 1_000_000) / 1_000_001.0


inst = json.load(sys.stdin)
pts = inst["points"]
k = inst["k"]
n = len(pts)

ni = _rng(1000003 * n + 7 * k + 1)

# k-means++ initialisation
centers = [list(pts[ni(0, n - 1)])]
for _ in range(k - 1):
    d = [min((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 for c in centers) for p in pts]
    s = sum(d)
    thresh = _uni(ni) * s
    acc = 0.0
    idx = 0
    for i, dd in enumerate(d):
        acc += dd
        if acc >= thresh:
            idx = i
            break
    centers.append(list(pts[idx]))

labels = [0] * n
for _ in range(60):
    new = [min(range(k), key=lambda j: (pts[i][0] - centers[j][0]) ** 2
                                       + (pts[i][1] - centers[j][1]) ** 2)
           for i in range(n)]
    if new == labels:
        break
    labels = new
    for j in range(k):
        xs = [pts[i] for i in range(n) if labels[i] == j]
        if xs:
            centers[j] = [sum(p[0] for p in xs) / len(xs),
                          sum(p[1] for p in xs) / len(xs)]

print(json.dumps({"labels": labels}))
