#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_G_0473 -- "Segment Atlas: an Unsupervised Customer
Segmentation Assigner" (family: ml-clustering-algo; format B, quality-metric).

THEME.  A retail analytics team must SEGMENT customers for a marketing campaign.
Each customer is a point in a 2-D behavioural feature space (e.g. a UMAP-style
embedding of spend recency/frequency, price sensitivity, browsing style, ...).
The team already knows -- from the campaign brief -- HOW MANY segments `k` they
must produce, but NOT which customer belongs to which segment.  The catch is that
real segment geometry is NOT always a tidy ball of points: some campaigns split
customers into compact spend tiers (BLOB-like clusters), some into two
interleaving lifestyle arcs (MOON-like, non-convex), and some into nested
loyalty shells around a core (RING/CIRCLE-like, concentric).  A centroid method
(k-means) nails the blobs but shreds the moons and rings; a good segmenter must
respect connectivity/manifold structure, not just Euclidean proximity.

The candidate is an UNSUPERVISED segmenter: it sees only the point cloud and the
target segment count `k`, and must output a segment label for every customer.
Quality is measured against the hidden "ground-truth" segmentation with the
ADJUSTED RAND INDEX (ARI) -- a permutation-invariant agreement score, so the
candidate never needs to guess which integer names the graders used.

This is a BREADTH task: an authentic unsupervised-clustering scorer over a
distribution of blob / moon / ring segmentation instances.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n": N, "k": K, "points": [[x0,y0], ..., [x_{N-1},y_{N-1}]]}
  stdout: ONE JSON object:
            {"labels": [l_0, ..., l_{N-1}]}      # segment id of each customer

  A segmentation is VALID iff `labels` is a list of exactly N integers, each in
  [0, N).  (It need NOT use exactly k distinct labels -- ARI tolerates over/under
  segmentation; but a degenerate one-segment or all-singleton answer earns ARI 0.)
  Wrong length, a non-integer / boolean / out-of-range label, a crash, a timeout,
  or non-JSON -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance the evaluator holds the
hidden true segmentation `truth`.  It computes:
    q_cand = ARI(truth, candidate labels)            # in [-0.5, 1], 1 = perfect
    q_base = ARI(truth, coordinate-threshold split)  # the evaluator's own weak
             reference: sort customers by their first feature and cut into k equal
             ranked bins -- a segmentation blind to 2-D / manifold structure.
  normalized with an affine anchor (weak reference -> 0.1, perfect ARI -> 1.0):
    r = clamp( 0.1 + 0.9 * (q_cand - q_base) / max(1e-9, 1.0 - q_base), 0, 1 )
  Matching the coordinate-threshold reference scores ~0.1; a perfect segmentation
  scores 1.0; doing worse than the reference scores < 0.1.  The final Ratio is the
  mean of r over all instances.

  Because several instances are noisy / non-convex (interleaving moons, nested
  rings) where even a connectivity-aware segmenter mislabels boundary customers,
  a strong candidate stays well below 1.0 -> genuine headroom.

ISOLATION.  The candidate is untrusted and runs in a FRESH SUBPROCESS via
`isorun.run_candidate`; it only ever sees the PUBLIC instance (points + k).  The
hidden ground truth, the reference segmentation and the ARI scoring are computed
by THIS parent process, so a frame-walking / introspecting candidate learns
nothing about the answer.

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


def _uni(ni):
    return ni(1, 1_000_000) / 1_000_001.0


def _gauss(ni):
    u1 = _uni(ni)
    u2 = _uni(ni)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


# ----------------------------- instance family -----------------------------
def _gen_blobs(seed, n, k, R, std):
    """k Gaussian spend-tier clusters.  Centres are placed on an ellipse that is
    ELONGATED along the second feature so the clusters overlap when projected onto
    the first feature -- a coordinate-threshold split cannot separate them, but a
    centroid or connectivity method can."""
    ni = _rng(seed)
    centers = []
    for j in range(k):
        a = 2.0 * math.pi * j / k + 0.3
        centers.append((0.3 * R * math.cos(a), R * math.sin(a)))
    pts = []
    lab = []
    for i in range(n):
        j = i % k
        cx, cy = centers[j]
        pts.append([cx + std * _gauss(ni), cy + std * _gauss(ni)])
        lab.append(j)
    return pts, lab


def _gen_moons(seed, n, noise):
    """Two interleaving half-moon lifestyle arcs (non-convex, k=2)."""
    ni = _rng(seed)
    pts = []
    lab = []
    for i in range(n):
        j = i % 2
        t = math.pi * _uni(ni)
        if j == 0:
            x = math.cos(t)
            y = math.sin(t)
        else:
            x = 1.0 - math.cos(t)
            y = 0.5 - math.sin(t)
        pts.append([x + noise * _gauss(ni), y + noise * _gauss(ni)])
        lab.append(j)
    return pts, lab


def _gen_rings(seed, n, radii, noise):
    """Concentric loyalty shells: len(radii) nested rings around a common core."""
    ni = _rng(seed)
    k = len(radii)
    pts = []
    lab = []
    for i in range(n):
        j = i % k
        t = 2.0 * math.pi * _uni(ni)
        r = radii[j]
        pts.append([r * math.cos(t) + noise * _gauss(ni),
                    r * math.sin(t) + noise * _gauss(ni)])
        lab.append(j)
    return pts, lab


def _build_instances():
    """Deterministic distribution: 4 blob, 4 moon, 4 ring segmentation instances,
    with a couple of harder/noisier held-out cases per shape for generalization."""
    out = []
    specs = [
        ("blob", _gen_blobs, (4740, 200, 3, 4.0, 0.55)),
        ("blob", _gen_blobs, (4741, 200, 4, 4.0, 0.60)),
        ("blob", _gen_blobs, (4742, 180, 3, 3.6, 0.85)),
        ("blob", _gen_blobs, (4749, 220, 4, 3.8, 0.75)),
        ("moon", _gen_moons, (4743, 200, 0.09)),
        ("moon", _gen_moons, (4744, 200, 0.13)),
        ("moon", _gen_moons, (4745, 200, 0.16)),
        ("moon", _gen_moons, (4750, 220, 0.11)),
        ("ring", _gen_rings, (4746, 200, [0.45, 1.0], 0.05)),
        ("ring", _gen_rings, (4747, 210, [0.35, 0.7, 1.0], 0.045)),
        ("ring", _gen_rings, (4748, 200, [0.5, 1.0], 0.09)),
        ("ring", _gen_rings, (4751, 216, [0.33, 0.66, 1.0], 0.05)),
    ]
    for shape, fn, args in specs:
        pts, truth = fn(*args)
        k = len(set(truth))
        seed = args[0]
        out.append({
            "name": f"{shape}{seed}",
            "n": len(pts),
            "k": k,
            "points": pts,     # public
            "truth": truth,    # HIDDEN
        })
    return out


# ----------------------------- adjusted rand index -------------------------
def _c2(x):
    return x * (x - 1) // 2


def _ari(truth, pred):
    """Adjusted Rand Index between two label lists of equal length."""
    n = len(truth)
    cont = {}
    a = {}
    b = {}
    for t, p in zip(truth, pred):
        cont[(t, p)] = cont.get((t, p), 0) + 1
        a[t] = a.get(t, 0) + 1
        b[p] = b.get(p, 0) + 1
    sc = sum(_c2(v) for v in cont.values())
    sa = sum(_c2(v) for v in a.values())
    sb = sum(_c2(v) for v in b.values())
    cn = _c2(n)
    if cn == 0:
        return 0.0
    expected = sa * sb / cn
    max_index = 0.5 * (sa + sb)
    denom = max_index - expected
    if abs(denom) < 1e-12:
        # both partitions trivial/identical (e.g. all-one or all-singletons)
        return 1.0 if sc == sa == sb else 0.0
    return (sc - expected) / denom


# ----------------------------- reference segmentation ----------------------
def _reference_labels(inst):
    """Weak baseline: sort customers by their FIRST feature and cut into k equal
    ranked bins.  Structure-blind (ignores the second feature entirely and any
    manifold shape), so it is a genuinely weak segmentation to anchor r=0.1."""
    pts = inst["points"]
    k = inst["k"]
    n = len(pts)
    order = sorted(range(n), key=lambda i: (pts[i][0], i))
    lab = [0] * n
    per = n / k
    for rank, i in enumerate(order):
        lab[i] = min(k - 1, int(rank / per))
    return lab


# ----------------------------- answer validation ---------------------------
def _valid_labels(inst, answer):
    """Return the label list if the answer is structurally valid, else None."""
    if not isinstance(answer, dict):
        return None
    labels = answer.get("labels")
    if not isinstance(labels, list):
        return None
    n = inst["n"]
    if len(labels) != n:
        return None
    for v in labels:
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        if v < 0 or v >= n:
            return None
    return labels


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        truth = inst["truth"]
        q_base = _ari(truth, _reference_labels(inst))
        denom = 1.0 - q_base
        if denom < 1e-9:
            denom = 1e-9
        public = {
            "name": inst["name"],
            "n": inst["n"],
            "k": inst["k"],
            "points": [list(p) for p in inst["points"]],
        }
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        labels = _valid_labels(inst, ans)
        if labels is None:
            vec.append(0.0)
            continue
        try:
            q_cand = _ari(truth, labels)
        except Exception:
            vec.append(0.0)
            continue
        if not (q_cand == q_cand) or q_cand in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (q_cand - q_base) / denom
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
