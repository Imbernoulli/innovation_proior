#!/usr/bin/env python3
"""
FROZEN evaluator for fsx_C_0391 -- "SwarmSort: Drone Delivery Zone Discovery"
(family: classical-ml-algorithm; format B, quality-metric).

THEME.  A city-wide drone delivery swarm drops parcels all over a metro map.  At
the end of each shift the operations centre has a cloud of 2-D drop coordinates
(where every parcel was actually delivered) but the ORIGINAL routing zone that
each drop belonged to has been lost.  To rebalance depots for tomorrow the centre
must re-DISCOVER the delivery zones purely from the geometry of the drop cloud:
partition the drops into exactly K zones so that drops that truly belonged to the
same routing zone end up together.  Only the operator (this evaluator) knows the
true zone each drop came from -- that ground truth is HIDDEN.  The model sees only
the anonymised coordinates + the target zone count K, and must output a labelling.

This is unsupervised CLUSTERING skinned as a logistics problem: points = parcel
drops, target partition = K routing zones, quality = how well the recovered
partition matches the hidden true zones, measured by the Adjusted Rand Index (ARI)
-- a permutation-invariant, chance-corrected agreement score.

CANDIDATE CONTRACT (isolated stdin -> stdout program).
  stdin : ONE JSON object (the PUBLIC instance):
            {"name": str, "n": int, "dim": 2, "k": int,
             "points": [[x0,y0], [x1,y1], ..., [x_{n-1},y_{n-1}]]}   # floats
  stdout: ONE JSON object:
            {"labels": [c_0, c_1, ..., c_{n-1}]}   # integer zone id per drop
          The label VALUES are arbitrary integers (ARI is invariant to relabelling
          and does not require exactly K distinct ids), but there must be exactly n
          of them and each must be a plain finite integer.  Wrong length, a
          non-integer / boolean / non-finite entry, a crash, a timeout, or non-JSON
          -> that instance scores 0.0.

SCORING (deterministic; no wall-time).  Per instance the evaluator computes the
ARI of the candidate labelling against the HIDDEN true zones (ari_cand), plus a
weak internal reference partition -- an axis (x-coordinate) equal-count split into
K contiguous bands (ari_base).  A perfect recovery has ARI 1.0.  We anchor the
weak axis-split at 0.1 and the (generally unreachable) perfect recovery at 1.0:
    r = clamp( 0.1 + 0.9 * (ari_cand - ari_base) / max(1e-9, 1.0 - ari_base), 0, 1 )
  A candidate reproducing the axis-split scores ~0.1; matching the true zones
  scores 1.0; doing worse than the axis-split scores < 0.1.  Non-convex zones
  (interleaving crescents, concentric rings) cannot be recovered by centroidal
  methods, so even strong K-means-style clusterers stay well below 1.0 there ->
  headroom.

AGGREGATION.  The reported Ratio is the GEOMETRIC MEAN of the per-instance r
values (the Vector).  gmean rewards methods that transfer across the whole zoo of
zone shapes and severely penalises a clusterer that overfits one dataset family
while collapsing on another -- a single near-zero instance drags the whole score
down.

ISOLATION.  The candidate is untrusted and runs in a FRESH OS-SANDBOXED SUBPROCESS
via `isorun.run_candidate`; it only ever sees the PUBLIC instance.  The hidden true
zones and all references are computed by THIS parent process, so a frame-walking /
introspecting / source-reading candidate learns nothing useful.

CLI:  python3 evaluator.py <solution.py>
Prints:
  Ratio: <geometric mean of r over all instances, in [0,1]>
  Vector: [r_1, r_2, ...]
"""
import sys, json, math
import isorun


# ----------------------------- deterministic RNG ---------------------------
class LCG:
    """Pure-python 64-bit LCG -> deterministic floats/gaussians (no numpy)."""
    __slots__ = ("s",)

    def __init__(self, seed):
        self.s = (seed * 2862933555777941757 + 3037000493) & ((1 << 64) - 1)

    def _next(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return self.s

    def uniform(self):
        # 53-bit mantissa uniform in [0,1)
        return (self._next() >> 11) * (1.0 / (1 << 53))

    def gauss(self):
        # Box-Muller; guard against log(0)
        u1 = self.uniform()
        if u1 < 1e-12:
            u1 = 1e-12
        u2 = self.uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def randint(self, lo, hi):  # inclusive
        return lo + (self._next() >> 17) % (hi - lo + 1)


# ----------------------------- instance family -----------------------------
def _blobs(rng, n, centers, spreads):
    """n points split as evenly as possible across len(centers) isotropic blobs."""
    k = len(centers)
    pts, lab = [], []
    for i in range(n):
        c = i % k
        cx, cy = centers[c]
        sx, sy = spreads[c]
        pts.append([cx + sx * rng.gauss(), cy + sy * rng.gauss()])
        lab.append(c)
    return pts, lab


def _aniso(rng, n, centers, spreads, shear):
    pts, lab = _blobs(rng, n, centers, spreads)
    out = []
    for (x, y) in pts:
        out.append([x + shear * y, y])
    return out, lab


def _moons(rng, n, noise):
    pts, lab = [], []
    for i in range(n):
        c = i % 2
        t = math.pi * (i // 2) / max(1, (n // 2) - 1)
        if c == 0:
            x, y = math.cos(t), math.sin(t)
        else:
            x, y = 1.0 - math.cos(t), 0.5 - math.sin(t)
        pts.append([x + noise * rng.gauss(), y + noise * rng.gauss()])
        lab.append(c)
    return pts, lab


def _rings(rng, n, radii, noise):
    """Concentric rings sharing a centre -> not linearly / centroidally separable."""
    k = len(radii)
    pts, lab = [], []
    for i in range(n):
        c = i % k
        ang = 2.0 * math.pi * rng.uniform()
        r = radii[c] + noise * rng.gauss()
        pts.append([r * math.cos(ang), r * math.sin(ang)])
        lab.append(c)
    return pts, lab


def _grid_centers(cols, rows, gap):
    return [[cx * gap, cy * gap] for cy in range(rows) for cx in range(cols)]


def _build_instances():
    """Deterministic instance zoo.  Centre layouts are chosen so a single
    x-threshold split is genuinely weak (many zones share an x-band)."""
    insts = []

    def add(seed, name, k, pts, lab):
        insts.append({"name": name, "n": len(pts), "dim": 2, "k": k,
                      "points": pts, "labels": lab})

    # 1. 3x3 grid of tight blobs (9 zones) -- easy shape, but x-split fails
    r = LCG(1001)
    C = _grid_centers(3, 3, 6.0)
    S = [[0.7, 0.7]] * 9
    p, l = _blobs(r, 270, C, S)
    add(1001, "grid3x3", 9, p, l)

    # 2. 5 depot blobs arranged on a circle
    r = LCG(1002)
    C = [[8.0 * math.cos(2 * math.pi * j / 5), 8.0 * math.sin(2 * math.pi * j / 5)] for j in range(5)]
    S = [[1.0, 1.0]] * 5
    p, l = _blobs(r, 250, C, S)
    add(1002, "depots5", 5, p, l)

    # 3. 4 blobs with strongly VARIED spreads (k-means one-size fails)
    r = LCG(1003)
    C = [[0, 0], [10, 0], [0, 10], [10, 10]]
    S = [[0.5, 0.5], [2.6, 2.6], [1.4, 1.4], [0.8, 0.8]]
    p, l = _blobs(r, 240, C, S)
    add(1003, "varied4", 4, p, l)

    # 4. 3 anisotropic (sheared) blobs stacked vertically
    r = LCG(1004)
    C = [[0, 0], [0, 7], [0, 14]]
    S = [[2.2, 0.5], [2.2, 0.5], [2.2, 0.5]]
    p, l = _aniso(r, 240, C, S, 1.3)
    add(1004, "aniso3", 3, p, l)

    # 5. two interleaving crescents (non-convex; centroidal methods struggle)
    r = LCG(1005)
    p, l = _moons(r, 240, 0.09)
    add(1005, "moons", 2, p, l)

    # 6. two concentric rings (non-convex; centroidal methods fail hard)
    r = LCG(1006)
    p, l = _rings(r, 240, [2.0, 5.0], 0.22)
    add(1006, "rings", 2, p, l)

    # 7. 4 overlapping blobs on a small grid
    r = LCG(1007)
    C = _grid_centers(2, 2, 3.4)
    S = [[1.5, 1.5]] * 4
    p, l = _blobs(r, 260, C, S)
    add(1007, "overlap4", 4, p, l)

    # ---- larger / held-out ----
    # 8. 2x4 grid of blobs (8 zones), tighter, bigger
    r = LCG(1008)
    C = _grid_centers(4, 2, 5.0)
    S = [[0.85, 0.85]] * 8
    p, l = _blobs(r, 320, C, S)
    add(1008, "grid2x4", 8, p, l)

    # 9. 6 varied blobs on a circle, larger n
    r = LCG(1009)
    C = [[9.0 * math.cos(2 * math.pi * j / 6), 9.0 * math.sin(2 * math.pi * j / 6)] for j in range(6)]
    S = [[0.6, 0.6], [1.8, 1.8], [1.0, 1.0], [2.2, 2.2], [0.9, 0.9], [1.5, 1.5]]
    p, l = _blobs(r, 360, C, S)
    add(1009, "varied6", 6, p, l)

    # 10. 5 anisotropic sheared blobs, larger n, held-out
    r = LCG(1010)
    C = [[0, 0], [6, 0], [12, 0], [3, 6], [9, 6]]
    S = [[1.9, 0.55]] * 5
    p, l = _aniso(r, 350, C, S, 1.1)
    add(1010, "aniso5", 5, p, l)

    return insts


# ----------------------------- metric: ARI ---------------------------------
def _ari(a, b):
    """Adjusted Rand Index between two integer labellings a, b (same length)."""
    n = len(a)
    if n == 0:
        return 1.0
    from collections import defaultdict
    cont = defaultdict(int)
    ra = defaultdict(int)
    cb = defaultdict(int)
    for x, y in zip(a, b):
        cont[(x, y)] += 1
        ra[x] += 1
        cb[y] += 1

    def comb2(v):
        return v * (v - 1) // 2

    sum_ij = sum(comb2(v) for v in cont.values())
    sum_a = sum(comb2(v) for v in ra.values())
    sum_b = sum(comb2(v) for v in cb.values())
    tot = comb2(n)
    if tot == 0:
        return 1.0
    expected = (sum_a * sum_b) / tot
    max_index = (sum_a + sum_b) / 2.0
    denom = max_index - expected
    if abs(denom) < 1e-12:
        # both partitions trivial (all-one or all-singleton) -> agree by definition
        return 1.0 if sum_a == sum_b == sum_ij else 0.0
    return (sum_ij - expected) / denom


# ----------------------------- weak reference ------------------------------
def _axis_split_labels(points, k):
    """Deterministic weak baseline: sort by x (tie-break index), cut into k
    contiguous equal-count bands.  Reproduced exactly by the 'trivial' tier."""
    n = len(points)
    order = sorted(range(n), key=lambda i: (points[i][0], i))
    labels = [0] * n
    base = n // k
    extra = n % k
    idx = 0
    for band in range(k):
        cnt = base + (1 if band < extra else 0)
        for _ in range(cnt):
            labels[order[idx]] = band
            idx += 1
    return labels


# ----------------------------- validation ----------------------------------
def _validate_labels(inst, answer):
    if not isinstance(answer, dict):
        return None
    labels = answer.get("labels")
    if not isinstance(labels, list):
        return None
    n = inst["n"]
    if len(labels) != n:
        return None
    out = []
    for v in labels:
        if isinstance(v, bool) or not isinstance(v, int):
            return None
        out.append(v)
    return out


# ----------------------------- scoring driver ------------------------------
def main():
    if len(sys.argv) < 2:
        print("usage: evaluator.py <solution.py>")
        sys.exit(2)
    cand = sys.argv[1]
    instances = _build_instances()

    vec = []
    for inst in instances:
        true_lab = inst["labels"]
        k = inst["k"]
        base_lab = _axis_split_labels(inst["points"], k)
        ari_base = _ari(base_lab, true_lab)
        denom = 1.0 - ari_base
        if denom < 1e-9:
            denom = 1e-9

        public = {"name": inst["name"], "n": inst["n"], "dim": inst["dim"],
                  "k": k, "points": [list(p) for p in inst["points"]]}
        ans, st = isorun.run_candidate(cand, public, timeout=20)
        if st != "OK":
            vec.append(0.0)
            continue
        try:
            cand_lab = _validate_labels(inst, ans)
        except Exception:
            cand_lab = None
        if cand_lab is None:
            vec.append(0.0)
            continue
        try:
            ari_cand = _ari(cand_lab, true_lab)
        except Exception:
            vec.append(0.0)
            continue
        r = 0.1 + 0.9 * (ari_cand - ari_base) / denom
        if not (r == r) or r in (float("inf"), float("-inf")):
            vec.append(0.0)
            continue
        if r < 0.0:
            r = 0.0
        elif r > 1.0:
            r = 1.0
        vec.append(r)

    # geometric mean (rewards cross-dataset transfer; one near-zero drags it down)
    if not vec:
        gm = 0.0
    else:
        acc = 0.0
        for x in vec:
            acc += math.log(x) if x > 1e-12 else math.log(1e-12)
        gm = math.exp(acc / len(vec))
    if gm < 0.0:
        gm = 0.0
    elif gm > 1.0:
        gm = 1.0
    print("Ratio: %.6f" % gm)
    print("Vector: " + json.dumps([round(x, 6) for x in vec]))


if __name__ == "__main__":
    main()
