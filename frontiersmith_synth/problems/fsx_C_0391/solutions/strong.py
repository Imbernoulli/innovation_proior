# TIER: strong
# Z-score standardised, K-means++ seeded, multi-restart Lloyd K-means keeping the
# lowest-inertia partition.  Standardisation removes per-axis scale so anisotropic
# / varied-spread zones are recovered; K-means++ + several seeded restarts escape
# the bad local optima that trip up a single plain pass.  Still centroidal, so it
# CANNOT recover truly non-convex zones (interleaving crescents, concentric rings)
# -> the normalised score stays well below 1.0 on those instances.
import sys, json, math

inst = json.load(sys.stdin)
pts = inst["points"]
k = inst["k"]
n = len(pts)
dim = len(pts[0])


class LCG:
    def __init__(self, seed):
        self.s = (seed * 2862933555777941757 + 3037000493) & ((1 << 64) - 1)

    def _n(self):
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return self.s

    def u(self):
        return (self._n() >> 11) * (1.0 / (1 << 53))


# ---- standardise per axis ----
mean = [0.0] * dim
for p in pts:
    for d in range(dim):
        mean[d] += p[d]
mean = [m / n for m in mean]
var = [0.0] * dim
for p in pts:
    for d in range(dim):
        t = p[d] - mean[d]
        var[d] += t * t
std = [math.sqrt(v / n) if v > 0 else 1.0 for v in var]
X = [[(p[d] - mean[d]) / std[d] for d in range(dim)] for p in pts]


def dist2(a, b):
    d = 0.0
    for i in range(dim):
        t = a[i] - b[i]
        d += t * t
    return d


def kpp_init(rng):
    c0 = int(rng.u() * n) % n
    cents = [list(X[c0])]
    dmin = [dist2(X[i], cents[0]) for i in range(n)]
    for _ in range(1, k):
        tot = sum(dmin)
        if tot <= 0:
            cents.append(list(X[int(rng.u() * n) % n]))
        else:
            target = rng.u() * tot
            acc, pick = 0.0, n - 1
            for i in range(n):
                acc += dmin[i]
                if acc >= target:
                    pick = i
                    break
            cents.append(list(X[pick]))
        for i in range(n):
            d = dist2(X[i], cents[-1])
            if d < dmin[i]:
                dmin[i] = d
    return cents


def lloyd(cents):
    labels = [0] * n
    for _ in range(40):
        changed = False
        for i in range(n):
            best, bd = 0, None
            for c in range(k):
                d = dist2(X[i], cents[c])
                if bd is None or d < bd:
                    bd, best = d, c
            if labels[i] != best:
                changed = True
            labels[i] = best
        sums = [[0.0] * dim for _ in range(k)]
        cnts = [0] * k
        for i in range(n):
            c = labels[i]
            cnts[c] += 1
            for d in range(dim):
                sums[c][d] += X[i][d]
        for c in range(k):
            if cnts[c] > 0:
                cents[c] = [sums[c][d] / cnts[c] for d in range(dim)]
        if not changed:
            break
    inertia = 0.0
    for i in range(n):
        inertia += dist2(X[i], cents[labels[i]])
    return labels, inertia


best_labels, best_inertia = None, None
for seed in range(8):
    rng = LCG(9973 * (seed + 1) + 7)
    labels, inertia = lloyd(kpp_init(rng))
    if best_inertia is None or inertia < best_inertia:
        best_inertia, best_labels = inertia, labels

print(json.dumps({"labels": best_labels}))
