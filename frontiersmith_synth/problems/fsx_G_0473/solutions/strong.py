# TIER: strong
# Normalised spectral clustering on a k-nearest-neighbour affinity graph.
#
#   1. Build a symmetric mNN graph with Gaussian edge weights
#      A_ij = exp(-||xi-xj||^2 / (2 sigma^2))  (sigma = mean m-th-NN distance).
#   2. Symmetric normalise  S = D^-1/2 A D^-1/2.
#   3. Recover the top-k eigenvectors of S by orthogonal (subspace) iteration
#      with Gram-Schmidt re-orthonormalisation -> a spectral embedding that
#      "unrolls" non-convex manifolds (moons) and concentric shells (rings).
#   4. Row-normalise the embedding and k-means it (best of several seeded inits
#      by inertia -- a purely unsupervised model-selection criterion).
#
# Because it clusters along graph connectivity rather than Euclidean centroids,
# it recovers blobs, interleaving moons AND nested rings -- dominating both the
# structure-blind threshold and raw k-means on aggregate.  It still mislabels
# boundary customers on the noisiest instances, leaving headroom below 1.0.
# Deterministic: every RNG is seeded from the instance size.
import sys, json, math


def _rng(seed):
    state = (seed * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

    def nxt(lo, hi):
        nonlocal state
        state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return lo + (state >> 17) % (hi - lo + 1)

    return nxt


def _uni(ni):
    return ni(1, 1_000_000) / 1_000_001.0


def _gauss(ni):
    return math.sqrt(-2.0 * math.log(_uni(ni))) * math.cos(2.0 * math.pi * _uni(ni))


def _kmeans(data, k, seed, iters=40):
    ni = _rng(seed)
    n = len(data)
    dim = len(data[0])
    centers = [list(data[ni(0, n - 1)])]
    for _ in range(k - 1):
        d = [min(sum((p[t] - c[t]) ** 2 for t in range(dim)) for c in centers) for p in data]
        s = sum(d)
        thresh = _uni(ni) * s
        acc = 0.0
        idx = 0
        for i, dd in enumerate(d):
            acc += dd
            if acc >= thresh:
                idx = i
                break
        centers.append(list(data[idx]))
    lab = [0] * n
    for _ in range(iters):
        new = [min(range(k), key=lambda j: sum((data[i][t] - centers[j][t]) ** 2
                                               for t in range(dim)))
               for i in range(n)]
        if new == lab:
            break
        lab = new
        for j in range(k):
            xs = [data[i] for i in range(n) if lab[i] == j]
            if xs:
                centers[j] = [sum(p[t] for p in xs) / len(xs) for t in range(dim)]
    inertia = 0.0
    for i in range(n):
        c = centers[lab[i]]
        inertia += sum((data[i][t] - c[t]) ** 2 for t in range(dim))
    return lab, inertia


def _kmeans_best(data, k, seed, restarts=8):
    best = None
    best_in = None
    for rr in range(restarts):
        lab, inertia = _kmeans(data, k, seed + rr * 101)
        if best_in is None or inertia < best_in:
            best_in = inertia
            best = lab
    return best


inst = json.load(sys.stdin)
pts = inst["points"]
k = inst["k"]
n = len(pts)
seed0 = 1000003 * n + 7 * k + 1

if k <= 1 or n <= k:
    print(json.dumps({"labels": [0] * n}))
    sys.exit(0)

m = min(8, n - 1)

# pairwise squared distances
D2 = [[0.0] * n for _ in range(n)]
for i in range(n):
    pi = pts[i]
    for j in range(i + 1, n):
        pj = pts[j]
        d = (pi[0] - pj[0]) ** 2 + (pi[1] - pj[1]) ** 2
        D2[i][j] = d
        D2[j][i] = d

knn = [sorted(range(n), key=lambda j: (D2[i][j], j))[1:m + 1] for i in range(n)]
sig = sum(math.sqrt(D2[i][knn[i][-1]]) for i in range(n)) / n
sig2 = 2.0 * sig * sig if sig > 0 else 1.0

A = [[0.0] * n for _ in range(n)]
for i in range(n):
    for j in knn[i]:
        w = math.exp(-D2[i][j] / sig2)
        A[i][j] = w
        A[j][i] = w

deg = [sum(A[i]) or 1e-12 for i in range(n)]
dinv = [1.0 / math.sqrt(deg[i]) for i in range(n)]
S = [[A[i][j] * dinv[i] * dinv[j] for j in range(n)] for i in range(n)]

# top-k eigenvectors via orthogonal iteration
ni = _rng(seed0 + 3)
Q = [[_gauss(ni) for _ in range(k)] for _ in range(n)]


def _gs(Q):
    for c in range(k):
        for c2 in range(c):
            dot = sum(Q[i][c] * Q[i][c2] for i in range(n))
            for i in range(n):
                Q[i][c] -= dot * Q[i][c2]
        nrm = math.sqrt(sum(Q[i][c] ** 2 for i in range(n))) or 1e-12
        for i in range(n):
            Q[i][c] /= nrm
    return Q


Q = _gs(Q)
for _ in range(120):
    Y = [[0.0] * k for _ in range(n)]
    for i in range(n):
        Si = S[i]
        for c in range(k):
            Y[i][c] = sum(Si[j] * Q[j][c] for j in range(n))
    Q = _gs(Y)

emb = []
for i in range(n):
    row = Q[i][:]
    nr = math.sqrt(sum(v * v for v in row)) or 1e-12
    emb.append([v / nr for v in row])

labels = _kmeans_best(emb, k, seed0 + 11)
print(json.dumps({"labels": labels}))
