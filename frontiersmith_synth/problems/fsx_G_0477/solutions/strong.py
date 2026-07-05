# TIER: strong
# Rank ensemble of a GLOBAL multivariate model (Mahalanobis distance under the full
# sample covariance, i.e. whitening) and a LOCAL density model (distance to the k-th
# nearest neighbour in standardized space). Mahalanobis catches global + correlation-
# break + contextual faults (off the correlation manifold); kNN catches the density /
# valley faults that a single-Gaussian model misses (they sit near the global mean).
# Combining them by rank-averaging is robust across the diverse regimes.
import sys, json, math

inst = json.load(sys.stdin)
X = inst["X"]; n = inst["n"]; d = inst["d"]

# ---- standardize columns ----
mean = [0.0] * d
for row in X:
    for j in range(d):
        mean[j] += row[j]
for j in range(d):
    mean[j] /= n
std = [0.0] * d
for row in X:
    for j in range(d):
        dv = row[j] - mean[j]
        std[j] += dv * dv
for j in range(d):
    std[j] = math.sqrt(std[j] / max(1, n - 1)) or 1e-9
Z = [[(row[j] - mean[j]) / std[j] for j in range(d)] for row in X]

# ---- covariance of standardized data (= correlation matrix), regularized ----
C = [[0.0] * d for _ in range(d)]
for row in Z:
    for a in range(d):
        ra = row[a]
        for b in range(a, d):
            C[a][b] += ra * row[b]
for a in range(d):
    for b in range(a, d):
        C[a][b] /= max(1, n - 1)
        C[b][a] = C[a][b]
for a in range(d):
    C[a][a] += 1e-3   # ridge for invertibility


def inv(M):
    m = len(M)
    A = [M[i][:] + [1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) < 1e-12:
            A[col][col] += 1e-6
            piv = col
        A[col], A[piv] = A[piv], A[col]
        pv = A[col][col]
        inv_pv = 1.0 / pv
        for k in range(2 * m):
            A[col][k] *= inv_pv
        for r in range(m):
            if r != col:
                f = A[r][col]
                if f != 0.0:
                    for k in range(2 * m):
                        A[r][k] -= f * A[col][k]
    return [row[m:] for row in A]


Cinv = inv(C)

maha = []
for row in Z:
    Cx = [sum(Cinv[a][b] * row[b] for b in range(d)) for a in range(d)]
    maha.append(sum(row[a] * Cx[a] for a in range(d)))

# ---- kNN: distance to k-th nearest neighbour in standardized space ----
k = min(15, max(5, n // 30))
knn = []
for i in range(n):
    zi = Z[i]
    dmin = []  # keep k smallest squared distances
    for jx in range(n):
        if jx == i:
            continue
        zj = Z[jx]
        dd = 0.0
        for a in range(d):
            t = zi[a] - zj[a]
            dd += t * t
        if len(dmin) < k:
            dmin.append(dd)
            if len(dmin) == k:
                dmin.sort()
        elif dd < dmin[-1]:
            # insert into sorted list
            lo, hi = 0, k - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if dmin[mid] < dd:
                    lo = mid + 1
                else:
                    hi = mid
            dmin.insert(lo, dd)
            dmin.pop()
    knn.append(math.sqrt(dmin[-1]) if dmin else 0.0)


def to_ranks(v):
    n = len(v)
    order = sorted(range(n), key=lambda i: v[i])
    r = [0.0] * n
    for rank, idx in enumerate(order):
        r[idx] = rank / max(1, n - 1)
    return r


rm = to_ranks(maha)
rk = to_ranks(knn)
# OR-of-views: a snapshot is anomalous if it is extreme under EITHER the global
# (Mahalanobis) or the local-density (kNN) model. Max-combine is robust to the density
# regime, where the valley fault is deliberately central under the global model.
scores = [max(rm[i], rk[i]) + 0.05 * (rm[i] + rk[i]) for i in range(n)]
print(json.dumps({"scores": scores}))
