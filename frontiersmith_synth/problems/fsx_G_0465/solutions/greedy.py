# TIER: greedy
# k-nearest-neighbour patient imputation. Standardize each column by its observed
# mean/std, then for a masked cell (i,j) find the K other patients that DO have
# column j observed and are closest to patient i over their commonly-observed
# columns (mean squared standardized difference), and predict the (distance-weighted)
# average of their column-j values. Falls back to the column mean when no neighbour
# shares an observed column. Exploits row similarity but not the full multivariate
# low-rank structure.
import sys, json, math

inst = json.load(sys.stdin)
N, D = inst["N"], inst["D"]
M = inst["matrix"]
mask = inst["masked"]
K = 8

cmean = [0.0] * D
csd = [1.0] * D
for j in range(D):
    vals = [M[i][j] for i in range(N) if M[i][j] is not None]
    if vals:
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        cmean[j] = m
        csd[j] = math.sqrt(var) if var > 1e-12 else 1.0

# standardized rows with observed mask
Z = [[None] * D for _ in range(N)]
for i in range(N):
    for j in range(D):
        if M[i][j] is not None:
            Z[i][j] = (M[i][j] - cmean[j]) / csd[j]

preds = []
for (i, j) in mask:
    cand = []
    for r in range(N):
        if r == i or Z[r][j] is None:
            continue
        ssd = 0.0
        cnt = 0
        for c in range(D):
            if c == j:
                continue
            if Z[i][c] is not None and Z[r][c] is not None:
                d = Z[i][c] - Z[r][c]
                ssd += d * d
                cnt += 1
        if cnt == 0:
            continue
        dist = math.sqrt(ssd / cnt)
        cand.append((dist, M[r][j]))
    if not cand:
        preds.append(cmean[j])
        continue
    cand.sort(key=lambda t: t[0])
    nn = cand[:K]
    wsum = 0.0
    vsum = 0.0
    for dist, val in nn:
        w = 1.0 / (1e-6 + dist)
        wsum += w
        vsum += w * val
    preds.append(vsum / wsum if wsum > 0 else cmean[j])

print(json.dumps({"preds": preds}))
