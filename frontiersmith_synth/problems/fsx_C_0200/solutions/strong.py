# TIER: strong
# A PC-style constraint routine specialized for the equal-shock-variance
# regime:
#   1. Skeleton by order-0 and order-1 (partial) correlation pruning: keep a
#      pair only if it stays correlated after conditioning on every single
#      other candidate neighbor -- kills chains/indirect edges.
#   2. Orientation by the equal-variance identifiability signal: under equal
#      shock variances a source has smaller marginal variance than its
#      descendants, so sorting columns by increasing empirical variance
#      recovers a valid topological order.  Orient each skeleton edge from
#      the lower-variance endpoint to the higher-variance endpoint.
# Recovers far more structure than marginal thresholding; still imperfect on
# dense graphs / higher-order dependencies, leaving headroom.
import sys, json, math

inst = json.load(sys.stdin)
d = inst["d"]; n = inst["n"]; X = inst["data"]

mean = [0.0] * d
for row in X:
    for j in range(d):
        mean[j] += row[j]
for j in range(d):
    mean[j] /= n
var = [0.0] * d
for row in X:
    for j in range(d):
        dv = row[j] - mean[j]
        var[j] += dv * dv
for j in range(d):
    var[j] /= n
std = [math.sqrt(v) or 1e-12 for v in var]

r = [[0.0] * d for _ in range(d)]
for a in range(d):
    r[a][a] = 1.0
for a in range(d):
    for b in range(a + 1, d):
        s = 0.0
        for row in X:
            s += (row[a] - mean[a]) * (row[b] - mean[b])
        rho = (s / n) / (std[a] * std[b])
        r[a][b] = r[b][a] = rho


def pcorr(a, b, k):
    rab, rak, rbk = r[a][b], r[a][k], r[b][k]
    den = math.sqrt(max(1e-12, (1 - rak * rak) * (1 - rbk * rbk)))
    return (rab - rak * rbk) / den


A0 = 0.16   # order-0 marginal keep threshold
A1 = 0.11   # order-1 conditional keep threshold

adj = [[False] * d for _ in range(d)]
for a in range(d):
    for b in range(a + 1, d):
        if abs(r[a][b]) > A0:
            adj[a][b] = adj[b][a] = True

# order-1 conditional independence pruning
for a in range(d):
    for b in range(a + 1, d):
        if not adj[a][b]:
            continue
        for k in range(d):
            if k == a or k == b:
                continue
            if not (adj[a][k] or adj[b][k]):
                continue
            if abs(pcorr(a, b, k)) < A1:
                adj[a][b] = adj[b][a] = False
                break

# orientation by increasing variance (equal-variance identifiability)
rank = sorted(range(d), key=lambda j: var[j])
pos = [0] * d
for p, j in enumerate(rank):
    pos[j] = p

edges = []
for a in range(d):
    for b in range(a + 1, d):
        if adj[a][b]:
            if pos[a] < pos[b]:
                edges.append([a, b])
            else:
                edges.append([b, a])
print(json.dumps({"edges": edges}))
