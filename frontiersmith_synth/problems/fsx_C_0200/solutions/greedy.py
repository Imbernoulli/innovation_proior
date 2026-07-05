# TIER: greedy
# Marginal-correlation thresholding: connect any pair whose |Pearson r|
# exceeds a fixed threshold, and orient every kept edge by COLUMN INDEX
# (lower index -> higher index).  No conditioning, so transitive/indirect
# correlations become false edges; and because the public column order is a
# hidden permutation of the causal order, index-orientation is wrong about
# half the time.  A cheap-but-flawed influence map.
import sys, json, math

inst = json.load(sys.stdin)
d = inst["d"]; n = inst["n"]; X = inst["data"]

# column means / standard deviations
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
    std[j] = math.sqrt(std[j] / n) or 1e-12

# correlation matrix
r = [[0.0] * d for _ in range(d)]
for a in range(d):
    for b in range(a + 1, d):
        s = 0.0
        for row in X:
            s += (row[a] - mean[a]) * (row[b] - mean[b])
        rho = (s / n) / (std[a] * std[b])
        r[a][b] = r[b][a] = rho

THR = 0.20
edges = []
for a in range(d):
    for b in range(a + 1, d):
        if abs(r[a][b]) > THR:
            edges.append([a, b])          # orient by index a<b
print(json.dumps({"edges": edges}))
