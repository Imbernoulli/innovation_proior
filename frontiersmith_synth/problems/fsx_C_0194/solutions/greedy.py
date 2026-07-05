# TIER: greedy
# Marginal-correlation skeleton + variance ordering. Build an undirected edge
# whenever two stations' readings are strongly correlated, then orient every
# such edge from the lower-variance station to the higher-variance one (a source
# disturbance has small variance; downstream stations accumulate variance).
# This is cheap but crude: marginal correlation also fires on INDIRECT
# (ancestor/descendant) pairs, so it adds many transitive edges -> higher SHD.
import sys, json, math

inst = json.load(sys.stdin)
n = inst["n"]
X = inst["samples"]
m = len(X)

# column means and (centered) variances
mean = [0.0] * n
for row in X:
    for k in range(n):
        mean[k] += row[k]
mean = [s / m for s in mean]
var = [0.0] * n
for row in X:
    for k in range(n):
        d = row[k] - mean[k]
        var[k] += d * d
var = [v / m for v in var]

# pairwise correlation
tau = 0.20
edges = []
for i in range(n):
    for j in range(i + 1, n):
        cov = 0.0
        for row in X:
            cov += (row[i] - mean[i]) * (row[j] - mean[j])
        cov /= m
        denom = math.sqrt(max(var[i], 1e-12) * max(var[j], 1e-12))
        corr = cov / denom
        if abs(corr) >= tau:
            # orient low-variance -> high-variance (ties by index)
            if (var[i], i) <= (var[j], j):
                edges.append([i, j])
            else:
                edges.append([j, i])

print(json.dumps({"edges": edges}))
