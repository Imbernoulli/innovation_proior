# TIER: greedy
# A single plain Lloyd K-means pass with a cheap deterministic init: pick K
# initial centroids evenly spaced through the points sorted by x.  No restarts, no
# feature scaling.  Recovers well-separated round zones but is sensitive to the
# init and to varied / anisotropic spreads, and cannot bend around non-convex
# zones (crescents, rings).
import sys, json

inst = json.load(sys.stdin)
pts = inst["points"]
k = inst["k"]
n = len(pts)


def dist2(a, b):
    d = 0.0
    for i in range(len(a)):
        t = a[i] - b[i]
        d += t * t
    return d


order = sorted(range(n), key=lambda i: (pts[i][0], i))
cents = [list(pts[order[(j * n) // k]]) for j in range(k)]

labels = [0] * n
for _ in range(25):
    changed = False
    for i in range(n):
        best, bd = 0, None
        for c in range(k):
            d = dist2(pts[i], cents[c])
            if bd is None or d < bd:
                bd, best = d, c
        if labels[i] != best:
            changed = True
        labels[i] = best
    sums = [[0.0] * len(pts[0]) for _ in range(k)]
    cnts = [0] * k
    for i in range(n):
        c = labels[i]
        cnts[c] += 1
        for d in range(len(pts[0])):
            sums[c][d] += pts[i][d]
    for c in range(k):
        if cnts[c] > 0:
            cents[c] = [sums[c][d] / cnts[c] for d in range(len(pts[0]))]
    if not changed:
        break

print(json.dumps({"labels": labels}))
