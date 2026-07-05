# TIER: greedy
# Max-sum dispersion ordering.  Start from the point farthest from the pool mean,
# then repeatedly add the point that maximizes the SUM of distances to the
# already-selected set.  This spreads the queried points out, so it covers the
# feature space (and the rare clusters) faster than a random order.  But summing
# distances (instead of the k-center MINIMUM distance) rewards points that are far
# from EVERYTHING, so the budget piles up in a few extreme directions and inner /
# less-extreme rare clusters get covered late -- it beats random but trails a true
# k-center coverage order.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
pool = inst["pool"]
d = inst["d"]


def sqd(a, b):
    s = 0.0
    for j in range(d):
        t = a[j] - b[j]
        s += t * t
    return s


mean = [0.0] * d
for row in pool:
    for j in range(d):
        mean[j] += row[j]
for j in range(d):
    mean[j] /= n

start = 0
best = -1.0
for i in range(n):
    v = sqd(pool[i], mean)
    if v > best:
        best = v
        start = i

selected = [start]
in_sel = [False] * n
in_sel[start] = True
sumd = [sqd(pool[i], pool[start]) for i in range(n)]

while len(selected) < n:
    best_i = -1
    best_v = -1.0
    for i in range(n):
        if in_sel[i]:
            continue
        if sumd[i] > best_v:
            best_v = sumd[i]
            best_i = i
    selected.append(best_i)
    in_sel[best_i] = True
    b = pool[best_i]
    for i in range(n):
        if not in_sel[i]:
            sumd[i] += sqd(pool[i], b)

print(json.dumps({"order": selected}))
