# TIER: strong
# Farthest-first / k-center coverage order.  Start from the point farthest from
# the pool mean, then repeatedly add the pool point whose MINIMUM distance to the
# already-selected set is largest.  This greedily maximizes coverage of the
# feature space, so it plants a labeled representative in every distinct cluster
# -- including the rare ones -- as early as possible.  Under the nearest-centroid
# model this yields a class centroid for every latent class within a handful of
# labels, so the learning curve rises fast.  It still cannot reach the
# immediate-jump ceiling (the first labels can only cover so many classes), so the
# normalized score stays below 1.0 -> headroom.
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

# deterministic start: farthest from the mean (break ties by lowest index)
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
mind = [sqd(pool[i], pool[start]) for i in range(n)]

while len(selected) < n:
    best_i = -1
    best_v = -1.0
    for i in range(n):
        if in_sel[i]:
            continue
        if mind[i] > best_v:
            best_v = mind[i]
            best_i = i
    selected.append(best_i)
    in_sel[best_i] = True
    b = pool[best_i]
    for i in range(n):
        if not in_sel[i]:
            dd = sqd(pool[i], b)
            if dd < mind[i]:
                mind[i] = dd

print(json.dumps({"order": selected}))
