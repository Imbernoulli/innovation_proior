# TIER: greedy
# Feature-only nearest-centroid classifier.  Compute the mean topic vector of the
# labeled seeds in each subfield, then assign every query paper to the subfield
# whose centroid is closest (squared Euclidean).  This uses the noisy feature
# signal but IGNORES the citation graph entirely, so it beats majority but leaves
# the homophily structure on the table.
import sys, json

inst = json.load(sys.stdin)
k = inst["k"]
dim = inst["dim"]
feats = inst["features"]
train_ids = inst["train_ids"]
train_labels = inst["train_labels"]
query_ids = inst["query_ids"]

# class centroids from labeled seeds
sums = [[0.0] * dim for _ in range(k)]
cnt = [0] * k
for nid, c in zip(train_ids, train_labels):
    row = feats[nid]
    for d in range(dim):
        sums[c][d] += row[d]
    cnt[c] += 1
cent = []
for c in range(k):
    if cnt[c] > 0:
        cent.append([sums[c][d] / cnt[c] for d in range(dim)])
    else:
        cent.append([0.0] * dim)


def nearest(row):
    best = 0
    bestd = None
    for c in range(k):
        dd = 0.0
        cc = cent[c]
        for d in range(dim):
            diff = row[d] - cc[d]
            dd += diff * diff
        if bestd is None or dd < bestd:
            bestd = dd
            best = c
    return best


labels = [nearest(feats[q]) for q in query_ids]
print(json.dumps({"labels": labels}))
