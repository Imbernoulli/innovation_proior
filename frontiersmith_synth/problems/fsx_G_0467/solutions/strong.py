# TIER: strong
# Feature-primed label propagation on the citation graph.
#   1) Build a feature prior: soft class scores from distance to seed centroids.
#   2) Spread labels over citation edges.  Seed nodes are CLAMPED to their true
#      one-hot every iteration; other nodes update as a blend of (a) the average
#      class distribution of their citation neighbours and (b) their own feature
#      prior.  Homophily makes neighbour aggregation informative, while the
#      feature prior anchors isolated / low-degree papers.
#   3) After convergence, argmax gives each query paper's subfield.
# Combining structure + features beats the feature-only centroid rule, but the
# noisy graph and imperfect features keep accuracy below 1.0 -> headroom.
import sys, json, math

inst = json.load(sys.stdin)
n = inst["n"]
k = inst["k"]
dim = inst["dim"]
feats = inst["features"]
edges = inst["edges"]
train_ids = inst["train_ids"]
train_labels = inst["train_labels"]
query_ids = inst["query_ids"]

# ---- feature prior via nearest-centroid, turned into a soft distribution ----
sums = [[0.0] * dim for _ in range(k)]
cnt = [0] * k
for nid, c in zip(train_ids, train_labels):
    row = feats[nid]
    for d in range(dim):
        sums[c][d] += row[d]
    cnt[c] += 1
cent = [[(sums[c][d] / cnt[c]) if cnt[c] else 0.0 for d in range(dim)]
        for c in range(k)]


def feat_prior(row):
    d2 = []
    for c in range(k):
        s = 0.0
        cc = cent[c]
        for d in range(dim):
            diff = row[d] - cc[d]
            s += diff * diff
        d2.append(s)
    # softmax over negative (scaled) distances
    m = min(d2)
    w = [math.exp(-(x - m) / (2.0 * dim)) for x in d2]
    z = sum(w) or 1.0
    return [x / z for x in w]


prior = [feat_prior(feats[i]) for i in range(n)]

# ---- adjacency ----
adj = [[] for _ in range(n)]
for u, v in edges:
    adj[u].append(v)
    adj[v].append(u)

# ---- initialise label distributions ----
is_seed = [False] * n
seed_dist = [None] * n
for nid, c in zip(train_ids, train_labels):
    is_seed[nid] = True
    oneh = [0.0] * k
    oneh[c] = 1.0
    seed_dist[nid] = oneh

dist = []
for i in range(n):
    if is_seed[i]:
        dist.append(list(seed_dist[i]))
    else:
        dist.append(list(prior[i]))

# ---- iterate propagation ----
ALPHA = 0.75          # weight on neighbour aggregation vs. feature prior
ITERS = 30
for _ in range(ITERS):
    new = []
    for i in range(n):
        if is_seed[i]:
            new.append(seed_dist[i])
            continue
        nb = adj[i]
        if nb:
            agg = [0.0] * k
            for j in nb:
                dj = dist[j]
                for c in range(k):
                    agg[c] += dj[c]
            inv = 1.0 / len(nb)
            agg = [x * inv for x in agg]
            row = [ALPHA * agg[c] + (1.0 - ALPHA) * prior[i][c] for c in range(k)]
        else:
            row = list(prior[i])
        z = sum(row) or 1.0
        new.append([x / z for x in row])
    dist = new

labels = []
for q in query_ids:
    row = dist[q]
    best = 0
    for c in range(1, k):
        if row[c] > row[best]:
            best = c
    labels.append(best)

print(json.dumps({"labels": labels}))
