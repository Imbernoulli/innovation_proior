# TIER: strong
# A properly designed unsupervised retrieval kernel that undoes ALL three nuisances:
#   (1) L1-normalize each descriptor into a composition  -> removes the per-protein
#       abundance / length factor.
#   (2) per-channel Z-SCORE across the batch (subtract channel mean, divide by channel
#       std) -> centers out the constant per-channel baseline offsets AND equalizes the
#       loud nuisance channels against the quiet informative ones.
#   (3) cosine similarity in the standardized space, nearest-neighbour ranked.
# Centering (which the greedy kernel skips) is what lets the informative channels
# dominate the geometry, so same-fold proteins cluster and mAP jumps well above the
# raw-distance baseline.  Families still overlap and the informative channels are
# noisy, so it stays far from perfect retrieval -- headroom remains for even better
# channel-weighting kernels.
import sys, json, math

inst = json.load(sys.stdin)
N = inst["n"]
D = inst["dim"]
feats = inst["features"]

# (1) composition normalization -> removes per-protein abundance/length
comp = []
for v in feats:
    s = 0.0
    for x in v:
        s += abs(x)
    if s < 1e-12:
        s = 1e-12
    comp.append([x / s for x in v])

# (2) per-channel z-score across the batch
mean = [0.0] * D
for d in range(D):
    acc = 0.0
    for i in range(N):
        acc += comp[i][d]
    mean[d] = acc / N
std = [0.0] * D
for d in range(D):
    acc = 0.0
    for i in range(N):
        dv = comp[i][d] - mean[d]
        acc += dv * dv
    sd = math.sqrt(acc / N)
    std[d] = sd if sd > 1e-12 else 1e-12

z = [[(comp[i][d] - mean[d]) / std[d] for d in range(D)] for i in range(N)]
zn = []
for v in z:
    nn = math.sqrt(sum(x * x for x in v))
    zn.append(nn if nn > 1e-12 else 1e-12)

# (3) cosine nearest-neighbour ranking in standardized space
ranking = []
for i in range(N):
    zi = z[i]
    ni = zn[i]
    scored = []
    for j in range(N):
        if j == i:
            continue
        dot = 0.0
        zj = z[j]
        for a, b in zip(zi, zj):
            dot += a * b
        scored.append((dot / (ni * zn[j]), j))
    scored.sort(key=lambda t: (-t[0], t[1]))
    ranking.append([j for _, j in scored])

print(json.dumps({"ranking": ranking}))
