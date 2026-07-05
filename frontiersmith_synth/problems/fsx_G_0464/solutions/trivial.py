# TIER: trivial
# Raw squared-Euclidean nearest-neighbour retrieval on the descriptors AS GIVEN.
# This reproduces the evaluator's weak reference: distance is dominated by the
# per-protein abundance/length factor and the loud nuisance channels, not by fold,
# so it scores ~0.1.
import sys, json

inst = json.load(sys.stdin)
N = inst["n"]
feats = inst["features"]

ranking = []
for i in range(N):
    fi = feats[i]
    scored = []
    for j in range(N):
        if j == i:
            continue
        d2 = 0.0
        fj = feats[j]
        for a, b in zip(fi, fj):
            dv = a - b
            d2 += dv * dv
        scored.append((-d2, j))
    scored.sort(key=lambda t: (-t[0], t[1]))
    ranking.append([j for _, j in scored])

print(json.dumps({"ranking": ranking}))
