# TIER: trivial
# Coordinate-threshold segmentation: sort customers by their FIRST feature and cut
# into k equal ranked bins.  This is exactly the evaluator's weak reference
# segmentation (structure-blind: it ignores the second feature and all manifold
# shape), so it reproduces q_base and scores ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
pts = inst["points"]
k = inst["k"]
n = len(pts)

order = sorted(range(n), key=lambda i: (pts[i][0], i))
labels = [0] * n
per = n / k
for rank, i in enumerate(order):
    labels[i] = min(k - 1, int(rank / per))

print(json.dumps({"labels": labels}))
