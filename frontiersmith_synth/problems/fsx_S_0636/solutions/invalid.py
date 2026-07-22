# TIER: invalid
# Colors every node i % k, ignoring adjacency entirely. On any clique or dense
# cluster larger than k this immediately puts the same color on two interfering
# nodes, so the coloring is IMPROPER and the evaluator scores it 0.0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]; k = inst["k"]

colors = [i % k for i in range(n)]
print(json.dumps({"colors": colors}))
