# TIER: trivial
# Declare a single giant outbreak: everyone in cluster 0. Structurally valid, so it
# earns the 0.1 floor, but ARI = 0 -> no credit beyond the floor on any dataset.
import sys, json
inst = json.load(sys.stdin)
n = len(inst["X"])
print(json.dumps({"labels": [0] * n}))
