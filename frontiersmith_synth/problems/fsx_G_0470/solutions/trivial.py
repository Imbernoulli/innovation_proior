# TIER: trivial
# Identity transform: use the raw features unchanged.
import sys, json
inst = json.load(sys.stdin)
d = inst["d"]
W = [[1.0 if i == j else 0.0 for j in range(d)] for i in range(d)]
print(json.dumps({"W": W}))
