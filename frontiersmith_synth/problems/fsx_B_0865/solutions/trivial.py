# TIER: trivial
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
# Minimal effort: a handful of blind row/col sweeps, then stop -- far
# short of using the whole budget, no matter how large it is.
K = 16
ops = []
while len(ops) < K:
    ops.append({"type": "row", "omega": 1.0})
    if len(ops) < K:
        ops.append({"type": "col", "omega": 1.0})
print(json.dumps({"ops": ops}))
