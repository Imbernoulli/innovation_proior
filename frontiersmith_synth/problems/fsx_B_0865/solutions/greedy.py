# TIER: greedy
import sys, json

inst = json.load(sys.stdin)
budget = inst["budget"]

# The obvious textbook recipe: plain alternating Sinkhorn/RAS sweeps,
# omega=1 (exact projection), using the whole budget. No look at whether
# the matrix has block structure -- just uniform row/col sweeps.
ops = []
while len(ops) < budget:
    ops.append({"type": "row", "omega": 1.0})
    if len(ops) < budget:
        ops.append({"type": "col", "omega": 1.0})

print(json.dumps({"ops": ops}))
