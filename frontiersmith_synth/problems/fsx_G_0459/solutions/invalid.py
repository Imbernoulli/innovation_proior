# TIER: invalid
# Not a valid permutation: send index 0 for every slot.  The order has the right
# length but repeats a single index, so it fails the strict permutation check and
# the evaluator scores every instance 0.0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]

print(json.dumps({"order": [0] * n}))
