# TIER: invalid
# Emit a constant "order" of all zeros.  This is never a permutation of the cut
# indices (it repeats index 0 and omits the rest), so the evaluator rejects it as an
# infeasible dispatch priority and scores every instance 0.0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
print(json.dumps({"order": [0] * n}))
