# TIER: invalid
# Emit a label list of the WRONG length (N-1).  The evaluator's structural
# validation rejects it on every instance -> 0.0 everywhere.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
print(json.dumps({"labels": [0] * (n - 1)}))
