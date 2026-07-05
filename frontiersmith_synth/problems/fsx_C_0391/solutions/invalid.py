# TIER: invalid
# Emit a labelling of the WRONG length (half the drops).  The evaluator requires
# exactly n integer labels, so this fails validation on every instance -> 0.0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]

print(json.dumps({"labels": [0] * (n // 2)}))
