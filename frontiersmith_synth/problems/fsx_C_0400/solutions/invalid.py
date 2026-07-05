# TIER: invalid
# Emit a label list of the WRONG length (one shorter than the test batch).  The
# evaluator's strict shape check rejects it, so every instance scores 0.0.
import sys, json

inst = json.load(sys.stdin)
m = len(inst["test"])
print(json.dumps({"labels": [1] * max(0, m - 1)}))
