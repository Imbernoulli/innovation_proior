# TIER: trivial
# Constant scorer: "I cannot tell faults apart." Every snapshot gets the same score.
# ROC-AUC of a constant vector is exactly 0.5 -> reward exactly 0.10 on every instance.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
print(json.dumps({"scores": [1.0] * n}))
