# TIER: invalid
# Emit a head with a non-finite (NaN) weight.  The evaluator rejects any
# non-finite weight, so every instance scores 0.0.
import sys, json

inst = json.load(sys.stdin)
d = inst["d"]
w = [1.0] * d
w[0] = float("nan")
print(json.dumps({"w": w, "b": 0.0}))
