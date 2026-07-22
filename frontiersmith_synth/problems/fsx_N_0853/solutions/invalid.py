# TIER: invalid
# Emits out-of-range spray doses (2.0 > 1.0 upper bound) -- the evaluator's
# strict validation must reject this on every instance and score it 0.0.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]

print(json.dumps({"spray": [2.0] * T, "release": [0.0] * T}))
