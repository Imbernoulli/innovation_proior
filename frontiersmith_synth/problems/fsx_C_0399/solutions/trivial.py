# TIER: trivial
# Do nothing: commit an all-zero schedule so every step leaves the iterate at z0.
# The final residual equals ||F(z0)|| = base_res, which the evaluator anchors to 0.1.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
print(json.dumps({"a": [0.0] * T, "m": [0.0] * T, "o": [0.0] * T}))
