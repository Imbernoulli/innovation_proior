# TIER: trivial
# Reproduce the weak reference relay exactly: plain gradient descent-ascent with the
# conservative baseline step and NO optimism (b = 0).  This matches q_base on every
# instance, so it scores ~0.1 everywhere.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
eta0 = inst["baseline_step"]

print(json.dumps({"a": [eta0] * T, "b": [0.0] * T}))
