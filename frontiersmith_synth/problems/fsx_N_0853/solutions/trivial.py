# TIER: trivial
# Never spray, never release. Zero risk, zero action -- this is exactly the
# evaluator's internal "do nothing" reference, so it scores ~0.1 on every
# instance by construction.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]

print(json.dumps({"spray": [0.0] * T, "release": [0.0] * T}))
