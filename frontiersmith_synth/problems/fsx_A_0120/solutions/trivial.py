# TIER: trivial
# One giant zone: every cell gets label 0.  This exactly reproduces the
# evaluator's one-zone reference (zero walls, full mismatch), so it scores ~0.1
# on every instance.
import sys, json

inst = json.load(sys.stdin)
N = inst["H"] * inst["W"]
print(json.dumps({"labels": [0] * N}))
