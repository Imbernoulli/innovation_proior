# TIER: trivial
# Do nothing: never charge, never discharge. Zero profit, zero risk, zero
# aging. Reproduces the evaluator's do-nothing baseline exactly, so it scores
# 0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]

print(json.dumps({"actions": [0.0] * T}))
