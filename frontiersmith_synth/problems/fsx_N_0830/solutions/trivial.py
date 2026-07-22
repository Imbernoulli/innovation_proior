# TIER: trivial
# Reject every candidate. Zero risk, zero value -- this is exactly the
# evaluator's internal weak baseline, so it scores ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]

print(json.dumps({"actions": [0] * N, "recalls": []}))
