# TIER: trivial
# Reject every applicant. Zero risk, zero revenue -- this is exactly the
# evaluator's internal weak baseline, so it scores ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]

print(json.dumps({"decisions": [0] * N}))
