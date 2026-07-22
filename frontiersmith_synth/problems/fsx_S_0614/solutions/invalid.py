# TIER: invalid
# Cram every item into bin 0.  Total weight vastly exceeds any single bin's capacity, so
# the assignment is INFEASIBLE and the evaluator scores it 0.0 -- exercising the capacity
# feasibility check.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]
print(json.dumps({"assign": [0] * N}))
