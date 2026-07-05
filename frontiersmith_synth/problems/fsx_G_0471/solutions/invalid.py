# TIER: invalid
# Emits example ids equal to N -- one past the last valid id [0, N-1].  The
# evaluator rejects the out-of-range schedule as infeasible, so the instance
# scores 0.0.
import sys, json

inst = json.load(sys.stdin)
N = inst["n_examples"]

print(json.dumps({"schedule": [N] * 50}))
