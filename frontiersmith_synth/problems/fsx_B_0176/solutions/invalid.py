# TIER: invalid
# Do-nothing design: hold zero spare modules anywhere. Cheapest possible holding
# cost, but the aggregate availability collapses far below beta (the evaluator sets
# beta strictly above the all-zero fill on every instance), so this is INFEASIBLE
# everywhere -> scores 0 on every instance.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]
print(json.dumps({"stock": [0.0] * N}))
