# TIER: invalid
# Hold no safety stock anywhere. The programme fill rate collapses far below beta
# (and every local life-support floor is violated), so this is infeasible on every
# instance -> 0.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]
print(json.dumps({"stock": [0.0] * N}))
