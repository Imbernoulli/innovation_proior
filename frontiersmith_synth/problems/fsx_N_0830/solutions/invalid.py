# TIER: invalid
# Emits an "actions" list of the WRONG length (only decides half the
# stream). The evaluator requires len(actions) == N for every instance,
# so this is rejected as infeasible -> scores 0.0 everywhere.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]

print(json.dumps({"actions": [1] * (N // 2), "recalls": []}))
