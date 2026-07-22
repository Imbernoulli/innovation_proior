# TIER: invalid
# Emits a decisions list with the WRONG length (only half the applicants
# are decided). The evaluator requires len(decisions) == N for every
# instance, so this is rejected as infeasible -> scores 0.0 everywhere.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]

print(json.dumps({"decisions": [1] * (N // 2)}))
