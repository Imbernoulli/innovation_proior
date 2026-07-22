# TIER: invalid
# Emits an out-of-range gain (gain_back = 99, far outside the validated
# [-3, 3] window) -> the evaluator rejects the answer as infeasible on every
# instance -> scores 0.0.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"gain_back": 99.0, "gain_fwd": 0.0, "target_frac": 1.0, "cap_frac": 1.0}))
