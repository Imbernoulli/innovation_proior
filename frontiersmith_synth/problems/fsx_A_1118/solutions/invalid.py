# TIER: invalid
# Reads the instance but returns an infeasible / non-finite policy (missing
# key + NaN); the evaluator must reject this and score 0.0 on every instance.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"policy": {"base": float("nan"), "cap_gain": 0.0, "drift_gain": 0.0}}))
