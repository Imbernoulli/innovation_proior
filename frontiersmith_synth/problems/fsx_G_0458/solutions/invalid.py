# TIER: invalid
# Emit NEGATIVE weights.  A resampling/reweighting policy must be non-negative
# (you cannot resample a patient a negative number of times), so the evaluator
# rejects this as infeasible and scores it 0.0 on every instance.
import sys, json

inst = json.load(sys.stdin)
N = inst["n"]
print(json.dumps({"weights": [-1.0] * N}))
