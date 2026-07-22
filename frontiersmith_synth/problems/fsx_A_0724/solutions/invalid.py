# TIER: invalid
# Deliberately broken answer: allocation is the wrong length, contains negative and
# non-integer entries, and blows past every per-measurement cap and the total budget.
# The evaluator must reject this and score every instance 0.0.
import sys, json

inst = json.load(sys.stdin)
M = inst.get("n_measurements", 5)

bad_alloc = [-3] + [999999] * M + [2.5]  # wrong length, negative, over-cap, non-int
print(json.dumps({"alloc": bad_alloc}))
