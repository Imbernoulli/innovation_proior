# TIER: invalid
# Cram every item onto a single porter load.  Every instance in this family has a
# total weight well above one load's capacity (and more than K distinct categories),
# so load 0 is over-weight AND over-category -> the plan is infeasible -> the
# evaluator scores it 0.0.
import sys, json

inst = json.load(sys.stdin)
N = inst["n"]

print(json.dumps({"assign": [0] * N}))
