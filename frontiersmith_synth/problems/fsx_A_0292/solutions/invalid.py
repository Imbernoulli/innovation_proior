# TIER: invalid
# Cram every group onto a single tour.  The queue's total headcount (and total
# docent-minute demand) far exceeds one tour's capacity on every instance, so
# tour 0 overflows both axes -> the layout is infeasible -> the evaluator scores
# it 0.0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]

print(json.dumps({"assign": [0] * n}))
