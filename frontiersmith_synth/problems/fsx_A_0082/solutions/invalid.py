# TIER: invalid
# Cram every party onto a single gondola.  Whenever the queue's total headcount
# exceeds one gondola's capacity (true for every instance in this family), gondola
# 0 is overfilled -> the layout is infeasible -> the evaluator scores it 0.0.
import sys, json

inst = json.load(sys.stdin)
N = inst["n"]

print(json.dumps({"assign": [0] * N}))
