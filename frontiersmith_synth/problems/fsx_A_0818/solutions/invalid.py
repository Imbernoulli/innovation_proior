# TIER: invalid
# Drill every well far outside the survey box. Every instance's box bounds
# are read from the input, so this is unconditionally out of bounds ->
# the evaluator rejects the whole answer -> scores 0.0 on every instance.
import sys, json

inst = json.load(sys.stdin)
D = inst["dim"]
box = inst["box"]
Q = inst["budget"]

pt = [box[d][1] + 1.0e6 for d in range(D)]
print(json.dumps({"queries": [pt for _ in range(Q)]}))
