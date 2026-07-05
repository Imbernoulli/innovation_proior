# TIER: invalid
# Place every job at the same slot (0, 0).  For J >= 2 the slots are not pairwise
# distinct, so the placement fails validation -> the evaluator scores it 0.0.
import sys, json

inst = json.load(sys.stdin)
J = inst["j"]

place = [[0, 0] for _ in range(J)]
print(json.dumps({"place": place}))
