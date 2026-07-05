# TIER: invalid
# Drill every well at the same cell (0, 0).  For K >= 2 the wells are not pairwise
# distinct, so the placement fails validation -> the evaluator scores it 0.0.
import sys, json

inst = json.load(sys.stdin)
k = inst["k"]

wells = [[0, 0] for _ in range(k)]
print(json.dumps({"wells": wells}))
