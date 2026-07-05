# TIER: invalid
# Emit an out-of-range label (k, which is never a valid subfield in 0..k-1) for
# every query paper.  The evaluator rejects any label outside [0, k), so the
# whole layout is infeasible -> the instance scores 0.0.
import sys, json

inst = json.load(sys.stdin)
k = inst["k"]
query_ids = inst["query_ids"]

print(json.dumps({"labels": [k] * len(query_ids)}))
