# TIER: invalid
# Assign every job to machine index k, which is OUT OF RANGE (valid machines are
# 0..k-1).  The evaluator rejects the assignment as infeasible -> scores 0.0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
k = inst["k"]
print(json.dumps({"assign": [k] * n}))
