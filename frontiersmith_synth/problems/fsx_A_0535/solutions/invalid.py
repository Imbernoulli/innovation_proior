# TIER: invalid
# Emits NEGATIVE tolls (subsidies), which the evaluator rejects as infeasible.
# Every instance must therefore score 0.0.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"tolls": [-1.0] * inst["m"]}))
