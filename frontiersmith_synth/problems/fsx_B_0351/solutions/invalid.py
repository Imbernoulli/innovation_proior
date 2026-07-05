# TIER: invalid
# Configurations with out-of-range coordinates (5.0 is not a valid inverter setting in
# [0,1]). The evaluator rejects any point outside [0,1]^n, so the whole batch is
# infeasible and scores 0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
budget = inst["budget"]
pts = [[5.0] * n for _ in range(budget)]
print(json.dumps({"points": pts}))
