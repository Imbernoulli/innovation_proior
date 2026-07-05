# TIER: trivial
# Fly the SAME "midpoint" mission profile budget-many times: x = [0.5]*n for every
# profile. That profile is Pareto-optimal (distance vars = 0.5 -> on the cost frontier),
# but every mission collapses to a single cost point, so the portfolio dominates no more
# 4D volume than one point. This is exactly the evaluator's baseline construction -> ~0.1.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
budget = inst["budget"]
pts = [[0.5] * n for _ in range(budget)]
print(json.dumps({"points": pts}))
