# TIER: trivial
# Trial the SAME "midpoint" recipe budget-many times: x = [0.5]*n for every point.
# That configuration is Pareto-optimal (distance vars = 0.5 -> on the cost frontier),
# but all trials collapse to a single cost point, so the batch dominates no more volume
# than one point. This is exactly the evaluator's baseline construction -> ~0.1.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
budget = inst["budget"]
pts = [[0.5] * n for _ in range(budget)]
print(json.dumps({"points": pts}))
