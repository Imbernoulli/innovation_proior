# TIER: trivial
# Commission the SAME configuration B times: the domain center (all 0.5).
# All copies collapse to one Pareto point, so the dominated hypervolume equals
# the evaluator's single-point baseline -> normalized score ~= 0.1.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
B = inst["B"]
print(json.dumps({"points": [[0.5] * n for _ in range(B)]}))
