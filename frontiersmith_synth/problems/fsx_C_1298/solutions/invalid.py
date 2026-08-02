# TIER: invalid
# Dumps every ant into source 0, EVERY round, at double the colony's actual
# ant budget (2*A instead of A). The row sum exceeds A, so the evaluator
# rejects the layout on every instance (score 0.0) instead of silently
# clamping or ignoring the overshoot.
import sys, json

inst = json.load(sys.stdin)
K = inst["K"]
T = inst["T"]
A = inst["A"]

row = [0] * K
row[0] = 2 * A
print(json.dumps({"alloc": [row[:] for _ in range(T)]}))
