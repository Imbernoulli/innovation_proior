# TIER: invalid
# Deliberately broken: dumps the entire budget onto station 0, blowing past
# that station's dock capacity (and, for small budgets, past the per-station
# bound in general) -- the grader must reject this as infeasible -> 0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
budget = inst["budget"]
alloc = [0] * n
alloc[0] = budget + 10_000  # way out of range for any capacity

print(json.dumps({"init": alloc}))
