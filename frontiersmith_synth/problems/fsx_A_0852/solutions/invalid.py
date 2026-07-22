# TIER: invalid
# Massively overspend the budget on a single solver -- violates
# sum(alloc.values()) <= budget, so validation must reject every instance.
import sys, json

inst = json.load(sys.stdin)
heuristics = inst["heuristics"]
budget = inst["budget"]

alloc = {}
if heuristics:
    alloc[heuristics[0]["id"]] = budget * 10 + 1000

print(json.dumps({"alloc": alloc}))
