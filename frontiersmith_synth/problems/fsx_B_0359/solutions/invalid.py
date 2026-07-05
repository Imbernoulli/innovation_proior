# TIER: invalid
# Well-formed layouts, but with every distance wiring var driven to 0.0
# (far from the 0.5 optimum).  That inflates DTLZ2's g term, so every cost
# blows past the reference point -> no layout is counted -> hypervolume 0.
import sys, json
inst = json.load(sys.stdin)
n, budget = inst["n"], inst["budget"]
pts = [[0.0] * n for _ in range(min(budget, 5))]
print(json.dumps({"points": pts}))
