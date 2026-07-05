# TIER: trivial
# Reproduce the evaluator's weak COST-ASCENDING FILL operator: buy the cheapest
# sites first while the budget lasts, ignoring what they actually watch.  This
# exactly matches q_base, so it scores ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
M, B, tc = inst["M"], inst["B"], inst["tc"]

order = sorted(range(M), key=lambda j: (tc[j], j))
build = []
spent = 0
for j in order:
    if spent + tc[j] <= B:
        build.append(j)
        spent += tc[j]

print(json.dumps({"build": build}))
