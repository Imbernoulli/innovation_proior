# TIER: trivial
# Static per-variable majority vote: for each variable, sum the weight of clauses where
# it appears positively vs negatively, and take whichever side has more total weight.
# No search, no structure-awareness -- this reproduces the evaluator's own baseline()
# exactly, so it always scores 0.1.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
clauses = inst["clauses"]
weights = inst["weights"]

pos = [0.0] * n
neg = [0.0] * n
for c, w in zip(clauses, weights):
    for lit in c:
        v = abs(lit) - 1
        if lit > 0:
            pos[v] += w
        else:
            neg[v] += w

assign = [1 if pos[v] >= neg[v] else 0 for v in range(n)]
print(json.dumps({"assign": assign}))
