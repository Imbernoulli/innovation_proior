# TIER: trivial
# All-cheapest policy: assign every cut to option 0 (the base yard switcher).
# Always feasible (budget floor), reproduces the evaluator baseline -> ~0.1.
import sys, json
inst = json.load(sys.stdin)
n = inst["n_cuts"]
print(json.dumps({"assign": [0] * n}))
