# TIER: trivial
# Do nothing: rest every day, cross no corridor. Delivers 0 cargo, never
# risks a crack. Reproduces the evaluator's 0.1 anchor exactly.
import sys, json

inst = json.load(sys.stdin)
n = inst["n_days"]
print(json.dumps({"routes": [-1] * n, "masses": [0.0] * n}))
