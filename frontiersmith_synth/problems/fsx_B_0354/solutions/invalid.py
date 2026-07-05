# TIER: invalid
# Stockpile absurdly many kits everywhere.  The acquisition spend blows past the
# budget on every instance, so the layout is infeasible -> the evaluator scores it
# 0.0 on every instance.
import sys, json

inst = json.load(sys.stdin)
N = inst["n_sites"]
print(json.dumps({"q": [1e6] * N, "q0": 1e6}))
