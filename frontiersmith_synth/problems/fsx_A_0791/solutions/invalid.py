# TIER: invalid
# Post a price above the allowed maximum every period -- infeasible under the
# evaluator's strict [0, p_max] validation, so every instance scores 0.0.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
p_max = inst["p_max"]

print(json.dumps({"prices": [p_max * 2.0 + 1.0] * T}))
