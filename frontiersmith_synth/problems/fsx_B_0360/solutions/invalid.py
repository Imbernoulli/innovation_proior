# TIER: invalid
# Assign every cut to its heaviest (highest-fuel) option -> total fuel always exceeds
# the daily budget -> infeasible on every instance -> rejected -> 0.
import sys, json
inst = json.load(sys.stdin)
N = inst["n_cuts"]
M = inst["n_options"]
print(json.dumps({"assign": [M - 1] * N}))
