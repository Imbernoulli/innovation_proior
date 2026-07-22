# TIER: invalid
# Parks every car (lower AND upper, every shaft) at floor 0. Since the safety
# gap G is always positive, floor(upper) - floor(lower) = 0 < G for every
# shaft at t=0 -- a physically impossible starting configuration (the two
# cars would occupy the same/overlapping floors). The evaluator must reject
# this outright (score 0), regardless of how the calls are assigned.
import sys, json

inst = json.load(sys.stdin)
S = inst["S"]
calls = inst["calls"]

assign = [[0, 0] for _ in calls]
park = [0] * (2 * S)

print(json.dumps({"assign": assign, "park": park}))
