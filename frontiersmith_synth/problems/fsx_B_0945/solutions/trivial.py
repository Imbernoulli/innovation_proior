# TIER: trivial
# Never move: stay at the starting paddock for the whole season. This is the
# evaluator's own reference "do nothing" construction (obj_base), so it always
# scores ~0.1. Grass at the home paddock gets grazed down to nothing and only
# recovers from its own logistic regrowth plus whatever diffusion trickles in
# from neighbours that are never touched.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
start = inst["start"]

print(json.dumps({"visits": [start] * T}))
