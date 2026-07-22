# TIER: trivial
# Naive wheel: visit every colour once, in the INPUT order (id 0,1,2,...), ignoring
# the routing question entirely.  Lot size is just a generic demand share of a fixed
# guessed cycle length -- exactly the evaluator's own weak reference construction, so
# this reproduces ~0.1.
import sys, json

inst = json.load(sys.stdin)
colors = inst["colors"]
k = inst["k"]
max_lot = inst["max_lot"]

generic_T = 260.0 + 40.0 * k
wheel = []
for c in colors:
    lot = max(c["min_lot"], int(round(c["demand"] * generic_T)))
    lot = min(lot, max_lot)
    wheel.append({"color": c["id"], "lot": lot})

print(json.dumps({"wheel": wheel}))
