# TIER: trivial
# Top off every locker to safety factor 4 -- always feasible for both the programme
# fill floor and every local life-support floor, but the heaviest possible holding
# cost. This is exactly the evaluator's baseline design, so it maps to ~0.1.
import sys, json

inst = json.load(sys.stdin)
sd = inst["sd"]
stock = [4.0 * s for s in sd]
print(json.dumps({"stock": stock}))
