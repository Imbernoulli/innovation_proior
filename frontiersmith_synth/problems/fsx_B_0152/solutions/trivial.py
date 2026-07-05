# TIER: trivial
# Gold-plate every depot: hold safety stock equal to safety factor 4 at every node.
# This is the evaluator's baseline design -- always feasible (fill rate ~1 >> beta)
# but the heaviest possible holding cost -- so it scores exactly ~0.1. No inventory
# reasoning at all: just "stock everything to the hilt".
import sys, json

inst = json.load(sys.stdin)
sd = inst["sd"]
print(json.dumps({"stock": [4.0 * s for s in sd]}))
