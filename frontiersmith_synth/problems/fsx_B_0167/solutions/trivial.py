# TIER: trivial
# Gold-plate the gantry: give every bar the maximum allowed cross-section a_max.
# This is the evaluator's baseline design -- the stiffest and heaviest possible, so it
# trivially clears every gate (lowest stress, highest buckling margin, least sag) but at
# the worst weight -- hence it maps to exactly ~0.1. No structural reasoning at all.
import sys, json

inst = json.load(sys.stdin)
M = len(inst["bars"])
print(json.dumps({"areas": [inst["a_max"]] * M}))
