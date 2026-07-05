# TIER: invalid
# Uniform sizing at the MINIMUM allowed area. The members are far too thin: axial
# stresses blow past the yield limit (and the deck sags well beyond the service
# gate), so every instance is infeasible and scores 0.
import sys, json

inst = json.load(sys.stdin)
M = len(inst["bars"])
print(json.dumps({"areas": [inst["a_min"]] * M}))
