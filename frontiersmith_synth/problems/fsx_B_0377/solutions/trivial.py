# TIER: trivial
# Uniform a_max sizing: heaviest design, always feasible on all gates -> ~0.1.
import sys, json
inst = json.load(sys.stdin)
M = len(inst["bars"])
print(json.dumps({"areas": [inst["a_max"]] * M}))
