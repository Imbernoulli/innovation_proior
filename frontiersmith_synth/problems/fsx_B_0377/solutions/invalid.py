# TIER: invalid
# Uniform a_min sizing: stresses blow past yield everywhere -> infeasible -> 0.
import sys, json
inst = json.load(sys.stdin)
M = len(inst["bars"])
print(json.dumps({"areas": [inst["a_min"]] * M}))
