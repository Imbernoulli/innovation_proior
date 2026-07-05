# TIER: trivial
# Uniform sizing at the maximum allowed area. This is the heaviest possible rack
# gantry and always satisfies both gates (it is the evaluator's baseline), so it is
# feasible everywhere and scores exactly ~0.1 -- no engineering, just "make every
# member the biggest steel section".
import sys, json

inst = json.load(sys.stdin)
M = len(inst["bars"])
print(json.dumps({"areas": [inst["a_max"]] * M}))
