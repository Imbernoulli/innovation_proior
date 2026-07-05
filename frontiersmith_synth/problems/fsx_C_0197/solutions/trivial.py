# TIER: trivial
# Identity foraging curve f(s)=s  (== the evaluator's reference baseline).
import sys, json
inst = json.load(sys.stdin)
grid = inst["grid"]
print(json.dumps({"ys": [float(x) for x in grid]}))
