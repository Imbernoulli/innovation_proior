# TIER: trivial
# Spend nothing: rely entirely on the pilot batch, allocate zero extra
# budget to every region. Always feasible; reproduces the grader baseline.
import sys, json

inst = json.load(sys.stdin)
R = len(inst["regions"])
print(json.dumps({"alloc": [0.0] * R}))
