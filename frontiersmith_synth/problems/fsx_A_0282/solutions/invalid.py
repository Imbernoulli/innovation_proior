# TIER: invalid
# Load EVERY container onto a single truck.  For any non-trivial instance this
# overloads the mass and/or bulk limit, so the plan is rejected and scores 0.0.
import sys, json

inst = json.load(sys.stdin)
n = len(inst["mass"])
print(json.dumps({"assign": [0] * n}))
