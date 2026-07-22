# TIER: invalid
# Blows the budget: allocates the FULL budget to every region simultaneously
# (sum >> budget), which the feasibility check must reject -> score 0.
import sys, json

inst = json.load(sys.stdin)
R = len(inst["regions"])
B = inst["budget"]
print(json.dumps({"alloc": [float(B)] * R}))
