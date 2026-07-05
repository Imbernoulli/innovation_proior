# TIER: invalid
# Cheapest-possible design: give every strut the minimum cross-section. This
# is by far the lightest, but the gantry is grossly over-stressed and sways
# past its limit -> the feasibility gate rejects it -> score 0.
import sys, json
inst = json.load(sys.stdin)
m = len(inst["bars"])
print(json.dumps({"areas": [inst["a_min"]] * m}))
