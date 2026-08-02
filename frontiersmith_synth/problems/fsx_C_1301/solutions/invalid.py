# TIER: invalid
# Only ever answers for van 0, no matter how many vans the fleet has -- the
# submitted plan is missing every other van's decision, which fails validation
# (every van id must appear exactly once) -> the evaluator scores every instance
# 0.0.
import sys, json

inst = json.load(sys.stdin)

print(json.dumps({"vans": [{"id": 0, "charge_at_p1": 0, "charge_at_p2": 0}]}))
