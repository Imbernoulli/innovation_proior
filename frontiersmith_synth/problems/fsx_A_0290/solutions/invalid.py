# TIER: invalid
# Build EVERY candidate site.  The total cost far exceeds the budget on every
# instance, so the plan is rejected and scores 0.0.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"build": list(range(inst["M"]))}))
