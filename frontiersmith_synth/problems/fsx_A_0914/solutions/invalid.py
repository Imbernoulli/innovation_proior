# TIER: invalid
# Every lot is 0 tons, below every colour's minimum batch size -- the furnace can't
# run a zero-ton campaign, so this fails validation and scores 0.0 on every instance.
import sys, json

inst = json.load(sys.stdin)
colors = inst["colors"]

wheel = [{"color": c["id"], "lot": 0} for c in colors]
print(json.dumps({"wheel": wheel}))
