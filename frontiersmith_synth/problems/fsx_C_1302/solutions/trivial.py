# TIER: trivial
# Do-nothing baseline: never bid. Spends no budget, wins nothing, trains the
# competitor with zero aggression. Establishes the score floor.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
print(json.dumps({"bids": [0.0] * T}))
