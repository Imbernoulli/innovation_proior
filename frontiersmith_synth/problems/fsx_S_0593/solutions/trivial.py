# TIER: trivial
# Quote nothing: post zero size on both sides so nothing ever fills.  Inventory stays
# 0, cash stays 0, PnL = 0 -- exactly the evaluator's do-nothing reference, so this
# scores 0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
z = [0.0] * T
print(json.dumps({"hb": [0.0] * T, "ha": [0.0] * T, "zb": z, "za": z}))
