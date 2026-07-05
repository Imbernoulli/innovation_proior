# TIER: trivial
# Worst-fit priority: maximize leftover budget (phi2 = (res-s)/C) so each canister
# is stowed into the EMPTIEST fitting module. This spreads load and wastes module
# budget -- it exactly reproduces the evaluator's weak reference rule, so it scores
# ~0.1 on every instance.
import sys, json

json.load(sys.stdin)  # read the public instance (unused: this is a fixed rule)
print(json.dumps({"weights": [0.0, 0.0, 1.0, 0.0]}))
