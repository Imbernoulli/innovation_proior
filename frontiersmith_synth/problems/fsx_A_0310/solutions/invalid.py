# TIER: invalid
# Visit EVERY system in index order.  The flown path length far exceeds the fuel
# budget L on every instance, so the route is rejected and scores 0.0.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"route": list(range(inst["N"]))}))
