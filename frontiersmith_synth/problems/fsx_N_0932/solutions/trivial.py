# TIER: trivial
# Do nothing clever: buy the entire quantity Q from the outside option in round 1,
# at whatever outside0 happens to be. This is exactly the evaluator's own
# do-nothing reference (cost_base = Q * outside0), so it anchors at ~0.1.
import sys, json

inst = json.load(sys.stdin)
Q = inst["Q"]

actions = [{"type": "outside", "qty": Q}]
print(json.dumps({"actions": actions}))
