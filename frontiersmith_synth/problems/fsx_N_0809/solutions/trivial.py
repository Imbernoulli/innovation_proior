# TIER: trivial
# Do-nothing controller: post zero toll on every edge, every round.
import sys, json

inst = json.load(sys.stdin)
E = inst["E"]
T = inst["T"]
tolls = [[0.0] * E for _ in range(T)]
print(json.dumps({"tolls": tolls}))
