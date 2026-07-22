# TIER: invalid
# Deliberately broken: wrong shape (missing rows) AND a NaN slipped in.
import sys, json

inst = json.load(sys.stdin)
E = inst["E"]
T = inst["T"]
tolls = [[float("nan")] * E for _ in range(max(0, T - 1))]  # wrong length, non-finite
print(json.dumps({"tolls": tolls}))
