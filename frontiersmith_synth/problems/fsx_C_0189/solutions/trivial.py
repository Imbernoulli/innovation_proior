# TIER: trivial
# Take no steps at all: zero step sizes leave z at z0, so the residual norm equals
# the baseline -> normalized score ~ 0.1 on every instance.
import sys, json
inst = json.load(sys.stdin)
K = inst["K"]
print(json.dumps({"eta": [0.0] * K, "gamma": [0.0] * K}))
