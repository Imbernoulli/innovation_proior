# TIER: trivial
# Do nothing: charge no tolls. The selfish user equilibrium is left untouched,
# so the achieved total latency equals the do-nothing baseline (~0.1).
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"tolls": [0.0] * inst["m"]}))
