# TIER: trivial
# Open-door harbor master: bar 0 everywhere -> admit every vessel that still
# fits, in arrival order.  This reproduces the evaluator's open-door anchor, so
# it scores ~0.1 on every tide.
import sys, json

inst = json.load(sys.stdin)
K = inst["K"]; R = inst["R_buckets"]; S = inst["S_buckets"]
bars = [[[0.0 for _ in range(S)] for _ in range(R)] for _ in range(K)]
print(json.dumps({"bars": bars}))
