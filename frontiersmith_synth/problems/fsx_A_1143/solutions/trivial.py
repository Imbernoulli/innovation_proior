# TIER: trivial
# Do nothing: bid 0 on every lot. Wins nothing, spends nothing, never breaches
# budget. Reproduces the evaluator's 0.1 anchor exactly (utility 0).
import sys, json

inst = json.load(sys.stdin)
n = inst["n_lots"]
print(json.dumps({"bids": [0.0] * n}))
