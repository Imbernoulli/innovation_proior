# TIER: trivial
# Naive gradient descent-ascent: use the supplied conservative reference step at
# every round, no optimism, no momentum (beta = gamma = 0).  This reproduces the
# evaluator's reference method exactly, so it scores ~0.1 on every instance.
import sys, json

inst = json.load(sys.stdin)
K = inst["budget"]
eta = inst["ref_step"]

print(json.dumps({"alpha": [eta] * K, "beta": [0.0] * K, "gamma": [0.0] * K}))
