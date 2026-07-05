# TIER: trivial
# Reproduce the evaluator's REFERENCE schedule: a constant, conservative
# extragradient step (alpha_k = beta_k = ref = 0.4/||H||) for every iteration.
# This is exactly the weak baseline, so it scores ~0.1 on every reef.
import sys, json

inst = json.load(sys.stdin)
K = inst["K"]
ra = inst["ref_alpha"]
rb = inst["ref_beta"]
print(json.dumps({"alpha": [ra] * K, "beta": [rb] * K}))
