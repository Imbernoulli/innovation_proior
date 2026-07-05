# TIER: trivial
# Uniform weights: treat every patient equally, i.e. do NOTHING about the class
# imbalance.  This reproduces the evaluator's weak baseline exactly, so it scores
# ~0.1 on every instance -- the classifier collapses onto the healthy majority and
# the rare disease subtypes get near-zero recall.
import sys, json

inst = json.load(sys.stdin)
N = inst["n"]
print(json.dumps({"weights": [1.0] * N}))
