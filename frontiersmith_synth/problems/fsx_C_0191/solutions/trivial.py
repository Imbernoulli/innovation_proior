# TIER: trivial
# Trivial thermal-response curve: the IDENTITY phi(x)=x. This reproduces the
# evaluator's own baseline activation, so the one-hidden-layer net collapses to a
# linear classifier -> per-hall normalized score ~0.1 everywhere.
import sys, json

inst = json.load(sys.stdin)
grid = inst["grid"]
print(json.dumps({"activation": list(grid)}))
