# TIER: invalid
# Emit a labelling of the WRONG length (one label short).  The evaluator requires
# exactly H*W labels, so this fails validation and scores 0.0 on every instance.
import sys, json

inst = json.load(sys.stdin)
N = inst["H"] * inst["W"]
print(json.dumps({"labels": [0] * (N - 1)}))
