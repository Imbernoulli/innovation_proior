# TIER: trivial
# No regularization at all: the fixed trainer overfits the 25 noisy points.
# This reproduces the evaluator's baseline construction -> normalized score ~= 0.1.
import sys, json
json.load(sys.stdin)
print(json.dumps({"ridge": 0.0}))
