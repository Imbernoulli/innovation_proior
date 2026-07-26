# TIER: invalid
# Malformed: repeats job 0 for every slot instead of emitting a permutation. The
# evaluator's strict feasibility check (order must be a permutation of 0..n-1) rejects
# this on every instance -> score 0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
print(json.dumps({"order": [0] * n}))
