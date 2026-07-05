# TIER: invalid
# Emit a degenerate "ranking" that puts every candidate at position 0.  This is not
# a permutation of the candidate indices (duplicates, out of shape), so the evaluator
# rejects it and scores the session 0.0.
import sys, json

inst = json.load(sys.stdin)
m = len(inst["items"])
print(json.dumps({"ranking": [0] * m}))
