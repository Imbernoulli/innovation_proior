# TIER: trivial
# Identity order: calibrate stages in their natural id order 0,1,...,C-1.
# This is exactly the evaluator's reference order, so it scores ~0.1 on every
# instance -- no attempt to exploit stage popularity or cluster structure.
import sys, json
inst = json.load(sys.stdin)
C = inst["n_stages"]
print(json.dumps({"order": list(range(C))}))
