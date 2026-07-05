# TIER: invalid
# Emits a non-permutation (every clause set to catalog index 0). It has the right
# length but is not a permutation of 0..C-1, so the evaluator rejects it as
# malformed and it scores 0 on every instance.
import sys, json

inst = json.load(sys.stdin)
C = inst["n_clauses"]
print(json.dumps({"order": [0] * C}))
