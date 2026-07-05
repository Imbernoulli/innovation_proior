# TIER: invalid
# Emits a non-permutation (all zeros): not a valid global ordering of the
# module types, so the evaluator rejects it and scores 0.
import sys, json

inst = json.load(sys.stdin)
K = inst["K"]
print(json.dumps({"order": [0] * K}))
