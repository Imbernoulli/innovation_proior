# TIER: invalid
# Emit a malformed admit vector: wrong length and an out-of-range entry (2 is not in
# {0,1}).  The evaluator rejects any mis-shaped / non-binary vector, so this scores 0.0.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]
print(json.dumps({"admit": [2] * (N - 1)}))
