# TIER: invalid
# Reference an out-of-range column index (F itself, which is never a valid 0..F-1
# column).  The evaluator rejects the selection -> every dataset scores 0.0.
import sys, json

inst = json.load(sys.stdin)
F = inst["n_features"]
print(json.dumps({"features": [0, F]}))
