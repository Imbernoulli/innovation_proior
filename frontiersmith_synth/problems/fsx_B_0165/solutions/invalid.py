# TIER: invalid
# Proposes a configuration with out-of-range coordinates (5.0 is outside
# [0,1]); the evaluator rejects it -> score 0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
print(json.dumps({"points": [[5.0] * n]}))
