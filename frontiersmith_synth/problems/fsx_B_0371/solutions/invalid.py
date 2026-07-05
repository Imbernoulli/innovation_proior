# TIER: invalid
# Emit a phase mask of the wrong shape: a flat 1-D list instead of the required
# M x M grid.  The evaluator's shape check rejects it, so every instance scores 0.0.
import sys, json

inst = json.load(sys.stdin)
M = inst["M"]

print(json.dumps({"phase": [0.0] * M}))
