# TIER: invalid
# Emit a malformed plan: negative shed amounts (and the wrong inner length).  The
# evaluator rejects any non-finite / negative / mis-shaped entry, so this scores 0.0.
import sys, json

inst = json.load(sys.stdin)
T = inst["T"]
print(json.dumps({"shed": [[-1.0] for _ in range(T)]}))
