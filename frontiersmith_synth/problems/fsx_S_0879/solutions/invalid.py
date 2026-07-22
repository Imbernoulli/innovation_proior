# TIER: invalid
# Emit a malformed order: duplicate indices (job 0 listed 5 times) and an out-of-range
# index. The evaluator rejects any duplicate / out-of-range entry, so this scores 0.0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
print(json.dumps({"order": [0, 0, 0, 0, 0, n + 5]}))
