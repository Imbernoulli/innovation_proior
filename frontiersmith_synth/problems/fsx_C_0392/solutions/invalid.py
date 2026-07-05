# TIER: invalid
# Emit an out-of-range edge (references variable index n, which does not exist).
# The evaluator rejects any edge with an index >= n, so every region scores 0.0.
# A classic off-by-one / bad-index structure-learning bug.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
# also throw in a self-loop for good measure; either one is fatal
print(json.dumps({"edges": [[0, n], [1, 1]]}))
