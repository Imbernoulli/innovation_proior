# TIER: invalid
# Dumps every guest into room 0, ignoring the clash graph entirely. Except
# for degenerate single-guest instances this immediately seats two clashing
# guests together, so the evaluator rejects the layout and scores it 0.0.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
print(json.dumps({"room": [0] * n}))
