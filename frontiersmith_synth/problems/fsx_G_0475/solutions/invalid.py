# TIER: invalid
# Emits a malformed answer: the "pred" list has the wrong length (one entry) and a
# non-string element.  The evaluator rejects the shape -> every instance scores 0.
import sys, json

inst = json.load(sys.stdin)
print(json.dumps({"pred": [12345]}))
