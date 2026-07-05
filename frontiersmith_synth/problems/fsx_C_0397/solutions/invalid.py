# TIER: invalid
# Emits a label list of the wrong length -> evaluator rejects it -> 0.
import sys, json
inst = json.load(sys.stdin)
print(json.dumps({"labels": [0, 1, 2]}))
