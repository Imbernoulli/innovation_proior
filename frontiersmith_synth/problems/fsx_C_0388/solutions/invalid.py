# TIER: invalid
# Wrong-shape answer (weight vector length != m) -> evaluator rejects -> 0.
import sys, json
inst = json.load(sys.stdin)
print(json.dumps({"w": [1.0, 2.0], "b": 0.0}))
