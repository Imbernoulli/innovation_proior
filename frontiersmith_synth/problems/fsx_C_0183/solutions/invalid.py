# TIER: invalid
# Wrong-length schedule (T-1 entries) -> evaluator rejects -> score 0.
import sys, json
inst = json.load(sys.stdin)
print(json.dumps({"steps": [0.1] * max(0, inst["T"] - 1)}))
