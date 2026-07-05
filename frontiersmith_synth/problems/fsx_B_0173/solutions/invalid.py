# TIER: invalid
import sys, json
inst = json.load(sys.stdin)
n = inst["n"]
# decision coordinates outside the [0,1] box -> the evaluator rejects the answer -> score 0
print(json.dumps({"points": [[3.0] * n for _ in range(5)]}))
