# TIER: invalid
# Deliberately broken: returns every cell as -1, which is out of the valid symbol range
# [0, n-1] for every instance. The evaluator rejects this on shape/range validation
# alone -> scores 0 on every instance.
import sys, json

inst = json.load(sys.stdin)
n = inst["n"]
print(json.dumps({"grid": [[-1] * n for _ in range(n)]}))
