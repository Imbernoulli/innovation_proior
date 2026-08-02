# TIER: invalid
# Claims to cut cell (0, 0) twice within the very first year. `quota` is always
# >= 1 in every instance, so this is a duplicate-cell violation the evaluator must
# reject outright -> the instance scores 0.0.
import sys, json

inst = json.load(sys.stdin)
T = inst["horizon"]

harvests = [[[0, 0], [0, 0]]] + [[] for _ in range(T - 1)]
print(json.dumps({"harvests": harvests}))
