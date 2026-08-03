# TIER: invalid
# Return a NON-permutation: every slot set to 0. This is not a bijection over the
# predicate indices, so the evaluator rejects it as infeasible -> score 0.
import sys, json
inst = json.load(sys.stdin)
M = inst["M"]
print(json.dumps({"order": [0] * M}))
