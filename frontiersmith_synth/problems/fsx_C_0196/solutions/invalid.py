# TIER: invalid
# A malformed DFA: it claims 2 states but every transition points to state 9, which
# does not exist (out of range for K=2).  The evaluator rejects it -> score 0.0 on
# every instance.
import sys, json

inst = json.load(sys.stdin)
D = inst["n_types"]

print(json.dumps({"start": 0, "accept": [1, 1], "trans": [[9] * D, [9] * D]}))
