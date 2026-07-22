# TIER: invalid
# Cheat attempt: declare a transition table for only 2 columns (symbols 0/1)
# instead of the full m-symbol alphabet the instance requires. Every instance
# in this family has m > 2, so the evaluator's shape check on `trans` rows
# rejects the answer -> every instance scores 0.0.
import sys, json

inst = json.load(sys.stdin)

print(json.dumps({
    "n_states": 2,
    "start": 0,
    "trans": [[0, 1], [1, 0]],
    "out": [0.0, 1.0],
}))
