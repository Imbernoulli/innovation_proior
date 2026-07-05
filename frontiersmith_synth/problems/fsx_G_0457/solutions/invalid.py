# TIER: invalid
# Emits out-of-range "probabilities" (5.0) for every test event.  A forecast value
# outside [0,1] is not a probability, so the evaluator rejects the whole answer and
# scores this instance 0.0.
import sys, json

inst = json.load(sys.stdin)
n_test = inst["n_test"]
print(json.dumps({"forecast": [5.0] * n_test}))
