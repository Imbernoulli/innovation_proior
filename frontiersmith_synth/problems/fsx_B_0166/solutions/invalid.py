# TIER: invalid
# Emits a malformed order (repeated channel id, not a permutation of 0..N-1). The evaluator
# must reject it and score every instance 0.0.
import sys, json

inst = json.load(sys.stdin)
N = inst["N"]
order = [0] * N            # all zeros: right length, but repeats -> not a permutation
print(json.dumps({"order": order}))
