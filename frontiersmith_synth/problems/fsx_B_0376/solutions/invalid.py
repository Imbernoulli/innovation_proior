# TIER: invalid
# Not a valid global order: emits all zeros (duplicate stage ids, missing the
# rest). It is NOT a permutation of 0..C-1, so the evaluator's strict validator
# rejects it -> score 0 on every instance.
import sys, json
inst = json.load(sys.stdin)
C = inst["n_stages"]
print(json.dumps({"order": [0] * C}))
