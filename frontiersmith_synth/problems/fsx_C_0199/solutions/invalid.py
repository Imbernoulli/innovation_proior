# TIER: invalid
# Emits a malformed labeling: wrong length and a non-finite entry. The evaluator's
# strict validation rejects it -> score 0 on every dataset -> geometric mean 0.
import sys, json
inst = json.load(sys.stdin)
n = len(inst["X"])
print(json.dumps({"labels": [float("nan")] * (n - 1)}))
