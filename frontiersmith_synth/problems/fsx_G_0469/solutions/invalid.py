# TIER: invalid
# Emit non-finite probabilities -> the evaluator rejects the answer (nan) and
# scores this instance 0.0.
import sys, json
inst = json.load(sys.stdin)
ts = inst["test_score"]
out = [float("nan") for _ in ts]
print(json.dumps({"prob": out}))
