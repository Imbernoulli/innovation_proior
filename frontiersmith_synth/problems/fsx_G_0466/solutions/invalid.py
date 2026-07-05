# TIER: invalid
# Emits a non-finite forecast (NaN) -> the evaluator's finiteness check
# rejects it -> score 0.
import sys, json
inst = json.load(sys.stdin)
H = inst["horizon"]
print(json.dumps({"forecast": [float("nan")] * H}))
