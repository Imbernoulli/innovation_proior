# TIER: invalid
# Emits a curve of the wrong length (and a non-finite entry) -> the evaluator must reject it -> 0.
import sys, json
inst = json.load(sys.stdin)
print(json.dumps({"ys": [1e309, 0.0, 0.0]}))
